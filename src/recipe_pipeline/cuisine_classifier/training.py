import logging
import os
import random
import warnings
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import DocBin
from spacy.training import Example

from .config import TRAIN, OUT
from .utils import configure_device, set_global_seed

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:
    torch = None

try:
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info
except ImportError:
    DataLoader = None
    IterableDataset = None
    get_worker_info = None

# ---------- Data Loading ----------

def collate_examples(batch):
    """Identity collate function for DataLoader (must be at module level for pickling)."""
    return batch


class ShardedIterableDataset(IterableDataset if IterableDataset is not None else object):
    """
    Iterable dataset that streams Examples from sharded .spacy files.
    
    Features:
    - Shuffles shard order at iteration start
    - Shuffles docs within each shard
    - Supports multi-worker data loading via worker_init_fn
    - Memory efficient: only one shard in memory at a time
    """
    
    def __init__(
        self, 
        vocab: spacy.vocab.Vocab, 
        dir_path: Path, 
        shuffle: bool = True,
        max_docs: Optional[int] = None,
        seed: int = 42,
        task_type: str = "textcat"  # "textcat" for classification, "ner" for NER
    ):
        """
        Initialize the ShardedIterableDataset.
        
        Args:
            vocab: spaCy Vocab object for deserializing docs
            dir_path: Directory containing .spacy shard files
            shuffle: Whether to shuffle shards and docs within shards
            max_docs: Optional limit on total documents to yield
            seed: Random seed for reproducibility
            task_type: "textcat" for text classification, "ner" for NER
        """
        super().__init__()
        self.vocab = vocab
        self.dir_path = dir_path
        self.shuffle = shuffle
        self.max_docs = max_docs
        self.base_seed = seed
        self.task_type = task_type
        
        # Create blank nlp for Example creation (lightweight, picklable)
        self.blank_nlp = spacy.blank("en")
        
        # Get all shard paths
        self.all_shard_paths = sorted(p for p in dir_path.glob("*.spacy"))
        if not self.all_shard_paths:
            raise RuntimeError(f"No .spacy files found in {dir_path}")
        
        self._epoch = 0
        
        logger.debug(f"ShardedIterableDataset: {len(self.all_shard_paths)} shards in {dir_path}")
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across epochs."""
        self._epoch = epoch
    
    def _get_worker_shards(self) -> List[Path]:
        """Get the subset of shards assigned to this worker."""
        worker_info = get_worker_info() if get_worker_info is not None else None
        
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        # Shuffle shards deterministically based on epoch
        rng = random.Random(self.base_seed + self._epoch)
        shards = self.all_shard_paths.copy()
        if self.shuffle:
            rng.shuffle(shards)
        
        # Subsample shards if max_docs is set
        # We estimate ~2000 docs per shard.
        if self.max_docs is not None:
            docs_per_shard_est = 2000
            shards_needed = max(1, int(self.max_docs / docs_per_shard_est))
            if shards_needed < len(shards):
                logger.debug(f"Subsampling: using {shards_needed} shards out of {len(shards)} to approx {self.max_docs} docs")
                shards = shards[:shards_needed]
        
        # Split shards among workers (round-robin assignment)
        if num_workers > 1:
            worker_shards = [s for i, s in enumerate(shards) if i % num_workers == worker_id]
            logger.debug(
                f"Worker {worker_id}/{num_workers}: assigned {len(worker_shards)}/{len(shards)} shards"
            )
            return worker_shards
        
        return shards
    
    def __iter__(self):
        """
        Iterate through shards, yielding Example objects.
        """
        shards = self._get_worker_shards()
        
        worker_info = get_worker_info() if get_worker_info is not None else None
        worker_id = worker_info.id if worker_info else 0
        local_rng = random.Random(self.base_seed + self._epoch + worker_id)
        
        docs_yielded = 0
        
        for shard_path in shards:
            # Check max_docs limit (safety break)
            if self.max_docs is not None and docs_yielded >= self.max_docs:
                break
            
            try:
                db = DocBin().from_disk(shard_path)
                cached_examples: List = []
                for doc in db.get_docs(self.vocab):
                    if self.task_type == "textcat":
                        cached_examples.append((doc.text, dict(doc.cats)))
                    else:
                        ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
                        cached_examples.append((doc.text, ents))

                if self.shuffle:
                    local_rng.shuffle(cached_examples)
                
                for payload in cached_examples:
                    if self.max_docs is not None and docs_yielded >= self.max_docs:
                        break
                    
                    if self.task_type == "textcat":
                        text, cats = payload
                        example = Example.from_dict(
                            self.blank_nlp.make_doc(text), 
                            {"cats": cats}
                        )
                    else:  # NER
                        text, ents = payload
                        example = Example.from_dict(
                            self.blank_nlp.make_doc(text), 
                            {"entities": ents}
                        )
                    
                    yield example
                    docs_yielded += 1
                    
            except Exception as e:
                logger.warning(f"Error loading shard {shard_path}: {e}")
                continue


def sharded_worker_init_fn(worker_id: int):
    """Worker init function for DataLoader with ShardedIterableDataset."""
    worker_info = get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed % (2**32)
        random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)


# ---------- Transformer / tok2vec training using DocBins ----------

def build_nlp_transformer(all_labels: List[str]) -> spacy.language.Language:
    """Build a transformer + textcat classifier with optional layer freezing."""
    try:
        import spacy_transformers  # noqa: F401
    except Exception as e:
        raise RuntimeError("spacy-transformers is not available.") from e

    from .config import TRAIN as _TRAIN

    nlp = spacy.blank("en")
    trf_cfg = {
        "model": {
            "@architectures": "spacy-transformers.TransformerModel.v3",
            "name": _TRAIN.TRANSFORMER_MODEL,
            "tokenizer_config": {"use_fast": True},
            "transformer_config": {},
            "mixed_precision": bool(_TRAIN.USE_AMP),
            "grad_scaler_config": {"enabled": bool(_TRAIN.USE_AMP)},
            "get_spans": {
                "@span_getters": "spacy-transformers.strided_spans.v1",
                "window": int(_TRAIN.WINDOW),
                "stride": int(_TRAIN.STRIDE),
            },
        },
        "set_extra_annotations": {
            "@annotation_setters": "spacy-transformers.null_annotation_setter.v1"
        },
    }
    max_batch_items = getattr(_TRAIN, "MAX_BATCH_ITEMS", None)
    if max_batch_items is None:
        max_batch_items = min(1024, 8 * max(1, int(_TRAIN.BATCH_SIZE)))
    trf_cfg["max_batch_items"] = int(max_batch_items)
    nlp.add_pipe("transformer", config=trf_cfg)
    
    # Add textcat component for multi-class classification (single label per example)
    # Note: exclusive_classes is set via model config, not pipe config in spaCy v3+
    textcat = nlp.add_pipe("textcat")
    for label in all_labels:
        textcat.add_label(label)

    # Optional layer freezing
    if _TRAIN.FREEZE_LAYERS > 0:
        try:
            trf = nlp.get_pipe("transformer").model
            hf = trf.transformer.model
            blocks = None
            if hasattr(hf, "transformer") and hasattr(hf.transformer, "layer"):  # distilbert
                blocks = hf.transformer.layer
            elif hasattr(hf, "encoder") and hasattr(hf.encoder, "layer"):        # bert/roberta
                blocks = hf.encoder.layer
            if blocks is not None:
                k = min(_TRAIN.FREEZE_LAYERS, len(blocks))
                for i in range(k):
                    for p in blocks[i].parameters():
                        p.requires_grad = False
                logger.info(f"[transformer] Froze {k} lower layer(s).")
        except Exception as e:
            warnings.warn(f"Could not freeze layers: {e}")
    return nlp


def build_nlp_tok2vec(all_labels: List[str]) -> spacy.language.Language:
    """CPU-friendly tok2vec + textcat fallback."""
    nlp = spacy.blank("en")
    nlp.add_pipe("tok2vec")
    # Note: exclusive_classes is set via model config, not pipe config in spaCy v3+
    textcat = nlp.add_pipe("textcat")
    for label in all_labels:
        textcat.add_label(label)
    logger.info("Using tok2vec fallback (no transformers).")
    return nlp


def choose_nlp(all_labels: List[str]) -> Tuple[Language, str]:
    """Choose transformer or tok2vec pipeline based on config and available deps."""
    from .config import TRAIN as _TRAIN

    # Check if debug mode forces tok2vec
    if hasattr(_TRAIN, 'USE_TOK2VEC_DEBUG') and _TRAIN.USE_TOK2VEC_DEBUG:
        logger.info("Debug mode: using tok2vec instead of transformers (faster, CPU-friendly)")
        return build_nlp_tok2vec(all_labels), "tok2vec"

    if torch is not None:
        has_trf = True
        try:
            import spacy_transformers  # noqa
        except Exception:
            has_trf = False
        if has_trf:
            try:
                return build_nlp_transformer(all_labels), "transformer"
            except Exception as e:
                warnings.warn(f"Falling back to tok2vec due to: {e}")
                return build_nlp_tok2vec(all_labels), "tok2vec"
    # No torch or transformers
    return build_nlp_tok2vec(all_labels), "tok2vec"


def iter_examples_from_docbins(
    nlp: Language,
    dir_path: Path,
    shuffle: bool = False,
    max_docs: Optional[int] = None,
) -> Iterable[Example]:
    shard_paths = sorted(p for p in dir_path.glob("*.spacy"))
    logger.debug(f"iter_examples_from_docbins: found {len(shard_paths)} shard(s) in {dir_path}")
    if shuffle:
        random.shuffle(shard_paths)

    count = 0
    for sp_i, sp_path in enumerate(shard_paths, start=1):
        db = DocBin().from_disk(sp_path)
        for d in db.get_docs(nlp.vocab):
            # For textcat, we use the cats dict directly
            yield Example.from_dict(nlp.make_doc(d.text), {"cats": d.cats})
            count += 1
            if max_docs is not None and count >= max_docs:
                logger.debug(f"max_docs={max_docs} reached, stopping iterator.")
                return


def sample_validation(nlp: Language, dir_path: Path, cap: int = 1500) -> List[Example]:
    out: List[Example] = []
    for eg in iter_examples_from_docbins(nlp, dir_path, shuffle=False):
        out.append(eg)
        if len(out) >= cap:
            break
    return out


def compounding_batch(epoch: int, total_epochs: int, start: int = 128, end: int = 256) -> int:
    """Compute batch size that compounds from start to end over epochs."""
    if total_epochs <= 1:
        return end
    r = epoch / (total_epochs - 1)
    return max(1, int(round(start * ((end / start) ** r))))


def gather_train_metadata(
    dir_path: Path,
    *,
    warm_limit: int = 256,
) -> Tuple[List[str], List[Tuple[str, dict]], int]:
    """Scan training docbins once to collect labels, warmup samples, and counts."""
    shard_paths = sorted(p for p in dir_path.glob("*.spacy"))
    if not shard_paths:
        raise RuntimeError(f"No .spacy shards found in {dir_path}")

    blank = spacy.blank("en")
    labels = set()
    warm_samples: List[Tuple[str, dict]] = []
    total_docs = 0

    for sp_path in shard_paths:
        db = DocBin().from_disk(sp_path)
        for doc in db.get_docs(blank.vocab):
            labels.update(doc.cats.keys())
            if len(warm_samples) < warm_limit:
                warm_samples.append((doc.text, dict(doc.cats)))
            total_docs += 1

    return sorted(labels), warm_samples, total_docs


def train_classifier_from_docbins(
    train_dir: Path | None = None,
    valid_dir: Path | None = None,
    out_model_dir: Path | None = None,
) -> Language:
    from .config import TRAIN as _TRAIN, OUT as _OUT
    import time

    train_dir = train_dir or _OUT.TRAIN_DIR
    valid_dir = valid_dir or _OUT.VALID_DIR
    out_model_dir = out_model_dir or _OUT.MODEL_DIR

    logger.debug(f"train_dir={train_dir}")
    logger.debug(f"valid_dir={valid_dir}")
    logger.debug(f"out_model_dir={out_model_dir}")

    warm_cap = min(256, max(16, 100))

    logger.info("Scanning training docbins for labels and warm-up samples (single pass)...")
    t_scan = time.time()
    all_labels, warm_samples, total_train_docs = gather_train_metadata(train_dir, warm_limit=warm_cap)
    logger.info(
        "Found %s canonical cuisine labels and %s training docs (scan %.1fs)",
        len(all_labels),
        f"{total_train_docs:,}",
        time.time() - t_scan,
    )
    if len(all_labels) > 10:
        logger.debug("Label sample: %s", all_labels[:10])

    # Configure device FIRST (before creating pipeline)
    configure_device()
    set_global_seed(_TRAIN.RANDOM_SEED)

    t0 = time.time()
    logger.debug("Choosing pipeline...")
    nlp, mode = choose_nlp(all_labels)
    
    # Log device info for transformer models
    if mode == "transformer" and torch is not None:
        try:
            trf = nlp.get_pipe("transformer")
            if hasattr(trf, "model"):
                model_ref = trf.model.get_ref("model")
                if hasattr(model_ref, "device"):
                    logger.info(f"[device] Transformer model device: {model_ref.device}")
                
                # Also check the underlying PyTorch model
                if hasattr(trf, "model") and hasattr(trf.model, "get_ref"):
                    try:
                        transformer_model = trf.model.get_ref("model")
                        if hasattr(transformer_model, "transformer"):
                            # Check the actual PyTorch model device
                            pytorch_model = transformer_model.transformer.model
                            if hasattr(pytorch_model, "device"):
                                logger.info(f"[device] PyTorch model device: {pytorch_model.device}")
                            # Check first parameter device
                            if hasattr(pytorch_model, "parameters"):
                                first_param = next(pytorch_model.parameters(), None)
                                if first_param is not None:
                                    logger.info(f"[device] Model parameters on device: {first_param.device}")
                    except Exception as e:
                        logger.debug(f"Could not inspect PyTorch model device: {e}")
        except Exception as e:
            logger.debug(f"Could not determine transformer device: {e}")
    
    logger.debug(f"choose_nlp() done in {time.time() - t0:.1f}s")
    logger.info(f"Pipeline mode: {mode}")
    
    # Log debug mode status
    if hasattr(_TRAIN, 'MAX_TRAIN_DOCS') and _TRAIN.MAX_TRAIN_DOCS is not None:
        logger.info(f"Debug mode: max_train_docs={_TRAIN.MAX_TRAIN_DOCS}")

    # Warm init using cached samples
    logger.debug("Building warm-up examples from cached samples...")
    warm: List[Example] = []
    for text, cats in warm_samples:
        warm.append(Example.from_dict(nlp.make_doc(text), {"cats": cats}))
    logger.debug("Prepared %s warm examples", len(warm))

    logger.debug("Calling nlp.initialize(...)")
    t0 = time.time()
    optimizer = nlp.initialize(lambda: warm)
    logger.debug(f"nlp.initialize(...) finished in {time.time() - t0:.1f}s")

    if hasattr(optimizer, "learn_rate"):
        optimizer.learn_rate = float(_TRAIN.LR)
        logger.debug(f"Set optimizer learn_rate={optimizer.learn_rate}")

    logger.debug("Building validation snapshot...")
    t0 = time.time()
    valid_snapshot = sample_validation(nlp, valid_dir, cap=_TRAIN.EVAL_SNAPSHOT_MAX)
    logger.debug(f"Validation snapshot size={len(valid_snapshot)} "
          f"built in {time.time() - t0:.1f}s")

    best_f1 = -1.0
    bad_epochs = 0

    # Determine whether to use multi-worker DataLoader
    num_workers = getattr(_TRAIN, 'DATA_LOADER_WORKERS', 0)
    if torch is not None and torch.cuda.is_available() and num_workers > 0:
        logger.warning(
            "Disabling DataLoader workers (%s) on GPU to avoid spaCy vector pickling errors; using single-threaded loader.",
            num_workers,
        )
        num_workers = 0

    # Silence HF tokenizers fork warnings
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    use_multiworker_loader = (
        DataLoader is not None and 
        IterableDataset is not None and 
        torch is not None and
        num_workers > 0
    )
    
    if use_multiworker_loader:
        logger.info(f"Using DataLoader with {num_workers} workers")
        # Windows multiprocessing fix
        import platform
        if platform.system() == "Windows":
            try:
                import multiprocessing
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
    else:
        logger.info("Using single-threaded data loading (no DataLoader workers)")
    
    # Helper for batching when not using DataLoader
    def batch_iter(source, sz):
        buf = []
        for item in source:
            buf.append(item)
            if len(buf) >= sz:
                yield buf
                buf = []
        if buf: 
            yield buf
    
    # Create reusable dataset (used for both single and multi-threaded loading)
    max_docs_limit = getattr(_TRAIN, 'MAX_TRAIN_DOCS', None)
    effective_train_docs = min(total_train_docs, max_docs_limit) if max_docs_limit else total_train_docs
    train_dataset = ShardedIterableDataset(
        vocab=nlp.vocab,
        dir_path=train_dir,
        shuffle=True,
        max_docs=max_docs_limit,
        seed=_TRAIN.RANDOM_SEED,
        task_type="textcat"
    )
    limit_msg = f", limited to {max_docs_limit:,} docs" if max_docs_limit else ""
    logger.info(
        "Using ShardedIterableDataset with %s shards (%s docs)%s",
        len(train_dataset.all_shard_paths),
        f"{total_train_docs:,}",
        limit_msg,
    )
    logger.info(
        "Effective training docs after cap: %s (raw=%s); labels=%s; batch_size=%s",
        f"{effective_train_docs:,}",
        f"{total_train_docs:,}",
        len(all_labels),
        getattr(_TRAIN, "BATCH_SIZE", "unset"),
    )
    
    configured_batch = max(1, int(getattr(_TRAIN, "BATCH_SIZE", 64)))
    
    for epoch in range(_TRAIN.N_EPOCHS):
        logger.info(f"===== Epoch {epoch + 1}/{_TRAIN.N_EPOCHS} =====")
        losses: dict = {}
        micro_bs = configured_batch
        logger.info("Using micro-batch size (μbs) = %s", micro_bs)
        updates = 0
        examples_seen = 0

        t_epoch = time.time()
        
        if use_multiworker_loader:
            # Use DataLoader with multiple workers
            train_dataset.set_epoch(epoch)
            
            loader = DataLoader(
                train_dataset,
                batch_size=micro_bs,
                shuffle=False,  # Shuffling handled by ShardedIterableDataset
                num_workers=num_workers,
                collate_fn=collate_examples,
                pin_memory=False,
                worker_init_fn=sharded_worker_init_fn,
                persistent_workers=True if num_workers > 0 else False,
            )
            data_source = loader
        else:
            # Single-threaded: Iterate dataset directly
            # This reuses the same sampling/shuffling logic as the dataset class
            train_dataset.set_epoch(epoch)
            data_source = batch_iter(train_dataset, micro_bs)
            
        for batch in data_source:
            nlp.update(batch, sgd=optimizer, drop=_TRAIN.DROPOUT, losses=losses)
            updates += 1
            examples_seen += len(batch)

            if updates % 20 == 0:
                elapsed = time.time() - t_epoch
                logger.info(
                    f"  Epoch {epoch+1} | Updates: {updates} | Examples: {examples_seen} | "
                    f"Loss: {losses.get('textcat', 0):.2f} | Time: {elapsed:.1f}s"
                )

            if (
                torch is not None
                and torch.cuda.is_available()
                and updates % _TRAIN.CLEAR_CACHE_EVERY == 0
            ):
                logger.debug("emptying CUDA cache...")
                torch.cuda.empty_cache()

        logger.debug(f"Finished epoch {epoch+1} updates={updates} "
              f"in {time.time() - t_epoch:.1f}s")

        logger.debug("Running evaluation on validation snapshot...")
        with nlp.select_pipes(disable=[p for p in nlp.pipe_names if p != "textcat"]):
            scores = nlp.evaluate(valid_snapshot)

        # For textcat, we look at textcat_score
        f1 = float(scores.get("textcat_score", 0.0))

        logger.info(
            f"Epoch {epoch + 1:02d}/{_TRAIN.N_EPOCHS} | μbs={micro_bs:<3d} "
            f"| loss={losses.get('textcat', 0):.1f} | F1={f1:.3f}"
        )

        improved = f1 > best_f1 + 1e-6
        if improved:
            best_f1 = f1
            bad_epochs = 0
            out_model_dir.mkdir(parents=True, exist_ok=True)
            nlp.to_disk(out_model_dir)
            logger.info(f"  ↳ Saved model-best → {out_model_dir} (F1={f1:.3f})")
        else:
            bad_epochs += 1
            if bad_epochs >= _TRAIN.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping after {bad_epochs} non-improving epoch(s).")
                break

    logger.info(f"Best F1 observed: {best_f1 if best_f1 >= 0 else 0.0}")
    return nlp


__all__ = [
    "build_nlp_transformer",
    "build_nlp_tok2vec",
    "choose_nlp",
    "iter_examples_from_docbins",
    "sample_validation",
    "compounding_batch",
    "train_classifier_from_docbins",
    "ShardedIterableDataset",
    "sharded_worker_init_fn",
]
