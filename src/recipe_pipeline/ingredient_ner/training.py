import logging
import random
import warnings
import time
import bisect
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

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
    from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
except ImportError:
    Dataset = None
    DataLoader = None
    IterableDataset = None
    get_worker_info = None


# ---------- Model Building ----------
def collate_examples(batch):
    """
    Identity function for DataLoader. 
    Must be defined at module level to be picklable on Windows.
    """
    return batch

def build_nlp_transformer() -> spacy.language.Language:
    """Build a small-window transformer + NER with optional layer freezing."""
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
        "max_batch_items": 4096,
    }
    nlp.add_pipe("transformer", config=trf_cfg)
    ner = nlp.add_pipe("ner")
    ner.add_label("INGREDIENT")

    if _TRAIN.FREEZE_LAYERS > 0:
        _freeze_transformer_layers(nlp, _TRAIN.FREEZE_LAYERS)
        
    return nlp


def _freeze_transformer_layers(nlp, k_layers):
    try:
        trf = nlp.get_pipe("transformer").model
        hf = trf.transformer.model
        blocks = None
        if hasattr(hf, "transformer") and hasattr(hf.transformer, "layer"):  # distilbert
            blocks = hf.transformer.layer
        elif hasattr(hf, "encoder") and hasattr(hf.encoder, "layer"):        # bert/roberta
            blocks = hf.encoder.layer
        
        if blocks is not None:
            k = min(k_layers, len(blocks))
            for i in range(k):
                for p in blocks[i].parameters():
                    p.requires_grad = False
            logger.info(f"[transformer] Froze {k} lower layer(s).")
    except Exception as e:
        warnings.warn(f"Could not freeze layers: {e}")


def build_nlp_tok2vec() -> spacy.language.Language:
    """CPU-friendly tok2vec + NER fallback."""
    nlp = spacy.blank("en")
    nlp.add_pipe("tok2vec")
    ner = nlp.add_pipe("ner")
    ner.add_label("INGREDIENT")
    logger.info("Using tok2vec fallback (no transformers).")
    return nlp


def choose_nlp() -> Tuple[Language, str]:
    from .config import TRAIN as _TRAIN
    
    # Debug override
    if hasattr(_TRAIN, 'USE_TOK2VEC_DEBUG') and _TRAIN.USE_TOK2VEC_DEBUG:
        logger.info("Debug mode: using tok2vec instead of transformers")
        return build_nlp_tok2vec(), "tok2vec"

    if torch is not None:
        try:
            import spacy_transformers  # noqa
            return build_nlp_transformer(), "transformer"
        except Exception as e:
            logger.warning(f"Falling back to tok2vec: {e}")
            return build_nlp_tok2vec(), "tok2vec"
            
    return build_nlp_tok2vec(), "tok2vec"


# ---------- Data Loading (The Optimized Part) ----------

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
        seed: int = 42
    ):
        """
        Initialize the ShardedIterableDataset.
        
        Args:
            vocab: spaCy Vocab object for deserializing docs
            dir_path: Directory containing .spacy shard files
            shuffle: Whether to shuffle shards and docs within shards
            max_docs: Optional limit on total documents to yield (via shard subsampling)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.vocab = vocab
        self.dir_path = dir_path
        self.shuffle = shuffle
        self.max_docs = max_docs
        self.base_seed = seed
        
        # Create blank nlp for Example creation (lightweight, picklable)
        self.blank_nlp = spacy.blank("en")
        
        # Get all shard paths
        self.all_shard_paths = sorted(p for p in dir_path.glob("*.spacy"))
        if not self.all_shard_paths:
            raise RuntimeError(f"No .spacy files found in {dir_path}")
        
        # These will be set by worker_init_fn for multi-worker support
        self._worker_id = 0
        self._num_workers = 1
        self._epoch = 0
        
        logger.debug(f"ShardedIterableDataset: {len(self.all_shard_paths)} shards in {dir_path}")
    
    def set_epoch(self, epoch: int):
        """Set epoch for deterministic shuffling across epochs."""
        self._epoch = epoch
    
    def _get_worker_shards(self) -> List[Path]:
        """Get the subset of shards assigned to this worker."""
        # When using DataLoader with num_workers > 0, split shards among workers
        worker_info = get_worker_info() if get_worker_info is not None else None
        
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = self._worker_id
            num_workers = self._num_workers
        
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
        
        For each shard:
        1. Load all docs into memory
        2. Shuffle docs locally (if shuffle=True)
        3. Yield Example objects one by one
        """
        # Get shards assigned to this worker
        shards = self._get_worker_shards()
        
        # Create local RNG for within-shard shuffling
        worker_info = get_worker_info() if get_worker_info is not None else None
        worker_id = worker_info.id if worker_info else 0
        local_rng = random.Random(self.base_seed + self._epoch + worker_id)
        
        docs_yielded = 0
        
        for shard_path in shards:
            # Check max_docs limit (safety break)
            if self.max_docs is not None and docs_yielded >= self.max_docs:
                break
            
            try:
                # Load shard into memory
                db = DocBin().from_disk(shard_path)
                docs = list(db.get_docs(self.vocab))
                
                # Shuffle docs within this shard
                if self.shuffle:
                    local_rng.shuffle(docs)
                
                # Yield Examples
                for doc in docs:
                    if self.max_docs is not None and docs_yielded >= self.max_docs:
                        break
                    
                    # Convert doc to Example
                    ents = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
                    example = Example.from_dict(
                        self.blank_nlp.make_doc(doc.text), 
                        {"entities": ents}
                    )
                    
                    yield example
                    docs_yielded += 1
                    
            except Exception as e:
                logger.warning(f"Error loading shard {shard_path}: {e}")
                continue
        
        logger.debug(f"Worker {worker_id}: yielded {docs_yielded} examples")


def sharded_worker_init_fn(worker_id: int):
    """
    Worker init function for DataLoader with ShardedIterableDataset.
    
    This is called in each worker process to set up worker-specific state.
    The actual shard splitting is handled inside the dataset's __iter__.
    """
    worker_info = get_worker_info()
    if worker_info is not None:
        # Set numpy/random seeds for this worker
        seed = worker_info.seed % (2**32)
        random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
        
        logger.debug(
            f"Worker {worker_id} initialized with seed {seed}, "
            f"dataset id: {id(worker_info.dataset)}"
        )
def iter_examples_from_docbins(nlp: Language, dir_path: Path, shuffle: bool = False) -> Iterable[Example]:
    """Fallback iterator for non-PyTorch environments."""
    shard_paths = sorted(p for p in dir_path.glob("*.spacy"))
    if shuffle:
        random.shuffle(shard_paths)

    for sp_path in shard_paths:
        db = DocBin().from_disk(sp_path)
        for d in db.get_docs(nlp.vocab):
            ents = [(e.start_char, e.end_char, e.label_) for e in d.ents]
            yield Example.from_dict(nlp.make_doc(d.text), {"entities": ents})


def compounding_batch(epoch: int, total_epochs: int, start: int = 128, end: int = 256) -> int:
    """Compute batch size that compounds from start to end over epochs."""
    if total_epochs <= 1: return end
    r = epoch / (total_epochs - 1)
    return max(1, int(round(start * ((end / start) ** r))))


def train_ner_from_docbins(
    train_dir: Path | None = None,
    valid_dir: Path | None = None,
    out_model_dir: Path | None = None,
) -> Language:
    from .config import TRAIN as _TRAIN, OUT as _OUT
    
    train_dir = train_dir or _OUT.TRAIN_DIR
    valid_dir = valid_dir or _OUT.VALID_DIR
    out_model_dir = out_model_dir or _OUT.MODEL_DIR

    configure_device()
    set_global_seed(_TRAIN.RANDOM_SEED)

    # 1. Setup Pipeline
    nlp, mode = choose_nlp()
    logger.info(f"Pipeline initialized in mode: {mode}")

    # 2. Warmup / Initialization
    # We grab a small set of examples to initialize the weights/labels
    logger.info("Initializing optimizer with warmup examples...")
    warmup_examples = []
    for i, eg in enumerate(iter_examples_from_docbins(nlp, train_dir, shuffle=True)):
        warmup_examples.append(eg)
        if i >= 100: break
    
    if not warmup_examples:
        raise ValueError(f"No training data found in {train_dir}")
        
    optimizer = nlp.initialize(lambda: warmup_examples)
    if hasattr(optimizer, "learn_rate"):
        optimizer.learn_rate = float(_TRAIN.LR)

    # 3. Validation Snapshot (Fixed size)
    logger.info("Creating validation snapshot...")
    valid_snapshot = []
    for i, eg in enumerate(iter_examples_from_docbins(nlp, valid_dir, shuffle=False)):
        valid_snapshot.append(eg)
        if i >= _TRAIN.EVAL_SNAPSHOT_MAX: break

    # 4. Training Loop
    best_f1 = -1.0
    bad_epochs = 0
    
    # Determine whether to use multi-worker DataLoader
    num_workers = _TRAIN.DATA_LOADER_WORKERS
    use_multiworker_loader = (
        DataLoader is not None and 
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
    
    # Log dataset size info
    shard_files = list(train_dir.glob("*.spacy"))
    n_shards = len(shard_files)
    max_docs = _TRAIN.MAX_TRAIN_DOCS
    estimated_total = n_shards * 2000  # Approximate based on shard_size
    if max_docs:
        logger.info(f"Training shards: {n_shards} (~{estimated_total:,} docs), limited to {max_docs:,} examples")
    else:
        logger.info(f"Training shards: {n_shards} (~{estimated_total:,} docs total)")
    
    logger.info(f"Starting training for {_TRAIN.N_EPOCHS} epochs...")
    
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
    
    # Create dataset (used for both single and multi-threaded loading)
    train_dataset = ShardedIterableDataset(
        vocab=nlp.vocab,
        dir_path=train_dir,
        shuffle=True,
        max_docs=_TRAIN.MAX_TRAIN_DOCS,
        seed=_TRAIN.RANDOM_SEED
    )
    
    for epoch in range(_TRAIN.N_EPOCHS):
        losses = {}
        micro_bs = compounding_batch(epoch, _TRAIN.N_EPOCHS, start=128, end=256)
        updates = 0
        
        # Prepare Data Source
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

        # Iterate Batches
        t0 = time.time()
        examples_seen = 0
        for batch in data_source:
            nlp.update(batch, sgd=optimizer, drop=_TRAIN.DROPOUT, losses=losses)
            updates += 1
            examples_seen += len(batch)
            
            # Progress logging every 50 updates
            if updates % 50 == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"  Epoch {epoch+1} | Updates: {updates} | Examples: {examples_seen} | "
                    f"Loss: {losses.get('ner', 0):.2f} | Time: {elapsed:.1f}s"
                )
            
            # CUDA Cache Clearing
            if updates % _TRAIN.CLEAR_CACHE_EVERY == 0 and torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluation
        with nlp.select_pipes(disable=[p for p in nlp.pipe_names if p != "ner"]):
            scores = nlp.evaluate(valid_snapshot)
        
        f1 = scores.get("ents_f", 0.0)
        p = scores.get("ents_p", 0.0)
        r = scores.get("ents_r", 0.0)
        
        logger.info(
            f"Epoch {epoch+1:02d} | Loss: {losses.get('ner', 0):.2f} | "
            f"F1: {f1:.3f} (P={p:.3f}, R={r:.3f}) | Time: {time.time()-t0:.1f}s"
        )
        
        # Early Stopping
        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            bad_epochs = 0
            out_model_dir.mkdir(parents=True, exist_ok=True)
            nlp.to_disk(out_model_dir)
            logger.info(f"  -> Saved best model to {out_model_dir}")
        else:
            bad_epochs += 1
            if bad_epochs >= _TRAIN.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

    return nlp

__all__ = [
    "train_ner_from_docbins",
    "ShardedIterableDataset",
    "sharded_worker_init_fn",
    "iter_examples_from_docbins",
]