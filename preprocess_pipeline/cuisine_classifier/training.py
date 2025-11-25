import logging
import random
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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
        "max_batch_items": 4096,
    }
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


def compounding_batch(epoch: int, total_epochs: int, start: int = 8, end: int = 16) -> int:
    if total_epochs <= 1:
        return end
    r = epoch / (total_epochs - 1)
    return max(1, int(round(start * ((end / start) ** r))))


def count_examples_in_docbins(nlp: Language, dir_path: Path) -> int:
    total = 0
    shard_paths = sorted(p for p in dir_path.glob("*.spacy"))
    logger.debug(f"Counting examples in {dir_path}, found {len(shard_paths)} shard(s)")
    for sp_i, sp_path in enumerate(shard_paths, start=1):
        db = DocBin().from_disk(sp_path)
        n_docs = sum(1 for _ in db.get_docs(nlp.vocab))
        total += n_docs
        logger.debug(f"  shard #{sp_i}: {sp_path.name} has {n_docs} docs")
    logger.debug(f"Total train examples in {dir_path}: {total}")
    return total


def get_all_labels_from_docbins(dir_path: Path) -> List[str]:
    """Extract all unique cuisine labels from docbins."""
    all_labels = set()
    shard_paths = sorted(p for p in dir_path.glob("*.spacy"))
    blank = spacy.blank("en")
    
    for sp_path in shard_paths:
        db = DocBin().from_disk(sp_path)
        for d in db.get_docs(blank.vocab):
            all_labels.update(d.cats.keys())
    
    return sorted(list(all_labels))


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

    # Get all labels from training data
    logger.info("Extracting all cuisine labels from training data...")
    all_labels = get_all_labels_from_docbins(train_dir)
    logger.info(f"Found {len(all_labels)} unique cuisine labels: {all_labels[:10]}{'...' if len(all_labels) > 10 else ''}")

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

    # Warm init
    logger.debug("Collecting warm-up examples...")
    t0 = time.time()
    warm: List[Example] = []
    for eg_i, eg in enumerate(
        iter_examples_from_docbins(nlp, train_dir, shuffle=True),
        start=1,
    ):
        warm.append(eg)
        if eg_i % 50 == 0:
            logger.debug(f"warm example #{eg_i}")
        if eg_i >= min(256, max(16, 100)):
            break
    logger.debug(f"Collected {len(warm)} warm examples in {time.time() - t0:.1f}s")

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

    logger.debug("Sanity-checking training set size...")
    _ = count_examples_in_docbins(nlp, train_dir)

    for epoch in range(_TRAIN.N_EPOCHS):
        logger.info(f"===== Epoch {epoch + 1}/{_TRAIN.N_EPOCHS} =====")
        losses: dict = {}
        micro_bs = compounding_batch(epoch, _TRAIN.N_EPOCHS, start=8, end=16)
        logger.debug(f"micro-batch size (μbs) = {micro_bs}")
        buf: List[Example] = []
        updates = 0

        t_epoch = time.time()
        for eg_i, eg in enumerate(iter_examples_from_docbins(nlp, train_dir, shuffle=True), start=1):
            buf.append(eg)
            if len(buf) < micro_bs:
                continue
            nlp.update(buf, sgd=optimizer, drop=_TRAIN.DROPOUT, losses=losses)
            buf.clear()
            updates += 1

            if updates % 20 == 0:
                logger.debug(f"epoch={epoch+1} updates={updates} "
                      f"seen_examples≈{eg_i} loss={losses.get('textcat', 0):.2f}")

            if (torch is not None) and torch.cuda.is_available() \
                    and updates % _TRAIN.CLEAR_CACHE_EVERY == 0:
                logger.debug("emptying CUDA cache...")
                torch.cuda.empty_cache()

        if buf:
            logger.debug(f"Flushing last batch of size {len(buf)}")
            nlp.update(buf, sgd=optimizer, drop=_TRAIN.DROPOUT, losses=losses)
            buf.clear()

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
    "get_all_labels_from_docbins",
    "train_classifier_from_docbins",
]

