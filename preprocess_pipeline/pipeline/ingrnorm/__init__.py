
from .io import materialize_parquet_source
from .parquet_utils import vocab_from_parquet_listcol
from .spacy_normalizer import apply_spacy_normalizer_to_parquet, SpacyIngredientNormalizer
from .w2v_dedupe import w2v_dedupe
from .sbert_dedupe import sbert_dedupe
from .dedupe_map import load_jsonl_map, write_jsonl_map, apply_map_to_parquet_streaming
from .encoder import IngredientEncoder

__all__ = [
    "materialize_parquet_source",
    "vocab_from_parquet_listcol",
    "apply_spacy_normalizer_to_parquet",
    "SpacyIngredientNormalizer",
    "w2v_dedupe",
    "sbert_dedupe",
    "load_jsonl_map",
    "write_jsonl_map",
    "apply_map_to_parquet_streaming",
    "IngredientEncoder",
]