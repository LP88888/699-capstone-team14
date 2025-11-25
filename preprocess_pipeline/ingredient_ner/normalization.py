import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from .utils import normalize_token


def load_jsonl_map(path: Union[str, Path, None]) -> Dict[str, str]:
    """Load dedupe map from JSONL lines: {'from': '...', 'to': '...'}."""
    mapping: Dict[str, str] = {}
    if path is None:
        return mapping
    p = Path(path)
    if not p.exists():
        return mapping
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                src = normalize_token(str(obj.get("from", "")))
                dst = normalize_token(str(obj.get("to", "")))
                if src and dst:
                    mapping[src] = dst
    return mapping


def load_encoder_maps(
    id2tok_path: Optional[Path],
    tok2id_path: Optional[Path],
) -> Tuple[Optional[dict], Optional[dict]]:
    """Load IngredientEncoder id2tok / tok2id maps."""
    if not id2tok_path or not tok2id_path:
        return None, None
    if (not Path(id2tok_path).exists()) or (not Path(tok2id_path).exists()):
        return None, None
    with open(id2tok_path, "r", encoding="utf-8") as f:
        id2tok_raw = json.load(f)
    with open(tok2id_path, "r", encoding="utf-8") as f:
        tok2id_raw = json.load(f)
    id2tok = {int(k): str(v) for k, v in id2tok_raw.items()}
    tok2id = {str(k): int(v) for k, v in tok2id_raw.items()}
    return id2tok, tok2id


def apply_dedupe(tok: str, mapping: Optional[Dict[str, str]]) -> str:
    return mapping.get(tok, tok) if mapping else tok


__all__ = ["load_jsonl_map", "load_encoder_maps", "apply_dedupe"]