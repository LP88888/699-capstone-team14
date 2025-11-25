
from __future__ import annotations
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any

def setup_logging(cfg: Optional[Dict[str, Any]] = None, *, force: bool = False) -> None:
    """Initialize root logging once. Safe to call multiple times if force=False."""
    if logging.getLogger().handlers and not force:
        return

    cfg = cfg or {}
    log_cfg = (cfg.get("logging") or {})

    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = log_cfg.get("fmt", "[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    datefmt = log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    handlers = []

    # Console
    if log_cfg.get("console", True):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handlers.append(ch)

    # Rotating file
    file_path = log_cfg.get("file")
    if file_path:
        fp = Path(file_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        rotate_cfg = log_cfg.get("rotate", {}) or {}
        max_bytes = int(rotate_cfg.get("max_bytes", 10_485_760))  # 10MB
        backup_count = int(rotate_cfg.get("backup_count", 5))
        fh = logging.handlers.RotatingFileHandler(
            filename=str(fp),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers)

    # Quiet noisy deps
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)