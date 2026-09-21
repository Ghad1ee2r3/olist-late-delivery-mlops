"""Logging setup. Read level and file path from config, not from the code.

Two places: the terminal, and logs/service.log.
Same format in both so I can search one line later.
"""

import logging
from pathlib import Path

from src.config import load_config, resolve_path

LOGGER_NAME = "olist"


def setup_logging(cfg: dict = None) -> logging.Logger:
    if cfg is None:
        cfg = load_config()
    log_cfg = cfg.get("logging", {})
    level_name = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = log_cfg.get(
        "format",
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_path = resolve_path(log_cfg.get("log_file", "logs/service.log"))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "predict") -> logging.Logger:
    parent = logging.getLogger(LOGGER_NAME)
    if not parent.handlers:
        setup_logging()
    return parent.getChild(name)
