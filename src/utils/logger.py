from __future__ import annotations

import logging


LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def setup_logger(name: str = "stutter_detection") -> logging.Logger:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
