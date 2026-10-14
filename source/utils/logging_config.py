from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO, fmt: str = LOG_FORMAT) -> None:
    logging.basicConfig(level=level, format=fmt)
