from __future__ import annotations

import logging

from source.config import load_config
from source.logic.reid.engine import ReIDEngine
from worker import VideoJobs

log = logging.getLogger("ps1.api")


class AppState:
    cfg: dict
    engine: ReIDEngine


S = AppState()


def init_state(config_path: str | None = None) -> None:
    S.cfg = load_config(config_path)
    S.engine = ReIDEngine(S.cfg)
    e = S.engine
    log.info("Gallery: %d images / %d identities (%s); ReID backend %s on %s",
             len(e.gallery), len(e.gallery.identities), e.consistency.get("status"), e.reid.name, e.reid.device)


def is_initialised() -> bool:
    return hasattr(S, "engine")


VIDEO_JOBS = VideoJobs(lambda: S.cfg, lambda: S.engine)
