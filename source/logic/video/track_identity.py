from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from source.logic.reid.open_set import UNKNOWN, OpenSetMatcher
from source.logic.reid.engine import ReIDEngine
from source.logic.video.tracker import Track

PENDING = "PENDING"


@dataclass
class IdentityPolicy:
    reid_interval: int = 3
    stable_interval: int = 15
    vote_window: int = 8
    min_votes: int = 2
    min_agreement: float = 0.6
    min_known_fraction: float = 0.25
    min_box_height: int = 64
    min_box_width: int = 24
    min_det_confidence: float = 0.4
    skip_truncated: bool = True
    max_overlap: float = 0.3
    reset_overlap: float = 0.5
    batch_size: int = 32

    @classmethod
    def from_config(cls, d: dict | None) -> "IdentityPolicy":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown identity policy settings: {sorted(unknown)}")
        p = cls(**d)
        if p.reid_interval < 1 or p.stable_interval < 1 or p.vote_window < 1 or p.min_votes < 1:
            raise ValueError("reid_interval, stable_interval, vote_window and min_votes must be >= 1")
        if p.min_votes > p.vote_window:
            raise ValueError("min_votes cannot exceed vote_window")
        return p


@dataclass
class Observation:
    frame_index: int
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    det_confidence: float
    decision: str
    identity: str
    top1_candidate: str
    top1_score: float
    top2_score: float | None
    margin: float | None
    reason: str


@dataclass
class TrackLabel:
    status: str
    identity: str
    score: float | None
    votes: int
    n_obs: int

    @property
    def text(self) -> str:
        if self.status == "KNOWN":
            return f"{self.identity} | Score: {self.score:.2f}"
        return "UNKNOWN (pending)" if self.status == PENDING else "UNKNOWN"


@dataclass
class TrackState:
    track_id: int
    first_frame: int
    last_frame: int
    frames_visible: int = 0
    total_obs: int = 0
    last_sample_pframe: int | None = None
    obs_since_change: int = 0
    label_changes: int = 0
    occluded: bool = False
    resets: int = 0
    window: deque = field(default_factory=deque)
    label: TrackLabel = field(default_factory=lambda: TrackLabel(PENDING, UNKNOWN, None, 0, 0))
    ever_known: Counter = field(default_factory=Counter)


def decide_label(window, p: IdentityPolicy) -> TrackLabel:
    n = len(window)
    if n < p.min_votes:
        return TrackLabel(PENDING, UNKNOWN, None, 0, n)
    accepted = [o for o in window if o.decision == "KNOWN"]
    if accepted:
        pid, v = Counter(o.identity for o in accepted).most_common(1)[0]
        if v >= p.min_votes and v / len(accepted) >= p.min_agreement and v / n >= p.min_known_fraction:
            score = float(np.mean([o.top1_score for o in accepted if o.identity == pid]))
            return TrackLabel("KNOWN", pid, score, v, n)
    return TrackLabel("UNKNOWN", UNKNOWN, None, 0, n)


def _overlap_share(box, others) -> float:
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1)
    best = 0.0
    for a1, b1, a2, b2 in others:
        iw, ih = min(x2, a2) - max(x1, a1), min(y2, b2) - max(y1, b1)
        if iw > 0 and ih > 0:
            best = max(best, iw * ih / area)
    return best


def crop_quality(track: Track, all_boxes: list, frame_shape, p: IdentityPolicy) -> str | None:
    x1, y1, x2, y2 = track.box
    h, w = frame_shape[:2]
    if (y2 - y1) < p.min_box_height or (x2 - x1) < p.min_box_width:
        return "too_small"
    if track.score < p.min_det_confidence:
        return "low_det_confidence"
    if p.skip_truncated and (x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1):
        return "truncated_at_border"
    if p.max_overlap < 1.0 and _overlap_share(track.box, [b for b in all_boxes if b != track.box]) > p.max_overlap:
        return "occluded_overlap"
    return None


class TrackIdentityManager:
    def __init__(self, engine: ReIDEngine, matcher: OpenSetMatcher | None = None,
                 policy: IdentityPolicy | None = None):
        self.engine = engine
        self.matcher = matcher or engine.matcher
        self.policy = policy or IdentityPolicy()
        self.states: dict[int, TrackState] = {}
        self.skipped = Counter()
        self.crop_sizes: list[tuple[int, int]] = []
        self.embed_batches = 0
        self.identity_resets = 0

    def _due(self, st: TrackState, pframe: int) -> bool:
        if st.last_sample_pframe is None:
            return True
        p = self.policy
        stable = len(st.window) >= p.vote_window and st.obs_since_change >= p.vote_window // 2
        return pframe - st.last_sample_pframe >= (p.stable_interval if stable else p.reid_interval)

    def step(self, frame_index: int, pframe: int, frame_rgb: np.ndarray,
             tracks: list[Track]) -> tuple[dict[int, TrackLabel], list[Observation]]:
        p = self.policy
        boxes = [t.box for t in tracks]
        due: list[Track] = []
        for t in tracks:
            st = self.states.get(t.track_id)
            if st is None:
                st = self.states[t.track_id] = TrackState(t.track_id, frame_index, frame_index,
                                                          window=deque(maxlen=p.vote_window))
            st.last_frame = frame_index
            st.frames_visible += 1
            if p.reset_overlap < 1.0:
                cover = _overlap_share(t.box, [b for b in boxes if b != t.box])
                if cover > p.reset_overlap:
                    st.occluded = True
                elif st.occluded and cover <= p.max_overlap:
                    st.occluded = False
                    if st.window:
                        st.window.clear()
                        st.label = TrackLabel(PENDING, UNKNOWN, None, 0, 0)
                        st.last_sample_pframe = None
                        st.obs_since_change = 0
                        st.resets += 1
                        self.identity_resets += 1
            if self._due(st, pframe):
                reason = crop_quality(t, boxes, frame_rgb.shape, p)
                if reason is None:
                    due.append(t)
                else:
                    self.skipped[reason] += 1

        new_obs: list[Observation] = []
        if due:
            crops = []
            for t in due:
                x1, y1, x2, y2 = t.box
                crops.append(Image.fromarray(np.ascontiguousarray(frame_rgb[y1:y2, x1:x2])))
                self.crop_sizes.append((y2 - y1, x2 - x1))
            embs = self.engine.embed(crops, batch_size=p.batch_size)
            self.embed_batches += 1
            for t, emb in zip(due, embs):
                m = self.engine.identify_embedding(emb, top_k=2, matcher=self.matcher)
                top1 = m.ranking.identity_ids[0] if len(m.ranking.identity_ids) else ""
                o = Observation(frame_index, t.track_id, *t.box, t.score, m.decision, format_pid(m.identity),
                                format_pid(top1), m.top1_score, m.top2_score, m.margin, m.reason)
                new_obs.append(o)
                st = self.states[t.track_id]
                st.window.append(o)
                st.total_obs += 1
                st.last_sample_pframe = pframe
                new = decide_label(st.window, p)
                if (new.status, new.identity) != (st.label.status, st.label.identity):
                    if st.label.status != PENDING:
                        st.label_changes += 1
                    st.obs_since_change = 0
                else:
                    st.obs_since_change += 1
                st.label = new

        labels = {}
        for t in tracks:
            st = self.states[t.track_id]
            labels[t.track_id] = st.label
            if st.label.status == "KNOWN":
                st.ever_known[st.label.identity] += 1
        return labels, new_obs

    def track_summaries(self) -> list[dict]:
        out = []
        for st in sorted(self.states.values(), key=lambda s: s.track_id):
            out.append({
                "track_id": st.track_id, "first_frame": st.first_frame, "last_frame": st.last_frame,
                "frames_visible": st.frames_visible, "reid_observations": st.total_obs,
                "final_status": st.label.status, "final_identity": st.label.identity,
                "final_score": None if st.label.score is None else round(st.label.score, 4),
                "label_changes": st.label_changes, "occlusion_resets": st.resets,
                "identities_shown": dict(st.ever_known),
            })
        return out


def format_pid(pid) -> str:
    if isinstance(pid, (int, np.integer)):
        return f"{int(pid):04d}"
    return str(pid)
