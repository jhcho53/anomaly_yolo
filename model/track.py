import numpy as np
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple
from utils.inf_utils import iou_xyxy


# ======================
# Simple tracker (majority vote)
# ======================
class Track:
    """
    Per-person track that accumulates helmet/no-helmet votes over recent frames.

    Attributes:
        tid: Unique track id.
        bbox: Latest bbox [x1, y1, x2, y2] (float32).
        last_frame: Index of the last frame where this track was updated.
        votes: Deque of recent helmet votes (True/False) or None when not a rider.
        prev_smoothed: Previous smoothed helmet state (True/False) or None if insufficient votes yet.
        last_nohelmet_log_time: Wall time (sec) of the last NoHelmet log for cooldown (managed by caller).
        rider_recent: Whether this detection was considered a rider in the latest update.
    """
    __slots__ = (
        "tid",
        "bbox",
        "last_frame",
        "votes",
        "prev_smoothed",
        "last_nohelmet_log_time",
        "rider_recent",
    )

    def __init__(self, tid: int, bbox: np.ndarray, frame_idx: int, vote_window: int) -> None:
        self.tid: int = tid
        self.bbox: np.ndarray = bbox.astype(np.float32)
        self.last_frame: int = frame_idx
        self.votes: Deque[Optional[bool]] = deque(maxlen=int(vote_window))
        self.prev_smoothed: Optional[bool] = None
        self.last_nohelmet_log_time: float = 0.0
        self.rider_recent: bool = False


class SimpleHelmetTracker:
    """
    Greedy IoU-based tracker + majority-vote smoothing for helmet state.

    Args:
        vote_window: Number of recent frames to keep for voting.
        vote_min_valid: Minimum number of non-None votes required to compute a smoothed state.
        vote_threshold: Ratio threshold (>=) for deciding Helmet when enough votes exist.
        iou_thresh: IoU threshold for greedy assignment (strictly greater than this threshold is considered a match).
        max_age_frames: Remove tracks not updated for this many frames.
    """

    def __init__(
        self,
        vote_window: int = 5,
        vote_min_valid: int = 3,
        vote_threshold: float = 0.5,
        iou_thresh: float = 0.3,
        max_age_frames: int = 30,
    ) -> None:
        self.vote_window = int(vote_window)
        self.vote_min_valid = int(vote_min_valid)
        self.vote_threshold = float(vote_threshold)
        self.iou_thresh = float(iou_thresh)
        self.max_age_frames = int(max_age_frames)

        self.tracks: List[Track] = []
        self.next_tid: int = 1
        self.frame_idx: int = 0

    # ---------------- Internal: greedy IoU assignment ----------------
    def _assign(self, dets: List[np.ndarray]) -> Tuple[Dict[int, Track], List[int]]:
        """
        Assign detections to existing tracks using greedy IoU matching.

        Returns:
            matches: dict {det_idx -> Track}
            unmatched_dets: list of detection indices not matched to any track
        """
        matches: Dict[int, Track] = {}

        if not self.tracks or not dets:
            return matches, list(range(len(dets)))

        ious = np.zeros((len(self.tracks), len(dets)), dtype=np.float32)
        for ti, tr in enumerate(self.tracks):
            for di, db in enumerate(dets):
                ious[ti, di] = iou_xyxy(tr.bbox, db)

        # Consider only pairs with IoU > threshold; sort by IoU descending
        pairs = [
            (ti, di, ious[ti, di])
            for ti in range(len(self.tracks))
            for di in range(len(dets))
            if ious[ti, di] > self.iou_thresh
        ]
        pairs.sort(key=lambda x: x[2], reverse=True)

        used_t, used_d = set(), set()
        for ti, di, _ in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches[di] = self.tracks[ti]

        unmatched_dets = [di for di in range(len(dets)) if di not in used_d]
        return matches, unmatched_dets

    # ---------------- Public API ----------------
    def update(
        self,
        person_boxes: Sequence[Sequence[float]],
        rider_flags: Sequence[bool],
        inst_has_helmet_list: Sequence[Optional[bool]],
    ) -> Dict[int, Tuple[int, Optional[bool], bool]]:
        """
        Update the tracker with current detections and votes.

        Args:
            person_boxes: List of person bboxes [x1, y1, x2, y2].
            rider_flags: List of booleans indicating whether the person is a rider.
            inst_has_helmet_list: List where each element is True/False (helmet vote) for riders, or None for non-riders.

        Returns:
            det_to_out: dict {det_idx -> (track_id, smoothed_state, was_event)}
                - smoothed_state: True/False when enough votes exist, else None
                - was_event: True only on transition Helmet→NoHelmet (or first-time NoHelmet from unknown)
        """
        self.frame_idx += 1

        # Defensive alignment: clamp to the shortest length to avoid index errors.
        n = min(len(person_boxes), len(rider_flags), len(inst_has_helmet_list))
        if n != len(person_boxes) or n != len(rider_flags) or n != len(inst_has_helmet_list):
            # Truncate silently; callers should keep inputs aligned.
            person_boxes = person_boxes[:n]
            rider_flags = rider_flags[:n]
            inst_has_helmet_list = inst_has_helmet_list[:n]

        dets = [np.array(b, dtype=np.float32) for b in person_boxes]
        matches, unmatched = self._assign(dets)

        # Update matched tracks
        for di, tr in matches.items():
            tr.bbox = dets[di]
            tr.last_frame = self.frame_idx
            tr.rider_recent = bool(rider_flags[di])
            tr.votes.append(bool(inst_has_helmet_list[di]) if tr.rider_recent else None)

        # Create new tracks for unmatched detections
        for di in unmatched:
            tr = Track(self.next_tid, dets[di], self.frame_idx, self.vote_window)
            tr.rider_recent = bool(rider_flags[di])
            tr.votes.append(bool(inst_has_helmet_list[di]) if tr.rider_recent else None)
            self.tracks.append(tr)
            matches[di] = tr
            self.next_tid += 1

        # Drop stale tracks
        self.tracks = [tr for tr in self.tracks if self.frame_idx - tr.last_frame <= self.max_age_frames]

        # Build output mapping and detect Helmet→NoHelmet transitions
        det_to_out: Dict[int, Tuple[int, Optional[bool], bool]] = {}
        for di, tr in matches.items():
            valid = [v for v in tr.votes if v is not None]
            smoothed: Optional[bool] = None
            if len(valid) >= self.vote_min_valid:
                helmet_ratio = sum(1 for v in valid if v) / float(len(valid))
                smoothed = (helmet_ratio >= self.vote_threshold)

            was_event = False
            # Trigger event on transition Helmet→NoHelmet, or first-time NoHelmet from unknown
            if smoothed is not None and tr.prev_smoothed is not None:
                if tr.prev_smoothed is True and smoothed is False:
                    was_event = True
            elif smoothed is False and tr.prev_smoothed is None:
                was_event = True

            tr.prev_smoothed = smoothed
            det_to_out[di] = (tr.tid, smoothed, was_event)

        return det_to_out
