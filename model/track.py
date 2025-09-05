import numpy as np
from collections import deque
from utils.inf_utils import iou_xyxy
# ======================
# 간단 트래커(다수결)
# ======================
class Track:
    __slots__ = ("tid","bbox","last_frame","votes","prev_smoothed","last_nohelmet_log_time","rider_recent")
    def __init__(self, tid, bbox, frame_idx, vote_window):
        self.tid = tid
        self.bbox = bbox.astype(np.float32)
        self.last_frame = frame_idx
        self.votes = deque(maxlen=vote_window)  # True/False/None
        self.prev_smoothed = None
        self.last_nohelmet_log_time = 0.0
        self.rider_recent = False

class SimpleHelmetTracker:
    def __init__(self, vote_window=5, vote_min_valid=3, vote_threshold=0.5, iou_thresh=0.3, max_age_frames=30):
        self.vote_window = int(vote_window)
        self.vote_min_valid = int(vote_min_valid)
        self.vote_threshold = float(vote_threshold)
        self.iou_thresh = float(iou_thresh)
        self.max_age_frames = int(max_age_frames)
        self.tracks = []
        self.next_tid = 1
        self.frame_idx = 0

    def _assign(self, dets):
        matches = {}  # det_idx -> track
        if not self.tracks or not dets:
            return matches, list(range(len(dets)))
        ious = np.zeros((len(self.tracks), len(dets)), dtype=np.float32)
        for ti, tr in enumerate(self.tracks):
            for di, db in enumerate(dets):
                ious[ti, di] = iou_xyxy(tr.bbox, db)
        pairs = [(ti, di, ious[ti, di]) for ti in range(len(self.tracks)) for di in range(len(dets)) if ious[ti, di] > self.iou_thresh]
        pairs.sort(key=lambda x: x[2], reverse=True)
        used_t = set(); used_d = set()
        for ti, di, _ in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            matches[di] = self.tracks[ti]
        unmatched_dets = [di for di in range(len(dets)) if di not in used_d]
        return matches, unmatched_dets

    def update(self, person_boxes, rider_flags, inst_has_helmet_list):
        self.frame_idx += 1
        dets = [np.array(b, dtype=np.float32) for b in person_boxes]
        matches, unmatched = self._assign(dets)
        for di, tr in matches.items():
            tr.bbox = dets[di]
            tr.last_frame = self.frame_idx
            tr.rider_recent = bool(rider_flags[di])
            tr.votes.append(bool(inst_has_helmet_list[di]) if tr.rider_recent else None)
        for di in unmatched:
            tr = Track(self.next_tid, dets[di], self.frame_idx, self.vote_window)
            tr.rider_recent = bool(rider_flags[di])
            tr.votes.append(bool(inst_has_helmet_list[di]) if tr.rider_recent else None)
            self.tracks.append(tr)
            matches[di] = tr
            self.next_tid += 1
        self.tracks = [tr for tr in self.tracks if self.frame_idx - tr.last_frame <= self.max_age_frames]
        det_to_out = {}
        for di, tr in matches.items():
            valid = [v for v in tr.votes if v is not None]
            smoothed = None
            if len(valid) >= self.vote_min_valid:
                helmet_ratio = sum(1 for v in valid if v) / float(len(valid))
                smoothed = (helmet_ratio >= self.vote_threshold)
            was_event = False
            if smoothed is not None and tr.prev_smoothed is not None:
                if (tr.prev_smoothed is True) and (smoothed is False):
                    was_event = True
            elif smoothed is False and tr.prev_smoothed is None:
                was_event = True
            tr.prev_smoothed = smoothed
            det_to_out[di] = (tr.tid, smoothed, was_event)
        return det_to_out