import os
import cv2
import csv
import json
import time
import argparse
import tempfile
import threading
import queue
from pathlib import Path
from collections import deque, defaultdict
from typing import Optional, List, Dict, Tuple

from ultralytics import YOLO
import torch

class HazardWorker(threading.Thread):
    """trash/fire/smoke/weapon + rider(2인 이상/헬멧 미착용) 처리."""
    def __init__(self, q: "queue.Queue", log_dir: Path,
                 vlm_enable: bool, vlm_model, vlm_tokenizer, gen_cfg_dict: dict,
                 vlm_segments: int, vlm_max_num: int, vlm_input_annotated: bool,
                 log_cooldown: float, nohelmet_cooldown: float,
                 nohelmet_hold: int, multi_rider_hold: int, multi_rider_cooldown: float,
                 hazard_require_crowd: bool,
                 crowd_state: dict, crowd_lock: threading.Lock,
                 stop_event: threading.Event):
        super().__init__(daemon=True)
        self.q = q
        self.log_dir = log_dir
        self.vlm_enable = vlm_enable
        self.vlm_model = vlm_model
        self.vlm_tokenizer = vlm_tokenizer
        self.gen_cfg_dict = gen_cfg_dict
        self.vlm_segments = vlm_segments
        self.vlm_max_num = vlm_max_num
        self.vlm_input_annotated = vlm_input_annotated

        self.log_cooldown = float(log_cooldown)
        self.nohelmet_cooldown = float(nohelmet_cooldown)
        self.nohelmet_hold = int(nohelmet_hold)
        self.multi_rider_hold = int(multi_rider_hold)
        self.multi_rider_cooldown = float(multi_rider_cooldown)

        self.hazard_require_crowd = bool(hazard_require_crowd)
        self.crowd_state = crowd_state
        self.crowd_lock = crowd_lock

        self.stop_event = stop_event

        # 내부 상태
        self.last_hazard_log_time = {
            "trash_bag": 0.0, "fire": 0.0, "smoke": 0.0, "weapon": 0.0
        }
        self._nohelmet_hold = defaultdict(int)    # tid -> hold count
        self._nohelmet_last_log = defaultdict(float)  # tid -> last log t
        self._multi_rider_hold = defaultdict(int)     # tuple(sorted tids) -> hold
        self._multi_rider_last_log = defaultdict(float)

    def _vlm_maybe(self, task, event_tag: str, extra: dict):
        if not self.vlm_enable:
            return
        # VLM은 heavy → 여기서만 직렬 처리
        frames = task["frames_vis"] if self.vlm_input_annotated else task["frames_raw"]
        fps = task["fps"]
        qa = _vlm_call_on_frames(
            self.vlm_model, self.vlm_tokenizer, frames,
            self.gen_cfg_dict, num_segments=self.vlm_segments,
            max_num=self.vlm_max_num, fps_for_fallback=fps
        )
        # JSONL append
        rec = {
            "stamp_iso": task["stamp_meta"]["iso"],
            "frame_idx": task["frame_idx"],
            "event": event_tag,
            "qa": qa,
        }
        rec.update(extra or {})
        try:
            with open(Path(self.log_dir) / "vlm_qa.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[VLM] jsonl append failed: {e}")

    def _crowd_ok(self) -> bool:
        if not self.hazard_require_crowd:
            return True
        with self.crowd_lock:
            return bool(self.crowd_state.get("smoothed", False))

    def run(self):
        while not self.stop_event.is_set() or not self.q.empty():
            try:
                task = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:
                self.q.task_done()
                break

            stamp_meta = task["stamp_meta"]; now_t = stamp_meta["t"]
            frame_idx = task["frame_idx"]
            total_persons = task["total_persons"]

            # --- (1) 일반 hazards ---
            if self._crowd_ok():
                for name, cnt in task["hazard_counts"].items():
                    if cnt <= 0:
                        continue
                    last_t = self.last_hazard_log_time.get(name, 0.0)
                    if (now_t - last_t) >= self.log_cooldown:
                        log_hazard(
                            log_dir=self.log_dir,
                            stamp_meta=stamp_meta,
                            event_type=name,
                            count=int(cnt),
                            persons=int(total_persons),
                        )
                        self.last_hazard_log_time[name] = now_t

            # --- (2) rider: no-helmet ---
            for tr in task["track_infos"]:  # dict: tid, smoothed_has_helmet, votes_valid, helmet_ratio, bbox
                tid = tr["tid"]
                smoothed_has_helmet = tr["smoothed_has_helmet"]
                if smoothed_has_helmet is False:
                    self._nohelmet_hold[tid] += 1
                    if self._nohelmet_hold[tid] >= max(1, self.nohelmet_hold) and \
                       (now_t - self._nohelmet_last_log[tid]) >= self.nohelmet_cooldown:
                        log_no_helmet(
                            log_dir=self.log_dir,
                            stamp_meta=stamp_meta,
                            track_id=int(tid),
                            bbox=tr["bbox"],
                            persons=int(total_persons),
                            vote_window=int(task["vote_window"]),
                            valid_votes=int(tr["votes_valid"]),
                            threshold=float(task["vote_threshold"]),
                            helmet_ratio=float(tr["helmet_ratio"]),
                        )
                        self._nohelmet_last_log[tid] = now_t
                        self._nohelmet_hold[tid] = 0

                        # VLM QA
                        if task["any_detection"]:
                            self._vlm_maybe(
                                task, event_tag="no_helmet",
                                extra={
                                    "persons": total_persons,
                                    "pm": task["pm_count"],
                                    "helmet": task["helmet_count"],
                                    "trash_bag": task["hazard_counts"]["trash_bag"],
                                    "fire": task["hazard_counts"]["fire"],
                                    "smoke": task["hazard_counts"]["smoke"],
                                    "weapon": task["hazard_counts"]["weapon"],
                                    "track_id": int(tid),
                                    "event_detail": "no_helmet_vote"
                                }
                            )
                else:
                    self._nohelmet_hold[tid] = 0

            # --- (3) rider: 2인 이상 탑승 ---
            # key = tuple(sorted rider tids) per PM (길이>=2)
            for pm_idx, rider_tids in task["pm_to_rider_tids"].items():
                tids = [t for t in rider_tids if t is not None]
                if len(tids) >= 2:
                    key = tuple(sorted(tids))
                    self._multi_rider_hold[key] += 1
                    if self._multi_rider_hold[key] >= max(1, task["multi_rider_hold"]) and \
                       (now_t - self._multi_rider_last_log[key]) >= task["multi_rider_cooldown"]:
                        log_multi_rider(
                            log_dir=self.log_dir,
                            stamp_meta=stamp_meta,
                            frame_idx=frame_idx,
                            pm_idx=int(pm_idx),
                            rider_tids=tids,
                            persons=int(total_persons)
                        )
                        self._multi_rider_last_log[key] = now_t
                        self._multi_rider_hold[key] = 0
                else:
                    # 홀드 초기화
                    key = tuple(sorted(tids)) if tids else None
                    if key in self._multi_rider_hold:
                        self._multi_rider_hold[key] = 0

            self.q.task_done()

# ======================
# Worker threads
# ======================
class CrowdWorker(threading.Thread):
    def __init__(self, q: "queue.Queue", log_dir: Path,
                 vote_window: int, vote_min_valid: int, vote_threshold: float,
                 cooldown: float, crowd_state: dict, crowd_lock: threading.Lock, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.q = q
        self.log_dir = log_dir
        self.voter = BinaryVoter(vote_window, vote_min_valid, vote_threshold, cooldown)
        self.crowd_state = crowd_state
        self.crowd_lock = crowd_lock
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set() or not self.q.empty():
            try:
                task = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if task is None:  # sentinel
                self.q.task_done()
                break
            stamp_meta = task["stamp_meta"]
            frame_idx = task["frame_idx"]
            total_persons = task["total_persons"]
            now_t = stamp_meta["t"]

            is_crowd = (total_persons >= task["crowd_person_threshold"])
            crowd_smoothed, crowd_start, cv_stats = self.voter.push(is_crowd, now_t)

            # 공유 crowd 상태 업데이트
            with self.crowd_lock:
                self.crowd_state["smoothed"] = bool(crowd_smoothed) if crowd_smoothed is not None else self.crowd_state.get("smoothed", False)
                self.crowd_state["last_update_t"] = now_t

            if crowd_start and self.voter.can_log(now_t):
                log_hazard(
                    log_dir=self.log_dir,
                    stamp_meta=stamp_meta,
                    event_type="crowd",
                    count=int(total_persons),
                    persons=int(total_persons),
                )
                log_crowd_vote(
                    log_dir=self.log_dir,
                    stamp_meta=stamp_meta,
                    frame_idx=frame_idx,
                    persons=total_persons,
                    vote_window=self.voter.window.maxlen,
                    valid=cv_stats["valid"],
                    threshold=self.voter.threshold,
                    true_ratio=cv_stats["ratio_true"],
                    smoothed=crowd_smoothed
                )
                self.voter.mark_logged(now_t)

            self.q.task_done()

# ======================
# Voting helpers
# ======================
class BinaryVoter:
    """최근 프레임 기반 이진 투표 스무딩 (window/min_valid/threshold + cooldown)."""
    def __init__(self, window: int, min_valid: int, threshold: float, cooldown: float):
        self.window = deque(maxlen=int(window))
        self.min_valid = int(min_valid)
        self.threshold = float(threshold)
        self.cooldown = float(cooldown)
        self.last_smoothed = None   # True/False/None
        self.last_log_t = 0.0

    def push(self, value: Optional[bool], now_t: float):
        self.window.append(value)
        valid = [v for v in self.window if v is not None]
        true_cnt = sum(1 for v in valid if v)
        ratio_true = (true_cnt / len(valid)) if len(valid) > 0 else 0.0
        smoothed = None
        if len(valid) >= self.min_valid:
            smoothed = (ratio_true >= self.threshold)
        was_event = (smoothed is True) and (self.last_smoothed is not True)
        self.last_smoothed = smoothed if smoothed is not None else self.last_smoothed
        stats = {"valid": len(valid), "true_cnt": true_cnt, "ratio_true": ratio_true}
        return self.last_smoothed, was_event, stats

    def can_log(self, now_t: float) -> bool:
        return (now_t - self.last_log_t) >= self.cooldown

    def mark_logged(self, now_t: float):
        self.last_log_t = now_t


def _append_csv_line(csv_path: Path, header: list, row: list):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)


def log_crowd_vote(log_dir: Path, stamp_meta: dict, frame_idx: int, persons: int,
                   vote_window: int, valid: int, threshold: float,
                   true_ratio: float, smoothed: Optional[bool]):
    header = ["stamp_iso", "frame_idx", "persons", "vote_window", "valid",
              "threshold", "true_ratio", "smoothed"]
    row = [stamp_meta.get("iso", ""), int(frame_idx), int(persons),
           int(vote_window), int(valid), float(threshold),
           float(true_ratio), "" if smoothed is None else int(bool(smoothed))]
    _append_csv_line(Path(log_dir) / "crowd_vote.csv", header, row)


def log_multi_rider(log_dir: Path, stamp_meta: dict, frame_idx: int,
                    pm_idx: int, rider_tids: List[int], persons: int):
    """2인 이상 탑승 이벤트 로거."""
    header = ["stamp_iso", "frame_idx", "pm_idx", "rider_tids", "persons"]
    row = [stamp_meta.get("iso", ""), int(frame_idx), int(pm_idx),
           ";".join(map(str, rider_tids)), int(persons)]
    _append_csv_line(Path(log_dir) / "rider_multi.csv", header, row)


