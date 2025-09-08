#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inf_video.py — Local video → YOLO one-pass + parallel voting/logging + VLM QA (frames→video fallback)

- Auto-supports 4/7-class weights (person/pm/trash_bag/helmet[/fire/smoke/weapon])
- Producer(메인) → Consumers(스레드):
  * CrowdWorker: crowd 전담 (투표 스무딩 + crowd 로깅)
  * HazardWorker: trash/fire/smoke/weapon + rider(2인 이상/헬멧 미착용) 로깅 + no-helmet VLM
- InternVL (Only) loader: utils.video_vlm.init_model
- Events trigger VLM with recent frames (frames→video fallback 내장)
- JSONL: runs/.../logs/vlm_qa.jsonl
"""

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

# ====== Torch / NMS fallback ======
def _install_torchvision_nms_fallback(force: bool = False) -> bool:
    try:
        import torchvision
    except Exception as e:
        print(f"[WARN] torchvision import failed: {e} -> using pure PyTorch NMS fallback")
        torchvision = None

    def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.float(); b = b.float()
        tl = torch.max(a[:, None, :2], b[None, :, :2])
        br = torch.min(a[:, None, 2:], b[None, :, 2:])
        wh = (br - tl).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter + 1e-7
        return inter / union

    def _nms_fallback(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.new_zeros((0,), dtype=torch.long)
        device = boxes.device
        boxes = boxes.to(dtype=torch.float32)
        scores = scores.to(dtype=torch.float32)
        order = torch.argsort(scores, descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            ious = _box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]
            remain = (ious <= float(iou_thres)).nonzero(as_tuple=False).squeeze(1)
            order = order[1:][remain]
        return torch.tensor(keep, dtype=torch.long, device=device)

    patched = False
    if force:
        try:
            import torchvision
            torchvision.ops.nms = _nms_fallback; patched = True
        except Exception:
            patched = True
    else:
        try:
            import torchvision
            try:
                _ = torchvision.ops.nms(torch.zeros((1, 4)), torch.zeros((1,)), 0.5)
                patched = False
            except Exception:
                torchvision.ops.nms = _nms_fallback; patched = True
        except Exception:
            patched = True

    if patched:
        print("[WARN] Using pure-PyTorch NMS fallback (torchvision C++ ops unavailable).")
    else:
        print("[INFO] Using torchvision built-in NMS (C++ ops).")
    return patched

_install_torchvision_nms_fallback()
# ---------------------------------------------------------------

# ====== VLM import (InternVL only) ======
from utils.video_vlm import (
    init_model as vlm_init_model,
    run_frames_inference,
    run_video_inference,
)

# ====== Logging / utilities ======
from utils.log import _video_stamp_meta, log_hazard, log_no_helmet
from utils.inf_utils import (
    expand_box, iou_xyxy, draw_label,
    _expand_person_roi, _helmet_center_in_head_region,
    resolve_class_ids, _expand_box_xy
)
from model.track import SimpleHelmetTracker
from model.worker import HazardWorker, CrowdWorker, BinaryVoter
# ======================
# Defaults (constants)
# ======================
DEFAULT_WEIGHTS = "Pretrained/4_class.pt"
DEFAULT_SOURCE = "input.mp4"
DEFAULT_OUT_DIR = "runs/parallel_video"

DEFAULT_CONF_THRES = 0.25
DEFAULT_IOU_THRES  = 0.5

# ---- Rider association tuning
DEFAULT_RIDER_X_EXPAND = 1.15
DEFAULT_RIDER_Y_EXPAND = 1.90
DEFAULT_RIDER_IOU_THRES = 0.05
DEFAULT_RIDER_XCENTER_FRAC = 0.6
DEFAULT_MAX_RIDERS_PER_PM = 2  # 2인 이상 탑승 검출 허용 수

DEFAULT_HEAD_REGION_RATIO = 0.6
DEFAULT_ROI_TOP_EXTRA = 0.2
DEFAULT_ROI_SIDE_EXTRA = 0.05

DEFAULT_PERSON_DRAW_MIN_COUNT = 6
DEFAULT_CROWD_PERSON_THRESHOLD = 6

# Helmet per-track voting (SimpleHelmetTracker 내부)
DEFAULT_VOTE_WINDOW = 5
DEFAULT_VOTE_MIN_VALID = 3
DEFAULT_VOTE_THRESHOLD = 0.5
DEFAULT_TRACK_IOU_THRESH = 0.3
DEFAULT_TRACK_MAX_AGE_FRAMES = 30

# Logging cooldown / holds
DEFAULT_LOG_COOLDOWN_SEC = 5.0
DEFAULT_NO_HELMET_COOLDOWN_SEC = 10.0
DEFAULT_NOHELMET_HOLD = 2
DEFAULT_MULTI_RIDER_HOLD = 2
DEFAULT_MULTI_RIDER_COOLDOWN_SEC = 10.0

# 병렬 제어
DEFAULT_HAZARD_REQUIRE_CROWD = 0  # 1이면 crowd=True일 때만 hazard 로깅

DEFAULT_DEVICE = ""        # "", "cpu", "cuda:0"
DEFAULT_PREVIEW = False
DEFAULT_IMGSZ = 0
DEFAULT_OUT_SUFFIX = "_out.mp4"

# ---- VLM (InternVL Only)
DEFAULT_VLM_ENABLE = True
DEFAULT_VLM_MODEL  = "OpenGVLab/InternVL3-1B"
DEFAULT_VLM_DEVICE_MAP = "auto"
DEFAULT_VLM_MAXTOK = 64
DEFAULT_VLM_COND   = "event"        # "event" | "any"
DEFAULT_VLM_LANG   = "ko"           # "ko" | "en"
DEFAULT_VLM_SAVE_IMG = True
DEFAULT_VLM_MIN_INTERVAL = 1.5
DEFAULT_VLM_CLIP_SEC = 2.0
DEFAULT_VLM_SEGMENTS = 8
DEFAULT_VLM_MAX_NUM  = 1
DEFAULT_VLM_INPUT_ANNOTATED = 0

# Crowd voting
DEFAULT_CROWD_VOTE_WINDOW = 15
DEFAULT_CROWD_VOTE_MIN_VALID = 7
DEFAULT_CROWD_VOTE_THRESHOLD = 0.6

# Visualization colors (BGR)
COLOR_PM      = (255, 160, 0)
COLOR_PERSON  = (0, 200, 0)
COLOR_RIDER   = (0, 140, 255)
COLOR_TRASH   = (180, 180, 30)
COLOR_FIRE    = (0, 0, 255)
COLOR_SMOKE   = (160, 160, 160)
COLOR_WEAPON  = (180, 0, 180)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ======================
# VLM helpers (with frames→video fallback)
# ======================
def _write_frames_to_temp_video(frames: List, fps: float) -> str:
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = tmp.name
    tmp.close()
    writer = cv2.VideoWriter(path, fourcc, max(fps, 1.0), (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def _vlm_call_on_frames(
    vlm_model, vlm_tokenizer, frames: List,
    gen_cfg_dict: dict, num_segments=8, max_num=1, fps_for_fallback=30.0
):
    if not frames:
        return []
    cfg_no_cache = {k: v for k, v in (gen_cfg_dict or {}).items() if k != "use_cache"}
    cfg_with_cache = dict(cfg_no_cache, use_cache=True)
    cfg_candidates = [cfg_no_cache, cfg_with_cache]

    def _try_frames(cfg):
        try:
            return run_frames_inference(
                model=vlm_model, tokenizer=vlm_tokenizer,
                frames=frames, generation_config=cfg,
                num_segments=num_segments, max_num=max_num
            )
        except Exception as e:
            print(f"[VLM] frames inference failed ({'use_cache' in cfg}): {e}")
            return None

    def _try_video(cfg):
        try:
            tmp = _write_frames_to_temp_video(frames, fps_for_fallback)
            try:
                return run_video_inference(
                    model=vlm_model, tokenizer=vlm_tokenizer,
                    video_path=tmp, generation_config=cfg,
                    num_segments=num_segments, max_num=max_num
                )
            finally:
                try: os.remove(tmp)
                except Exception: pass
        except Exception as e:
            print(f"[VLM] video inference failed ({'use_cache' in cfg}): {e}")
            return None

    for cfg in cfg_candidates:
        qa = _try_frames(cfg)
        if qa: return qa
    for cfg in cfg_candidates:
        qa = _try_video(cfg)
        if qa: return qa
    print("[VLM] caption failed: all attempts exhausted")
    return []

# ======================
# Video inference app
# ======================
class VideoPMHelmetApp:
    def __init__(self, args):
        # I/O
        self.weights = args.weights
        self.source = args.source
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_suffix = args.out_suffix

        # YOLO thresholds
        self.conf_thres = args.conf_thres
        self.iou_thres  = args.iou_thres

        # ROI/head region parameters
        self.head_region_ratio = args.head_region_ratio
        self.roi_top_extra     = args.roi_top_extra
        self.roi_side_extra    = args.roi_side_extra

        # Crowd visualization threshold
        self.person_draw_min_count  = args.person_draw_min_count
        self.crowd_person_threshold = args.crowd_person_threshold

        # Tracker / voting parameters
        self.vote_window = args.vote_window
        self.vote_min_valid = args.vote_min_valid
        self.vote_threshold = args.vote_threshold
        self.track_iou_thresh = args.track_iou_thresh
        self.track_max_age_frames = args.track_max_age_frames

        # Logging
        self.log_dir = self.out_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_cooldown = args.log_cooldown_sec
        self.nohelmet_cooldown = args.no_helmet_cooldown_sec

        # Inference device / display / size
        self.device = args.device
        self.preview = bool(args.preview)
        self.imgsz = args.imgsz

        # VLM settings
        self.vlm_enable = bool(args.vlm_enable)
        self.vlm_model_name  = args.vlm_model
        if args.vlm_device_map == "auto":
            self.vlm_device_map = "auto"
        elif args.vlm_device_map == "None":
            self.vlm_device_map = None
        else:
            try:
                self.vlm_device_map = json.loads(args.vlm_device_map)
            except Exception:
                self.vlm_device_map = args.vlm_device_map

        self.vlm_maxtok = args.vlm_max_new_tokens
        self.vlm_cond   = args.vlm_condition.lower().strip()
        self.vlm_lang   = args.vlm_lang
        self.vlm_save_img = bool(args.vlm_save_annotated)
        self.vlm_min_interval = float(args.vlm_min_interval)
        self.vlm_clip_sec = float(args.vlm_clip_sec)
        self.vlm_segments = int(args.vlm_segments)
        self.vlm_max_num  = int(args.vlm_max_num)
        self.vlm_input_annotated = bool(args.vlm_input_annotated)

        self.vlm_images_dir = self.out_dir / "vlm_images"
        self.vlm_images_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_jsonl_path = self.log_dir / "vlm_qa.jsonl"
        self._last_caption_t = -1e9
        self._video_stamp_meta = _video_stamp_meta

        # YOLO model
        print(f"[INFO] Loading YOLO weights: {self.weights}")
        self.model = YOLO(self.weights)
        if self.device:
            try:
                self.model.to(self.device)
                print(f"[INFO] Moved model to device: {self.device}")
            except Exception as e:
                print(f"[WARN] Failed to move model to '{self.device}': {e}")

        # Class id mapping (auto 4/7 support)
        self.ids = resolve_class_ids(self.model)
        print(f"[INFO] Resolved class ids: {self.ids}")

        # Tracker
        self.tracker = SimpleHelmetTracker(
            vote_window=self.vote_window,
            vote_min_valid=self.vote_min_valid,
            vote_threshold=self.vote_threshold,
            iou_thresh=self.track_iou_thresh,
            max_age_frames=self.track_max_age_frames,
        )

        # Rider association params
        self.rider_x_expand      = float(args.rider_x_expand)
        self.rider_y_expand      = float(args.rider_y_expand)
        self.rider_iou_thres     = float(args.rider_iou_thres)
        self.rider_xcenter_frac  = float(args.rider_xcenter_frac)
        self.max_riders_per_pm   = int(args.max_riders_per_pm)

        # 병렬/홀드/쿨다운
        self.nohelmet_hold = int(args.nohelmet_hold)
        self.multi_rider_hold = int(args.multi_rider_hold)
        self.multi_rider_cooldown = float(args.multi_rider_cooldown_sec)
        self.hazard_require_crowd = bool(args.hazard_require_crowd)

        # VLM loader (InternVL Only)
        self.vlm_model = None
        self.vlm_tokenizer = None
        self.gen_cfg_dict = dict(
            max_new_tokens=int(self.vlm_maxtok),
            do_sample=False,
            top_p=1.0,
            num_beams=1,
            repetition_penalty=1.05,
        )
        if self.vlm_enable:
            try:
                self.vlm_model, self.vlm_tokenizer = vlm_init_model(self.vlm_model_name, device_map=self.vlm_device_map)
                print(f"[VLM] Loaded via video_vlm.init_model: {self.vlm_model_name} (device_map={self.vlm_device_map})")
            except Exception as e:
                print(f"[VLM] load failed: {e}")
                self.vlm_enable = False

        # ---- Parallel queues/workers ----
        self.stop_event = threading.Event()
        self.crowd_q: "queue.Queue" = queue.Queue(maxsize=64)
        self.hazard_q: "queue.Queue" = queue.Queue(maxsize=64)

        # crowd shared state
        self._crowd_state = {"smoothed": False, "last_update_t": 0.0}
        self._crowd_lock = threading.Lock()

        self.crowd_worker = CrowdWorker(
            q=self.crowd_q, log_dir=self.log_dir,
            vote_window=args.crowd_vote_window,
            vote_min_valid=args.crowd_vote_min_valid,
            vote_threshold=args.crowd_vote_threshold,
            cooldown=self.log_cooldown,
            crowd_state=self._crowd_state, crowd_lock=self._crowd_lock,
            stop_event=self.stop_event
        )
        self.hazard_worker = HazardWorker(
            q=self.hazard_q, log_dir=self.log_dir,
            vlm_enable=self.vlm_enable, vlm_model=self.vlm_model, vlm_tokenizer=self.vlm_tokenizer,
            gen_cfg_dict=self.gen_cfg_dict,
            vlm_segments=self.vlm_segments, vlm_max_num=self.vlm_max_num,
            vlm_input_annotated=self.vlm_input_annotated,
            log_cooldown=self.log_cooldown, nohelmet_cooldown=self.nohelmet_cooldown,
            nohelmet_hold=self.nohelmet_hold,
            multi_rider_hold=self.multi_rider_hold,
            multi_rider_cooldown=self.multi_rider_cooldown,
            hazard_require_crowd=self.hazard_require_crowd,
            crowd_state=self._crowd_state, crowd_lock=self._crowd_lock,
            stop_event=self.stop_event
        )
        self.crowd_worker.start()
        self.hazard_worker.start()

    def _save_img_for_vlm(self, img_bgr, vis_bgr, stem):
        img_path = self.vlm_images_dir / f"{stem}.jpg"
        try:
            cv2.imwrite(str(img_path), vis_bgr if self.vlm_save_img else img_bgr)
            return str(img_path)
        except Exception:
            return ""

    # ---------- Rider association allowing up to K riders per PM ----------
    def _associate_riders(self, pm_boxes, person_boxes, w, h) -> Tuple[List[bool], Dict[int, List[int]]]:
        """
        Returns:
          rider_flags: [len(person_boxes)] bool
          pm_to_person_idxs: {pm_idx: [person_idx,...]}
        """
        rider_flags = [False] * len(person_boxes)
        pm_to_person = {i: [] for i in range(len(pm_boxes))}
        assigned_person = set()

        for pm_idx, pm in enumerate(pm_boxes):
            pm_e = _expand_box_xy(pm, self.rider_x_expand, self.rider_y_expand, w, h)
            pm_cx = 0.5 * (pm[0] + pm[2])
            pm_w  = max(1.0, (pm[2] - pm[0]))
            xc_tol = 0.5 * pm_w * max(0.0, min(1.5, self.rider_xcenter_frac))

            candidates = []
            for i, pb in enumerate(person_boxes):
                if i in assigned_person:
                    continue
                px_cx = 0.5 * (pb[0] + pb[2])
                if abs(px_cx - pm_cx) > xc_tol:
                    continue
                # require vertical overlap
                if pb[3] < pm[1] or pb[1] > pm[3]:
                    continue
                iou = iou_xyxy(pm_e, pb)
                if iou < self.rider_iou_thres:
                    continue
                score = iou - 0.001 * (abs(px_cx - pm_cx) / pm_w)
                candidates.append((score, i))

            candidates.sort(reverse=True)
            for _, i in candidates:
                if i in assigned_person:
                    continue
                pm_to_person[pm_idx].append(i)
                rider_flags[i] = True
                assigned_person.add(i)
                if len(pm_to_person[pm_idx]) >= self.max_riders_per_pm:
                    break

        return rider_flags, pm_to_person

    # ---------- Per-frame inference ----------
    def run_inference(self, frame):
        """Run YOLO + rider/helmet logic and return (annotated_frame, stats, raw_frame)."""
        h, w = frame.shape[:2]

        keep_ids = sorted({v for v in self.ids.values() if v is not None})
        pred_kwargs = dict(conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        if keep_ids: pred_kwargs["classes"] = keep_ids
        if self.imgsz and self.imgsz > 0: pred_kwargs["imgsz"] = self.imgsz

        res = self.model.predict(frame, **pred_kwargs)[0]
        if res.boxes is None or len(res.boxes) == 0:
            stats = {
                "total_persons": 0,
                "det_to_track": {},
                "person_boxes": [],
                "rider_flags": [],
                "pm_to_person": {},
                "inst_has_helmet": [],
                "trash_count": 0,
                "fire_count": 0,
                "smoke_count": 0,
                "weapon_count": 0,
                "pm_count": 0,
                "helmet_count": 0,
            }
            return frame, [], [], [], [], [], [], stats, frame

        xyxy = res.boxes.xyxy.cpu().numpy()
        cls  = res.boxes.cls.cpu().numpy().astype(int)

        get = lambda key: self.ids.get(key, None)
        pm_boxes      = [xyxy[i] for i, c in enumerate(cls) if c == get("pm")]
        person_boxes  = [xyxy[i] for i, c in enumerate(cls) if c == get("person")]
        helmet_boxes  = [xyxy[i] for i, c in enumerate(cls) if c == get("helmet")]
        trash_boxes   = [xyxy[i] for i, c in enumerate(cls) if c == get("trash_bag")] if get("trash_bag") is not None else []
        fire_boxes    = [xyxy[i] for i, c in enumerate(cls) if c == get("fire")]       if get("fire") is not None else []
        smoke_boxes   = [xyxy[i] for i, c in enumerate(cls) if c == get("smoke")]      if get("smoke") is not None else []
        weapon_boxes  = [xyxy[i] for i, c in enumerate(cls) if c == get("weapon")]     if get("weapon") is not None else []

        # Rider association (allow up to K per PM)
        rider_flags, pm_to_person = self._associate_riders(pm_boxes, person_boxes, w, h)

        # Instant helmet decision (riders only)
        inst_has_helmet = [False] * len(person_boxes)
        if len(helmet_boxes) > 0:
            for i, (pb, is_rider) in enumerate(zip(person_boxes, rider_flags)):
                if not is_rider:
                    continue
                roi = _expand_person_roi(pb, w, h, self.roi_top_extra, self.roi_side_extra)
                for hb in helmet_boxes:
                    if _helmet_center_in_head_region(hb, roi, self.head_region_ratio):
                        inst_has_helmet[i] = True
                        break

        # Majority-vote smoothing (tracker)
        det_to_track = self.tracker.update(person_boxes, rider_flags, inst_has_helmet)

        # Visualization — PM
        vis = frame.copy()
        for pm in pm_boxes:
            x1, y1, x2, y2 = pm.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PM, 2)
            draw_label(vis, x1, y1, "PM", COLOR_PM)

        total_persons = len(person_boxes)

        # Visualization — Rider
        for det_idx, (pb, is_rider) in enumerate(zip(person_boxes, rider_flags)):
            if not is_rider:
                continue
            x1, y1, x2, y2 = pb.astype(int)
            tid, smoothed, _ = det_to_track.get(det_idx, (None, None, False))
            show_has_helmet = smoothed if smoothed is not None else inst_has_helmet[det_idx]
            label = "Rider"
            if tid is not None:
                label += f"#{tid}"
            label += " | " + ("Helmet" if show_has_helmet else "NoHelmet")
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_RIDER, 2)
            draw_label(vis, x1, y1, label, COLOR_RIDER)

        # Visualization — general Person (only when #people is large)
        if total_persons >= self.person_draw_min_count:
            for (pb, is_rider) in zip(person_boxes, rider_flags):
                if is_rider: continue
                x1, y1, x2, y2 = pb.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PERSON, 2)
                draw_label(vis, x1, y1, "Person", COLOR_PERSON)
            text = f"Persons: {total_persons}"
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
            x0 = w - tw - 16; y0 = 16 + th + 8
            cv2.rectangle(vis, (x0 - 8, 8), (x0 + tw + 8, y0), (50, 50, 50), -1)
            cv2.putText(vis, text, (x0, 16 + th), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Visualization — other classes
        for tb in trash_boxes:
            x1, y1, x2, y2 = tb.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_TRASH, 2)
            draw_label(vis, x1, y1, "Trash", COLOR_TRASH)
        for fb in fire_boxes:
            x1, y1, x2, y2 = fb.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_FIRE, 2)
            draw_label(vis, x1, y1, "Fire", COLOR_FIRE)
        for sb in smoke_boxes:
            x1, y1, x2, y2 = sb.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_SMOKE, 2)
            draw_label(vis, x1, y1, "Smoke", COLOR_SMOKE)
        for wb in weapon_boxes:
            x1, y1, x2, y2 = wb.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_WEAPON, 2)
            draw_label(vis, x1, y1, "Weapon", COLOR_WEAPON)

        stats = {
            "total_persons": total_persons,
            "det_to_track": det_to_track,
            "person_boxes": person_boxes,
            "rider_flags": rider_flags,
            "pm_to_person": pm_to_person,
            "inst_has_helmet": inst_has_helmet,
            "trash_count": len(trash_boxes),
            "fire_count": len(fire_boxes),
            "smoke_count": len(smoke_boxes),
            "weapon_count": len(weapon_boxes),
            "pm_count": len(pm_boxes),
            "helmet_count": len(helmet_boxes),
        }
        return vis, pm_boxes, trash_boxes, fire_boxes, smoke_boxes, weapon_boxes, person_boxes, stats, frame

    # ---------- Main loop ----------
    def run(self):
        src = Path(self.source)
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {src}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Video opened: {src} | {W}x{H} @ {fps:.2f} FPS | frames={total}")

        out_path = self.out_dir / f"{src.stem}{self.out_suffix}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open writer: {out_path}")

        # Recent frame buffers for events — keep both raw/annotated
        clip_len_frames = max(1, int(self.vlm_clip_sec * fps))
        recent_raw_frames = deque(maxlen=clip_len_frames)
        recent_vis_frames = deque(maxlen=clip_len_frames)

        base_unix = time.time()
        frame_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                # Save raw frame first
                recent_raw_frames.append(frame.copy())

                # Inference + annotated frame
                vis, pm_boxes, trash_boxes, fire_boxes, smoke_boxes, weapon_boxes, person_boxes, stats, raw_frame = self.run_inference(frame)

                # Save annotated frame
                recent_vis_frames.append(vis.copy())

                # ===== Prepare tasks =====
                stamp_meta = self._video_stamp_meta(base_unix, frame_idx, fps)
                now_t = stamp_meta["t"]
                any_detection = (
                    stats["total_persons"] > 0 or
                    stats["pm_count"] > 0 or
                    stats["helmet_count"] > 0 or
                    stats["trash_count"] > 0 or
                    stats["fire_count"] > 0 or
                    stats["smoke_count"] > 0 or
                    stats["weapon_count"] > 0
                )

                # (A) Crowd task (비차단)
                crowd_task = {
                    "stamp_meta": stamp_meta,
                    "frame_idx": frame_idx,
                    "total_persons": stats["total_persons"],
                    "crowd_person_threshold": self.crowd_person_threshold,
                }
                try:
                    self.crowd_q.put_nowait(crowd_task)
                except queue.Full:
                    # 드랍해도 다음 프레임에서 회복
                    pass

                # (B) Hazard task (비차단)
                #  - per-track info 모으기 (no-helmet 로깅 통계용)
                track_infos = []
                # det_idx -> (tid, smoothed_has_helmet, was_event)
                det_to_track = stats["det_to_track"]
                for det_idx, (pb, is_rider) in enumerate(zip(stats["person_boxes"], stats["rider_flags"])):
                    if not is_rider:
                        continue
                    tid, smoothed, _ = det_to_track.get(det_idx, (None, None, False))
                    if tid is None:
                        continue
                    # tracker 내부 votes로 통계 계산
                    tr_obj = next((t for t in self.tracker.tracks if t.tid == tid), None)
                    if tr_obj is not None and hasattr(tr_obj, "votes"):
                        valid = [v for v in tr_obj.votes if v is not None]
                        helmet_ratio = (sum(1 for v in valid if v)/len(valid)) if len(valid) > 0 else 0.0
                        votes_valid = len(valid)
                    else:
                        helmet_ratio = 0.0; votes_valid = 0
                    track_infos.append({
                        "tid": int(tid),
                        "smoothed_has_helmet": None if smoothed is None else bool(smoothed),
                        "votes_valid": int(votes_valid),
                        "helmet_ratio": float(helmet_ratio),
                        "bbox": pb
                    })

                # PM → rider track ids 매핑
                pm_to_rider_tids = {pm_idx: [] for pm_idx in range(len(pm_boxes))}
                for pm_idx, person_idxs in stats["pm_to_person"].items():
                    tids = []
                    for det_idx in person_idxs:
                        tid, _, _ = det_to_track.get(det_idx, (None, None, False))
                        tids.append(None if tid is None else int(tid))
                    pm_to_rider_tids[pm_idx] = tids

                hazard_task = {
                    "stamp_meta": stamp_meta,
                    "frame_idx": frame_idx,
                    "fps": fps,
                    "total_persons": stats["total_persons"],
                    "hazard_counts": {
                        "trash_bag": stats["trash_count"],
                        "fire":      stats["fire_count"],
                        "smoke":     stats["smoke_count"],
                        "weapon":    stats["weapon_count"],
                    },
                    "pm_count": stats["pm_count"],
                    "helmet_count": stats["helmet_count"],
                    "track_infos": track_infos,
                    "vote_window": self.vote_window,
                    "vote_threshold": self.vote_threshold,
                    "pm_to_rider_tids": pm_to_rider_tids,
                    "multi_rider_hold": self.multi_rider_hold,
                    "multi_rider_cooldown": self.multi_rider_cooldown,
                    "any_detection": any_detection,
                    "frames_raw": list(recent_raw_frames),
                    "frames_vis": list(recent_vis_frames),
                }
                try:
                    self.hazard_q.put_nowait(hazard_task)
                except queue.Full:
                    pass

                # (C) 'any' mode VLM (선택): crowd/hazard와 독립적으로 주기적 캡션
                if self.vlm_enable and self.vlm_cond == "any" and any_detection:
                    if (stamp_meta["t"] - self._last_caption_t) >= self.vlm_min_interval:
                        frames_for_vlm = list(recent_vis_frames) if self.vlm_input_annotated else list(recent_raw_frames)
                        qa = _vlm_call_on_frames(
                            self.vlm_model, self.vlm_tokenizer,
                            frames_for_vlm, self.gen_cfg_dict,
                            num_segments=self.vlm_segments,
                            max_num=self.vlm_max_num,
                            fps_for_fallback=fps
                        )
                        rec = {
                            "stamp_iso": stamp_meta["iso"],
                            "frame_idx": frame_idx,
                            "event": "any",
                            "qa": qa,
                            "persons": stats["total_persons"],
                            "pm": stats["pm_count"],
                            "helmet": stats["helmet_count"],
                            "trash_bag": stats["trash_count"],
                            "fire": stats["fire_count"],
                            "smoke": stats["smoke_count"],
                            "weapon": stats["weapon_count"],
                            "event_detail": "any_detection"
                        }
                        try:
                            with open(self.vlm_jsonl_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        except Exception as e:
                            print(f"[VLM] jsonl append failed: {e}")
                        self._last_caption_t = stamp_meta["t"]

                # ===== Output =====
                writer.write(vis)
                if self.preview:
                    cv2.imshow("PM/Rider/Helmet (Video, Parallel)", vis)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (27, ord('q')):
                        print("[INFO] Interrupted by user.")
                        break

        finally:
            # 종료 시 워커 플러시
            self.stop_event.set()
            try:
                self.crowd_q.put_nowait(None)
            except Exception:
                pass
            try:
                self.hazard_q.put_nowait(None)
            except Exception:
                pass
            self.crowd_worker.join(timeout=3.0)
            self.hazard_worker.join(timeout=3.0)

            cap.release()
            writer.release()
            if self.preview:
                try: cv2.destroyAllWindows()
                except Exception: pass
            print(f"[SAVED] {out_path}")
            print(f"[INFO] Logs dir: {self.log_dir.resolve()}")
            if self.vlm_enable:
                print(f"[INFO] VLM images dir: {self.vlm_images_dir.resolve()}")
                print(f"[INFO] VLM QA jsonl: {self.vlm_jsonl_path.resolve()}")


# ======================
# CLI
# ======================
def parse_args():
    p = argparse.ArgumentParser(
        description="Local Video → YOLO one-pass w/ parallel voting/logging, InternVL QA (frames→video fallback)"
    )
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="input video path")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help=".pt path (4 or 7 classes)")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="output directory")
    p.add_argument("--out_suffix", type=str, default=DEFAULT_OUT_SUFFIX, help="output filename suffix")

    p.add_argument("--conf_thres", type=float, default=DEFAULT_CONF_THRES)
    p.add_argument("--iou_thres",  type=float, default=DEFAULT_IOU_THRES)

    p.add_argument("--head_region_ratio", type=float, default=DEFAULT_HEAD_REGION_RATIO)
    p.add_argument("--roi_top_extra",     type=float, default=DEFAULT_ROI_TOP_EXTRA)
    p.add_argument("--roi_side_extra",    type=float, default=DEFAULT_ROI_SIDE_EXTRA)

    p.add_argument("--person_draw_min_count",  type=int,   default=DEFAULT_PERSON_DRAW_MIN_COUNT)
    p.add_argument("--crowd_person_threshold", type=int,   default=DEFAULT_CROWD_PERSON_THRESHOLD)

    p.add_argument("--vote_window",           type=int,   default=DEFAULT_VOTE_WINDOW)
    p.add_argument("--vote_min_valid",        type=int,   default=DEFAULT_VOTE_MIN_VALID)
    p.add_argument("--vote_threshold",        type=float, default=DEFAULT_VOTE_THRESHOLD)
    p.add_argument("--track_iou_thresh",      type=float, default=DEFAULT_TRACK_IOU_THRESH)
    p.add_argument("--track_max_age_frames",  type=int,   default=DEFAULT_TRACK_MAX_AGE_FRAMES)

    p.add_argument("--log_cooldown_sec",       type=float, default=DEFAULT_LOG_COOLDOWN_SEC)
    p.add_argument("--no_helmet_cooldown_sec", type=float, default=DEFAULT_NO_HELMET_COOLDOWN_SEC)

    p.add_argument("--device", type=str, default=DEFAULT_DEVICE, help="''|cpu|cuda:0")
    p.add_argument("--imgsz",  type=int, default=DEFAULT_IMGSZ,  help="inference size; 0=auto")
    p.add_argument("--preview", type=int, default=int(DEFAULT_PREVIEW), help="1 to show preview window")

    # VLM options
    p.add_argument("--vlm_enable", type=int, default=int(DEFAULT_VLM_ENABLE), help="1 to enable InternVL QA")
    p.add_argument("--vlm_model", type=str, default=DEFAULT_VLM_MODEL, help="e.g., OpenGVLab/InternVL3-1B")
    p.add_argument("--vlm_device_map", type=str, default=DEFAULT_VLM_DEVICE_MAP,
                   help="device map: 'auto' | 'None' | JSON string (e.g., '{\"\": \"cuda:0\"}')")
    p.add_argument("--vlm_max_new_tokens", type=int, default=DEFAULT_VLM_MAXTOK)
    p.add_argument("--vlm_condition", type=str, default=DEFAULT_VLM_COND, help="event|any")
    p.add_argument("--vlm_lang", type=str, default=DEFAULT_VLM_LANG, help="ko|en")
    p.add_argument("--vlm_save_annotated", type=int, default=int(DEFAULT_VLM_SAVE_IMG), help="1: save annotated image for QA log")
    p.add_argument("--vlm_min_interval", type=float, default=DEFAULT_VLM_MIN_INTERVAL, help="min seconds between QA in 'any' mode")

    # Video option: event clip length / sampling and input source for VLM
    p.add_argument("--vlm_clip_sec", type=float, default=DEFAULT_VLM_CLIP_SEC, help="seconds of recent frames to feed to VLM per event")
    p.add_argument("--vlm_segments", type=int, default=DEFAULT_VLM_SEGMENTS, help="run_frames_inference num_segments")
    p.add_argument("--vlm_max_num", type=int, default=DEFAULT_VLM_MAX_NUM, help="video_vlm tiling 'max_num'")
    p.add_argument("--vlm_input_annotated", type=int, default=DEFAULT_VLM_INPUT_ANNOTATED,
                   help="1: feed annotated frames (with boxes) to VLM; 0: feed raw frames")

    # Rider association options
    p.add_argument("--rider_x_expand",     type=float, default=DEFAULT_RIDER_X_EXPAND,
                   help="PM horizontal expand factor for rider association")
    p.add_argument("--rider_y_expand",     type=float, default=DEFAULT_RIDER_Y_EXPAND,
                   help="PM vertical expand factor for rider association")
    p.add_argument("--rider_iou_thres",    type=float, default=DEFAULT_RIDER_IOU_THRES,
                   help="PM-person IoU threshold for rider association")
    p.add_argument("--rider_xcenter_frac", type=float, default=DEFAULT_RIDER_XCENTER_FRAC,
                   help="Allowed fraction of PM width around its center for person's X-center")
    p.add_argument("--max_riders_per_pm",  type=int,   default=DEFAULT_MAX_RIDERS_PER_PM,
                   help="maximum riders per PM association (>=2 to detect double riders)")

    # Crowd voting options
    p.add_argument("--crowd_vote_window", type=int, default=DEFAULT_CROWD_VOTE_WINDOW,
                   help="crowd voting window (frames)")
    p.add_argument("--crowd_vote_min_valid", type=int, default=DEFAULT_CROWD_VOTE_MIN_VALID,
                   help="minimum valid frames required inside crowd window")
    p.add_argument("--crowd_vote_threshold", type=float, default=DEFAULT_CROWD_VOTE_THRESHOLD,
                   help="fraction of True needed to mark crowd=True")

    # Holds / cooldowns / coupling
    p.add_argument("--nohelmet_hold", type=int, default=DEFAULT_NOHELMET_HOLD,
                   help="consecutive frames of smoothed no-helmet required before logging")
    p.add_argument("--multi_rider_hold", type=int, default=DEFAULT_MULTI_RIDER_HOLD,
                   help="consecutive frames with >=2 riders (same group) before logging")
    p.add_argument("--multi_rider_cooldown_sec", type=float, default=DEFAULT_MULTI_RIDER_COOLDOWN_SEC,
                   help="cooldown seconds per (rider group) double-ride event")
    p.add_argument("--hazard_require_crowd", type=int, default=DEFAULT_HAZARD_REQUIRE_CROWD,
                   help="1: log hazards only when crowd is True (smoothed); 0: independent")

    return p.parse_args()


def main():
    args = parse_args()
    app = VideoPMHelmetApp(args)
    app.run()


if __name__ == "__main__":
    main()
