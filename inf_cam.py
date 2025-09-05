#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inf_cam.py — ROS2 Humble real-time YOLO one-pass inference + majority-vote stabilization + event logging + InternVL-only QA

Overview
- Automatically supports 4-class or 7-class weights (.pt)
  * 4cls: person, pm, trash bag, helmet
  * 7cls: person, pm, trash bag, helmet, fire, smoke, weapon
- Rider detection: expand a PM box by 2×; if IoU > 0.01 with a person → Rider
- Helmet detection (riders only): if a helmet box center falls within the upper 60% of the head ROI
  (ROI = top 20%, left/right ±5%) → Helmet
- Majority-vote smoothing: smooth helmet state across recent frames (vote_window) using vote_threshold
- CSV logging:
  * no_helmet.csv: when a rider transitions Helmet→NoHelmet (per-track cooldown)
  * hazard_*.csv: when #people ≥ crowd_person_threshold (default 6) and trash_bag/fire/smoke/weapon is detected (cooldown)

InternVL (Only) QA
- VLM load: utils.video_vlm.init_model (safe loading, e.g., device_map="auto")
- On event, extract the most recent N seconds (vlm_clip_sec) of frames and run single-turn QA
  (one-sentence summary guided by detection context)
  - Prefer run_frames_inference; fallback to temporary MP4 + run_video_inference on failure
- JSONL logging: logs/vlm_qa.jsonl
- Option: save caption images (logs/vlm_images/)
"""

import os
import csv
import cv2
import json
import tempfile
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import List, Optional

import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO

# ====== VLM import ======
from utils.video_vlm import (
    init_model as vlm_init_model,
    run_frames_inference,
    run_video_inference,
)

# ====== Logging/utility ======
from utils.inf_utils import (
    expand_box, iou_xyxy, draw_label,
    _expand_person_roi, _helmet_center_in_head_region,
    resolve_class_ids,
)
from model.track import SimpleHelmetTracker

# ======================
# Defaults (constants)
# ======================
DEFAULT_WEIGHTS = "Pretrained/4_class.pt"
DEFAULT_INPUT_TOPIC = "/camera/image_raw"           # or .../compressed
DEFAULT_OUTPUT_TOPIC = "/pm_helmet/image_annotated"

DEFAULT_CONF_THRES = 0.25
DEFAULT_IOU_THRES  = 0.5

DEFAULT_HEAD_REGION_RATIO = 0.6
DEFAULT_ROI_TOP_EXTRA = 0.2
DEFAULT_ROI_SIDE_EXTRA = 0.05

DEFAULT_PERSON_DRAW_MIN_COUNT = 6
DEFAULT_CROWD_PERSON_THRESHOLD = 6

DEFAULT_VOTE_WINDOW = 5
DEFAULT_VOTE_MIN_VALID = 3
DEFAULT_VOTE_THRESHOLD = 0.5
DEFAULT_TRACK_IOU_THRESH = 0.3
DEFAULT_TRACK_MAX_AGE_FRAMES = 30

DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_COOLDOWN_SEC = 5.0
DEFAULT_NO_HELMET_COOLDOWN_SEC = 10.0

DEFAULT_DEVICE = ""        # "", "cpu", "cuda:0"
DEFAULT_PREVIEW = False
DEFAULT_IMGSZ = 0         # 0 → keep original size (Ultralytics default)

# ---- VLM (InternVL Only) defaults
DEFAULT_VLM_ENABLE = True
DEFAULT_VLM_MODEL  = "OpenGVLab/InternVL3-1B"
DEFAULT_VLM_DEVICE_MAP = "auto"     # 'auto' | 'None' | JSON string
DEFAULT_VLM_MAXTOK = 64
DEFAULT_VLM_COND   = "event"        # 'event' | 'any'
DEFAULT_VLM_LANG   = "ko"           # 'ko' | 'en'
DEFAULT_VLM_SAVE_IMG = True         # Save caption image (annotated/original chosen below)
DEFAULT_VLM_MIN_INTERVAL = 1.5      # In 'any' mode, minimum QA interval (sec)
DEFAULT_VLM_CLIP_SEC = 2.0          # Clip length for QA around event (sec)
DEFAULT_VLM_SEGMENTS = 8            # run_frames_inference num_segments
DEFAULT_VLM_MAX_NUM  = 1            # InternVL tiling upper bound
DEFAULT_VLM_TARGET_FPS = 15.0       # Estimated FPS for buffer sizing

# Visualization colors (BGR)
COLOR_PM      = (255, 160, 0)
COLOR_PERSON  = (0, 200, 0)
COLOR_RIDER   = (0, 140, 255)
COLOR_TRASH   = (180, 180, 30)
COLOR_FIRE    = (0, 0, 255)
COLOR_SMOKE   = (160, 160, 160)
COLOR_WEAPON  = (180, 0, 180)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _install_torchvision_nms_fallback(force: bool = False) -> bool:
    """
    Monkey-patch torchvision NMS with a pure PyTorch fallback when C++ ops are unavailable.
    Returns: True if fallback is applied, False if original C++ NMS is used.
    """
    try:
        import torchvision
    except Exception as e:
        print(f"[WARN] torchvision import failed: {e} -> using pure PyTorch NMS fallback")
        torchvision = None

    # Vectorized IoU
    def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: [M,4], b: [N,4] with (x1,y1,x2,y2)
        a = a.float()
        b = b.float()
        tl = torch.max(a[:, None, :2], b[None, :, :2])       # [M,N,2]
        br = torch.min(a[:, None, 2:], b[None, :, 2:])       # [M,N,2]
        wh = (br - tl).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]                      # [M,N]
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])   # [M]
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])   # [N]
        union = area_a[:, None] + area_b[None, :] - inter + 1e-7
        return inter / union

    def _nms_fallback(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
        """
        Pure-PyTorch NMS.
        boxes: [N,4], scores: [N]
        return: keep indices (LongTensor)
        """
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
            ious = _box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]  # [N-1]
            remain = (ious <= float(iou_thres)).nonzero(as_tuple=False).squeeze(1)
            order = order[1:][remain]

        return torch.tensor(keep, dtype=torch.long, device=device)

    patched = False
    if force:
        try:
            import torchvision
            torchvision.ops.nms = _nms_fallback
            patched = True
        except Exception:
            patched = True  # torchvision may be absent; fallback only
    else:
        try:
            import torchvision
            try:
                _ = torchvision.ops.nms(torch.zeros((1, 4)), torch.zeros((1,)), 0.5)
                patched = False  # C++ NMS works
            except Exception:
                torchvision.ops.nms = _nms_fallback
                patched = True
        except Exception:
            patched = True  # torchvision import failed → fallback only

    if patched:
        print("[WARN] Using pure-PyTorch NMS fallback (torchvision C++ ops unavailable).")
    else:
        print("[INFO] Using torchvision built-in NMS (C++ ops).")
    return patched


# Apply NMS fallback before using YOLO
_install_torchvision_nms_fallback()

# ======================
# VLM helpers (frames→video fallback)
# ======================
def _write_frames_to_temp_video(frames: List[np.ndarray], fps: float) -> str:
    """Write frames to a temporary MP4 file and return the path."""
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
    vlm_model, vlm_tokenizer, frames: List[np.ndarray],
    gen_cfg_dict: dict, num_segments=8, max_num=1, fps_for_fallback=15.0,
    prompt: Optional[str] = None, lang: str = "ko", hint: Optional[str] = None
):
    """
    Try InternVL with two generation configs:
      1) without 'use_cache'
      2) with use_cache=True
    For each config, try frames→video (fallback) in order.
    """
    if not frames:
        return []

    cfg_no_cache = {k: v for k, v in (gen_cfg_dict or {}).items() if k != "use_cache"}
    cfg_with_cache = dict(cfg_no_cache, use_cache=True)
    cfg_candidates = [cfg_no_cache, cfg_with_cache]

    def _try_frames(cfg):
        try:
            return run_frames_inference(
                model=vlm_model,
                tokenizer=vlm_tokenizer,
                frames=frames,
                generation_config=cfg,
                num_segments=num_segments,
                max_num=max_num,
                prompt=prompt,
                lang=lang,
                hint=hint,
            )
        except Exception as e:
            print(f"[VLM] frames inference failed ({'use_cache' in cfg}): {e}")
            return None

    def _try_video(cfg):
        try:
            tmp = _write_frames_to_temp_video(frames, fps_for_fallback)
            try:
                return run_video_inference(
                    model=vlm_model,
                    tokenizer=vlm_tokenizer,
                    video_path=tmp,
                    generation_config=cfg,
                    num_segments=num_segments,
                    max_num=max_num,
                    prompt=prompt,
                    lang=lang,
                    hint=hint,
                )
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        except Exception as e:
            print(f"[VLM] video inference failed ({'use_cache' in cfg}): {e}")
            return None

    # Try frames first
    for cfg in cfg_candidates:
        qa = _try_frames(cfg)
        if qa:
            return qa

    # Fallback to video
    for cfg in cfg_candidates:
        qa = _try_video(cfg)
        if qa:
            return qa

    print("[VLM] caption failed: all attempts exhausted")
    return []


# ======================
# ROS2 node
# ======================
class PMHelmetNode(Node):
    def __init__(self):
        super().__init__("pm_helmet_inference_vote_log")

        # ---- ROS parameter declarations
        self.declare_parameter("weights", DEFAULT_WEIGHTS)
        self.declare_parameter("input_topic", DEFAULT_INPUT_TOPIC)
        self.declare_parameter("output_topic", DEFAULT_OUTPUT_TOPIC)

        self.declare_parameter("conf_thres", DEFAULT_CONF_THRES)
        self.declare_parameter("iou_thres", DEFAULT_IOU_THRES)

        self.declare_parameter("head_region_ratio", DEFAULT_HEAD_REGION_RATIO)
        self.declare_parameter("roi_top_extra", DEFAULT_ROI_TOP_EXTRA)
        self.declare_parameter("roi_side_extra", DEFAULT_ROI_SIDE_EXTRA)

        self.declare_parameter("person_draw_min_count", DEFAULT_PERSON_DRAW_MIN_COUNT)
        self.declare_parameter("crowd_person_threshold", DEFAULT_CROWD_PERSON_THRESHOLD)

        self.declare_parameter("vote_window", DEFAULT_VOTE_WINDOW)
        self.declare_parameter("vote_min_valid", DEFAULT_VOTE_MIN_VALID)
        self.declare_parameter("vote_threshold", DEFAULT_VOTE_THRESHOLD)
        self.declare_parameter("track_iou_thresh", DEFAULT_TRACK_IOU_THRESH)
        self.declare_parameter("track_max_age_frames", DEFAULT_TRACK_MAX_AGE_FRAMES)

        self.declare_parameter("log_dir", DEFAULT_LOG_DIR)
        self.declare_parameter("log_cooldown_sec", DEFAULT_LOG_COOLDOWN_SEC)
        self.declare_parameter("no_helmet_cooldown_sec", DEFAULT_NO_HELMET_COOLDOWN_SEC)

        self.declare_parameter("device", DEFAULT_DEVICE)
        self.declare_parameter("preview", DEFAULT_PREVIEW)
        self.declare_parameter("imgsz", DEFAULT_IMGSZ)

        # ---- VLM parameters
        self.declare_parameter("vlm_enable", DEFAULT_VLM_ENABLE)
        self.declare_parameter("vlm_model", DEFAULT_VLM_MODEL)
        self.declare_parameter("vlm_device_map", DEFAULT_VLM_DEVICE_MAP)
        self.declare_parameter("vlm_max_new_tokens", DEFAULT_VLM_MAXTOK)
        self.declare_parameter("vlm_condition", DEFAULT_VLM_COND)    # event|any
        self.declare_parameter("vlm_lang", DEFAULT_VLM_LANG)          # ko|en
        self.declare_parameter("vlm_save_annotated", int(DEFAULT_VLM_SAVE_IMG))  # 1/0
        self.declare_parameter("vlm_min_interval", DEFAULT_VLM_MIN_INTERVAL)
        self.declare_parameter("vlm_clip_sec", DEFAULT_VLM_CLIP_SEC)
        self.declare_parameter("vlm_segments", DEFAULT_VLM_SEGMENTS)
        self.declare_parameter("vlm_max_num", DEFAULT_VLM_MAX_NUM)
        self.declare_parameter("vlm_target_fps", DEFAULT_VLM_TARGET_FPS)

        # ---- Get parameters
        self.weights = self.get_parameter("weights").value
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value

        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.iou_thres  = float(self.get_parameter("iou_thres").value)

        self.head_region_ratio = float(self.get_parameter("head_region_ratio").value)
        self.roi_top_extra     = float(self.get_parameter("roi_top_extra").value)
        self.roi_side_extra    = float(self.get_parameter("roi_side_extra").value)

        self.person_draw_min_count  = int(self.get_parameter("person_draw_min_count").value)
        self.crowd_person_threshold = int(self.get_parameter("crowd_person_threshold").value)

        vw = int(self.get_parameter("vote_window").value)
        vm = int(self.get_parameter("vote_min_valid").value)
        vt = float(self.get_parameter("vote_threshold").value)
        ti = float(self.get_parameter("track_iou_thresh").value)
        ta = int(self.get_parameter("track_max_age_frames").value)

        self.log_dir = Path(self.get_parameter("log_dir").value)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_cooldown = float(self.get_parameter("log_cooldown_sec").value)
        self.nohelmet_cooldown = float(self.get_parameter("no_helmet_cooldown_sec").value)

        self.device = self.get_parameter("device").value
        self.preview = bool(self.get_parameter("preview").value)
        self.imgsz = int(self.get_parameter("imgsz").value)

        # ---- VLM setup
        self.vlm_enable = bool(self.get_parameter("vlm_enable").value)
        self.vlm_model_name = self.get_parameter("vlm_model").value
        dm_raw = self.get_parameter("vlm_device_map").value
        if dm_raw == "auto":
            self.vlm_device_map = "auto"
        elif dm_raw == "None":
            self.vlm_device_map = None
        else:
            try:
                self.vlm_device_map = json.loads(dm_raw)
            except Exception:
                self.vlm_device_map = dm_raw

        self.vlm_maxtok = int(self.get_parameter("vlm_max_new_tokens").value)
        self.vlm_cond   = str(self.get_parameter("vlm_condition").value).lower().strip()
        self.vlm_lang   = str(self.get_parameter("vlm_lang").value)
        self.vlm_save_img = bool(int(self.get_parameter("vlm_save_annotated").value))
        self.vlm_min_interval = float(self.get_parameter("vlm_min_interval").value)
        self.vlm_clip_sec = float(self.get_parameter("vlm_clip_sec").value)
        self.vlm_segments = int(self.get_parameter("vlm_segments").value)
        self.vlm_max_num  = int(self.get_parameter("vlm_max_num").value)
        self.vlm_target_fps = float(self.get_parameter("vlm_target_fps").value)

        # ---- Load YOLO
        self.get_logger().info(f"Loading YOLO weights: {self.weights}")
        self.model = YOLO(self.weights)
        if self.device:
            try:
                self.model.to(self.device)
                self.get_logger().info(f"Moved model to device: {self.device}")
            except Exception as e:
                self.get_logger().warning(f"Failed to move model to '{self.device}': {e}")

        # Class id mapping (auto-support for 4/7 classes)
        self.ids = resolve_class_ids(self.model)
        self.get_logger().info(f"Resolved class ids: {self.ids}")

        # Tracker (majority vote)
        self.tracker = SimpleHelmetTracker(vote_window=vw, vote_min_valid=vm, vote_threshold=vt,
                                           iou_thresh=ti, max_age_frames=ta)

        # QoS & I/O
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()

        # Input topic type autodetection
        if self.input_topic.endswith("/compressed"):
            self.sub_compressed = self.create_subscription(CompressedImage, self.input_topic, self.compressed_cb, qos)
            self.sub_image = None
            self.get_logger().info(f"Subscribed: {self.input_topic} (CompressedImage)")
        else:
            self.sub_image = self.create_subscription(Image, self.input_topic, self.image_cb, qos)
            compressed_topic = self.input_topic + "/compressed"
            self.sub_compressed = self.create_subscription(CompressedImage, compressed_topic, self.compressed_cb, qos)
            self.get_logger().info(f"Subscribed: {self.input_topic} (Image) and {compressed_topic} (CompressedImage)")

        self.pub = self.create_publisher(Image, self.output_topic, qos)
        self.get_logger().info(f"Publishing annotated image: {self.output_topic}")
        self.get_logger().info(f"CONF_THRES={self.conf_thres}, IOU_THRES={self.iou_thres}, vote_window={vw}, vote_min_valid={vm}, vote_threshold={vt}")

        # Processing flag (simple backpressure)
        self.processing = False

        # Hazard log cooldown timestamps
        self.last_hazard_log_time = {
            "trash_bag": 0.0, "fire": 0.0, "smoke": 0.0, "weapon": 0.0, "crowd": 0.0,
        }

        # ====== VLM loading/buffer/logging ======
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
                self.get_logger().info(f"[VLM] Loaded via video_vlm.init_model: {self.vlm_model_name} (device_map={self.vlm_device_map})")
            except Exception as e:
                self.get_logger().error(f"[VLM] load failed: {e}")
                self.vlm_enable = False

        # Recent frame buffer: (frame, recv_time_sec)
        maxlen = max(1, int(self.vlm_target_fps * self.vlm_clip_sec * 2))
        self.recent_frames = deque(maxlen=maxlen)

        # VLM logging/images
        self.vlm_images_dir = self.log_dir / "vlm_images"
        self.vlm_images_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_jsonl_path = self.log_dir / "vlm_qa.jsonl"
        self._last_qa_t = -1e9

    # ---------- Core inference + visualization ----------
    def run_inference(self, frame):
        h, w = frame.shape[:2]

        # Keep only valid (non-None) class IDs; pass to Ultralytics if non-empty
        keep_ids = sorted({v for v in self.ids.values() if v is not None})
        pred_kwargs = dict(conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        if keep_ids:
            pred_kwargs["classes"] = keep_ids
        if self.imgsz and self.imgsz > 0:
            pred_kwargs["imgsz"] = self.imgsz

        res = self.model.predict(frame, **pred_kwargs)[0]
        if res.boxes is None or len(res.boxes) == 0:
            stats = {
                "total_persons": 0,
                "det_to_track": {},
                "person_boxes": [],
                "rider_flags": [],
                "inst_has_helmet": [],
                "trash_count": 0,
                "fire_count": 0,
                "smoke_count": 0,
                "weapon_count": 0,
            }
            return frame, 0, [], [], [], [], [], stats

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

        # Rider decision: PM (expanded ×2) IoU>0.01 with person
        rider_flags = [False] * len(person_boxes)
        for pm in pm_boxes:
            pm2 = expand_box(pm, factor=2.0, img_w=w, img_h=h)
            for i, pb in enumerate(person_boxes):
                if iou_xyxy(pm2, pb) > 0.01:
                    rider_flags[i] = True

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
        for pm in pm_boxes:
            x1, y1, x2, y2 = pm.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PM, 2)
            draw_label(frame, x1, y1, "PM", COLOR_PM)

        total_persons = len(person_boxes)

        # Visualization — Rider (trackID + smoothed state)
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RIDER, 2)
            draw_label(frame, x1, y1, label, COLOR_RIDER)

        # Visualization — general Person (only when many people)
        if total_persons >= self.person_draw_min_count:
            for (pb, is_rider) in zip(person_boxes, rider_flags):
                if is_rider:
                    continue
                x1, y1, x2, y2 = pb.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
                draw_label(frame, x1, y1, "Person", COLOR_PERSON)

            text = f"Persons: {total_persons}"
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
            x0 = w - tw - 16
            y0 = 16 + th + 8
            cv2.rectangle(frame, (x0 - 8, 8), (x0 + tw + 8, y0), (50, 50, 50), -1)
            cv2.putText(frame, text, (x0, 16 + th), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Visualization — other classes
        for tb in trash_boxes:
            x1, y1, x2, y2 = tb.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TRASH, 2)
            draw_label(frame, x1, y1, "Trash", COLOR_TRASH)
        for fb in fire_boxes:
            x1, y1, x2, y2 = fb.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FIRE, 2)
            draw_label(frame, x1, y1, "Fire", COLOR_FIRE)
        for sb in smoke_boxes:
            x1, y1, x2, y2 = sb.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_SMOKE, 2)
            draw_label(frame, x1, y1, "Smoke", COLOR_SMOKE)
        for wb in weapon_boxes:
            x1, y1, x2, y2 = wb.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WEAPON, 2)
            draw_label(frame, x1, y1, "Weapon", COLOR_WEAPON)

        stats = {
            "total_persons": total_persons,
            "det_to_track": det_to_track,
            "person_boxes": person_boxes,
            "rider_flags": rider_flags,
            "inst_has_helmet": inst_has_helmet,
            "trash_count": len(trash_boxes),
            "fire_count": len(fire_boxes),
            "smoke_count": len(smoke_boxes),
            "weapon_count": len(weapon_boxes),
        }
        return frame, total_persons, trash_boxes, fire_boxes, smoke_boxes, weapon_boxes, person_boxes, stats

    # ---------- CSV logging helpers ----------
    def _append_csv(self, file_path: Path, header: list, row: list):
        existed = file_path.exists()
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not existed:
                writer.writerow(header)
            writer.writerow(row)

    def _stamp_to_meta(self, header_stamp) -> dict:
        """Convert ROS header stamp into a dictionary with sec/nsec/iso/float time."""
        if header_stamp is not None:
            sec = int(getattr(header_stamp, "sec", 0))
            nsec = int(getattr(header_stamp, "nanosec", getattr(header_stamp, "nanosec_", 0)))
        else:
            now = self.get_clock().now().to_msg()
            sec = int(now.sec)
            nsec = int(now.nanosec)
        t = sec + nsec * 1e-9
        iso = datetime.fromtimestamp(t).isoformat()
        return {"sec": sec, "nsec": nsec, "iso": iso, "t": t}

    def log_no_helmet(self, stamp_meta, track_id, bbox, persons, vote_window, valid_votes, threshold, helmet_ratio):
        """Append a No-Helmet transition event into no_helmet.csv."""
        fp = self.log_dir / "no_helmet.csv"
        header = ["stamp_sec","stamp_nsec","iso","track_id","bbox_x1","bbox_y1","bbox_x2","bbox_y2",
                  "persons","vote_window","valid_votes","vote_threshold","helmet_ratio"]
        row = [stamp_meta["sec"], stamp_meta["nsec"], stamp_meta["iso"], track_id,
               int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
               persons, vote_window, valid_votes, threshold, round(helmet_ratio,3)]
        self._append_csv(fp, header, row)

    def log_hazard(self, stamp_meta, event_type, count, persons):
        """Append a hazard event into hazard_<event>.csv."""
        fp = self.log_dir / f"hazard_{event_type}.csv"
        header = ["stamp_sec","stamp_nsec","iso","event","count","persons"]
        row = [stamp_meta["sec"], stamp_meta["nsec"], stamp_meta["iso"], event_type, count, persons]
        self._append_csv(fp, header, row)

    # ---------- VLM helpers ----------
    def _recent_append(self, frame_bgr: np.ndarray, t_sec: float):
        self.recent_frames.append((frame_bgr, float(t_sec)))

    def _recent_clip_frames(self, now_t: float) -> List[np.ndarray]:
        """Return frames within [now - clip_sec, now]. If empty, return the latest frame if available."""
        t0 = now_t - self.vlm_clip_sec
        frames = [f for (f, ts) in self.recent_frames if ts >= t0]
        if not frames and self.recent_frames:
            frames = [self.recent_frames[-1][0]]
        return frames

    def _save_img_for_vlm(self, raw_bgr, vis_bgr, stem: str) -> str:
        """Save an image for captioning (annotated or raw) and return its path."""
        img_path = self.vlm_images_dir / f"{stem}.jpg"
        try:
            cv2.imwrite(str(img_path), vis_bgr if self.vlm_save_img else raw_bgr)
            return str(img_path)
        except Exception:
            return ""

    def _log_vlm_qa(self, stamp_meta, frame_idx: int, event_type: str, qa_list, extra=None, image_path=""):
        """Append a VLM QA record into logs/vlm_qa.jsonl."""
        rec = {
            "stamp_iso": stamp_meta["iso"],
            "ros_stamp": {"sec": int(stamp_meta["sec"]), "nsec": int(stamp_meta["nsec"])},
            "frame_idx": frame_idx,
            "event": event_type,
            "image_path": image_path,
            "qa": qa_list,
            "lang": self.vlm_lang,
        }
        if extra:
            rec.update(extra)
        try:
            with open(self.vlm_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            self.get_logger().error(f"[VLM] jsonl append failed: {e}")

    def _build_hint(self, event_type: str, total_persons: int, stats: dict, det_to_track: dict) -> str:
        """Compose a concise hint string summarizing scene context for InternVL."""
        rider_flags = stats.get("rider_flags", [])
        inst_has_helmet = stats.get("inst_has_helmet", [])
        helmet_on = helmet_off = helmet_unknown = 0
        for idx, is_rider in enumerate(rider_flags):
            if not is_rider:
                continue
            smoothed = det_to_track.get(idx, (None, None, False))[1]
            if smoothed is True:
                helmet_on += 1
            elif smoothed is False:
                helmet_off += 1
            else:
                if idx < len(inst_has_helmet) and inst_has_helmet[idx]:
                    helmet_on += 1
                else:
                    helmet_unknown += 1
        riders = int(sum(1 for f in rider_flags if f))
        crowd = (total_persons >= self.crowd_person_threshold)
        hint = (
            f"persons={total_persons}, riders={riders}, "
            f"helmet_on={helmet_on}, helmet_off={helmet_off}, helmet_unknown={helmet_unknown}, "
            f"crowd_6plus={'Yes' if crowd else 'No'}, "
            f"trash_bag={stats.get('trash_count',0)}, fire={stats.get('fire_count',0)}, "
            f"smoke={stats.get('smoke_count',0)}, weapon={stats.get('weapon_count',0)}, "
            f"event={event_type}"
        )
        return hint

    # ---------- Unified processing/publish ----------
    def _process_and_publish(self, frame, header):
        # 1) Timestamp + push original frame into recent buffer
        stamp_meta = self._stamp_to_meta(header.stamp if header else None)
        now_t = stamp_meta["t"]
        self._recent_append(frame.copy(), now_t)

        # 2) Run inference/visualization on a copy
        work = frame.copy()
        vis, total_persons, trash_boxes, fire_boxes, smoke_boxes, weapon_boxes, person_boxes, stats = self.run_inference(work)

        # Determine if any detection exists
        any_detection = (
            total_persons > 0 or
            stats["trash_count"] > 0 or
            stats["fire_count"]  > 0 or
            stats["smoke_count"] > 0 or
            stats["weapon_count"]> 0
        )

        det_to_track = stats["det_to_track"]
        rider_flags = stats["rider_flags"]

        # (A) No-helmet (majority vote) — log on state transition (per-track cooldown) + VLM
        for det_idx, (pb, is_rider) in enumerate(zip(person_boxes, rider_flags)):
            if not is_rider:
                continue
            info = det_to_track.get(det_idx)
            if not info:
                continue
            tid, smoothed, was_event = info
            if smoothed is False and was_event:
                tr = next((t for t in self.tracker.tracks if t.tid == tid), None)
                if tr is None:
                    continue
                if (now_t - tr.last_nohelmet_log_time) >= self.nohelmet_cooldown:
                    valid = [v for v in tr.votes if v is not None]
                    ratio = (sum(1 for v in valid if v) / len(valid)) if len(valid) > 0 else 0.0
                    self.log_no_helmet(stamp_meta, tid, pb, total_persons,
                                       self.tracker.vote_window, len(valid),
                                       self.tracker.vote_threshold, ratio)
                    tr.last_nohelmet_log_time = now_t

                    # VLM (on 'event' mode, or 'any' mode with detection)
                    if self.vlm_enable and (self.vlm_cond == "event" or (self.vlm_cond == "any" and any_detection)):
                        frames_for_vlm = self._recent_clip_frames(now_t)
                        hint = self._build_hint("no_helmet", total_persons, stats, det_to_track)
                        qa = _vlm_call_on_frames(
                            self.vlm_model, self.vlm_tokenizer, frames_for_vlm,
                            self.gen_cfg_dict, num_segments=self.vlm_segments,
                            max_num=self.vlm_max_num, fps_for_fallback=self.vlm_target_fps,
                            prompt=None, lang=self.vlm_lang, hint=hint
                        )
                        stem = f"ros_f{int(now_t*1000):013d}_nohelmet"
                        img_path = self._save_img_for_vlm(frame, vis, stem) if self.vlm_save_img else ""
                        extra = {
                            "persons": total_persons,
                            "trash_bag": stats["trash_count"],
                            "fire": stats["fire_count"],
                            "smoke": stats["smoke_count"],
                            "weapon": stats["weapon_count"],
                            "track_id": int(tid),
                            "event_detail": "no_helmet_transition"
                        }
                        self._log_vlm_qa(stamp_meta, frame_idx=-1, event_type="no_helmet",
                                         qa_list=qa, extra=extra, image_path=img_path)

        # (B) Crowd (persons ≥ threshold) + hazard logging (cooldown) + VLM
        is_crowd = (total_persons >= self.crowd_person_threshold)
        if is_crowd:
            hazards = [
                ("trash_bag", stats["trash_count"]),
                ("fire",      stats["fire_count"]),
                ("smoke",     stats["smoke_count"]),
                ("weapon",    stats["weapon_count"]),
            ]
            any_hazard_present = any(count > 0 for _, count in hazards)

            # (B0) Call VLM even for crowd-only scenes (no explicit hazards) when cooldown allows
            if self.vlm_enable and (self.vlm_cond == "event" or (self.vlm_cond == "any" and any_detection)) and not any_hazard_present:
                last_t = self.last_hazard_log_time.get("crowd", 0.0)
                if (now_t - last_t) >= self.log_cooldown:
                    frames_for_vlm = self._recent_clip_frames(now_t)
                    hint = self._build_hint("crowd", total_persons, stats, det_to_track)
                    qa = _vlm_call_on_frames(
                        self.vlm_model, self.vlm_tokenizer, frames_for_vlm,
                        self.gen_cfg_dict, num_segments=self.vlm_segments,
                        max_num=self.vlm_max_num, fps_for_fallback=self.vlm_target_fps,
                        prompt=None, lang=self.vlm_lang, hint=hint
                    )
                    stem = f"ros_f{int(now_t*1000):013d}_crowd"
                    img_path = self._save_img_for_vlm(frame, vis, stem) if self.vlm_save_img else ""
                    extra = {
                        "persons": total_persons,
                        "trash_bag": stats["trash_count"],
                        "fire": stats["fire_count"],
                        "smoke": stats["smoke_count"],
                        "weapon": stats["weapon_count"],
                        "event_detail": "crowd_only"
                    }
                    self._log_vlm_qa(stamp_meta, frame_idx=-1, event_type="crowd",
                                     qa_list=qa, extra=extra, image_path=img_path)
                    self.last_hazard_log_time["crowd"] = now_t

            # (B1) Log hazards when present (per-type cooldown) + VLM
            for name, count in hazards:
                if count <= 0:
                    continue
                last_t = self.last_hazard_log_time.get(name, 0.0)
                if (now_t - last_t) >= self.log_cooldown:
                    self.log_hazard(stamp_meta, name, count, total_persons)
                    self.last_hazard_log_time[name] = now_t

                    if self.vlm_enable and (self.vlm_cond == "event" or (self.vlm_cond == "any" and any_detection)):
                        frames_for_vlm = self._recent_clip_frames(now_t)
                        hint = self._build_hint(f"hazard_{name}", total_persons, stats, det_to_track)
                        qa = _vlm_call_on_frames(
                            self.vlm_model, self.vlm_tokenizer, frames_for_vlm,
                            self.gen_cfg_dict, num_segments=self.vlm_segments,
                            max_num=self.vlm_max_num, fps_for_fallback=self.vlm_target_fps,
                            prompt=None, lang=self.vlm_lang, hint=hint
                        )
                        stem = f"ros_f{int(now_t*1000):013d}_{name}"
                        img_path = self._save_img_for_vlm(frame, vis, stem) if self.vlm_save_img else ""
                        extra = {
                            "persons": total_persons,
                            "trash_bag": stats["trash_count"],
                            "fire": stats["fire_count"],
                            "smoke": stats["smoke_count"],
                            "weapon": stats["weapon_count"],
                            "event_detail": f"hazard_{name}"
                        }
                        self._log_vlm_qa(stamp_meta, frame_idx=-1, event_type=name,
                                         qa_list=qa, extra=extra, image_path=img_path)

        # (C) 'any' mode: run QA whenever there is any detection, respecting min interval
        if self.vlm_enable and self.vlm_cond == "any" and any_detection:
            if (now_t - self._last_qa_t) >= self.vlm_min_interval:
                frames_for_vlm = self._recent_clip_frames(now_t)
                hint = self._build_hint("any_detection", total_persons, stats, det_to_track)
                qa = _vlm_call_on_frames(
                    self.vlm_model, self.vlm_tokenizer, frames_for_vlm,
                    self.gen_cfg_dict, num_segments=self.vlm_segments,
                    max_num=self.vlm_max_num, fps_for_fallback=self.vlm_target_fps,
                    prompt=None, lang=self.vlm_lang, hint=hint
                )
                stem = f"ros_f{int(now_t*1000):013d}_any"
                img_path = self._save_img_for_vlm(frame, vis, stem) if self.vlm_save_img else ""
                extra = {
                    "persons": total_persons,
                    "trash_bag": stats["trash_count"],
                    "fire": stats["fire_count"],
                    "smoke": stats["smoke_count"],
                    "weapon": stats["weapon_count"],
                    "event_detail": "any_detection"
                }
                self._log_vlm_qa(stamp_meta, frame_idx=-1, event_type="any",
                                 qa_list=qa, extra=extra, image_path=img_path)
                self._last_qa_t = now_t

        # Publish annotated image
        out = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        if header:
            out.header = header
        self.pub.publish(out)

        # Optional preview window
        if self.preview:
            cv2.imshow("PM/Rider/Helmet (ROS2)", vis)
            cv2.waitKey(1)

    # ---------- ROS callbacks ----------
    def image_cb(self, msg: Image):
        if self.processing:
            return
        self.processing = True
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._process_and_publish(frame, msg.header)
        except Exception as e:
            self.get_logger().error(f"inference error: {e}")
        finally:
            self.processing = False

    def compressed_cb(self, msg: CompressedImage):
        if self.processing:
            return
        self.processing = True  # simple backpressure
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            self._process_and_publish(frame, msg.header)
        except Exception as e:
            self.get_logger().error(f"inference error (compressed): {e}")
        finally:
            self.processing = False


# ======================
# Entry point
# ======================
def main():
    rclpy.init()
    node = PMHelmetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
