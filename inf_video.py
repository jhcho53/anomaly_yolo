#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inf_video.py — Local Video → YOLO one-pass + Majority Vote + CSV Event Logs + InternVL QA (frames→video 폴백)

- 4/7 클래스 자동 지원 (person/pm/trash_bag/helmet[/fire/smoke/weapon])
- Rider/Helmet 판정 + 다수결 안정화 + CSV 이벤트 로깅
- InternVL(Only) 로드: utils.video_vlm.init_model
- 이벤트 발생 시 최근 프레임 버퍼를 VLM에 투입:
  * 우선: utils.video_vlm.run_frames_inference
  * 폴백: 임시 mp4 생성 → utils.video_vlm.run_video_inference
- VLM 결과(JSONL): runs/.../logs/vlm_qa.jsonl

옵션:
  --vlm_input_annotated 0|1
    0: 원본 프레임을 VLM 입력으로 사용(기본, 권장)
    1: 박스가 그려진 주석 프레임을 VLM 입력으로 사용(디버깅용)
"""

import os
import re
import cv2
import csv
import json
import time
import argparse
import tempfile
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional, List

from ultralytics import YOLO

# ====== VLM import (InternVL 전용) ======
from utils.video_vlm import (
    init_model as vlm_init_model,
    run_frames_inference,
    run_video_inference,
)
# ====== 로깅/유틸 ======
from utils.log import _video_stamp_meta, log_hazard, log_no_helmet
from utils.inf_utils import (
    expand_box, iou_xyxy, draw_label,
    _expand_person_roi, _helmet_center_in_head_region,
    resolve_class_ids,
)
from model.track import SimpleHelmetTracker

# ======================
# 기본값(상수)
# ======================
DEFAULT_WEIGHTS = "Pretrained/4_class.pt"
DEFAULT_SOURCE = "input.mp4"
DEFAULT_OUT_DIR = "runs/pm_helmet_video"

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

DEFAULT_LOG_COOLDOWN_SEC = 5.0
DEFAULT_NO_HELMET_COOLDOWN_SEC = 10.0

DEFAULT_DEVICE = ""        # "", "cpu", "cuda:0"
DEFAULT_PREVIEW = False
DEFAULT_IMGSZ = 0
DEFAULT_OUT_SUFFIX = "_out.mp4"

# ---- VLM(InternVL Only)
DEFAULT_VLM_ENABLE = True
DEFAULT_VLM_MODEL  = "OpenGVLab/InternVL3-1B"
DEFAULT_VLM_DEVICE_MAP = "auto"     # "auto" | "None" | JSON 문자열
DEFAULT_VLM_MAXTOK = 64
DEFAULT_VLM_COND   = "event"        # "event" | "any"
DEFAULT_VLM_LANG   = "ko"           # "ko" | "en"
DEFAULT_VLM_SAVE_IMG = True         # True: 캡션용 이미지 저장
DEFAULT_VLM_MIN_INTERVAL = 1.5      # any 모드에서 캡션 최소 간격(초)
DEFAULT_VLM_CLIP_SEC = 2.0          # 이벤트 시 마지막 N초 프레임을 VLM에 투입
DEFAULT_VLM_SEGMENTS = 8            # run_frames_inference num_segments
DEFAULT_VLM_MAX_NUM  = 1            # InternVL 타일링 최대 수 (video_vlm)
DEFAULT_VLM_INPUT_ANNOTATED = 0     # 1: VLM 입력으로 주석 프레임 사용, 0: 원본 프레임 사용

# 시각화 색상 (BGR)
COLOR_PM      = (255, 160, 0)
COLOR_PERSON  = (0, 200, 0)
COLOR_RIDER   = (0, 140, 255)
COLOR_HELMET  = (0, 0, 255)
COLOR_TRASH   = (180, 180, 30)
COLOR_FIRE    = (0, 0, 255)
COLOR_SMOKE   = (160, 160, 160)
COLOR_WEAPON  = (180, 0, 180)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ======================
# VLM 헬퍼(프레임→비디오 폴백 포함)
# ======================
def _write_frames_to_temp_video(frames: List[np.ndarray], fps: float) -> str:
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
    gen_cfg_dict: dict, num_segments=8, max_num=1, fps_for_fallback=30.0
):
    """
    InternVL 버전 차이에 안전하도록 generation_config 후보를 두 번 시도:
      1) use_cache 미포함(dict)
      2) use_cache=True 포함(dict)
    각 후보에 대해 frames→video(폴백) 순서로 시도.
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
                generation_config=cfg,     # InternVL chat 4번째 위치 인자와 호환
                num_segments=num_segments,
                max_num=max_num
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
                    max_num=max_num
                )
            finally:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
        except Exception as e:
            print(f"[VLM] video inference failed ({'use_cache' in cfg}): {e}")
            return None

    for cfg in cfg_candidates:
        qa = _try_frames(cfg)
        if qa:
            return qa

    for cfg in cfg_candidates:
        qa = _try_video(cfg)
        if qa:
            return qa

    print("[VLM] caption failed: all attempts exhausted")
    return []

# ======================
# 비디오 추론 앱
# ======================
class VideoPMHelmetApp:
    def __init__(self, args):
        self.weights = args.weights
        self.source = args.source
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_suffix = args.out_suffix

        self.conf_thres = args.conf_thres
        self.iou_thres  = args.iou_thres

        self.head_region_ratio = args.head_region_ratio
        self.roi_top_extra     = args.roi_top_extra
        self.roi_side_extra    = args.roi_side_extra

        self.person_draw_min_count  = args.person_draw_min_count
        self.crowd_person_threshold = args.crowd_person_threshold

        self.vote_window = args.vote_window
        self.vote_min_valid = args.vote_min_valid
        self.vote_threshold = args.vote_threshold
        self.track_iou_thresh = args.track_iou_thresh
        self.track_max_age_frames = args.track_max_age_frames

        self.log_dir = self.out_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_cooldown = args.log_cooldown_sec
        self.nohelmet_cooldown = args.no_helmet_cooldown_sec

        self.device = args.device
        self.preview = bool(args.preview)
        self.imgsz = args.imgsz

        # VLM 설정
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
        self.vlm_save_img = bool(args.vlm_save_annotated)  # 로그 이미지 저장 주석 유무
        self.vlm_min_interval = float(args.vlm_min_interval)
        self.vlm_clip_sec = float(args.vlm_clip_sec)
        self.vlm_segments = int(args.vlm_segments)
        self.vlm_max_num  = int(args.vlm_max_num)
        self.vlm_input_annotated = bool(args.vlm_input_annotated)  # ★ VLM 입력 프레임 소스 토글

        self.vlm_images_dir = self.out_dir / "vlm_images"
        self.vlm_images_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_jsonl_path = self.log_dir / "vlm_qa.jsonl"
        self._last_caption_t = -1e9
        self._video_stamp_meta = _video_stamp_meta

        # YOLO 모델
        print(f"[INFO] Loading YOLO weights: {self.weights}")
        self.model = YOLO(self.weights)
        if self.device:
            try:
                self.model.to(self.device)
                print(f"[INFO] Moved model to device: {self.device}")
            except Exception as e:
                print(f"[WARN] Failed to move model to '{self.device}': {e}")

        # 클래스 id 매핑
        self.ids = resolve_class_ids(self.model)
        print(f"[INFO] Resolved class ids: {self.ids}")

        # 트래커
        self.tracker = SimpleHelmetTracker(
            vote_window=self.vote_window,
            vote_min_valid=self.vote_min_valid,
            vote_threshold=self.vote_threshold,
            iou_thresh=self.track_iou_thresh,
            max_age_frames=self.track_max_age_frames,
        )

        # Hazard 로그 쿨다운 타임
        self.last_hazard_log_time = {
            "trash_bag": 0.0, "fire": 0.0, "smoke": 0.0, "weapon": 0.0
        }

        # VLM 로더 (InternVL Only)
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

    def _save_img_for_vlm(self, img_bgr, vis_bgr, stem):
        img_path = self.vlm_images_dir / f"{stem}.jpg"
        try:
            # 로그 저장 이미지는 --vlm_save_annotated 토글로 결정
            cv2.imwrite(str(img_path), vis_bgr if self.vlm_save_img else img_bgr)
            return str(img_path)
        except Exception:
            return ""

    def _log_vlm_qa(self, stamp_meta, frame_idx, event_type, qa_list, extra=None, image_path=""):
        rec = {
            "stamp_iso": stamp_meta["iso"],
            "frame_idx": frame_idx,
            "event": event_type,
            "image_path": image_path,
            "qa": qa_list,
        }
        if extra:
            rec.update(extra)
        try:
            with open(self.vlm_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[VLM] jsonl append failed: {e}")

    # ---------- 프레임 추론 ----------
    def run_inference(self, frame):
        h, w = frame.shape[:2]
        keep_ids = sorted(set(self.ids.values()))
        pred_kwargs = dict(conf=self.conf_thres, iou=self.iou_thres, classes=keep_ids, verbose=False)
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
                "pm_count": 0,
                "helmet_count": 0,
            }
            return frame, 0, [], [], [], [], [], stats, frame

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

        # Rider 판정
        rider_flags = [False] * len(person_boxes)
        for pm in pm_boxes:
            pm2 = expand_box(pm, factor=2.0, img_w=w, img_h=h)
            for i, pb in enumerate(person_boxes):
                if iou_xyxy(pm2, pb) > 0.01:
                    rider_flags[i] = True

        # 헬멧 즉시판정(라이더만)
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

        # 다수결 스무딩(트래커)
        det_to_track = self.tracker.update(person_boxes, rider_flags, inst_has_helmet)

        # 시각화 — PM
        vis = frame.copy()
        for pm in pm_boxes:
            x1, y1, x2, y2 = pm.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PM, 2)
            draw_label(vis, x1, y1, "PM", COLOR_PM)

        total_persons = len(person_boxes)

        # 시각화 — Rider
        for det_idx, (pb, is_rider) in enumerate(zip(person_boxes, rider_flags)):
            if not is_rider:
                continue
            x1, y1, x2, y2 = pb.astype(int)
            tid, smoothed, _ = det_to_track.get(det_idx, (None, None, False))
            show_has_helmet = smoothed if smoothed is not None else inst_has_helmet[det_idx]
            label = f"Rider"
            if tid is not None:
                label += f"#{tid}"
            label += " | " + ("Helmet" if show_has_helmet else "NoHelmet")
            cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_RIDER, 2)
            draw_label(vis, x1, y1, label, COLOR_RIDER)

        # 시각화 — 일반 Person (임계 이상)
        if total_persons >= self.person_draw_min_count:
            for (pb, is_rider) in zip(person_boxes, rider_flags):
                if is_rider:
                    continue
                x1, y1, x2, y2 = pb.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PERSON, 2)
                draw_label(vis, x1, y1, "Person", COLOR_PERSON)
            text = f"Persons: {total_persons}"
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
            x0 = w - tw - 16; y0 = 16 + th + 8
            cv2.rectangle(vis, (x0 - 8, 8), (x0 + tw + 8, y0), (50, 50, 50), -1)
            cv2.putText(vis, text, (x0, 16 + th), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 기타 클래스 시각화
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
            "inst_has_helmet": inst_has_helmet,
            "trash_count": len(trash_boxes),
            "fire_count": len(fire_boxes),
            "smoke_count": len(smoke_boxes),
            "weapon_count": len(weapon_boxes),
            "pm_count": len(pm_boxes),
            "helmet_count": len(helmet_boxes),
        }
        return vis, total_persons, trash_boxes, fire_boxes, smoke_boxes, weapon_boxes, person_boxes, stats, frame

    # ---------- 메인 루프 ----------
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

        out_path = self.out_dir / f"{src.stem}{DEFAULT_OUT_SUFFIX}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open writer: {out_path}")

        # 최근 프레임 버퍼 (이벤트용) — 원본/주석 분리
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

                # 원본 프레임 버퍼에 먼저 저장
                recent_raw_frames.append(frame.copy())

                # 추론 + 주석 프레임 생성
                vis, total_persons, trash_boxes, fire_boxes, smoke_boxes, weapon_boxes, person_boxes, stats, raw_frame = self.run_inference(frame)

                # 주석 프레임 버퍼에도 저장
                recent_vis_frames.append(vis.copy())

                # ===== 로깅/QA =====
                stamp_meta = self._video_stamp_meta(base_unix, frame_idx, fps)
                now_t = stamp_meta["t"]
                any_detection = (
                    total_persons > 0 or
                    stats["pm_count"] > 0 or
                    stats["helmet_count"] > 0 or
                    stats["trash_count"] > 0 or
                    stats["fire_count"] > 0 or
                    stats["smoke_count"] > 0 or
                    stats["weapon_count"] > 0
                )

                det_to_track = stats["det_to_track"]
                rider_flags = stats["rider_flags"]

                # (A) 무헬멧 전이 이벤트
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
                            ratio = (sum(1 for v in valid if v)/len(valid)) if len(valid) > 0 else 0.0

                            # CSV 로그
                            log_no_helmet(
                                log_dir=self.log_dir,
                                stamp_meta=stamp_meta,
                                track_id=int(tid),
                                bbox=pb,  # [x1,y1,x2,y2]
                                persons=int(total_persons),
                                vote_window=int(self.tracker.vote_window),
                                valid_votes=int(len(valid)),
                                threshold=float(self.tracker.vote_threshold),
                                helmet_ratio=float(ratio),
                            )
                            tr.last_nohelmet_log_time = now_t

                            # VLM QA (event 모드 또는 any+검출)
                            if self.vlm_enable and (self.vlm_cond == "event" or (self.vlm_cond == "any" and any_detection)):
                                frames_for_vlm = list(recent_vis_frames) if self.vlm_input_annotated else list(recent_raw_frames)
                                qa = _vlm_call_on_frames(
                                    self.vlm_model, self.vlm_tokenizer,
                                    frames_for_vlm, self.gen_cfg_dict,
                                    num_segments=self.vlm_segments,
                                    max_num=self.vlm_max_num,
                                    fps_for_fallback=fps
                                )
                                stem = f"{src.stem}_f{frame_idx:06d}_nohelmet"
                                img_path = self._save_img_for_vlm(raw_frame, vis, stem) if self.vlm_save_img else ""
                                extra = {
                                    "persons": total_persons,
                                    "pm": stats["pm_count"],
                                    "helmet": stats["helmet_count"],
                                    "trash_bag": stats["trash_count"],
                                    "fire": stats["fire_count"],
                                    "smoke": stats["smoke_count"],
                                    "weapon": stats["weapon_count"],
                                    "track_id": int(tid),
                                    "event_detail": "no_helmet_transition"
                                }
                                self._log_vlm_qa(stamp_meta, frame_idx, "no_helmet", qa, extra, img_path)

                # (B) 군집 + Hazard
                is_crowd = (total_persons >= self.crowd_person_threshold)
                if is_crowd:
                    hazards = [
                        ("trash_bag", stats["trash_count"]),
                        ("fire",      stats["fire_count"]),
                        ("smoke",     stats["smoke_count"]),
                        ("weapon",    stats["weapon_count"]),
                    ]
                    for name, cnt in hazards:
                        if cnt <= 0:
                            continue
                        last_t = self.last_hazard_log_time.get(name, 0.0)
                        if (now_t - last_t) >= self.log_cooldown:
                            # CSV 로그
                            log_hazard(
                                log_dir=self.log_dir,
                                stamp_meta=stamp_meta,
                                event_type=name,
                                count=int(cnt),
                                persons=int(total_persons),
                            )
                            self.last_hazard_log_time[name] = now_t

                            if self.vlm_enable and (self.vlm_cond == "event" or (self.vlm_cond == "any" and any_detection)):
                                frames_for_vlm = list(recent_vis_frames) if self.vlm_input_annotated else list(recent_raw_frames)
                                qa = _vlm_call_on_frames(
                                    self.vlm_model, self.vlm_tokenizer,
                                    frames_for_vlm, self.gen_cfg_dict,
                                    num_segments=self.vlm_segments,
                                    max_num=self.vlm_max_num,
                                    fps_for_fallback=fps
                                )
                                stem = f"{src.stem}_f{frame_idx:06d}_{name}"
                                img_path = self._save_img_for_vlm(raw_frame, vis, stem) if self.vlm_save_img else ""
                                extra = {
                                    "persons": total_persons,
                                    "pm": stats["pm_count"],
                                    "helmet": stats["helmet_count"],
                                    "trash_bag": stats["trash_count"],
                                    "fire": stats["fire_count"],
                                    "smoke": stats["smoke_count"],
                                    "weapon": stats["weapon_count"],
                                    "event_detail": f"hazard_{name}"
                                }
                                self._log_vlm_qa(stamp_meta, frame_idx, name, qa, extra, img_path)

                # (옵션) any 모드: 검출이 있는 모든 시점, 최소 간격 보장
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
                        stem = f"{src.stem}_f{frame_idx:06d}_any"
                        img_path = self._save_img_for_vlm(raw_frame, vis, stem) if self.vlm_save_img else ""
                        extra = {
                            "persons": total_persons,
                            "pm": stats["pm_count"],
                            "helmet": stats["helmet_count"],
                            "trash_bag": stats["trash_count"],
                            "fire": stats["fire_count"],
                            "smoke": stats["smoke_count"],
                            "weapon": stats["weapon_count"],
                            "event_detail": "any_detection"
                        }
                        self._log_vlm_qa(stamp_meta, frame_idx, "any", qa, extra, img_path)
                        self._last_caption_t = stamp_meta["t"]

                # ===== 출력 =====
                writer.write(vis)
                if self.preview:
                    cv2.imshow("PM/Rider/Helmet (Video)", vis)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (27, ord('q')):
                        print("[INFO] Interrupted by user.")
                        break

        finally:
            cap.release()
            writer.release()
            if self.preview:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
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
        description="Local Video → YOLO one-pass w/ majority voting, event logs, InternVL QA (frames→video fallback)"
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

    # VLM 옵션
    p.add_argument("--vlm_enable", type=int, default=int(DEFAULT_VLM_ENABLE), help="1 to enable InternVL QA")
    p.add_argument("--vlm_model", type=str, default=DEFAULT_VLM_MODEL, help="e.g., OpenGVLab/InternVL3-1B")
    p.add_argument("--vlm_device_map", type=str, default=DEFAULT_VLM_DEVICE_MAP,
                   help="device map: 'auto' | 'None' | JSON string (e.g., '{\"\": \"cuda:0\"}')")
    p.add_argument("--vlm_max_new_tokens", type=int, default=DEFAULT_VLM_MAXTOK)
    p.add_argument("--vlm_condition", type=str, default=DEFAULT_VLM_COND, help="event|any")
    p.add_argument("--vlm_lang", type=str, default=DEFAULT_VLM_LANG, help="ko|en")
    p.add_argument("--vlm_save_annotated", type=int, default=int(DEFAULT_VLM_SAVE_IMG), help="1: save annotated image for QA log")
    p.add_argument("--vlm_min_interval", type=float, default=DEFAULT_VLM_MIN_INTERVAL, help="min seconds between QA in 'any' mode")

    # 새 옵션: 이벤트 클립 길이/샘플링 및 입력 소스
    p.add_argument("--vlm_clip_sec", type=float, default=DEFAULT_VLM_CLIP_SEC, help="seconds of recent frames to feed to VLM per event")
    p.add_argument("--vlm_segments", type=int, default=DEFAULT_VLM_SEGMENTS, help="run_frames_inference num_segments")
    p.add_argument("--vlm_max_num", type=int, default=DEFAULT_VLM_MAX_NUM, help="video_vlm tiling 'max_num'")
    p.add_argument("--vlm_input_annotated", type=int, default=DEFAULT_VLM_INPUT_ANNOTATED,
                   help="1: feed annotated frames (with boxes) to VLM; 0: feed raw frames")

    return p.parse_args()

def main():
    args = parse_args()
    app = VideoPMHelmetApp(args)
    app.run()

if __name__ == "__main__":
    main()
