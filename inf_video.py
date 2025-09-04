#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inf_video_vlm.py — Local Video → YOLO one-pass + Majority Vote + CSV Event Logs + InternVL3-1B Captions

- 4/7 클래스 자동 지원 (person/pm/trash_bag/helmet[/fire/smoke/weapon])
- Rider/Helmet 판정 + 다수결 안정화 + CSV 이벤트 로깅
- InternVL3-1B로 이벤트 프레임 캡션 생성 + 이미지/JSONL 로깅
- VLM 로딩은 사용자 제공 init_model() 함수를 그대로 사용(device_map="auto" 기본)

의존성:
  pip install --no-deps --no-cache-dir ultralytics pillow transformers accelerate sentencepiece safetensors
  (PyTorch는 Jetson용 CUDA 빌드 유지! --no-deps 사용)
"""

import os
import re
import csv
import cv2
import time
import json
import argparse
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Optional

from ultralytics import YOLO

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
DEFAULT_IMGSZ = 0         # 0이면 원본 크기(ultralytics 기본)
DEFAULT_OUT_SUFFIX = "_out.mp4"

# ---- VLM(InternVL3-1B) 기본값
DEFAULT_VLM_ENABLE = True
DEFAULT_VLM_MODEL  = "OpenGVLab/InternVL3-1B"
DEFAULT_VLM_DEVICE_MAP = "auto"  # "auto" | "None" | JSON 문자열
DEFAULT_VLM_MAXTOK = 64
DEFAULT_VLM_COND   = "event"     # "event" | "any"
DEFAULT_VLM_LANG   = "ko"        # "ko" | "en"
DEFAULT_VLM_SAVE_ANN = True      # True: 주석 프레임 저장, False: 원본 프레임 저장
DEFAULT_VLM_MIN_INTERVAL = 1.5   # any 모드에서 캡션 최소 간격(초)

# 시각화 색상 (BGR)
COLOR_PM      = (255, 160, 0)
COLOR_PERSON  = (0, 200, 0)
COLOR_RIDER   = (0, 140, 255)
COLOR_HELMET  = (0, 0, 255)    # (라벨 배경 색상용, 헬멧 박스 그리진 않음)
COLOR_TRASH   = (180, 180, 30)
COLOR_FIRE    = (0, 0, 255)
COLOR_SMOKE   = (160, 160, 160)
COLOR_WEAPON  = (180, 0, 180)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ======================
# 유틸
# ======================
def expand_box(xyxy, factor, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1; h = y2 - y1
    cx = x1 + w / 2.0; cy = y1 + h / 2.0
    nw = w * factor; nh = h * factor
    nx1 = max(0, cx - nw / 2.0); ny1 = max(0, cy - nh / 2.0)
    nx2 = min(img_w - 1, cx + nw / 2.0); ny2 = min(img_h - 1, cy + nh / 2.0)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-6
    return inter / union

def draw_label(img, x, y, text, color, scale=0.5, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _expand_person_roi(pb, w, h, top_extra, side_extra):
    x1, y1, x2, y2 = pb
    pw = x2 - x1; ph = y2 - y1
    nx1 = max(0, x1 - pw * side_extra)
    nx2 = min(w - 1, x2 + pw * side_extra)
    ny1 = max(0, y1 - ph * top_extra)
    ny2 = min(h - 1, y2)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

def _helmet_center_in_head_region(helmet_box, roi, head_ratio):
    hx1, hy1, hx2, hy2 = helmet_box
    cx = (hx1 + hx2) / 2.0; cy = (hy1 + hy2) / 2.0
    rx1, ry1, rx2, ry2 = roi
    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
        return False
    head_y_limit = ry1 + (ry2 - ry1) * head_ratio
    return cy <= head_y_limit

# ---------- 클래스 매핑(4cls/7cls 자동) ----------
def _normalize(s: str):
    s = s.lower().strip()
    s = re.sub(r"[_\-\s]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())

def _find_one(norm_map, candidates):
    cands = [_normalize(c) for c in candidates]
    # 1) 정확 일치
    for i, n in norm_map.items():
        if n in cands:
            return i
    # 2) 부분 포함(양방향)
    for i, n in norm_map.items():
        if any(c in n or n in c for c in cands):
            return i
    return None

def resolve_class_ids(model):
    raw = model.names
    id_to_name = {int(k): v for k, v in (raw.items() if isinstance(raw, dict) else enumerate(raw))}
    norm_map = {i: _normalize(n) for i, n in id_to_name.items()}

    CANDS = {
        "person":    ["person"],
        "pm":        ["pm", "personal mobility", "kickboard", "e scooter", "escooter", "electric scooter", "micromobility", "micro mobility", "e_scooter"],
        "trash_bag": ["trash bag", "garbage bag", "plastic bag", "trash", "garbage"],
        "helmet":    ["helmet", "hardhat", "safety helmet"],
        "fire":      ["fire"],
        "smoke":     ["smoke"],
        "weapon":    ["weapon"],
    }

    ids = {}
    for key, cands in CANDS.items():
        idx = _find_one(norm_map, cands)
        if idx is not None:
            ids[key] = idx

    required = ["person", "pm", "helmet"]
    missing = [k for k in required if k not in ids]
    if missing:
        raise ValueError(f"[ERROR] 필수 클래스 누락: {missing} | model.names={id_to_name}")
    return ids

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

        # update matched
        for di, tr in matches.items():
            tr.bbox = dets[di]
            tr.last_frame = self.frame_idx
            tr.rider_recent = bool(rider_flags[di])
            tr.votes.append(bool(inst_has_helmet_list[di]) if tr.rider_recent else None)

        # create tracks for unmatched
        for di in unmatched:
            tr = Track(self.next_tid, dets[di], self.frame_idx, self.vote_window)
            tr.rider_recent = bool(rider_flags[di])
            tr.votes.append(bool(inst_has_helmet_list[di]) if tr.rider_recent else None)
            self.tracks.append(tr)
            matches[di] = tr
            self.next_tid += 1

        # prune old
        self.tracks = [tr for tr in self.tracks if self.frame_idx - tr.last_frame <= self.max_age_frames]

        # outputs
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

# ======================
# InternVL 로더 (사용자 제공 방식)
# ======================
import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig

def init_model(path='OpenGVLab/InternVL3-1B', device_map: Optional[object] = "auto"):
    """
    InternVL 로더 (안정성 우선):
      - dtype: CUDA면 FP16, 아니면 FP32
      - device_map 기본 "auto" (가장 안전)
      - 호환성 fallback: dtype → torch_dtype, device_map={"": "cuda:0"} → None
    """
    use_cuda = torch.cuda.is_available()
    prefer_dtype = torch.float16 if use_cuda else torch.float32

    kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # device_map 처리
    if device_map is None:
        pass
    elif device_map == "auto":
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = device_map  # 사용자가 직접 넘긴 맵(문자열/dict)

    # 1차: 최신 Transformers는 dtype 인자를 권장
    try:
        model = AutoModel.from_pretrained(
            path,
            dtype=prefer_dtype,
            **kwargs
        ).eval()
    except TypeError:
        # 구버전 호환: torch_dtype 사용
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=prefer_dtype,
            **kwargs
        ).eval()
    except KeyError:
        # device_map 관련 키 이슈 → 재시도
        try:
            km = kwargs.copy()
            km["device_map"] = {"": "cuda:0"} if use_cuda else None
            model = AutoModel.from_pretrained(
                path,
                dtype=prefer_dtype,
                **km
            ).eval()
        except Exception:
            # 최후 수단: device_map 해제 후 로드 → 이후 to(cuda)
            model = AutoModel.from_pretrained(
                path,
                dtype=prefer_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).eval()
            if use_cuda:
                model = model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

# ======================
# InternVL3-1B 캡셔너 (init_model 사용 + generation_config + 타입 호환 보강)
# ======================
class InternVLCaptioner:
    """
    1) 제공된 init_model()로 InternVL3-1B 로드 (device_map='auto' 기본)
    2) 실패 시 transformers.pipeline('image-to-text') 폴백
    - InternVL 경로: 입력은 **RGB numpy.ndarray**로 전달 ('.shape' 필요 버전 호환)
    - generation_config를 기본 제공, 구버전 시그니처 자동 폴백
    """
    def __init__(self, model_name, device_map="auto", max_new_tokens=64, lang="ko"):
        self.model_name = model_name
        self.device_map = device_map   # "auto" | None | dict/str
        self.max_new_tokens = int(max_new_tokens)
        self.lang = lang
        self.avail = False
        self.backend = None  # "internvl" | "pipeline"
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.gen_cfg = None  # GenerationConfig (internvl 경로에서 사용)

    def load(self):
        # 우선: 사용자 init_model()로 로드
        try:
            self.model, self.tokenizer = init_model(self.model_name, device_map=self.device_map)
            self.gen_cfg = GenerationConfig(
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                repetition_penalty=1.05,
            )
            self.backend = "internvl"
            self.avail = True
            print(f"[VLM] Loaded InternVL via init_model: {self.model_name} (device_map={self.device_map})")
            return self
        except Exception as e1:
            print(f"[VLM] init_model load failed: {e1}")

        # 폴백: image-to-text 파이프라인
        try:
            from transformers import pipeline
            self.pipeline = pipeline(task="image-to-text", model=self.model_name)
            self.backend = "pipeline"
            self.avail = True
            print(f"[VLM] Fallback pipeline(image-to-text) loaded: {self.model_name}")
            return self
        except Exception as e2:
            print(f"[VLM] Fallback pipeline load failed: {e2}")
            self.avail = False
            return self

    @staticmethod
    def _bgr_to_rgb_np(img_bgr):
        # InternVL 일부 버전은 PIL이 아니라 numpy RGB를 기대(.shape 사용)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _to_pil(self, img_bgr):
        # 파이프라인 경로에선 PIL을 선호
        from PIL import Image
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def caption(self, img_bgr, prompt=None):
        if not self.avail:
            return None
        if prompt is None:
            prompt = ("이 이미지를 간단하고 핵심적으로 한국어로 설명해줘."
                      if self.lang == "ko"
                      else "Describe this image briefly and concisely in English.")
        try:
            if self.backend == "internvl":
                rgb_np = self._bgr_to_rgb_np(img_bgr)  # numpy 배열
                # 1) 최신 시그니처: generation_config 필요
                try:
                    text = self.model.chat(
                        self.tokenizer, rgb_np, prompt,
                        history=[], generation_config=self.gen_cfg
                    )
                except TypeError:
                    # 2) 구버전 시그니처: generation_config 없이
                    text = self.model.chat(self.tokenizer, rgb_np, prompt, history=[])
                if isinstance(text, (list, tuple)):
                    text = text[-1] if text else ""
                return str(text)
            else:
                # pipeline 경로도 PIL 입력 사용
                pil = self._to_pil(img_bgr)
                out = self.pipeline(pil, max_new_tokens=self.max_new_tokens)
                if isinstance(out, list) and out:
                    return str(out[0].get("generated_text") or out[0].get("caption") or out[0])
                return str(out)
        except Exception as e:
            print(f"[VLM] caption failed: {e}")
            return None

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
        self.vlm_model  = args.vlm_model
        # device_map 파싱: "auto" | "None" | JSON 문자열(딕셔너리)
        self.vlm_device_map = None
        if args.vlm_device_map == "auto":
            self.vlm_device_map = "auto"
        elif args.vlm_device_map == "None":
            self.vlm_device_map = None
        else:
            try:
                self.vlm_device_map = json.loads(args.vlm_device_map)
            except Exception:
                # 문자열 그대로 전달
                self.vlm_device_map = args.vlm_device_map

        self.vlm_maxtok = args.vlm_max_new_tokens
        self.vlm_cond   = args.vlm_condition.lower().strip()
        self.vlm_lang   = args.vlm_lang
        self.vlm_save_annotated = bool(args.vlm_save_annotated)
        self.vlm_min_interval   = float(args.vlm_min_interval)
        self.vlm_images_dir = self.out_dir / "vlm_images"
        self.vlm_images_dir.mkdir(parents=True, exist_ok=True)
        self.vlm_jsonl_path = self.log_dir / "vlm_captions.jsonl"
        self._last_caption_t = -1e9  # 비디오 상대시각 기준

        # 모델
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

        # VLM 로더
        self.captioner = None
        if self.vlm_enable:
            self.captioner = InternVLCaptioner(
                model_name=self.vlm_model,
                device_map=self.vlm_device_map,
                max_new_tokens=self.vlm_maxtok,
                lang=self.vlm_lang
            ).load()
            if not self.captioner.avail:
                print("[VLM] Captioner unavailable. Proceeding without captions.")

    # ---------- 로그 유틸 ----------
    def _append_csv(self, file_path: Path, header: list, row: list):
        existed = file_path.exists()
        with open(file_path, "a", newline="") as f:
            w = csv.writer(f)
            if not existed:
                w.writerow(header)
            w.writerow(row)

    def _video_stamp_meta(self, base_unix: float, frame_idx: int, fps: float):
        t = (frame_idx / (fps if fps > 1e-6 else 30.0))
        ts_unix = base_unix + t
        sec = int(ts_unix)
        nsec = int((ts_unix - sec) * 1e9)
        iso = datetime.fromtimestamp(ts_unix).isoformat()
        return {"sec": sec, "nsec": nsec, "iso": iso, "t": t}

    def log_no_helmet(self, stamp_meta, track_id, bbox, persons, vote_window, valid_votes, threshold, helmet_ratio):
        fp = self.log_dir / "no_helmet.csv"
        header = ["stamp_sec","stamp_nsec","iso","track_id","bbox_x1","bbox_y1","bbox_x2","bbox_y2",
                  "persons","vote_window","valid_votes","vote_threshold","helmet_ratio"]
        row = [stamp_meta["sec"], stamp_meta["nsec"], stamp_meta["iso"], track_id,
               int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
               persons, vote_window, valid_votes, threshold, round(helmet_ratio,3)]
        self._append_csv(fp, header, row)

    def log_hazard(self, stamp_meta, event_type, count, persons):
        fp = self.log_dir / f"hazard_{event_type}.csv"
        header = ["stamp_sec","stamp_nsec","iso","event","count","persons"]
        row = [stamp_meta["sec"], stamp_meta["nsec"], stamp_meta["iso"], event_type, count, persons]
        self._append_csv(fp, header, row)

    # ----- 도메인 특화 프롬프트 빌더 -----
    def _make_domain_prompt(self, event_type: str, total_persons: int, stats: dict, det_to_track: dict) -> str:
        """
        YOLO 탐지값을 힌트로 제공하면서, 안전/치안 요소를 한 줄 요약하도록 유도.
        """
        rider_flags = stats.get("rider_flags", [])
        inst_has_helmet = stats.get("inst_has_helmet", [])
        # 스무딩 결과 우선 사용, 없으면 inst_has_helmet 사용
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
                # 즉시판정 참고
                if inst_has_helmet[idx]:
                    helmet_on += 1
                else:
                    helmet_unknown += 1
        riders = sum(1 for f in rider_flags if f)
        crowd = (total_persons >= self.crowd_person_threshold)

        # 힌트 집계
        hint = {
            "persons": total_persons,
            "riders": riders,
            "helmet_on": helmet_on,
            "helmet_off": helmet_off,
            "helmet_unknown": helmet_unknown,
            "crowd_6plus": "예" if crowd else "아니오",
            "trash_bag": stats.get("trash_count", 0),
            "fire": stats.get("fire_count", 0),
            "smoke": stats.get("smoke_count", 0),
            "weapon": stats.get("weapon_count", 0),
            "event": event_type,
        }

        if self.vlm_lang == "ko":
            prompt = (
                "아래 힌트를 참고해 보이는 사실만 근거로 안전 점검 요약을 한국어 한 문장으로 작성하세요. "
                "정확하고 간결하게, 과장/추측 금지.\n"
                f"힌트: 사람={hint['persons']}, 라이더={hint['riders']}, 헬멧착용={hint['helmet_on']}명, "
                f"무헬멧={hint['helmet_off']}명, 불명={hint['helmet_unknown']}명, "
                f"군집(6+명)={hint['crowd_6plus']}, 쓰레기봉투={hint['trash_bag']}, "
                f"화재={hint['fire']}, 연기={hint['smoke']}, 흉기={hint['weapon']}, 이벤트={hint['event']}.\n"
                "출력 형식 예: '사람≈N, 라이더 M명(헬멧 O/X/혼재), 군집 O/X, 쓰봉 O/X, 화재 O/X, 연기 O/X, 흉기 O/X, 메모: …'"
            )
        else:
            prompt = (
                "Using the hints below, write a concise, factual one‑sentence safety summary in English. "
                "No speculation; be precise.\n"
                f"Hints: persons={hint['persons']}, riders={hint['riders']}, helmet_on={hint['helmet_on']}, "
                f"helmet_off={hint['helmet_off']}, unknown={hint['helmet_unknown']}, "
                f"crowd_6plus={hint['crowd_6plus']}, trash_bag={hint['trash_bag']}, "
                f"fire={hint['fire']}, smoke={hint['smoke']}, weapon={hint['weapon']}, event={hint['event']}.\n"
                "Output format example: 'people≈N, riders M (helmet on/off/mixed), crowd Y/N, trash Y/N, fire Y/N, smoke Y/N, weapon Y/N, note: …'"
            )
        return prompt

    def _save_image_and_caption(self, img_bgr, vis_bgr, stamp_meta, frame_idx, event_type, info_dict, prompt=None):
        """이미지 저장 + 캡션 생성 + JSONL 기록"""
        if not (self.captioner and self.captioner.avail):
            return

        # any 모드에서 과도한 캡션 방지
        if self.vlm_cond == "any":
            if (stamp_meta["t"] - self._last_caption_t) < self.vlm_min_interval:
                return

        img_to_log = vis_bgr if self.vlm_save_annotated else img_bgr
        stem = f"{Path(self.source).stem}_f{frame_idx:06d}_{event_type}"
        img_path = self.vlm_images_dir / f"{stem}.jpg"
        try:
            cv2.imwrite(str(img_path), img_to_log)
        except Exception as e:
            print(f"[VLM] image save failed: {e}")
            return

        caption = self.captioner.caption(img_to_log, prompt=prompt)
        rec = {
            "stamp_iso": stamp_meta["iso"],
            "frame_idx": frame_idx,
            "event": event_type,
            "image_path": str(img_path),
            "caption": caption,
        }
        rec.update(info_dict or {})

        try:
            with open(self.vlm_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[VLM] jsonl append failed: {e}")
        self._last_caption_t = stamp_meta["t"]

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

        # 시각화 — Rider(트랙ID + 스무딩 결과)
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

        # 시각화 — 일반 Person (사람 수가 임계 이상일 때만)
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
        return vis, total_persons, trash_boxes, fire_boxes, smoke_boxes, weapon_boxes, person_boxes, stats, frame  # raw frame 포함

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

        base_unix = time.time()  # 로그용 기준시각
        frame_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                vis, total_persons, trash_boxes, fire_boxes, smoke_boxes, weapon_boxes, person_boxes, stats, raw_frame = self.run_inference(frame)

                # ===== 로깅/캡션 =====
                stamp_meta = self._video_stamp_meta(base_unix, frame_idx, fps)
                now_t = stamp_meta["t"]  # 비디오 상대시각

                any_detection = (
                    total_persons > 0 or
                    stats["pm_count"] > 0 or
                    stats["helmet_count"] > 0 or
                    stats["trash_count"] > 0 or
                    stats["fire_count"] > 0 or
                    stats["smoke_count"] > 0 or
                    stats["weapon_count"] > 0
                )

                # (A) 무헬멧(다수결) — 상태 전이 시 로깅(트랙별 쿨다운) + 캡션
                det_to_track = stats["det_to_track"]
                rider_flags = stats["rider_flags"]
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
                            self.log_no_helmet(stamp_meta, tid, pb, total_persons,
                                               self.tracker.vote_window, len(valid),
                                               self.tracker.vote_threshold, ratio)
                            tr.last_nohelmet_log_time = now_t

                            # VLM 캡션 — 도메인 특화 프롬프트
                            if self.vlm_enable and (self.vlm_cond == "event" or (self.vlm_cond == "any" and any_detection)):
                                domain_prompt = self._make_domain_prompt(
                                    event_type="no_helmet",
                                    total_persons=total_persons,
                                    stats=stats,
                                    det_to_track=det_to_track
                                )
                                info_dict = {
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
                                self._save_image_and_caption(
                                    img_bgr=raw_frame, vis_bgr=vis, stamp_meta=stamp_meta,
                                    frame_idx=frame_idx, event_type="no_helmet", info_dict=info_dict,
                                    prompt=domain_prompt
                                )

                # (B) 군집 + Hazard 로깅(쿨다운) + 캡션
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
                            self.log_hazard(stamp_meta, name, cnt, total_persons)
                            self.last_hazard_log_time[name] = now_t

                            # VLM 캡션 — 도메인 특화 프롬프트
                            if self.vlm_enable and (self.vlm_cond == "event" or (self.vlm_cond == "any" and any_detection)):
                                domain_prompt = self._make_domain_prompt(
                                    event_type=f"hazard_{name}",
                                    total_persons=total_persons,
                                    stats=stats,
                                    det_to_track=det_to_track
                                )
                                info_dict = {
                                    "persons": total_persons,
                                    "pm": stats["pm_count"],
                                    "helmet": stats["helmet_count"],
                                    "trash_bag": stats["trash_count"],
                                    "fire": stats["fire_count"],
                                    "smoke": stats["smoke_count"],
                                    "weapon": stats["weapon_count"],
                                    "event_detail": f"hazard_{name}"
                                }
                                self._save_image_and_caption(
                                    img_bgr=raw_frame, vis_bgr=vis, stamp_meta=stamp_meta,
                                    frame_idx=frame_idx, event_type=name, info_dict=info_dict,
                                    prompt=domain_prompt
                                )

                # (옵션) any 모드: 검출이 있는 모든 프레임 캡션(최소 간격 적용)
                if self.vlm_enable and self.vlm_cond == "any" and any_detection:
                    domain_prompt = self._make_domain_prompt(
                        event_type="any_detection",
                        total_persons=total_persons,
                        stats=stats,
                        det_to_track=det_to_track
                    )
                    info_dict = {
                        "persons": total_persons,
                        "pm": stats["pm_count"],
                        "helmet": stats["helmet_count"],
                        "trash_bag": stats["trash_count"],
                        "fire": stats["fire_count"],
                        "smoke": stats["smoke_count"],
                        "weapon": stats["weapon_count"],
                        "event_detail": "any_detection"
                    }
                    self._save_image_and_caption(
                        img_bgr=raw_frame, vis_bgr=vis, stamp_meta=stamp_meta,
                        frame_idx=frame_idx, event_type="any", info_dict=info_dict,
                        prompt=domain_prompt
                    )

                # ===== 출력 =====
                writer.write(vis)
                if self.preview:
                    cv2.imshow("PM/Rider/Helmet (Video)", vis)
                    # ESC or q
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
                print(f"[INFO] VLM captions jsonl: {self.vlm_jsonl_path.resolve()}")

# ======================
# CLI
# ======================
def parse_args():
    p = argparse.ArgumentParser(description="Local Video → YOLO one-pass w/ majority voting, event logs, InternVL captions")
    p.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="input video path")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help=".pt path (4 or 7 classes)")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="output directory")
    p.add_argument("--out_suffix", type=str, default=DEFAULT_OUT_SUFFIX, help="output filename suffix (appended to stem)")

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
    p.add_argument("--vlm_enable", type=int, default=int(DEFAULT_VLM_ENABLE), help="1 to enable InternVL captioning")
    p.add_argument("--vlm_model", type=str, default=DEFAULT_VLM_MODEL, help="e.g., OpenGVLab/InternVL3-1B")
    p.add_argument("--vlm_device_map", type=str, default=DEFAULT_VLM_DEVICE_MAP,
                   help="device map for InternVL loader: 'auto' | 'None' | JSON string (e.g., '{\"\": \"cuda:0\"}')")
    p.add_argument("--vlm_max_new_tokens", type=int, default=DEFAULT_VLM_MAXTOK)
    p.add_argument("--vlm_condition", type=str, default=DEFAULT_VLM_COND, help="event|any")
    p.add_argument("--vlm_lang", type=str, default=DEFAULT_VLM_LANG, help="ko|en")
    p.add_argument("--vlm_save_annotated", type=int, default=int(DEFAULT_VLM_SAVE_ANN), help="1: save annotated frame (vis), 0: save raw frame")
    p.add_argument("--vlm_min_interval", type=float, default=DEFAULT_VLM_MIN_INTERVAL, help="min seconds between captions in 'any' mode")
    return p.parse_args()

def main():
    args = parse_args()
    app = VideoPMHelmetApp(args)
    app.run()

if __name__ == "__main__":
    main()
