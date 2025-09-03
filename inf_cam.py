#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 camera subscriber -> Ultralytics YOLO one-pass -> temporal majority vote -> annotated image publisher
+ Logging: no-helmet always, hazards only when crowd (persons >= 6)

Supports both 4-class and 7-class models (auto-resolve).
"""

import re
import os
import csv
import cv2
import math
import time
import json
import numpy as np
from pathlib import Path
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.time import Time

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
from datetime import datetime

# ======================
# 기본 설정 (ROS 파라미터로 덮어쓰기 가능)
# ======================
WEIGHTS = "/home/jaehyeon/Desktop/neubility/Dataset/runs/train0/final_s/weights/best.pt"
INPUT_TOPIC = "/camera/image_raw"                # 또는 "/camera/image_raw/compressed"
OUTPUT_TOPIC = "/pm_helmet/image_annotated"
CONF_THRES = 0.25
IOU_THRES  = 0.5

# 헬멧 판정 파라미터
HEAD_REGION_RATIO = 0.6
ROI_TOP_EXTRA = 0.2
ROI_SIDE_EXTRA = 0.05

# 시각화 색상 (BGR)
COLOR_PM      = (255, 160, 0)
COLOR_PERSON  = (0, 200, 0)
COLOR_RIDER   = (0, 140, 255)
COLOR_HELMET  = (0, 0, 255)   # (라벨 배경 색상용, 헬멧 박스는 그리지 않음)
COLOR_TRASH   = (180, 180, 30)
COLOR_FIRE    = (0, 0, 255)
COLOR_SMOKE   = (160, 160, 160)
COLOR_WEAPON  = (180, 0, 180)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 시각화 정책
VIS_SHOW_RIDERS_ALWAYS = True
PERSON_DRAW_MIN_COUNT  = 6     # 일반 Person은 이 값 이상일 때만 박스 표시
CROWD_PERSON_THRESHOLD = 6     # 군집 판단 임계치

# ========== 유틸 ==========
def expand_box(xyxy, factor, img_w, img_h):
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    nw = w * factor
    nh = h * factor
    nx1 = max(0, cx - nw / 2.0)
    ny1 = max(0, cy - nh / 2.0)
    nx2 = min(img_w - 1, cx + nw / 2.0)
    ny2 = min(img_h - 1, cy + nh / 2.0)
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

def _expand_person_roi(pb, w, h, top_extra=ROI_TOP_EXTRA, side_extra=ROI_SIDE_EXTRA):
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

# ---------- 클래스 매핑(4cls/7cls 자동 지원) ----------
def _normalize(s: str):
    s = s.lower().strip()
    s = re.sub(r"[_\-\s]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())

def _find_one(norm_map, candidates):
    cands = [_normalize(c) for c in candidates]
    for i, n in norm_map.items():
        if n in cands: return i
    for i, n in norm_map.items():
        if any(c in n or n in c for c in cands): return i
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
        self.votes = deque(maxlen=vote_window)  # values: True/False/None
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
        # dets: list of np.array xyxy
        matches = {}     # det_idx -> track
        if not self.tracks or not dets:
            return matches, list(range(len(dets)))
        ious = np.zeros((len(self.tracks), len(dets)), dtype=np.float32)
        for ti, tr in enumerate(self.tracks):
            for di, db in enumerate(dets):
                ious[ti, di] = iou_xyxy(tr.bbox, db)
        # 모든 (ti,di) 중 iou>thresh를 내림차순으로 정렬하여 그리디 매칭
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
        """
        person_boxes: [N,4] xyxy
        rider_flags:  [N] bool
        inst_has_helmet_list: [N] bool (라이더만 True/False, 비라이더는 False로 들어와도 됨)
        returns: for each det idx -> (track_id, smoothed_has_helmet or None, was_new_nohelmet_event)
        """
        self.frame_idx += 1
        dets = [np.array(b, dtype=np.float32) for b in person_boxes]
        matches, unmatched = self._assign(dets)

        # 업데이트/추가
        for di, tr in matches.items():
            tr.bbox = dets[di]
            tr.last_frame = self.frame_idx
            tr.rider_recent = bool(rider_flags[di])
            # 표결 업데이트: 라이더일 때만 유효표
            if tr.rider_recent:
                tr.votes.append(bool(inst_has_helmet_list[di]))
            else:
                tr.votes.append(None)

        for di in unmatched:
            tr = Track(self.next_tid, dets[di], self.frame_idx, self.vote_window)
            tr.rider_recent = bool(rider_flags[di])
            tr.votes.append(bool(inst_has_helmet_list[di]) if tr.rider_recent else None)
            self.tracks.append(tr)
            matches[di] = tr
            self.next_tid += 1

        # 오래된 트랙 제거
        alive = []
        for tr in self.tracks:
            if self.frame_idx - tr.last_frame <= self.max_age_frames:
                alive.append(tr)
        self.tracks = alive

        # 결과/이벤트
        det_to_out = {}
        for di, tr in matches.items():
            valid = [v for v in tr.votes if v is not None]
            smoothed = None
            if len(valid) >= self.vote_min_valid:
                helmet_ratio = sum(1 for v in valid if v) / float(len(valid))
                smoothed = (helmet_ratio >= self.vote_threshold)
            # no-helmet 신규 이벤트 발생 여부(상태 전이 감지)
            was_event = False
            if smoothed is not None and tr.prev_smoothed is not None:
                if (tr.prev_smoothed is True) and (smoothed is False):
                    was_event = True
            elif smoothed is False and tr.prev_smoothed is None:
                # 초기값이 바로 NoHelmet이면 이벤트로 간주(선택 사항)
                was_event = True
            tr.prev_smoothed = smoothed
            det_to_out[di] = (tr.tid, smoothed, was_event)
        return det_to_out

# ======================
# ROS2 노드
# ======================
class PMHelmetNode(Node):
    def __init__(self):
        super().__init__("pm_helmet_inference_vote_log")

        # ---- ROS Parameters
        self.declare_parameter("weights", WEIGHTS)
        self.declare_parameter("input_topic", INPUT_TOPIC)
        self.declare_parameter("output_topic", OUTPUT_TOPIC)
        self.declare_parameter("conf_thres", CONF_THRES)
        self.declare_parameter("iou_thres", IOU_THRES)
        self.declare_parameter("head_region_ratio", HEAD_REGION_RATIO)
        self.declare_parameter("roi_top_extra", ROI_TOP_EXTRA)
        self.declare_parameter("roi_side_extra", ROI_SIDE_EXTRA)
        self.declare_parameter("preview", False)
        self.declare_parameter("device", "")

        # 다수결/트래킹/로그 파라미터
        self.declare_parameter("vote_window", 5)
        self.declare_parameter("vote_min_valid", 3)
        self.declare_parameter("vote_threshold", 0.5)
        self.declare_parameter("track_iou_thresh", 0.3)
        self.declare_parameter("track_max_age_frames", 30)

        self.declare_parameter("log_dir", "logs")
        self.declare_parameter("log_cooldown_sec", 5.0)         # hazard 공용 쿨다운
        self.declare_parameter("no_helmet_cooldown_sec", 10.0)  # 트랙별 무헬멧 쿨다운

        # 파라미터 적용
        global CONF_THRES, IOU_THRES, HEAD_REGION_RATIO, ROI_TOP_EXTRA, ROI_SIDE_EXTRA
        CONF_THRES = float(self.get_parameter("conf_thres").value)
        IOU_THRES  = float(self.get_parameter("iou_thres").value)
        HEAD_REGION_RATIO = float(self.get_parameter("head_region_ratio").value)
        ROI_TOP_EXTRA = float(self.get_parameter("roi_top_extra").value)
        ROI_SIDE_EXTRA = float(self.get_parameter("roi_side_extra").value)

        weights = self.get_parameter("weights").value
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.preview = bool(self.get_parameter("preview").value)
        device = self.get_parameter("device").value

        vw = int(self.get_parameter("vote_window").value)
        vm = int(self.get_parameter("vote_min_valid").value)
        vt = float(self.get_parameter("vote_threshold").value)
        ti = float(self.get_parameter("track_iou_thresh").value)
        ta = int(self.get_parameter("track_max_age_frames").value)

        self.log_dir = Path(self.get_parameter("log_dir").value)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_cooldown = float(self.get_parameter("log_cooldown_sec").value)
        self.nohelmet_cooldown = float(self.get_parameter("no_helmet_cooldown_sec").value)

        # ---- YOLO Load
        self.get_logger().info(f"Loading YOLO weights: {weights}")
        self.model = YOLO(weights)
        if device:
            try:
                self.model.to(device)
                self.get_logger().info(f"Moved model to device: {device}")
            except Exception as e:
                self.get_logger().warn(f"Failed to move model to '{device}': {e}")

        # 클래스 id 맵
        self.ids = resolve_class_ids(self.model)
        self.get_logger().info(f"Resolved class ids: {self.ids}")

        # 트래커
        self.tracker = SimpleHelmetTracker(vote_window=vw, vote_min_valid=vm, vote_threshold=vt,
                                           iou_thresh=ti, max_age_frames=ta)

        # QoS & I/O
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1)

        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(Image, self.input_topic, self.image_cb, qos)
        compressed_topic = self.input_topic if self.input_topic.endswith("/compressed") else self.input_topic + "/compressed"
        self.sub_compressed = self.create_subscription(CompressedImage, compressed_topic, self.compressed_cb, qos)
        self.pub = self.create_publisher(Image, self.output_topic, qos)

        self.processing = False

        # 로그 상태(쿨다운 타이머)
        self.last_hazard_log_time = {
            "trash_bag": 0.0, "fire": 0.0, "smoke": 0.0, "weapon": 0.0
        }

        self.get_logger().info(f"Subscribed: {self.input_topic} (+ {compressed_topic})")
        self.get_logger().info(f"Publishing annotated image: {self.output_topic}")
        self.get_logger().info(f"CONF_THRES={CONF_THRES}, IOU_THRES={IOU_THRES}, vote_window={vw}, vote_min_valid={vm}, vote_threshold={vt}")

    # ---------- 공용 추론/시각화 + 로깅 ----------
    def run_inference(self, frame):
        h, w = frame.shape[:2]
        keep_ids = sorted(set(self.ids.values()))
        res = self.model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, classes=keep_ids, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return frame, 0, [], [], [], [], [], {}

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

        # 헬멧 즉시판정 (라이더만)
        inst_has_helmet = [False] * len(person_boxes)
        if len(helmet_boxes) > 0:
            for i, (pb, is_rider) in enumerate(zip(person_boxes, rider_flags)):
                if not is_rider: 
                    continue
                roi = _expand_person_roi(pb, w, h)
                for hb in helmet_boxes:
                    if _helmet_center_in_head_region(hb, roi, HEAD_REGION_RATIO):
                        inst_has_helmet[i] = True
                        break

        # 트래커 업데이트(다수결)
        det_to_track = self.tracker.update(person_boxes, rider_flags, inst_has_helmet)

        # 시각화: PM
        for pm in pm_boxes:
            x1, y1, x2, y2 = pm.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PM, 2)
            draw_label(frame, x1, y1, "PM", COLOR_PM)

        total_persons = len(person_boxes)

        # 시각화: 라이더(스무딩 결과 반영, 트랙ID 표기)
        if VIS_SHOW_RIDERS_ALWAYS:
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_RIDER, 2)
                draw_label(frame, x1, y1, label, COLOR_RIDER)

        # 시각화: 일반 Person (군집일 때만)
        if total_persons >= PERSON_DRAW_MIN_COUNT:
            for (pb, is_rider) in zip(person_boxes, rider_flags):
                if is_rider and VIS_SHOW_RIDERS_ALWAYS:
                    continue
                x1, y1, x2, y2 = pb.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
                draw_label(frame, x1, y1, "Person", COLOR_PERSON)

            text = f"Persons: {total_persons}"
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
            x0 = w - tw - 16; y0 = 16 + th + 8
            cv2.rectangle(frame, (x0 - 8, 8), (x0 + tw + 8, y0), (50, 50, 50), -1)
            cv2.putText(frame, text, (x0, 16 + th), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 기타 클래스 시각화
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

    # ---------- 로깅 ----------
    def _append_csv(self, file_path: Path, header: list, row: list):
        existed = file_path.exists()
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not existed:
                writer.writerow(header)
            writer.writerow(row)

    def _stamp_to_meta(self, header_stamp) -> dict:
        # header_stamp: builtin_interfaces.msg.Time
        if header_stamp is not None:
            sec = int(getattr(header_stamp, "sec", 0))
            nsec = int(getattr(header_stamp, "nanosec", getattr(header_stamp, "nanosec_", 0)))
        else:
            now = self.get_clock().now().to_msg()
            sec = int(now.sec); nsec = int(now.nanosec)
        t = sec + nsec * 1e-9
        iso = datetime.fromtimestamp(t).isoformat()
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
        # event_type in {"trash_bag","fire","smoke","weapon"}
        fp = self.log_dir / f"hazard_{event_type}.csv"
        header = ["stamp_sec","stamp_nsec","iso","event","count","persons"]
        row = [stamp_meta["sec"], stamp_meta["nsec"], stamp_meta["iso"], event_type, count, persons]
        self._append_csv(fp, header, row)

    # ---------- 콜백 ----------
    def _process_and_publish(self, frame, header):
        vis, total_persons, trash_boxes, fire_boxes, smoke_boxes, weapon_boxes, person_boxes, stats = self.run_inference(frame)

        # ===== 로깅 로직 =====
        stamp_meta = self._stamp_to_meta(header.stamp if header else None)
        now_t = stamp_meta["t"]

        # (A) 무헬멧(다수결) 로그: 항상
        det_to_track = stats["det_to_track"]
        rider_flags = stats["rider_flags"]
        inst_has_helmet = stats["inst_has_helmet"]

        for det_idx, (pb, is_rider) in enumerate(zip(person_boxes, rider_flags)):
            if not is_rider:
                continue
            info = det_to_track.get(det_idx)
            if not info:
                continue
            tid, smoothed, was_event = info
            if smoothed is False and was_event:
                # 트랙 객체에서 쿨다운/비율 계산
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

        # (B) 군집(사람 수 ≥ 6) + Hazard(쓰봉/화재/연기/흉기) 로그
        is_crowd = (total_persons >= CROWD_PERSON_THRESHOLD)
        if is_crowd:
            hazards = [
                ("trash_bag", len(trash_boxes)),
                ("fire",      len(fire_boxes)),
                ("smoke",     len(smoke_boxes)),
                ("weapon",    len(weapon_boxes)),
            ]
            for name, cnt in hazards:
                if cnt <= 0: 
                    continue
                last_t = self.last_hazard_log_time.get(name, 0.0)
                if (now_t - last_t) >= self.log_cooldown:
                    self.log_hazard(stamp_meta, name, cnt, total_persons)
                    self.last_hazard_log_time[name] = now_t

        # 퍼블리시
        out = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
        if header:
            out.header = header
        self.pub.publish(out)

        if self.preview:
            cv2.imshow("PM/Rider/Helmet (ROS2)", vis)
            cv2.waitKey(1)

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
        self.processing = True
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            self._process_and_publish(frame, msg.header)
        except Exception as e:
            self.get_logger().error(f"inference error (compressed): {e}")
        finally:
            self.processing = False


def main():
    rclpy.init()
    node = PMHelmetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.preview:
            try: cv2.destroyAllWindows()
            except Exception: pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
