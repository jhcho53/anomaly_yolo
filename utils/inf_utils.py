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
from datetime import datetime
from typing import Optional, List, Tuple
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
    for i, n in norm_map.items():
        if n in cands:
            return i
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