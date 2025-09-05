import re
import cv2
import numpy as np
from typing import Dict, Optional

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ======================
# Geometry / drawing utils
# ======================
def expand_box(xyxy: np.ndarray, factor: float, img_w: int, img_h: int) -> np.ndarray:
    """
    Expand a bounding box by a given factor while clamping to image boundaries.

    Args:
        xyxy: [x1, y1, x2, y2] float/int array.
        factor: Scale factor (>1 expands, <1 shrinks).
        img_w, img_h: Image width/height for clamping.

    Returns:
        np.ndarray of shape (4,) and dtype float32.
    """
    x1, y1, x2, y2 = map(float, xyxy)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    nw = w * factor
    nh = h * factor
    nx1 = max(0.0, cx - nw / 2.0)
    ny1 = max(0.0, cy - nh / 2.0)
    nx2 = min(float(img_w - 1), cx + nw / 2.0)
    ny2 = min(float(img_h - 1), cy + nh / 2.0)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute IoU between two boxes in (x1, y1, x2, y2) format.

    Args:
        a, b: arrays or sequences [x1, y1, x2, y2].

    Returns:
        IoU in [0, 1].
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-6  # epsilon to avoid zero-division
    return float(inter / union)


def draw_label(
    img: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: tuple,
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """
    Draw a filled label box with text (BGR color).

    Args:
        img: BGR image (in-place drawing).
        x, y: Anchor point (bottom-left of the label box).
        text: Label text.
        color: (B, G, R).
        scale: Font scale.
        thickness: Line thickness.
    """
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    # Background rectangle
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    # Text
    cv2.putText(img, text, (x + 3, y - 4), FONT, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _expand_person_roi(
    pb: np.ndarray,
    w: int,
    h: int,
    top_extra: float,
    side_extra: float,
) -> np.ndarray:
    """
    Expand a person's ROI slightly to include head/shoulder margins.

    Args:
        pb: person box [x1, y1, x2, y2].
        w, h: image width/height for clamping.
        top_extra: fraction of box height to extend upward.
        side_extra: fraction of box width to extend sideways.

    Returns:
        Expanded ROI [x1, y1, x2, y2], float32.
    """
    x1, y1, x2, y2 = map(float, pb)
    pw = x2 - x1
    ph = y2 - y1
    nx1 = max(0.0, x1 - pw * side_extra)
    nx2 = min(float(w - 1), x2 + pw * side_extra)
    ny1 = max(0.0, y1 - ph * top_extra)
    ny2 = min(float(h - 1), y2)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)


def _helmet_center_in_head_region(
    helmet_box: np.ndarray,
    roi: np.ndarray,
    head_ratio: float,
) -> bool:
    """
    Decide whether the helmet box center is within the 'head region' of a person's ROI.

    The head region is defined as the upper (head_ratio) portion of the ROI height.

    Args:
        helmet_box: [x1, y1, x2, y2].
        roi: expanded person ROI [x1, y1, x2, y2].
        head_ratio: fraction (e.g., 0.6 means upper 60%).

    Returns:
        True if helmet center lies in the head region; otherwise False.
    """
    hx1, hy1, hx2, hy2 = map(float, helmet_box)
    cx = (hx1 + hx2) / 2.0
    cy = (hy1 + hy2) / 2.0
    rx1, ry1, rx2, ry2 = map(float, roi)
    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
        return False
    head_y_limit = ry1 + (ry2 - ry1) * head_ratio
    return cy <= head_y_limit


# ======================
# Class mapping (auto 4/7 classes)
# ======================
def _normalize(s: str) -> str:
    """
    Normalize class names: lowercase, collapse separators, strip non-alnum (keeps spaces).
    """
    s = s.lower().strip()
    s = re.sub(r"[_\-\s]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


def _find_one(norm_map: Dict[int, str], candidates) -> Optional[int]:
    """
    Find a class id in norm_map that matches any candidate (exact, then partial).
    """
    cands = [_normalize(c) for c in candidates]
    # Exact match
    for i, n in norm_map.items():
        if n in cands:
            return i
    # Soft (substring) match
    for i, n in norm_map.items():
        if any(c in n or n in c for c in cands):
            return i
    return None


def resolve_class_ids(model) -> Dict[str, int]:
    """
    Resolve class indices for the expected labels from a Ultralytics YOLO model.

    Expected keys (if present in model.names):
      - 'person' (required)
      - 'pm' (personal mobility / e-scooter) (required)
      - 'trash_bag'
      - 'helmet' (required)
      - 'fire'
      - 'smoke'
      - 'weapon'

    Raises:
        ValueError if any required classes are missing.

    Returns:
        Dict mapping name -> class index (e.g., {'person': 0, 'pm': 1, 'helmet': 2, ...}).
    """
    raw = model.names
    id_to_name = {int(k): v for k, v in (raw.items() if isinstance(raw, dict) else enumerate(raw))}
    norm_map = {i: _normalize(n) for i, n in id_to_name.items()}
    CANDS = {
        "person":    ["person"],
        "pm":        ["pm", "personal mobility", "kickboard", "e scooter", "escooter",
                      "electric scooter", "micromobility", "micro mobility", "e_scooter"],
        "trash_bag": ["trash bag", "garbage bag", "plastic bag", "trash", "garbage"],
        "helmet":    ["helmet", "hardhat", "safety helmet"],
        "fire":      ["fire"],
        "smoke":     ["smoke"],
        "weapon":    ["weapon"],
    }
    ids: Dict[str, int] = {}
    for key, cands in CANDS.items():
        idx = _find_one(norm_map, cands)
        if idx is not None:
            ids[key] = idx

    required = ["person", "pm", "helmet"]
    missing = [k for k in required if k not in ids]
    if missing:
        raise ValueError(
            f"[ERROR] Required classes missing: {missing} | model.names={id_to_name}"
        )
    return ids
