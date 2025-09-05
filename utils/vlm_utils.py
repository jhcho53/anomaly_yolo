#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_only_infer.py — Generate QA captions from videos/frames using VLM (InternVL) only

- Removes any GRU-based anomaly detection.
- Slices a video into fixed-length clips (clip_sec) and queries the VLM per clip.
- Prefers frame-direct inference (run_frames_inference); falls back to a temporary MP4
  then run_video_inference when needed.
- Tries generation_config twice for robustness across InternVL variants:
  (1) without 'use_cache', (2) with use_cache=True.

Dependencies:
  * utils/video_vlm.py (or a sibling video_vlm.py) providing:
      - init_model(path, device_map)
      - run_frames_inference(model, tokenizer, frames=..., generation_config=..., num_segments, max_num)
      - run_video_inference(model, tokenizer, video_path, generation_config=..., num_segments, max_num)
"""

import os
import cv2
import json
import tempfile
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional

# ---- InternVL backends (prefer frames, fallback to video) ----
try:
    from video_vlm import init_model as vlm_init_model
    from video_vlm import run_frames_inference, run_video_inference
except Exception:
    from utils.video_vlm import init_model as vlm_init_model
    try:
        from utils.video_vlm import run_frames_inference, run_video_inference
    except Exception as e:
        raise ImportError("utils/video_vlm.py (or video_vlm.py) with init_model/run_* is required.") from e


# =========================
# 0) ROI & common utilities
# =========================

def _ensure_roi_from_first_frame(frames: List[np.ndarray], roi):
    """
    If roi is None, return a full-frame ROI from the first frame.
    """
    if roi is not None:
        return tuple(map(int, roi))
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    return (0, 0, w, h)


def _crop_frame(frame_bgr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a frame to ROI with bounds clamped to the frame size.
    Falls back to the original frame if the resulting area is empty.
    """
    H, W = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, roi)
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    return frame_bgr[y1:y2, x1:x2]


# =========================
# 1) Temporary MP4 writer (for video fallback)
# =========================

def _write_frames_to_temp_video(frames: List[np.ndarray], fps: float) -> str:
    """Write BGR frames to a temporary MP4 file and return the path."""
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


# =========================
# 2) VLM call (frames-first, video fallback, use_cache robustness)
# =========================

def _vlm_call_on_frames(
    vlm_model,
    vlm_tokenizer,
    frames: List[np.ndarray],
    gen_cfg_dict: Optional[dict] = None,
    num_segments: int = 8,
    max_num: int = 1,
    fps_for_fallback: float = 30.0
) -> List[Tuple[str, str]]:
    """
    Call InternVL robustly across minor API differences:
      1) Try generation_config without 'use_cache'
      2) Try again with use_cache=True
    For each config, try frames-direct → video fallback (temp MP4).
    """
    if not frames:
        return []

    base = dict(
        max_new_tokens=int((gen_cfg_dict or {}).get("max_new_tokens", 64)),
        do_sample=bool((gen_cfg_dict or {}).get("do_sample", False)),
        top_p=float((gen_cfg_dict or {}).get("top_p", 1.0)),
        num_beams=int((gen_cfg_dict or {}).get("num_beams", 1)),
        repetition_penalty=float((gen_cfg_dict or {}).get("repetition_penalty", 1.05)),
        temperature=float((gen_cfg_dict or {}).get("temperature", 1.0)),
    )
    cfg_no_cache = {k: v for k, v in base.items() if k != "use_cache"}
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


# =========================
# 3) Frame-level API
# =========================

def run_vlm_on_frames(
    frames: List[np.ndarray],
    vlm_model,
    vlm_tokenizer,
    generation_config: Optional[dict] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    num_segments: int = 8,
    max_num: int = 1,
    fps_for_fallback: float = 30.0,
    tag: str = "frames_clip"
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Run a single VLM QA pass on the in-memory frame list.
    Returns: { tag: [(Q, A), ...] }
    """
    if not frames:
        return {tag: []}

    roi_eff = _ensure_roi_from_first_frame(frames, roi)
    if roi_eff is not None:
        frames = [_crop_frame(f, roi_eff) for f in frames]

    qa = _vlm_call_on_frames(
        vlm_model, vlm_tokenizer, frames,
        gen_cfg_dict=generation_config,
        num_segments=num_segments, max_num=max_num,
        fps_for_fallback=fps_for_fallback
    )
    return {tag: qa}


# =========================
# 4) Streaming video chunk API (fixed-length, no GRU)
# =========================

def _iter_video_chunks(
    video_path: str,
    clip_sec: float = 3.0,
    step_sec: Optional[float] = None,
    process_last: bool = True
):
    """
    Stream-decode a video via OpenCV and yield fixed-length chunks.
    Yields: (chunk_frames: List[np.ndarray], start_frame_idx: int, end_frame_idx: int, fps: float)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 25.0

    clip_frames = max(1, int(round(clip_sec * fps)))
    step_sec = step_sec if step_sec is not None else clip_sec
    step_frames = max(1, int(round(step_sec * fps)))

    buf = deque()
    start_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)

        # Emit a chunk once enough frames have accumulated
        if len(buf) >= clip_frames:
            frames = list(buf)[:clip_frames]
            end_idx = start_idx + clip_frames - 1
            yield frames, start_idx, end_idx, fps

            # Slide the window by step_frames
            for _ in range(min(step_frames, len(buf))):
                buf.popleft()
            start_idx = start_idx + step_frames

    # Emit the final (possibly shorter) tail
    if process_last and len(buf) > 0:
        frames = list(buf)
        end_idx = start_idx + len(frames) - 1
        yield frames, start_idx, end_idx, fps

    cap.release()


def run_vlm_over_video(
    video_path: str,
    vlm_model,
    vlm_tokenizer,
    generation_config: Optional[dict] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    clip_sec: float = 3.0,
    step_sec: Optional[float] = None,
    process_last: bool = True,
    num_segments: int = 8,
    max_num: int = 1
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Slice a video into clips of length clip_sec and run QA for each chunk.
    Returns: { "s-e(sec)": [(Q, A), ...], ... }
    """
    results: Dict[str, List[Tuple[str, str]]] = {}
    for frames, s_idx, e_idx, fps in _iter_video_chunks(video_path, clip_sec, step_sec, process_last):
        # Apply ROI
        roi_eff = _ensure_roi_from_first_frame(frames, roi)
        if roi_eff is not None:
            frames = [_crop_frame(f, roi_eff) for f in frames]

        qa = _vlm_call_on_frames(
            vlm_model, vlm_tokenizer, frames,
            gen_cfg_dict=generation_config,
            num_segments=num_segments, max_num=max_num,
            fps_for_fallback=fps
        )
        s_sec = s_idx / fps
        e_sec = e_idx / fps
        key = f"{s_sec:.2f}-{e_sec:.2f}s"
        results[key] = qa
    return results


# =========================
# 5) End-to-end wrappers (backward compatible)
# =========================

def run_inference_on_frames(
    frames: List[np.ndarray],
    vlm_model,
    vlm_tokenizer,
    generation_config: Optional[dict] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    num_segments: int = 8,
    vlm_max_num: int = 1,
    tag: str = "stream_clip",
    fps_for_vlm_fallback: float = 30.0
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Backward-compatible wrapper for frame inputs → {tag: [(Q, A), ...]}.
    """
    return run_vlm_on_frames(
        frames=frames,
        vlm_model=vlm_model,
        vlm_tokenizer=vlm_tokenizer,
        generation_config=generation_config,
        roi=roi,
        num_segments=num_segments,
        max_num=vlm_max_num,
        fps_for_fallback=fps_for_vlm_fallback,
        tag=tag
    )


def run_inference(
    video_path: str,
    # VLM
    vlm_model=None,
    vlm_tokenizer=None,
    generation_config: Optional[dict] = None,
    # slicing / preprocessing
    clip_sec: float = 3.0,
    step_sec: Optional[float] = None,
    process_last: bool = True,
    roi: Optional[Tuple[int, int, int, int]] = None,
    num_segments: int = 8,
    vlm_max_num: int = 1
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Backward-compatible wrapper for file path → {"s-e(sec)": [(Q, A), ...]}.
    Internally uses fixed-length chunks with frames-first / video-fallback logic.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    return run_vlm_over_video(
        video_path=video_path,
        vlm_model=vlm_model,
        vlm_tokenizer=vlm_tokenizer,
        generation_config=generation_config,
        roi=roi,
        clip_sec=clip_sec,
        step_sec=step_sec,
        process_last=process_last,
        num_segments=num_segments,
        max_num=vlm_max_num
    )


# =========================
# 6) Convenience: VLM loader + simple example
# =========================

_VLM_CACHE: Dict[str, Tuple[object, object]] = {}

def load_vlm(model_name="OpenGVLab/InternVL3-1B", device_map="auto"):
    """
    Cached VLM loader (delegates to video_vlm.init_model).
    """
    key = f"{model_name}::{device_map}"
    if key in _VLM_CACHE:
        return _VLM_CACHE[key]
    model, tok = vlm_init_model(model_name, device_map=device_map)
    _VLM_CACHE[key] = (model, tok)
    return model, tok


if __name__ == "__main__":
    # Minimal runnable example
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--model", type=str, default="OpenGVLab/InternVL3-1B")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--clip_sec", type=float, default=3.0)
    p.add_argument("--step_sec", type=float, default=None)
    p.add_argument("--roi", type=str, default="")
    p.add_argument("--num_segments", type=int, default=8)
    p.add_argument("--max_num", type=int, default=1)
    p.add_argument("--maxtok", type=int, default=64)
    args = p.parse_args()

    roi = None
    if args.roi:
        parts = [int(x) for x in args.roi.replace(",", " ").split()]
        assert len(parts) == 4, "ROI must be 'x1,y1,x2,y2'"
        roi = tuple(parts)  # type: ignore

    vlm_m, vlm_t = load_vlm(args.model, device_map=args.device_map)
    gen = dict(max_new_tokens=args.maxtok, do_sample=False, num_beams=1, top_p=1.0, repetition_penalty=1.05)

    res = run_inference(
        video_path=args.source,
        vlm_model=vlm_m,
        vlm_tokenizer=vlm_t,
        generation_config=gen,
        clip_sec=args.clip_sec,
        step_sec=args.step_sec,
        roi=roi,
        num_segments=args.num_segments,
        vlm_max_num=args.max_num
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
