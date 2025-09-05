#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_only_infer.py — VLM(InternVL)만 사용해 비디오/프레임에서 QA 캡션 생성

- GRU 기반 이상탐지 완전 제거
- 비디오를 고정 길이 청크(clip_sec)로 슬라이싱하여 각 청크를 VLM에 질의
- 프레임 직접 추론(run_frames_inference) 우선, 불가하면 임시 MP4 → run_video_inference 폴백
- generation_config는 'use_cache' 유무 두 가지 구성을 순차 시도(버전 차이 안전)

필요 모듈:
  * utils/video_vlm.py (또는 동일 디렉토리 video_vlm.py):
      - init_model(path, device_map)
      - run_frames_inference(model, tokenizer, frames=..., generation_config=..., num_segments, max_num)
      - run_video_inference(model, tokenizer, video_path, generation_config=..., num_segments, max_num)
"""

import os
import cv2
import json
import time
import tempfile
import numpy as np
from collections import deque
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# ---- InternVL backends (프레임 우선, 불가 시 비디오 폴백) ----
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
# 0) ROI & 공통 유틸
# =========================

def _ensure_roi_from_first_frame(frames: List[np.ndarray], roi):
    """
    roi가 None이면 첫 프레임 전체 영역을 ROI로 반환.
    """
    if roi is not None:
        return tuple(map(int, roi))
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    return (0, 0, w, h)

def _ensure_roi_from_video(video_path: str, roi):
    if roi is not None:
        return tuple(map(int, roi))
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if w > 0 and h > 0:
            return (0, 0, w, h)
    return None

def _crop_frame(frame_bgr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, roi)
    return frame_bgr[y1:y2, x1:x2]


# =========================
# 1) 임시 mp4 작성(비디오 폴백용)
# =========================

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


# =========================
# 2) VLM 실행 (프레임 → 비디오 폴백 + use_cache 안전화)
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
    InternVL 버전 차이 안전:
      1) use_cache 미포함(dict)
      2) use_cache=True 포함(dict)
    각 후보에 대해 frames 경로 우선 → 비디오 폴백 순서로 시도.
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
    cfg_no_cache = {k: v for k, v in base.items() if k != "use_cache"}  # 보장
    cfg_with_cache = dict(cfg_no_cache, use_cache=True)

    cfg_candidates = [cfg_no_cache, cfg_with_cache]

    def _try_frames(cfg):
        try:
            return run_frames_inference(
                model=vlm_model,
                tokenizer=vlm_tokenizer,
                frames=frames,                 # 프레임 직접
                generation_config=cfg,         # 4번째 위치 인자로 chat()에 전달됨
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
                try: os.remove(tmp)
                except Exception: pass
        except Exception as e:
            print(f"[VLM] video inference failed ({'use_cache' in cfg}): {e}")
            return None

    # 1) 프레임 경로 우선
    for cfg in cfg_candidates:
        qa = _try_frames(cfg)
        if qa:
            return qa

    # 2) 비디오 폴백
    for cfg in cfg_candidates:
        qa = _try_video(cfg)
        if qa:
            return qa

    print("[VLM] caption failed: all attempts exhausted")
    return []


# =========================
# 3) 프레임 단위 API
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
    메모리에 있는 프레임 리스트를 받아 한 번의 VLM QA를 수행.
    반환: { tag: [(Q, A), ...] }
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
# 4) 비디오 스트리밍 청크 API (GRU 없이 고정 길이)
# =========================

def _iter_video_chunks(
    video_path: str,
    clip_sec: float = 3.0,
    step_sec: Optional[float] = None,
    process_last: bool = True
):
    """
    OpenCV로 스트리밍 디코드하며 고정 길이 청크를 생성하는 제너레이터.
    Yields: (chunk_frames: List[np.ndarray], start_frame_idx:int, end_frame_idx:int, fps:float)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:  # 합리적 기본값
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    clip_frames = max(1, int(round(clip_sec * fps)))
    step_sec = step_sec if step_sec is not None else clip_sec
    step_frames = max(1, int(round(step_sec * fps)))

    buf = deque()
    start_idx = 0
    read_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(frame)
        read_idx += 1

        # 충분히 쌓였으면 1개 청크 배출
        if len(buf) >= clip_frames:
            frames = list(buf)[:clip_frames]
            end_idx = start_idx + clip_frames - 1
            yield frames, start_idx, end_idx, fps

            # step 만큼 슬라이드
            for _ in range(min(step_frames, len(buf))):
                buf.popleft()
            start_idx = start_idx + step_frames

    # 마지막 잔여 처리
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
    비디오를 clip_sec 단위로 슬라이스해 각 청크에 대해 QA 수행.
    반환: { "s-e(sec)": [(Q,A), ...], ... }
    """
    results: Dict[str, List[Tuple[str, str]]] = {}
    for frames, s_idx, e_idx, fps in _iter_video_chunks(video_path, clip_sec, step_sec, process_last):
        # ROI 적용
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
# 5) 엔드투엔드 래퍼 (이전 인터페이스 유지)
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
    (이전 시그니처 유지) 프레임 리스트 입력 → {tag: [(Q,A), ...]}
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
    # VLM 인자
    vlm_model=None,
    vlm_tokenizer=None,
    generation_config: Optional[dict] = None,
    # 슬라이싱/전처리
    clip_sec: float = 3.0,
    step_sec: Optional[float] = None,
    process_last: bool = True,
    roi: Optional[Tuple[int, int, int, int]] = None,
    num_segments: int = 8,
    vlm_max_num: int = 1
) -> Dict[str, List[Tuple[str, str]]]:
    """
    (이전 시그니처 유지) 파일 경로 입력 → { "s-e(sec)": [(Q,A), ...] }
    내부는 고정 길이 청크 + 프레임 우선/비디오 폴백 공용 로직 사용.
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
# 6) 편의 유틸: VLM 로더 + 간단 예제
# =========================

_VLM_CACHE: Dict[str, Tuple[object, object]] = {}

def load_vlm(model_name="OpenGVLab/InternVL3-1B", device_map="auto"):
    """
    InternVL 로더 캐시 (video_vlm.init_model 사용).
    """
    key = f"{model_name}::{device_map}"
    if key in _VLM_CACHE:
        return _VLM_CACHE[key]
    model, tok = vlm_init_model(model_name, device_map=device_map)
    _VLM_CACHE[key] = (model, tok)
    return model, tok


if __name__ == "__main__":
    # 간단 실행 예시
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
