import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

from typing import List, Tuple, Optional

# ---------- Optional backends ----------
try:
    from decord import VideoReader, cpu
    _USE_DECORD = True
except Exception:
    VideoReader = None
    cpu = None
    _USE_DECORD = False

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    cv2 = None
    _HAVE_CV2 = False

try:
    import av
    _HAVE_AV = True
except Exception:
    av = None
    _HAVE_AV = False
# --------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def find_closest_aspect_ratio(ar, target_ratios, w, h, image_size):
    best, diff = (1, 1), float('inf')
    area = w * h
    for i, j in target_ratios:
        tar = i / j
        d = abs(ar - tar)
        if d < diff or (d == diff and area > 0.5 * image_size * image_size * i * j):
            best, diff = (i, j), d
    return best


def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    InternVL 비전 타워에서 사용하는 멀티-타일 전처리.
    """
    w, h = image.size
    ar = w / h
    ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )
    i, j = find_closest_aspect_ratio(ar, ratios, w, h, image_size)
    tw, th = image_size * i, image_size * j
    img_r = image.resize((tw, th))
    blocks = i * j
    tiles = []
    cols = tw // image_size
    for idx in range(blocks):
        x0 = (idx % cols) * image_size
        y0 = (idx // cols) * image_size
        tiles.append(img_r.crop((x0, y0, x0 + image_size, y0 + image_size)))
    if use_thumbnail and blocks > 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def split_model(model_name):
    """
    (선택) 다중 GPU용 장치 맵 헬퍼.
    주의: 기본적으로 사용하지 않습니다. 사용시 루트 키("")를 반드시 넣으세요.
    """
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    layers = getattr(getattr(cfg, 'llm_config', cfg), 'num_hidden_layers', None)
    if layers is None:
        raise ValueError("Cannot determine num_hidden_layers from config.")
    gpus = torch.cuda.device_count()
    if gpus <= 0:
        return {}
    per = math.ceil(layers / (gpus - 0.5))
    counts = [math.ceil(per * 0.5)] + [per] * (gpus - 1)
    dm = {"": 0}  # ★ 루트 기본은 GPU0
    cnt = 0
    for gpu, num in enumerate(counts):
        for _ in range(num):
            if cnt >= layers:
                break
            dm[f'language_model.model.layers.{cnt}'] = gpu
            cnt += 1
        if cnt >= layers:
            break
    for key in ['vision_model', 'mlp1',
                'language_model.model.tok_embeddings',
                'language_model.model.embed_tokens',
                'language_model.output',
                'language_model.model.norm',
                'language_model.model.rotary_emb',
                'language_model.lm_head',
                f'language_model.model.layers.{layers-1}']:
        dm[key] = 0
    return dm


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
        kwargs["device_map"] = device_map  # 사용자가 직접 넘긴 맵

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
        # device_map에서 키 누락 등 발생 시: 루트 매핑으로 재시도
        try:
            km = kwargs.copy()
            km["device_map"] = {"": "cuda:0"} if use_cuda else None
            model = AutoModel.from_pretrained(
                path,
                dtype=prefer_dtype,
                **km
            ).eval()
        except Exception:
            # 최후 수단: device_map 완전 해제 후 로드 → 이후 to(cuda)
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


def get_frame_indices(bound, fps, max_frame, num_segments, first_idx=0):
    if bound:
        s, e = bound
    else:
        s, e = -1e5, 1e5
    si = max(first_idx, round(s * fps))
    ei = min(round(e * fps), max_frame)
    span = max((ei - si) / float(max(1, num_segments)), 0.0)
    return [int(min(max_frame, max(first_idx, si + span/2 + span * i))) for i in range(num_segments)]


# ------------------------- Backend helpers -------------------------
def _safe_fps(val, default=25.0):
    try:
        v = float(val)
        return v if v > 1e-6 else default
    except Exception:
        return default


def _meta_decord(video_path):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = _safe_fps(vr.get_avg_fps(), default=25.0)
    max_frame = len(vr) - 1
    return vr, fps, max_frame


def _frames_decord(vr, indices):
    frames = []
    for idx in indices:
        arr = vr[idx].asnumpy()  # RGB
        frames.append(arr)
    return frames


def _meta_opencv(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video via OpenCV: {video_path}")
    fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS), default=25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frame = total - 1 if total > 0 else None
    return cap, fps, max_frame


def _frames_opencv(cap, indices):
    frames = []
    last_valid = None
    for idx in indices:
        ok = cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok2, bgr = cap.read()
        if ok and ok2 and bgr is not None:
            rgb = bgr[:, :, ::-1]
            last_valid = rgb
            frames.append(rgb)
        else:
            if last_valid is None:
                raise RuntimeError(f"Failed to read frame {idx} via OpenCV and no prior frame to fallback.")
            frames.append(last_valid)
    return frames


def _meta_pyav(video_path):
    container = av.open(video_path)
    streams = [s for s in container.streams if s.type == 'video']
    if not streams:
        container.close()
        raise RuntimeError("No video stream found in file.")
    vstream = streams[0]
    fps = _safe_fps(float(vstream.average_rate) if vstream.average_rate else None, default=25.0)
    if getattr(vstream, "frames", 0):
        max_frame = int(vstream.frames) - 1
        container.close()
        return fps, max_frame
    cnt = 0
    for _ in container.decode(video=0):
        cnt += 1
    container.close()
    max_frame = max(0, cnt - 1)
    return fps, max_frame


def _frames_pyav(video_path, indices):
    idx_set = set(int(i) for i in indices)
    max_idx = max(idx_set) if idx_set else -1
    out = {}
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in idx_set:
                out[i] = frame.to_ndarray(format="rgb24")
            if i >= max_idx and len(out) == len(idx_set):
                break
    frames, last_valid = [], None
    for i in indices:
        arr = out.get(int(i), None)
        if arr is None:
            if last_valid is None:
                raise RuntimeError(f"Missing frame {i} in PyAV decode and no prior frame.")
            arr = last_valid
        frames.append(arr)
        last_valid = arr
    return frames
# ------------------------------------------------------------------


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    video_path에서 프레임을 읽어 InternVL 전처리(타일링)까지 수행.
    반환:
      pixel_values: [sum_patches, 3, H, W] float32 (나중에 bfloat16/float16 캐스팅)
      patch_counts: 각 프레임 당 타일 개수 리스트
    """
    transform = build_transform(input_size)

    backend_used = None
    frames_np = []
    patch_counts = []

    try:
        if _USE_DECORD:
            vr, fps, max_frame = _meta_decord(video_path)
            indices = get_frame_indices(bound, fps, max_frame, num_segments)
            frames_np = _frames_decord(vr, indices)
            backend_used = 'decord'
        else:
            raise RuntimeError("Decord not available")
    except Exception:
        if _HAVE_CV2:
            try:
                cap, fps, max_frame = _meta_opencv(video_path)
                if max_frame is None:
                    if _HAVE_AV:
                        try:
                            fps2, max_frame2 = _meta_pyav(video_path)
                            fps = fps if fps > 1e-6 else fps2
                            max_frame = max_frame2
                        except Exception:
                            pass
                    if max_frame is None:
                        count = 0
                        ok = True
                        while ok:
                            ok, frm = cap.read()
                            if ok:
                                count += 1
                        cap.release()
                        cap = cv2.VideoCapture(video_path)
                        max_frame = max(0, count - 1)
                indices = get_frame_indices(bound, fps, max_frame, num_segments)
                frames_np = _frames_opencv(cap, indices)
                cap.release()
                backend_used = 'opencv'
            except Exception:
                backend_used = None

        if backend_used is None and _HAVE_AV:
            fps, max_frame = _meta_pyav(video_path)
            indices = get_frame_indices(bound, fps, max_frame, num_segments)
            frames_np = _frames_pyav(video_path, indices)
            backend_used = 'pyav'

    if backend_used is None or not frames_np:
        raise ImportError(
            "No usable video backend found. Install one of:\n"
            "- decord (source build on Jetson), or\n"
            "- OpenCV (apt-get install -y python3-opencv), or\n"
            "- PyAV (pip install av)."
        )

    # Preprocess into tiles/patches
    pixel_vals = []
    patch_counts = []
    for arr in frames_np:
        img = Image.fromarray(arr).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        tv = torch.stack([transform(t) for t in tiles])
        patch_counts.append(tv.shape[0])
        pixel_vals.append(tv)

    return torch.cat(pixel_vals, dim=0), patch_counts


def prepare_video_prompts(num_patches_list):
    return ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

# --------------------- BBox-focused single prompt --------------------- #
_DEFAULT_PROMPT_EN = (
    "In one concise sentence, summarize the scene from a safety perspective with a bounding‑box focus. "
)

def _build_bbox_prompt(lang: str = "ko", hint: Optional[str] = None) -> str:
    base = _DEFAULT_PROMPT_EN if (lang or "").lower().startswith("ko") else _DEFAULT_PROMPT_EN
    if hint:
        return f"{base}\nHint: {hint}"
    return base
# --------------------------------------------------------------------- #


def _to_device_dtype(pv: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    device = next(model.parameters()).device
    return pv.to(torch.float16 if device.type == "cuda" else torch.float32).to(device)


def _call_chat_single_turn(model, tokenizer, pixel_values, prompt, generation_config, **kwargs):
    """
    InternVL 구현 차이를 고려한 안전 호출자.
    1) 가장 단순한 서명으로 호출
    2) 실패 시 num_patches_list/history/return_history 변화해 재시도
    3) 최종 실패 시 예외 전파
    """
    # 1) 가장 단순한 형태 (권장)
    try:
        out = model.chat(tokenizer, pixel_values, prompt, generation_config, **kwargs)
        return out[0] if isinstance(out, tuple) else out
    except TypeError:
        pass

    # 2) return_history 제거/추가 등 가변 시도
    trial_kwargs = [
        kwargs,  # as-is
        {k: v for k, v in kwargs.items() if k != "return_history"},
        dict(kwargs, history=None),
        dict({k: v for k, v in kwargs.items() if k != "return_history"}, history=None),
    ]
    last_err = None
    for kw in trial_kwargs:
        try:
            out = model.chat(tokenizer, pixel_values, prompt, generation_config, **kw)
            return out[0] if isinstance(out, tuple) else out
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("InternVL chat failed")


def run_video_inference(
    model,
    tokenizer,
    video_path: str,
    generation_config,
    bound=None,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: int = 8,
    prompt: Optional[str] = None,
    lang: str = "ko",
    hint: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    파일 경로 방식(단일 턴):
      - 비디오에서 일정 간격 프레임 추출 → 타일 전처리 → 단일 프롬프트 질의
      - 반환은 [(prompt_used, answer)] 1개만
    """
    pv, patch_list = load_video(video_path, bound, input_size, max_num, num_segments)
    pv = _to_device_dtype(pv, model)

    prefix = prepare_video_prompts(patch_list)
    final_prompt = prefix + (prompt if prompt else _build_bbox_prompt(lang=lang, hint=hint))

    # InternVL 호출
    answer = _call_chat_single_turn(
        model, tokenizer, pv, final_prompt, generation_config,
        num_patches_list=patch_list  # InternVL에서 패치 수를 알려주는 케이스 지원
    )
    return [(final_prompt, answer)]


# ======================= frames_direct support =========================
def _uniform_indices_from_length(total: int, num_segments: int) -> List[int]:
    if total <= 0:
        return []
    num_segments = max(1, int(num_segments))
    if num_segments == 1:
        return [total - 1]  # 마지막 프레임 하나(게이팅용)
    return np.linspace(0, total - 1, num_segments).astype(int).tolist()


def _bgr_ndarray_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    if arr is None:
        raise ValueError("Empty frame array.")
    if _HAVE_CV2:
        arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        arr_rgb = arr[:, :, ::-1].copy()
    return Image.fromarray(arr_rgb)


def _frames_to_pixel_values(
    frames_bgr: List[np.ndarray],
    input_size: int = 448,
    max_num: int = 1
) -> Tuple[torch.Tensor, List[int]]:
    transform = build_transform(input_size)
    pixel_vals = []
    patch_counts: List[int] = []
    for bgr in frames_bgr:
        img = _bgr_ndarray_to_pil_rgb(bgr)
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        tv = torch.stack([transform(t) for t in tiles])  # [P,3,H,W]
        patch_counts.append(tv.shape[0])
        pixel_vals.append(tv)
    if not pixel_vals:
        raise ValueError("No frames after preprocessing.")
    return torch.cat(pixel_vals, dim=0), patch_counts


def run_frames_inference(
    model,
    tokenizer,
    frames_bgr: Optional[List[np.ndarray]] = None,
    generation_config=None,
    num_segments: int = 4,
    max_num: int = 1,
    input_size: int = 448,
    prompt: Optional[str] = None,
    lang: str = "ko",
    hint: Optional[str] = None,
    **kwargs
) -> List[Tuple[str, str]]:
    """
    프레임 직접 방식(단일 턴):
      - list[np.ndarray(BGR)] → 균등 샘플링 → 타일 전처리 → 단일 프롬프트 질의
      - 반환은 [(prompt_used, answer)] 1개만
      - frames_bgr 또는 frames 키워드로 입력 가능(호환)
    """
    # 호환성: frames 키워드도 허용
    if frames_bgr is None:
        frames_bgr = kwargs.pop("frames", None)

    # 프레임 없음 → 기본 프롬프트만 반환(응답은 빈 문자열)
    prompt_used = (prompt if prompt else _build_bbox_prompt(lang=lang, hint=hint))
    if not isinstance(frames_bgr, list) or len(frames_bgr) == 0:
        return [(prompt_used, "")]

    # 1) 프레임 균등 샘플링
    idxs = _uniform_indices_from_length(len(frames_bgr), num_segments)
    sampled = [frames_bgr[i] for i in idxs]

    # 2) 전처리(타일링) -> pixel_values + 각 프레임별 타일 수
    pv, patch_list = _frames_to_pixel_values(sampled, input_size=input_size, max_num=max_num)
    pv = _to_device_dtype(pv, model)

    # 3) 프롬프트 + 질의
    prefix = prepare_video_prompts(patch_list)
    final_prompt = prefix + prompt_used

    answer = _call_chat_single_turn(
        model, tokenizer, pv, final_prompt, generation_config,
        num_patches_list=patch_list
    )
    return [(final_prompt, answer)]
# ===========================================================================
