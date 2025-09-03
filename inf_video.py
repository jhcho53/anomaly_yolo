# yolo_pm_rider_helmet_video.py
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import re

# ======================
# 설정
# ======================
WEIGHTS = "/home/jaehyeon/Desktop/neubility/Dataset/runs/train0/final_s/weights/best.pt"  # 4cls 또는 7cls pt 경로
SOURCE  = "/media/jaehyeon/T311/Neubie/dataset/Seongbuk"  # 폴더 또는 단일 동영상 파일
OUTPUT_DIR = Path("runs/pm_rider_helmet_video")
CONF_THRES = 0.25
IOU_THRES  = 0.5
SHOW_PREVIEW = False  # 단일 파일 처리 시 미리보기 창 표시 여부

# 헬멧 판정 파라미터 (한 번 추론 기준)
HEAD_REGION_RATIO = 0.6   # 상단 60%에서만 헬멧 인정(오검 감소)
ROI_TOP_EXTRA = 0.2       # 머리 공간 확보 위해 위쪽 여유
ROI_SIDE_EXTRA = 0.05     # 좌우 여유

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

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

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
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
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
    pw = x2 - x1
    ph = y2 - y1
    nx1 = max(0, x1 - pw * side_extra)
    nx2 = min(w - 1, x2 + pw * side_extra)
    ny1 = max(0, y1 - ph * top_extra)   # 위로 살짝 확장
    ny2 = min(h - 1, y2)
    return np.array([nx1, ny1, nx2, ny2], dtype=np.float32)

def _helmet_center_in_head_region(helmet_box, roi, head_ratio):
    # 헬멧 중심이 ROI 내부 & 상단 head_ratio 내에 있는지 확인
    hx1, hy1, hx2, hy2 = helmet_box
    cx = (hx1 + hx2) / 2.0
    cy = (hy1 + hy2) / 2.0
    rx1, ry1, rx2, ry2 = roi
    if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
        return False
    head_y_limit = ry1 + (ry2 - ry1) * head_ratio
    return cy <= head_y_limit

# ---------- 클래스 매핑(4cls/7cls 자동 지원) ----------
def _normalize(s: str):
    s = s.lower().strip()
    s = re.sub(r"[_\-\s]+", " ", s)      # _, - 를 공백으로
    s = re.sub(r"[^a-z0-9 ]", "", s)     # 영숫자/공백만
    return " ".join(s.split())

def _find_one(norm_map, candidates):
    cands = [_normalize(c) for c in candidates]
    # 1) 정확 일치
    for i, n in norm_map.items():
        if n in cands: return i
    # 2) 부분 포함(양방향)
    for i, n in norm_map.items():
        if any(c in n or n in c for c in cands): return i
    return None

def resolve_class_ids(model):
    raw = model.names
    id_to_name = {int(k): v for k, v in (raw.items() if isinstance(raw, dict) else enumerate(raw))}
    norm_map = {i: _normalize(n) for i, n in id_to_name.items()}

    # 동의어 집합
    CANDS = {
        "person":    ["person"],
        "pm":        ["pm", "personal mobility", "kickboard", "e scooter", "escooter", "electric scooter", "micromobility", "micro mobility", "e_scooter"],
        "trash_bag": ["trash bag", "garbage bag", "plastic bag", "trash", "garbage"],
        "helmet":    ["helmet", "hardhat", "safety helmet"],
        "fire":      ["fire"],
        "smoke":     ["smoke"],
        "weapon":    ["weapon"],  # 필요시 "gun","knife" 등 데이터셋에 맞게 추가
    }

    ids = {}
    for key, cands in CANDS.items():
        idx = _find_one(norm_map, cands)
        if idx is not None:
            ids[key] = idx

    # 필수 클래스 확인(라이더/헬멧 로직에 필요)
    required = ["person", "pm", "helmet"]
    missing = [k for k in required if k not in ids]
    if missing:
        raise ValueError(f"[ERROR] 필수 클래스 누락: {missing} | model.names={id_to_name}")
    return ids  # trash_bag, fire, smoke, weapon은 있으면 사용

def process_frame(model, frame, ids):
    """
    한 번의 추론으로 사용 가능한 클래스 모두 검출한 뒤,
    - PM x2와 person IoU>0.01 -> Rider
    - Rider에 한해 헬멧 판정(상단 head_ratio 영역 내 헬멧 중심)
    - Rider는 항상 시각화, 사람은 인원 임계치 이상일 때만 시각화
    - trash_bag / fire / smoke / weapon은 있으면 박스 + 라벨
    """
    h, w = frame.shape[:2]

    # 요청할 클래스 인덱스 (모델이 가진 것만)
    keep_ids = sorted(set(ids.values()))
    res = model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, classes=keep_ids, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return frame

    xyxy = res.boxes.xyxy.cpu().numpy()
    cls  = res.boxes.cls.cpu().numpy().astype(int)
    # conf = res.boxes.conf.cpu().numpy()  # 필요 시 사용

    # 클래스별 분리 (존재 여부 체크)
    get = lambda key: ids.get(key, None)

    pm_boxes      = [xyxy[i] for i, c in enumerate(cls) if c == get("pm")]
    person_boxes  = [xyxy[i] for i, c in enumerate(cls) if c == get("person")]
    helmet_boxes  = [xyxy[i] for i, c in enumerate(cls) if c == get("helmet")]
    trash_boxes   = [xyxy[i] for i, c in enumerate(cls) if c == get("trash_bag")] if get("trash_bag") is not None else []
    fire_boxes    = [xyxy[i] for i, c in enumerate(cls) if c == get("fire")]       if get("fire") is not None else []
    smoke_boxes   = [xyxy[i] for i, c in enumerate(cls) if c == get("smoke")]      if get("smoke") is not None else []
    weapon_boxes  = [xyxy[i] for i, c in enumerate(cls) if c == get("weapon")]     if get("weapon") is not None else []

    # -------- Rider 판정: PM x2와 Person IoU ----------
    rider_flags = [False] * len(person_boxes)
    for pm in pm_boxes:
        pm2 = expand_box(pm, factor=2.0, img_w=w, img_h=h)
        for i, pb in enumerate(person_boxes):
            if iou_xyxy(pm2, pb) > 0.01:
                rider_flags[i] = True

    # -------- PM 시각화: 항상 ----------
    for pm in pm_boxes:
        x1, y1, x2, y2 = pm.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PM, 2)
        draw_label(frame, x1, y1, "PM", COLOR_PM)

    total_persons = len(person_boxes)

    # -------- Rider에 한해 헬멧 판정 ----------
    has_helmet_full = [False] * len(person_boxes)
    if len(helmet_boxes) > 0:
        for i, (pb, is_rider) in enumerate(zip(person_boxes, rider_flags)):
            if not is_rider:
                continue  # Rider가 아니면 헬멧 판단/표시를 하지 않음
            roi = _expand_person_roi(pb, w, h)  # 머리 공간을 약간 더 포함
            for hb in helmet_boxes:
                if _helmet_center_in_head_region(hb, roi, HEAD_REGION_RATIO):
                    has_helmet_full[i] = True
                    break

    # -------- 시각화 ----------
    # (1) Rider: 항상 표시 (Helmet/NoHelmet 라벨)
    if VIS_SHOW_RIDERS_ALWAYS:
        for idx, (pb, is_rider) in enumerate(zip(person_boxes, rider_flags)):
            if not is_rider:
                continue
            x1, y1, x2, y2 = pb.astype(int)
            color = COLOR_RIDER
            label = "Rider | " + ("Helmet" if has_helmet_full[idx] else "NoHelmet")
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, x1, y1, label, color)

    # (2) 일반 Person: 인원수 임계치 이상일 때만 표시 (Rider는 중복표시 방지)
    if total_persons >= PERSON_DRAW_MIN_COUNT:
        for (pb, is_rider) in zip(person_boxes, rider_flags):
            if is_rider and VIS_SHOW_RIDERS_ALWAYS:
                continue  # Rider는 이미 그림
            x1, y1, x2, y2 = pb.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PERSON, 2)
            draw_label(frame, x1, y1, "Person", COLOR_PERSON)

        # 우측 상단 카운트 텍스트
        text = f"Persons: {total_persons}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
        x0 = w - tw - 16
        y0 = 16 + th + 8
        cv2.rectangle(frame, (x0 - 8, 8), (x0 + tw + 8, y0), (50, 50, 50), -1)
        cv2.putText(frame, text, (x0, 16 + th), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # (3) 기타 클래스: 있으면 표시
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

    return frame

def process_video_file(model, video_path, ids, out_path, show=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 열 수 없음: {video_path}")
        return
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
        print(f"[WARN] 손상 또는 빈 비디오: {video_path}")
        cap.release()
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = float(fps)
    except:
        fps = 0.0
    if not fps or fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = process_frame(model, frame, ids)
        writer.write(vis)
        if show:
            cv2.imshow("PM/Rider/Helmet (video)", vis)
            if cv2.waitKey(1) == 27:
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"[Saved] {out_path}")

def _iter_videos(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and not p.name.startswith("._"):
            try:
                if p.stat().st_size > 0:
                    yield p
            except Exception:
                continue

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHTS)
    ids = resolve_class_ids(model)

    src_path = Path(SOURCE)
    if not src_path.exists():
        raise FileNotFoundError(f"SOURCE 경로가 존재하지 않습니다: {src_path}")

    # 폴더 모드: 하위 모든 동영상 처리(구조 보존)
    if src_path.is_dir():
        vids = list(_iter_videos(src_path))
        if not vids:
            print(f"[INFO] 폴더 내 동영상 없음: {src_path}")
            return
        for f in vids:
            rel_parent = f.parent.relative_to(src_path)  # 폴더 구조 보존
            out_file = OUTPUT_DIR / rel_parent / f"{f.stem}_out.mp4"
            process_video_file(model, f, ids, out_file, show=False)
        return

    # 단일 파일 모드(동영상만)
    if src_path.is_file() and src_path.suffix.lower() in VIDEO_EXTS:
        out_path = OUTPUT_DIR / f"{src_path.stem}_out.mp4"
        process_video_file(model, src_path, ids, out_path, show=SHOW_PREVIEW)
    else:
        raise ValueError(f"지원하지 않는 동영상 확장자 또는 파일이 아닙니다: {src_path}")

if __name__ == "__main__":
    run()
