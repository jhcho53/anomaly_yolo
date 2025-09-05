# utils/log.py
import csv
from pathlib import Path
from datetime import datetime
from typing import Mapping, Sequence, Union

__all__ = ["_append_csv", "_video_stamp_meta", "log_no_helmet", "log_hazard"]

PathLike = Union[str, Path]


# ---------- 공통 유틸 ----------
def _append_csv(file_path: Path, header: Sequence, row: Sequence) -> None:
    """
    CSV에 한 줄을 append. 파일이 없으면 header를 먼저 씁니다.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existed = file_path.exists()
    with open(file_path, "a", newline="") as f:
        w = csv.writer(f)
        if not existed and header:
            w.writerow(header)
        w.writerow(row)


def _video_stamp_meta(base_unix: float, frame_idx: int, fps: float) -> Mapping[str, Union[int, float, str]]:
    """
    비디오 기준시각(base_unix)과 프레임 인덱스, FPS로 로그 타임스탬프 메타를 생성.
    반환: {"sec", "nsec", "iso", "t"}  (t는 비디오 상대초)
    """
    t = (frame_idx / (fps if fps > 1e-6 else 30.0))
    ts_unix = base_unix + t
    sec = int(ts_unix)
    nsec = int((ts_unix - sec) * 1e9)
    iso = datetime.fromtimestamp(ts_unix).isoformat()
    return {"sec": sec, "nsec": nsec, "iso": iso, "t": t}


# ---------- 도메인 로그 ----------
def log_no_helmet(
    log_dir: PathLike,
    stamp_meta: Mapping[str, Union[int, float, str]],
    track_id: int,
    bbox: Sequence[Union[int, float]],  # [x1,y1,x2,y2]
    persons: int,
    vote_window: int,
    valid_votes: int,
    threshold: float,
    helmet_ratio: float,
) -> None:
    """
    무헬멧(헬멧→무헬멧 전이) 이벤트 로그를 CSV로 기록.
    """
    log_dir = Path(log_dir)
    fp = log_dir / "no_helmet.csv"
    header = [
        "stamp_sec", "stamp_nsec", "iso",
        "track_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "persons", "vote_window", "valid_votes", "vote_threshold", "helmet_ratio"
    ]
    x1, y1, x2, y2 = map(int, bbox[:4])
    row = [
        int(stamp_meta["sec"]), int(stamp_meta["nsec"]), str(stamp_meta["iso"]),
        int(track_id), x1, y1, x2, y2,
        int(persons), int(vote_window), int(valid_votes), float(threshold), round(float(helmet_ratio), 3)
    ]
    _append_csv(fp, header, row)


def log_hazard(
    log_dir: PathLike,
    stamp_meta: Mapping[str, Union[int, float, str]],
    event_type: str,     # "trash_bag" | "fire" | "smoke" | "weapon"
    count: int,
    persons: int,
) -> None:
    """
    군집(사람 수 임계 이상) 상태에서의 Hazard 감지 로그를 CSV로 기록.
    """
    log_dir = Path(log_dir)
    fp = log_dir / f"hazard_{event_type}.csv"
    header = ["stamp_sec", "stamp_nsec", "iso", "event", "count", "persons"]
    row = [
        int(stamp_meta["sec"]), int(stamp_meta["nsec"]), str(stamp_meta["iso"]),
        str(event_type), int(count), int(persons)
    ]
    _append_csv(fp, header, row)
