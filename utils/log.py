# utils/log.py
import csv
from pathlib import Path
from datetime import datetime
from typing import Mapping, Sequence, Union

__all__ = ["_append_csv", "_video_stamp_meta", "log_no_helmet", "log_hazard"]

PathLike = Union[str, Path]


# ---------- Common utilities ----------
def _append_csv(file_path: Path, header: Sequence, row: Sequence) -> None:
    """
    Append a single row to a CSV file. If the file does not exist yet, write the header first.

    Args:
        file_path: Destination CSV file path (must be a Path; callers using str should cast beforehand).
        header: Header row to write when the file is first created.
        row: Data row to append.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existed = file_path.exists()
    # Use UTF-8 to avoid locale-dependent encoding issues.
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not existed and header:
            w.writerow(header)
        w.writerow(row)


def _video_stamp_meta(base_unix: float, frame_idx: int, fps: float) -> Mapping[str, Union[int, float, str]]:
    """
    Build a timestamp metadata dict for a video log entry, given:
      - base_unix: wall-clock UNIX time (seconds) at the start of the video,
      - frame_idx: current frame index (0-based),
      - fps: frames per second.

    Returns:
        A mapping with:
          - "sec":   integer UNIX seconds
          - "nsec":  nanoseconds remainder
          - "iso":   ISO-8601 local timestamp string
          - "t":     video-relative time in seconds (frame_idx / fps; falls back to 30.0 FPS if fps is tiny)
    """
    eff_fps = fps if fps > 1e-6 else 30.0
    t = frame_idx / eff_fps
    ts_unix = base_unix + t
    sec = int(ts_unix)
    nsec = int((ts_unix - sec) * 1e9)
    iso = datetime.fromtimestamp(ts_unix).isoformat()
    return {"sec": sec, "nsec": nsec, "iso": iso, "t": t}


# ---------- Domain-specific logs ----------
def log_no_helmet(
    log_dir: PathLike,
    stamp_meta: Mapping[str, Union[int, float, str]],
    track_id: int,
    bbox: Sequence[Union[int, float]],  # [x1, y1, x2, y2]
    persons: int,
    vote_window: int,
    valid_votes: int,
    threshold: float,
    helmet_ratio: float,
) -> None:
    """
    Log a Helmet→NoHelmet transition event for a rider into CSV.

    Creates/updates: <log_dir>/no_helmet.csv
    """
    log_dir = Path(log_dir)
    fp = log_dir / "no_helmet.csv"
    header = [
        "stamp_sec", "stamp_nsec", "iso",
        "track_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "persons", "vote_window", "valid_votes", "vote_threshold", "helmet_ratio",
    ]
    x1, y1, x2, y2 = map(int, bbox[:4])
    row = [
        int(stamp_meta["sec"]), int(stamp_meta["nsec"]), str(stamp_meta["iso"]),
        int(track_id), x1, y1, x2, y2,
        int(persons), int(vote_window), int(valid_votes), float(threshold), round(float(helmet_ratio), 3),
    ]
    _append_csv(fp, header, row)


def log_hazard(
    log_dir: PathLike,
    stamp_meta: Mapping[str, Union[int, float, str]],
    event_type: str,  # e.g., "trash_bag" | "fire" | "smoke" | "weapon" | "crowd"
    count: int,
    persons: int,
) -> None:
    """
    Log a hazard (or crowd) event while in a crowding state into CSV.

    Creates/updates: <log_dir>/hazard_<event_type>.csv
    """
    log_dir = Path(log_dir)
    fp = log_dir / f"hazard_{event_type}.csv"
    header = ["stamp_sec", "stamp_nsec", "iso", "event", "count", "persons"]
    row = [
        int(stamp_meta["sec"]), int(stamp_meta["nsec"]), str(stamp_meta["iso"]),
        str(event_type), int(count), int(persons),
    ]
    _append_csv(fp, header, row)
