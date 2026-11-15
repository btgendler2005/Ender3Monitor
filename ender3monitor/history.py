"""Print history — one small record per monitoring session, newest first.

Everything lives in a single directory (default: `history/` at the project
root, next to maintenance.json, so it is CWD-independent):

    history/history.json     the records (capped at MAX_RECORDS)
    history/<id>.jpg         a small thumbnail of each print's final frame

Record schema (all JSON-native types):

    id                str    "YYYYmmdd_HHMMSS_xxxxxx"
    started_at        float  epoch seconds
    ended_at          float  epoch seconds
    duration_seconds  int
    job_name          str | None   sliced file name, when the G-code index knew it
    total_layers      int | None
    outcome           "completed" | "stopped" | "failed"
    failure_count     int    confirmed failures
    failure_types     [str]  distinct confirmed failure types
    false_alarms      int
    frames_analyzed   int
    ai_cost_usd       float
    thumbnail         str | None   bare filename inside the history directory
    timelapse         str | None   bare filename of the compiled MP4, if any
    events            [dict]       that print's event log, trimmed to MAX_EVENTS

All file I/O is best-effort: a missing or corrupt file reads as "no history",
and no method here raises on a disk error.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Default store lives at the project root, like maintenance.json.
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "history"

MAX_RECORDS = 200       # oldest records (and their thumbnails) are dropped past this
MAX_EVENTS = 100        # per-record event-log cap
THUMB_WIDTH = 480       # px; final photo is downscaled to this width
THUMB_QUALITY = 80

OUTCOMES = ("completed", "stopped", "failed")


class HistoryStore:
    def __init__(self, directory: Optional[Path] = None,
                 max_records: int = MAX_RECORDS) -> None:
        self.dir = Path(directory) if directory else _DEFAULT_DIR
        self.path = self.dir / "history.json"
        self.max_records = max(1, int(max_records))
        self._lock = threading.RLock()
        self._records: List[dict] = []
        self._load()

    # ── persistence ──
    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if isinstance(data, list):
                    self._records = [r for r in data
                                     if isinstance(r, dict) and r.get("id")]
        except Exception as exc:
            print(f"  [HISTORY] load error (ignored): {exc}")
            self._records = []

    def _save(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._records, indent=1))
            os.replace(tmp, self.path)          # atomic — never a half-written file
        except Exception as exc:
            print(f"  [HISTORY] save error (ignored): {exc}")

    def _unlink_thumb(self, record: dict) -> None:
        try:
            p = self.thumb_path_for(record)
            if p is not None:
                p.unlink(missing_ok=True)
        except Exception:
            pass

    # ── read side ──
    def list(self, limit: Optional[int] = None, include_events: bool = False) -> List[dict]:
        """Records newest first. Event logs are omitted unless asked for."""
        with self._lock:
            recs = self._records[:limit] if limit else list(self._records)
            if include_events:
                return [dict(r) for r in recs]
            return [{k: v for k, v in r.items() if k != "events"} for r in recs]

    def get(self, record_id: str) -> Optional[dict]:
        with self._lock:
            for r in self._records:
                if r.get("id") == record_id:
                    return dict(r)
        return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def thumb_path_for(self, record: Optional[dict]) -> Optional[Path]:
        """Where a record's thumbnail would live (not checked for existence).

        Reduced to a bare filename so a tampered history.json can't point
        outside the history directory.
        """
        name = (record or {}).get("thumbnail")
        if not name or not isinstance(name, str):
            return None
        return self.dir / Path(name).name

    # ── write side ──
    def new_id(self, when: Optional[float] = None) -> str:
        ts = datetime.fromtimestamp(when or time.time()).strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{uuid.uuid4().hex[:6]}"

    def save_thumbnail(self, record_id: str, frame) -> Optional[str]:
        """Write `frame` (BGR ndarray) as a small JPEG; returns its filename."""
        if frame is None:
            return None
        try:
            import cv2
            h, w = frame.shape[:2]
            if w > THUMB_WIDTH:
                frame = cv2.resize(frame, (THUMB_WIDTH, max(1, round(h * THUMB_WIDTH / w))),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
            if not ok:
                return None
            self.dir.mkdir(parents=True, exist_ok=True)
            name = f"{record_id}.jpg"
            (self.dir / name).write_bytes(buf.tobytes())
            return name
        except Exception as exc:
            print(f"  [HISTORY] thumbnail error (ignored): {exc}")
            return None

    def add(self, record: dict) -> dict:
        """Insert a record at the front, enforce the cap, persist."""
        rec = dict(record)
        rec.setdefault("id", self.new_id(rec.get("ended_at")))
        rec["events"] = list(rec.get("events") or [])[-MAX_EVENTS:]
        with self._lock:
            self._records.insert(0, rec)
            for dropped in self._records[self.max_records:]:
                self._unlink_thumb(dropped)
            del self._records[self.max_records:]
            self._save()
        return rec

    def update(self, record_id: str, **fields) -> bool:
        """Patch fields on an existing record (e.g. a timelapse compiled later)."""
        with self._lock:
            for r in self._records:
                if r.get("id") == record_id:
                    r.update(fields)
                    self._save()
                    return True
        return False

    def delete(self, record_id: str) -> bool:
        with self._lock:
            for i, r in enumerate(self._records):
                if r.get("id") == record_id:
                    self._unlink_thumb(r)
                    del self._records[i]
                    self._save()
                    return True
        return False


def _session_events(monitor, started_at: float) -> List[dict]:
    """This session's slice of the monitor's rolling event log."""
    out = []
    for ev in list(getattr(monitor, "events", None) or []):
        if not isinstance(ev, dict):
            continue
        t = ev.get("t")
        if isinstance(t, (int, float)) and t < started_at:
            continue                    # left over from an earlier session
        out.append(dict(ev))
    return out[-MAX_EVENTS:]


def build_record(monitor, outcome: str, *, now: Optional[float] = None) -> Optional[dict]:
    """Snapshot a finished monitoring session off a Monitor-like object.

    Returns None for sessions that never analysed a frame (nothing to record).
    Reads everything defensively with getattr so it works on partial fakes and
    on attributes other code adds later (e.g. false_alarm_count).
    """
    frames = int(getattr(monitor, "frame_count", 0) or 0)
    if frames <= 0:
        return None
    if outcome not in OUTCOMES:
        outcome = "stopped"
    ended = float(now if now is not None else time.time())

    status = getattr(getattr(monitor, "printer", None), "status", None)
    # Prefer the printer's own job timer, then time since the print was first
    # seen active, then the wall-clock length of the monitoring session.
    session_start = getattr(monitor, "_session_started_at", None)
    elapsed = getattr(status, "elapsed_seconds", None)
    if elapsed is None and getattr(monitor, "_print_active_since", None):
        elapsed = ended - monitor._print_active_since
    if elapsed is None and session_start:
        elapsed = ended - session_start
    duration = max(0, int(elapsed or 0))
    started = float(session_start) if session_start else ended - duration

    # The G-code context is cleared the moment the printer stops printing, so
    # prefer the name the monitor remembered while the job was still running.
    job_name, total_layers = getattr(monitor, "_history_job", None) or (None, None)
    job_name = job_name or getattr(status, "job_name", None)
    total_layers = total_layers or getattr(status, "total_layers", None)

    types = sorted(str(t) for t in (getattr(monitor, "_run_failure_types", None) or ())
                   if t and t not in ("none", "no_printer"))

    return {
        "started_at": round(started, 3),
        "ended_at": round(ended, 3),
        "duration_seconds": duration,
        "job_name": str(job_name) if job_name else None,
        "total_layers": int(total_layers) if total_layers else None,
        "outcome": outcome,
        "failure_count": int(getattr(monitor, "failure_count", 0) or 0),
        "failure_types": types,
        "false_alarms": int(getattr(monitor, "false_alarm_count", 0) or 0),
        "frames_analyzed": frames,
        "ai_cost_usd": round(float(getattr(monitor, "session_cost_usd", 0.0) or 0.0), 4),
        "thumbnail": None,
        "timelapse": getattr(monitor, "_history_timelapse", None) or None,
        "events": _session_events(monitor, started),
    }


def record_session(store: HistoryStore, monitor, outcome: str, frame=None,
                   *, now: Optional[float] = None) -> Optional[dict]:
    """Build a record for the session that just ended and persist it.

    Returns the stored record, or None when the session is skipped.
    """
    rec = build_record(monitor, outcome, now=now)
    if rec is None:
        return None
    rec["id"] = store.new_id(rec["ended_at"])
    rec["thumbnail"] = store.save_thumbnail(rec["id"], frame)
    return store.add(rec)


def format_recent(store: HistoryStore, limit: int = 5) -> str:
    """Plain-text summary of the most recent prints (Telegram /history)."""
    from ender3monitor.printer import _fmt_duration
    recs = store.list(limit=limit)
    if not recs:
        return "No prints recorded yet."
    icons = {"completed": "✅", "stopped": "⏹", "failed": "❌"}
    lines = [f"Last {len(recs)} print(s):"]
    for r in recs:
        when = datetime.fromtimestamp(r.get("ended_at") or 0).strftime("%b %d %H:%M")
        outcome = r.get("outcome") or "stopped"
        lines.append(
            f"{icons.get(outcome, '•')} {when} — {r.get('job_name') or 'Untitled print'}\n"
            f"    {outcome} · {_fmt_duration(r.get('duration_seconds')) or '—'}"
            f" · ${float(r.get('ai_cost_usd') or 0):.2f}"
        )
    return "\n".join(lines)
