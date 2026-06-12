"""Maintenance & health tracking — persists across restarts in a small JSON file.

Tracks cumulative print hours and per-print failure history so it can:
  • remind you to do upkeep (clean nozzle, lube rails, check belts) every
    MAINTENANCE_REMINDER_HOURS of printing
  • flag a likely developing problem — e.g. stopped-extrusion recurring across
    recent prints suggests a partial clog

record_print() returns a list of alert strings the caller should push out.
All file I/O is best-effort and never raises.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

# Default store lives at the project root, like .env, so it's CWD-independent.
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "maintenance.json"

_RECENT_WINDOW = 5          # how many recent prints to keep for trend detection
_CLOG_LOOKBACK = 3          # look at the last N prints…
_CLOG_THRESHOLD = 2         # …and warn if stopped-extrusion appears in ≥ this many


class MaintenanceTracker:
    def __init__(self, reminder_hours: int = 250, path: Optional[Path] = None) -> None:
        self.reminder_hours = max(1, reminder_hours)
        self.path = Path(path) if path else _DEFAULT_PATH
        self.data = {
            "total_print_seconds": 0,
            "prints_completed": 0,
            "recent_failures": [],          # list[list[str]] — failure types per recent print
            "last_reminder_hours": 0,       # highest reminder multiple already fired
        }
        self._load()

    # ── persistence ──
    def _load(self) -> None:
        try:
            if self.path.exists():
                self.data.update(json.loads(self.path.read_text()))
        except Exception as exc:
            print(f"  [MAINT] load error (ignored): {exc}")

    def _save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data, indent=2))
        except Exception as exc:
            print(f"  [MAINT] save error (ignored): {exc}")

    # ── read-side helpers ──
    @property
    def total_hours(self) -> float:
        return self.data["total_print_seconds"] / 3600.0

    def summary(self) -> str:
        d = self.data
        until_next = self.reminder_hours - (self.total_hours % self.reminder_hours)
        return (
            f"*Maintenance*\n"
            f"Total print time: {self.total_hours:.1f} h\n"
            f"Prints completed: {d['prints_completed']}\n"
            f"Next upkeep reminder in ~{until_next:.0f} h"
        )

    # ── update on each completed print ──
    def record_print(self, elapsed_seconds: Optional[int],
                     failure_types: Iterable[str]) -> List[str]:
        alerts: List[str] = []
        added = int(elapsed_seconds or 0)
        self.data["total_print_seconds"] += max(0, added)
        self.data["prints_completed"] += 1

        # Rolling window of distinct failure types seen during each print.
        types = sorted({t for t in failure_types if t and t not in ("none", "no_printer")})
        recent = self.data.get("recent_failures", [])
        recent.append(types)
        self.data["recent_failures"] = recent[-_RECENT_WINDOW:]

        # Upkeep reminder when we cross a new multiple of reminder_hours.
        milestone = int(self.total_hours // self.reminder_hours) * self.reminder_hours
        if milestone >= self.reminder_hours and milestone > self.data.get("last_reminder_hours", 0):
            self.data["last_reminder_hours"] = milestone
            alerts.append(
                f"🛠 {milestone} h of printing reached — time for upkeep: "
                f"clean the nozzle, wipe/lube the rails, and check belt tension."
            )

        # Developing-clog trend: stopped-extrusion in several recent prints.
        lookback = self.data["recent_failures"][-_CLOG_LOOKBACK:]
        clog_hits = sum(1 for t in lookback if "stopped extrusion" in t)
        if clog_hits >= _CLOG_THRESHOLD:
            alerts.append(
                f"⚠️ Stopped-extrusion flagged in {clog_hits} of the last "
                f"{len(lookback)} prints — possible partial clog. Consider a cold pull "
                f"or nozzle check."
            )

        self._save()
        return alerts
