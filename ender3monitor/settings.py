"""Runtime-editable settings — the operational tunables, persisted in JSON.

Why this exists (vs editing .env): the flat .env file has no merge story, can't
be written safely by the app, and never reflects runtime toggles. This module
is the single writer of an app-owned `settings.json`, edited only through the
validated API/UI — so concurrent hand-edits and lost toggles go away.

SECURITY MODEL (deliberate):
  • SCHEMA is a fixed ALLOWLIST. Only the keys defined here can ever be read or
    written. Secrets (API keys, SMTP/Telegram credentials, WEB_PASSWORD) are NOT
    in the schema and live only in .env — they can never leak through, or be
    altered by, the settings API.
  • Every write is validated server-side against the key's type and bounds.
    Unknown keys and out-of-range/!choice values are rejected, never stored.
  • Persistence is atomic (temp file + os.replace) and confined to one fixed
    path; nothing about the path is caller-controlled.

Precedence on load: settings.json  →  seed (from .env/Config)  →  schema default.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# App-owned store at the project root, like .env (CWD-independent, gitignored).
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "settings.json"


def _field(type_, default, group, label, *, live=True, help="",
           min=None, max=None, choices=None) -> dict:
    return {"type": type_, "default": default, "group": group, "label": label,
            "live": live, "help": help, "min": min, "max": max, "choices": choices}


# The ALLOWLIST. Anything not here is invisible to the settings API by design.
# No secret ever appears in this schema.
SCHEMA: Dict[str, dict] = {
    # ── Detection ──
    "capture_interval": _field(
        "int", 300, "Detection", "Analysis interval (s)", min=10, max=3600,
        help="Seconds between AI analyses — the main cost dial."),
    "confidence_threshold": _field(
        "float", 0.85, "Detection", "Alert threshold", min=0.0, max=1.0,
        help="Flag a failure only at/above this confidence."),
    "first_layer_interval": _field(
        "int", 60, "Detection", "First-layer interval (s)", min=10, max=3600,
        help="Tighter analysis cadence while on the first layer."),
    "first_layer_max_z": _field(
        "float", 0.6, "Detection", "First-layer max Z (mm)", min=0.0, max=20.0,
        help="Treat Z at/under this as the first layer."),

    # ── Camera ──
    "camera_flip": _field(
        "enum", "none", "Camera", "Image flip", live=True,
        choices=["none", "180", "vertical", "horizontal"],
        help="Rotate/flip the camera image (180 for an upside-down mount)."),

    # ── Automation ──
    "auto_start_on_print": _field(
        "bool", True, "Automation", "Auto-start on print",
        help="Begin monitoring automatically when the printer starts a print."),
    "auto_pause_on_failure": _field(
        "bool", False, "Automation", "Auto-act on failure",
        help="Intervene over USB when a failure is confirmed."),
    "auto_pause_action": _field(
        "enum", "pause", "Automation", "Failure action",
        choices=["pause", "cooldown", "estop"],
        help="pause = M25; cooldown = pause + heaters off; estop = M112."),

    # ── Timelapse ──
    "timelapse_mode": _field(
        "enum", "auto", "Timelapse", "Capture mode",
        choices=["auto", "layer", "time"],
        help="auto = layer-synced when USB Z is available, else time-based."),
    "timelapse_max_sessions": _field(
        "int", 20, "Timelapse", "Keep recent sessions", min=1, max=1000,
        help="Prune to this many recent print folders."),
    "timelapse_retention_days": _field(
        "int", 30, "Timelapse", "Retention (days)", min=0, max=3650,
        help="Also delete folders/MP4s older than this."),
    "timelapse_delete_frames_after_compile": _field(
        "bool", False, "Timelapse", "Delete frames after compile",
        help="Reclaim space by dropping JPEGs once compiled to MP4."),

    # ── Maintenance ──
    "maintenance_reminder_hours": _field(
        "int", 250, "Maintenance", "Upkeep reminder (h)", min=1, max=100000,
        help="Push an upkeep nudge every N hours of printing."),

    # ── Pricing (suggested sell price on the completion report) ──
    "pricing_enabled": _field(
        "bool", True, "Pricing", "Show suggested price",
        help="Add a suggested sell price to the print-complete summary."),
    "currency_symbol": _field(
        "enum", "$", "Pricing", "Currency", choices=["$", "€", "£", "¥"],
        help="Currency symbol for the price."),
    "filament_grams": _field(
        "float", 30.0, "Pricing", "Filament weight (g)", min=0.0, max=100000.0,
        help="Grams of filament per print (from your slicer). Used for material cost."),
    "filament_price_per_kg": _field(
        "float", 20.0, "Pricing", "Filament price (/kg)", min=0.0, max=100000.0,
        help="What you pay per kilogram of filament."),
    "electricity_rate_per_kwh": _field(
        "float", 0.15, "Pricing", "Electricity (/kWh)", min=0.0, max=100.0,
        help="Your power rate per kWh."),
    "printer_watts": _field(
        "int", 120, "Pricing", "Printer power (W)", min=0, max=5000,
        help="Average draw while printing (Ender 3 V3 SE ≈ 120 W)."),
    "machine_rate_per_hour": _field(
        "float", 1.0, "Pricing", "Machine rate (/h)", min=0.0, max=100000.0,
        help="Hourly charge for wear, depreciation, and failed-print buffer."),
    "labor_flat": _field(
        "float", 2.0, "Pricing", "Labor (flat)", min=0.0, max=100000.0,
        help="Flat post-processing / handling fee per print."),
    "markup_multiplier": _field(
        "float", 2.0, "Pricing", "Markup (×)", min=0.0, max=100.0,
        help="Multiply total cost by this for the sell price (2–3× typical)."),
}

_TRUE = {"1", "true", "yes", "on", "t", "y"}
_FALSE = {"0", "false", "no", "off", "f", "n"}


def _coerce(key: str, value: Any) -> Tuple[bool, Any]:
    """Validate+coerce a value for key against the schema. Returns (ok, value_or_error)."""
    spec = SCHEMA.get(key)
    if spec is None:
        return False, f"unknown setting '{key}'"
    t = spec["type"]
    try:
        if t == "bool":
            if isinstance(value, bool):
                return True, value
            s = str(value).strip().lower()
            if s in _TRUE:
                return True, True
            if s in _FALSE:
                return True, False
            return False, f"{key}: expected a boolean"
        if t == "int":
            v = int(value)
        elif t == "float":
            v = float(value)
        elif t == "enum":
            v = str(value)
            if v not in spec["choices"]:
                return False, f"{key}: must be one of {spec['choices']}"
            return True, v
        else:
            return False, f"{key}: unknown type"
    except (TypeError, ValueError):
        return False, f"{key}: expected {t}"

    if spec["min"] is not None and v < spec["min"]:
        return False, f"{key}: must be ≥ {spec['min']}"
    if spec["max"] is not None and v > spec["max"]:
        return False, f"{key}: must be ≤ {spec['max']}"
    return True, v


class Settings:
    def __init__(self, seed: Optional[Dict[str, Any]] = None,
                 path: Optional[Path] = None,
                 on_change: Optional[Callable[[str, Any], None]] = None) -> None:
        self.path = Path(path) if path else _DEFAULT_PATH
        self.on_change = on_change
        self._lock = threading.RLock()
        # Start from schema defaults, overlay the seed (from .env/Config), then
        # overlay the persisted file (highest precedence).
        self._values: Dict[str, Any] = {k: s["default"] for k, s in SCHEMA.items()}
        if seed:
            for k, v in seed.items():
                if k in SCHEMA:
                    ok, cv = _coerce(k, v)
                    if ok:
                        self._values[k] = cv
        existed = self._load()
        if not existed:
            self._save()   # persist the seeded baseline on first run

    # ── persistence ──
    def _load(self) -> bool:
        try:
            if not self.path.exists():
                return False
            data = json.loads(self.path.read_text())
            for k, v in (data or {}).items():
                if k in SCHEMA:                 # ignore stale/unknown keys
                    ok, cv = _coerce(k, v)
                    if ok:
                        self._values[k] = cv
            return True
        except Exception as exc:
            print(f"  [SETTINGS] load error (ignored): {exc}")
            return False

    def _save(self) -> None:
        try:
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._values, indent=2))
            os.replace(tmp, self.path)          # atomic on POSIX
        except Exception as exc:
            print(f"  [SETTINGS] save error (ignored): {exc}")

    # ── read ──
    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key, SCHEMA.get(key, {}).get("default"))

    def public_dict(self) -> Dict[str, Any]:
        """Current values — only schema (non-secret) keys, safe to send to the UI."""
        with self._lock:
            return dict(self._values)

    @staticmethod
    def schema_public() -> List[dict]:
        """Schema metadata for the UI. Never contains secrets or values."""
        return [{"key": k, **{kk: v[kk] for kk in
                              ("type", "group", "label", "live", "help",
                               "min", "max", "choices")}}
                for k, v in SCHEMA.items()]

    # ── write ──
    def update(self, changes: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Validate+apply a batch. Returns (applied_values, errors, restart_keys).

        All-or-nothing per key: invalid keys are rejected (collected in errors)
        and never stored; valid keys are coerced, persisted atomically, and the
        on_change hook is fired so the running app can apply live ones.
        """
        applied: Dict[str, Any] = {}
        errors: List[str] = []
        restart_keys: List[str] = []
        with self._lock:
            for key, raw in changes.items():
                ok, cv = _coerce(key, raw)
                if not ok:
                    errors.append(cv)            # cv is the error message
                    continue
                self._values[key] = cv
                applied[key] = cv
                if not SCHEMA[key]["live"]:
                    restart_keys.append(key)
            if applied:
                self._save()
        # Fire hooks outside the lock so handlers can touch the monitor freely.
        if self.on_change:
            for key, cv in applied.items():
                try:
                    self.on_change(key, cv)
                except Exception as exc:
                    print(f"  [SETTINGS] on_change({key}) error (ignored): {exc}")
        return applied, errors, restart_keys
