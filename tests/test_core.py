"""Unit tests for the parsing/decision logic that has bitten us before.

Run:  python3 -m pytest tests/ -q

Covers the pure-logic layer only — no camera, no printer serial, no network.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ender3monitor.analyzer import (
    _downscale, _normalize_type, _parse_response, _precheck_frame,
)
from ender3monitor.config import _parse_flip
from ender3monitor.maintenance import MaintenanceTracker
from ender3monitor.printer import (
    _M78_TOTAL_RE, _POS_RE, _PRINTTIME_RE, _SD_RE, _TEMP_RE, _fmt_duration,
)
from monitor import _frames_differ


# ── analyzer: failure-type normalization ────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("none", "none"),
    ("spaghetti", "spaghetti/stringing"),
    ("detached", "detached from bed"),
    ("layer_shift", "layer shift"),
    ("stopped_extrusion", "stopped extrusion"),
    ("warping", "warping"),
    ("no_printer", "no_printer"),
    ("Spaghetti", "spaghetti/stringing"),                 # case-insensitive
    ("STOPPED_EXTRUSION", "stopped extrusion"),
    ("detached from bed", "detached from bed"),           # display-name style
    # The exact garbage the model produced in production — must collapse to none
    ("spaghetti/stringing | detached from bed", "none"),
    ("spaghetti/stringing | layer shift | detached from bed | stopped extrusion"
     " | warping | none | no_printer", "none"),
    ("garbage_value", "none"),
    ("", "none"),
])
def test_normalize_type(raw, expected):
    assert _normalize_type(raw) == expected


# ── analyzer: response parsing ───────────────────────────────────────────────

def _resp(**kw):
    return _parse_response(json.dumps(kw), "test")


def test_parse_clean_failure_above_floor():
    r = _resp(failure_type="detached", confidence=0.9, description="dragged")
    assert r.failure_detected and r.failure_type == "detached from bed"


def test_parse_failure_below_floor_is_none():
    r = _resp(failure_type="spaghetti", confidence=0.6, description="maybe")
    assert not r.failure_detected and r.failure_type == "none"


def test_parse_compound_junk_never_fails():
    r = _resp(failure_type="spaghetti/stringing | detached from bed",
              confidence=0.99, description="x")
    assert not r.failure_detected and r.failure_type == "none"


def test_parse_no_printer_sentinel_preserved():
    r = _resp(failure_type="no_printer", confidence=0.0, description="empty")
    assert not r.failure_detected and r.failure_type == "no_printer"


def test_parse_garbage_text():
    r = _parse_response("I am not JSON at all", "test")
    assert not r.failure_detected and r.failure_type == "none"


def test_parse_falls_back_to_observations():
    r = _resp(failure_type="none", confidence=0.1, observations="all good")
    assert r.description == "all good"


# ── analyzer: pre-checks and downscale ───────────────────────────────────────

def test_precheck_rejects_black_frame():
    black = np.zeros((120, 160, 3), dtype=np.uint8)
    early = _precheck_frame(black, "test")
    assert early is not None and early.failure_type == "no_printer"


def test_precheck_passes_textured_frame():
    rng = np.random.default_rng(42)
    noisy = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    assert _precheck_frame(noisy, "test") is None


def test_downscale_geometry():
    big = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert _downscale(big).shape == (360, 640, 3)
    small = np.zeros((100, 320, 3), dtype=np.uint8)
    assert _downscale(small) is small


# ── printer: G-code reply parsing ────────────────────────────────────────────

def test_m105_with_v3se_fan_field():
    m = _TEMP_RE.search("ok T:22.62 /0.00 B:22.52 /0.00 @:0 B@:0 FAN0@:0")
    assert m and float(m.group(1)) == 22.62 and float(m.group(3)) == 22.52


def test_m27_progress_and_idle():
    m = _SD_RE.search("SD printing byte 45324/3105540")
    assert m and int(m.group(1)) == 45324 and int(m.group(2)) == 3105540
    assert _SD_RE.search("Not SD printing") is None


def test_m31_duration_forms():
    cases = {"echo:Print time: 1h 23m 45s": 5025,
             "echo:Print time: 2m 19s": 139,
             "echo:Print time: 45s": 45}
    for text, secs in cases.items():
        m = _PRINTTIME_RE.search(text)
        h, mn, s = (int(g or 0) for g in m.groups())
        assert h * 3600 + mn * 60 + s == secs, text


def test_m114_takes_position_z_not_step_count():
    m = _POS_RE.search("X:110.5 Y:90.2 Z:5.40 E:123.4 Count X:8840 Y:7216 Z:2160")
    assert m and float(m.group(1)) == 5.40


def test_m78_total_time_forms():
    cases = {"Total time: 8d 12h 34m 56s": 736496,
             "Total time: 192h 5m": 691500,
             "Total time: 45m 10s": 2710}
    for text, secs in cases.items():
        m = _M78_TOTAL_RE.search(text)
        d, h, mn, s = (int(g or 0) for g in m.groups())
        assert d * 86400 + h * 3600 + mn * 60 + s == secs, text


def test_fmt_duration():
    assert _fmt_duration(None) is None
    assert _fmt_duration(59) == "0m"
    assert _fmt_duration(3660) == "1h 1m"


# ── monitor: frame motion diff ───────────────────────────────────────────────

def test_frames_differ_handles_resolution_change():
    a = np.zeros((720, 1280, 3), dtype=np.uint8)
    b = np.zeros((480, 640, 3), dtype=np.uint8)     # camera re-enumerated smaller
    assert _frames_differ(a, b) is False             # must not raise (cv2 -209)


def test_frames_differ_detects_motion():
    a = np.zeros((100, 100, 3), dtype=np.uint8)
    b = np.full((100, 100, 3), 200, dtype=np.uint8)
    assert _frames_differ(a, b) is True
    assert _frames_differ(a, a.copy()) is False


# ── config: camera flip parsing ──────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("none", None), ("off", None), ("", None),
    ("180", -1), ("rotate180", -1),
    ("vertical", 0), ("v", 0),
    ("horizontal", 1), ("h", 1),
    ("nonsense", None),
])
def test_parse_flip(raw, expected):
    assert _parse_flip(raw) == expected


# ── maintenance: persistence + alerts ────────────────────────────────────────

def test_maintenance_roundtrip_and_alerts(tmp_path):
    store = tmp_path / "maint.json"
    t = MaintenanceTracker(reminder_hours=10, path=store)

    # 9 h: no reminder yet
    assert t.record_print(9 * 3600, set()) == []
    # +2 h crosses the 10 h milestone → reminder fires exactly once
    alerts = t.record_print(2 * 3600, set())
    assert any("10 h" in a for a in alerts)
    assert t.record_print(60, set()) == []          # no re-fire

    # Clog trend: stopped-extrusion in 2 of the last 3 prints
    t.record_print(60, {"stopped extrusion"})
    alerts = t.record_print(60, {"stopped extrusion"})
    assert any("partial clog" in a for a in alerts)

    # Persistence across instances
    t2 = MaintenanceTracker(reminder_hours=10, path=store)
    assert t2.data["prints_completed"] == 5
    assert t2.total_hours == pytest.approx(t.total_hours)
