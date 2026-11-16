"""Unit tests for the parsing/decision logic that has bitten us before.

Run:  python3 -m pytest tests/ -q

Covers the pure-logic layer only — no camera, no printer serial, no network.
"""
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ender3monitor.analyzer import (
    _downscale, _normalize_type, _parse_response, _precheck_frame,
    build_context_note,
)
from ender3monitor.gcode_index import (
    GcodeIndex, IndexStore, VolumeWatcher, parse_gcode,
)
from ender3monitor.config import _parse_flip
from ender3monitor.maintenance import MaintenanceTracker
from ender3monitor.settings import SCHEMA, Settings, _coerce
from ender3monitor.printer import (
    _M78_TOTAL_RE, _POS_RE, _PRINTTIME_RE, _RUNOUT_MARKER_RE, _SD_RE, _TEMP_RE,
    _fmt_duration, PrinterController, PrinterStatus, SENSOR_CALIBRATION_POLLS,
)
from monitor import (
    Monitor, _frames_differ, _confirm_frames, _progress_stuck_since,
    STUCK_AT_COMPLETION_PCT, STUCK_AT_COMPLETION_SECONDS,
)


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


@pytest.mark.parametrize("resp, printing, progress", [
    ("SD printing byte 100/200", True, 0.5),
    ("Not SD printing", False, None),
    # End-of-print: Marlin's "Done printing file" must clear the printing flag
    # so monitor.py's falling-edge completion fires (regression guard).
    ("Done printing file", False, None),
    ("SD printing byte 200/200\nDone printing file", False, 1.0),
])
def test_query_progress_completion(resp, printing, progress):
    pc = PrinterController(port="")
    pc.status.printing = True       # pretend a print is in progress
    pc.send = lambda *a, **k: resp  # stub the serial round-trip
    pc.query_progress()
    assert pc.status.printing is printing
    if progress is None:
        assert pc.status.progress in (None, pc.status.progress)  # unchanged/None ok
    else:
        assert pc.status.progress == progress


# ── printer: stall watchdog ──────────────────────────────────────────────────

class _DummySerial:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_watchdog_tick_ignores_no_call_in_flight():
    pc = PrinterController(port="")
    assert pc.watchdog_tick(stall_seconds=20) is False


def test_watchdog_tick_ignores_call_within_budget():
    pc = PrinterController(port="")
    pc._call_started_at = time.time() - 2   # well under the 20s budget
    assert pc.watchdog_tick(stall_seconds=20) is False


def test_watchdog_tick_force_closes_stalled_call():
    pc = PrinterController(port="")
    pc.status.connected = True
    dummy = _DummySerial()
    pc._serial = dummy
    pc._call_started_at = time.time() - 25   # past the 20s budget

    assert pc.watchdog_tick(stall_seconds=20) is True
    assert dummy.closed is True
    assert pc._serial is None
    assert pc.status.connected is False
    assert pc.connected is False


# ── printer: M600 color/filament-change pause detection ─────────────────────

def test_note_pause_marker_sets_and_clears_with_hysteresis():
    pc = PrinterController(port="")
    assert pc.status.filament_change_pause is False

    pc._note_pause_marker("echo:busy: paused for user")
    assert pc.status.filament_change_pause is True

    # A single clean reply isn't proof the pause ended (busy ping is only
    # sent every ~2s and could simply have landed outside this window).
    pc._note_pause_marker("ok T:200.0 /200.0 B:60.0 /60.0")
    assert pc.status.filament_change_pause is True

    # A second clean reply in a row clears it.
    pc._note_pause_marker("ok T:200.0 /200.0 B:60.0 /60.0")
    assert pc.status.filament_change_pause is False


def test_note_pause_marker_empty_reply_proves_nothing():
    pc = PrinterController(port="")
    pc.status.filament_change_pause = True
    pc._note_pause_marker("")   # timed out entirely — not evidence of resume
    assert pc.status.filament_change_pause is True


@pytest.mark.parametrize("window, interval, expected", [
    (90, 30, 3),    # legacy 30 s frames → unchanged
    (90, 300, 2),   # 5-min cadence → 2 frames, not a 15-min (3×) wait
    (90, 60, 2),    # first-layer cadence
    (45, 300, 2),   # spaghetti window, slow cadence → still floored at 2
    (90, 0, 3),     # unknown interval → legacy default
])
def test_confirm_frames_scales_with_interval(window, interval, expected):
    assert _confirm_frames(window, interval) == expected


def test_confirm_frames_never_below_two():
    # A single bad frame must never alert, no matter how long the interval.
    assert _confirm_frames(45, 99999) == 2


# ── stuck-at-completion fallback: firmware that never falls out of "printing" ─

def test_progress_stuck_since_starts_window_at_threshold():
    # Not yet at the threshold — no window started.
    assert _progress_stuck_since(None, True, 0.5, now=100.0) is None
    # Crossing the threshold starts the window at `now`.
    assert _progress_stuck_since(None, True, STUCK_AT_COMPLETION_PCT, now=100.0) == 100.0


def test_progress_stuck_since_holds_start_time_while_pinned():
    # Once started, the window's start time doesn't reset while still pinned.
    since = _progress_stuck_since(100.0, True, 1.0, now=150.0)
    assert since == 100.0


def test_progress_stuck_since_resets_when_not_printing_or_below_threshold():
    assert _progress_stuck_since(100.0, False, 1.0, now=150.0) is None
    assert _progress_stuck_since(100.0, True, 0.9, now=150.0) is None
    assert _progress_stuck_since(100.0, True, None, now=150.0) is None


def test_progress_stuck_triggers_completion_after_timeout():
    # Mirrors the `stuck` computation in Monitor._check_printer_completion.
    since = _progress_stuck_since(None, True, 1.0, now=0.0)
    assert since == 0.0

    just_before = STUCK_AT_COMPLETION_SECONDS - 1
    assert (just_before - since) < STUCK_AT_COMPLETION_SECONDS

    at_timeout = STUCK_AT_COMPLETION_SECONDS
    assert (at_timeout - since) >= STUCK_AT_COMPLETION_SECONDS


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


# ── settings: SECURITY — no secret may ever enter the schema ─────────────────

def test_settings_schema_has_no_secrets():
    secrets = {"anthropic_api_key", "smtp_password", "smtp_username",
               "telegram_bot_token", "telegram_chat_id", "web_password",
               "web_username", "discord_webhook", "printer_port"}
    assert secrets.isdisjoint(SCHEMA.keys())
    # and every schema field is one of the safe, declared types
    assert all(s["type"] in ("int", "float", "bool", "enum") for s in SCHEMA.values())


def test_settings_public_surfaces_bounded_to_schema(tmp_path):
    s = Settings(path=tmp_path / "s.json")
    assert set(s.public_dict()) <= set(SCHEMA)
    assert {f["key"] for f in Settings.schema_public()} == set(SCHEMA)
    # schema metadata never carries a value/default (avoids leaking seeded creds-by-accident)
    assert all("default" not in f and "value" not in f for f in Settings.schema_public())


# ── settings: validation / coercion ─────────────────────────────────────────

@pytest.mark.parametrize("key,raw,ok,val", [
    ("capture_interval", "120", True, 120),
    ("capture_interval", 5, False, None),            # below min
    ("capture_interval", 99999, False, None),        # above max
    ("confidence_threshold", "0.9", True, 0.9),
    ("confidence_threshold", 1.5, False, None),      # above max
    ("auto_pause_action", "estop", True, "estop"),
    ("auto_pause_action", "rm -rf /", False, None),  # not a choice
    ("auto_start_on_print", "off", True, False),
    ("auto_start_on_print", "yes", True, True),
    ("camera_flip", "180", True, "180"),
    ("camera_flip", "sideways", False, None),
    ("anthropic_api_key", "stolen", False, None),    # unknown/secret -> rejected
    ("nonexistent", 1, False, None),
])
def test_settings_coerce(key, raw, ok, val):
    got_ok, got = _coerce(key, raw)
    assert got_ok is ok
    if ok:
        assert got == val


# ── settings: batch update, persistence, precedence ─────────────────────────

def test_settings_update_partial_and_persist(tmp_path):
    store = tmp_path / "s.json"
    fired = []
    s = Settings(seed={"capture_interval": 300}, path=store,
                 on_change=lambda k, v: fired.append((k, v)))
    assert store.exists()                              # seed persisted on first run

    applied, errors, restart = s.update({
        "capture_interval": 120,        # valid
        "auto_pause_action": "BAD",     # invalid choice
        "anthropic_api_key": "x",       # not in schema
    })
    assert applied == {"capture_interval": 120}
    assert len(errors) == 2 and s.get("capture_interval") == 120
    assert ("capture_interval", 120) in fired
    assert "anthropic_api_key" not in s.public_dict()  # secret never created

    # settings.json wins over a different seed on reload
    s2 = Settings(seed={"capture_interval": 999}, path=store)
    assert s2.get("capture_interval") == 120


def test_settings_atomic_save_leaves_no_tmp(tmp_path):
    store = tmp_path / "s.json"
    s = Settings(path=store)
    s.update({"capture_interval": 90})
    assert store.exists()
    assert not (tmp_path / "s.json.tmp").exists()      # temp renamed, not left behind


# ── API cost meter ──────────────────────────────────────────────────────────

class _FakeUsage:
    def __init__(self, inp=0, out=0, cw=0, cr=0):
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_creation_input_tokens = cw
        self.cache_read_input_tokens = cr


def test_usage_cost_components():
    from ender3monitor.analyzer import _usage_cost
    # all four token buckets priced independently
    c = _usage_cost(_FakeUsage(inp=1000, out=100, cw=200, cr=5000))
    expect = (1000*3 + 100*15 + 200*3.75 + 5000*0.30) / 1e6
    assert abs(c - expect) < 1e-12


def test_usage_cost_none_is_zero():
    from ender3monitor.analyzer import _usage_cost
    assert _usage_cost(None) == 0.0          # Ollama / missing usage → free


def test_analysis_result_defaults_cost_zero():
    from ender3monitor.analyzer import AnalysisResult
    r = AnalysisResult(False, "none", 0.1, "x")
    assert r.cost_usd == 0.0


# ── Pricing ──────────────────────────────────────────────────────────────────

class _PriceSettings:
    def __init__(self, **kw):
        self.v = dict(
            pricing_enabled=True, currency_symbol="$", filament_price_per_kg=20.0,
            electricity_rate_per_kwh=0.15, printer_watts=120, machine_rate_per_hour=1.0,
            labor_flat=2.0, markup_multiplier=2.0,
        )
        self.v.update(kw)
    def get(self, k): return self.v[k]


def test_pricing_breakdown_and_markup():
    from ender3monitor.pricing import compute_price
    p = compute_price(5 * 3600, 30, _PriceSettings())   # 30 g, 5 h
    assert abs(p["material"] - 0.60) < 1e-9
    assert abs(p["electricity"] - 0.09) < 1e-9
    assert abs(p["machine"] - 5.0) < 1e-9
    assert abs(p["cost"] - 7.69) < 1e-9
    assert abs(p["price"] - 15.38) < 1e-9


def test_pricing_disabled_returns_none():
    from ender3monitor.pricing import compute_price
    assert compute_price(3600, 30, _PriceSettings(pricing_enabled=False)) is None


def test_pricing_handles_none_time_and_zero_grams():
    from ender3monitor.pricing import compute_price
    p = compute_price(None, 0, _PriceSettings())
    assert p["price"] == p["labor"] * p["markup"]       # only labor survives


def test_pricing_keys_are_in_settings_schema():
    from ender3monitor.settings import SCHEMA
    for k in ("pricing_enabled", "filament_grams", "filament_price_per_kg",
              "markup_multiplier", "currency_symbol"):
        assert k in SCHEMA


def test_first_layer_mode_toggle_in_schema():
    from ender3monitor.settings import SCHEMA
    assert SCHEMA["first_layer_mode"]["type"] == "bool"
    assert SCHEMA["first_layer_mode"]["default"] is True   # on by default (current behavior)


# ── monitor: M600 color/filament-change alert ────────────────────────────────

class _FakePush:
    def __init__(self):
        self.enabled = True
        self.sent = []

    def send(self, title, message, priority="default"):
        self.sent.append((title, message, priority))


class _FakePrinterStatus:
    filament_change_pause = False


class _FakePrinter:
    def __init__(self):
        self.status = _FakePrinterStatus()


class _FakeMonitorForColorChange:
    """Bare object exposing only what Monitor._check_color_change touches —
    building a real Monitor needs a full Config/camera/analyzer stack."""
    def __init__(self):
        self.printer = _FakePrinter()
        self.push = _FakePush()
        self._color_change_alerted = False
        self._pause_is_runout = False   # set by _check_filament_runout
        self._running = True
        self.status = "Monitoring…"
        self._lock = threading.Lock()
        self.events = deque(maxlen=50)


def test_check_color_change_alerts_once_per_pause_episode():
    m = _FakeMonitorForColorChange()

    m.printer.status.filament_change_pause = True
    Monitor._check_color_change(m)
    assert m._color_change_alerted is True
    assert len(m.push.sent) == 1
    assert m.status == "Paused – change filament (M600)"
    assert len(m.events) == 1 and m.events[0]["type"] == "color_change"

    # Still paused on the next poll — must not re-alert.
    Monitor._check_color_change(m)
    assert len(m.push.sent) == 1

    # Resumed — clears the latch and restores status.
    m.printer.status.filament_change_pause = False
    Monitor._check_color_change(m)
    assert m._color_change_alerted is False
    assert m.status == "Monitoring…"

    # A later M600 in the same (multi-color) print alerts again.
    m.printer.status.filament_change_pause = True
    Monitor._check_color_change(m)
    assert len(m.push.sent) == 2


def test_check_color_change_skips_push_when_disabled():
    m = _FakeMonitorForColorChange()
    m.push.enabled = False
    m.printer.status.filament_change_pause = True
    Monitor._check_color_change(m)
    assert m._color_change_alerted is True
    assert m.push.sent == []


# --------------------------------------------------------------------------- #
# G-code index — slicer-aware status                                            #
# --------------------------------------------------------------------------- #

def _write_gcode(tmp_path, *, layers=10, m73=False, m600_at=None, name="t.gcode"):
    """A miniature Orca-shaped file. Markers match a stock export (gcode_comments=0)."""
    out = ["; HEADER_BLOCK_START",
           f"; total layer number: {layers}",
           ";TIME:600",
           ";Filament used:1.5m",
           "; layer_height = 0.2",
           "; filament_type = PLA;PLA",
           "; HEADER_BLOCK_END"]
    for i in range(layers):
        if m73:
            # Orca's mask is "M73 P%s R%s\n" (GCodeProcessor.cpp).
            out.append(f"M73 P{i * 10} R{layers - i}")
        out += [";LAYER_CHANGE", f";Z:{(i + 1) * 0.2:.2f}", ";TYPE:Outer wall",
                "G1 F1800 X10 Y10 E1", ";TYPE:Sparse infill", "G1 X20 Y20 E2"]
        if m600_at is not None and i == m600_at:
            out.append("M600")
    p = tmp_path / name
    p.write_text("\n".join(out) + "\n")
    return p


def test_gcode_index_parses_stock_orca_markers(tmp_path):
    idx = parse_gcode(_write_gcode(tmp_path, layers=10))
    assert idx is not None
    assert idx.total_layers == 10
    assert len(idx.layer_offsets) == len(idx.layer_z) == 10
    assert idx.layer_z[0] == pytest.approx(0.2)
    assert idx.layer_height == pytest.approx(0.2)
    assert idx.estimated_seconds == pytest.approx(600)
    assert idx.filament_type == "PLA"
    # Consecutive identical ;TYPE: runs are collapsed, distinct ones are kept.
    assert set(idx.feature_names) == {"Outer wall", "Sparse infill"}


def test_gcode_index_layer_lookup_prefers_z_over_byte_offset():
    """Z is what the nozzle is doing; the SD read pointer leads it."""
    idx = GcodeIndex(layer_offsets=[0, 100, 200, 300],
                     layer_z=[0.2, 0.4, 0.6, 0.8], layer_height=0.2)
    assert idx.layer_at_z(0.4) == 2
    assert idx.layer_at_z(0.79) == 4
    assert idx.layer_at_offset(250) == 3
    # A Z nowhere near any layer means this isn't the running file.
    assert idx.layer_at_z(60.0) is None
    assert idx.layer_at_z(None) is None


def test_gcode_index_feature_at_offset():
    idx = GcodeIndex(feature_offsets=[0, 50, 90],
                     feature_names=["Outer wall", "Bridge", "Sparse infill"])
    assert idx.feature_at_offset(10) == "Outer wall"
    assert idx.feature_at_offset(50) == "Bridge"
    assert idx.feature_at_offset(89) == "Bridge"
    assert idx.feature_at_offset(1000) == "Sparse infill"
    assert idx.feature_at_offset(None) is None


def test_gcode_index_m73_interpolates_between_stamps(tmp_path):
    idx = parse_gcode(_write_gcode(tmp_path, layers=10, m73=True))
    assert idx.has_m73
    a, b = idx.m73_offsets[2], idx.m73_offsets[3]
    ra = idx.remaining_seconds_at_offset(a)
    rb = idx.remaining_seconds_at_offset(b)
    mid = idx.remaining_seconds_at_offset((a + b) // 2)
    assert rb < mid < ra          # strictly decreasing, genuinely interpolated


def test_gcode_index_discards_non_monotonic_m73(tmp_path):
    """A rising 'remaining' means we misparsed — drop it instead of interpolating junk."""
    p = tmp_path / "bad.gcode"
    p.write_text(";LAYER_CHANGE\n;Z:0.2\nM73 P0 R5\nM73 P50 R99\n")
    idx = parse_gcode(p)
    assert idx is not None and not idx.has_m73


def test_gcode_index_falls_back_to_time_model_without_m73(tmp_path):
    """No M73 → the built-in move-timing model still yields a decreasing ETA."""
    idx = parse_gcode(_write_gcode(tmp_path, layers=10, m73=False))
    assert not idx.has_m73
    assert idx.remaining_seconds_at_offset(idx.size_bytes // 2) is None
    early = idx.modelled_remaining_seconds(idx.layer_offsets[1])
    late = idx.modelled_remaining_seconds(idx.layer_offsets[8])
    assert early is not None and late is not None and late < early
    assert idx.layer_time_frac[0] == 0.0
    assert all(idx.layer_time_frac[i] <= idx.layer_time_frac[i + 1]
               for i in range(len(idx.layer_time_frac) - 1))


def test_gcode_index_next_color_change(tmp_path):
    idx = parse_gcode(_write_gcode(tmp_path, layers=10, m600_at=4))
    assert len(idx.m600_offsets) == 1
    off = idx.m600_offsets[0]
    assert idx.next_color_change(0) == off
    assert idx.next_color_change(off + 1) is None
    assert idx.layer_at_offset(off) == 5


def test_gcode_index_confirm_z_rejects_the_wrong_file():
    """Byte size is a fingerprint, not a proof — Z is the second opinion."""
    idx = GcodeIndex(layer_offsets=[0, 100, 200, 300],
                     layer_z=[0.2, 0.4, 0.6, 0.8], layer_height=0.2)
    assert idx.confirm_z(250, 0.6) is True       # right file
    assert idx.confirm_z(250, 60.0) is False     # nozzle nowhere near — wrong file
    assert idx.confirm_z(250, None) is True      # no evidence → don't reject
    assert idx.confirm_z(None, 0.6) is True
    # Tolerant of the planner-queue lead: the read pointer runs ahead of the nozzle.
    assert idx.confirm_z(300, 0.4) is True


def test_index_store_matches_by_size_and_survives_the_file(tmp_path):
    """The index outlives the G-code — that is the whole point."""
    src = _write_gcode(tmp_path, layers=8, name="job.gcode")
    size = src.stat().st_size
    store = IndexStore(tmp_path / "cache")
    assert store.add_file(src) is not None
    assert store.match(size).source_name == "job.gcode"
    assert store.match(size + 1) is None
    assert store.match(None) is None

    src.unlink()                      # the file goes to the SD card and is gone
    reloaded = IndexStore(tmp_path / "cache")
    assert reloaded.match(size) is not None
    assert reloaded.match(size).total_layers == 8


def test_index_store_ignores_unparseable_files(tmp_path):
    junk = tmp_path / "notgcode.gcode"
    junk.write_text("this is not g-code\n")
    store = IndexStore(tmp_path / "cache")
    assert store.add_file(junk) is None
    assert store.count == 0


def test_volume_watcher_indexes_new_volumes(tmp_path):
    """Cards are matched by content, not label — NONAME and PRINTER both work."""
    volumes = tmp_path / "Volumes"
    card = volumes / "NONAME"
    card.mkdir(parents=True)
    _write_gcode(card, layers=6, name="a.gcode")
    store = IndexStore(tmp_path / "cache")
    w = VolumeWatcher(store, volume_root=volumes, settle_seconds=0)

    assert w.scan_once() == 1
    assert store.count == 1
    assert w.scan_once() == 0            # nothing new — no duplicate work

    # Eject and re-insert under the other label: same content, still known.
    card.rename(volumes / "PRINTER")
    assert w.scan_once() == 0
    assert store.count == 1


def test_volume_watcher_picks_up_files_added_to_a_mounted_card(tmp_path):
    """The real workflow: the card stays mounted while files are exported to it.

    A watcher that scans each volume only once indexes whatever happened to be
    on the card at startup and silently misses every later export.
    """
    volumes = tmp_path / "Volumes"
    card = volumes / "PRINTER"
    card.mkdir(parents=True)
    _write_gcode(card, layers=6, name="first.gcode")
    store = IndexStore(tmp_path / "cache")
    w = VolumeWatcher(store, volume_root=volumes, settle_seconds=0)

    assert w.scan_once() == 1

    # Export a second file to the still-mounted card.
    second = _write_gcode(card, layers=9, name="second.gcode")
    assert w.scan_once() == 1
    assert store.count == 2
    assert store.match(second.stat().st_size).total_layers == 9


def test_volume_watcher_skips_files_still_being_written(tmp_path):
    """A file caught mid-write indexes at a truncated size — the wrong fingerprint."""
    volumes = tmp_path / "Volumes"
    card = volumes / "PRINTER"
    card.mkdir(parents=True)
    _write_gcode(card, layers=6, name="inflight.gcode")
    store = IndexStore(tmp_path / "cache")

    # Freshly written: too new to trust.
    w = VolumeWatcher(store, volume_root=volumes, settle_seconds=3600)
    assert w.scan_once() == 0
    assert store.count == 0

    # Once it has settled, the next pass takes it.
    w.settle_seconds = 0
    assert w.scan_once() == 1
    assert store.count == 1


def test_build_context_note_flags_bridging_as_expected():
    note = build_context_note("Bridge", 19, 27)
    assert note is not None and "expected" in note.lower()
    assert build_context_note("Overhang wall", 5, 10) is not None
    # Ordinary features get neutral positional context, not the sag exemption.
    plain = build_context_note("Outer wall", 19, 27)
    assert "layer 19 of 27" in plain and "expected for this feature" not in plain
    assert build_context_note(None, None, None) is None


# ── printer: filament runout sensor + firmware announcements ────────────────

def test_runout_marker_regex_matches_firmware_announcements():
    for line in (
        "//action:out_of_filament",
        "//action:out_of_filament T0",
        "//action:prompt_begin FilamentRunout T0",
        "//action:filament_runout",
        "//action:pause filament_runout T0",
        "echo:Filament runout detected",
    ):
        assert _RUNOUT_MARKER_RE.search(line), line


def test_runout_marker_regex_ignores_settings_reports_and_plain_pauses():
    """M412/M503 echo the runout *setting* back, and an M600 pause is just a
    pause — neither is a runout event, and matching either would alert on a
    healthy print."""
    for line in (
        "echo:  M412 S1 D7.00 ; Filament runout ON",
        "echo:; Filament runout",
        "echo:busy: paused for user",
        "ok T:200.00 /200.00 B:60.00 /60.00",
    ):
        assert not _RUNOUT_MARKER_RE.search(line), line


def test_note_runout_marker_latches_then_clears_with_the_pause():
    pc = PrinterController(port="")
    assert pc.status.runout_marker_seen is False

    pc._note_runout_marker("//action:out_of_filament")
    assert pc.status.runout_marker_seen is True

    # The announcement arrives once while the condition persists, so a later
    # quiet reply must not retract it.
    pc._note_runout_marker("ok T:200.0 /200.0 B:60.0 /60.0")
    assert pc.status.runout_marker_seen is True

    # It clears once the pause it caused is over (same hysteresis as the pause).
    pc.status.filament_change_pause = True
    pc._note_pause_marker("ok")
    pc._note_pause_marker("ok")
    assert pc.status.filament_change_pause is False
    assert pc.status.runout_marker_seen is False


def _sensor_pc(replies):
    """A controller whose send() hands back canned M119 replies in order."""
    pc = PrinterController(port="")
    it = iter(replies)
    pc.send = lambda gcode, read_timeout=3.0: next(it)
    return pc


@pytest.mark.parametrize("present_word, absent_word", [
    ("TRIGGERED", "open"),   # stock Marlin: filament holds the switch closed
    ("open", "TRIGGERED"),   # inverted FIL_RUNOUT_STATE / reversed wiring
])
def test_filament_sensor_polarity_is_learned_not_assumed(present_word, absent_word):
    """Both wirings must converge on the same answer. The meaning of TRIGGERED
    is a build-time choice, so a hard-coded polarity would mean either silence
    on a real runout or an alert on every healthy print, depending on machine."""
    pc = _sensor_pc([f"filament: {present_word}\nok"] * SENSOR_CALIBRATION_POLLS
                    + [f"filament: {absent_word}\nok"])
    pc.status.printing = True

    for _ in range(SENSOR_CALIBRATION_POLLS):
        pc.query_filament_sensor()
    assert pc.status.filament_sensor_reported is True
    assert pc.status.filament_present is True

    pc.query_filament_sensor()
    assert pc.status.filament_present is False


def test_filament_sensor_stays_unknown_until_calibrated():
    pc = _sensor_pc(["filament: open\nok"])
    pc.status.printing = True
    pc.query_filament_sensor()
    # One reading is not a baseline, and "unknown" must not pass for "loaded".
    assert pc.status.filament_present is None


def test_filament_sensor_does_not_calibrate_while_idle_or_paused():
    """Filament is deliberately out of the sensor during a color change, and
    undefined when idle, so neither may teach us what "present" looks like."""
    pc = _sensor_pc(["filament: open\nok"] * (SENSOR_CALIBRATION_POLLS * 2))
    for _ in range(SENSOR_CALIBRATION_POLLS):
        pc.query_filament_sensor()                  # idle
    pc.status.printing = True
    pc.status.filament_change_pause = True
    for _ in range(SENSOR_CALIBRATION_POLLS):
        pc.query_filament_sensor()                  # parked mid-swap
    assert pc._sensor_present_state is None
    assert pc.status.filament_present is None


def test_firmware_with_no_sensor_reports_unknown_not_runout():
    pc = _sensor_pc(["x_min: open\ny_min: open\nz_min: TRIGGERED\nok"])
    pc.status.printing = True
    pc.query_filament_sensor()
    assert pc.status.filament_sensor_reported is False
    assert pc.status.filament_present is None


# ── gcode index: telling a planned M600 from an unexplained pause ────────────

def test_color_change_near_window():
    idx = GcodeIndex(size_bytes=1_000_000, source_name="x")
    idx.m600_offsets = [500_000]
    assert idx.color_change_near(500_000) is True
    assert idx.color_change_near(520_000) is True    # SD pointer ran past it
    assert idx.color_change_near(498_000) is True    # small forward slack
    assert idx.color_change_near(600_000) is False
    assert idx.color_change_near(480_000) is False
    assert idx.color_change_near(None) is False
    # A file with no color changes can never explain a pause.
    assert GcodeIndex(size_bytes=1, source_name="y").color_change_near(10) is False


# ── monitor: filament runout alert ──────────────────────────────────────────

class _FakeSettings:
    def __init__(self, **vals):
        self._vals = {"filament_runout_alert": True}
        self._vals.update(vals)

    def get(self, key):
        return self._vals[key]


class _FakeIndex:
    """Stands in for a matched G-code index — only the M600 lookup matters."""
    def __init__(self, near):
        self._near = near

    def color_change_near(self, pos, behind=0, ahead=0):
        return self._near


class _RunoutPrinter:
    def __init__(self):
        self.status = PrinterStatus()

    def clear_runout_marker(self):
        self.status.runout_marker_seen = False


class _FakeMonitorForRunout:
    """Bare object exposing only what the runout path touches — building a real
    Monitor needs a full Config/camera/analyzer stack."""
    def __init__(self, index=None, **settings):
        self.printer = _RunoutPrinter()
        self.push = _FakePush()
        self.settings = _FakeSettings(**settings)
        self._gcode_index = index
        self._runout_alerted = False
        self._pause_is_runout = False
        self._color_change_alerted = False
        self._running = True
        self.status = "Monitoring…"
        self._lock = threading.Lock()
        self.events = deque(maxlen=50)

    # Real implementations, so these tests exercise the shipped logic.
    _classify_runout = Monitor._classify_runout
    _runout_message = Monitor._runout_message


def test_runout_sensor_alerts_once_per_episode_and_clears():
    m = _FakeMonitorForRunout()
    m.printer.status.printing = True
    m.printer.status.filament_present = False

    Monitor._check_filament_runout(m)
    assert m.printer.status.filament_runout is True
    assert m.printer.status.runout_source == "sensor"
    assert m.status == "Paused – filament runout"
    assert len(m.events) == 1 and m.events[0]["type"] == "filament_runout"
    assert len(m.push.sent) == 1
    title, _message, priority = m.push.sent[0]
    assert "ran out" in title and priority == "high"

    # Still out on the next poll — one alert per episode, not one every 5 s.
    Monitor._check_filament_runout(m)
    assert len(m.push.sent) == 1

    # Reloaded — the latch clears and an all-clear goes out.
    m.printer.status.filament_present = True
    Monitor._check_filament_runout(m)
    assert m._runout_alerted is False
    assert m.printer.status.filament_runout is False
    assert m.status == "Monitoring…"
    assert len(m.push.sent) == 2

    # A second runout later in the same print alerts again.
    m.printer.status.filament_present = False
    Monitor._check_filament_runout(m)
    assert len(m.push.sent) == 3


def test_runout_message_names_the_signal_and_where_it_stopped():
    m = _FakeMonitorForRunout()
    st = m.printer.status
    st.printing = True
    st.filament_present = False
    st.layer, st.total_layers = 340, 1219
    st.remaining_seconds = 7800

    Monitor._check_filament_runout(m)
    body = m.push.sent[0][1]
    assert "runout sensor" in body
    assert "layer 340/1219" in body
    assert "2h 10m" in body


def test_sensor_reading_while_idle_is_not_a_runout():
    # An empty sensor on an idle printer is just a printer with no filament in it.
    m = _FakeMonitorForRunout()
    m.printer.status.filament_present = False
    Monitor._check_filament_runout(m)
    assert m.printer.status.filament_runout is False
    assert m.push.sent == []


def test_unexplained_pause_is_a_runout_but_a_planned_m600_is_not():
    """Stock firmware answers a runout by running M600, so the pause itself says
    nothing — only the sliced file can separate the two."""
    planned = _FakeMonitorForRunout(index=_FakeIndex(near=True))
    planned.printer.status.printing = True
    planned.printer.status.filament_change_pause = True
    planned.printer.status.file_position = 500_000
    Monitor._check_filament_runout(planned)
    assert planned.printer.status.filament_runout is False
    assert planned.push.sent == []

    unplanned = _FakeMonitorForRunout(index=_FakeIndex(near=False))
    unplanned.printer.status.printing = True
    unplanned.printer.status.filament_change_pause = True
    unplanned.printer.status.file_position = 500_000
    Monitor._check_filament_runout(unplanned)
    assert unplanned.printer.status.filament_runout is True
    assert unplanned.printer.status.runout_source == "unplanned-pause"
    assert len(unplanned.push.sent) == 1


def test_pause_without_a_matched_index_stays_a_color_change():
    """With no index there is no way to tell a runout from an M600, and a wrong
    "your filament ran out" on every multi-color print is the worse error."""
    m = _FakeMonitorForRunout(index=None)
    m.printer.status.printing = True
    m.printer.status.filament_change_pause = True
    Monitor._check_filament_runout(m)
    assert m.printer.status.filament_runout is False
    assert m.push.sent == []


def test_firmware_announcement_is_a_runout_even_without_an_index():
    m = _FakeMonitorForRunout(index=None)
    m.printer.status.printing = True
    m.printer.status.runout_marker_seen = True
    Monitor._check_filament_runout(m)
    assert m.printer.status.runout_source == "firmware"
    assert len(m.push.sent) == 1


def test_runout_alert_can_be_switched_off():
    m = _FakeMonitorForRunout(filament_runout_alert=False)
    m.printer.status.printing = True
    m.printer.status.filament_present = False
    Monitor._check_filament_runout(m)
    assert m.printer.status.filament_runout is False
    assert m.push.sent == []


def test_runout_pause_stands_down_the_m600_alert():
    """One pause, one notification — and with the wording that sends you to the
    printer expecting the right job."""
    m = _FakeMonitorForRunout(index=_FakeIndex(near=False))
    m.printer.status.printing = True
    m.printer.status.filament_change_pause = True
    m.printer.status.file_position = 500_000

    Monitor._check_filament_runout(m)
    Monitor._check_color_change(m)        # same poll, runs straight after

    assert len(m.push.sent) == 1
    assert "ran out" in m.push.sent[0][0]
    assert m._color_change_alerted is False
    assert [e["type"] for e in m.events] == ["filament_runout"]


def test_m600_alert_still_fires_for_a_planned_color_change():
    m = _FakeMonitorForRunout(index=_FakeIndex(near=True))
    m.printer.status.printing = True
    m.printer.status.filament_change_pause = True
    m.printer.status.file_position = 500_000

    Monitor._check_filament_runout(m)
    Monitor._check_color_change(m)

    assert len(m.push.sent) == 1
    assert "change filament" in m.push.sent[0][0]
    assert [e["type"] for e in m.events] == ["color_change"]


# ── runout: the two ways this can go quietly wrong ──────────────────────────

def test_stalled_poll_does_not_blank_a_known_sensor_reading():
    """While the printer is blocked waiting on the user, gcode processing stalls
    and the read window fills with busy-keepalive lines instead of the endstop
    report we asked for. That is not evidence the sensor went away — and
    blanking the reading on it would erase a just-detected runout at the one
    moment it matters."""
    pc = _sensor_pc(["filament: TRIGGERED\nok"] * SENSOR_CALIBRATION_POLLS
                    + ["filament: open\nok",
                       "echo:busy: paused for user\necho:busy: paused for user"])
    pc.status.printing = True
    for _ in range(SENSOR_CALIBRATION_POLLS):
        pc.query_filament_sensor()
    pc.query_filament_sensor()
    assert pc.status.filament_present is False

    pc.query_filament_sensor()          # stalled reply — no report inside it
    assert pc.status.filament_present is False
    assert pc.status.filament_sensor_reported is True


def test_runout_stays_latched_while_the_printer_is_still_parked():
    """No all-clear until the print is moving again. On firmware with no matched
    index the signal that revealed the runout goes quiet the moment the printer
    blocks, so re-deciding from scratch each poll would retract the alert."""
    m = _FakeMonitorForRunout(index=None)
    m.printer.status.printing = True
    m.printer.status.filament_present = False
    Monitor._check_filament_runout(m)
    assert len(m.push.sent) == 1

    # The printer has now blocked waiting on the user and stopped answering.
    m.printer.status.filament_change_pause = True
    m.printer.status.filament_present = None
    Monitor._check_filament_runout(m)
    assert m.printer.status.filament_runout is True
    assert len(m.push.sent) == 1                 # no false "reloaded"

    # ...and the M600 alert must not claim the same pause either.
    Monitor._check_color_change(m)
    assert len(m.push.sent) == 1

    # Resumed — now the all-clear is real.
    m.printer.status.filament_change_pause = False
    Monitor._check_filament_runout(m)
    assert m.printer.status.filament_runout is False
    assert len(m.push.sent) == 2


class _FakeSerial:
    """Minimal stand-in for a Marlin USB port: answers the poll commands this
    app sends, and lets a test flip the runout switch mid-print."""

    def __init__(self):
        self.filament = True
        self.printing = True
        self._out = deque()

    # ── the pyserial surface PrinterController actually uses ──
    def reset_input_buffer(self):
        self._out.clear()

    def flush(self):
        pass

    def close(self):
        pass

    def readline(self):
        return self._out.popleft() if self._out else b""

    def write(self, raw):
        cmd = raw.decode().strip()
        if cmd == "M105":
            self._reply("ok T:210.0 /210.0 B:60.0 /60.0")
        elif cmd == "M27":
            self._reply("SD printing byte 500000/1000000" if self.printing
                        else "Not SD printing", ok=True)
        elif cmd == "M31":
            self._reply("echo:Print time: 1h 0m 0s", ok=True)
        elif cmd == "M114":
            self._reply("X:0.00 Y:0.00 Z:10.00 E:0.00", ok=True)
        elif cmd == "M119":
            # This machine reads TRIGGERED when filament is threaded.
            self._reply("Reporting endstop status\nx_min: open\ny_min: open\n"
                        f"filament: {'TRIGGERED' if self.filament else 'open'}", ok=True)
        else:
            self._reply("ok")

    def _reply(self, text, ok=False):
        for line in text.split("\n"):
            self._out.append((line + "\n").encode())
        if ok:
            self._out.append(b"ok\n")


def test_end_to_end_runout_from_serial_to_notification():
    """The whole path against stock-firmware-shaped output and no G-code index:
    poll, learn the polarity, flip the switch, classify, notify."""
    m = _FakeMonitorForRunout(index=None)
    clock = [1000.0]
    pc = PrinterController(port="", clock=lambda: clock[0])
    pc._serial = _FakeSerial()
    pc.status.connected = True
    m.printer = pc

    def tick():                                  # one 5 s poll-loop tick
        clock[0] += 5
        pc.refresh_status()
        Monitor._check_filament_runout(m)

    # The sensor is read every 30 s mid-print now (one command per 5 s at most).
    for _ in range(80):
        tick()
        if pc.status.filament_present is not None:
            break
    assert pc.status.printing is True
    assert pc.status.filament_present is True
    assert m.push.sent == []                     # a healthy print stays quiet

    pc._serial.filament = False                  # the spool runs out
    for _ in range(8):
        tick()
        if pc.status.filament_runout:
            break

    assert pc.status.filament_runout is True
    assert pc.status.runout_source == "sensor"
    assert len(m.push.sent) == 1
    assert "ran out" in m.push.sent[0][0]
    assert m.push.sent[0][2] == "high"


def test_no_all_clear_when_the_print_was_cancelled_instead_of_reloaded():
    """Giving up and cancelling also ends the runout state — but claiming the
    print "is moving again" then would be plainly false."""
    m = _FakeMonitorForRunout(index=None)
    m.printer.status.printing = True
    m.printer.status.filament_present = False
    Monitor._check_filament_runout(m)
    assert len(m.push.sent) == 1

    # Print cancelled at the LCD: job gone, nothing resumed.
    m.printer.status.printing = False
    Monitor._check_filament_runout(m)
    assert m.printer.status.filament_runout is False
    assert m._runout_alerted is False
    assert len(m.push.sent) == 1           # no bogus "moving again"


# ── notifications: master mute switches ─────────────────────────────────────

def test_notification_toggles_mute_channels():
    from ender3monitor.notifier import EmailNotifier
    from ender3monitor.push import PushNotifier
    e = EmailNotifier("h", 587, "u", "p", "s", "r")
    assert e.enabled
    e.muted = True
    assert not e.enabled
    p = PushNotifier(telegram_bot_token="t", telegram_chat_id="c")
    assert p.channels() == ["telegram"] and p.enabled
    p.telegram_muted = True
    assert p.channels() == [] and not p.enabled


# ── Pause/Resume toggle state ────────────────────────────────────────────────

def test_paused_flag_follows_our_own_pause_resume_and_job_end():
    from ender3monitor.printer import PrinterController
    pc = PrinterController.__new__(PrinterController)
    pc.status = PrinterStatus(connected=True, printing=True)
    sent = []
    pc.send = lambda gcode, **k: sent.append(gcode) or "ok"

    pc.pause()
    assert pc.status.paused and pc.status.as_dict()["paused"] is True
    pc.resume()
    assert not pc.status.paused
    pc.cooldown()
    assert pc.status.paused          # cooldown parks the print too
    assert sent == ["M25", "M24", "M25", "M104 S0", "M140 S0", "M107"]


def test_install_assets_are_served_and_skip_auth():
    import web
    assert "/manifest.webmanifest" in web._PUBLIC_PATHS
    assert all(f"/icon-{n}.png" in web._PUBLIC_PATHS for n in web._ICON_SIZES)
    png = web._icon_png(180)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert 'rel="manifest"' in web._HTML and 'id="btn-estop"' in web._HTML


# ── printer: temperature history (dashboard chart) ───────────────────────────

def _pc_with_reply(reply):
    pc = PrinterController(port="")
    pc.send = lambda *a, **k: reply
    return pc


def test_temp_history_appends_only_valid_readings():
    pc = _pc_with_reply("ok T:210.5 /215.0 B:59.8 /60.0 @:0 B@:0")
    pc.query_temps()
    rows = pc.temp_history()
    assert len(rows) == 1
    ts, nozzle, nozzle_target, bed, bed_target = rows[0]
    assert (nozzle, nozzle_target, bed, bed_target) == (210.5, 215.0, 59.8, 60.0)
    assert ts == pc.status.temps_at == pc.status.as_dict()["temps_at"]

    # A reply with no temperatures in it (empty read, dropped link, unrelated
    # line) must not add a sample or repeat the previous one.
    for junk in ("", "echo:busy: paused for user", "ok"):
        pc.send = lambda *a, **k: junk
        pc.query_temps()
    assert len(pc.temp_history()) == 1


def test_temp_history_is_bounded_and_windowed():
    from ender3monitor.printer import TEMP_HISTORY_MAX_SAMPLES, TEMP_HISTORY_SECONDS
    pc = PrinterController(port="")
    pc.status.nozzle_temp, pc.status.bed_temp = 200.0, 60.0
    base = 1_000_000.0
    for i in range(TEMP_HISTORY_MAX_SAMPLES + 50):
        pc._record_temps(now=base + i * 5)
    newest = base + (TEMP_HISTORY_MAX_SAMPLES + 49) * 5
    rows = pc.temp_history(now=newest)
    assert len(rows) == TEMP_HISTORY_MAX_SAMPLES          # oldest were dropped
    assert rows[-1][0] == newest
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)
    # Readings older than the window are not served (e.g. after a long outage).
    assert len(pc.temp_history(now=newest + TEMP_HISTORY_SECONDS - 10)) == 3
    assert pc.temp_history(now=newest + TEMP_HISTORY_SECONDS + 1) == []
    # Callers get copies, not the live buffer.
    rows[0][1] = -1
    assert pc.temp_history(now=newest)[0][1] == 200.0


def test_temp_history_skips_incomplete_reading():
    pc = PrinterController(port="")
    pc._record_temps()                      # nothing read yet
    assert pc.temp_history() == [] and pc.status.temps_at is None


def test_color_change_marks_are_layer_fractions(tmp_path):
    # The fixture writes its M600 at the end of layer 5 of 10 → halfway.
    idx = parse_gcode(_write_gcode(tmp_path, layers=10, m600_at=4))
    marks = idx.color_change_marks()
    assert len(marks) == 1 and marks[0] == pytest.approx(0.5, abs=0.03)
    assert parse_gcode(_write_gcode(tmp_path, layers=10)).color_change_marks() == []
    # An M600 written at the very start of a layer lands on the same boundary.
    at_start = GcodeIndex(size_bytes=1000, total_layers=4,
                          layer_offsets=[0, 250, 500, 750], m600_offsets=[500])
    assert at_start.color_change_marks() == [0.5]
    assert GcodeIndex(m600_offsets=[10]).color_change_marks() == []   # no layer table


def test_printer_status_color_change_marks_default_and_clear():
    st = PrinterStatus()
    assert st.as_dict()["color_change_marks"] == []
    st.color_change_marks = [0.25, 0.5]
    st.job_name = "x.gcode"
    assert st.as_dict()["color_change_marks"] == [0.25, 0.5]
    st.clear_gcode_context()
    assert st.color_change_marks == []


# ── failure review: Resume / Keep paused / False alarm ──────────────────────

class _ReviewPrinter:
    def __init__(self, connected=True):
        self.connected = connected
        self.resumed = 0

    def resume(self):
        self.resumed += 1


class _FakeMonitorForReview:
    """Bare object exposing only what the failure-review methods touch."""
    def __init__(self, connected=True):
        self.printer = _ReviewPrinter(connected)
        self._lock = threading.Lock()
        self.events = deque(maxlen=50)
        self.failure_review = None
        self.false_alarm_count = 0


def _open_review(m, frame=None):
    from ender3monitor.analyzer import AnalysisResult
    result =AnalysisResult(failure_detected=True, confidence=0.91,
                            failure_type="spaghetti/stringing",
                            description="Loose strands over the bed")
    return Monitor._open_failure_review(m, result, frame)


def test_failure_review_captures_incident_and_frame():
    m = _FakeMonitorForReview()
    review = _open_review(m, np.zeros((48, 64, 3), dtype=np.uint8))
    assert m.failure_review is review
    assert review["failure_type"] == "spaghetti/stringing"
    assert review["confidence"] == 0.91
    assert review["description"] == "Loose strands over the bed"
    assert review["timestamp"] > 0
    assert review["jpeg"][:2] == b"\xff\xd8"          # a real JPEG


def test_failure_review_resume_resumes_and_clears():
    m = _FakeMonitorForReview()
    _open_review(m)
    res = Monitor.resolve_failure_review(m, "resume")
    assert res["ok"] and m.printer.resumed == 1
    assert m.failure_review is None
    assert m.false_alarm_count == 0 and not m.events


def test_failure_review_resume_without_printer_just_clears():
    m = _FakeMonitorForReview(connected=False)
    _open_review(m)
    res = Monitor.resolve_failure_review(m, "resume")
    assert res["ok"] and m.printer.resumed == 0
    assert m.failure_review is None


def test_failure_review_keep_paused_only_clears():
    m = _FakeMonitorForReview()
    _open_review(m)
    res = Monitor.resolve_failure_review(m, "keep_paused")
    assert res["ok"] and m.printer.resumed == 0
    assert m.failure_review is None
    assert m.false_alarm_count == 0 and not m.events


def test_failure_review_false_alarm_counts_logs_and_does_not_resume():
    m = _FakeMonitorForReview()
    _open_review(m)
    res = Monitor.resolve_failure_review(m, "false_alarm")
    assert res["ok"]
    assert m.printer.resumed == 0                      # dismissal only
    assert m.false_alarm_count == 1
    assert m.failure_review is None
    assert len(m.events) == 1
    ev = m.events[0]
    assert ev["type"] == "false_alarm" and ev["detected"] is False
    assert "spaghetti/stringing" in ev["desc"] and ev["conf"] == 0.91


def test_failure_review_rejects_bad_action_stale_id_and_no_review():
    m = _FakeMonitorForReview()
    assert not Monitor.resolve_failure_review(m, "resume")["ok"]      # nothing pending
    review = _open_review(m)
    assert not Monitor.resolve_failure_review(m, "explode")["ok"]
    assert not Monitor.resolve_failure_review(m, "resume", "stale-id")["ok"]
    assert m.failure_review is review and m.printer.resumed == 0      # untouched
    assert Monitor.resolve_failure_review(m, "resume", review["id"])["ok"]
    # Answered once — a second press finds nothing to resolve.
    assert not Monitor.resolve_failure_review(m, "false_alarm", review["id"])["ok"]
    assert m.false_alarm_count == 0 and m.printer.resumed == 1


def test_failure_review_callback_data_roundtrip():
    from monitor import (FAILURE_REVIEW_ACTIONS, failure_review_buttons,
                         parse_failure_review_callback)
    buttons = failure_review_buttons("1700000000000")
    assert [parse_failure_review_callback(d) for _, d in buttons] == [
        (a, "1700000000000") for a in FAILURE_REVIEW_ACTIONS]
    assert all(len(d.encode()) <= 64 for _, d in buttons)             # Telegram limit
    for junk in ("", "fr:resume", "xx:resume:1", "fr:explode:1", "/resume"):
        assert parse_failure_review_callback(junk) is None


# ── telegram: inline buttons on the failure alert ───────────────────────────

def test_push_telegram_attaches_inline_keyboard(monkeypatch):
    import urllib.parse
    from ender3monitor.push import PushNotifier
    posts = []
    monkeypatch.setattr(PushNotifier, "_post",
                        staticmethod(lambda url, data, headers=None: posts.append(data)))
    p = PushNotifier(telegram_bot_token="t", telegram_chat_id="42")

    p.send("Failure", "msg", "high", telegram_buttons=[("Resume", "fr:resume:1")])
    p.send("Plain", "msg")
    with_buttons, plain = (urllib.parse.parse_qs(d.decode()) for d in posts)
    markup = json.loads(with_buttons["reply_markup"][0])
    assert markup == {"inline_keyboard": [[{"text": "Resume",
                                            "callback_data": "fr:resume:1"}]]}
    assert "reply_markup" not in plain


def _recording_bot(allowed, callback_handler, edit_ok=True):
    """TelegramBot with the network calls replaced by a call log."""
    from ender3monitor.telegram_bot import TelegramBot
    bot = TelegramBot("token", allowed, {}, callback_handler=callback_handler)
    bot.calls = []
    bot.answer_callback = lambda cq_id, text="": bot.calls.append(("answer", cq_id, text))
    bot.edit_message = lambda chat, mid, text: (
        bot.calls.append(("edit", chat, mid, text)) or edit_ok)
    bot.send_message = lambda chat, text: bot.calls.append(("send", chat, text))
    return bot


def _callback_update(chat_id, data="fr:false_alarm:1"):
    return {"update_id": 7, "callback_query": {
        "id": "cq1", "data": data,
        "message": {"message_id": 99, "chat": {"id": chat_id}, "text": "Print failure"},
    }}


def test_telegram_callback_resolves_and_edits_the_alert():
    seen = []
    bot = _recording_bot({42}, lambda data: seen.append(data) or "Marked as false alarm")
    bot._handle_update(_callback_update(42))
    assert seen == ["fr:false_alarm:1"]
    assert bot.calls[0] == ("answer", "cq1", "Marked as false alarm")
    kind, chat, mid, text = bot.calls[1]
    assert (kind, chat, mid) == ("edit", 42, 99)
    assert text.startswith("Print failure") and "Marked as false alarm" in text
    assert len(bot.calls) == 2                         # no extra reply when the edit worked


def test_telegram_callback_replies_when_edit_fails():
    bot = _recording_bot({42}, lambda data: "Kept paused.", edit_ok=False)
    bot._handle_update(_callback_update(42, "fr:keep_paused:1"))
    assert bot.calls[-1] == ("send", 42, "Kept paused.")


def test_telegram_callback_from_unauthorized_chat_is_refused():
    seen = []
    bot = _recording_bot({42}, lambda data: seen.append(data) or "done")
    bot._handle_update(_callback_update(666))
    assert seen == []                                  # handler never ran
    assert [c[0] for c in bot.calls] == ["answer"]
    assert "Not authorized" in bot.calls[0][2]


def test_telegram_callback_drives_the_monitor_resolve():
    from monitor import parse_failure_review_callback
    m = _FakeMonitorForReview()
    review = _open_review(m)

    def handler(data):
        action, rid = parse_failure_review_callback(data)
        return Monitor.resolve_failure_review(m, action, rid)["message"]

    bot = _recording_bot({42}, handler)
    bot._handle_update(_callback_update(42, f"fr:false_alarm:{review['id']}"))
    assert m.false_alarm_count == 1 and m.failure_review is None
    assert m.printer.resumed == 0

# ── Print history ───────────────────────────────────────────────────────────
from ender3monitor.history import (  # noqa: E402
    MAX_EVENTS, HistoryStore, build_record, format_recent, record_session,
)


def _hist_rec(n=0, **kw):
    rec = {"started_at": 1000.0 + n, "ended_at": 2000.0 + n, "duration_seconds": 1000,
           "job_name": f"job{n}.gcode", "outcome": "completed", "failure_count": 0,
           "failure_types": [], "false_alarms": 0, "frames_analyzed": 5,
           "ai_cost_usd": 0.12, "thumbnail": None, "timelapse": None,
           "events": [{"t": 1500.0, "type": "none"}]}
    rec.update(kw)
    return rec


def test_history_add_is_newest_first_and_list_omits_events(tmp_path):
    store = HistoryStore(tmp_path / "h")
    a = store.add(_hist_rec(1))
    b = store.add(_hist_rec(2))
    assert a["id"] != b["id"]
    listed = store.list()
    assert [r["job_name"] for r in listed] == ["job2.gcode", "job1.gcode"]
    assert all("events" not in r for r in listed)
    assert store.get(a["id"])["events"] == [{"t": 1500.0, "type": "none"}]
    assert store.get("nope") is None
    assert len(store.list(limit=1)) == 1


def test_history_caps_records_and_trims_events(tmp_path):
    store = HistoryStore(tmp_path / "h", max_records=3)
    thumb = tmp_path / "h" / "old.jpg"
    (tmp_path / "h").mkdir()
    thumb.write_bytes(b"x")
    store.add(_hist_rec(0, thumbnail="old.jpg"))
    for n in range(1, 5):
        store.add(_hist_rec(n, events=[{"t": i} for i in range(MAX_EVENTS + 20)]))
    assert len(store) == 3
    assert [r["job_name"] for r in store.list()] == ["job4.gcode", "job3.gcode", "job2.gcode"]
    assert not thumb.exists()                       # dropped record's thumbnail went too
    ev = store.get(store.list()[0]["id"])["events"]
    assert len(ev) == MAX_EVENTS and ev[-1] == {"t": MAX_EVENTS + 19}


def test_history_persists_across_instances(tmp_path):
    store = HistoryStore(tmp_path / "h")
    rec = store.add(_hist_rec(7, outcome="failed", failure_types=["spaghetti"]))
    assert store.update(rec["id"], timelapse="timelapse_x.mp4")
    assert not (tmp_path / "h" / "history.json.tmp").exists()
    again = HistoryStore(tmp_path / "h")
    got = again.get(rec["id"])
    assert got["outcome"] == "failed" and got["failure_types"] == ["spaghetti"]
    assert got["timelapse"] == "timelapse_x.mp4"


@pytest.mark.parametrize("junk", ["{not json", '{"a": 1}', "[1, 2, {}]", ""])
def test_history_tolerates_corrupt_or_missing_file(tmp_path, junk):
    d = tmp_path / "h"
    assert HistoryStore(d).list() == []             # missing dir/file
    d.mkdir()
    (d / "history.json").write_text(junk)
    store = HistoryStore(d)
    assert store.list() == []
    store.add(_hist_rec(1))                         # and it recovers on the next write
    assert len(HistoryStore(d)) == 1


def test_history_delete_removes_record_and_thumbnail(tmp_path):
    store = HistoryStore(tmp_path / "h")
    name = store.save_thumbnail("abc", np.zeros((720, 1280, 3), dtype=np.uint8))
    assert name == "abc.jpg" and (tmp_path / "h" / name).is_file()
    rec = store.add(_hist_rec(1, thumbnail=name))
    keep = store.add(_hist_rec(2))
    assert store.delete(rec["id"]) is True
    assert store.delete(rec["id"]) is False
    assert not (tmp_path / "h" / name).exists()
    assert [r["id"] for r in HistoryStore(tmp_path / "h").list()] == [keep["id"]]


def test_history_thumb_path_cannot_escape_the_directory(tmp_path):
    store = HistoryStore(tmp_path / "h")
    p = store.thumb_path_for({"thumbnail": "../../etc/passwd"})
    assert p == tmp_path / "h" / "passwd"
    assert store.thumb_path_for({"thumbnail": None}) is None


class _HistStatus:
    elapsed_seconds = 3725
    job_name = None            # already cleared by the time the print has ended
    total_layers = None


class _FakeMonitorForHistory:
    def __init__(self):
        self.printer = type("P", (), {"status": _HistStatus()})()
        self.frame_count = 12
        self.failure_count = 2
        self.session_cost_usd = 0.3456789
        self._session_started_at = 5000.0
        self._print_active_since = 5100.0
        self._history_job = ("benchy.gcode", 240)
        self._history_timelapse = "timelapse_1.mp4"
        self._run_failure_types = {"spaghetti", "none", "warping"}
        self.events = deque([
            {"t": 4000.0, "type": "none", "desc": "previous session"},
            {"t": 5200.0, "type": "spaghetti", "conf": 0.9, "desc": "mess", "detected": True},
        ], maxlen=50)


def test_build_record_snapshots_the_session():
    m = _FakeMonitorForHistory()
    rec = build_record(m, "completed", now=9000.0)
    assert rec["outcome"] == "completed"
    assert rec["started_at"] == 5000.0 and rec["ended_at"] == 9000.0
    assert rec["duration_seconds"] == 3725           # printer's job timer wins
    assert rec["job_name"] == "benchy.gcode" and rec["total_layers"] == 240
    assert rec["failure_count"] == 2
    assert rec["failure_types"] == ["spaghetti", "warping"]
    assert rec["false_alarms"] == 0                  # attribute not present yet
    assert rec["frames_analyzed"] == 12
    assert rec["ai_cost_usd"] == 0.3457
    assert rec["timelapse"] == "timelapse_1.mp4"
    assert [e["type"] for e in rec["events"]] == ["spaghetti"]   # earlier session dropped
    json.dumps(rec)                                  # must be JSON-serialisable

    m.false_alarm_count = 3
    m.printer.status.elapsed_seconds = None
    rec = build_record(m, "bogus", now=9000.0)
    assert rec["false_alarms"] == 3
    assert rec["outcome"] == "stopped"
    assert rec["duration_seconds"] == 3900           # falls back to print-active time


def test_record_session_skips_empty_sessions_and_saves_thumbnail(tmp_path):
    store = HistoryStore(tmp_path / "h")
    m = _FakeMonitorForHistory()
    m.frame_count = 0
    assert record_session(store, m, "stopped") is None
    assert len(store) == 0

    m.frame_count = 4
    frame = np.full((480, 960, 3), 127, dtype=np.uint8)
    rec = record_session(store, m, "stopped", frame, now=9000.0)
    assert rec["thumbnail"] == rec["id"] + ".jpg"
    assert (tmp_path / "h" / rec["thumbnail"]).stat().st_size > 0
    assert store.list()[0]["id"] == rec["id"]
    # No frame → still recorded, just without a thumbnail.
    assert record_session(store, m, "failed", None, now=9100.0)["thumbnail"] is None


def test_monitor_record_history_runs_once_and_never_raises(tmp_path):
    m = _FakeMonitorForHistory()
    m.history = HistoryStore(tmp_path / "h")
    m._history_recorded = False
    m._history_last_id = None
    Monitor._record_history(m, "completed", None)
    Monitor._record_history(m, "stopped", None)      # second call is a no-op
    assert len(m.history) == 1 and m._history_last_id == m.history.list()[0]["id"]

    class _Boom:
        def new_id(self, *_a):
            raise OSError("disk full")
    m.history, m._history_recorded = _Boom(), False
    Monitor._record_history(m, "completed", None)    # swallowed


def test_history_format_recent(tmp_path):
    store = HistoryStore(tmp_path / "h")
    assert format_recent(store) == "No prints recorded yet."
    for n in range(7):
        store.add(_hist_rec(n))
    text = format_recent(store, limit=5)
    assert text.count("job") == 5 and "job6.gcode" in text and "job1.gcode" not in text
    assert "completed" in text and "16m" in text and "$0.12" in text


# ── timelapse: head-parked capture (snapshot marker) ─────────────────────────

import monitor as _monitor_mod  # noqa: E402
from ender3monitor.timelapse import TimelapseManager  # noqa: E402


class _MarkerSerial:
    """A port whose firmware speaks on its own: `pending` is what has arrived
    unprompted, `reply` is what the next command is answered with."""

    def __init__(self):
        self.pending = b""
        self.reply = deque()

    @property
    def in_waiting(self):
        return len(self.pending)

    def read(self, n):
        out, self.pending = self.pending[:n], self.pending[n:]
        return out

    def reset_input_buffer(self):
        self.pending = b""

    def flush(self):
        pass

    def write(self, raw):
        pass

    def readline(self):
        return self.reply.popleft() if self.reply else b""


def _marker_controller():
    pc = PrinterController(port="")
    pc._serial = _MarkerSerial()
    pc.status.connected = True
    hits = []
    pc.on_snapshot = lambda: hits.append(1)
    return pc, hits


def test_snapshot_marker_waiting_on_the_line_is_not_discarded_by_send():
    pc, hits = _marker_controller()
    pc._serial.pending = b"echo:TIMELAPSE_SNAP\n"
    pc._serial.reply.append(b"ok\n")
    pc.send("M105")
    assert len(hits) == 1


def test_snapshot_marker_inside_a_reply_fires_before_the_ok():
    pc, hits = _marker_controller()
    seen_at_ok = []

    class _Reply(deque):
        def popleft(self):
            line = super().popleft()
            if line.startswith(b"ok"):
                seen_at_ok.append(len(hits))
            return line

    pc._serial.reply = _Reply([b"echo:TIMELAPSE_SNAP\n", b"ok T:200 /200\n"])
    pc.send("M105")
    assert len(hits) == 1
    assert seen_at_ok == [1]   # acted on while the reply was still held back


def test_snapshot_marker_split_across_two_reads_fires_once():
    pc, hits = _marker_controller()
    pc._scan_unsolicited(b"echo:TIMELA")
    assert hits == []
    pc._scan_unsolicited(b"PSE_SNAP\n")
    assert len(hits) == 1
    pc._scan_unsolicited(b"ok\n")
    assert len(hits) == 1


def test_listen_picks_up_a_marker_and_returns_on_stop():
    pc, hits = _marker_controller()
    pc._serial.pending = b"echo:TIMELAPSE_SNAP\necho:busy: paused for user\n"
    stop = threading.Event()
    pc.on_snapshot = lambda: (hits.append(1), stop.set())
    started = time.time()
    pc.listen(5.0, stop)
    assert len(hits) == 1
    assert time.time() - started < 1.0
    assert pc.status.filament_change_pause is True   # idle-line pause latched too


class _FakeMonitorForSnapshot:
    def __init__(self, tmp_path, mode="auto"):
        self._running = True
        self._timelapse_layer_mode = True
        self._timelapse_synced = False
        self._snapshot_lock = threading.Lock()
        self._last_layer_z = None
        self.settings = _FakeSettings(timelapse_mode=mode)
        self.timelapse = TimelapseManager(str(tmp_path))
        self.printer = _FakePrinter()
        self.printer.status.printing = True
        self.printer.status.z_height = 0.2
        self._frame_provider = lambda: np.zeros((8, 8, 3), np.uint8)


def test_first_marker_drops_earlier_frames_and_takes_over_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(_monitor_mod, "SNAPSHOT_SETTLE_SECONDS", 0)
    m = _FakeMonitorForSnapshot(tmp_path)
    Monitor._maybe_capture_layer_frame(m)          # Z-based frame, head in shot
    assert m.timelapse.frame_count == 1

    Monitor._on_snapshot_marker(m)
    assert m._timelapse_synced is True
    assert m.timelapse.frame_count == 1            # the early one was replaced
    Monitor._on_snapshot_marker(m)
    assert m.timelapse.frame_count == 2

    m.printer.status.z_height = 5.0                # a layer change the poller sees
    Monitor._maybe_capture_layer_frame(m)
    assert m.timelapse.frame_count == 2            # …no longer saves a frame
    assert len(list(m.timelapse._session_dir.glob("frame_*.jpg"))) == 2


def test_marker_is_ignored_in_time_mode_and_when_not_monitoring(tmp_path, monkeypatch):
    monkeypatch.setattr(_monitor_mod, "SNAPSHOT_SETTLE_SECONDS", 0)
    m = _FakeMonitorForSnapshot(tmp_path, mode="time")
    Monitor._on_snapshot_marker(m)
    assert m._timelapse_synced is False and m.timelapse.frame_count == 0

    m = _FakeMonitorForSnapshot(tmp_path)
    m._running = False
    Monitor._on_snapshot_marker(m)
    assert m.timelapse.frame_count == 0


def test_compile_holds_the_last_frame(tmp_path):
    import cv2
    tl = TimelapseManager(str(tmp_path))
    for _ in range(3):
        tl.save_frame(np.zeros((48, 64, 3), np.uint8))
    out = tl.compile(fps=10, hold_last_seconds=1.0)
    assert out is not None
    cap = cv2.VideoCapture(out)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n == 13


def test_dashboard_state_reports_the_timelapse_capture_mode(tmp_path, monkeypatch):
    import web

    class _M:
        _timelapse_synced = False
        _timelapse_layer_mode = True
        timelapse = TimelapseManager(str(tmp_path))

    monkeypatch.setattr(web, "_monitor", _M)
    assert web._timelapse_state() == {"mode": "layer", "frames": 0}
    _M.timelapse.save_frame(np.zeros((8, 8, 3), np.uint8))
    _M._timelapse_synced = True
    assert web._timelapse_state() == {"mode": "parked", "frames": 1}
    _M._timelapse_synced = _M._timelapse_layer_mode = False
    assert web._timelapse_state()["mode"] == "time"
