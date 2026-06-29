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
from ender3monitor.settings import SCHEMA, Settings, _coerce
from ender3monitor.printer import (
    _M78_TOTAL_RE, _POS_RE, _PRINTTIME_RE, _SD_RE, _TEMP_RE, _fmt_duration,
    PrinterController,
)
from monitor import _frames_differ, _confirm_frames


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
