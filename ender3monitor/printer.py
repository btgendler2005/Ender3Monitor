"""Serial G-code control for Marlin-based printers (e.g. Ender 3 V3 SE).

The printer exposes a USB serial port that accepts plain-text G-code and
replies "ok". This module wraps that with a small, thread-safe controller used
for two things:

  1. Reading status — nozzle/bed temperatures (M105) and SD print progress (M27).
  2. Intervening on a confirmed failure — pause (M25), cool down, or emergency
     stop (M112).

pyserial is an optional dependency. If it (or the printer) is unavailable, the
controller stays in a disconnected state and every method is a safe no-op, so
the rest of the app runs unchanged.
"""
from __future__ import annotations

import glob
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional

try:
    import serial  # pyserial
    _HAVE_SERIAL = True
except Exception:
    serial = None
    _HAVE_SERIAL = False


# M105 reply looks like:  ok T:24.7 /0.0 B:23.5 /0.0 @:0 B@:0
_TEMP_RE = re.compile(r"T:\s*(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*).*?B:\s*(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)")
# M27 reply looks like:   SD printing byte 1234/56789   (or "Not SD printing")
_SD_RE = re.compile(r"SD printing byte\s+(\d+)\s*/\s*(\d+)")
# M31 reply looks like:   echo:Print time: 1h 23m 45s   (any of h/m/s may be absent)
_PRINTTIME_RE = re.compile(r"Print time:\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?", re.IGNORECASE)
# M114 reply looks like:  X:0.00 Y:0.00 Z:0.20 E:0.00 Count X:0 Y:0 Z:80
_POS_RE = re.compile(r"\bZ:\s*(-?\d+\.?\d*)")
# M78 (print statistics) has a line like:  Total time: 8d 12h 34m 56s
_M78_TOTAL_RE = re.compile(
    r"Total time:\s*(?:(\d+)\s*d)?\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*s)?",
    re.IGNORECASE,
)
# Marlin has no dedicated "M600 happened" reply — but while it's blocked waiting
# on the user (filament/color change, or any other M0/M1-style pause), its
# default-on HOST_KEEPALIVE_FEATURE re-sends this line every ~2s so the host
# doesn't think the connection died. Since gcode processing is fully stalled
# during that wait, it also rides along as unsolicited text inside whatever
# poll command's reply window happens to be open when it's sent (see `send()`).
# Mid-print, this is effectively always an embedded M600 color change.
_PAUSED_FOR_USER_RE = re.compile(r"paused for user", re.IGNORECASE)

# Timelapse snapshot marker. The sliced file parks the head at a fixed spot on
# every layer change, waits for the move to finish (M400) and then announces it
# with `M118 E1 TIMELAPSE_SNAP`, which the firmware echoes on the USB line. That
# is the one moment the head is out of the shot and the bed is in the same place
# as on every other layer, so the monitor photographs exactly then.
SNAPSHOT_MARKER = b"TIMELAPSE_SNAP"
# How often listen() looks at the idle line for that marker (seconds).
_LISTEN_SLICE_SECONDS = 0.03

# ── Serial traffic budget ────────────────────────────────────────────────────
# Every line the host sends lands in the same small command ring buffer that
# the SD reader feeds moves into (BUFSIZE is 4 on many Creality builds). A
# host that polls several commands every few seconds keeps slots occupied, and
# the SD file can't keep the planner fed — motion stops at the end of a line.
# So the controller now *listens* far more than it *asks*:
#   • temperatures and SD progress arrive as auto-reports (M155 / M27 S);
#   • whatever still has to be polled goes out one command at a time, with the
#     next only after the previous "ok", and no faster than
#     `min_command_interval` while a print may be running;
#   • commands that make the firmware drain the motion queue before replying
#     are never sent while a print may be running.
TEMP_AUTOREPORT_SECONDS = 5        # M155 S5
SD_AUTOREPORT_SECONDS = 10         # M27 S10
FALLBACK_POLL_SECONDS = 10         # polling floor when an auto-report is missing
AUTOREPORT_GRACE_INTERVALS = 3     # this many intervals without a report → poll
AUTOREPORT_RETRY_SECONDS = 60      # re-arm a report that stopped (printer reset)
PRINT_TIME_POLL_SECONDS = 60       # M31; elapsed time is extrapolated between
SENSOR_POLL_PRINTING_SECONDS = 30  # M119 during a print
SENSOR_POLL_IDLE_SECONDS = 60
POSITION_POLL_IDLE_SECONDS = 30    # M114 — idle only
STATS_POLL_IDLE_SECONDS = 300      # M78 — idle only
DEFAULT_MIN_COMMAND_INTERVAL = 5.0
IDLE_MIN_COMMAND_INTERVAL = 0.5
DEFAULT_OK_TIMEOUT = 5.0
SLOW_OK_SECONDS = 1.0              # an "ok" slower than this is counted as slow
# A timed-out command's "ok" usually turns up later; it must not be mistaken
# for the next command's. Forget the debt after this long in case it never does.
OK_DEBT_EXPIRY_SECONDS = 30.0

# Commands after which Marlin finishes (or blocks on) the queued motion before
# it replies, or that rewrite settings. Never sent while a print may be running.
MOTION_SYNC_COMMANDS = frozenset({
    "M114",          # some builds (M114_LEGACY, older Creality) planner.synchronize()
    "M400", "G4",    # wait for moves / dwell
    "M503", "M500", "M501", "M502",   # settings report/save/load/reset
    "G28", "G29", "G30", "G34", "M48", "M420",
    "M109", "M190", "M191", "M303",   # wait-for-temperature / autotune
    "M0", "M1", "M600",
})
_UNKNOWN_CMD_RE = re.compile(r"Unknown command", re.IGNORECASE)


def command_code(gcode: str) -> str:
    """The command word of a G-code line: "M155 S5" → "M155"."""
    word = gcode.strip().split(None, 1)[0] if gcode.strip() else ""
    return word.upper()


# ── Prometheus hooks (lazy, so this module imports without prometheus) ────────

def _ops():
    try:
        from ender3monitor import ops_metrics
        return ops_metrics
    except Exception:
        return None


def _metric(name: str, *labels: str) -> None:
    ops = _ops()
    if ops is not None:
        try:
            m = getattr(ops, name)
            (m.labels(*labels) if labels else m).inc()
        except Exception:
            pass


def _observe_ok(code: str, waited: float) -> None:
    ops = _ops()
    if ops is None:
        return
    try:
        ops.printer_ok_wait_seconds.labels(code).observe(waited)
        if waited > SLOW_OK_SECONDS:
            ops.printer_slow_ok_total.labels(code).inc()
    except Exception:
        pass


def _observe_ok_timeout(code: str) -> None:
    _metric("printer_ok_timeouts_total", code)
    _metric("printer_slow_ok_total", code)   # a timeout is the slowest "ok" of all


def _set_autoreport_gauge(kind: str, active: bool) -> None:
    ops = _ops()
    if ops is not None:
        try:
            ops.printer_autoreport_active.labels(kind).set(1 if active else 0)
        except Exception:
            pass

# Explicit filament-runout announcements. There is no single universal line:
# with Marlin's default FILAMENT_RUNOUT_SCRIPT ("M600") the firmware handles a
# runout entirely by itself and says nothing specific about it — over serial it
# is indistinguishable from an embedded color change (both just block waiting on
# the user, see above). Only the host-handled configurations announce the event:
# M412 H1 opens a host prompt, and ACTION_ON_FILAMENT_RUNOUT emits a host action.
# Matched here when present because it is instant and unambiguous; when absent
# the monitor falls back to classifying the pause (see Monitor._classify_runout).
#
# Deliberately narrow: the bare phrase "filament runout" also appears in
# Marlin's own *settings* reports (M412/M503 print "; Filament runout ON"), so
# matching that alone would turn a settings dump into a runout alert.
_RUNOUT_MARKER_RE = re.compile(
    r"action:\s*out_of_filament"
    r"|action:\s*(?:pause\s+)?filament_runout"
    r"|prompt_begin\s+FilamentRunout"
    r"|filament\s*runout\s*(?:detected|triggered)"
    r"|runout\s*detected",
    re.IGNORECASE,
)
# M119 (endstop report) includes a line for the filament sensor when the
# firmware has one configured:   filament: open   /   Filament 1: TRIGGERED
_FILAMENT_SENSOR_RE = re.compile(
    r"^\s*filament[^:\n]*:\s*(open|triggered)\b", re.IGNORECASE | re.MULTILINE)
# Something that marks a reply as a real endstop report rather than whatever
# else landed in the read window. Needed to tell "this firmware reports no
# filament sensor" apart from "we never got the report" (see below).
_ENDSTOP_REPORT_RE = re.compile(
    r"_(?:min|max)\s*:|Reporting endstop status", re.IGNORECASE)

# How many consecutive identical sensor readings are needed before we trust one
# as the "filament present" baseline (see _learn_sensor_polarity).
SENSOR_CALIBRATION_POLLS = 3

# Temperature history kept for the dashboard chart: ~30 minutes at the 5 s
# poll cadence. Bounded both by sample count and by age (see temp_history()).
TEMP_HISTORY_SECONDS = 30 * 60
TEMP_HISTORY_MAX_SAMPLES = 360


def _fmt_duration(seconds: Optional[int]) -> Optional[str]:
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


@dataclass
class PrinterStatus:
    connected: bool = False
    port: Optional[str] = None
    nozzle_temp: Optional[float] = None
    nozzle_target: Optional[float] = None
    bed_temp: Optional[float] = None
    bed_target: Optional[float] = None
    temps_at: Optional[float] = None        # unix time of the last good M105 reading
    printing: bool = False                  # an SD/USB print is currently active
    progress: Optional[float] = None        # 0..1, SD prints only
    elapsed_seconds: Optional[int] = None   # print job timer (M31)
    remaining_seconds: Optional[int] = None # estimated, from elapsed + progress
    z_height: Optional[float] = None        # current nozzle Z (M114), for first-layer / layer-timelapse
    # M114 is only sent while idle. Mid-print, z_height is None unless the
    # monitor estimated it from the matched G-code index (then this is True).
    z_estimated: bool = False
    lifetime_print_seconds: Optional[int] = None  # firmware EEPROM total (M78)
    filament_change_pause: bool = False     # printer is blocked waiting on the user (M600 etc.)
    # Latched by our own pause()/cooldown() and cleared by resume() or the job
    # ending, so the UI can offer one Pause/Resume toggle. A pause made on the
    # printer's own LCD is not visible over serial and is not reflected here.
    paused: bool = False

    # ── Filament runout ──
    # `filament_present` is the runout sensor read over M119 and normalised for
    # polarity (see _learn_sensor_polarity); None means "no sensor, or not yet
    # calibrated", and is never treated as a runout. `runout_marker_seen` is
    # latched when the firmware announces a runout itself. The last two are
    # written by the monitor once it has classified the event, and live here so
    # the UI, Telegram and Prometheus all read runout state from one place.
    filament_present: Optional[bool] = None
    filament_sensor_reported: bool = False  # firmware's M119 reports a filament line
    runout_marker_seen: bool = False        # firmware announced a runout over serial
    filament_runout: bool = False           # a runout is believed to be in progress
    runout_source: Optional[str] = None     # "sensor" | "firmware" | "unplanned-pause"

    # ── Slicer-aware fields ──
    # Populated by the monitor from a matching G-code index (see gcode_index.py)
    # and left as None whenever no index matches, so every consumer degrades to
    # the byte-percentage behaviour rather than showing a confident wrong answer.
    file_size: Optional[int] = None         # M27's total — the file's fingerprint
    file_position: Optional[int] = None     # M27's byte offset into that file
    job_name: Optional[str] = None          # filename the index was built from
    layer: Optional[int] = None             # current layer (1-based)
    total_layers: Optional[int] = None
    feature: Optional[str] = None           # active ;TYPE: e.g. "Outer wall"
    remaining_source: str = "bytes"         # "slicer" | "layers" | "bytes"
    next_color_change_seconds: Optional[int] = None  # until the next embedded M600
    # Where the file's embedded M600s fall, as 0..1 fractions of total layers.
    color_change_marks: List[float] = field(default_factory=list)

    last_error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "connected": self.connected,
            "port": self.port,
            "nozzle_temp": self.nozzle_temp,
            "nozzle_target": self.nozzle_target,
            "bed_temp": self.bed_temp,
            "bed_target": self.bed_target,
            "temps_at": self.temps_at,
            "printing": self.printing,
            "progress": self.progress,
            "elapsed_seconds": self.elapsed_seconds,
            "remaining_seconds": self.remaining_seconds,
            "z_height": self.z_height,
            "z_estimated": self.z_estimated,
            "lifetime_print_seconds": self.lifetime_print_seconds,
            "filament_change_pause": self.filament_change_pause,
            "paused": self.paused,
            "filament_present": self.filament_present,
            "filament_sensor_reported": self.filament_sensor_reported,
            "filament_runout": self.filament_runout,
            "runout_source": self.runout_source,
            "file_size": self.file_size,
            "file_position": self.file_position,
            "job_name": self.job_name,
            "layer": self.layer,
            "total_layers": self.total_layers,
            "feature": self.feature,
            "remaining_source": self.remaining_source,
            "next_color_change_seconds": self.next_color_change_seconds,
            "color_change_marks": list(self.color_change_marks),
            "layer_str": (f"{self.layer}/{self.total_layers}"
                          if self.layer and self.total_layers else None),
            "elapsed_str": _fmt_duration(self.elapsed_seconds),
            "remaining_str": _fmt_duration(self.remaining_seconds),
            "lifetime_str": _fmt_duration(self.lifetime_print_seconds),
            "next_color_change_str": _fmt_duration(self.next_color_change_seconds),
        }

    def clear_gcode_context(self) -> None:
        """Drop everything derived from a G-code index (no match, or job ended)."""
        self.job_name = None
        self.layer = None
        self.total_layers = None
        self.feature = None
        self.remaining_source = "bytes"
        self.next_color_change_seconds = None
        self.color_change_marks = []


def autodetect_port() -> Optional[str]:
    """Best-effort guess at the printer's serial device across platforms.

    On macOS the cu.* node is preferred over tty.* — tty.* can block on open
    waiting for a carrier-detect signal the printer never asserts. Re-globbing
    each time also means a replug that renames the device (e.g. usbserial-1210
    → usbserial-1310) is handled transparently, including on reconnect.
    """
    patterns = [
        "/dev/cu.usbserial*", "/dev/cu.usbmodem*", "/dev/cu.wchusbserial*",      # macOS (preferred)
        "/dev/tty.usbserial*", "/dev/tty.usbmodem*", "/dev/tty.wchusbserial*",   # macOS fallback
        "/dev/ttyUSB*", "/dev/ttyACM*",                                          # Linux
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    if sys.platform.startswith("win"):
        return "COM3"   # common default; user should set PRINTER_PORT explicitly
    return None


def _open_without_reset(port: str, baud: int):
    """Open the port without producing a DTR/RTS edge from our side.

    On boards that wire DTR to the MCU reset (classic AVR Marlin), it is the
    *transition* that resets the board. The OS asserts DTR and RTS when the
    port opens; pyserial then applies its own dtr/rts settings, so asking for
    anything other than "asserted" would itself toggle the line. We ask for
    asserted (= what the OS just did → no change), then clear HUPCL so closing
    the port — a reconnect, or the stall watchdog force-closing it — leaves
    DTR asserted instead of dropping it. The next open then finds DTR already
    asserted and there is no edge at all.

    What this cannot prevent: the very first open after the USB cable is
    plugged in (DTR goes from never-asserted to asserted). Connect while the
    printer is idle; once connected, reconnects are edge-free.
    """
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 2
    ser.dtr = True
    ser.rts = True
    ser.open()
    _clear_hupcl(ser)
    return ser


def _clear_hupcl(ser) -> None:
    """Keep DTR asserted when the port closes (POSIX only). Never raises."""
    try:
        import termios
        fd = ser.fileno()
        attrs = termios.tcgetattr(fd)
        if attrs[2] & termios.HUPCL:
            attrs[2] &= ~termios.HUPCL
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        pass   # Windows / no termios: nothing to clear


class PrinterController:
    """Thread-safe serial G-code controller. Safe to use even with no printer."""

    def __init__(self, port: str = "", baud: int = 115200,
                 min_command_interval: float = DEFAULT_MIN_COMMAND_INTERVAL,
                 ok_timeout: float = DEFAULT_OK_TIMEOUT,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._port_cfg = port
        self._baud = baud
        self._serial = None
        self._lock = threading.Lock()
        self.status = PrinterStatus()
        # Traffic budget (see the block above MOTION_SYNC_COMMANDS).
        self.min_command_interval = max(0.0, float(min_command_interval))
        self.ok_timeout = max(0.5, float(ok_timeout))
        self._clock = clock
        self._last_cmd_at: Optional[float] = None
        self._sent_times: deque = deque(maxlen=512)   # for commands/minute
        self._ok_debt = 0
        self._ok_debt_until = 0.0
        self._rx_buf = b""          # partial line from a passive read
        self._reset_link_state()
        # Rolling temperature history for the dashboard chart. Written by the
        # poll thread, read by web requests — hence its own small lock (never
        # the serial lock, which can be held for seconds during a slow reply).
        self._temp_history: deque = deque(maxlen=TEMP_HISTORY_MAX_SAMPLES)
        self._temp_history_lock = threading.Lock()
        # Hysteresis for clearing filament_change_pause: the busy-keepalive line
        # only arrives every ~2s, so a single reply window without it doesn't
        # prove the pause ended — require a couple of clean replies in a row.
        self._pause_clear_streak = 0
        # Timelapse snapshot marker (see SNAPSHOT_MARKER). on_snapshot is called
        # from whichever thread read the marker, the moment it is read.
        self.on_snapshot: Optional[Callable[[], None]] = None
        self._rx_tail = b""   # end of the last unsolicited chunk (split markers)
        # Runout-sensor polarity calibration (see _learn_sensor_polarity).
        self._sensor_present_state: Optional[bool] = None
        self._sensor_calibration_value: Optional[bool] = None
        self._sensor_calibration_streak = 0

        # Stall watchdog: some USB-serial adapters wedge a blocking read past
        # its configured timeout (observed in the field — readline() never
        # returned instead of erroring after ~2-4s), which freezes every poll
        # forever since Python can't interrupt a blocked thread from outside.
        # _call_started_at marks an in-flight send(); a separate watchdog
        # thread force-closes the port if one runs suspiciously long.
        self._call_started_at: Optional[float] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_stop: Optional[threading.Event] = None

    # ------------------------------------------------------------------ #
    # Connection                                                           #
    # ------------------------------------------------------------------ #

    def connect(self) -> bool:
        """Open the serial port. Returns True on success. Never raises."""
        if not _HAVE_SERIAL:
            self.status.last_error = "pyserial not installed (pip install pyserial)"
            return False
        if not self._port_cfg or self._port_cfg.lower() in ("none", "off", ""):
            return False   # printer control disabled by config

        port = autodetect_port() if self._port_cfg.lower() == "auto" else self._port_cfg
        if not port:
            self.status.last_error = "No serial port found (set PRINTER_PORT)."
            return False

        try:
            with self._lock:
                self._serial = _open_without_reset(port, self._baud)
                # Boards that do reset on open print a banner; give it a moment
                # and drop it. Nothing here writes to the printer.
                time.sleep(2.0)
                self._serial.reset_input_buffer()
                self._rx_buf = b""
                self._ok_debt = 0
            self._reset_link_state()
            self.status.connected = True
            self.status.port = port
            self.status.last_error = None
            return True
        except Exception as exc:
            self.status.connected = False
            self.status.last_error = f"Could not open {port}: {exc}"
            self._serial = None
            return False

    def disconnect(self) -> None:
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:
                    pass
            self._serial = None
        self.status.connected = False

    @property
    def connected(self) -> bool:
        return self.status.connected and self._serial is not None

    def _reset_link_state(self) -> None:
        """Forget what we learned about the firmware's reporting (new link).

        Auto-report settings live in the printer's RAM, so every (re)connect
        asks for them again, and until the first SD status arrives we don't
        know whether a print is running — which counts as "maybe printing".
        """
        # Per report kind: None = not asked yet, True = reports seen,
        # False = unsupported ("Unknown command", or asked and never reported).
        self._autoreport = {"temp": None, "sd": None}
        self._autoreport_asked_at = {"temp": None, "sd": None}
        self._autoreport_seen_at = {"temp": None, "sd": None}
        self._last_poll: dict = {}
        self._print_state_known = False
        self._elapsed_anchor: Optional[tuple] = None   # (seconds, clock())

    # ------------------------------------------------------------------ #
    # Stall watchdog                                                       #
    # ------------------------------------------------------------------ #

    def watchdog_tick(self, stall_seconds: float = 20.0) -> bool:
        """Force-close the port if a send() call has been stuck this long.

        Deliberately does NOT take `self._lock` — the whole point is to
        break a call that's wedged while holding it. Reads/writes `_serial`
        without the lock, which is an intentional, accepted race: the
        alternative is a poll thread stuck forever inside a blocking read
        that never honors its own timeout (seen in the field with some
        USB-serial adapters). Closing the fd out from under that call
        typically makes the blocked read error out, freeing the poll thread
        to notice the disconnect and reconnect on its next cycle.

        Returns True if it force-closed a stalled connection.
        """
        started = self._call_started_at
        if started is None or (time.time() - started) < stall_seconds:
            return False
        ser = self._serial
        self._serial = None
        self.status.connected = False
        self.status.last_error = f"Serial call stalled >{stall_seconds:.0f}s — watchdog force-closed the port"
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        try:
            from ender3monitor import ops_metrics as _ops
            _ops.printer_watchdog_stalls_total.inc()
        except Exception:
            pass
        return True

    def start_watchdog(self, stall_seconds: float = 20.0, check_interval: float = 5.0) -> None:
        """Start the background thread that runs watchdog_tick() periodically.

        Idempotent — safe to call more than once (e.g. from tests or a
        restart path); later calls are no-ops while a watchdog is running.
        """
        if self._watchdog_thread is not None:
            return
        self._watchdog_stop = threading.Event()

        def _loop() -> None:
            while not self._watchdog_stop.wait(check_interval):
                self.watchdog_tick(stall_seconds)

        self._watchdog_thread = threading.Thread(target=_loop, daemon=True)
        self._watchdog_thread.start()

    def stop_watchdog(self) -> None:
        if self._watchdog_thread is None:
            return
        self._watchdog_stop.set()
        self._watchdog_thread.join(timeout=2)
        self._watchdog_thread = None

    # ------------------------------------------------------------------ #
    # Low-level command exchange                                           #
    # ------------------------------------------------------------------ #

    def may_be_printing(self) -> bool:
        """True unless we positively know the printer is idle.

        Right after connecting — before the first SD status — a print may well
        be running, so the strict mid-print rules apply until we know better.
        """
        st = self.status
        return st.printing or st.filament_change_pause or not self._print_state_known

    def _command_gap(self) -> float:
        return (self.min_command_interval if self.may_be_printing()
                else IDLE_MIN_COMMAND_INTERVAL)

    def can_send(self, now: Optional[float] = None) -> bool:
        """Whether the rate limit allows a routine command right now."""
        now = self._clock() if now is None else now
        last = self._last_cmd_at
        return last is None or (now - last) >= self._command_gap()

    def commands_last_minute(self, now: Optional[float] = None) -> int:
        now = self._clock() if now is None else now
        return sum(1 for t in self._sent_times if now - t <= 60.0)

    def send(self, gcode: str, read_timeout: Optional[float] = None,
             priority: bool = False) -> str:
        """Send one G-code line and collect the reply up to its 'ok'.

        Exactly one command is ever in flight: the serial lock is held from the
        write until the "ok" (or the timeout), so a second caller waits.
        Routine commands also respect the rate limit; `priority` is for the
        operator's interventions (pause/resume/cool-down/e-stop), which must
        never be held back by a status poll. Commands that drain the motion
        queue are refused outright while a print may be running.

        Returns the accumulated response text (may be empty). Never raises —
        on error it records last_error, marks disconnected, and returns "".
        """
        if not self.connected:
            return ""
        code = command_code(gcode)
        if code in MOTION_SYNC_COMMANDS and self.may_be_printing():
            _metric("printer_commands_suppressed_total", code, "motion_sync")
            return ""
        timeout = self.ok_timeout if read_timeout is None else read_timeout
        self._call_started_at = time.time()   # cleared in `finally`; watchdog polls this
        try:
            with self._lock:
                if not priority and not self.can_send():
                    _metric("printer_commands_suppressed_total", code, "rate_limit")
                    return ""
                # Lines the firmware sent on its own since the last exchange
                # (auto-reports, the snapshot marker, a late "ok") are handled
                # first — never thrown away.
                self._consume_passive(self._read_pending())
                self._serial.write((gcode.strip() + "\n").encode("ascii", "ignore"))
                self._serial.flush()
                sent_at = self._clock()
                self._last_cmd_at = sent_at
                self._sent_times.append(sent_at)
                _metric("printer_commands_sent_total", code)
                deadline = time.time() + timeout
                lines = []
                got_ok = False
                while time.time() < deadline:
                    raw = self._serial.readline()
                    if not raw:
                        continue
                    # Acted on here, not after the 'ok': while the head
                    # dwells at the park spot the firmware holds our reply
                    # back, and by then the head has moved on.
                    self._scan_unsolicited(raw)
                    if self._rx_buf:            # finish a line a passive read split
                        raw, self._rx_buf = self._rx_buf + raw, b""
                    line = raw.decode("ascii", "ignore").strip()
                    if not line:
                        continue
                    if line[:2].lower() == "ok" and self._take_ok_debt():
                        continue                # a timed-out command's late "ok"
                    lines.append(line)
                    self._handle_report_line(line, solicited=code)
                    if line[:2].lower() == "ok":
                        got_ok = True
                        break
                waited = self._clock() - sent_at
                if got_ok:
                    _observe_ok(code, waited)
                else:
                    _observe_ok_timeout(code)
                    self._ok_debt = min(self._ok_debt + 1, 3)
                    self._ok_debt_until = self._clock() + OK_DEBT_EXPIRY_SECONDS
            resp = "\n".join(lines)
            self._note_pause_marker(resp)
            self._note_runout_marker(resp)
            return resp
        except Exception as exc:
            self.status.last_error = f"Serial write failed: {exc}"
            self.status.connected = False
            try:
                from ender3monitor import ops_metrics as _ops
                _ops.printer_serial_errors_total.inc()
            except Exception:
                pass
            return ""
        finally:
            self._call_started_at = None

    def _read_pending(self) -> bytes:
        """Bytes already waiting on the port, without blocking. Lock held."""
        ser = self._serial
        n = getattr(ser, "in_waiting", 0) if ser is not None else 0
        if not isinstance(n, int) or n <= 0:
            return b""
        return ser.read(n) or b""

    def _scan_unsolicited(self, data: bytes) -> None:
        """Look for the timelapse snapshot marker in raw bytes off the wire.

        Keeps the tail of the previous chunk so a marker split across two reads
        is still seen, and fires on_snapshot once per marker. Never raises.
        """
        if not data:
            return
        buf = self._rx_tail + data
        hits = buf.count(SNAPSHOT_MARKER)
        if hits:
            buf = buf[buf.rfind(SNAPSHOT_MARKER) + len(SNAPSHOT_MARKER):]
        self._rx_tail = buf[-(len(SNAPSHOT_MARKER) - 1):]
        cb = self.on_snapshot
        if hits and cb is not None:
            try:
                cb()
            except Exception as exc:
                print(f"\n  [PRINTER] snapshot callback failed: {exc}")

    def _take_ok_debt(self) -> bool:
        """Consume one owed "ok" (from a timed-out command), if any is owed.

        Marlin answers strictly in order, so the first "ok" after a timeout
        belongs to the command that timed out, not to the one now waiting.
        """
        if self._ok_debt <= 0:
            return False
        if self._clock() > self._ok_debt_until:
            self._ok_debt = 0       # never arrived; stop discarding real oks
            return False
        self._ok_debt -= 1
        return True

    def _consume_passive(self, data: bytes) -> None:
        """Handle bytes the firmware sent on its own. Lock held. Never sends.

        This is the whole reader: auto-reports update status, the snapshot
        marker fires the timelapse callback, and pause/runout announcements
        are latched. Nothing read here ever causes a command to be written.
        """
        if not data:
            return
        self._scan_unsolicited(data)
        buf = self._rx_buf + data
        *complete, self._rx_buf = buf.split(b"\n")
        if len(self._rx_buf) > 512:       # no newline in sight — drop junk
            self._rx_buf = b""
        for raw in complete:
            line = raw.decode("ascii", "ignore").strip()
            if not line:
                continue
            if line[:2].lower() == "ok":
                self._take_ok_debt()
                continue
            self._handle_report_line(line)
            # Latch-only: an idle line can announce a pause or a runout, but
            # silence here proves nothing.
            if _PAUSED_FOR_USER_RE.search(line):
                self._note_pause_marker(line)
            self._note_runout_marker(line)

    def _handle_report_line(self, line: str, solicited: Optional[str] = None) -> None:
        """Apply one status line, whether auto-reported or inside a reply.

        `solicited` is the command whose reply window the line arrived in: a
        temperature or SD line answering our own M105/M27 poll updates status
        but is not evidence that the auto-report is working.
        """
        if line[:2].lower() == "ok":
            return   # "ok T:…" is an M105 reply; query_temps() parses it
        now = self._clock()
        m = _TEMP_RE.search(line)
        if m and "T:" in line:
            self._apply_temps(m)
            if solicited != "M105":
                self._note_autoreport("temp", now)
            return
        if _SD_RE.search(line) or "Not SD printing" in line or "Done printing" in line:
            self._apply_sd_status(line)
            if solicited != "M27":
                self._note_autoreport("sd", now)

    def _note_autoreport(self, kind: str, now: float) -> None:
        self._autoreport_seen_at[kind] = now
        if self._autoreport[kind] is not True:
            self._autoreport[kind] = True
            _set_autoreport_gauge(kind, True)

    def autoreport_active(self, kind: str, now: Optional[float] = None) -> bool:
        """A report of this kind arrived within its grace window."""
        now = self._clock() if now is None else now
        seen = self._autoreport_seen_at.get(kind)
        interval = TEMP_AUTOREPORT_SECONDS if kind == "temp" else SD_AUTOREPORT_SECONDS
        return seen is not None and now - seen <= interval * AUTOREPORT_GRACE_INTERVALS

    def listen(self, duration: float, stop: Optional[threading.Event] = None,
               slice_seconds: float = _LISTEN_SLICE_SECONDS) -> None:
        """Passively read the serial line for `duration` seconds.

        This is where auto-reports and the timelapse marker arrive. A short
        slice keeps the marker latency to a few hundredths of a second while a
        timelapse is being captured; callers pass a longer one otherwise to
        save CPU (the OS buffers the bytes in the meantime). Only ever takes
        the serial lock for a non-blocking read, so commands from other
        threads (pause, e-stop) are not held up. Never sends. Never raises.
        """
        deadline = time.time() + duration
        while self.connected:
            try:
                with self._lock:
                    self._consume_passive(self._read_pending())
            except Exception:
                break   # send() owns error reporting; just stop listening
            remaining = deadline - time.time()
            if remaining <= 0 or (stop is not None and stop.is_set()):
                return
            pause = min(slice_seconds, remaining)
            if stop is not None:
                stop.wait(pause)
            else:
                time.sleep(pause)
        remaining = deadline - time.time()
        if remaining > 0:
            stop.wait(remaining) if stop is not None else time.sleep(remaining)

    def _note_pause_marker(self, resp: str) -> None:
        """Latch/clear filament_change_pause from a raw serial reply.

        Set the instant we see the busy-keepalive line; only clear after a
        couple of marker-free, non-empty replies in a row, since a single quiet
        window doesn't prove the printer stopped waiting (see the field note
        above `_PAUSED_FOR_USER_RE`) — an empty (timed-out) reply proves nothing
        either way and is ignored.
        """
        if _PAUSED_FOR_USER_RE.search(resp):
            self.status.filament_change_pause = True
            self._pause_clear_streak = 0
        elif resp and self.status.filament_change_pause:
            self._pause_clear_streak += 1
            if self._pause_clear_streak >= 2:
                self.status.filament_change_pause = False
                self._pause_clear_streak = 0
                # The pause is over, so whatever announced it is history too.
                self.status.runout_marker_seen = False

    def _note_runout_marker(self, resp: str) -> None:
        """Latch a firmware runout announcement from a raw serial reply.

        Only latches — clearing is deliberate, because the announcement arrives
        once while the condition persists. It is cleared when the pause it
        caused ends (above), when the print ends, or by the monitor when it
        closes out the runout episode.
        """
        if _RUNOUT_MARKER_RE.search(resp):
            self.status.runout_marker_seen = True

    def clear_runout_marker(self) -> None:
        """Forget a latched firmware runout announcement (episode over)."""
        self.status.runout_marker_seen = False

    # ------------------------------------------------------------------ #
    # Status queries                                                       #
    # ------------------------------------------------------------------ #

    def query_temps(self) -> None:
        """Update status with the latest nozzle/bed temperatures via M105."""
        resp = self.send("M105")
        for line in resp.splitlines():
            if line[:2].lower() == "ok":     # other lines were applied by send()
                m = _TEMP_RE.search(line)
                if m:
                    self._apply_temps(m)

    def _apply_temps(self, m) -> None:
        self.status.nozzle_temp = float(m.group(1))
        self.status.nozzle_target = float(m.group(2))
        self.status.bed_temp = float(m.group(3))
        self.status.bed_target = float(m.group(4))
        self._record_temps()

    def _record_temps(self, now: Optional[float] = None) -> None:
        """Append the current reading to the bounded temperature history."""
        st = self.status
        if st.nozzle_temp is None or st.bed_temp is None:
            return
        ts = round(time.time() if now is None else now, 1)
        if now is None and st.temps_at is not None and ts - st.temps_at < 1.0:
            # Same reading seen twice (an auto-report inside a poll window).
            with self._temp_history_lock:
                if self._temp_history:
                    self._temp_history[-1] = [
                        ts, st.nozzle_temp, st.nozzle_target, st.bed_temp, st.bed_target]
                    st.temps_at = ts
                    return
        st.temps_at = ts
        with self._temp_history_lock:
            self._temp_history.append(
                [ts, st.nozzle_temp, st.nozzle_target, st.bed_temp, st.bed_target])

    def temp_history(self, now: Optional[float] = None) -> List[list]:
        """Recent samples as [unix_ts, nozzle, nozzle_target, bed, bed_target].

        Oldest first, limited to the last TEMP_HISTORY_SECONDS so a long
        disconnect doesn't leave stale readings in the window. Returns copies,
        so callers can't mutate the buffer.
        """
        cutoff = (time.time() if now is None else now) - TEMP_HISTORY_SECONDS
        with self._temp_history_lock:
            return [list(row) for row in self._temp_history if row[0] >= cutoff]

    def query_progress(self) -> None:
        """Update SD-print state + percent via M27 (meaningful for SD/USB prints).

        Completion is recognised from either "Not SD printing" or Marlin's
        end-of-print "Done printing file" line. The latter matters because a
        finished print can report "Done printing file" before the next poll's
        "Not SD printing"; without it the `printing` flag could stay stuck True
        and the falling-edge completion in monitor.py would never fire.
        """
        self._apply_sd_status(self.send("M27"))

    def _apply_sd_status(self, resp: str) -> None:
        """Apply an SD status report — an M27 reply or an M27 S auto-report."""
        was_printing = self.status.printing
        m = _SD_RE.search(resp)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            self.status.progress = (done / total) if total > 0 else None
            # The byte offset and the file's total size are the only handles the
            # host gets on *which* file is running and where in it we are — the
            # G-code index keys off exactly these (see gcode_index.py).
            self.status.file_position = done
            self.status.file_size = total or None
            # A "byte N/N" reading alongside "Done printing file" means finished.
            self.status.printing = "Done printing" not in resp
        elif "Not SD printing" in resp or "Done printing" in resp:
            self.status.printing = False
            self.status.paused = False
            self.status.file_position = None
            self.status.file_size = None
            # No job, no runout — don't carry an announcement into the next print.
            self.status.runout_marker_seen = False
            self._elapsed_anchor = None
        else:
            return
        self._print_state_known = True
        if self.status.printing and not was_printing:
            # M114 is not sent mid-print, so the last idle Z (often the parked
            # height of the previous job) would be stale for the whole print.
            # The monitor fills in an estimate from the G-code index instead.
            self.status.z_height = None
            self.status.z_estimated = False

    def query_print_time(self) -> None:
        """Update elapsed time via M31 and estimate remaining from progress."""
        resp = self.send("M31")
        m = _PRINTTIME_RE.search(resp)
        if m and any(m.groups()):
            h = int(m.group(1) or 0)
            mn = int(m.group(2) or 0)
            s = int(m.group(3) or 0)
            self.status.elapsed_seconds = h * 3600 + mn * 60 + s
            self._elapsed_anchor = (self.status.elapsed_seconds, self._clock())
        self._update_remaining()

    def _update_remaining(self) -> None:
        # Estimate remaining: linear projection once we're a little way in.
        el, pct = self.status.elapsed_seconds, self.status.progress
        if self.status.printing and el is not None and pct is not None and pct > 0.02:
            self.status.remaining_seconds = int(el * (1.0 / pct - 1.0))
        elif not self.status.printing:
            self.status.remaining_seconds = None

    def query_position(self) -> None:
        """Update current nozzle Z height via M114 (for first-layer / layer timelapse)."""
        resp = self.send("M114")
        m = _POS_RE.search(resp)
        if m:
            self.status.z_height = float(m.group(1))
            self.status.z_estimated = False

    def query_filament_sensor(self) -> None:
        """Read the filament runout sensor via M119 (endstop states).

        Marlin reports the runout switch alongside the endstops when one is
        configured. Firmware without a sensor simply omits the line, which
        leaves `filament_present` at None — unknown, never "ran out".
        """
        resp = self.send("M119")
        m = _FILAMENT_SENSOR_RE.search(resp)
        if not m:
            # Only an identifiable endstop report proves there is no sensor.
            # Anything else means the report never arrived — while the printer
            # is blocked waiting on the user, gcode processing stalls and the
            # read window fills with busy-keepalive lines instead (see
            # _PAUSED_FOR_USER_RE). Blanking the reading on that would erase a
            # just-detected runout at the one moment it matters.
            if _ENDSTOP_REPORT_RE.search(resp):
                self.status.filament_sensor_reported = False
                self.status.filament_present = None
            return
        self.status.filament_sensor_reported = True
        raw = m.group(1).lower() == "triggered"
        self._learn_sensor_polarity(raw)
        self.status.filament_present = (
            None if self._sensor_present_state is None
            else raw == self._sensor_present_state
        )

    def _learn_sensor_polarity(self, raw: bool) -> None:
        """Learn which raw M119 state means "filament is present".

        Nominally Marlin prints TRIGGERED when filament is threaded, but the
        meaning is a build-time choice (FIL_RUNOUT_STATE, exposed by firmware
        builders as separate HIGH/LOW options) and the switch can be wired
        either way, so a hard-coded polarity is wrong on a real fraction of
        machines — and wrong here means either silence on a real runout or an
        alert on every healthy print. So calibrate instead: while a print is
        genuinely running and not blocked waiting on the user, filament is by
        definition loaded, so whatever the sensor reads then is the "present"
        state. A stable run of identical readings is required so one glitch
        can't set the baseline backwards.
        """
        if self._sensor_present_state is not None:
            return
        if not self.status.printing or self.status.filament_change_pause:
            # Idle or parked mid-swap — tells us nothing about the loaded state.
            self._sensor_calibration_value = None
            self._sensor_calibration_streak = 0
            return
        if raw == self._sensor_calibration_value:
            self._sensor_calibration_streak += 1
        else:
            self._sensor_calibration_value = raw
            self._sensor_calibration_streak = 1
        if self._sensor_calibration_streak >= SENSOR_CALIBRATION_POLLS:
            self._sensor_present_state = raw

    def query_statistics(self) -> None:
        """Read the firmware's lifetime total print time via M78 (EEPROM stats).

        Not all builds support M78; if unsupported the value just stays None.
        """
        resp = self.send("M78")
        m = _M78_TOTAL_RE.search(resp)
        if m and any(m.groups()):
            d = int(m.group(1) or 0)
            h = int(m.group(2) or 0)
            mn = int(m.group(3) or 0)
            s = int(m.group(4) or 0)
            self.status.lifetime_print_seconds = d * 86400 + h * 3600 + mn * 60 + s

    # ------------------------------------------------------------------ #
    # Scheduler                                                            #
    # ------------------------------------------------------------------ #

    def _request_autoreport(self, kind: str) -> None:
        """Ask the firmware to push this report on its own (M155 / M27 S)."""
        gcode = (f"M155 S{TEMP_AUTOREPORT_SECONDS}" if kind == "temp"
                 else f"M27 S{SD_AUTOREPORT_SECONDS}")
        resp = self.send(gcode)
        if not resp and not self.connected:
            return
        self._autoreport_asked_at[kind] = self._clock()
        if _UNKNOWN_CMD_RE.search(resp):
            self._autoreport[kind] = False
            _set_autoreport_gauge(kind, False)
            print(f"  [PRINTER] Firmware has no {gcode.split()[0]} auto-report — "
                  f"polling every {FALLBACK_POLL_SECONDS}s instead.")
        elif self._autoreport[kind] is None:
            self._autoreport[kind] = "asked"

    def _autoreport_due(self, kind: str, now: float):
        """Should we (re)arm this auto-report now?"""
        state = self._autoreport[kind]
        if state is False or self.autoreport_active(kind, now):
            return False
        asked = self._autoreport_asked_at[kind]
        if state is None:
            return True
        interval = TEMP_AUTOREPORT_SECONDS if kind == "temp" else SD_AUTOREPORT_SECONDS
        grace = interval * AUTOREPORT_GRACE_INTERVALS
        if state == "asked" and asked is not None and now - asked > grace:
            # Accepted but silent: e.g. M27 S on a build without
            # AUTO_REPORT_SD_STATUS just answers once and never again.
            self._autoreport[kind] = False
            _set_autoreport_gauge(kind, False)
            print(f"  [PRINTER] No {kind} auto-reports arrived — "
                  f"polling every {FALLBACK_POLL_SECONDS}s instead.")
            return False
        # Worked before, now stale (printer reset over a live USB link): re-arm.
        return state is True and (asked is None or now - asked >= AUTOREPORT_RETRY_SECONDS)

    def _needs_poll(self, kind: str, now: float) -> bool:
        """The auto-report for `kind` is known not to be carrying the data."""
        state = self._autoreport[kind]
        return state is not None and state != "asked" and not self.autoreport_active(kind, now)

    def _next_command(self, now: float):
        """The single most useful command to send now, or None.

        Arming an auto-report comes first (it saves every poll after it). Among
        the polls that are due, the most overdue relative to its own period
        wins, so when the budget is tight (no auto-reports at all, 1 command
        per 5 s) every poll still gets its turn — each just slips a little
        later, never sooner, than its period. Ties go to list order: print
        state, temperatures, then the slow-changing extras.
        """
        st = self.status
        for kind in ("sd", "temp"):
            if self._autoreport_due(kind, now):
                return f"{kind}-auto", (lambda k=kind: self._request_autoreport(k))

        polls = []
        if self._needs_poll("sd", now):
            polls.append(("M27", FALLBACK_POLL_SECONDS, self.query_progress))
        if self._needs_poll("temp", now):
            polls.append(("M105", FALLBACK_POLL_SECONDS, self.query_temps))
        if st.printing:
            polls.append(("M31", PRINT_TIME_POLL_SECONDS, self.query_print_time))
        # Runout sensor: skipped while blocked waiting on the user — the reply
        # would not reach us (see query_filament_sensor), and filament is
        # legitimately out of the sensor during a color-change swap.
        if not st.filament_change_pause and self._print_state_known:
            polls.append(("M119", SENSOR_POLL_PRINTING_SECONDS if st.printing
                          else SENSOR_POLL_IDLE_SECONDS, self.query_filament_sensor))
        if not self.may_be_printing():
            polls.append(("M114", POSITION_POLL_IDLE_SECONDS, self.query_position))
            polls.append(("M78", STATS_POLL_IDLE_SECONDS, self.query_statistics))

        best, best_score = None, 1.0
        for key, every, fn in polls:
            last = self._last_poll.get(key)
            score = float("inf") if last is None else (now - last) / every
            if score >= best_score and (best is None or score > best_score):
                best, best_score = (key, fn), score
        return best

    def refresh_status(self) -> None:
        """One scheduler tick: send at most what the traffic budget allows.

        While a print may be running that is at most one command, and only if
        `min_command_interval` has passed since the last one. While idle the
        due commands go out back to back (still one in flight at a time).
        Temperatures and SD progress normally arrive by themselves (see
        listen()), so most ticks during a print send nothing at all.
        """
        if not self.connected:
            return
        for _ in range(8):
            now = self._clock()
            if not self.can_send(now):
                break
            nxt = self._next_command(now)
            if nxt is None:
                break
            key, fn = nxt
            self._last_poll[key] = now
            fn()
            if not self.connected or self.may_be_printing():
                break
        self._extrapolate_elapsed()

    def _extrapolate_elapsed(self) -> None:
        """Advance the print timer between the (now rare) M31 readings."""
        anchor = self._elapsed_anchor
        if anchor is not None and self.status.printing:
            secs, at = anchor
            self.status.elapsed_seconds = int(secs + max(0.0, self._clock() - at))
        self._update_remaining()

    # ------------------------------------------------------------------ #
    # Interventions                                                        #
    # ------------------------------------------------------------------ #

    def pause(self) -> None:
        """Pause an SD print (M25)."""
        self.send("M25", priority=True)
        self.status.paused = True

    def resume(self) -> None:
        """Resume a paused SD print (M24)."""
        self.send("M24", priority=True)
        self.status.paused = False

    def cooldown(self) -> None:
        """Pause, then turn off hotend and bed heaters and the part fan."""
        self.send("M25", priority=True)        # pause first so the head parks
        self.send("M104 S0", priority=True)    # hotend off
        self.send("M140 S0", priority=True)    # bed off
        self.send("M107", priority=True)       # part fan off
        self.status.paused = True

    def emergency_stop(self) -> None:
        """Hard halt (M112). The printer must be power-cycled/reset afterwards."""
        self.send("M112", read_timeout=1.0, priority=True)

    def apply_failure_action(self, action: str) -> str:
        """Run the configured intervention. Returns a short human-readable result."""
        if not self.connected:
            return "printer not connected"
        action = (action or "pause").lower()
        if action == "cooldown":
            self.cooldown()
            return "paused and heaters off"
        if action == "estop":
            self.emergency_stop()
            return "emergency stop sent"
        self.pause()
        return "print paused"
