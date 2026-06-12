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
from dataclasses import dataclass
from typing import Optional

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
    printing: bool = False                  # an SD/USB print is currently active
    progress: Optional[float] = None        # 0..1, SD prints only
    elapsed_seconds: Optional[int] = None   # print job timer (M31)
    remaining_seconds: Optional[int] = None # estimated, from elapsed + progress
    z_height: Optional[float] = None        # current nozzle Z (M114), for first-layer / layer-timelapse
    lifetime_print_seconds: Optional[int] = None  # firmware EEPROM total (M78)

    last_error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "connected": self.connected,
            "port": self.port,
            "nozzle_temp": self.nozzle_temp,
            "nozzle_target": self.nozzle_target,
            "bed_temp": self.bed_temp,
            "bed_target": self.bed_target,
            "printing": self.printing,
            "progress": self.progress,
            "elapsed_seconds": self.elapsed_seconds,
            "remaining_seconds": self.remaining_seconds,
            "z_height": self.z_height,
            "lifetime_print_seconds": self.lifetime_print_seconds,
            "elapsed_str": _fmt_duration(self.elapsed_seconds),
            "remaining_str": _fmt_duration(self.remaining_seconds),
            "lifetime_str": _fmt_duration(self.lifetime_print_seconds),
        }


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


class PrinterController:
    """Thread-safe serial G-code controller. Safe to use even with no printer."""

    def __init__(self, port: str = "", baud: int = 115200) -> None:
        self._port_cfg = port
        self._baud = baud
        self._serial = None
        self._lock = threading.Lock()
        self.status = PrinterStatus()
        self._refresh_count = 0   # paces the slow M78 statistics query

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
                self._serial = serial.Serial(port, self._baud, timeout=2)
                # Marlin resets on connect; give it a moment and drain the banner.
                time.sleep(2.0)
                self._serial.reset_input_buffer()
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

    # ------------------------------------------------------------------ #
    # Low-level command exchange                                           #
    # ------------------------------------------------------------------ #

    def send(self, gcode: str, read_timeout: float = 3.0) -> str:
        """Send one G-code line and collect the reply up to its 'ok'.

        Returns the accumulated response text (may be empty). Never raises —
        on error it records last_error, marks disconnected, and returns "".
        """
        if not self.connected:
            return ""
        try:
            with self._lock:
                self._serial.reset_input_buffer()
                self._serial.write((gcode.strip() + "\n").encode("ascii", "ignore"))
                self._serial.flush()
                deadline = time.time() + read_timeout
                lines = []
                while time.time() < deadline:
                    raw = self._serial.readline()
                    if not raw:
                        continue
                    line = raw.decode("ascii", "ignore").strip()
                    if line:
                        lines.append(line)
                    if line.startswith("ok") or line.lower().startswith("ok"):
                        break
            return "\n".join(lines)
        except Exception as exc:
            self.status.last_error = f"Serial write failed: {exc}"
            self.status.connected = False
            return ""

    # ------------------------------------------------------------------ #
    # Status queries                                                       #
    # ------------------------------------------------------------------ #

    def query_temps(self) -> None:
        """Update status with the latest nozzle/bed temperatures via M105."""
        resp = self.send("M105")
        m = _TEMP_RE.search(resp)
        if m:
            self.status.nozzle_temp = float(m.group(1))
            self.status.nozzle_target = float(m.group(2))
            self.status.bed_temp = float(m.group(3))
            self.status.bed_target = float(m.group(4))

    def query_progress(self) -> None:
        """Update SD-print state + percent via M27 (meaningful for SD/USB prints)."""
        resp = self.send("M27")
        m = _SD_RE.search(resp)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            self.status.printing = True
            self.status.progress = (done / total) if total > 0 else None
        elif "Not SD printing" in resp:
            self.status.printing = False

    def query_print_time(self) -> None:
        """Update elapsed time via M31 and estimate remaining from progress."""
        resp = self.send("M31")
        m = _PRINTTIME_RE.search(resp)
        if m and any(m.groups()):
            h = int(m.group(1) or 0)
            mn = int(m.group(2) or 0)
            s = int(m.group(3) or 0)
            self.status.elapsed_seconds = h * 3600 + mn * 60 + s
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

    def refresh_status(self) -> None:
        """One combined poll: temps, print state/percent, time, and Z height.

        Lifetime statistics (M78) change slowly, so we query them only ~once a
        minute (and on the first refresh after connecting).
        """
        self.query_temps()
        if not self.connected:
            return
        self.query_progress()
        self.query_print_time()
        self.query_position()
        if self._refresh_count % 12 == 0:   # ~every 60 s at a 5 s poll
            self.query_statistics()
        self._refresh_count += 1

    # ------------------------------------------------------------------ #
    # Interventions                                                        #
    # ------------------------------------------------------------------ #

    def pause(self) -> None:
        """Pause an SD print (M25)."""
        self.send("M25")

    def resume(self) -> None:
        """Resume a paused SD print (M24)."""
        self.send("M24")

    def cooldown(self) -> None:
        """Pause, then turn off hotend and bed heaters and the part fan."""
        self.send("M25")        # pause first so the head parks
        self.send("M104 S0")    # hotend off
        self.send("M140 S0")    # bed off
        self.send("M107")       # part fan off

    def emergency_stop(self) -> None:
        """Hard halt (M112). The printer must be power-cycled/reset afterwards."""
        self.send("M112", read_timeout=1.0)

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
