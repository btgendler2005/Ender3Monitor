#!/usr/bin/env python3
"""Ender3Monitor – 3D print failure detection system."""

import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

# A frame provider returns the latest camera frame (already oriented/flipped)
# or None if no frame is available yet. Used by the web UI to share its single
# persistent capture thread with the analysis loop instead of opening the
# camera a second time.
FrameProvider = Callable[[], Optional[np.ndarray]]

from ender3monitor.config import Config
from ender3monitor.camera import CameraManager
from ender3monitor.analyzer import create_analyzer, AnalysisResult
from ender3monitor.notifier import EmailNotifier
from ender3monitor.metrics import MonitorMetrics
from ender3monitor.timelapse import TimelapseManager
from ender3monitor.printer import PrinterController
from ender3monitor.push import PushNotifier
from ender3monitor.maintenance import MaintenanceTracker
from ender3monitor.settings import Settings
from ender3monitor import ops_metrics as ops

# Camera-flip string (settings/UI) ↔ cv2.flip code (capture).
_FLIP_STR_TO_CODE = {"none": None, "180": -1, "vertical": 0, "horizontal": 1}
_FLIP_CODE_TO_STR = {None: "none", -1: "180", 0: "vertical", 1: "horizontal"}

PRINTER_POLL_INTERVAL = 5        # seconds between temperature polls over USB
PRINTER_RECONNECT_INTERVAL = 10  # seconds between reconnect attempts when dropped

# Analysis cadence comes from config.capture_interval (CAPTURE_INTERVAL_SECONDS).
TIMELAPSE_INTERVAL = 30     # seconds between timelapse frames (time-based mode)

# Layer-synced timelapse: capture when Z rises by at least this much (mm).
LAYER_Z_THRESHOLD_MM = 0.04
# A drop this large means the head re-homed / a new print began below the old
# baseline — re-baseline instead of waiting for Z to exceed the stale high mark.
LAYER_Z_RESET_MM = 1.0
# First-layer fallback window when no USB Z is available (seconds after print starts).
FIRST_LAYER_FALLBACK_SECONDS = 300
# Heaters are "at temp" within this many degrees of target (signal-based warmup).
WARMUP_TEMP_TOLERANCE = 5.0

# Print-complete detection: stop after this many consecutive still frames
NO_MOTION_LIMIT = 4         # 4 × 30 s = 2 minutes
MOTION_THRESHOLD = 5.0      # mean absolute pixel difference (0–255) to count as "changed"

# Failure confirmation: require sustained detection before alerting, to filter
# single-frame false positives. Calibrated in *wall-clock seconds*, not a fixed
# frame count — at long capture intervals a fixed count (e.g. 3) silently turned
# into many minutes (3 × 300 s = 15 min), so a real failure could end the print
# before ever crossing the alert threshold. We convert these windows into a
# frame count using the active capture interval (see _confirm_frames), floored
# at 2 so a lone bad frame never alerts regardless of interval.
FAILURE_CONFIRM_SECONDS = 90        # ~1.5 min of sustained failure before alerting
SPAGHETTI_CONFIRM_SECONDS = 45      # spaghetti is time-critical — confirm faster
FAILURE_CONFIRM_FRAMES = 3          # legacy default (used if interval is unknown)
SPAGHETTI_CONFIRM_FRAMES = 2


def _confirm_frames(window_seconds: int, interval_seconds: float) -> int:
    """Frames needed to span ``window_seconds`` at the current capture cadence.

    Floored at 2 so a single false-positive frame can never trigger an alert,
    and capped at the legacy 30 s-frame calibration so fast intervals don't
    demand an unreasonable streak.
    """
    if not interval_seconds or interval_seconds <= 0:
        return max(2, round(window_seconds / 30))
    return max(2, min(round(window_seconds / 30),
                      round(window_seconds / interval_seconds)))

# Startup grace period: skip failure detection while the printer warms up.
# The bed and nozzle take ~3-5 min to reach temperature; during this time
# the printer is stationary and the LLM would see a "nothing happening" frame.
STARTUP_GRACE_SECONDS = 300         # ~5 min warm-up window (converted to frames)

# Near the end of a print the head parks away from the model, leaving a gap
# that looks like stopped-extrusion. Suppress failure flagging past this %.
COMPLETION_SUPPRESS_PCT = 0.97

# Some firmware/boards never send "Not SD printing" or "Done printing" once a
# print actually finishes — M27 just keeps echoing the final "byte N/N" line
# forever, so the printing flag's falling edge never arrives. If progress has
# been pinned at/near 100% this long while `printing` is still (allegedly)
# True, force completion rather than getting stuck in "Finishing…" forever.
STUCK_AT_COMPLETION_PCT = 0.999
STUCK_AT_COMPLETION_SECONDS = 120


def _progress_stuck_since(current_since: Optional[float], printing: bool,
                           progress: Optional[float], now: float) -> Optional[float]:
    """Track how long progress has been pinned near 100% while still "printing".

    Returns the timestamp progress first reached the threshold (unchanged
    while it stays pinned), or None once it drops below the threshold or
    printing stops — resetting the window.
    """
    if printing and progress is not None and progress >= STUCK_AT_COMPLETION_PCT:
        return current_since if current_since is not None else now
    return None


def _frames_differ(f1: np.ndarray, f2: np.ndarray, threshold: float = MOTION_THRESHOLD) -> bool:
    """Return True if the two frames differ significantly (motion / print activity)."""
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    # If the camera reinitialises mid-session the resolution can change.
    # Resize gray2 to match gray1 so absdiff doesn't crash.
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    return float(cv2.absdiff(gray1, gray2).mean()) >= threshold


def _clear_line() -> None:
    # Clear current line + the description line below it
    print("\r\033[K\033[A\033[K", end="", flush=True)


def _header() -> None:
    print("\n" + "=" * 60)
    print("  Ender3Monitor – 3D Print Failure Detection")
    print("=" * 60)
    print("  Commands: start | stop | timelapse | quit")
    print("=" * 60 + "\n")


class Monitor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.camera: Optional[CameraManager] = None
        self.analyzer = create_analyzer(
            backend=config.analyzer_backend,
            anthropic_api_key=config.anthropic_api_key,
            anthropic_model=config.anthropic_model,
            ollama_model=config.ollama_model,
            ollama_host=config.ollama_host,
            capture_interval=config.capture_interval,
        )
        self.notifier = EmailNotifier(
            smtp_host=config.smtp_host,
            smtp_port=config.smtp_port,
            username=config.smtp_username,
            password=config.smtp_password,
            sender=config.smtp_sender,
            recipient=config.smtp_recipient,
        )
        self.metrics = MonitorMetrics()

        # Runtime-editable operational settings (seeded from .env/Config on first
        # run, then settings.json wins). Live keys are read straight from here in
        # the loop; constructed-object keys are applied via _apply_setting.
        self.settings = Settings(
            seed={
                "capture_interval": config.capture_interval,
                "confidence_threshold": config.confidence_threshold,
                "first_layer_interval": config.first_layer_interval,
                "first_layer_max_z": config.first_layer_max_z,
                "camera_flip": _FLIP_CODE_TO_STR.get(config.camera_flip, "none"),
                "auto_start_on_print": config.auto_start_on_print,
                "auto_pause_on_failure": config.auto_pause_on_failure,
                "auto_pause_action": config.auto_pause_action,
                "timelapse_mode": config.timelapse_mode,
                "timelapse_max_sessions": config.timelapse_max_sessions,
                "timelapse_retention_days": config.timelapse_retention_days,
                "timelapse_delete_frames_after_compile":
                    config.timelapse_delete_frames_after_compile,
                "maintenance_reminder_hours": config.maintenance_reminder_hours,
            },
            on_change=self._apply_setting,
        )

        self.timelapse = TimelapseManager(
            config.timelapse_dir,
            max_sessions=self.settings.get("timelapse_max_sessions"),
            retention_days=self.settings.get("timelapse_retention_days"),
            delete_frames_after_compile=self.settings.get("timelapse_delete_frames_after_compile"),
        )

        # Optional printer USB control + push notifications
        self.printer = PrinterController(config.printer_port, config.printer_baud)
        self.push = PushNotifier(
            ntfy_topic=config.ntfy_topic,
            discord_webhook=config.discord_webhook,
            telegram_bot_token=config.telegram_bot_token,
            telegram_chat_id=config.telegram_chat_id,
        )
        self.maintenance = MaintenanceTracker(
            reminder_hours=self.settings.get("maintenance_reminder_hours"))

        self._running = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Background printer temperature/progress poller
        self._printer_poll_stop = threading.Event()
        self._printer_poll_thread: Optional[threading.Thread] = None
        self._init_printer()

        # UI state
        self.status = "Idle"
        self.last_result: Optional[AnalysisResult] = None
        self.frame_count = 0
        self.failure_count = 0
        self.last_frame_time: Optional[datetime] = None
        # Server-side event log (survives browser reloads) + per-print API cost.
        self.events: deque = deque(maxlen=50)
        self.session_cost_usd: float = 0.0

        # Warmup gating — signal-based when USB connected, else time-based grace
        self._grace_until: float = 0.0   # set in start() (no-USB fallback)
        self._warmup_done: bool = False

        # Auto-start: begin monitoring when the printer starts a print (rising edge)
        self.auto_start_enabled: bool = self.settings.get("auto_start_on_print")
        self._poll_prev_printing: Optional[bool] = None   # None until first poll
        self._default_camera_index: Optional[int] = None
        self._default_frame_provider: Optional[FrameProvider] = None

        # Print-complete detection
        self._no_motion_count: int = 0
        self._prev_analysis_frame: Optional[np.ndarray] = None
        self._print_motion_seen: bool = False   # True once the printer first moves

        # Printer-authoritative completion (preferred over camera stillness)
        self._seen_printer_printing: bool = False  # saw an active SD/USB print
        self._printer_was_printing: bool = False   # previous poll's printing state
        self._print_active_since: Optional[float] = None  # when printing/motion first seen
        self._high_progress_since: Optional[float] = None  # when progress first pinned near 100%

        # Layer-synced timelapse (driven by the printer Z poller)
        self._timelapse_layer_mode: bool = False
        self._last_layer_z: Optional[float] = None

        # Per-run analysis stats (for the completion report)
        self._conf_sum: float = 0.0
        self._conf_n: int = 0
        self._conf_peak: float = 0.0
        self._run_failure_types: set = set()   # confirmed failure types this run

        # Failure confirmation — require consecutive frames before alerting
        self._pending_failure_type: Optional[str] = None
        self._pending_failure_count: int = 0
        self._pending_failure_alerted: bool = False   # alert once per incident

        # Optional external frame source (web UI shares its capture thread).
        # When set, the loop samples this instead of opening the camera.
        self._frame_provider: Optional[FrameProvider] = None

    # ------------------------------------------------------------------ #
    # Printer USB connection + polling                                     #
    # ------------------------------------------------------------------ #

    def _init_printer(self) -> None:
        """Connect to the printer (if configured) and start the poller.

        The poller is started even if the first connect fails, so the printer
        can be plugged in later (or recover from a bumped cable) without a
        restart.
        """
        if not self.config.printer_port:
            return   # USB control disabled
        if self.printer.connect():
            print(f"  Printer connected on {self.printer.status.port}.")
        else:
            print(f"  Printer not connected: {self.printer.status.last_error}")
            print("  Will keep retrying in the background…")
        # Some USB-serial adapters wedge a blocking read past its own
        # timeout, freezing the poll thread forever (field-observed). The
        # watchdog runs on its own thread so it can force-close the port and
        # unstick a stall even though the poll thread itself is unresponsive.
        self.printer.start_watchdog()
        self._printer_poll_thread = threading.Thread(
            target=self._printer_poll_loop, daemon=True
        )
        self._printer_poll_thread.start()

    def _apply_setting(self, key: str, value) -> None:
        """Push a changed setting into already-constructed objects.

        Loop-read keys (interval, threshold, first-layer, auto-pause, timelapse
        mode) need no action — the loop reads them live from self.settings.
        camera_flip's stream side is applied by the web layer (it owns the
        stream); here we cover the standalone-CLI camera.
        """
        if key == "auto_start_on_print":
            self.auto_start_enabled = bool(value)
        elif key == "maintenance_reminder_hours":
            self.maintenance.reminder_hours = max(1, int(value))
        elif key == "timelapse_max_sessions":
            self.timelapse.max_sessions = max(1, int(value))
        elif key == "timelapse_retention_days":
            self.timelapse.retention_days = max(0, int(value))
        elif key == "timelapse_delete_frames_after_compile":
            self.timelapse.delete_frames_after_compile = bool(value)
        elif key == "camera_flip" and self.camera is not None:
            self.camera._flip = _FLIP_STR_TO_CODE.get(value)

    def set_default_source(self, camera_index: Optional[int],
                           frame_provider: Optional[FrameProvider]) -> None:
        """Register a default camera/frame source so auto-start can begin
        monitoring on its own (called by the web UI with its shared stream)."""
        self._default_camera_index = camera_index
        self._default_frame_provider = frame_provider

    def _maybe_auto_start(self) -> None:
        """Begin monitoring on a fresh idle→printing transition (rising edge)."""
        printing = bool(self.printer.connected and self.printer.status.printing)
        prev = self._poll_prev_printing
        self._poll_prev_printing = printing
        if prev is None:
            return   # first observation — establish baseline, don't act
        rising_edge = printing and not prev
        if (rising_edge and self.auto_start_enabled and not self._running
                and self._default_frame_provider is not None):
            print("\n  [AUTO] Printer started a print — auto-starting monitoring.")
            self.start(camera_index=self._default_camera_index,
                       frame_provider=self._default_frame_provider)

    def _printer_poll_loop(self) -> None:
        """Poll temps/progress while connected; auto-reconnect when not.

        A failed serial write inside query_temps() flips the controller to
        disconnected, so a yanked cable is noticed within one poll cycle and
        we begin retrying connect() roughly every RECONNECT seconds.
        """
        reconnect_ticks = max(1, round(PRINTER_RECONNECT_INTERVAL / PRINTER_POLL_INTERVAL))
        ticks = 0
        was_connected = self.printer.connected
        while not self._printer_poll_stop.is_set():
            if self.printer.connected:
                self.printer.refresh_status()   # temps + print state + time + Z
                self._maybe_auto_start()        # begin monitoring on print start
                self._maybe_capture_layer_frame()
                self._check_printer_completion()  # falling-edge finish at 5 s cadence
            elif ticks % reconnect_ticks == 0:
                self.printer.connect()          # quiet retry; logged on transition below

            # Mirror live telemetry into Prometheus gauges each poll.
            ops.update_printer(self.printer.status)

            # Log + clean up on connect/disconnect transitions
            now_connected = self.printer.connected
            if was_connected and not now_connected:
                print("\n  [PRINTER] Connection lost — retrying in the background…")
                s = self.printer.status
                s.nozzle_temp = s.nozzle_target = None
                s.bed_temp = s.bed_target = None
                s.progress = s.elapsed_seconds = s.remaining_seconds = None
            elif not was_connected and now_connected:
                print(f"\n  [PRINTER] Reconnected on {self.printer.status.port}.")
                ops.printer_reconnects_total.inc()
            was_connected = now_connected

            ticks += 1
            self._printer_poll_stop.wait(timeout=PRINTER_POLL_INTERVAL)

    def _maybe_capture_layer_frame(self) -> None:
        """Layer-synced timelapse: save one frame whenever the Z height steps up.

        Runs in the printer poll loop (every ~5 s) so it sees fresh Z. Only active
        while monitoring with a shared frame source (web UI). Z-hop travel moves
        are mostly sampled-over at 5 s, so this yields ~one frame per layer.
        """
        if not (self._running and self._timelapse_layer_mode and self._frame_provider):
            return
        if not self.printer.status.printing:
            return   # ignore homing/parking moves outside an active print
        z = self.printer.status.z_height
        if z is None:
            return
        # The head ended the previous job parked high (e.g. Z=80). Without this
        # reset the baseline would start there and a new print whose layers are
        # all lower would never trigger a capture for the entire run.
        if self._last_layer_z is not None and z < self._last_layer_z - LAYER_Z_RESET_MM:
            self._last_layer_z = z
            return
        if self._last_layer_z is None or z >= self._last_layer_z + LAYER_Z_THRESHOLD_MM:
            frame = self._frame_provider()
            if frame is not None:
                self.timelapse.save_frame(frame)
                self._last_layer_z = z

    def _check_printer_completion(self) -> None:
        """Detect print completion from the printer's own state (falling edge).

        Runs in the 5 s poll loop rather than the analysis loop, so completion
        is noticed within seconds of the job ending — at a 5-minute analysis
        cadence the old in-loop check could fire up to 5 minutes late, by which
        time the part may already be off the bed (empty-bed "final photo").
        """
        printing = bool(self.printer.status.printing)
        progress = self.printer.status.progress
        if self._running and printing:
            self._seen_printer_printing = True
            if self._print_active_since is None:
                self._print_active_since = time.time()

        # Safety net for firmware that never reports "Not SD printing"/"Done
        # printing" once the job truly ends (see STUCK_AT_COMPLETION_* above).
        self._high_progress_since = _progress_stuck_since(
            self._high_progress_since, printing, progress, time.time())
        stuck = (
            self._high_progress_since is not None
            and (time.time() - self._high_progress_since) >= STUCK_AT_COMPLETION_SECONDS
        )

        falling = self._printer_was_printing and not printing
        # Log every printing-state transition so a missed completion is
        # diagnosable from the console without reproducing under a debugger.
        if printing != self._printer_was_printing:
            pct = self.printer.status.progress
            print(f"\n  [PRINTER] printing {self._printer_was_printing} → {printing}"
                  f" (progress={pct:.0%} running={self._running}"
                  f" seen_printing={self._seen_printer_printing})"
                  if pct is not None else
                  f"\n  [PRINTER] printing {self._printer_was_printing} → {printing}"
                  f" (running={self._running} seen_printing={self._seen_printer_printing})")
        self._printer_was_printing = printing

        if not ((falling or stuck) and self._running and self._seen_printer_printing):
            return

        # Grab the final frame NOW, while the finished part is still on the bed.
        frame = None
        if self._frame_provider is not None:
            frame = self._frame_provider()
        elif self.camera is not None:
            frame = self.camera.snapshot()
        if frame is None:
            frame = self._prev_analysis_frame

        with self._lock:
            self.status = "Print Complete"
            self._running = False
        self._stop_event.set()   # wake the analysis loop so it exits promptly
        if stuck and not falling:
            print(f"\n  ✓ Print complete (progress pinned at {progress:.0%} for "
                  f"{STUCK_AT_COMPLETION_SECONDS}s — printer never reported "
                  "end-of-print, forcing completion).")
        else:
            print("\n  ✓ Print complete (reported by printer).")

        # Heavy I/O (email, timelapse compile, Telegram video) runs on its own
        # thread so this poll loop keeps polling temps without a long stall.
        threading.Thread(
            target=self._send_completion, args=(frame,), daemon=True
        ).start()

    def _warmup_gate(self):
        """Decide whether to skip analysis during warmup.

        Returns (skip: bool, status_message: Optional[str]). With USB connected
        we gate on real signals — printing must have started and heaters must be
        within tolerance of target. Without USB we fall back to the time-based
        grace window. Once cleared, the gate stays open for the rest of the run.
        """
        if self._warmup_done:
            return False, None

        pr = self.printer.status if self.printer.connected else None
        if pr is not None:
            n, nt = pr.nozzle_temp, pr.nozzle_target
            b, bt = pr.bed_temp, pr.bed_target

            # Heaters still climbing → warming up (checked FIRST: the printer's
            # "printing" flag (M27) can lag through the whole heat phase, so we
            # must not let it mask a clearly-warming-up print as "Waiting…").
            heating = (
                (nt and n is not None and n < nt - WARMUP_TEMP_TOLERANCE) or
                (bt and b is not None and b < bt - WARMUP_TEMP_TOLERANCE)
            )
            if heating:
                msg = "Warming up…"
                if n is not None and nt:
                    msg += f" (nozzle {n:.0f}/{nt:.0f}°"
                    if b is not None and bt:
                        msg += f", bed {b:.0f}/{bt:.0f}°"
                    msg += ")"
                return True, msg

            # Heaters at/above target. If a heat target is set (or M27 confirms
            # printing), the job is underway → start analyzing. Don't gate on the
            # unreliable printing flag alone.
            if pr.printing or (nt and nt > 0) or (bt and bt > 0):
                self._warmup_done = True
                return False, None

            # Genuinely idle (no heat set, not printing) — wait briefly for the
            # job to spin up, then proceed so we never stall forever.
            if time.time() < self._grace_until:
                return True, "Waiting for print to start…"
            self._warmup_done = True
            return False, None

        # No USB — time-based grace fallback.
        if time.time() < self._grace_until:
            secs = int(self._grace_until - time.time())
            return True, (f"Warming up… ({secs // 60}m {secs % 60}s remaining)"
                          if secs > 0 else "Warming up… (almost ready)")
        self._warmup_done = True
        return False, None

    def _in_first_layer(self) -> bool:
        """True while the nozzle is on/near the first layer (failure-prone)."""
        if not self.settings.get("first_layer_mode"):
            return False   # feature off → use the main interval throughout
        pr = self.printer.status if self.printer.connected else None
        if pr and pr.printing and pr.z_height is not None:
            return pr.z_height <= self.settings.get("first_layer_max_z")
        # Fallback with no USB Z: first few minutes after printing/motion began.
        if self._print_active_since is not None:
            return (time.time() - self._print_active_since) <= FIRST_LAYER_FALLBACK_SECONDS
        return False

    def close(self) -> None:
        """Release printer resources (call on app shutdown)."""
        self._printer_poll_stop.set()
        if self._printer_poll_thread and self._printer_poll_thread.is_alive():
            self._printer_poll_thread.join(timeout=2)
        self.printer.stop_watchdog()
        self.printer.disconnect()

    # ------------------------------------------------------------------ #
    # Monitoring loop                                                       #
    # ------------------------------------------------------------------ #

    def _monitoring_loop(self) -> None:
        last_capture = 0.0
        last_timelapse = 0.0

        self.metrics.monitoring_active.set(1)
        try:
            while self._running:
                # Cadence is dynamic: tighter while on the first layer. The printer
                # poller keeps z_height fresh, so we can decide before sleeping.
                first_layer = self._in_first_layer()
                interval = (self.settings.get("first_layer_interval") if first_layer
                            else self.settings.get("capture_interval"))
                # Time-based timelapse only when NOT in layer-synced mode.
                time_timelapse = not self._timelapse_layer_mode

                # Sleep until the next event (analysis or, in time mode, timelapse).
                now = time.time()
                next_event = last_capture + interval
                if time_timelapse:
                    next_event = min(next_event, last_timelapse + TIMELAPSE_INTERVAL)
                sleep_for = max(1.0, next_event - now)
                # Event.wait() returns immediately when stop() sets _stop_event,
                # so we never block the UI for the full interval waiting on a sleep.
                self._stop_event.wait(timeout=sleep_for)

                if not self._running:
                    break

                now = time.time()
                needs_analysis = now - last_capture >= interval
                needs_timelapse = time_timelapse and (now - last_timelapse >= TIMELAPSE_INTERVAL)

                if not (needs_analysis or needs_timelapse):
                    continue

                # Sample a frame. With an external provider (web UI), pull the
                # latest frame from its shared capture thread — no second camera
                # handle. Otherwise open/grab/release via snapshot(), which keeps
                # OpenCV's capture thread from running continuously between frames.
                if self._frame_provider is not None:
                    frame = self._frame_provider()
                else:
                    frame = self.camera.snapshot() if self.camera else None

                if frame is None:
                    with self._lock:
                        self.status = "Camera error – no frame"
                    continue

                # Time-based timelapse (layer-synced capture happens in the poller)
                if needs_timelapse:
                    self.timelapse.save_frame(frame)
                    last_timelapse = now

                # Analysis frame
                if needs_analysis:
                    last_capture = now

                    # ── Warmup gate (signal-based w/ USB, else time-based) ── #
                    skip_warmup, warmup_msg = self._warmup_gate()
                    if skip_warmup:
                        with self._lock:
                            self.status = warmup_msg or "Warming up…"
                        self._prev_analysis_frame = frame.copy()
                        if self._running:
                            _print_status(self)
                        continue   # skip motion check and LLM until warmed up

                    # Printer-reported completion is detected in the 5 s poll loop
                    # (_check_printer_completion); here we only read the flags it
                    # maintains. use_printer_completion disables the camera-motion
                    # fallback once the printer has proven it reports SD status.
                    pr = self.printer.status if self.printer.connected else None
                    use_printer_completion = self._seen_printer_printing

                    # Near the end the head parks away from the model → looks like a
                    # gap/stopped-extrusion. Suppress failure flagging past the cutoff.
                    near_completion = bool(
                        pr and pr.progress is not None and pr.progress >= COMPLETION_SUPPRESS_PCT
                    )

                    # ── Camera motion completion (fallback only) ──────────── #
                    # Used only when the printer can't tell us (no USB / not reporting
                    # SD status). Once we trust the printer, skip this entirely so a
                    # parked head at the end isn't mistaken for completion or failure.
                    if not use_printer_completion:
                        if self._prev_analysis_frame is not None:
                            if _frames_differ(frame, self._prev_analysis_frame):
                                if self._no_motion_count > 0:
                                    print(
                                        f"\n  [MOTION] Change detected — "
                                        f"resetting idle counter (was {self._no_motion_count})"
                                    )
                                self._print_motion_seen = True
                                if self._print_active_since is None:
                                    self._print_active_since = time.time()
                                self._no_motion_count = 0
                            else:
                                if self._print_motion_seen:
                                    self._no_motion_count += 1
                                    print(
                                        f"\n  [MOTION] No change detected "
                                        f"({self._no_motion_count}/{NO_MOTION_LIMIT} frames)"
                                    )
                                    if self._no_motion_count >= NO_MOTION_LIMIT:
                                        with self._lock:
                                            self.status = "Print Complete"
                                            self._running = False
                                        print(
                                            "\n  ✓ Print appears complete — "
                                            "no change for 4 consecutive frames (2 min)."
                                        )
                                        self._send_completion(frame)
                                        _print_status(self)
                                        break
                                else:
                                    print("\n  [MOTION] No change yet — waiting for print to start")
                        self._prev_analysis_frame = frame.copy()

                    # ── LLM analysis ───────────────────────────────────── #
                    with self._lock:
                        self.status = ("Analyzing first layer…" if first_layer
                                       else "Analyzing frame…")
                    _t0 = time.perf_counter()
                    try:
                        result = self.analyzer.analyze_frame(frame, first_layer=first_layer)
                        ops.analysis_duration_seconds.labels(
                            self.config.analyzer_backend).observe(time.perf_counter() - _t0)
                    except Exception as exc:
                        ops.analysis_errors_total.labels(self.config.analyzer_backend).inc()
                        with self._lock:
                            self.status = f"Analysis error: {exc}"
                        continue

                    confirmed_failure = False
                    with self._lock:
                        self.frame_count += 1
                        self.last_result = result
                        self.last_frame_time = datetime.now()
                        self.session_cost_usd += getattr(result, "cost_usd", 0.0) or 0.0
                        # Server-side event log (replayed to browsers on reload).
                        self.events.append({
                            "t": time.time(),
                            "type": result.failure_type,
                            "conf": round(result.confidence, 3),
                            "desc": (result.description or "")[:140],
                            "detected": result.failure_detected,
                        })
                        # Per-run confidence stats (for the completion report)
                        self._conf_sum += result.confidence
                        self._conf_n += 1
                        self._conf_peak = max(self._conf_peak, result.confidence)
                        threshold = self.settings.get("confidence_threshold")
                        self.metrics.record_analysis(
                            result.confidence,
                            result.failure_type,
                            threshold,
                        )

                        # Only update status if stop() hasn't already set it to Idle
                        if self._running:
                            is_failure = (
                                result.failure_detected
                                and result.confidence >= threshold
                                and result.failure_type not in ("no_printer", "none")
                                # Don't flag a "gap" failure when the print is basically done
                                and not near_completion
                            )
                            if is_failure:
                                # Spaghetti is time-critical — alert on the first frame.
                                # All other failures require consecutive confirmation.
                                is_spaghetti = "spaghetti" in result.failure_type.lower()
                                confirm_needed = _confirm_frames(
                                    SPAGHETTI_CONFIRM_SECONDS if is_spaghetti
                                    else FAILURE_CONFIRM_SECONDS,
                                    interval,
                                )

                                # Track consecutive frames with the same failure type
                                if result.failure_type == self._pending_failure_type:
                                    self._pending_failure_count += 1
                                else:
                                    self._pending_failure_type = result.failure_type
                                    self._pending_failure_count = 1
                                    self._pending_failure_alerted = False

                                self.status = (
                                    f"Possible failure – {result.failure_type} "
                                    f"({self._pending_failure_count}/{confirm_needed})"
                                    if self._pending_failure_count < confirm_needed
                                    else f"FAILURE DETECTED – {result.failure_type}"
                                )

                                if (self._pending_failure_count >= confirm_needed
                                        and not self._pending_failure_alerted):
                                    # Confirmed — alert once per incident. Do the
                                    # actual I/O (email/push/serial) outside the lock.
                                    self._pending_failure_alerted = True
                                    self.failure_count += 1
                                    self._no_motion_count = 0
                                    confirmed_failure = True
                            else:
                                # Clean frame (or suppressed near completion) —
                                # reset pending failure
                                self._pending_failure_type = None
                                self._pending_failure_count = 0
                                self._pending_failure_alerted = False
                                if near_completion:
                                    self.status = "Finishing… (near complete)"
                                elif result.failure_type == "no_printer":
                                    self.status = "Monitoring… (no printer in frame)"
                                elif first_layer:
                                    self.status = "Monitoring… (first layer)"
                                else:
                                    self.status = "Monitoring…"

                    # Side effects for a confirmed failure — outside the lock so
                    # slow email/serial I/O doesn't block UI state reads.
                    if confirmed_failure:
                        self._handle_confirmed_failure(result, frame)

                    if self._running:
                        _print_status(self)

        finally:
            self.metrics.monitoring_active.set(0)
            # snapshot() releases the camera after every frame, so nothing
            # persistent to clean up here.

    def _handle_confirmed_failure(self, result: AnalysisResult, frame: np.ndarray) -> None:
        """All side effects for a confirmed failure: email, push, auto-pause."""
        self._run_failure_types.add(result.failure_type)   # for maintenance trend tracking

        # 1. Email (existing behaviour)
        self._send_alert(result, frame)

        # 2. Push notification
        if self.push.enabled:
            msg = result.description or f"{result.failure_type} detected"
            self.push.send(
                title=f"⚠️ Print failure: {result.failure_type}",
                message=f"{msg}\nConfidence {result.confidence:.0%}",
                priority="high",
            )

        # 3. Auto-pause / cooldown / e-stop over USB
        if self.settings.get("auto_pause_on_failure"):
            action = self.settings.get("auto_pause_action")
            if self.printer.connected:
                action_result = self.printer.apply_failure_action(action)
                print(f"\n  [PRINTER] Auto-{action}: {action_result}")
                if self.push.enabled:
                    self.push.send(
                        title="🛑 Printer action taken",
                        message=f"Auto-{action}: {action_result}",
                        priority="high",
                    )
            else:
                print("\n  [PRINTER] Auto-pause enabled but printer not connected.")

    def _send_alert(self, result: AnalysisResult, frame: np.ndarray) -> None:
        if not self.notifier.enabled:
            return
        try:
            self.notifier.send_alert(result, frame)
            print(f"\n  [EMAIL] Alert sent to {self.config.smtp_recipient}")
        except Exception as exc:
            print(f"\n  [EMAIL ERROR] {exc}")

    def _send_completion(self, frame: Optional[np.ndarray]) -> None:
        # Email (existing behaviour) — needs a frame for the attachment
        if frame is not None and self.notifier.enabled:
            try:
                self.notifier.send_completion(frame, self.frame_count)
                print(f"\n  [EMAIL] Completion notice sent to {self.config.smtp_recipient}")
            except Exception as exc:
                print(f"\n  [EMAIL ERROR] {exc}")

        # Rich report (stats + final photo + compiled timelapse) to push channels
        try:
            self._send_completion_report(frame)
        except Exception as exc:
            print(f"\n  [REPORT ERROR] {exc}")

        # Maintenance/health tracking — log this print and push any reminders
        try:
            elapsed = self.printer.status.elapsed_seconds
            if elapsed is None and self._print_active_since:
                elapsed = int(time.time() - self._print_active_since)
            for alert in self.maintenance.record_print(elapsed, self._run_failure_types):
                print(f"\n  [MAINT] {alert}")
                if self.push.enabled:
                    self.push.send(title="Maintenance", message=alert, priority="default")
        except Exception as exc:
            print(f"\n  [MAINT ERROR] {exc}")

    def _build_report_stats(self) -> str:
        """One-line-per-stat summary text for the completion report."""
        from ender3monitor.printer import _fmt_duration
        avg = (self._conf_sum / self._conf_n) if self._conf_n else 0.0
        # Prefer the printer's job timer; fall back to wall-clock since print start.
        elapsed = self.printer.status.elapsed_seconds
        if elapsed is None and self._print_active_since:
            elapsed = int(time.time() - self._print_active_since)
        lines = [
            f"Duration: {_fmt_duration(elapsed) or '—'}",
            f"Frames analyzed: {self.frame_count}",
            f"Failures: {self.failure_count}",
            f"Avg/peak confidence: {avg:.0%} / {self._conf_peak:.0%}",
            f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        if self.session_cost_usd > 0:
            lines.append(f"AI cost: ${self.session_cost_usd:.2f}")

        # Suggested sell price (manual filament weight + configured rates).
        from ender3monitor.pricing import compute_price, format_price_lines
        price = compute_price(elapsed, self.settings.get("filament_grams"), self.settings)
        lines += format_price_lines(price)
        return "\n".join(lines)

    def _send_completion_report(self, frame: np.ndarray) -> None:
        if not self.push.enabled:
            return
        stats = self._build_report_stats()
        # 1. Text summary (all channels)
        self.push.send(title="✅ Print complete", message=stats, priority="default")
        # 2. Final photo (Telegram)
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                self.push.send_photo(buf.tobytes(), caption="Final frame")
        # 3. Compiled timelapse video (Telegram, if not too large)
        try:
            mp4 = self.timelapse.compile(**self._timelapse_compile_kwargs())
            if mp4:
                self.push.send_video(mp4, caption="Timelapse")
        except Exception as exc:
            print(f"\n  [REPORT] timelapse compile/send skipped: {exc}")

    # ------------------------------------------------------------------ #
    # Public control methods                                               #
    # ------------------------------------------------------------------ #

    def start(self, camera_index: Optional[int] = None,
              frame_provider: Optional[FrameProvider] = None) -> None:
        if self._running:
            print("  Already monitoring.")
            return

        # Camera setup — caller can override (e.g. web UI passes chosen index)
        if camera_index is not None:
            idx = camera_index
        elif self.config.camera_index == -1:
            try:
                idx = CameraManager.select_camera()
            except RuntimeError as exc:
                print(f"  {exc}")
                print("  Tip: another process (e.g. the web UI) may be using the camera,")
                print("       or set CAMERA_INDEX in .env to skip auto-detection.")
                return
        else:
            idx = self.config.camera_index

        # CameraManager is kept for metadata (camera_index display). When a
        # frame_provider is supplied the loop uses that instead of opening the
        # camera, so we must NOT probe it here — that would grab a second handle
        # and fight the provider's persistent capture thread.
        self.camera = CameraManager(idx, flip=self.config.camera_flip)
        self._frame_provider = frame_provider
        if frame_provider is None:
            # Standalone capture — verify the camera is reachable before starting.
            if self.camera.snapshot() is None:
                print(f"  Camera error: cannot read from camera {idx}.")
                self.camera = None
                return

        self.timelapse.reset_session()
        self._stop_event.clear()
        self._running = True
        self.frame_count = 0
        self.failure_count = 0
        self.session_cost_usd = 0.0
        self.last_result = None
        self.status = "Monitoring…"
        self._grace_until = time.time() + STARTUP_GRACE_SECONDS
        self._warmup_done = False
        self._no_motion_count = 0
        self._prev_analysis_frame = None
        self._print_motion_seen = False
        self._seen_printer_printing = False
        self._printer_was_printing = False
        self._print_active_since = None
        self._high_progress_since = None
        self._pending_failure_type = None
        self._pending_failure_count = 0
        self._pending_failure_alerted = False
        self._last_layer_z = None
        self._conf_sum = 0.0
        self._conf_n = 0
        self._conf_peak = 0.0
        self._run_failure_types = set()
        # Layer-synced timelapse when the printer reports Z (auto/layer modes).
        tl_mode = self.settings.get("timelapse_mode")
        self._timelapse_layer_mode = (
            tl_mode == "layer"
            or (tl_mode == "auto" and self.printer.connected)
        )

        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()
        print(f"  Monitoring started (camera {idx}). "
              f"Analysis every {self.settings.get('capture_interval')}s.")

    def stop(self) -> None:
        if not self._running:
            if self.status == "Print Complete":
                print("  Print already marked as complete — monitor has stopped.")
            else:
                print("  Not currently monitoring.")
            return
        self._running = False
        self._stop_event.set()   # wake the sleeping loop immediately
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3)
        self._stop_event.clear()
        if self.camera:
            self.camera.release()
            self.camera = None
        self._frame_provider = None   # release reference to the web UI's capture
        if self.status != "Print Complete":
            self.status = "Idle"
        print("  Monitoring stopped.")

    def _timelapse_overlay_lines(self) -> list:
        """Build the burn-in caption lines per the configured toggles.

        Empty when the overlay is off, so callers can pass it unconditionally.
        """
        if not self.settings.get("timelapse_overlay"):
            return []
        lines: list = []
        if self.settings.get("timelapse_overlay_elapsed"):
            elapsed = self.printer.status.elapsed_seconds
            if elapsed is None and self._print_active_since:
                elapsed = int(time.time() - self._print_active_since)
            dur = _fmt_duration(elapsed)
            if dur:
                lines.append(f"Print time: {dur}")
        if self.settings.get("timelapse_overlay_filament"):
            grams = self.settings.get("filament_grams")
            if grams:
                lines.append(f"Filament: {grams:g} g")
        if self.settings.get("timelapse_overlay_date"):
            lines.append(datetime.now().strftime("%Y-%m-%d"))
        return lines

    def _timelapse_compile_kwargs(self) -> dict:
        """Aspect/fit/fps/overlay for compile(), read live from settings."""
        return {
            "fps": int(self.settings.get("timelapse_fps")),
            "aspect": self.settings.get("camera_aspect"),
            "fit": self.settings.get("camera_fit"),
            "overlay_lines": self._timelapse_overlay_lines(),
        }

    def compile_timelapse(self) -> Optional[str]:
        print("  Compiling timelapse…")
        result = self.timelapse.compile(**self._timelapse_compile_kwargs())
        if result:
            print(f"  Saved: {result}")
        return result

    def compile_timelapse_session(self, session_dir) -> Optional[str]:
        print(f"  Compiling timelapse session {session_dir}…")
        result = self.timelapse.compile_session(session_dir, **self._timelapse_compile_kwargs())
        if result:
            print(f"  Saved: {result}")
        return result


# ------------------------------------------------------------------ #
# Terminal UI helpers                                                   #
# ------------------------------------------------------------------ #

def _print_status(mon: Monitor) -> None:
    ts = mon.last_frame_time.strftime("%H:%M:%S") if mon.last_frame_time else "—"
    conf = f"{mon.last_result.confidence:.1%}" if mon.last_result else "—"
    ftype = mon.last_result.failure_type if mon.last_result else "—"
    raw_desc = mon.last_result.description if mon.last_result else ""
    desc = (raw_desc[:72] + "…") if len(raw_desc) > 73 else raw_desc
    _clear_line()
    print(
        f"  [{ts}] Status: {mon.status:<22} | "
        f"Frames: {mon.frame_count:>4} | Failures: {mon.failure_count:>3} | "
        f"Confidence: {conf} | Type: {ftype}\n"
        f"           {desc}"
    )


def _print_full_status(mon: Monitor) -> None:
    print()
    print(f"  Status        : {mon.status}")
    if mon.last_result:
        print(f"  Last result   : {mon.last_result.summary}")
        print(f"  Description   : {mon.last_result.description}")
    print(f"  Frames analyzed: {mon.frame_count}")
    print(f"  Failures found : {mon.failure_count}")
    print(f"  Camera index   : {mon.camera.camera_index if mon.camera else 'N/A'}")
    print()


# ------------------------------------------------------------------ #
# Entry point                                                           #
# ------------------------------------------------------------------ #

def main() -> None:
    try:
        config = Config.from_env()
    except KeyError as exc:
        print(f"Missing required environment variable: {exc}")
        print("Copy .env.example to .env and fill in your values.")
        sys.exit(1)

    mon = Monitor(config)
    mon.metrics.start_server(config.metrics_port)

    _header()

    COMMANDS = {
        "start": mon.start,
        "stop": mon.stop,
        "timelapse": mon.compile_timelapse,
    }

    while True:
        try:
            cmd = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting…")
            mon.stop()
            break

        if cmd in ("quit", "exit", "q"):
            mon.stop()
            break
        elif cmd == "status":
            _print_full_status(mon)
        elif cmd in COMMANDS:
            COMMANDS[cmd]()
        elif cmd == "":
            _print_full_status(mon)
        else:
            print(f"  Unknown command '{cmd}'. Commands: start | stop | timelapse | status | quit")


if __name__ == "__main__":
    main()
