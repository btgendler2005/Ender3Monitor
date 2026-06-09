#!/usr/bin/env python3
"""Ender3Monitor – 3D print failure detection system."""

import sys
import threading
import time
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

PRINTER_POLL_INTERVAL = 5    # seconds between temperature polls over USB

CAPTURE_INTERVAL = 30       # seconds between analysis frames
TIMELAPSE_INTERVAL = 30     # seconds between timelapse frames

# Print-complete detection: stop after this many consecutive still frames
NO_MOTION_LIMIT = 4         # 4 × 30 s = 2 minutes
MOTION_THRESHOLD = 5.0      # mean absolute pixel difference (0–255) to count as "changed"

# Failure confirmation: require this many consecutive frames before alerting.
# Eliminates single-frame false positives — a real failure persists.
FAILURE_CONFIRM_FRAMES = 3          # 3 × 30 s = 1.5 min of sustained failure before alerting
SPAGHETTI_CONFIRM_FRAMES = 2        # spaghetti needs 2 frames — 1 min — reduces blob/stringing false alarms

# Startup grace period: skip failure detection while the printer warms up.
# The bed and nozzle take ~3-5 min to reach temperature; during this time
# the printer is stationary and the LLM would see a "nothing happening" frame.
STARTUP_GRACE_FRAMES = 10           # 10 × 30 s = 5 min warm-up window


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
        self.timelapse = TimelapseManager(config.timelapse_dir)

        # Optional printer USB control + push notifications
        self.printer = PrinterController(config.printer_port, config.printer_baud)
        self.push = PushNotifier(
            ntfy_topic=config.ntfy_topic,
            discord_webhook=config.discord_webhook,
            telegram_bot_token=config.telegram_bot_token,
            telegram_chat_id=config.telegram_chat_id,
        )

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

        # Startup grace period — skip detection while printer warms up
        self._grace_frames_remaining: int = STARTUP_GRACE_FRAMES

        # Print-complete detection
        self._no_motion_count: int = 0
        self._prev_analysis_frame: Optional[np.ndarray] = None
        self._print_motion_seen: bool = False   # True once the printer first moves

        # Failure confirmation — require consecutive frames before alerting
        self._pending_failure_type: Optional[str] = None
        self._pending_failure_count: int = 0

        # Optional external frame source (web UI shares its capture thread).
        # When set, the loop samples this instead of opening the camera.
        self._frame_provider: Optional[FrameProvider] = None

    # ------------------------------------------------------------------ #
    # Printer USB connection + polling                                     #
    # ------------------------------------------------------------------ #

    def _init_printer(self) -> None:
        """Connect to the printer (if configured) and start the temp poller."""
        if not self.config.printer_port:
            return   # USB control disabled
        if self.printer.connect():
            print(f"  Printer connected on {self.printer.status.port}.")
            self._printer_poll_thread = threading.Thread(
                target=self._printer_poll_loop, daemon=True
            )
            self._printer_poll_thread.start()
        else:
            print(f"  Printer not connected: {self.printer.status.last_error}")

    def _printer_poll_loop(self) -> None:
        """Poll temperatures (and SD progress) for the live UI."""
        ticks = 0
        while not self._printer_poll_stop.is_set():
            if self.printer.connected:
                self.printer.query_temps()
                if ticks % 3 == 0:          # progress less often
                    self.printer.query_progress()
            ticks += 1
            self._printer_poll_stop.wait(timeout=PRINTER_POLL_INTERVAL)

    def close(self) -> None:
        """Release printer resources (call on app shutdown)."""
        self._printer_poll_stop.set()
        if self._printer_poll_thread and self._printer_poll_thread.is_alive():
            self._printer_poll_thread.join(timeout=2)
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
                # Sleep until the next event (analysis or timelapse) rather than
                # waking every second and grabbing a frame we'll discard.
                now = time.time()
                next_event = min(
                    last_capture + CAPTURE_INTERVAL,
                    last_timelapse + TIMELAPSE_INTERVAL,
                )
                sleep_for = max(1.0, next_event - now)
                # Event.wait() returns immediately when stop() sets _stop_event,
                # so we never block the UI for up to 30 s waiting on a sleep.
                self._stop_event.wait(timeout=sleep_for)

                if not self._running:
                    break

                now = time.time()
                needs_analysis = now - last_capture >= CAPTURE_INTERVAL
                needs_timelapse = now - last_timelapse >= TIMELAPSE_INTERVAL

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

                # Timelapse frame every 60 s
                if needs_timelapse:
                    self.timelapse.save_frame(frame)
                    last_timelapse = now

                # Analysis frame every 30 s
                if needs_analysis:
                    last_capture = now

                    # ── Startup grace period ───────────────────────────── #
                    if self._grace_frames_remaining > 0:
                        self._grace_frames_remaining -= 1
                        secs_left = self._grace_frames_remaining * CAPTURE_INTERVAL
                        mins_left = secs_left // 60
                        with self._lock:
                            self.status = (
                                f"Warming up… ({mins_left}m {secs_left % 60}s remaining)"
                                if secs_left > 0 else "Warming up… (almost ready)"
                            )
                        self._prev_analysis_frame = frame.copy()
                        if self._running:
                            _print_status(self)
                        continue   # skip motion check and LLM during grace period

                    # ── Motion / completion detection ──────────────────── #
                    if self._prev_analysis_frame is not None:
                        if _frames_differ(frame, self._prev_analysis_frame):
                            if self._no_motion_count > 0:
                                print(
                                    f"\n  [MOTION] Change detected — "
                                    f"resetting idle counter (was {self._no_motion_count})"
                                )
                            self._print_motion_seen = True
                            self._no_motion_count = 0
                        else:
                            if self._print_motion_seen:
                                # Only count toward completion after printing has started
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
                        self.status = "Analyzing frame…"
                    try:
                        result = self.analyzer.analyze_frame(frame)
                    except Exception as exc:
                        with self._lock:
                            self.status = f"Analysis error: {exc}"
                        continue

                    confirmed_failure = False
                    with self._lock:
                        self.frame_count += 1
                        self.last_result = result
                        self.last_frame_time = datetime.now()
                        self.metrics.record_analysis(
                            result.confidence,
                            result.failure_type,
                            self.config.confidence_threshold,
                        )

                        # Only update status if stop() hasn't already set it to Idle
                        if self._running:
                            is_failure = (
                                result.failure_detected
                                and result.confidence >= self.config.confidence_threshold
                                and result.failure_type not in ("no_printer", "none")
                            )
                            if is_failure:
                                # Spaghetti is time-critical — alert on the first frame.
                                # All other failures require consecutive confirmation.
                                is_spaghetti = "spaghetti" in result.failure_type.lower()
                                confirm_needed = (
                                    SPAGHETTI_CONFIRM_FRAMES if is_spaghetti
                                    else FAILURE_CONFIRM_FRAMES
                                )

                                # Track consecutive frames with the same failure type
                                if result.failure_type == self._pending_failure_type:
                                    self._pending_failure_count += 1
                                else:
                                    self._pending_failure_type = result.failure_type
                                    self._pending_failure_count = 1

                                self.status = (
                                    f"Possible failure – {result.failure_type} "
                                    f"({self._pending_failure_count}/{confirm_needed})"
                                    if self._pending_failure_count < confirm_needed
                                    else f"FAILURE DETECTED – {result.failure_type}"
                                )

                                if self._pending_failure_count == confirm_needed:
                                    # Confirmed — alert once per incident. Do the
                                    # actual I/O (email/push/serial) outside the lock.
                                    self.failure_count += 1
                                    self._no_motion_count = 0
                                    confirmed_failure = True
                            else:
                                # Clean frame — reset pending failure
                                self._pending_failure_type = None
                                self._pending_failure_count = 0
                                if result.failure_type == "no_printer":
                                    self.status = "Monitoring… (no printer in frame)"
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
        if self.config.auto_pause_on_failure:
            if self.printer.connected:
                action_result = self.printer.apply_failure_action(self.config.auto_pause_action)
                print(f"\n  [PRINTER] Auto-{self.config.auto_pause_action}: {action_result}")
                if self.push.enabled:
                    self.push.send(
                        title="🛑 Printer action taken",
                        message=f"Auto-{self.config.auto_pause_action}: {action_result}",
                        priority="high",
                    )
            else:
                print("\n  [PRINTER] Auto-pause enabled but printer not connected.")

    def _send_alert(self, result: AnalysisResult, frame: np.ndarray) -> None:
        try:
            self.notifier.send_alert(result, frame)
            print(f"\n  [EMAIL] Alert sent to {self.config.smtp_recipient}")
        except Exception as exc:
            print(f"\n  [EMAIL ERROR] {exc}")

    def _send_completion(self, frame: np.ndarray) -> None:
        try:
            self.notifier.send_completion(frame, self.frame_count)
            print(f"\n  [EMAIL] Completion notice sent to {self.config.smtp_recipient}")
        except Exception as exc:
            print(f"\n  [EMAIL ERROR] {exc}")
        if self.push.enabled:
            self.push.send(
                title="✅ Print complete",
                message=f"Monitoring finished after {self.frame_count} analyzed frames.",
                priority="default",
            )

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
        self.last_result = None
        self.status = "Monitoring…"
        self._grace_frames_remaining = STARTUP_GRACE_FRAMES
        self._no_motion_count = 0
        self._prev_analysis_frame = None
        self._print_motion_seen = False
        self._pending_failure_type = None
        self._pending_failure_count = 0

        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()
        print(f"  Monitoring started (camera {idx}). Analysis every {CAPTURE_INTERVAL}s.")

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

    def compile_timelapse(self) -> Optional[str]:
        print("  Compiling timelapse…")
        result = self.timelapse.compile()
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
