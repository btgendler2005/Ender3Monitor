#!/usr/bin/env python3
"""Ender3Monitor – 3D print failure detection system."""

import sys
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from ender3monitor.config import Config
from ender3monitor.camera import CameraManager
from ender3monitor.analyzer import create_analyzer, AnalysisResult
from ender3monitor.notifier import EmailNotifier
from ender3monitor.metrics import MonitorMetrics
from ender3monitor.timelapse import TimelapseManager

CAPTURE_INTERVAL = 30       # seconds between analysis frames
TIMELAPSE_INTERVAL = 60     # seconds between timelapse frames

# Print-complete detection: stop after this many consecutive still frames
NO_MOTION_LIMIT = 4         # 4 × 30 s = 2 minutes
MOTION_THRESHOLD = 5.0      # mean absolute pixel difference (0–255) to count as "changed"

# Failure confirmation: require this many consecutive frames before alerting.
# Eliminates single-frame false positives — a real failure persists.
FAILURE_CONFIRM_FRAMES = 2  # 2 × 30 s = 1 minute of sustained failure


def _frames_differ(f1: np.ndarray, f2: np.ndarray, threshold: float = MOTION_THRESHOLD) -> bool:
    """Return True if the two frames differ significantly (motion / print activity)."""
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
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

        self._running = False
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # UI state
        self.status = "Idle"
        self.last_result: Optional[AnalysisResult] = None
        self.frame_count = 0
        self.failure_count = 0
        self.last_frame_time: Optional[datetime] = None

        # Print-complete detection
        self._no_motion_count: int = 0
        self._prev_analysis_frame: Optional[np.ndarray] = None

        # Failure confirmation — require consecutive frames before alerting
        self._pending_failure_type: Optional[str] = None
        self._pending_failure_count: int = 0

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

                # Open, grab one frame, release immediately — this prevents
                # OpenCV's internal capture thread from running continuously
                # at 30 fps between monitoring intervals.
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

                    # ── Motion / completion detection ──────────────────── #
                    if self._prev_analysis_frame is not None:
                        if _frames_differ(frame, self._prev_analysis_frame):
                            if self._no_motion_count > 0:
                                print(
                                    f"\n  [MOTION] Change detected — "
                                    f"resetting idle counter (was {self._no_motion_count})"
                                )
                            self._no_motion_count = 0
                        else:
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
                                # Track consecutive frames with the same failure type
                                if result.failure_type == self._pending_failure_type:
                                    self._pending_failure_count += 1
                                else:
                                    self._pending_failure_type = result.failure_type
                                    self._pending_failure_count = 1

                                self.status = (
                                    f"Possible failure – {result.failure_type} "
                                    f"({self._pending_failure_count}/{FAILURE_CONFIRM_FRAMES})"
                                    if self._pending_failure_count < FAILURE_CONFIRM_FRAMES
                                    else f"FAILURE DETECTED – {result.failure_type}"
                                )

                                if self._pending_failure_count == FAILURE_CONFIRM_FRAMES:
                                    # Confirmed — alert once per incident
                                    self.failure_count += 1
                                    self._no_motion_count = 0
                                    self._send_alert(result, frame)
                            else:
                                # Clean frame — reset pending failure
                                self._pending_failure_type = None
                                self._pending_failure_count = 0
                                if result.failure_type == "no_printer":
                                    self.status = "Monitoring… (no printer in frame)"
                                else:
                                    self.status = "Monitoring…"

                    if self._running:
                        _print_status(self)

        finally:
            self.metrics.monitoring_active.set(0)
            # snapshot() releases the camera after every frame, so nothing
            # persistent to clean up here.

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

    # ------------------------------------------------------------------ #
    # Public control methods                                               #
    # ------------------------------------------------------------------ #

    def start(self, camera_index: Optional[int] = None) -> None:
        if self._running:
            print("  Already monitoring.")
            return

        # Camera setup — caller can override (e.g. web UI passes chosen index)
        if camera_index is not None:
            idx = camera_index
        elif self.config.camera_index == -1:
            idx = CameraManager.select_camera()
        else:
            idx = self.config.camera_index

        self.camera = CameraManager(idx, flip=self.config.camera_flip)
        # Verify the camera is reachable before starting the thread
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
        self._no_motion_count = 0
        self._prev_analysis_frame = None
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
        if self.status != "Print Complete":
            self.status = "Idle"
        print("  Monitoring stopped.")

    def compile_timelapse(self) -> None:
        print("  Compiling timelapse…")
        result = self.timelapse.compile()
        if result:
            print(f"  Saved: {result}")


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
