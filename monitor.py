#!/usr/bin/env python3
"""Ender3Monitor – 3D print failure detection system."""

import sys
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np

from config import Config
from camera import CameraManager
from analyzer import PrintAnalyzer, AnalysisResult
from notifier import EmailNotifier
from metrics import MonitorMetrics
from timelapse import TimelapseManager

CAPTURE_INTERVAL = 30       # seconds between analysis frames
TIMELAPSE_INTERVAL = 60     # seconds between timelapse frames


def _clear_line() -> None:
    print("\r" + " " * 80 + "\r", end="", flush=True)


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
        self.analyzer = PrintAnalyzer(config.anthropic_api_key)
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
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # UI state
        self.status = "Idle"
        self.last_result: Optional[AnalysisResult] = None
        self.frame_count = 0
        self.failure_count = 0
        self.last_frame_time: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    # Monitoring loop                                                       #
    # ------------------------------------------------------------------ #

    def _monitoring_loop(self) -> None:
        last_capture = 0.0
        last_timelapse = 0.0

        self.metrics.monitoring_active.set(1)
        try:
            while self._running:
                now = time.time()
                frame = self.camera.capture_frame() if self.camera else None

                if frame is None:
                    with self._lock:
                        self.status = "Camera error – no frame"
                    time.sleep(2)
                    continue

                # Timelapse frame every 60 s
                if now - last_timelapse >= TIMELAPSE_INTERVAL:
                    self.timelapse.save_frame(frame)
                    last_timelapse = now

                # Analysis frame every 30 s
                if now - last_capture >= CAPTURE_INTERVAL:
                    last_capture = now
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

                        if (
                            result.failure_detected
                            and result.confidence >= self.config.confidence_threshold
                        ):
                            self.failure_count += 1
                            self.status = f"FAILURE DETECTED – {result.failure_type}"
                            self._send_alert(result, frame)
                        else:
                            self.status = "Monitoring…"

                    _print_status(self)

                time.sleep(1)
        finally:
            self.metrics.monitoring_active.set(0)

    def _send_alert(self, result: AnalysisResult, frame: np.ndarray) -> None:
        try:
            self.notifier.send_alert(result, frame)
            print(f"\n  [EMAIL] Alert sent to {self.config.smtp_recipient}")
        except Exception as exc:
            print(f"\n  [EMAIL ERROR] {exc}")

    # ------------------------------------------------------------------ #
    # Public control methods                                               #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._running:
            print("  Already monitoring.")
            return

        # Camera setup
        if self.config.camera_index == -1:
            idx = CameraManager.select_camera()
        else:
            idx = self.config.camera_index

        self.camera = CameraManager(idx)
        try:
            self.camera.open()
        except RuntimeError as exc:
            print(f"  Camera error: {exc}")
            return

        self.timelapse.reset_session()
        self._running = True
        self.frame_count = 0
        self.failure_count = 0
        self.last_result = None
        self.status = "Monitoring…"

        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitor_thread.start()
        print(f"  Monitoring started (camera {idx}). Analysis every {CAPTURE_INTERVAL}s.")

    def stop(self) -> None:
        if not self._running:
            print("  Not currently monitoring.")
            return
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        if self.camera:
            self.camera.release()
            self.camera = None
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
    _clear_line()
    print(
        f"  [{ts}] Status: {mon.status:<30} | "
        f"Frames: {mon.frame_count:>4} | Failures: {mon.failure_count:>3} | "
        f"Confidence: {conf} | Type: {ftype}"
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
