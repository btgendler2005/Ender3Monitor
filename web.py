#!/usr/bin/env python3
"""Ender3Monitor – Web UI.

    python web.py            # start on default port 8080
    python web.py 9090       # custom port

Then open http://localhost:8080 in your browser.
The CLI (python monitor.py) continues to work independently.
"""
from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import secrets
import signal
import threading
import time
import traceback
from contextlib import asynccontextmanager
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Optional, Set

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse, FileResponse
from starlette.datastructures import Headers
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ender3monitor")

from ender3monitor.camera import CameraManager
from ender3monitor.camera_worker import FrameExchange, capture_worker, frame_shm_size
from ender3monitor.config import Config
from ender3monitor.telegram_bot import TelegramBot
from ender3monitor.settings import Settings
from ender3monitor import ops_metrics as ops
from monitor import Monitor, _FLIP_STR_TO_CODE

# "spawn" (not "fork") gives the capture worker a genuinely fresh interpreter —
# and, critically on macOS, a fresh AVFoundation session. See camera_worker.py.
_MP_CTX = mp.get_context("spawn")

# ── global state ──────────────────────────────────────────────────────────────

_config: Optional[Config] = None
_monitor: Optional[Monitor] = None
_clients: Set[WebSocket] = set()
_telegram: Optional[TelegramBot] = None
DEFAULT_PORT = 8080

# ── Live-view tuning ────────────────────────────────────────────────────────
# Env-overridable so weaker hardware (e.g. a Raspberry Pi) can dial the live
# stream down without code edits. Analysis quality is unaffected — it samples
# raw frames independently of these.
_STREAM_FPS = int(os.getenv("STREAM_FPS", "12"))          # live-view fps to the browser
_STREAM_QUALITY = int(os.getenv("STREAM_QUALITY", "70"))  # live-view JPEG quality
_STREAM_WIDTH = int(os.getenv("STREAM_WIDTH", "1280"))
_STREAM_HEIGHT = int(os.getenv("STREAM_HEIGHT", "720"))


class StreamCapture:
    """Single persistent camera owner.

    A supervisor thread keeps a capture *worker process* (see
    ender3monitor/camera_worker.py) running and reads frames it writes into a
    shared-memory buffer. The MJPEG stream serves the latest frame at
    _STREAM_FPS for a smooth live view, while the analysis loop and timelapse
    sample it on their own (much slower) schedule.

    The camera runs in a child process (not opened directly here) so that a
    USB camera dropping mid-session — e.g. a docking station getting unplugged
    — can be recovered by killing and respawning that one process. On macOS,
    OpenCV's AVFoundation backend gets left in a wedged state for the rest of
    a process's life once a camera drops out from under it; recreating the
    cv2.VideoCapture object in the *same* process (the old approach here)
    doesn't reliably recover it, even after the camera is replugged — only a
    fresh process does. A brand-new child process each time gets exactly that,
    without restarting the printer connection, web server, or anything else.
    """

    def __init__(self, index: int, flip: Optional[int],
                 width: int = _STREAM_WIDTH, height: int = _STREAM_HEIGHT,
                 fps: int = _STREAM_FPS, quality: int = _STREAM_QUALITY) -> None:
        self.index = index
        self.flip = flip
        self.width = width
        self.height = height
        self._encode_interval = 1.0 / max(1, fps)
        self.quality = quality

        self._latest_raw = None       # numpy BGR, already flipped
        self._latest_jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._running = False
        self._paused = False          # when True, kill the worker (for scans)
        self._thread: Optional[threading.Thread] = None

        # Worker-process state — touched only from the supervisor thread (_loop).
        self._process: Optional[mp.process.BaseProcess] = None
        self._shm: Optional[SharedMemory] = None
        self._frames: Optional[FrameExchange] = None
        self._frame_ready = None
        self._stop_worker = None
        self._last_spawn_attempt = 0.0
        self._RESPAWN_BACKOFF = 1.0   # seconds between respawn attempts

        # Demand tracking — with no MJPEG viewers and no recent snapshot
        # requests, the loop drops to ~2 fps reads and ~0.5 fps encodes instead
        # of reading the camera at full rate and JPEG-encoding 12 fps around the
        # clock. Analysis/timelapse sample latest_frame() at ≥5 s cadence, so a
        # ≤0.5 s-old idle frame is just as good for them.
        self._viewers = 0                  # active MJPEG stream connections
        self._last_jpeg_demand = 0.0       # last latest_jpeg() call (snapshots)
        self._IDLE_DEMAND_WINDOW = 5.0     # seconds a snapshot keeps us "active"
        self._IDLE_READ_SLEEP = 0.5        # idle: ~2 fps camera reads
        self._IDLE_ENCODE_INTERVAL = 2.0   # idle: refresh the jpeg every 2 s
        # Set by add_viewer()/latest_jpeg() so an idle loop wakes immediately
        # instead of finishing its _IDLE_READ_SLEEP before serving a new viewer.
        self._wake = threading.Event()

        # Push-based frame delivery: every published jpeg bumps _jpeg_seq and
        # wakes the asyncio waiters registered by next_jpeg(), so MJPEG viewers
        # get a frame the moment it is encoded rather than on their own poll
        # tick (which added ~40 ms average latency and dropped frames to beat).
        self._jpeg_seq = 0
        self._jpeg_ts = 0.0
        self._waiters: set = set()   # {(loop, asyncio.Event)}

    # ── lifecycle ──
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    # ── worker process management (supervisor-thread only) ──
    def _spawn_worker(self) -> None:
        self._last_spawn_attempt = time.time()
        shm = SharedMemory(create=True, size=frame_shm_size(self.width, self.height))
        frame_ready = _MP_CTX.Event()
        stop_worker = _MP_CTX.Event()
        proc = _MP_CTX.Process(
            target=capture_worker,
            args=(self.index, self.width, self.height,
                  shm.name, frame_ready, stop_worker),
            daemon=True,
        )
        proc.start()
        self._process, self._shm = proc, shm
        self._frames = FrameExchange(shm.buf, self.width, self.height)
        self._frame_ready, self._stop_worker = frame_ready, stop_worker

    def _teardown_worker(self) -> None:
        if self._stop_worker is not None:
            self._stop_worker.set()
        if self._process is not None:
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        if self._frames is not None:
            self._frames.release()   # numpy views would block shm.close()
            self._frames = None
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except FileNotFoundError:
                pass
        self._process = self._shm = None
        self._frame_ready = self._stop_worker = None

    def _loop(self) -> None:
        next_encode = 0.0
        while self._running:
            # Paused (or reindexing): drop the worker so a scan can use the camera.
            if self._paused:
                if self._process is not None:
                    self._teardown_worker()
                time.sleep(0.1)
                continue

            if self._process is None or not self._process.is_alive():
                if self._process is not None:
                    ops.camera_frames_total.labels("fail").inc()
                    self._teardown_worker()   # reap the dead worker, free its shm
                if time.time() - self._last_spawn_attempt < self._RESPAWN_BACKOFF:
                    time.sleep(0.05)
                    continue
                self._spawn_worker()
                time.sleep(0.05)
                continue

            if not self._frame_ready.wait(timeout=0.5):
                continue   # no new frame yet; loop back and re-check liveness
            self._frame_ready.clear()

            # cv2.flip already writes a new array, so flip straight from the
            # shared slot instead of copying first and flipping the copy.
            # read() hands back only a frame the worker didn't touch mid-copy.
            flip = self.flip
            frame, _ = self._frames.read(
                (lambda v: cv2.flip(v, flip)) if flip is not None else None)
            if frame is None:
                continue   # writer lapped us repeatedly; take the next frame
            ops.camera_frames_total.labels("ok").inc()
            with self._lock:
                self._latest_raw = frame
            now = time.time()
            active = self._has_demand()
            encode_interval = (self._encode_interval if active
                               else self._IDLE_ENCODE_INTERVAL)
            # Schedule encodes on a fixed timeline (next_encode += interval)
            # rather than "interval since the last encode": frames only arrive
            # every ~33-40 ms, so the latter always overshoots to the next
            # frame and a 12 fps target actually ran at ~9 fps on a 26 fps
            # camera. Re-anchor if we fell behind or the interval shrank
            # (idle -> active), so a new viewer gets a frame immediately.
            if next_encode - now > encode_interval or now - next_encode > encode_interval:
                next_encode = now
            if now >= next_encode:
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ok2:
                    self._publish_jpeg(buf.tobytes())
                next_encode += encode_interval
            if not active:
                # Idle: don't spin the camera at full rate for nobody — but
                # wake at once if a viewer or snapshot request arrives.
                self._wake.wait(self._IDLE_READ_SLEEP)
                self._wake.clear()
        self._teardown_worker()

    def _publish_jpeg(self, data: bytes) -> None:
        with self._lock:
            self._latest_jpeg = data
            self._jpeg_seq += 1
            self._jpeg_ts = time.time()
            waiters = list(self._waiters)
        for loop, ev in waiters:
            try:
                loop.call_soon_threadsafe(ev.set)
            except RuntimeError:
                pass   # that event loop has closed

    # ── controls ──
    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def set_index(self, index: int) -> None:
        """Switch to a different camera. Briefly pauses so the old worker is
        torn down before the new one is spawned."""
        if index == self.index:
            return
        self.index = index
        with self._lock:
            self._latest_raw = None
            self._latest_jpeg = None
        self._paused = True
        time.sleep(0.3)
        self._paused = False

    # ── demand tracking ──
    def add_viewer(self) -> None:
        with self._lock:
            self._viewers += 1
        self._wake.set()

    def remove_viewer(self) -> None:
        with self._lock:
            self._viewers = max(0, self._viewers - 1)

    def _has_demand(self) -> bool:
        return (self._viewers > 0
                or (time.time() - self._last_jpeg_demand) < self._IDLE_DEMAND_WINDOW)

    # ── readers ──
    def latest_jpeg(self) -> Optional[bytes]:
        # Reading the jpeg signals demand — wakes the encoder for a few seconds
        # so an on-demand /snapshot (UI, Telegram) gets a fresh image next call.
        self._last_jpeg_demand = time.time()
        self._wake.set()
        with self._lock:
            return self._latest_jpeg

    def jpeg_age(self) -> float:
        """Seconds since the latest jpeg was encoded (inf if none yet)."""
        with self._lock:
            return time.time() - self._jpeg_ts if self._latest_jpeg is not None else float("inf")

    async def next_jpeg(self, last_seq: int, timeout: float):
        """Return (seq, jpeg) as soon as a jpeg newer than `last_seq` exists.

        Returns immediately if one already does; otherwise waits up to
        `timeout` seconds for the capture thread to publish one and returns
        whatever is current (possibly unchanged) when the wait ends.
        """
        loop = asyncio.get_running_loop()
        ev = asyncio.Event()
        waiter = (loop, ev)
        with self._lock:
            if self._jpeg_seq != last_seq and self._latest_jpeg is not None:
                return self._jpeg_seq, self._latest_jpeg
            self._waiters.add(waiter)
        try:
            await asyncio.wait_for(ev.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            with self._lock:
                self._waiters.discard(waiter)
        with self._lock:
            return self._jpeg_seq, self._latest_jpeg

    def latest_frame(self):
        """Return a copy of the latest raw frame (for analysis/timelapse)."""
        with self._lock:
            return None if self._latest_raw is None else self._latest_raw.copy()


# The single shared capture instance (created in lifespan once config is loaded)
_stream: Optional[StreamCapture] = None

# Set the instant a shutdown signal arrives so long-lived responses (the MJPEG
# stream) can stop themselves and let the connection close, instead of being
# force-cancelled past uvicorn's graceful window (which logs a CancelledError).
_shutting_down = threading.Event()


def _install_shutdown_hook() -> None:
    """Chain a flag-setting handler ahead of uvicorn's signal handlers.

    Called from lifespan startup, by which point uvicorn has already installed
    its own SIGINT/SIGTERM handlers — we wrap them so the printer's normal
    shutdown still runs, but our streaming generators learn about it first.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous = signal.getsignal(sig)

        def handler(signum, frame, _previous=previous):
            _shutting_down.set()
            if callable(_previous):
                _previous(signum, frame)

        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Not in the main thread (e.g. under --reload) — skip; the graceful
            # timeout still bounds shutdown, just a touch less tidily.
            pass


def _resolve_stream_index() -> int:
    """Pick the camera index for the live stream at startup."""
    if _config and _config.camera_index >= 0:
        return _config.camera_index
    return 0   # default; user can switch in the UI


# ── Telegram command handlers ─────────────────────────────────────────────────

def _printer_status_lines() -> list:
    pr = _monitor.printer.status if _monitor else None
    if not (pr and pr.connected):
        return ["Printer: not connected"]
    def t(v, tgt):
        return "—" if v is None else f"{round(v)}°" + (f"/{round(tgt)}°" if tgt else "")
    lines = [f"Nozzle: {t(pr.nozzle_temp, pr.nozzle_target)}   Bed: {t(pr.bed_temp, pr.bed_target)}"]
    if pr.filament_runout:
        lines.append("🧵 Paused — filament runout")
    elif pr.filament_change_pause:
        lines.append("🎨 Paused — change filament (M600)")
    if pr.printing and pr.progress is not None:
        d = pr.as_dict()
        extra = []
        if d.get("layer_str"):     extra.append("layer " + d['layer_str'])
        if d.get("elapsed_str"):   extra.append(d['elapsed_str'] + " elapsed")
        if d.get("remaining_str"): extra.append("~" + d['remaining_str'] + " left")
        lines.append(f"Progress: {pr.progress*100:.1f}%   " + "   ".join(extra))
        if d.get("feature"):
            lines.append(f"Printing: {d['feature']}")
        if d.get("next_color_change_str"):
            lines.append(f"🎨 Color change in ~{d['next_color_change_str']}")
    elif pr.printing:
        lines.append("Printing…")
    return lines


def _tg_status(args):
    s = _state()
    out = ["*Ender3Monitor*", f"Status: {s.get('status')}",
           f"Frames: {s.get('frame_count', 0)}   Failures: {s.get('failure_count', 0)}"]
    ft = s.get("failure_type")
    if ft and ft not in ("none", "no_printer"):
        out.append(f"Last: {ft} ({(s.get('confidence') or 0)*100:.0f}%)")
    out += _printer_status_lines()
    life = s.get("printer", {}).get("lifetime_str")
    out.append(f"Lifetime: {life} (printer)" if life
               else f"Lifetime: {_monitor.maintenance.total_hours:.1f} h (tracked)")
    return "\n".join(out)


def _tg_snapshot(args):
    jpeg = _stream.latest_jpeg() if _stream else None
    return ("photo", jpeg, "📷 Live snapshot")


def _require_printer():
    return _monitor and _monitor.printer.connected


def _tg_pause(args):
    if not _require_printer(): return "Printer not connected."
    _monitor.printer.pause(); return "⏸ Print paused."


def _tg_resume(args):
    if not _require_printer(): return "Printer not connected."
    _monitor.printer.resume(); return "▶️ Print resumed."


def _tg_cooldown(args):
    if not _require_printer(): return "Printer not connected."
    _monitor.printer.cooldown(); return "❄️ Paused — heaters off."


def _tg_go(args):
    if _monitor is None: return "Not ready."
    if _monitor._running: return "Already monitoring."
    provider = _stream.latest_frame if _stream else None
    cam = _stream.index if _stream else None
    _monitor.start(camera_index=cam, frame_provider=provider)
    return "▶️ Monitoring started."


def _tg_stop(args):
    if _monitor is None: return "Not ready."
    _monitor.stop(); return "⏹ Monitoring stopped."


def _tg_maintenance(args):
    if _monitor is None:
        return "Not ready."
    lines = [_monitor.maintenance.summary()]
    life = _monitor.printer.status.as_dict().get("lifetime_str") if _monitor.printer.connected else None
    if life:
        lines.append(f"Printer lifetime (EEPROM): {life}")
    return "\n".join(lines)


def _tg_history(args):
    """/history → the last 5 recorded prints (date, name, outcome, duration, cost)."""
    store = getattr(_monitor, "history", None)
    if store is None:
        return "Not ready."
    from ender3monitor.history import format_recent
    return format_recent(store, limit=5)


def _tg_autostart(args):
    if _monitor is None:
        return "Not ready."
    arg = (args[0].lower() if args else "")
    if arg in ("on", "off"):
        _monitor.settings.update({"auto_start_on_print": arg == "on"})
    return f"Auto-start is {'ON' if _monitor.auto_start_enabled else 'OFF'}."


def _tg_ask(args):
    """Free-form question about the live frame, answered by the vision model."""
    if _monitor is None or _stream is None:
        return "Not ready."
    question = " ".join(args).strip()
    if not question:
        return "Ask me something about the print, e.g. `/ask is the first layer sticking?`"
    frame = _stream.latest_frame()
    if frame is None:
        return "No camera frame available right now."
    return _monitor.analyzer.ask(frame, question)


_TIMELAPSE_LIST_SIZE = 3


def _read_mp4_as_video_reply(mp4: str, caption: str):
    try:
        with open(mp4, "rb") as f:
            blob = f.read()
    except Exception as exc:
        return f"Compiled but couldn't read the file: {exc}"
    if len(blob) > 49 * 1024 * 1024:
        return (f"Timelapse compiled ({len(blob) / 1e6:.0f} MB) but exceeds "
                f"Telegram's 50 MB bot upload limit.\nSaved at: {mp4}")
    return ("video", blob, caption)


def _tg_timelapse_list(args):
    sessions = _monitor.timelapse.list_sessions(limit=_TIMELAPSE_LIST_SIZE)
    if not sessions:
        return "No timelapse sessions yet."
    items = [f"Last {len(sessions)} timelapse session(s) — "
             f"reply `/timelapse <n>` to compile + send one:"]
    for i, s in enumerate(sessions, start=1):
        caption = f"{i}) {s['name']} — {s['frame_count']} frame(s)"
        if s["last_frame"] is not None:
            try:
                jpeg = s["last_frame"].read_bytes()
                items.append(("photo", jpeg, caption))
                continue
            except Exception:
                pass
        items.append(caption + " (no frames)")
    return items


def _tg_timelapse_pick(index: int):
    sessions = _monitor.timelapse.list_sessions(limit=_TIMELAPSE_LIST_SIZE)
    if not (1 <= index <= len(sessions)):
        return f"No session #{index}. Run /timelapse list to see the available ones."
    session = sessions[index - 1]
    mp4 = _monitor.compile_timelapse_session(session["path"])
    if not mp4:
        return f"Session {session['name']} has no frames to compile."
    return _read_mp4_as_video_reply(mp4, f"Timelapse — {session['name']}")


def _tg_timelapse(args):
    """/timelapse           → current/most-recent session, compiled + sent as video
       /timelapse list      → last few sessions with a thumbnail of each's final frame
       /timelapse <n>       → compile + send the n-th session from that list
    """
    if _monitor is None:
        return "Not ready."
    sub = args[0].lower() if args else ""
    if sub == "list":
        return _tg_timelapse_list(args)
    if sub.isdigit():
        return _tg_timelapse_pick(int(sub))

    mp4 = _monitor.compile_timelapse()
    if not mp4:
        return "No timelapse frames yet — start monitoring a print first."
    return _read_mp4_as_video_reply(mp4, "Timelapse")


def _tg_failure_callback(data: str) -> str:
    """Inline-button press under a failure alert → resolve the pending review."""
    from monitor import parse_failure_review_callback
    parsed = parse_failure_review_callback(data)
    if parsed is None or _monitor is None:
        return "Unknown button."
    action, review_id = parsed
    ops.telegram_commands_total.labels(f"failure_{action}").inc()
    res = _monitor.resolve_failure_review(action, review_id)
    icon = {"resume": "▶️", "keep_paused": "⏸", "false_alarm": "🚫"}[action]
    return f"{icon} {res['message']}" if res["ok"] else res["message"]


def _build_telegram_handlers():
    handlers = {
        "status":   (_tg_status,   "current status + temps + progress"),
        "snapshot": (_tg_snapshot, "live camera photo"),
        "photo":    (_tg_snapshot, "alias of /snapshot"),
        "ask":      (_tg_ask,      "ask about the print, e.g. /ask how's the first layer?"),
        "pause":    (_tg_pause,    "pause the print"),
        "resume":   (_tg_resume,   "resume the print"),
        "cooldown": (_tg_cooldown, "pause + heaters off"),
        "go":       (_tg_go,       "start monitoring"),
        "stop":     (_tg_stop,     "stop monitoring"),
        "maintenance": (_tg_maintenance, "print hours + upkeep status"),
        "history":  (_tg_history,  "last 5 prints: outcome, duration, AI cost"),
        "autostart": (_tg_autostart, "auto-start on print: /autostart on|off"),
        "timelapse": (_tg_timelapse, "compile + send the timelapse so far; "
                                     "/timelapse list or /timelapse <n> for older ones"),
    }

    def _help(args):
        lines = ["*Ender3Monitor commands*"]
        for cmd, (_, desc) in handlers.items():
            if desc.startswith("alias"):
                continue
            lines.append(f"/{cmd} — {desc}")
        lines.append("/help — this list")
        return "\n".join(lines)

    handlers["help"] = (_help, "alias")             # listed manually below
    handlers["start"] = (_help, "alias of /help")   # /start is Telegram's default

    # Wrap every handler to count command usage in Prometheus.
    def _counted(name, fn):
        def wrapper(args):
            ops.telegram_commands_total.labels(name).inc()
            return fn(args)
        return wrapper
    return {cmd: (_counted(cmd, fn), desc) for cmd, (fn, desc) in handlers.items()}


def _parse_allowed_chats(raw: str) -> set:
    out = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if part:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _monitor, _stream, _telegram
    _install_shutdown_hook()
    _config = Config.from_env()
    _monitor = Monitor(_config)
    ops.register_system_collector()          # host CPU/mem/disk on the metrics endpoint
    _monitor.metrics.start_server(_config.metrics_port)
    _stream = StreamCapture(_resolve_stream_index(),
                            flip=_config.camera_flip)
    _stream.start()
    # Let the printer poller auto-start monitoring using the shared stream.
    _monitor.set_default_source(_stream.index, _stream.latest_frame)

    # Interactive Telegram bot (optional)
    if _config.telegram_bot_token:
        allowed = _parse_allowed_chats(_config.telegram_allowed_chats)
        _telegram = TelegramBot(_config.telegram_bot_token, allowed, _build_telegram_handlers(),
                                callback_handler=_tg_failure_callback)
        _telegram.start()
        _monitor.telegram_review_buttons = True   # failure alerts get reply buttons
        log.info("Telegram bot started (authorized chats: %s)",
                 allowed if allowed else "none yet — DM the bot to get your chat ID")

    push_task = asyncio.create_task(_push_loop())
    yield
    push_task.cancel()
    try:
        await push_task
    except asyncio.CancelledError:
        pass
    if _telegram:
        _telegram.stop()
    if _monitor and _monitor._running:
        _monitor.stop()
    if _monitor:
        _monitor.close()       # disconnect printer / stop temp poller
    if _stream:
        _stream.stop()


app = FastAPI(title="Ender3Monitor", lifespan=lifespan)


# ── HTTP Basic Auth (optional, via WEB_USERNAME / WEB_PASSWORD) ───────────────
# Gates every HTTP route and the websocket. Without it, anyone on the LAN can
# watch the camera and drive the printer (pause / e-stop / heaters).

def _auth_enabled() -> bool:
    return bool(_config and _config.web_username and _config.web_password)


def _check_basic_auth(header: str) -> bool:
    """Validate an 'Authorization: Basic …' header against the configured creds."""
    if not header.startswith("Basic "):
        return False
    try:
        user, _, pwd = base64.b64decode(header[6:]).decode("utf-8").partition(":")
    except Exception:
        return False
    # compare_digest on both fields — no early-exit timing oracle
    return (secrets.compare_digest(user, _config.web_username)
            and secrets.compare_digest(pwd, _config.web_password))


# NOTE: these are pure-ASGI middlewares (not @app.middleware("http"), which is
# Starlette's BaseHTTPMiddleware). BaseHTTPMiddleware wraps responses in an
# anyio memory stream; when a long-lived StreamingResponse like /stream is
# cancelled at shutdown it raises a noisy "Exception in ASGI application"
# CancelledError traceback. Pure-ASGI middleware passes cancellation straight
# through and shuts down cleanly.

_ICON_SIZES = (180, 192, 512)
_PUBLIC_PATHS = {"/manifest.webmanifest"} | {f"/icon-{n}.png" for n in _ICON_SIZES}


class _AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not _auth_enabled():
            await self.app(scope, receive, send)
            return
        # Home-screen install assets: the OS fetches these without the page's
        # Basic credentials, and they reveal nothing about the printer.
        if scope.get("path") in _PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        if _check_basic_auth(Headers(scope=scope).get("authorization", "")):
            await self.app(scope, receive, send)
            return
        response = Response(
            status_code=401,
            content="Authentication required",
            headers={"WWW-Authenticate": 'Basic realm="Ender3Monitor"'},
        )
        await response(scope, receive, send)


# ── Prometheus HTTP request metrics ───────────────────────────────────────────

# /stream is a long-lived MJPEG response — its "duration" is the whole stream
# lifetime, so we count it but don't pollute the latency histogram with it.
_NO_LATENCY_PATHS = {"/stream"}


class _MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        ops.http_requests_in_progress.inc()
        start = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            ops.http_requests_in_progress.dec()
            # Use the matched route template (not the raw URL) to bound cardinality.
            route = scope.get("route")
            path = getattr(route, "path", None) or "other"
            method = scope.get("method", "")
            ops.http_requests_total.labels(method, path, str(status["code"])).inc()
            if path not in _NO_LATENCY_PATHS:
                ops.http_request_duration_seconds.labels(method, path).observe(
                    time.perf_counter() - start)


# Add auth first, then metrics, so metrics wraps auth (outermost) and counts
# 401s too — matching the original decorator ordering.
app.add_middleware(_AuthMiddleware)
app.add_middleware(_MetricsMiddleware)


# ── WebSocket broadcast ───────────────────────────────────────────────────────

_PUSH_CHECK_INTERVAL = 0.5   # how often state is checked for changes
_PUSH_KEEPALIVE = 10.0       # resend unchanged state at least this often
_WS_SEND_TIMEOUT = 2.0       # a client slower than this is dropped, not waited on
_last_push_msg: Optional[str] = None
_last_push_time = 0.0


async def _send_all(msg: str) -> None:
    """Send `msg` to every browser concurrently; drop clients that fail or stall."""
    global _last_push_msg, _last_push_time
    _last_push_msg, _last_push_time = msg, time.monotonic()
    clients = list(_clients)
    if not clients:
        return
    results = await asyncio.gather(
        *(asyncio.wait_for(ws.send_text(msg), _WS_SEND_TIMEOUT) for ws in clients),
        return_exceptions=True,
    )
    for ws, res in zip(clients, results):
        if isinstance(res, BaseException):
            _clients.discard(ws)
    ops.ws_clients.set(len(_clients))


async def _push_loop() -> None:
    """Push state to connected browsers as soon as it changes.

    Checks every 0.5 s but only sends when the serialized state differs from
    the last push (or the keepalive lapses), so status/temperature changes
    reach the UI ~4x sooner than the old fixed 2 s push while idle traffic —
    and browser re-renders — drop to almost nothing.
    """
    while True:
        await asyncio.sleep(_PUSH_CHECK_INTERVAL)
        if not _clients:
            continue
        msg = json.dumps(_state())
        if (msg != _last_push_msg
                or time.monotonic() - _last_push_time >= _PUSH_KEEPALIVE):
            await _send_all(msg)


def _failure_review_state() -> Optional[dict]:
    """The pending failure review for the UI — everything but the JPEG bytes."""
    fr = _monitor.failure_review
    if fr is None:
        return None
    return {
        "id": fr["id"],
        "failure_type": fr["failure_type"],
        "confidence": fr["confidence"],
        "description": fr["description"],
        "timestamp": fr["timestamp"],
        "image_url": (f"/api/failure/snapshot?t={fr['id']}" if fr.get("jpeg") else None),
    }


def _timelapse_state() -> dict:
    """How this session's timelapse frames are being taken, for the dashboard.

    mode: "parked" once the sliced file's snapshot marker has been seen (head
    parked in the same spot every frame), else "layer" (Z-step) or "time".
    """
    if getattr(_monitor, "_timelapse_synced", False):
        mode = "parked"
    elif getattr(_monitor, "_timelapse_layer_mode", False):
        mode = "layer"
    else:
        mode = "time"
    return {"mode": mode, "frames": _monitor.timelapse.frame_count}


def _state() -> dict:
    if _monitor is None:
        return {"status": "starting", "is_running": False}
    r = _monitor.last_result
    return {
        "status": _monitor.status,
        "is_running": _monitor._running,
        "frame_count": _monitor.frame_count,
        "failure_count": _monitor.failure_count,
        "confidence": r.confidence if r else 0.0,
        "failure_type": r.failure_type if r else "none",
        "failure_detected": r.failure_detected if r else False,
        "description": r.description if r else "",
        "backend": r.backend if r else "",
        "camera_index": _monitor.camera.camera_index if _monitor.camera else None,
        "printer": _monitor.printer.status.as_dict(),
        "push_channels": _monitor.push.channels(),
        "auto_start": _monitor.auto_start_enabled,
        "threshold": _monitor.settings.get("confidence_threshold"),
        "cost_usd": round(_monitor.session_cost_usd, 4),
        "events": list(_monitor.events),
        "failure_review": _failure_review_state(),
        "false_alarm_count": _monitor.false_alarm_count,
        "confidence_history": list(_monitor.confidence_history),
        "timelapse": _timelapse_state(),
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    # http-middleware doesn't cover the websocket scope — enforce auth here.
    # Browsers re-send the page's cached Basic credentials on the WS upgrade.
    if _auth_enabled() and not _check_basic_auth(ws.headers.get("authorization", "")):
        await ws.close(code=4401)
        return
    await ws.accept()
    _clients.add(ws)
    ops.ws_clients.set(len(_clients))
    await ws.send_text(json.dumps(_state()))
    try:
        while True:
            await ws.receive_text()   # keep-alive drain
    except WebSocketDisconnect:
        pass
    finally:
        # Any exit (not only a clean disconnect) must unregister the client,
        # or broadcasts keep trying to write to a dead socket.
        _clients.discard(ws)
        ops.ws_clients.set(len(_clients))


# ── REST API ──────────────────────────────────────────────────────────────────

class AutoStartBody(BaseModel):
    enabled: bool


@app.post("/api/autostart")
async def api_autostart(body: AutoStartBody):
    if _monitor is None:
        return JSONResponse({"error": "not initialised"}, 503)
    # Through Settings so the choice persists (on_change syncs auto_start_enabled).
    _monitor.settings.update({"auto_start_on_print": body.enabled})
    await _broadcast()
    return {"auto_start": _monitor.auto_start_enabled}


_camera_cache: list = []   # last hardware scan result: [(index, w, h), ...]


@app.get("/api/cameras")
async def api_cameras(scan: bool = False):
    """Return available cameras.

    If CAMERA_INDEX is set (>= 0), return it immediately without opening
    any camera — fast path, no contention with the stream loop.

    Pass ?scan=true to force a full hardware scan regardless.
    """
    try:
        configured = _config.camera_index if _config else -1

        if not scan:
            # Fast path — never probe hardware on a plain page load. A scan
            # pauses the live capture and can take many seconds, which made
            # every page load slow and left the preview stalled until a manual
            # refresh. Return the last scan if we have one, otherwise just the
            # camera the stream is already using. The Rescan button passes
            # ?scan=true to force a real hardware scan.
            current = configured if configured >= 0 else (_stream.index if _stream else 0)
            cams = _camera_cache or [(current, 1280, 720)]
            return {
                "cameras": [{"index": i, "width": w, "height": h} for i, w, h in cams],
                "configured": current,
            }

        # Pause the persistent capture so the scan has exclusive access to the
        # camera hardware, then give it a moment to release its handle.
        if _stream:
            _stream.set_paused(True)
        await asyncio.sleep(0.4)
        try:
            loop = asyncio.get_running_loop()
            cameras = await asyncio.wait_for(
                loop.run_in_executor(None, CameraManager.list_available_cameras),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            log.warning("Camera scan timed out")
            return JSONResponse({"error": "Camera scan timed out", "cameras": [], "configured": configured})
        finally:
            if _stream:
                _stream.set_paused(False)
        _camera_cache[:] = cameras
        return {
            "cameras": [{"index": i, "width": w, "height": h} for i, w, h in cameras],
            "configured": configured if configured >= 0 else (_stream.index if _stream else 0),
        }
    except Exception:
        if _stream:
            _stream.set_paused(False)
        log.error("Unhandled error in /api/cameras:\n%s", traceback.format_exc())
        return JSONResponse({"error": "Internal error — see server log", "cameras": [], "configured": -1})


class StartBody(BaseModel):
    camera_index: Optional[int] = None


async def _broadcast() -> None:
    """Push current state to every connected browser immediately."""
    if not _clients:
        return
    await _send_all(json.dumps(_state()))


@app.post("/api/start")
async def api_start(body: StartBody = StartBody()):
    if _monitor is None:
        return JSONResponse({"error": "not initialised"}, 503)
    cam = body.camera_index
    if cam is None and _config and _config.camera_index >= 0:
        cam = _config.camera_index

    # Point the persistent capture at the chosen camera (if specified), then
    # hand the monitor that same capture as its frame source — no second handle.
    loop = asyncio.get_running_loop()
    if _stream is not None and cam is not None and cam != _stream.index:
        # set_index sleeps while the old worker is torn down — keep that off
        # the event loop so the stream and websocket don't freeze meanwhile.
        await loop.run_in_executor(None, _stream.set_index, cam)
        await asyncio.sleep(0.5)   # let the new camera produce a first frame

    provider = _stream.latest_frame if _stream is not None else None
    cam_for_monitor = cam if cam is not None else (_stream.index if _stream else None)

    await loop.run_in_executor(
        None,
        lambda: _monitor.start(camera_index=cam_for_monitor, frame_provider=provider),
    )
    await _broadcast()
    return _state()


@app.post("/api/stop")
async def api_stop():
    if _monitor is None:
        return JSONResponse({"error": "not initialised"}, 503)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _monitor.stop)
    await _broadcast()
    return _state()


@app.post("/api/timelapse")
async def api_timelapse():
    if _monitor is None:
        return JSONResponse({"error": "not initialised"}, 503)
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, _monitor.compile_timelapse)
    if not path:
        return JSONResponse(
            {"error": "No frames to compile yet — start monitoring first.", "file": None}
        )
    return {"message": "timelapse compiled", "file": Path(path).name}


class PrinterActionBody(BaseModel):
    action: str   # pause | resume | cooldown | estop


@app.post("/api/printer")
async def api_printer(body: PrinterActionBody):
    """Manual printer control over USB."""
    if _monitor is None:
        return JSONResponse({"error": "not initialised"}, 503)
    pr = _monitor.printer
    if not pr.connected:
        return JSONResponse({"error": "Printer not connected"}, 409)

    action = (body.action or "").lower()
    handlers = {
        "pause":    (pr.pause,           "Print paused"),
        "resume":   (pr.resume,          "Print resumed"),
        "cooldown": (pr.cooldown,        "Paused — heaters off"),
        "estop":    (pr.emergency_stop,  "Emergency stop sent — power-cycle the printer"),
    }
    if action not in handlers:
        return JSONResponse({"error": f"unknown action '{action}'"}, 400)

    fn, msg = handlers[action]
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, fn)
    await _broadcast()
    return {"message": msg}


class FailureResolveBody(BaseModel):
    action: str   # resume | keep_paused | false_alarm


@app.get("/api/failure/snapshot")
async def api_failure_snapshot():
    """The frame that triggered the pending failure review."""
    fr = _monitor.failure_review if _monitor is not None else None
    if fr is None or not fr.get("jpeg"):
        return JSONResponse({"error": "no failure snapshot"}, 404)
    return Response(fr["jpeg"], media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/failure/resolve")
async def api_failure_resolve(body: FailureResolveBody):
    """Answer the pending failure review: resume, keep paused, or false alarm."""
    if _monitor is None:
        return JSONResponse({"error": "not initialised"}, 503)
    from monitor import FAILURE_REVIEW_ACTIONS
    action = (body.action or "").lower()
    if action not in FAILURE_REVIEW_ACTIONS:
        return JSONResponse({"error": f"unknown action '{action}'"}, 400)

    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, _monitor.resolve_failure_review, action)
    await _broadcast()
    if not res["ok"]:
        return JSONResponse({"error": res["message"]}, 409)
    return {"message": res["message"]}


def _timelapse_dir() -> Path:
    return Path(_config.timelapse_dir).resolve() if _config else Path("timelapse_frames").resolve()


@app.get("/download/timelapse")
async def download_timelapse(name: Optional[str] = None):
    """Download a compiled timelapse MP4. With ?name=…, serves that file;
    otherwise serves the most recently compiled one. Path-traversal safe —
    only .mp4 files directly inside the timelapse dir are served."""
    base = _timelapse_dir()
    if name:
        # Reduce to a bare filename so "../" etc. can't escape the directory
        candidate = (base / Path(name).name).resolve()
    else:
        mp4s = sorted(base.glob("timelapse_*.mp4"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            return JSONResponse({"error": "No timelapse available."}, 404)
        candidate = mp4s[0].resolve()

    if candidate.parent != base or candidate.suffix != ".mp4" or not candidate.is_file():
        return JSONResponse({"error": "Timelapse not found."}, 404)

    return FileResponse(str(candidate), media_type="video/mp4", filename=candidate.name)


# ── Print history ─────────────────────────────────────────────────────────────

def _history_store():
    return getattr(_monitor, "history", None) if _monitor is not None else None


def _history_thumb_file(store, rec: Optional[dict]) -> Optional[Path]:
    """Resolved thumbnail path for a record, or None. Path-traversal safe —
    only .jpg files directly inside the history dir are served."""
    p = store.thumb_path_for(rec)
    if p is None:
        return None
    base = store.dir.resolve()
    candidate = (base / Path(p).name).resolve()
    if candidate.parent != base or candidate.suffix != ".jpg" or not candidate.is_file():
        return None
    return candidate


def _history_public(store, rec: dict) -> dict:
    """A record as the UI sees it, plus what is actually still on disk."""
    out = dict(rec)
    out["has_thumbnail"] = _history_thumb_file(store, rec) is not None
    tl = rec.get("timelapse")
    tl_ok = False
    if tl and isinstance(tl, str):
        base = _timelapse_dir()
        cand = (base / Path(tl).name).resolve()
        tl_ok = cand.parent == base and cand.suffix == ".mp4" and cand.is_file()
    out["timelapse_available"] = tl_ok
    return out


@app.get("/api/history")
async def api_history():
    """All recorded prints, newest first, without their event logs."""
    store = _history_store()
    if store is None:
        return JSONResponse({"error": "not initialised"}, 503)
    return {"prints": [_history_public(store, r) for r in store.list()]}


@app.get("/api/history/{record_id}")
async def api_history_get(record_id: str):
    store = _history_store()
    if store is None:
        return JSONResponse({"error": "not initialised"}, 503)
    rec = store.get(record_id)
    if rec is None:
        return JSONResponse({"error": "Print not found."}, 404)
    return _history_public(store, rec)


@app.get("/api/history/{record_id}/thumb")
async def api_history_thumb(record_id: str):
    store = _history_store()
    if store is None:
        return JSONResponse({"error": "not initialised"}, 503)
    # The file is looked up via the stored record, never built from the URL.
    path = _history_thumb_file(store, store.get(record_id))
    if path is None:
        return JSONResponse({"error": "No thumbnail."}, 404)
    return FileResponse(str(path), media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})


@app.delete("/api/history/{record_id}")
async def api_history_delete(record_id: str):
    store = _history_store()
    if store is None:
        return JSONResponse({"error": "not initialised"}, 503)
    if not store.delete(record_id):
        return JSONResponse({"error": "Print not found."}, 404)
    return {"deleted": record_id}


@app.get("/api/status")
async def api_status():
    return _state()


@app.get("/api/temps")
async def api_temps():
    """Temperature history for the dashboard chart (kept out of the websocket
    state so pushes stay small). Rows: [unix_ts, nozzle, nozzle_target, bed,
    bed_target], oldest first."""
    from ender3monitor.printer import TEMP_HISTORY_SECONDS
    samples = _monitor.printer.temp_history() if _monitor is not None else []
    return {"window_seconds": TEMP_HISTORY_SECONDS, "now": time.time(),
            "samples": samples}


@app.post("/api/events/clear")
async def api_events_clear():
    if _monitor is None:
        return JSONResponse({"error": "not initialised"}, 503)
    _monitor.events.clear()
    await _broadcast()
    return {"cleared": True}


@app.get("/api/settings")
async def api_settings_get():
    """Schema + current values for the settings panel.

    Only ever returns the non-secret allowlist (Settings has no secret keys).
    `info` carries read-only context — never credentials.
    """
    if _monitor is None or _config is None:
        return JSONResponse({"error": "not initialised"}, 503)
    model = (_config.anthropic_model if _config.analyzer_backend == "anthropic"
             else _config.ollama_model)
    return {
        "schema": Settings.schema_public(),
        "values": _monitor.settings.public_dict(),
        "info": {
            "backend": _config.analyzer_backend,
            "model": model,
            "camera_index": _config.camera_index,
            "auth_enabled": _auth_enabled(),
        },
    }


@app.post("/api/settings")
async def api_settings_post(request: Request):
    """Validate + apply a batch of setting changes.

    Every key is validated server-side by Settings.update against the schema;
    unknown or out-of-range values are rejected and never stored. Only schema
    keys exist, so there is no path to read or write a secret here.
    """
    if _monitor is None:
        return JSONResponse({"error": "not initialised"}, 503)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, 400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "expected a JSON object of key→value"}, 400)

    applied, errors, restart_required = _monitor.settings.update(body)

    # camera_flip's stream side is web-owned (the monitor handled CLI camera).
    if "camera_flip" in applied and _stream is not None:
        _stream.flip = _FLIP_STR_TO_CODE.get(applied["camera_flip"])

    await _broadcast()
    return {
        "applied": applied,
        "errors": errors,
        "restart_required": restart_required,
        "values": _monitor.settings.public_dict(),
    }


class PreviewBody(BaseModel):
    camera_index: int


@app.post("/api/preview")
async def api_preview(body: PreviewBody):
    """Switch the live-view camera without starting monitoring — lets the user
    confirm the right camera in the UI before hitting Start."""
    if _stream is None:
        return JSONResponse({"error": "stream not ready"}, 503)
    if _monitor and _monitor._running:
        return JSONResponse({"error": "stop monitoring before switching camera"}, 409)
    # Off the event loop: set_index blocks ~0.3 s while the worker restarts.
    await asyncio.get_running_loop().run_in_executor(
        None, _stream.set_index, body.camera_index)
    return {"camera_index": body.camera_index}


_SNAPSHOT_MAX_AGE = 1.0   # seconds; older idle frames trigger a fresh encode


@app.get("/snapshot")
async def snapshot():
    """On-demand JPEG snapshot (single frame from the live buffer).

    With no live viewers the buffer is only refreshed every ~2 s, so a stale
    frame triggers a wake-up and a short wait (≤1 s) for a fresh one.
    """
    data = _stream.latest_jpeg() if _stream else None   # also wakes the encoder
    if _stream is not None and (data is None or _stream.jpeg_age() > _SNAPSHOT_MAX_AGE):
        with _stream._lock:
            seq = _stream._jpeg_seq
        _, fresh = await _stream.next_jpeg(seq, _SNAPSHOT_MAX_AGE)
        data = fresh or data
    if data is None:
        return Response(status_code=204)
    return Response(data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


async def _mjpeg_generator():
    """Yield MJPEG frames from the shared live buffer as they are encoded.

    Registers as a viewer so the capture loop runs at full rate (_STREAM_FPS
    encodes) while at least one stream is open. Each frame is pushed the
    moment the encoder publishes it (next_jpeg), instead of on a fixed poll
    tick that added up to one frame interval of latency and periodically
    skipped frames when the two clocks drifted. Unchanged frames are never
    resent. The 0.5 s wait timeout bounds how long shutdown can go unnoticed.
    """
    if _stream is None:
        return
    _stream.add_viewer()
    try:
        last_seq = -1
        while not _shutting_down.is_set():
            seq, frame = await _stream.next_jpeg(last_seq, 0.5)
            if frame is None or seq == last_seq:
                continue
            last_seq = seq
            # Content-Length lets the browser decode each part as soon as its
            # last byte arrives rather than waiting for the next boundary.
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                frame +
                b"\r\n"
            )
    finally:
        if _stream:
            _stream.remove_viewer()


@app.get("/stream")
async def stream():
    """MJPEG live stream — open in any browser or <img src='/stream'>."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


# ── Home-screen install (manifest + icons) ───────────────────────────────────

@app.get("/manifest.webmanifest")
async def manifest():
    """Lets a phone add the dashboard to its home screen as a full-screen app."""
    return JSONResponse({
        "name": "Ender3Monitor",
        "short_name": "Ender3",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#070b14",
        "theme_color": "#0d1120",
        "icons": [{"src": f"/icon-{n}.png", "sizes": f"{n}x{n}", "type": "image/png"}
                  for n in _ICON_SIZES if n != 180],
    }, media_type="application/manifest+json",
       headers={"Cache-Control": "public, max-age=86400"})


_icon_cache: dict = {}


def _icon_png(size: int) -> bytes:
    """Draw the app icon (a printer gantry over a bed) once per size."""
    if size not in _icon_cache:
        n = 512
        img = np.full((n, n, 3), (32, 17, 13), np.uint8)          # --surf, BGR
        blue = (247, 142, 79)                                      # --blue, BGR
        aa = cv2.LINE_AA
        cv2.line(img, (136, 372), (136, 132), blue, 30, aa)       # left upright
        cv2.line(img, (376, 372), (376, 132), blue, 30, aa)       # right upright
        cv2.line(img, (136, 132), (376, 132), blue, 30, aa)       # top beam
        cv2.line(img, (104, 392), (408, 392), blue, 34, aa)       # bed
        cv2.rectangle(img, (220, 190), (292, 246), blue, -1, aa)  # print head
        cv2.fillConvexPoly(img, np.array([[232, 246], [280, 246], [256, 290]]), blue, aa)
        cv2.rectangle(img, (212, 330), (300, 368), (240, 228, 221), -1, aa)  # part
        if size != n:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        _icon_cache[size] = buf.tobytes() if ok else b""
    return _icon_cache[size]


@app.get("/icon-{size}.png")
async def icon(size: int):
    if size not in _ICON_SIZES:
        return Response(status_code=404)
    return Response(_icon_png(size), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


# ── HTML / CSS / JS  (single-file, no build step) ────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # The page is a ~100 KB static string: compress it once and let the browser
    # revalidate with an ETag instead of re-downloading it on every reload.
    global _html_gz, _html_etag
    if _html_gz is None:
        _html_gz = gzip.compress(_HTML.encode("utf-8"), compresslevel=6)
        _html_etag = '"%s"' % hashlib.md5(_HTML.encode("utf-8")).hexdigest()
    headers = {"ETag": _html_etag, "Cache-Control": "no-cache", "Vary": "Accept-Encoding"}
    if request.headers.get("if-none-match") == _html_etag:
        return Response(status_code=304, headers=headers)
    if "gzip" in request.headers.get("accept-encoding", ""):
        return Response(_html_gz, media_type="text/html; charset=utf-8",
                        headers={**headers, "Content-Encoding": "gzip"})
    return HTMLResponse(_HTML, headers=headers)


_html_gz: Optional[bytes] = None
_html_etag = ""


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0d1120">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Ender3">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="apple-touch-icon" href="/icon-180.png">
<link rel="icon" type="image/png" href="/icon-192.png">
<title>Ender3Monitor</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{overflow-x:hidden;max-width:100%}

:root{
  --bg:#070b14;
  --surf:#0d1120;
  --surf2:#121729;
  --surf3:#181f30;
  --border:rgba(255,255,255,0.07);
  --blue:#4f8ef7;
  --green:#22c55e;
  --red:#f87171;
  --amber:#fbbf24;
  --text:#dde4f0;
  --muted:#4a5878;
  --r:12px;
  --shadow:0 4px 24px rgba(0,0,0,.45);
}

body{
  background:var(--bg);
  color:var(--text);
  font-family:system-ui,-apple-system,'Segoe UI',sans-serif;
  font-size:14px;
  line-height:1.5;
  min-height:100vh;
  display:flex;
  flex-direction:column;
  overflow-x:hidden;            /* never scroll sideways on small screens */
}

/* ── Header ── */
header{
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:14px 28px;
  background:var(--surf);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:10;
}
.logo{display:flex;align-items:center;gap:10px;font-size:17px;font-weight:700;letter-spacing:-.3px}
.logo svg{opacity:.85}

.pill{
  display:flex;align-items:center;gap:7px;
  padding:5px 14px;border-radius:100px;
  font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  transition:all .3s;
}
.pill .dot{width:7px;height:7px;border-radius:50%;transition:background .3s}
.pill-idle  {background:rgba(74,88,120,.18);color:var(--muted)}
.pill-idle .dot{background:var(--muted)}
.pill-run   {background:rgba(79,142,247,.12);color:var(--blue)}
.pill-run .dot{background:var(--blue);animation:blink 2s ease-in-out infinite}
.pill-fail  {background:rgba(248,113,113,.12);color:var(--red)}
.pill-fail .dot{background:var(--red);animation:blink .8s ease-in-out infinite}
.pill-done  {background:rgba(34,197,94,.12);color:var(--green)}
.pill-done .dot{background:var(--green)}
.pill-pause {background:rgba(251,191,36,.12);color:var(--amber)}
.pill-pause .dot{background:var(--amber);animation:blink 1.2s ease-in-out infinite}

@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}

/* ── Layout ── */
.page{
  flex:1;
  display:grid;
  grid-template-columns:1fr 320px;
  grid-template-rows:auto 1fr;
  gap:18px;
  padding:22px 28px;
  max-width:1180px;
  width:100%;
  margin:0 auto;
  align-items:start;
}

/* ── Cards ── */
.card{
  background:var(--surf);
  border:1px solid var(--border);
  border-radius:var(--r);
  padding:22px;
  box-shadow:var(--shadow);
}
.card-label{
  font-size:10.5px;font-weight:700;
  text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin-bottom:14px;
}

/* ── Camera ── */
.cam-card{display:flex;flex-direction:column;gap:14px}
#cam{
  width:100%;aspect-ratio:16/9;
  border-radius:8px;object-fit:cover;
  background:#000;display:block;
}
.cam-meta{font-size:11.5px;color:var(--muted);display:flex;justify-content:space-between;
          flex-wrap:wrap;gap:4px 10px;min-width:0}
.cam-meta span{min-width:0;overflow-wrap:anywhere}
#stream-url{word-break:break-all}

/* ── Stats ── */
.stats-card{display:flex;flex-direction:column;gap:20px}

.conf-block{}
.conf-header{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px}
.conf-num{font-size:32px;font-weight:800;line-height:1;letter-spacing:-.5px}
.conf-track{height:5px;background:var(--surf3);border-radius:3px;overflow:hidden}
.conf-fill{height:100%;border-radius:3px;transition:width .7s cubic-bezier(.4,0,.2,1),background .4s}

.type-row{display:flex;flex-direction:column;gap:4px}
.type-val{
  font-size:15px;font-weight:600;
  padding:7px 12px;border-radius:7px;
  background:var(--surf3);
  width:fit-content;
  transition:color .3s,background .3s;
}
.type-ok  {color:var(--green);background:rgba(34,197,94,.1)}
.type-fail{color:var(--red);background:rgba(248,113,113,.1)}

.counts{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.count-box{
  background:var(--surf3);border-radius:10px;
  padding:9px 12px;
}
.count-box .lbl{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:3px}
.count-box .num{font-size:18px;font-weight:800;line-height:1.1;white-space:nowrap}
.cost-line{margin-top:10px;font-size:10px;color:var(--muted);opacity:.7}
.cost-line span{font-weight:600}

/* ── Print status pills ── */
.print-pills{display:none;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:10px}
.print-pills .count-box .num{font-size:16px}
.print-pills .pct .num{color:var(--blue)}
.stat-chip{font-size:11px;color:var(--muted);background:var(--surf3);border-radius:999px;padding:4px 10px}

/* ── Description ── */
.desc-card{}
#description{
  font-size:13.5px;line-height:1.7;color:var(--text);
  border-left:2px solid var(--blue);
  padding-left:14px;
  transition:opacity .3s;
}

/* ── Camera selector ── */
.cam-select-row{display:flex;gap:10px;align-items:center}
#cam-select{
  flex:1;
  background:var(--surf3);border:1px solid var(--border);
  color:var(--text);border-radius:8px;padding:8px 12px;
  font-size:13px;appearance:none;cursor:pointer;
}
#cam-select:focus{outline:none;border-color:var(--blue)}
#cam-select:disabled{opacity:.4;cursor:not-allowed}
.btn-scan{background:var(--surf3);border-color:var(--border);color:var(--muted);padding:8px 12px}
.btn-scan:hover{color:var(--text)}

/* ── Buttons ── */
button{
  display:flex;align-items:center;gap:7px;
  padding:9px 18px;border-radius:9px;
  border:1px solid var(--border);
  background:var(--surf);color:var(--text);
  font-size:13px;font-weight:500;
  cursor:pointer;transition:all .15s;
  white-space:nowrap;
}
button:hover{background:var(--surf3);border-color:rgba(255,255,255,.13)}
button:active{transform:scale(.97)}
.btn-start{background:rgba(79,142,247,.12);border-color:rgba(79,142,247,.3);color:var(--blue)}
.btn-start:hover{background:rgba(79,142,247,.2)}
.btn-stop{background:rgba(248,113,113,.08);border-color:rgba(248,113,113,.25);color:var(--red)}
.btn-stop:hover{background:rgba(248,113,113,.16)}
button:disabled{opacity:.35;cursor:not-allowed;transform:none}

/* ── Event log ── */
.log-section{
  padding:0 28px 28px;
  max-width:1180px;width:100%;margin:0 auto;
}
.log-header{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:10px;
}
.log-header h2{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.log-clear{font-size:11px;color:var(--muted);background:none;border:none;cursor:pointer;padding:2px 6px}
.log-clear:hover{color:var(--text)}
#log{
  background:var(--surf);border:1px solid var(--border);
  border-radius:var(--r);max-height:210px;overflow-y:auto;
}
#log::-webkit-scrollbar{width:4px}
#log::-webkit-scrollbar-track{background:transparent}
#log::-webkit-scrollbar-thumb{background:var(--surf3);border-radius:2px}

.log-empty{
  padding:20px;text-align:center;
  color:var(--muted);font-size:12px;
}
.ev{
  display:grid;
  grid-template-columns:64px 116px 1fr 44px;
  gap:12px;align-items:start;       /* top-align so multi-line rows never overlap */
  padding:10px 18px;
  border-bottom:1px solid var(--border);
  font-size:12px;
}
.ev:last-child{border-bottom:none}
.ev-time{color:var(--muted);font-family:monospace;font-size:11.5px;padding-top:2px}
.badge{
  padding:3px 9px;border-radius:5px;
  font-size:10.5px;font-weight:700;
  text-transform:uppercase;letter-spacing:.05em;
  width:fit-content;max-width:100%;
  min-width:0;                      /* allow shrink inside the grid track */
  overflow-wrap:anywhere;           /* wrap long/compound failure types */
  line-height:1.35;
}
.badge-ok  {background:rgba(34,197,94,.12);color:var(--green)}
.badge-fail{background:rgba(248,113,113,.12);color:var(--red)}
.badge-done{background:rgba(79,142,247,.12);color:var(--blue)}
.ev-desc{
  color:var(--text);
  min-width:0;                      /* let it wrap instead of overflowing */
  overflow-wrap:anywhere;
  line-height:1.5;
  display:-webkit-box;
  -webkit-line-clamp:3;             /* wrap up to 3 lines, then ellipsis */
  -webkit-box-orient:vertical;
  overflow:hidden;
}
.ev-conf{color:var(--muted);font-family:monospace;text-align:right;padding-top:2px}

/* ── Charts ── */
.charts-card{grid-column:1/-1;min-width:0}
.ch-legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin-bottom:10px;font-size:12px;color:var(--muted)}
.ch-key{display:inline-flex;align-items:center;gap:7px;white-space:nowrap}
.ch-key b{color:var(--text);font-weight:700;font-variant-numeric:tabular-nums}
.ch-key b span{color:var(--muted);font-weight:500}
.ch-swatch{width:14px;height:0;border-top:2px solid var(--muted);border-radius:2px;flex:none}
.ch-swatch.dash{border-top-style:dashed}
.ch-svg{display:block;width:100%;height:170px;overflow:visible}
.ch-grid{stroke:var(--border);stroke-width:1;shape-rendering:crispEdges}
.ch-axis{fill:var(--muted);font-size:10.5px;font-family:inherit;font-variant-numeric:tabular-nums}
.ch-empty{fill:var(--muted);font-size:12px;font-style:italic}
.ch-line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ch-target{fill:none;stroke-width:1.25;stroke-dasharray:4 4;opacity:.7}
.ch-nozzle{stroke:var(--amber)}
.ch-bed{stroke:var(--blue)}
.ch-dot{stroke:var(--surf);stroke-width:2}
.ch-dot.ch-nozzle{fill:var(--amber)}
.ch-dot.ch-bed{fill:var(--blue)}
.ch-layer{margin-top:16px;padding-top:14px;border-top:1px solid var(--border)}
.ch-layer-head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;
               font-size:12px;color:var(--muted);margin-bottom:7px}
.ch-layer-head b{color:var(--text);font-weight:700;font-variant-numeric:tabular-nums}
.ch-bar{position:relative;height:8px;background:var(--surf3);border-radius:4px}
.ch-bar-fill{height:100%;width:0;background:var(--blue);border-radius:4px;transition:width .7s cubic-bezier(.4,0,.2,1)}
.ch-tick{position:absolute;top:-3px;bottom:-3px;width:2px;margin-left:-1px;border-radius:1px;
         background:var(--amber);box-shadow:0 0 0 1px var(--surf)}
.ch-tick.done{background:var(--muted)}
.ch-layer-note{font-size:10.5px;color:var(--muted);margin-top:7px}

/* ── Failure review ── */
.fr-card{display:none;grid-column:1/-1;border-color:rgba(248,113,113,.35)}
.fr-card.show{display:block}
.fr-card .card-label{color:var(--red)}
.fr-body{display:flex;gap:18px;align-items:flex-start}
.fr-img{
  flex:0 0 300px;width:300px;max-width:100%;aspect-ratio:16/9;
  object-fit:cover;border-radius:8px;background:var(--surf3);display:block;
}
.fr-info{flex:1;min-width:0;display:flex;flex-direction:column;gap:10px}
.fr-head{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
.fr-type{font-size:20px;font-weight:700;color:var(--red);overflow-wrap:anywhere}
.fr-conf{font-size:13px;font-weight:600;color:var(--amber)}
.fr-time{font-size:11.5px;color:var(--muted)}
.fr-desc{font-size:13px;line-height:1.55;overflow-wrap:anywhere}
.fr-actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:4px}
.fr-actions button{justify-content:center;padding:11px 20px;font-size:14px;font-weight:600}
.fr-resume{background:rgba(34,197,94,.12);border-color:rgba(34,197,94,.3);color:var(--green)}
.fr-resume:hover{background:rgba(34,197,94,.2)}
.fr-keep{background:rgba(251,191,36,.1);border-color:rgba(251,191,36,.3);color:var(--amber)}
.fr-keep:hover{background:rgba(251,191,36,.18)}
.fr-false{background:var(--surf3);color:var(--blue)}
.fr-actions button:disabled{opacity:.5;cursor:default}
.fr-note{font-size:11.5px;color:var(--muted)}
.conf-spark{margin-top:10px}
.conf-spark svg{display:block;width:100%;height:36px;background:var(--surf3);border-radius:6px}
.conf-spark-meta{display:flex;justify-content:space-between;gap:8px;margin-top:5px;
                 font-size:10.5px;color:var(--muted)}
@media (max-width:760px){
  .fr-body{flex-direction:column;gap:12px}
  .fr-img{flex:none;width:100%}
  .fr-actions{flex-wrap:nowrap;gap:8px}
  .fr-actions button{flex:1 1 0;min-width:0;padding:15px 4px;font-size:13.5px}
}
/* ── /Failure review ── */

/* ── Toast ── */
#toast{
  position:fixed;bottom:28px;left:50%;transform:translateX(-50%) translateY(80px);
  background:var(--surf2);border:1px solid var(--border);
  padding:10px 20px;border-radius:10px;font-size:13px;
  transition:transform .25s;pointer-events:none;z-index:100;
  box-shadow:var(--shadow);
}
#toast.show{transform:translateX(-50%) translateY(0)}

/* ── Header right cluster + icon button ── */
.header-right{display:flex;align-items:center;gap:12px}
.icon-btn{
  display:flex;align-items:center;justify-content:center;
  width:34px;height:34px;padding:0;border-radius:9px;
  background:var(--surf3);border:1px solid var(--border);color:var(--muted);
  cursor:pointer;transition:all .15s;
}
.icon-btn:hover{color:var(--text);border-color:rgba(255,255,255,.14)}

/* ── Modal ── */
.modal-overlay{
  position:fixed;inset:0;z-index:200;
  background:rgba(0,0,0,.55);backdrop-filter:blur(2px);
  display:none;align-items:flex-start;justify-content:center;
  padding:40px 16px;overflow-y:auto;
}
.modal-overlay.show{display:flex}
.modal{
  background:var(--surf);border:1px solid var(--border);border-radius:14px;
  width:100%;max-width:560px;box-shadow:var(--shadow);
  display:flex;flex-direction:column;max-height:calc(100vh - 80px);
}
.modal-head{
  display:flex;align-items:center;justify-content:space-between;
  padding:18px 22px;border-bottom:1px solid var(--border);
}
.modal-head h2{font-size:16px;font-weight:700}
.modal-body{padding:8px 22px;overflow-y:auto}
.modal-foot{
  display:flex;align-items:center;gap:10px;
  padding:14px 22px;border-top:1px solid var(--border);
}
#settings-msg{font-size:12px}
#settings-msg.err{color:var(--red)}
#settings-msg.ok{color:var(--green)}
.sec-warn{
  margin:14px 22px 0;padding:10px 12px;border-radius:9px;font-size:12px;line-height:1.5;
  background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.3);color:var(--amber);
}
.sec-warn code{background:rgba(0,0,0,.25);padding:1px 5px;border-radius:4px;font-size:11px}
.set-group-label{
  font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin:18px 0 8px;
}
.set-row{
  display:flex;align-items:center;gap:12px;
  padding:9px 0;border-bottom:1px solid var(--border);
}
.set-row:last-child{border-bottom:none}
.set-info{flex:1;min-width:0}
.set-name{font-size:13.5px;display:flex;align-items:center;gap:7px}
.set-help{font-size:11px;color:var(--muted);margin-top:2px}
.set-restart{
  font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
  color:var(--amber);background:rgba(251,191,36,.12);padding:1px 6px;border-radius:4px;
}
.set-control{flex:0 0 auto}
.set-control input[type=number],.set-control select{
  background:var(--surf3);border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:7px 10px;font-size:13px;width:130px;
}
.set-control input:focus,.set-control select:focus{outline:none;border-color:var(--blue)}
/* toggle */
.tgl{display:inline-block;position:relative;width:42px;height:24px;cursor:pointer;flex:0 0 auto}
.tgl input{opacity:0;width:0;height:0;position:absolute}
.tgl .track{position:absolute;inset:0;background:var(--surf3);border:1px solid var(--border);
  border-radius:100px;transition:.2s}
.tgl .knob{position:absolute;top:3px;left:3px;width:16px;height:16px;border-radius:50%;
  background:var(--muted);transition:.2s}
.tgl input:checked + .track{background:rgba(79,142,247,.25);border-color:var(--blue)}
.tgl input:checked + .track .knob{transform:translateX(18px);background:var(--blue)}
.set-info-row{font-size:11.5px;color:var(--muted);padding:6px 0;display:flex;justify-content:space-between}

/* ── Status headline ── */
.headline{max-width:1180px;width:100%;margin:0 auto;padding:20px 28px 0}
.hl-main{font-size:26px;font-weight:800;letter-spacing:-.5px;line-height:1.2;overflow-wrap:anywhere}
.hl-sub{font-size:13.5px;color:#8291b3;margin-top:5px;display:flex;flex-wrap:wrap;gap:2px 0}
.hl-sub span:not(:last-child)::after{content:'·';margin:0 7px;color:var(--muted)}
.hl-sub .warn{color:var(--amber)}
.hl-bar{height:4px;background:var(--surf3);border-radius:2px;overflow:hidden;margin-top:12px;display:none}
.hl-bar div{height:100%;width:0;background:var(--blue);border-radius:2px;transition:width .7s}
.headline.t-idle .hl-main{color:#8291b3}
.headline.t-warn .hl-main{color:var(--amber)}
.headline.t-warn .hl-bar div{background:var(--amber)}
.headline.t-fail .hl-main{color:var(--red)}
.headline.t-fail .hl-bar div{background:var(--red)}
.headline.t-done .hl-main{color:var(--green)}
.headline.t-done .hl-bar div{background:var(--green)}

/* ── Details (collapsed secondary stats) ── */
.details{border-top:1px solid var(--border);padding-top:14px}
.details summary{
  font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);cursor:pointer;list-style:none;
  display:flex;justify-content:space-between;align-items:center;
}
.details summary::-webkit-details-marker{display:none}
.details summary::after{content:'+';font-size:15px;line-height:1;font-weight:500}
.details[open] summary::after{content:'–'}
.details[open] summary{margin-bottom:12px}
.details-meta{font-size:11px;color:var(--muted);margin-top:10px;display:flex;flex-direction:column;gap:4px}

/* ── Sticky action bar ── */
body{padding-bottom:calc(84px + env(safe-area-inset-bottom))}
#toast{bottom:calc(96px + env(safe-area-inset-bottom));opacity:0;transition:transform .25s,opacity .25s}
#toast.show{opacity:1}
.actionbar{
  position:fixed;left:0;right:0;bottom:0;z-index:50;
  background:rgba(13,17,32,.94);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  border-top:1px solid var(--border);
  padding:8px max(12px,env(safe-area-inset-right)) calc(8px + env(safe-area-inset-bottom)) max(12px,env(safe-area-inset-left));
}
.ab-inner{display:flex;gap:6px;max-width:560px;margin:0 auto}
.ab-btn{
  flex:1 1 0;min-width:0;min-height:54px;
  flex-direction:column;justify-content:center;gap:3px;
  padding:7px 4px;border-radius:11px;
  background:var(--surf3);font-size:10.5px;font-weight:600;
  position:relative;overflow:hidden;
  touch-action:manipulation;user-select:none;-webkit-user-select:none;-webkit-touch-callout:none;
}
.ab-btn svg{width:18px;height:18px;flex:0 0 auto}
.ab-btn:hover{background:#1d2336}
.ab-monitor{flex-grow:1.25;color:var(--blue);background:rgba(79,142,247,.12);border-color:rgba(79,142,247,.3)}
.ab-monitor:hover{background:rgba(79,142,247,.2)}
.ab-monitor.on{color:var(--text);background:var(--surf3);border-color:var(--border)}
.ab-monitor.on:hover{background:#1d2336}
.ab-pause.resume{color:var(--green);background:rgba(34,197,94,.1);border-color:rgba(34,197,94,.3)}
.ab-estop{color:var(--red);background:rgba(248,113,113,.1);border-color:rgba(248,113,113,.3)}
.ab-estop:hover{background:rgba(248,113,113,.16)}
.ab-estop:active{transform:none}
.ab-estop > *{position:relative}
.ab-estop .hold-fill{
  position:absolute;inset:0;background:rgba(248,113,113,.5);
  transform:scaleX(0);transform-origin:left;pointer-events:none;
}
.ab-estop.holding .hold-fill{transform:scaleX(1);transition:transform 1.5s linear}
.ab-printer{display:none}
.actionbar.has-printer .ab-printer{display:flex}
.ab-more{flex:0 0 50px}

/* ── More sheet ── */
.sheet-overlay{
  position:fixed;inset:0;z-index:150;background:rgba(0,0,0,.55);
  display:none;align-items:flex-end;justify-content:center;
}
.sheet-overlay.show{display:flex}
.sheet{
  background:var(--surf);border:1px solid var(--border);border-bottom:none;
  border-radius:16px 16px 0 0;width:100%;max-width:560px;
  padding:10px 18px calc(20px + env(safe-area-inset-bottom));
  display:flex;flex-direction:column;gap:16px;max-height:85vh;overflow-y:auto;
  box-shadow:var(--shadow);
}
.sheet-grip{width:36px;height:4px;border-radius:2px;background:var(--surf3);margin:0 auto}
.sheet-label{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin-bottom:8px}
.sheet-row{display:flex;align-items:center;justify-content:space-between;gap:12px}
.sheet-row .set-help{margin-top:2px}
.sheet-wide{width:100%;justify-content:center;padding:13px 18px;font-size:14px}
.sheet a{color:var(--blue);text-decoration:none;overflow-wrap:anywhere}

/* ── Mobile / narrow screens ── */
@media (max-width:760px){
  .headline{padding:14px 16px 0}
  .hl-main{font-size:22px}
  /* Camera runs edge to edge: it is the thing you came to look at */
  .card.cam-card{padding:0;overflow:hidden;gap:0}
  .cam-card #cam{border-radius:0}
  .cam-card #description{margin:14px 16px 16px}
  .modal-overlay{padding:0}
  .modal{max-width:none;border-radius:0;min-height:100vh;max-height:100vh}
  .set-control input[type=number],.set-control select{width:108px}
  header{padding:12px 16px}
  .logo{font-size:15px}
  /* The headline below already says the status in words; keep just the dot */
  .pill{padding:7px}
  #pill-text{display:none}

  /* Stack the two-column layout into one; camera no longer spans rows */
  .page{
    grid-template-columns:1fr;
    grid-template-rows:none;
    gap:14px;
    padding:14px 14px;
  }
  .card{padding:16px}

  .log-section{padding:0 14px 22px}

  /* Event log: time + badge + confidence on top row, description below */
  .ev{
    /* minmax(0,1fr) — not plain 1fr — lets the middle track shrink below its
       content's min-width so a long compound badge wraps instead of blowing
       out the row width (the classic CSS grid overflow trap). */
    grid-template-columns:auto minmax(0,1fr) auto;
    grid-template-areas:
      "time badge conf"
      "desc desc desc";
    row-gap:6px;column-gap:10px;
    align-items:center;
    padding:12px 14px;
  }
  .ev-time{grid-area:time}
  /* base .badge already has width:fit-content + max-width:100%; combined with
     the minmax(0,1fr) track that clamps & wraps long compound types while
     keeping short badges compact */
  .badge{grid-area:badge;justify-self:start}
  .ev-conf{grid-area:conf;padding-top:0}
  .ev-desc{grid-area:desc;-webkit-line-clamp:4}
}
button.busy { cursor: progress; opacity: .6; }
</style>
</head>
<body>

<header>
  <div class="logo">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.75" stroke-linecap="round">
      <path d="M3 21h18M5 21V9l7-6 7 6v12"/>
      <rect x="9" y="13" width="6" height="8" rx="1"/>
      <path d="M9 13h6"/>
    </svg>
    Ender3Monitor
  </div>
  <div class="header-right">
    <div class="pill pill-idle" id="pill">
      <div class="dot"></div>
      <span id="pill-text">Idle</span>
    </div>
    <a class="icon-btn" id="history-btn" href="/history" title="Print history" style="text-decoration:none">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 12a9 9 0 1 0 3-6.7L3 8"/>
        <path d="M3 3v5h5"/>
        <path d="M12 7v5l3 2"/>
      </svg>
    </a>
    <button class="icon-btn" id="settings-btn" title="Settings" onclick="openSettings()">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"/>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
      </svg>
    </button>
  </div>
</header>

<!-- Settings modal -->
<div id="settings-overlay" class="modal-overlay" onclick="if(event.target===this)closeSettings()">
  <div class="modal" role="dialog" aria-label="Settings">
    <div class="modal-head">
      <h2>Settings</h2>
      <button class="icon-btn" onclick="closeSettings()" title="Close">✕</button>
    </div>
    <div id="settings-secwarn" class="sec-warn" style="display:none">
      ⚠ No dashboard login set — anyone on your network can change these and
      control the printer. Set <code>WEB_USERNAME</code>/<code>WEB_PASSWORD</code> in
      <code>.env</code>.
    </div>
    <div id="settings-body" class="modal-body"></div>
    <div class="modal-foot">
      <span id="settings-msg"></span>
      <div style="flex:1"></div>
      <button onclick="closeSettings()">Cancel</button>
      <button class="btn-start" id="settings-save" onclick="saveSettings()">Save</button>
    </div>
  </div>
</div>

<!-- Status headline: is it OK, and when is it done -->
<div class="headline t-idle" id="headline" aria-live="polite">
  <div class="hl-main" id="hl-main">Connecting…</div>
  <div class="hl-sub" id="hl-sub"></div>
  <div class="hl-bar" id="hl-bar"><div id="hl-bar-fill"></div></div>
</div>

<div class="page">

  <!-- Failure review -->
  <div class="card fr-card" id="fr-card">
    <div class="card-label">Failure detected — review</div>
    <div class="fr-body">
      <img class="fr-img" id="fr-img" alt="Flagged frame">
      <div class="fr-info">
        <div class="fr-head">
          <span class="fr-type" id="fr-type">—</span>
          <span class="fr-conf" id="fr-conf"></span>
          <span class="fr-time" id="fr-time"></span>
        </div>
        <div class="fr-desc" id="fr-desc"></div>
        <div class="fr-actions">
          <button class="fr-resume" onclick="resolveFailure('resume')">Resume</button>
          <button class="fr-keep" onclick="resolveFailure('keep_paused')">Keep paused</button>
          <button class="fr-false" onclick="resolveFailure('false_alarm')">False alarm</button>
        </div>
        <div class="fr-note">False alarm only dismisses this alert — it does not resume the print.</div>
      </div>
    </div>
  </div>
  <!-- /Failure review -->

  <!-- Camera, with the AI's latest observation as its caption -->
  <div class="card cam-card">
    <img id="cam" src="/stream" alt="Live camera view of the printer">
    <p id="description" style="color:var(--muted);font-style:italic">
      Waiting for first analysis…
    </p>
  </div>

  <!-- Stats -->
  <div class="card stats-card">
    <div class="conf-block">
      <div class="card-label">Confidence</div>
      <div class="conf-header">
        <div class="conf-num" id="conf-num">—</div>
        <div style="font-size:11px;color:var(--muted)" id="conf-thresh">threshold —</div>
      </div>
      <div class="conf-track">
        <div class="conf-fill" id="conf-fill" style="width:0;background:var(--green)"></div>
      </div>
      <!-- Confidence sparkline -->
      <div class="conf-spark" id="conf-spark">
        <svg viewBox="0 0 100 36" preserveAspectRatio="none" role="img"
             aria-label="Confidence over this print, with the alert threshold">
          <line id="spark-thresh" x1="0" x2="100" y1="18" y2="18" stroke="var(--red)"
                stroke-width="1" stroke-dasharray="4 3" opacity=".7"
                vector-effect="non-scaling-stroke"/>
          <polyline id="spark-line" fill="none" stroke="var(--blue)" stroke-width="1.5"
                    stroke-linejoin="round" stroke-linecap="round"
                    vector-effect="non-scaling-stroke" points=""/>
        </svg>
        <div class="conf-spark-meta">
          <span id="spark-span">no analyses yet</span>
          <span id="spark-false"></span>
        </div>
      </div>
      <!-- /Confidence sparkline -->
    </div>

    <div class="type-row">
      <div class="card-label">Failure Type</div>
      <div class="type-val type-ok" id="type-val">none</div>
    </div>

    <!-- Printer (USB) — hidden until a printer is connected -->
    <div id="printer-block" style="display:none">
      <div class="card-label">Printer</div>
      <div class="counts">
        <div class="count-box">
          <div class="lbl">Nozzle</div>
          <div class="num" id="temp-nozzle" style="font-size:16px">—</div>
        </div>
        <div class="count-box">
          <div class="lbl">Bed</div>
          <div class="num" id="temp-bed" style="font-size:16px">—</div>
        </div>
      </div>
      <div class="print-pills" id="print-pills">
        <div class="count-box pct"><div class="lbl">Progress</div><div class="num" id="pp-pct">—</div></div>
        <div class="count-box"><div class="lbl">Elapsed</div><div class="num" id="pp-elapsed">—</div></div>
        <div class="count-box"><div class="lbl">Left</div><div class="num" id="pp-left">—</div></div>
        <div id="pp-chips" style="grid-column:1/-1;display:flex;flex-wrap:wrap;gap:6px"></div>
      </div>
      <div style="font-size:11px;color:var(--muted);margin-top:8px" id="printer-progress"></div>
    </div>

    <details class="details" id="details">
      <summary>Details</summary>
      <div class="counts">
        <div class="count-box">
          <div class="lbl">Frames</div>
          <div class="num" id="frames">0</div>
        </div>
        <div class="count-box">
          <div class="lbl">Failures</div>
          <div class="num" id="failures" style="color:var(--text)">0</div>
        </div>
      </div>
      <div class="cost-line" id="cost-box" title="Estimated AI API cost for this print">
        Est. AI cost · this print <span id="cost">$0.00</span>
      </div>
      <div class="cost-line" id="tl-line"></div>
      <div class="details-meta">
        <span id="backend-row"></span>
        <span id="cam-label"></span>
      </div>
    </details>
  </div>

  <!-- Charts -->
  <div class="card charts-card" id="charts-card" style="display:none">
    <div class="card-label">Temperatures</div>
    <div class="ch-legend" id="ch-legend"></div>
    <svg class="ch-svg" id="ch-temps" viewBox="0 0 600 170" role="img"
         aria-label="Nozzle and bed temperature over the last 30 minutes"></svg>
    <div class="ch-layer" id="ch-layer" style="display:none">
      <div class="ch-layer-head"><span id="ch-layer-label">Print progress</span><b id="ch-layer-pct"></b></div>
      <div class="ch-bar" id="ch-bar"><div class="ch-bar-fill" id="ch-bar-fill"></div></div>
      <div class="ch-layer-note" id="ch-layer-note"></div>
    </div>
  </div>
  <!-- /Charts -->

</div>

<!-- Sticky action bar: the controls you reach for mid-print -->
<nav class="actionbar" id="actionbar" aria-label="Print controls">
  <div class="ab-inner">
    <button class="ab-btn ab-monitor" id="btn-monitor" onclick="toggleMonitor()">
      <svg id="mon-icon" viewBox="0 0 24 24" fill="currentColor"><polygon points="6 4 20 12 6 20"/></svg>
      <span id="mon-label">Start AI</span>
    </button>
    <button class="ab-btn ab-printer ab-pause" id="btn-pause" onclick="togglePause()">
      <svg id="pause-icon" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>
      <span id="pause-label">Pause</span>
    </button>
    <button class="ab-btn ab-printer" id="btn-cooldown" onclick="doCooldown()"
            title="Pause the print and turn the heaters off">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 2v20M4.9 6l14.2 12M19.1 6L4.9 18"/></svg>
      <span>Cooldown</span>
    </button>
    <button class="ab-btn ab-printer ab-estop" id="btn-estop"
            title="Hold to emergency stop" aria-label="Emergency stop — press and hold">
      <span class="hold-fill"></span>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="8 2 16 2 22 8 22 16 16 22 8 22 2 16 2 8"/><path d="M12 8v5M12 16.5v.01"/></svg>
      <span>E‑Stop</span>
    </button>
    <button class="ab-btn ab-more" id="btn-more" onclick="openMore()" aria-label="More controls">
      <svg viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/></svg>
      <span>More</span>
    </button>
  </div>
</nav>

<!-- More sheet: the rarely-needed controls -->
<div id="more-overlay" class="sheet-overlay" onclick="if(event.target===this)closeMore()">
  <div class="sheet" role="dialog" aria-label="More controls">
    <div class="sheet-grip"></div>
    <div>
      <div class="sheet-label">Camera</div>
      <div class="cam-select-row">
        <select id="cam-select" title="Select camera" onchange="previewCamera()">
          <option value="">Scanning cameras…</option>
        </select>
        <button class="btn-scan" onclick="scanCameras(true)" title="Rescan all cameras">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="23 4 23 10 17 10"/>
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
          </svg>
        </button>
      </div>
    </div>
    <div class="sheet-row">
      <div>
        <div>Auto-start</div>
        <div class="set-help">Start AI monitoring when the printer begins a print</div>
      </div>
      <label class="tgl">
        <input type="checkbox" id="autostart-cb" onchange="toggleAutoStart()">
        <span class="track"><span class="knob"></span></span>
      </label>
    </div>
    <button class="sheet-wide" onclick="doTimelapse()" id="btn-timelapse">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 3v12m0 0l-4-4m4 4l4-4"/><path d="M5 21h14"/>
      </svg>
      Download Timelapse
    </button>
    <div>
      <div class="sheet-label">Live stream link</div>
      <a id="stream-url" href="/stream" target="_blank">/stream</a>
    </div>
    <button class="sheet-wide" onclick="closeMore()">Close</button>
  </div>
</div>

<!-- Event log -->
<div class="log-section">
  <div class="log-header">
    <h2>Event Log</h2>
    <button class="log-clear" onclick="clearLog()">Clear</button>
  </div>
  <div id="log">
    <div class="log-empty">No events yet — start monitoring to begin.</div>
  </div>
</div>

<div id="toast"></div>

<script>
let ws, lastFrames = -1, camList = [], lastState = {}, lastEventsKey = null, wsWasDown = false;

// ── WebSocket ──────────────────────────────────────────────────────────────
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen    = () => {
    // Back after a drop (e.g. server restart): the old MJPEG socket died with
    // it and the <img> stays frozen on its last frame unless reopened.
    if (wsWasDown) { wsWasDown = false; restartStream(); }
  };
  ws.onmessage = e => render(JSON.parse(e.data));
  ws.onclose   = () => {
    // Make stale data obvious instead of silently showing the last state.
    wsWasDown = true;
    document.getElementById('pill').className = 'pill pill-fail';
    document.getElementById('pill-text').textContent = 'Disconnected — reconnecting…';
    setTimeout(connect, 2500);
  };
}

// ── Live stream: recover from drops, pause in background tabs ──────────────
const BLANK_IMG = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==';
let streamRetry = null;
function restartStream() {
  const img = document.getElementById('cam');
  clearTimeout(streamRetry);
  if (document.hidden) return;              // resumed by visibilitychange
  img.src = '/stream?t=' + Date.now();      // cache-buster forces a new socket
}
(function() {
  const img = document.getElementById('cam');
  img.addEventListener('error', () => {
    if (img.src === BLANK_IMG || document.hidden) return;
    clearTimeout(streamRetry);
    streamRetry = setTimeout(restartStream, 2000);
  });
  // A hidden tab still pulls ~12 fps and keeps the encoder at full rate for
  // nobody; close the stream while hidden and reopen it when visible.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) { clearTimeout(streamRetry); img.src = BLANK_IMG; }
    else restartStream();
  });
})();

// Disable a button while its request is in flight (no double-submits, and
// the user sees the click registered even if the server takes a moment).
async function busy(id, fn) {
  const btn = typeof id === 'string' ? document.getElementById(id) : id;
  if (btn) { if (btn.disabled) return; btn.disabled = true; btn.classList.add('busy'); }
  try { return await fn(); }
  finally { if (btn) { btn.disabled = false; btn.classList.remove('busy'); } }
}

// ── Render state ───────────────────────────────────────────────────────────
function render(d) {
  lastState = d;
  renderHeadline(d);

  // Pill
  const pill = document.getElementById('pill');
  const s = d.status || '';
  pill.className = 'pill ' + pillClass(s);
  document.getElementById('pill-text').textContent = s;

  // Confidence — colors and label track the real configured threshold
  const pct = (d.confidence || 0) * 100;
  const thr = (d.threshold ?? 0.7) * 100;
  document.getElementById('conf-num').textContent =
    d.frame_count > 0 ? pct.toFixed(1) + '%' : '—';
  document.getElementById('conf-thresh').textContent = `threshold ${Math.round(thr)}%`;
  const fill = document.getElementById('conf-fill');
  fill.style.width = pct + '%';
  fill.style.background = pct >= thr ? 'var(--red)' : pct >= thr * 0.55 ? 'var(--amber)' : 'var(--green)';

  // Type
  const tv = document.getElementById('type-val');
  const ft = d.failure_type || 'none';
  tv.textContent = ft;
  tv.className = 'type-val ' + (d.failure_detected ? 'type-fail' : 'type-ok');

  // Counts
  document.getElementById('frames').textContent   = d.frame_count   ?? 0;
  const fel = document.getElementById('failures');
  fel.textContent   = d.failure_count  ?? 0;
  fel.style.color   = (d.failure_count ?? 0) > 0 ? 'var(--red)' : 'var(--text)';

  // Live API cost (this print)
  if (typeof d.cost_usd === 'number') {
    document.getElementById('cost').textContent =
      '≈$' + d.cost_usd.toFixed(d.cost_usd < 1 ? 3 : 2);
  }

  // Timelapse capture mode (Details)
  const tl = d.timelapse;
  const tlLine = document.getElementById('tl-line');
  if (tl && d.is_running) {
    const how = {parked: 'head-parked', layer: 'per layer (Z)', time: 'timed'}[tl.mode] || tl.mode;
    tlLine.innerHTML = 'Timelapse · ' + how + ' <span>' + tl.frames +
                       (tl.frames === 1 ? ' frame' : ' frames') + '</span>';
    tlLine.title = tl.mode === 'parked'
      ? 'The sliced file is parking the head for every frame.'
      : 'No snapshot marker seen from the sliced file in this print.';
  } else {
    tlLine.textContent = '';
  }

  // Description
  if (d.description) {
    const desc = document.getElementById('description');
    desc.textContent   = d.description;
    desc.style.color   = 'var(--text)';
    desc.style.fontStyle = 'normal';
  }

  // Backend
  if (d.backend) {
    document.getElementById('backend-row').textContent = 'Backend: ' + d.backend;
  }

  // Auto-start toggle reflects server state
  if (typeof d.auto_start === 'boolean') {
    const cb = document.getElementById('autostart-cb');
    if (cb && document.activeElement !== cb) cb.checked = d.auto_start;
  }

  // Printer (USB) temps + progress
  const p = d.printer || {};
  const pb = document.getElementById('printer-block');
  if (p.connected) {
    pb.style.display = 'block';
    const fmt = (t, tgt) => (t == null ? '—'
      : Math.round(t) + '°' + (tgt ? ' / ' + Math.round(tgt) + '°' : ''));
    document.getElementById('temp-nozzle').textContent = fmt(p.nozzle_temp, p.nozzle_target);
    document.getElementById('temp-bed').textContent    = fmt(p.bed_temp, p.bed_target);
    const prog = document.getElementById('printer-progress');
    let showPills = false;
    if (p.filament_runout) {
      // Shown ahead of the M600 line: a runout also parks the printer waiting
      // on the user, and calling that a color change would send you to the
      // printer expecting the wrong job.
      prog.textContent = '🧵 Paused — filament runout';
      prog.style.color = 'var(--red)';
    } else if (p.filament_change_pause) {
      prog.textContent = '🎨 Paused — change filament (M600)';
      prog.style.color = 'var(--amber)';
    } else if (p.printing && p.progress != null) {
      prog.textContent = '';
      const src = {slicer: ' (slicer)', model: ' (modelled)'}[p.remaining_source] || '';
      document.getElementById('pp-pct').textContent = (p.progress * 100).toFixed(1) + '%';
      document.getElementById('pp-elapsed').textContent = p.elapsed_str || '—';
      document.getElementById('pp-left').textContent =
        p.remaining_str ? '~' + p.remaining_str : '—';
      document.getElementById('pp-left').title = src ? 'Estimate source:' + src : '';
      const chips = [];
      if (p.layer_str) chips.push('Layer ' + p.layer_str);
      if (p.feature) chips.push(p.feature);
      if (p.next_color_change_str) chips.push('🎨 color change in ~' + p.next_color_change_str);
      const tlc = d.timelapse;
      if (d.is_running && tlc && tlc.mode === 'parked') {
        chips.push('🎞 Head-parked timelapse · ' + tlc.frames +
                   (tlc.frames === 1 ? ' frame' : ' frames'));
      }
      const box = document.getElementById('pp-chips');
      box.innerHTML = '';
      chips.forEach(t => {
        const c = document.createElement('span');
        c.className = 'stat-chip'; c.textContent = t; box.appendChild(c);
      });
      showPills = true;
    } else if (p.printing) {
      prog.textContent = 'Printing…';
      prog.style.color = 'var(--muted)';
    } else {
      prog.textContent = p.port ? 'Idle · ' + p.port : 'Idle';
      prog.style.color = 'var(--muted)';
    }
    document.getElementById('print-pills').style.display = showPills ? 'grid' : 'none';
  } else {
    pb.style.display = 'none';
  }

  // Camera label
  if (d.camera_index != null) {
    document.getElementById('cam-label').textContent = `Camera ${d.camera_index}`;
  }

  // Action bar
  renderActionBar(d);

  // Event log — rendered from the server-side history so it survives reloads.
  if (Array.isArray(d.events)) renderEvents(d.events);
  renderCharts(d);
  renderFailureReview(d); renderSparkline(d);
}

// ── Status headline ────────────────────────────────────────────────────────
// One line that answers "is it OK, and when is it done" — worst news first.
function renderHeadline(d) {
  const p = d.printer || {};
  const s = d.status || '';
  const pctStr = p.progress != null ? Math.round(p.progress * 100) + '%' : '';
  let tone = 'idle', main, sub = [], bar = false;

  if (p.connected && p.filament_runout) {
    tone = 'fail'; main = 'Paused — filament runout';
  } else if (p.connected && p.filament_change_pause) {
    tone = 'warn'; main = 'Paused — change filament';
  } else if (d.is_running && d.failure_detected) {
    tone = 'fail';
    main = 'Possible failure: ' + String(d.failure_type || 'unknown').replace(/_/g, ' ');
  } else if (/error/i.test(s)) {
    tone = 'warn'; main = s;
  } else if (p.connected && p.printing && p.paused) {
    tone = 'warn'; main = 'Paused' + (pctStr ? ' · ' + pctStr : '');
  } else if (p.connected && p.printing) {
    tone = 'run'; main = 'Printing' + (pctStr ? ' · ' + pctStr : '');
  } else if (/complete/i.test(s)) {
    tone = 'done'; main = 'Print complete';
  } else if (d.is_running) {
    tone = 'run'; main = s || 'Monitoring…';
  } else {
    main = s === 'starting' ? 'Starting…' : p.connected ? 'Printer idle' : 'Idle';
  }

  if (p.connected && p.printing) {
    bar = p.progress != null;
    if (p.remaining_str) sub.push(['~' + p.remaining_str + ' left']);
    if (p.layer_str) sub.push(['layer ' + p.layer_str]);
    if (p.feature && tone === 'run') sub.push([p.feature]);
    if (p.next_color_change_str) sub.push(['color change in ~' + p.next_color_change_str]);
    // A print running with nobody watching is the thing worth shouting about.
    if (!d.is_running) sub.push(['AI monitoring off', 'warn']);
  } else if (d.is_running && tone !== 'run') {
    sub.push([s]);
  } else if (!d.is_running && tone === 'idle' && s !== 'starting') {
    sub.push(['AI monitoring off']);
  }

  document.getElementById('headline').className = 'headline t-' + tone;
  document.getElementById('hl-main').textContent = main;
  const subEl = document.getElementById('hl-sub');
  subEl.innerHTML = '';
  sub.forEach(([text, cls]) => {
    const el = document.createElement('span');
    el.textContent = text; if (cls) el.className = cls;
    subEl.appendChild(el);
  });
  document.getElementById('hl-bar').style.display = bar ? 'block' : 'none';
  if (bar) document.getElementById('hl-bar-fill').style.width = (p.progress * 100) + '%';
  document.title = (tone === 'fail' ? '⚠ ' : '') +
    (p.connected && p.printing && pctStr ? pctStr + ' · ' : '') + 'Ender3Monitor';
}

// ── Action bar ─────────────────────────────────────────────────────────────
const ICON_PLAY  = '<polygon points="6 4 20 12 6 20"/>';
const ICON_STOP  = '<rect x="5" y="5" width="14" height="14" rx="2"/>';
const ICON_PAUSE = '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>';

function renderActionBar(d) {
  const p = d.printer || {};
  document.getElementById('actionbar').classList.toggle('has-printer', !!p.connected);

  const mon = document.getElementById('btn-monitor');
  mon.classList.toggle('on', !!d.is_running);
  document.getElementById('mon-label').textContent = d.is_running ? 'Stop AI' : 'Start AI';
  document.getElementById('mon-icon').innerHTML = d.is_running ? ICON_STOP : ICON_PLAY;

  const pb = document.getElementById('btn-pause');
  pb.classList.toggle('resume', !!p.paused);
  document.getElementById('pause-label').textContent = p.paused ? 'Resume' : 'Pause';
  document.getElementById('pause-icon').innerHTML = p.paused ? ICON_PLAY : ICON_PAUSE;
}

function toggleMonitor() {
  if (!lastState.is_running) return doStart();
  const p = lastState.printer || {};
  if (p.connected && p.printing &&
      !confirm('Stop AI monitoring? The print keeps running, unwatched.')) return;
  doStop();
}

function togglePause() {
  doPrinter((lastState.printer || {}).paused ? 'resume' : 'pause');
}

function doCooldown() {
  if (confirm('Pause the print and turn the heaters off?')) doPrinter('cooldown');
}

// E-Stop: press and hold, so a stray thumb can't brick a 10-hour print.
(function() {
  const btn = document.getElementById('btn-estop');
  const HOLD_MS = 1500;
  let timer = null;
  const start = () => {
    if (timer) return;
    btn.classList.add('holding');
    timer = setTimeout(() => {
      timer = null;
      btn.classList.remove('holding');
      if (navigator.vibrate) navigator.vibrate(200);
      doPrinter('estop');
    }, HOLD_MS);
  };
  const cancel = () => {
    if (!timer) return;
    clearTimeout(timer); timer = null;
    btn.classList.remove('holding');
    toast('Hold for 1.5 s to emergency stop');
  };
  btn.addEventListener('pointerdown', e => { e.preventDefault(); start(); });
  ['pointerup', 'pointerleave', 'pointercancel', 'blur'].forEach(ev =>
    btn.addEventListener(ev, cancel));
  btn.addEventListener('contextmenu', e => e.preventDefault());
  const isKey = e => e.key === ' ' || e.key === 'Enter';
  btn.addEventListener('keydown', e => { if (isKey(e)) { e.preventDefault(); start(); } });
  btn.addEventListener('keyup',   e => { if (isKey(e)) cancel(); });
})();

// ── More sheet ─────────────────────────────────────────────────────────────
function openMore()  { document.getElementById('more-overlay').classList.add('show'); }
function closeMore() { document.getElementById('more-overlay').classList.remove('show'); }

function pillClass(s) {
  const l = s.toLowerCase();
  if (l.includes('failure') || l.includes('detected')) return 'pill-fail';
  if (l.includes('complete'))                            return 'pill-done';
  if (l.includes('paused') || l.includes('change filament')) return 'pill-pause';
  if (l.includes('monitor') || l.includes('analyz'))    return 'pill-run';
  return 'pill-idle';
}

// ── Camera selection ───────────────────────────────────────────────────────
async function scanCameras(forceHardwareScan = false) {
  const sel = document.getElementById('cam-select');
  sel.disabled = true;
  sel.innerHTML = forceHardwareScan
    ? '<option value="">Scanning hardware…</option>'
    : '<option value="">Loading…</option>';
  try {
    const url = forceHardwareScan ? '/api/cameras?scan=true' : '/api/cameras';
    const res  = await fetch(url);
    const data = await res.json();
    camList = data.cameras || [];
    const configured = data.configured;

    if (data.error) {
      sel.innerHTML = `<option value="">${data.error}</option>`;
    } else if (camList.length === 0) {
      sel.innerHTML = '<option value="">No cameras found</option>';
    } else {
      sel.innerHTML = camList.map(c => {
        const label = `Camera ${c.index} — ${c.width}×${c.height}`;
        const selected = (c.index === configured || (configured < 0 && camList.indexOf(c) === 0)) ? 'selected' : '';
        return `<option value="${c.index}" ${selected}>${label}</option>`;
      }).join('');
    }
  } catch(e) {
    sel.innerHTML = '<option value="">Scan failed — retry</option>';
  }
  sel.disabled = false;
}

function selectedCam() {
  const sel = document.getElementById('cam-select');
  const v = sel ? parseInt(sel.value, 10) : NaN;
  return isNaN(v) ? null : v;
}

// Switch the live preview to the chosen camera (only when not monitoring)
async function previewCamera() {
  const cam = selectedCam();
  if (cam === null) return;
  try {
    const res = await fetch('/api/preview', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ camera_index: cam }),
    });
    if (res.ok) {
      toast(`Live view → Camera ${cam}`);
    } else if (res.status === 409) {
      toast('Stop monitoring before switching camera');
    }
  } catch(e) {}
}

// ── Controls ───────────────────────────────────────────────────────────────
async function doStart() {
  const cam = selectedCam();
  const body = cam !== null ? { camera_index: cam } : {};
  const data = await busy('btn-monitor', () => api('start', body));
  if (data) render(data);   // after busy() so render sets the final disabled state
}
async function doStop() {
  const data = await busy('btn-monitor', () => api('stop'));
  if (data) render(data);
}
async function doTimelapse() {
  const btn = document.getElementById('btn-timelapse');
  btn.disabled = true;
  toast('Compiling timelapse…');
  const data = await api('timelapse');
  btn.disabled = false;
  if (data && data.file) {
    toast('Downloading ' + data.file);
    // Content-Disposition: attachment triggers a download without navigating away
    window.location = '/download/timelapse?name=' + encodeURIComponent(data.file);
  } else {
    toast((data && data.error) ? data.error : 'No timelapse to download yet');
  }
}

// Toggle auto-start-on-print
async function toggleAutoStart() {
  const cb = document.getElementById('autostart-cb');
  try {
    const res = await fetch('/api/autostart', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ enabled: cb.checked }),
    });
    const d = await res.json();
    toast('Auto-start ' + (d.auto_start ? 'on' : 'off'));
  } catch(e) { toast('Could not update auto-start'); }
}

// Manual printer control over USB
async function doPrinter(action, btn) {
  await busy(btn, async () => {
    try {
      const res = await fetch('/api/printer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ action }),
      });
      const data = await res.json();
      toast(res.ok ? (data.message || 'Done') : (data.error || 'Printer command failed'));
    } catch(e) {
      toast('Printer command failed');
    }
  });
}

// Returns parsed JSON on success, null on error
async function api(action, body = null) {
  try {
    const res = await fetch(`/api/${action}`, {
      method: 'POST',
      headers: body ? {'Content-Type': 'application/json'} : {},
      body:    body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      // Show the server's own message (e.g. "Printer not connected") — the
      // status text alone is often empty or just "Conflict".
      let msg = res.statusText || ('HTTP ' + res.status);
      try { const j = await res.json(); if (j && j.error) msg = j.error; } catch(e) {}
      toast('Error: ' + msg);
      return null;
    }
    return await res.json();
  } catch(e) { toast('Network error'); return null; }
}

// ── Event log (server-driven; persists across reloads) ──────────────────────
function esc(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function renderEvents(events) {
  // Rebuilding the log's DOM on every push re-laid-out up to 50 rows and
  // wiped any text selection/hover; only rebuild when the events changed.
  const key = JSON.stringify(events);
  if (key === lastEventsKey) return;
  lastEventsKey = key;
  const log = document.getElementById('log');
  if (!events.length) {
    log.innerHTML = '<div class="log-empty">No events yet — start monitoring to begin.</div>';
    return;
  }
  // Newest first.
  const rows = events.slice().reverse().map(e => {
    const t = new Date((e.t || 0) * 1000).toTimeString().slice(0, 8);
    const isFail = e.detected;
    const ft = e.type || 'none';
    const cls = isFail ? 'badge-fail' : 'badge-ok';
    const lbl = isFail ? ft : (ft === 'no_printer' ? 'no printer' : 'ok');
    const conf = e.conf != null ? Math.round(e.conf * 100) + '%' : '';
    return `<div class="ev">
      <span class="ev-time">${t}</span>
      <span class="badge ${cls}">${esc(lbl)}</span>
      <span class="ev-desc">${esc((e.desc || '').slice(0, 220))}</span>
      <span class="ev-conf">${conf}</span>
    </div>`;
  });
  log.innerHTML = rows.join('');
}

async function clearLog() {
  await api('events/clear');   // server clears; next state push re-renders empty
  lastEventsKey = '[]';        // the empty list is already shown below
  document.getElementById('log').innerHTML =
    '<div class="log-empty">Log cleared.</div>';
}

// ── Stream link — show full LAN URL so user can open on phone ─────────────
(function() {
  const a = document.getElementById('stream-url');
  if (a) {
    const url = `${location.protocol}//${location.hostname}:${location.port}/stream`;
    a.href = url;
    a.textContent = url;
  }
})();

// ── Charts (temperature history + layer progress) ──────────────────────────
// History comes once from /api/temps; after that each websocket state carries
// the newest reading (printer.temps_at), appended here so pushes stay small.
const CH_WINDOW = 1800, CH_MAX = 360, CH_GAP = 30;   // seconds, samples, seconds
let chTemps = [], chLast = null, chLoadedAt = 0;

function chMerge(rows) {
  const seen = new Set(chTemps.map(r => r[0]));
  rows.forEach(r => {
    if (Array.isArray(r) && r[0] != null && r[1] != null && r[3] != null && !seen.has(r[0])) {
      seen.add(r[0]); chTemps.push(r);
    }
  });
  chTemps.sort((a, b) => a[0] - b[0]);
  if (chTemps.length) {
    const cutoff = chTemps[chTemps.length - 1][0] - CH_WINDOW;
    chTemps = chTemps.filter(r => r[0] >= cutoff).slice(-CH_MAX);
  }
}

async function loadTemps() {
  chLoadedAt = Date.now();
  try {
    const r = await fetch('/api/temps');
    if (!r.ok) return;
    const j = await r.json();
    chMerge(j.samples || []);
    if (chLast) renderCharts(chLast);
  } catch (e) { /* chart just fills in from live updates */ }
}

function chNiceMax(v) {
  const steps = [30, 60, 90, 120, 150, 180, 240, 300, 360, 450, 600];
  for (const s of steps) if (v * 1.05 <= s) return s;
  return Math.ceil(v * 1.05 / 150) * 150;
}

// Pure: builds the SVG markup for the temperature plot at a given pixel size.
function chTempsSvg(rows, w, h) {
  const L = 34, R = 10, T = 8, B = 20, pw = Math.max(10, w - L - R), ph = h - T - B;
  if (rows.length < 2) {
    return `<text class="ch-empty" x="${w / 2}" y="${h / 2}" text-anchor="middle">` +
           'Collecting temperature data…</text>';
  }
  const tEnd = rows[rows.length - 1][0], t0 = tEnd - CH_WINDOW;
  let top = 0;
  rows.forEach(r => { for (let i = 1; i < 5; i++) if (r[i] != null && r[i] > top) top = r[i]; });
  const yMax = chNiceMax(Math.max(top, 25));
  const X = t => L + Math.min(1, Math.max(0, (t - t0) / CH_WINDOW)) * pw;
  const Y = v => T + (1 - Math.min(1, Math.max(0, v / yMax))) * ph;
  const f = n => n.toFixed(1);
  // One path per series; a new sub-path starts after a gap in the data (a
  // disconnect) or, for targets, wherever the heater is off.
  const path = (col, isTarget) => {
    let d = '', prev = null;
    rows.forEach(r => {
      const v = r[col];
      if (v == null || (isTarget && v <= 0)) { prev = null; return; }
      const brk = prev == null || r[0] - prev > CH_GAP;
      d += (brk ? 'M' : 'L') + f(X(r[0])) + ' ' + f(Y(v));
      prev = r[0];
    });
    return d;
  };
  let out = '';
  for (let i = 0; i <= 3; i++) {
    const v = yMax * i / 3, y = Math.round(Y(v)) + .5;
    out += `<line class="ch-grid" x1="${L}" x2="${L + pw}" y1="${y}" y2="${y}"/>`;
    if (i) out += `<text class="ch-axis" x="${L - 6}" y="${y + 3.5}" text-anchor="end">${Math.round(v)}°</text>`;
  }
  out += `<text class="ch-axis" x="${L}" y="${h - 4}">30 min ago</text>`;
  if (pw > 220) out += `<text class="ch-axis" x="${L + pw / 2}" y="${h - 4}" text-anchor="middle">15 min</text>`;
  out += `<text class="ch-axis" x="${L + pw}" y="${h - 4}" text-anchor="end">now · °C</text>`;
  const series = [[4, 'ch-target ch-bed', true], [2, 'ch-target ch-nozzle', true],
                  [3, 'ch-line ch-bed', false], [1, 'ch-line ch-nozzle', false]];
  series.forEach(([col, cls, isTarget]) => {
    const d = path(col, isTarget);
    if (d) out += `<path class="${cls}" d="${d}"/>`;
  });
  const last = rows[rows.length - 1];
  out += `<circle class="ch-dot ch-bed" cx="${f(X(last[0]))}" cy="${f(Y(last[3]))}" r="4"/>`;
  out += `<circle class="ch-dot ch-nozzle" cx="${f(X(last[0]))}" cy="${f(Y(last[1]))}" r="4"/>`;
  return out;
}

function chLegendHtml(p) {
  const val = (t, tgt) => t == null ? '—'
    : Math.round(t) + '°' + (tgt ? ' <span>/ ' + Math.round(tgt) + '°</span>' : '');
  return '<span class="ch-key"><i class="ch-swatch" style="border-color:var(--amber)"></i>Nozzle <b>' +
           val(p.nozzle_temp, p.nozzle_target) + '</b></span>' +
         '<span class="ch-key"><i class="ch-swatch" style="border-color:var(--blue)"></i>Bed <b>' +
           val(p.bed_temp, p.bed_target) + '</b></span>' +
         '<span class="ch-key"><i class="ch-swatch dash"></i>Target</span>';
}

// Pure: tick marks for the layer bar. marks are 0..1 fractions of total layers.
function chTicksHtml(marks, progress) {
  return (marks || []).filter(m => m > 0 && m < 1).map(m =>
    `<i class="ch-tick${m <= progress ? ' done' : ''}" style="left:${(m * 100).toFixed(2)}%" title="Color change"></i>`
  ).join('');
}

function renderCharts(d) {
  chLast = d;
  const p = d.printer || {};
  const card = document.getElementById('charts-card');
  if (!card) return;
  if (!p.connected) { card.style.display = 'none'; return; }
  card.style.display = 'block';

  if (p.temps_at != null && p.nozzle_temp != null && p.bed_temp != null) {
    const lastTs = chTemps.length ? chTemps[chTemps.length - 1][0] : null;
    // Missed readings (tab asleep, socket dropped): refill from the server.
    if (lastTs != null && p.temps_at - lastTs > CH_GAP && Date.now() - chLoadedAt > 10000) loadTemps();
    chMerge([[p.temps_at, p.nozzle_temp, p.nozzle_target, p.bed_temp, p.bed_target]]);
  }

  document.getElementById('ch-legend').innerHTML = chLegendHtml(p);
  const svg = document.getElementById('ch-temps');
  const w = Math.max(240, Math.round(svg.clientWidth || 600)), h = 170;
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.innerHTML = chTempsSvg(chTemps, w, h);

  const layer = document.getElementById('ch-layer');
  if (p.printing && p.progress != null) {
    const frac = Math.min(1, Math.max(0, p.progress));
    const marks = p.total_layers ? (p.color_change_marks || []) : [];
    layer.style.display = 'block';
    document.getElementById('ch-layer-label').textContent =
      p.layer_str ? 'Layer ' + p.layer_str : 'Print progress';
    document.getElementById('ch-layer-pct').textContent = (frac * 100).toFixed(1) + '%';
    document.getElementById('ch-bar-fill').style.width = (frac * 100).toFixed(2) + '%';
    const bar = document.getElementById('ch-bar');
    bar.querySelectorAll('.ch-tick').forEach(n => n.remove());
    bar.insertAdjacentHTML('beforeend', chTicksHtml(marks, frac));
    const n = marks.filter(m => m > 0 && m < 1).length;
    const note = document.getElementById('ch-layer-note');
    note.textContent = n ? n + ' color change' + (n > 1 ? 's' : '') + ' marked' : '';
    note.style.display = n ? 'block' : 'none';
  } else {
    layer.style.display = 'none';
  }
}

// Redraw at the new pixel width so axis text stays a readable size.
if (window.ResizeObserver) {
  new ResizeObserver(() => { if (chLast) renderCharts(chLast); })
    .observe(document.getElementById('ch-temps'));
}

// ── Failure review ─────────────────────────────────────────────────────────
let frShownId = null, frBusy = false, sparkKey = '';

function renderFailureReview(d) {
  const card = document.getElementById('fr-card');
  if (!card) return;
  const fr = d.failure_review;
  if (!fr) {
    card.classList.remove('show');
    frShownId = null;
    return;
  }
  card.classList.add('show');
  if (fr.id === frShownId) return;   // same incident — don't reload the image
  frShownId = fr.id;
  const img = document.getElementById('fr-img');
  if (fr.image_url) { img.src = fr.image_url; img.style.display = ''; }
  else { img.removeAttribute('src'); img.style.display = 'none'; }
  document.getElementById('fr-type').textContent = fr.failure_type || 'failure';
  document.getElementById('fr-conf').textContent =
    fr.confidence != null ? Math.round(fr.confidence * 100) + '% confidence' : '';
  document.getElementById('fr-time').textContent =
    fr.timestamp ? new Date(fr.timestamp * 1000).toTimeString().slice(0, 8) : '';
  document.getElementById('fr-desc').innerHTML =
    esc(fr.description || 'No description from the analyzer.');
}

function renderSparkline(d) {
  const line = document.getElementById('spark-line');
  if (!line) return;
  const hist = Array.isArray(d.confidence_history) ? d.confidence_history : [];
  const thr = d.threshold ?? 0.7;
  const fa = d.false_alarm_count || 0;
  const key = hist.length + ':' + (hist.length ? hist[hist.length - 1][0] : 0) + ':' + thr + ':' + fa;
  if (key === sparkKey) return;
  sparkKey = key;
  // 0–100% maps to y 34–2 (2px breathing room top and bottom of the 36px box).
  const y = c => (34 - Math.max(0, Math.min(1, c)) * 32).toFixed(1);
  const ty = y(thr);
  const tl = document.getElementById('spark-thresh');
  tl.setAttribute('y1', ty); tl.setAttribute('y2', ty);
  let pts = '';
  if (hist.length === 1) pts = `0,${y(hist[0][1])} 100,${y(hist[0][1])}`;
  else pts = hist.map((p, i) => `${(i / (hist.length - 1) * 100).toFixed(1)},${y(p[1])}`).join(' ');
  line.setAttribute('points', pts);
  document.getElementById('spark-span').textContent = hist.length
    ? `last ${hist.length} ${hist.length === 1 ? 'analysis' : 'analyses'} · dashed = threshold`
    : 'no analyses yet';
  document.getElementById('spark-false').textContent =
    fa ? `${fa} false alarm${fa === 1 ? '' : 's'}` : '';
}

async function resolveFailure(action) {
  if (frBusy) return;
  frBusy = true;
  const btns = document.querySelectorAll('#fr-card .fr-actions button');
  btns.forEach(b => b.disabled = true);
  try {
    const res = await fetch('/api/failure/resolve', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ action }),
    });
    let data = {};
    try { data = await res.json(); } catch(e) {}
    toast(res.ok ? (data.message || 'Done') : (data.error || 'Could not resolve the failure'));
  } catch(e) {
    toast('Network error');
  } finally {
    frBusy = false;
    btns.forEach(b => b.disabled = false);
  }
}

// ── Toast ──────────────────────────────────────────────────────────────────
function toast(msg, ms=3000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), ms);
}

// ── Settings panel ──────────────────────────────────────────────────────────
let settingsSchema = null, settingsValues = {};

async function openSettings() {
  const overlay = document.getElementById('settings-overlay');
  const body = document.getElementById('settings-body');
  const msg = document.getElementById('settings-msg');
  msg.textContent = ''; msg.className = '';
  body.innerHTML = '<div style="padding:24px;color:var(--muted)">Loading…</div>';
  overlay.classList.add('show');
  let data;
  try {
    const res = await fetch('/api/settings');
    if (!res.ok) throw new Error(res.status);
    data = await res.json();
  } catch(e) {
    body.innerHTML = '<div style="padding:24px;color:var(--red)">Could not load settings.</div>';
    return;
  }
  settingsSchema = data.schema; settingsValues = data.values;
  document.getElementById('settings-secwarn').style.display =
    data.info && data.info.auth_enabled ? 'none' : 'block';
  renderSettings(body, data);
}

function renderSettings(body, data) {
  // Group fields in schema order
  const groups = [];
  const byGroup = {};
  for (const f of settingsSchema) {
    if (!byGroup[f.group]) { byGroup[f.group] = []; groups.push(f.group); }
    byGroup[f.group].push(f);
  }
  let html = '';
  for (const g of groups) {
    html += `<div class="set-group-label">${esc(g)}</div>`;
    for (const f of byGroup[g]) html += fieldRow(f);
  }
  // Read-only structural info (never secrets)
  if (data.info) {
    html += `<div class="set-group-label">System (set in .env — restart to change)</div>`;
    html += infoRow('AI backend', `${esc(data.info.backend)} · ${esc(data.info.model||'—')}`);
    html += infoRow('Configured camera index', data.info.camera_index);
    html += infoRow('Dashboard login', data.info.auth_enabled ? 'enabled' : 'OFF');
  }
  body.innerHTML = html;
}

function fieldRow(f) {
  const v = settingsValues[f.key];
  const restart = f.live ? '' : '<span class="set-restart">restart</span>';
  let control = '';
  if (f.type === 'bool') {
    control = `<label class="tgl"><input type="checkbox" data-key="${f.key}" ${v ? 'checked':''}>
                 <span class="track"><span class="knob"></span></span></label>`;
  } else if (f.type === 'enum') {
    control = `<select data-key="${f.key}">` +
      f.choices.map(c => `<option ${c===v?'selected':''} value="${esc(c)}">${esc(c)}</option>`).join('') +
      `</select>`;
  } else {
    const step = f.type === 'float' ? '0.05' : '1';
    const mn = f.min!=null ? `min="${f.min}"` : '', mx = f.max!=null ? `max="${f.max}"` : '';
    control = `<input type="number" data-key="${f.key}" data-type="${f.type}" value="${esc(v)}" step="${step}" ${mn} ${mx}>`;
  }
  return `<div class="set-row">
    <div class="set-info">
      <div class="set-name">${esc(f.label)} ${restart}</div>
      ${f.help ? `<div class="set-help">${esc(f.help)}</div>` : ''}
    </div>
    <div class="set-control">${control}</div>
  </div>`;
}

function infoRow(label, val) {
  return `<div class="set-info-row"><span>${esc(label)}</span><span>${esc(val)}</span></div>`;
}

function closeSettings() {
  document.getElementById('settings-overlay').classList.remove('show');
}

async function saveSettings() {
  const body = {};
  document.querySelectorAll('#settings-body [data-key]').forEach(el => {
    const k = el.dataset.key;
    if (el.type === 'checkbox') body[k] = el.checked;
    else if (el.dataset.type) body[k] = el.value === '' ? null : Number(el.value);
    else body[k] = el.value;
  });
  // Only send changed values
  const changed = {};
  for (const k in body) if (body[k] !== settingsValues[k]) changed[k] = body[k];
  const msg = document.getElementById('settings-msg');
  if (!Object.keys(changed).length) { msg.className=''; msg.textContent='No changes.'; return; }

  const btn = document.getElementById('settings-save');
  btn.disabled = true; msg.className=''; msg.textContent = 'Saving…';
  try {
    const res = await fetch('/api/settings', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(changed)
    });
    const data = await res.json();
    settingsValues = data.values || settingsValues;
    if (data.errors && data.errors.length) {
      msg.className='err'; msg.textContent = data.errors.join('  ·  ');
    } else {
      msg.className='ok';
      msg.textContent = (data.restart_required && data.restart_required.length)
        ? 'Saved — some changes need a restart.' : 'Saved ✓';
      toast('Settings saved');
      setTimeout(closeSettings, 700);
    }
  } catch(e) {
    msg.className='err'; msg.textContent = 'Save failed.';
  } finally {
    btn.disabled = false;
  }
}

// Esc closes the modal and the More sheet
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeSettings(); closeMore(); } });

// ── Boot ───────────────────────────────────────────────────────────────────
scanCameras();
loadTemps();
connect();
</script>
</body>
</html>
"""

# ── Print history page (separate from the dashboard's _HTML) ─────────────────

@app.get("/history", response_class=HTMLResponse)
async def history_page():
    return HTMLResponse(_HISTORY_HTML, headers={"Cache-Control": "no-cache"})


_HISTORY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Print history – Ender3Monitor</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{overflow-x:hidden;max-width:100%}

:root{
  --bg:#070b14;
  --surf:#0d1120;
  --surf2:#121729;
  --surf3:#181f30;
  --border:rgba(255,255,255,0.07);
  --blue:#4f8ef7;
  --green:#22c55e;
  --red:#f87171;
  --amber:#fbbf24;
  --text:#dde4f0;
  --muted:#4a5878;
  --soft:#8693ad;
  --r:12px;
}

body{
  background:var(--bg);
  color:var(--text);
  font-family:system-ui,-apple-system,'Segoe UI',sans-serif;
  font-size:14px;
  line-height:1.5;
  min-height:100vh;
  overflow-x:hidden;
}

header{
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  padding:14px 28px;
  background:var(--surf);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:10;
}
.logo{display:flex;align-items:center;gap:10px;font-size:17px;font-weight:700;
  letter-spacing:-.3px;color:var(--text);text-decoration:none;min-width:0}
.logo svg{opacity:.85;flex:none}
.logo span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.header-right{display:flex;align-items:center;gap:12px;flex:none}
.icon-btn{
  display:flex;align-items:center;justify-content:center;
  width:34px;height:34px;padding:0;border-radius:9px;
  background:var(--surf3);border:1px solid var(--border);color:var(--muted);
  cursor:pointer;transition:all .15s;text-decoration:none;
}
.icon-btn:hover{color:var(--text);border-color:rgba(255,255,255,.14)}
.back{
  display:flex;align-items:center;gap:6px;height:34px;padding:0 12px 0 8px;
  border-radius:9px;background:var(--surf3);border:1px solid var(--border);
  color:var(--soft);font-size:13px;font-weight:600;text-decoration:none;
  transition:all .15s;
}
.back:hover{color:var(--text);border-color:rgba(255,255,255,.14)}

.page{max-width:900px;margin:0 auto;padding:22px 28px 48px}
.page-head{display:flex;align-items:baseline;justify-content:space-between;
  gap:12px;margin-bottom:14px}
.page-head h1{font-size:20px;font-weight:700;letter-spacing:-.3px}
.count{color:var(--soft);font-size:12.5px;font-variant-numeric:tabular-nums}

.list{display:flex;flex-direction:column;gap:10px}
.card{
  background:var(--surf);border:1px solid var(--border);border-radius:var(--r);
  overflow:hidden;
}
.card.open{border-color:rgba(255,255,255,.13)}
.card-head{
  display:flex;align-items:center;gap:14px;width:100%;
  padding:12px;background:none;border:0;color:inherit;font:inherit;
  text-align:left;cursor:pointer;
}
.card-head:hover{background:rgba(255,255,255,.02)}
.card-head:focus-visible{outline:2px solid var(--blue);outline-offset:-2px;border-radius:var(--r)}
.thumb{
  flex:none;width:96px;height:72px;border-radius:8px;overflow:hidden;
  background:var(--surf3);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;color:var(--muted);
}
.thumb img{width:100%;height:100%;object-fit:cover;display:block}
.body{flex:1;min-width:0;display:flex;flex-direction:column;gap:3px}
.row1{display:flex;align-items:center;gap:8px;min-width:0}
.name{flex:1;min-width:0;font-size:14.5px;font-weight:600;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.when{color:var(--soft);font-size:12.5px}
.stats{display:flex;flex-wrap:wrap;gap:2px 14px;color:var(--soft);font-size:12.5px;
  font-variant-numeric:tabular-nums}
.stats b{color:var(--text);font-weight:600}
.stats .bad b{color:var(--red)}
.chev{flex:none;color:var(--muted);transition:transform .15s}
.card.open .chev{transform:rotate(180deg)}

.badge{
  flex:none;display:inline-flex;align-items:center;
  padding:2px 9px;border-radius:100px;
  font-size:10.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
  white-space:nowrap;
}
.b-completed{background:rgba(34,197,94,.12);color:var(--green)}
.b-stopped{background:rgba(134,147,173,.14);color:var(--soft)}
.b-failed{background:rgba(248,113,113,.12);color:var(--red)}
.b-ok{background:rgba(34,197,94,.12);color:var(--green)}
.b-fail{background:rgba(248,113,113,.12);color:var(--red)}

.detail{display:none;border-top:1px solid var(--border);padding:14px 12px 12px;
  background:var(--surf2)}
.card.open .detail{display:block}
.facts{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
  gap:10px 16px;margin-bottom:14px}
.fact .k{color:var(--soft);font-size:11px;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase}
.fact .v{font-size:13.5px;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}
.sec{color:var(--soft);font-size:11px;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;margin-bottom:6px}
.log{
  max-height:320px;overflow-y:auto;border:1px solid var(--border);
  border-radius:8px;background:var(--surf);
}
.ev{display:grid;grid-template-columns:auto auto minmax(0,1fr) auto;gap:8px;
  align-items:start;padding:7px 10px;font-size:12.5px;
  border-bottom:1px solid var(--border)}
.ev:last-child{border-bottom:0}
.ev-time{color:var(--soft);font-family:monospace;font-size:11.5px;padding-top:2px}
.ev-desc{min-width:0;overflow-wrap:anywhere}
.ev-conf{color:var(--soft);font-family:monospace;font-size:11.5px;padding-top:2px}
.note{color:var(--soft);font-size:12.5px;padding:12px 10px}
.actions{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.btn{
  display:inline-flex;align-items:center;gap:7px;min-height:36px;padding:0 14px;
  border-radius:9px;background:var(--surf3);border:1px solid var(--border);
  color:var(--text);font:inherit;font-size:13px;font-weight:600;
  text-decoration:none;cursor:pointer;transition:all .15s;
}
.btn:hover{border-color:rgba(255,255,255,.18)}
.btn-primary{background:rgba(79,142,247,.14);border-color:rgba(79,142,247,.3);color:var(--blue)}
.btn-danger{color:var(--red);margin-left:auto}
.btn-danger:hover{border-color:rgba(248,113,113,.4)}

.empty{
  text-align:center;padding:56px 20px;border:1px dashed rgba(255,255,255,.1);
  border-radius:var(--r);color:var(--soft);
}
.empty svg{color:var(--muted);margin-bottom:10px}
.empty h2{color:var(--text);font-size:16px;font-weight:600;margin-bottom:4px}
.empty a{color:var(--blue);text-decoration:none}

@media (max-width:600px){
  header{padding:12px 16px}
  .page{padding:16px 12px 40px}
  .card-head{gap:11px;padding:10px}
  .thumb{width:76px;height:57px}
  .chev{display:none}
  .ev{grid-template-columns:auto minmax(0,1fr) auto}
  .ev .badge{justify-self:start}
  .ev-desc{grid-column:1 / -1;grid-row:2}
  .btn-danger{margin-left:0}
}
</style>
</head>
<body>

<header>
  <a class="logo" href="/" title="Back to dashboard">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.75" stroke-linecap="round">
      <path d="M3 21h18M5 21V9l7-6 7 6v12"/>
      <rect x="9" y="13" width="6" height="8" rx="1"/>
      <path d="M9 13h6"/>
    </svg>
    <span>Ender3Monitor</span>
  </a>
  <div class="header-right">
    <a class="back" href="/">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M15 18l-6-6 6-6"/>
      </svg>
      Dashboard
    </a>
  </div>
</header>

<main class="page">
  <div class="page-head">
    <h1>Print history</h1>
    <span class="count" id="count"></span>
  </div>
  <div class="list" id="list"><div class="note">Loading…</div></div>
</main>

<script>
function esc(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function fmtDuration(sec) {
  sec = Math.max(0, Math.round(Number(sec) || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
  if (h) return h + 'h ' + m + 'm';
  if (m) return m + 'm';
  return sec + 's';
}

function fmtDate(epoch) {
  if (!epoch) return '—';
  return new Date(epoch * 1000).toLocaleString(undefined,
    {month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'});
}

function fmtCost(v) {
  v = Number(v) || 0;
  return '$' + (v > 0 && v < 0.01 ? v.toFixed(3) : v.toFixed(2));
}

const OUTCOMES = {completed: 'Completed', stopped: 'Stopped', failed: 'Failed'};
function outcomeBadge(o) {
  const key = OUTCOMES[o] ? o : 'stopped';
  return `<span class="badge b-${key}">${OUTCOMES[key]}</span>`;
}

const PHOTO_ICON = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
  <rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="10" r="1.6"/>
  <path d="M21 16l-5-5-8 8"/></svg>`;

function cardHtml(p) {
  const id = encodeURIComponent(p.id);
  const thumb = p.has_thumbnail
    ? `<img src="/api/history/${id}/thumb" alt="" loading="lazy">`
    : PHOTO_ICON;
  const fails = Number(p.failure_count) || 0;
  const alarms = Number(p.false_alarms) || 0;
  return `<article class="card" data-id="${esc(p.id)}">
    <button class="card-head" type="button" aria-expanded="false">
      <span class="thumb">${thumb}</span>
      <span class="body">
        <span class="row1">
          <span class="name">${esc(p.job_name || fmtDate(p.ended_at))}</span>
          ${outcomeBadge(p.outcome)}
        </span>
        <span class="when">${p.job_name ? esc(fmtDate(p.ended_at)) + ' · ' : ''}${esc(fmtDuration(p.duration_seconds))}</span>
        <span class="stats">
          <span>AI <b>${esc(fmtCost(p.ai_cost_usd))}</b></span>
          <span class="${fails ? 'bad' : ''}"><b>${fails}</b> failure${fails === 1 ? '' : 's'}</span>
          <span><b>${alarms}</b> false alarm${alarms === 1 ? '' : 's'}</span>
        </span>
      </span>
      <svg class="chev" width="18" height="18" viewBox="0 0 24 24" fill="none"
           stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M6 9l6 6 6-6"/></svg>
    </button>
    <div class="detail"></div>
  </article>`;
}

function fact(k, v) {
  return `<div class="fact"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div></div>`;
}

function eventsHtml(events) {
  if (!events || !events.length) {
    return '<div class="note">No events were logged for this print.</div>';
  }
  return events.slice().reverse().map(e => {
    const t = new Date((e.t || 0) * 1000).toTimeString().slice(0, 8);
    const ft = e.type || 'none';
    const lbl = e.detected ? ft : (ft === 'no_printer' ? 'no printer' : (ft === 'none' ? 'ok' : ft));
    const conf = e.conf != null ? Math.round(e.conf * 100) + '%' : '';
    return `<div class="ev">
      <span class="ev-time">${esc(t)}</span>
      <span class="badge ${e.detected ? 'b-fail' : 'b-ok'}">${esc(String(lbl).replace(/_/g, ' '))}</span>
      <span class="ev-desc">${esc(e.desc || '')}</span>
      <span class="ev-conf">${esc(conf)}</span>
    </div>`;
  }).join('');
}

function detailHtml(p) {
  const types = (p.failure_types || []).join(', ');
  let h = '<div class="facts">'
    + fact('Started', fmtDate(p.started_at))
    + fact('Ended', fmtDate(p.ended_at))
    + fact('Duration', fmtDuration(p.duration_seconds))
    + fact('Frames analysed', p.frames_analyzed != null ? p.frames_analyzed : '—')
    + (p.total_layers ? fact('Layers', p.total_layers) : '')
    + (types ? fact('Failure types', types) : '')
    + '</div>';
  h += '<div class="sec">Event log</div><div class="log">' + eventsHtml(p.events) + '</div>';
  h += '<div class="actions">';
  if (p.timelapse && p.timelapse_available) {
    h += `<a class="btn btn-primary" href="/download/timelapse?name=${encodeURIComponent(p.timelapse)}">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 3v12M7 10l5 5 5-5M5 21h14"/></svg>
      Download timelapse</a>`;
  }
  h += '<button class="btn btn-danger" type="button" data-delete>Delete</button></div>';
  return h;
}

const listEl = document.getElementById('list');
const countEl = document.getElementById('count');

function renderList(prints) {
  countEl.textContent = prints.length ? prints.length + (prints.length === 1 ? ' print' : ' prints') : '';
  if (!prints.length) {
    listEl.innerHTML = `<div class="empty">
      <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/></svg>
      <h2>No prints yet</h2>
      <p>Each monitored print is saved here when it finishes or is stopped.
         <a href="/">Back to the dashboard</a></p>
    </div>`;
    return;
  }
  listEl.innerHTML = prints.map(cardHtml).join('');
}

async function load() {
  try {
    const r = await fetch('/api/history');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const data = await r.json();
    renderList(data.prints || []);
  } catch (err) {
    countEl.textContent = '';
    listEl.innerHTML = '<div class="note">Could not load print history ('
      + esc(err.message) + '). <a class="btn" href="/history">Retry</a></div>';
  }
}

async function toggle(card) {
  const open = card.classList.toggle('open');
  card.querySelector('.card-head').setAttribute('aria-expanded', open ? 'true' : 'false');
  if (!open || card.dataset.loaded) return;
  const detail = card.querySelector('.detail');
  detail.innerHTML = '<div class="note">Loading…</div>';
  try {
    const r = await fetch('/api/history/' + encodeURIComponent(card.dataset.id));
    if (!r.ok) throw new Error('HTTP ' + r.status);
    detail.innerHTML = detailHtml(await r.json());
    card.dataset.loaded = '1';
  } catch (err) {
    detail.innerHTML = '<div class="note">Could not load this print (' + esc(err.message) + ').</div>';
  }
}

async function removeCard(card) {
  if (!confirm('Delete this print from the history?')) return;
  try {
    const r = await fetch('/api/history/' + encodeURIComponent(card.dataset.id), {method: 'DELETE'});
    if (!r.ok && r.status !== 404) throw new Error('HTTP ' + r.status);
    load();
  } catch (err) {
    alert('Delete failed: ' + err.message);
  }
}

listEl.addEventListener('click', e => {
  const card = e.target.closest('.card');
  if (!card) return;
  if (e.target.closest('[data-delete]')) { removeCard(card); return; }
  if (e.target.closest('.card-head')) toggle(card);
});

load();
</script>
</body>
</html>
"""

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    print(f"\n  Ender3Monitor  →  http://localhost:{port}\n")
    # timeout_graceful_shutdown caps how long uvicorn waits for open connections
    # (e.g. the never-ending /stream socket) before force-closing them, so a
    # single Ctrl+C exits cleanly instead of hanging until a second Ctrl+C.
    uvicorn.run("web:app", host="0.0.0.0", port=port, reload=False,
                timeout_graceful_shutdown=5)
