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
from ender3monitor.camera_worker import capture_worker
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
        shm = SharedMemory(create=True, size=self.height * self.width * 3)
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
        self._frame_ready, self._stop_worker = frame_ready, stop_worker

    def _teardown_worker(self) -> None:
        if self._stop_worker is not None:
            self._stop_worker.set()
        if self._process is not None:
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except FileNotFoundError:
                pass
        self._process = self._shm = None
        self._frame_ready = self._stop_worker = None

    def _loop(self) -> None:
        last_encode = 0.0
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

            frame = np.ndarray((self.height, self.width, 3), dtype=np.uint8,
                                buffer=self._shm.buf).copy()
            ops.camera_frames_total.labels("ok").inc()
            if self.flip is not None:
                frame = cv2.flip(frame, self.flip)
            with self._lock:
                self._latest_raw = frame
            now = time.time()
            active = self._has_demand()
            encode_interval = (self._encode_interval if active
                               else self._IDLE_ENCODE_INTERVAL)
            if now - last_encode >= encode_interval:
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ok2:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
                last_encode = now
            if not active:
                # Idle: don't spin the camera at full rate for nobody.
                time.sleep(self._IDLE_READ_SLEEP)
        self._teardown_worker()

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
        with self._lock:
            return self._latest_jpeg

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
    if pr.filament_change_pause:
        lines.append("🎨 Paused — change filament (M600)")
    if pr.printing and pr.progress is not None:
        d = pr.as_dict()
        extra = []
        if d.get("elapsed_str"):   extra.append(d['elapsed_str'] + " elapsed")
        if d.get("remaining_str"): extra.append("~" + d['remaining_str'] + " left")
        lines.append(f"Progress: {pr.progress*100:.1f}%   " + "   ".join(extra))
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
        _telegram = TelegramBot(_config.telegram_bot_token, allowed, _build_telegram_handlers())
        _telegram.start()
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

class _AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not _auth_enabled():
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

async def _push_loop() -> None:
    """Push state to all connected browsers every 2 s."""
    while True:
        await asyncio.sleep(2)
        if not _clients:
            continue
        msg = json.dumps(_state())
        dead: Set[WebSocket] = set()
        for ws in list(_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        _clients.difference_update(dead)


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


@app.get("/api/cameras")
async def api_cameras(scan: bool = False):
    """Return available cameras.

    If CAMERA_INDEX is set (>= 0), return it immediately without opening
    any camera — fast path, no contention with the stream loop.

    Pass ?scan=true to force a full hardware scan regardless.
    """
    try:
        configured = _config.camera_index if _config else -1

        if configured >= 0 and not scan:
            # Fast path — camera already chosen, no need to probe hardware
            return {
                "cameras": [{"index": configured, "width": 1280, "height": 720}],
                "configured": configured,
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
        return {
            "cameras": [{"index": i, "width": w, "height": h} for i, w, h in cameras],
            "configured": configured,
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
    msg = json.dumps(_state())
    dead: Set[WebSocket] = set()
    for ws in list(_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


@app.post("/api/start")
async def api_start(body: StartBody = StartBody()):
    if _monitor is None:
        return JSONResponse({"error": "not initialised"}, 503)
    cam = body.camera_index
    if cam is None and _config and _config.camera_index >= 0:
        cam = _config.camera_index

    # Point the persistent capture at the chosen camera (if specified), then
    # hand the monitor that same capture as its frame source — no second handle.
    if _stream is not None and cam is not None and cam != _stream.index:
        _stream.set_index(cam)
        await asyncio.sleep(0.5)   # let the new camera produce a first frame

    provider = _stream.latest_frame if _stream is not None else None
    cam_for_monitor = cam if cam is not None else (_stream.index if _stream else None)

    loop = asyncio.get_running_loop()
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


@app.get("/api/status")
async def api_status():
    return _state()


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
    _stream.set_index(body.camera_index)
    return {"camera_index": body.camera_index}


@app.get("/snapshot")
async def snapshot():
    """On-demand JPEG snapshot (single frame from the live buffer)."""
    data = _stream.latest_jpeg() if _stream else None
    if data is None:
        return Response(status_code=204)
    return Response(data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


async def _mjpeg_generator():
    """Yield MJPEG frames from the shared live buffer at _STREAM_FPS.

    Registers as a viewer so the capture loop runs at full rate while at least
    one stream is open. Skips unchanged frames (identity check — the encoder
    replaces the buffer object on every encode) so we never resend the same
    JPEG, which matters on phone connections.
    """
    if _stream:
        _stream.add_viewer()
    try:
        interval = 1.0 / _STREAM_FPS
        last_sent = None
        while not _shutting_down.is_set():
            frame = _stream.latest_jpeg() if _stream else None
            if frame is not None and frame is not last_sent:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
                last_sent = frame
            await asyncio.sleep(interval)
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


# ── HTML / CSS / JS  (single-file, no build step) ────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_HTML)


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
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
.cam-card{grid-row:1/3;display:flex;flex-direction:column;gap:14px}
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
  background:var(--surf3);border-radius:9px;
  padding:14px 16px;
}
.count-box .lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:4px}
.count-box .num{font-size:26px;font-weight:800;line-height:1}

/* ── Printer controls ── */
.printer-controls{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}
.pbtn{
  padding:9px 10px;border-radius:8px;
  border:1px solid var(--border);background:var(--surf3);color:var(--text);
  font-size:12.5px;font-weight:600;cursor:pointer;transition:all .15s;
  justify-content:center;
}
.pbtn:hover{border-color:rgba(255,255,255,.16);background:#1d2336}
.pbtn:active{transform:scale(.97)}
.pbtn-stop{background:rgba(248,113,113,.1);border-color:rgba(248,113,113,.3);color:var(--red)}
.pbtn-stop:hover{background:rgba(248,113,113,.2)}

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
.autostart-toggle{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--muted);
  cursor:pointer;padding:0 6px;user-select:none}
.autostart-toggle input{width:15px;height:15px;accent-color:var(--blue);cursor:pointer}

/* ── Controls ── */
.controls{
  display:flex;gap:10px;flex-wrap:wrap;
  padding:0 28px 4px;
  max-width:1180px;width:100%;margin:0 auto;
}
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

/* ── Mobile / narrow screens ── */
@media (max-width:760px){
  .modal-overlay{padding:0}
  .modal{max-width:none;border-radius:0;min-height:100vh;max-height:100vh}
  .set-control input[type=number],.set-control select{width:108px}
  header{padding:12px 16px}
  .logo{font-size:15px}
  .pill{padding:5px 10px;font-size:10px}

  /* Stack the two-column layout into one; camera no longer spans rows */
  .page{
    grid-template-columns:1fr;
    grid-template-rows:none;
    gap:14px;
    padding:14px 14px;
  }
  .cam-card{grid-row:auto}
  .card{padding:16px}

  /* Controls: full-width camera picker, big tappable buttons */
  .controls{padding:0 14px 4px;gap:8px}
  .cam-select-row{flex:1 0 100%;min-width:0}
  #cam-select{min-width:0}
  .controls > button{flex:1 1 40%;justify-content:center;padding:12px 14px;font-size:14px}
  .btn-scan{flex:0 0 auto}

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

<div class="page">

  <!-- Camera -->
  <div class="card cam-card">
    <div class="card-label">Camera Feed</div>
    <img id="cam" src="/stream" alt="Camera preview">
    <div class="cam-meta">
      <span id="cam-label">—</span>
      <span id="stream-link" style="font-size:11px;color:var(--muted)">
        📱 <a id="stream-url" href="/stream" target="_blank"
             style="color:var(--muted);text-decoration:none">/stream</a>
      </span>
    </div>
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
    </div>

    <div class="type-row">
      <div class="card-label">Failure Type</div>
      <div class="type-val type-ok" id="type-val">none</div>
    </div>

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

    <div class="count-box" id="cost-box" style="margin-top:10px">
      <div class="lbl">Est. API cost · this print</div>
      <div class="num" id="cost" style="color:var(--green)">$0.00</div>
    </div>

    <!-- Printer (USB) — hidden until a printer is connected -->
    <div id="printer-block" style="display:none">
      <div class="card-label" style="margin-top:18px">Printer</div>
      <div class="counts">
        <div class="count-box">
          <div class="lbl">Nozzle</div>
          <div class="num" id="temp-nozzle" style="font-size:20px">—</div>
        </div>
        <div class="count-box">
          <div class="lbl">Bed</div>
          <div class="num" id="temp-bed" style="font-size:20px">—</div>
        </div>
      </div>
      <div style="font-size:11px;color:var(--muted);margin-top:8px" id="printer-progress"></div>
      <div class="printer-controls">
        <button class="pbtn" onclick="doPrinter('pause')">Pause</button>
        <button class="pbtn" onclick="doPrinter('resume')">Resume</button>
        <button class="pbtn" onclick="doPrinter('cooldown')">Cooldown</button>
        <button class="pbtn pbtn-stop" onclick="doPrinter('estop')">E‑Stop</button>
      </div>
    </div>

    <div style="font-size:11px;color:var(--muted);border-top:1px solid var(--border);
                padding-top:14px" id="backend-row"></div>
  </div>

  <!-- Description -->
  <div class="card desc-card">
    <div class="card-label">AI Observation</div>
    <p id="description" style="color:var(--muted);font-style:italic">
      Waiting for first analysis…
    </p>
  </div>

</div>

<!-- Controls -->
<div class="controls">
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
  <button class="btn-start" id="btn-start" onclick="doStart()">
    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg>
    Start
  </button>
  <button class="btn-stop" id="btn-stop" onclick="doStop()" disabled>
    <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>
    Stop
  </button>
  <button onclick="doTimelapse()" id="btn-timelapse">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M12 3v12m0 0l-4-4m4 4l4-4"/><path d="M5 21h14"/>
    </svg>
    Download Timelapse
  </button>
  <label class="autostart-toggle" title="Automatically start monitoring when the printer begins a print">
    <input type="checkbox" id="autostart-cb" onchange="toggleAutoStart()">
    <span>Auto-start</span>
  </label>
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
let ws, lastFrames = -1, camList = [];

// ── WebSocket ──────────────────────────────────────────────────────────────
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = e => render(JSON.parse(e.data));
  ws.onclose   = () => setTimeout(connect, 2500);
}

// ── Render state ───────────────────────────────────────────────────────────
function render(d) {
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
    if (p.filament_change_pause) {
      prog.textContent = '🎨 Paused — change filament (M600)';
      prog.style.color = 'var(--amber)';
    } else if (p.printing && p.progress != null) {
      const parts = [(p.progress * 100).toFixed(1) + '%'];
      if (p.elapsed_str)   parts.push(p.elapsed_str + ' elapsed');
      if (p.remaining_str) parts.push('~' + p.remaining_str + ' left');
      prog.textContent = parts.join('  ·  ');
      prog.style.color = 'var(--muted)';
    } else if (p.printing) {
      prog.textContent = 'Printing…';
      prog.style.color = 'var(--muted)';
    } else {
      prog.textContent = p.port ? 'Idle · ' + p.port : 'Idle';
      prog.style.color = 'var(--muted)';
    }
  } else {
    pb.style.display = 'none';
  }

  // Camera label
  if (d.camera_index != null) {
    document.getElementById('cam-label').textContent = `Camera ${d.camera_index}`;
  }

  // Buttons
  document.getElementById('btn-start').disabled = !!d.is_running;
  document.getElementById('btn-stop').disabled  = !d.is_running;

  // Event log — rendered from the server-side history so it survives reloads.
  if (Array.isArray(d.events)) renderEvents(d.events);
}

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
  const data = await api('start', body);
  if (data) render(data);
}
async function doStop() {
  const data = await api('stop');
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
async function doPrinter(action) {
  if (action === 'estop' &&
      !confirm('Emergency stop halts the printer immediately and requires a power-cycle to recover. Continue?')) {
    return;
  }
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
}

// Returns parsed JSON on success, null on error
async function api(action, body = null) {
  try {
    const res = await fetch(`/api/${action}`, {
      method: 'POST',
      headers: body ? {'Content-Type': 'application/json'} : {},
      body:    body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) { toast('Error: ' + res.statusText); return null; }
    return await res.json();
  } catch(e) { toast('Network error'); return null; }
}

// ── Event log (server-driven; persists across reloads) ──────────────────────
function esc(s) {
  return String(s).replace(/[&<>"]/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function renderEvents(events) {
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

// ── Toast ──────────────────────────────────────────────────────────────────
function toast(msg, ms=3000) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), ms);
}

// ── Settings panel ──────────────────────────────────────────────────────────
let settingsSchema = null, settingsValues = {};

function esc(s){ return String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

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

// Esc closes the modal
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSettings(); });

// ── Boot ───────────────────────────────────────────────────────────────────
scanCameras();
connect();
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
