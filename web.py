#!/usr/bin/env python3
"""Ender3Monitor – Web UI.

    python web.py            # start on default port 8080
    python web.py 9090       # custom port

Then open http://localhost:8080 in your browser.
The CLI (python monitor.py) continues to work independently.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Set

import cv2
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ender3monitor")

from ender3monitor.camera import CameraManager
from ender3monitor.config import Config
from ender3monitor.telegram_bot import TelegramBot
from monitor import Monitor

# ── global state ──────────────────────────────────────────────────────────────

_config: Optional[Config] = None
_monitor: Optional[Monitor] = None
_clients: Set[WebSocket] = set()
_telegram: Optional[TelegramBot] = None
DEFAULT_PORT = 8080

# ── Live-view tuning ────────────────────────────────────────────────────────
_STREAM_FPS = 12          # live-view frames per second served to the browser
_STREAM_QUALITY = 70      # JPEG quality for the live stream
_STREAM_WIDTH = 1280
_STREAM_HEIGHT = 720


class StreamCapture:
    """Single persistent camera owner.

    One background thread keeps the camera open and continuously reads frames
    into a shared buffer. The MJPEG stream serves this buffer at _STREAM_FPS for
    a smooth live view, while the analysis loop and timelapse sample the latest
    raw frame on their own (much slower) schedule. Because there is exactly one
    handle on the camera, there is no contention — the prior design opened the
    camera separately for the stream and for each analysis frame, which raced on
    macOS USB drivers.
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
        self._paused = False          # when True, release the camera (for scans)
        self._thread: Optional[threading.Thread] = None

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

    def _open(self):
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self) -> None:
        cap = None
        last_encode = 0.0
        while self._running:
            # Paused (or reindexing): drop the camera handle so a scan can use it.
            if self._paused:
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(0.1)
                continue
            if cap is None:
                cap = self._open()
                if cap is None:
                    time.sleep(1.0)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                time.sleep(0.3)
                continue
            if self.flip is not None:
                frame = cv2.flip(frame, self.flip)
            with self._lock:
                self._latest_raw = frame
            now = time.time()
            if now - last_encode >= self._encode_interval:
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if ok2:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
                last_encode = now
        if cap is not None:
            cap.release()

    # ── controls ──
    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def set_index(self, index: int) -> None:
        """Switch to a different camera. Briefly pauses so the old handle is
        released before the new one opens."""
        if index == self.index:
            return
        self.index = index
        with self._lock:
            self._latest_raw = None
            self._latest_jpeg = None
        self._paused = True
        time.sleep(0.3)
        self._paused = False

    # ── readers ──
    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def latest_frame(self):
        """Return a copy of the latest raw frame (for analysis/timelapse)."""
        with self._lock:
            return None if self._latest_raw is None else self._latest_raw.copy()


# The single shared capture instance (created in lifespan once config is loaded)
_stream: Optional[StreamCapture] = None


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
    out.append(f"Lifetime: {_monitor.maintenance.total_hours:.1f} h printed")
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
    return _monitor.maintenance.summary()


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
    return handlers


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
    _config = Config.from_env()
    _monitor = Monitor(_config)
    _monitor.metrics.start_server(_config.metrics_port)
    _stream = StreamCapture(_resolve_stream_index(),
                            flip=_config.camera_flip)
    _stream.start()

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
    if _telegram:
        _telegram.stop()
    if _monitor and _monitor._running:
        _monitor.stop()
    if _monitor:
        _monitor.close()       # disconnect printer / stop temp poller
    if _stream:
        _stream.stop()


app = FastAPI(title="Ender3Monitor", lifespan=lifespan)


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
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _clients.add(ws)
    await ws.send_text(json.dumps(_state()))
    try:
        while True:
            await ws.receive_text()   # keep-alive drain
    except WebSocketDisconnect:
        _clients.discard(ws)


# ── REST API ──────────────────────────────────────────────────────────────────

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
    """Yield MJPEG frames from the shared live buffer at _STREAM_FPS."""
    interval = 1.0 / _STREAM_FPS
    while True:
        frame = _stream.latest_jpeg() if _stream else None
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )
        await asyncio.sleep(interval)


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

/* ── Mobile / narrow screens ── */
@media (max-width:760px){
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
  <div class="pill pill-idle" id="pill">
    <div class="dot"></div>
    <span id="pill-text">Idle</span>
  </div>
</header>

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
        <div style="font-size:11px;color:var(--muted)">threshold 70%</div>
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

  // Confidence
  const pct = (d.confidence || 0) * 100;
  document.getElementById('conf-num').textContent =
    d.frame_count > 0 ? pct.toFixed(1) + '%' : '—';
  const fill = document.getElementById('conf-fill');
  fill.style.width = pct + '%';
  fill.style.background = pct >= 70 ? 'var(--red)' : pct >= 40 ? 'var(--amber)' : 'var(--green)';

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
    if (p.printing && p.progress != null) {
      const parts = [(p.progress * 100).toFixed(1) + '%'];
      if (p.elapsed_str)   parts.push(p.elapsed_str + ' elapsed');
      if (p.remaining_str) parts.push('~' + p.remaining_str + ' left');
      prog.textContent = parts.join('  ·  ');
    } else if (p.printing) {
      prog.textContent = 'Printing…';
    } else {
      prog.textContent = p.port ? 'Idle · ' + p.port : 'Idle';
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

  // New frame → log entry
  if (d.frame_count > lastFrames && d.frame_count > 0) {
    lastFrames = d.frame_count;
    addLog(d);
  }
}

function pillClass(s) {
  const l = s.toLowerCase();
  if (l.includes('failure') || l.includes('detected')) return 'pill-fail';
  if (l.includes('complete'))                            return 'pill-done';
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

// ── Event log ──────────────────────────────────────────────────────────────
function addLog(d) {
  const log = document.getElementById('log');
  log.querySelector('.log-empty')?.remove();

  const now  = new Date().toTimeString().slice(0,8);
  const isFail = d.failure_detected;
  const isDone = (d.status||'').toLowerCase().includes('complete');
  const cls  = isFail ? 'badge-fail' : isDone ? 'badge-done' : 'badge-ok';
  const lbl  = isFail ? (d.failure_type||'failure') : isDone ? 'complete' : 'ok';
  const desc = (d.description||'').slice(0, 220);
  const conf = d.confidence ? (d.confidence*100).toFixed(0)+'%' : '';

  const row = document.createElement('div');
  row.className = 'ev';
  row.innerHTML = `
    <span class="ev-time">${now}</span>
    <span class="badge ${cls}">${lbl}</span>
    <span class="ev-desc">${desc}</span>
    <span class="ev-conf">${conf}</span>
  `;
  log.prepend(row);
  while (log.children.length > 60) log.removeChild(log.lastChild);
}

function clearLog() {
  const log = document.getElementById('log');
  log.innerHTML = '<div class="log-empty">Log cleared.</div>';
  lastFrames = -1;
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
    uvicorn.run("web:app", host="0.0.0.0", port=port, reload=False)
