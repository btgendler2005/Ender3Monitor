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
import traceback
from contextlib import asynccontextmanager
from typing import Optional, Set

import cv2
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ender3monitor")

from ender3monitor.camera import CameraManager
from ender3monitor.config import Config
from monitor import Monitor

# ── global state ──────────────────────────────────────────────────────────────

_config: Optional[Config] = None
_monitor: Optional[Monitor] = None
_clients: Set[WebSocket] = set()
DEFAULT_PORT = 8080

# Shared MJPEG frame — written by the capture loop, read by stream clients
_live_frame: Optional[bytes] = None
_STREAM_FPS = 1          # captures per second (1 fps is plenty for a printer)
_STREAM_QUALITY = 70     # JPEG quality for the live stream

# When set, the stream capture loop pauses so a camera scan can have
# exclusive access to the hardware (probing indices conflicts with the
# continuous single-camera stream on most USB webcam drivers).
_stream_paused = False


def _capture_frame_sync() -> Optional[bytes]:
    """Grab one frame from the camera and return it as JPEG bytes (runs in thread)."""
    idx = 0
    if _config and _config.camera_index >= 0:
        idx = _config.camera_index
    elif _monitor and _monitor.camera:
        idx = _monitor.camera.camera_index
    frame = CameraManager(idx, flip=_config.camera_flip if _config else None).snapshot()
    if frame is None:
        return None
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _STREAM_QUALITY])
    return buf.tobytes()


async def _stream_capture_loop() -> None:
    """Background task: refresh the shared live frame at _STREAM_FPS."""
    global _live_frame
    loop = asyncio.get_running_loop()
    while True:
        if _stream_paused:
            await asyncio.sleep(0.2)
            continue
        try:
            data = await loop.run_in_executor(None, _capture_frame_sync)
            if data:
                _live_frame = data
        except Exception:
            pass
        await asyncio.sleep(1.0 / _STREAM_FPS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config, _monitor
    _config = Config.from_env()
    _monitor = Monitor(_config)
    _monitor.metrics.start_server(_config.metrics_port)
    push_task    = asyncio.create_task(_push_loop())
    capture_task = asyncio.create_task(_stream_capture_loop())
    yield
    push_task.cancel()
    capture_task.cancel()
    if _monitor and _monitor._running:
        _monitor.stop()


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
    global _stream_paused
    try:
        configured = _config.camera_index if _config else -1

        if configured >= 0 and not scan:
            # Fast path — camera already chosen, no need to probe hardware
            return {
                "cameras": [{"index": configured, "width": 1280, "height": 720}],
                "configured": configured,
            }

        # Pause the stream loop so the scan has exclusive access to the
        # camera hardware, then give any in-flight capture time to release.
        _stream_paused = True
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
            _stream_paused = False
        return {
            "cameras": [{"index": i, "width": w, "height": h} for i, w, h in cameras],
            "configured": configured,
        }
    except Exception:
        _stream_paused = False
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
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: _monitor.start(camera_index=cam))
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
    await loop.run_in_executor(None, _monitor.compile_timelapse)
    return {"message": "timelapse compiled"}


@app.get("/api/status")
async def api_status():
    return _state()


@app.get("/snapshot")
async def snapshot():
    """On-demand JPEG snapshot (single frame)."""
    data = _live_frame
    if data is None:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(None, _capture_frame_sync)
    if data is None:
        return Response(status_code=204)
    return Response(data, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


async def _mjpeg_generator():
    """Yield MJPEG frames from the shared live frame buffer."""
    while True:
        frame = _live_frame
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )
        await asyncio.sleep(1.0 / _STREAM_FPS)


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
.cam-meta{font-size:11.5px;color:var(--muted);display:flex;justify-content:space-between}

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
  grid-template-columns:68px 90px 1fr 46px;
  gap:12px;align-items:center;
  padding:10px 18px;
  border-bottom:1px solid var(--border);
  font-size:12px;
}
.ev:last-child{border-bottom:none}
.ev-time{color:var(--muted);font-family:monospace;font-size:11.5px}
.badge{
  padding:2px 9px;border-radius:5px;
  font-size:10.5px;font-weight:700;
  text-transform:uppercase;letter-spacing:.05em;
  width:fit-content;
}
.badge-ok  {background:rgba(34,197,94,.12);color:var(--green)}
.badge-fail{background:rgba(248,113,113,.12);color:var(--red)}
.badge-done{background:rgba(79,142,247,.12);color:var(--blue)}
.ev-desc{color:var(--text);overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.ev-conf{color:var(--muted);font-family:monospace;text-align:right}

/* ── Toast ── */
#toast{
  position:fixed;bottom:28px;left:50%;transform:translateX(-50%) translateY(80px);
  background:var(--surf2);border:1px solid var(--border);
  padding:10px 20px;border-radius:10px;font-size:13px;
  transition:transform .25s;pointer-events:none;z-index:100;
  box-shadow:var(--shadow);
}
#toast.show{transform:translateX(-50%) translateY(0)}
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
    <select id="cam-select" title="Select camera">
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
  <button onclick="doTimelapse()">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16" fill="currentColor"/>
    </svg>
    Compile Timelapse
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
async function doTimelapse() { await api('timelapse'); toast('Timelapse compiling…') }

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
  const desc = (d.description||'').slice(0, 90);
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
