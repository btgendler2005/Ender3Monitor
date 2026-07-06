"""Interactive Telegram bot — two-way control over Telegram.

Long-polls the Telegram Bot API (getUpdates) on a background thread and
dispatches slash-commands to handlers supplied by the app: check status, grab a
camera snapshot, pause/resume/cooldown the printer, start/stop monitoring.

Security: only chat IDs in `allowed_chats` may run commands. Anyone else gets a
one-line reply with their chat ID so you can authorize them (set
TELEGRAM_ALLOWED_CHATS in .env). This means remote control needs NO exposed web
server — Telegram is the transport and the allowlist is the auth.

Stdlib only (urllib). Best-effort: network hiccups are swallowed and retried.

Handler protocol — each handler is (func, description) where func(args: list[str])
returns either:
  • a str                       → sent as a text reply
  • ("photo", jpeg_bytes, str)  → sent as a photo with a caption
  • ("video", mp4_bytes, str)   → sent as a video with a caption
  • a list of any of the above  → sent as a sequence of messages/media
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
import uuid
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

HandlerItem = Union[str, Tuple[str, bytes, str]]
HandlerResult = Union[HandlerItem, List[HandlerItem]]
Handler = Callable[[list], HandlerResult]

_API = "https://api.telegram.org/bot{token}/{method}"


class TelegramBot:
    def __init__(self, token: str, allowed_chats: Set[int],
                 handlers: Dict[str, Tuple[Handler, str]]) -> None:
        self.token = token
        self.allowed_chats = set(allowed_chats)
        self.handlers = handlers
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._offset = 0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if not self.token:
            return
        # Skip any backlog so we don't replay old commands on startup.
        try:
            updates = self._get_updates(timeout=0)
            if updates:
                self._offset = updates[-1]["update_id"] + 1
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------ #
    # Polling loop                                                         #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for upd in self._get_updates(timeout=30):
                    self._offset = upd["update_id"] + 1
                    self._handle_update(upd)
            except Exception:
                # Network blip / timeout — back off briefly and retry.
                self._stop.wait(timeout=3)

    def _handle_update(self, upd: dict) -> None:
        msg = upd.get("message") or upd.get("channel_post")
        if not msg:
            return
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return

        # /command@botname args…  →  command, [args]
        parts = text.split()
        cmd = parts[0][1:].split("@")[0].lower()
        args = parts[1:]

        if chat_id not in self.allowed_chats:
            self.send_message(
                chat_id,
                f"⛔ Not authorized.\nYour chat ID is `{chat_id}`.\n"
                "Add it to TELEGRAM_ALLOWED_CHATS in .env and restart to enable control.",
            )
            return

        handler = self.handlers.get(cmd)
        if handler is None:
            self.send_message(chat_id, f"Unknown command /{cmd}. Try /help.")
            return

        try:
            result = handler[0](args)
        except Exception as exc:
            self.send_message(chat_id, f"⚠️ /{cmd} failed: {exc}")
            return

        items = result if isinstance(result, list) else [result]
        for item in items:
            self._send_result_item(chat_id, item)

    def _send_result_item(self, chat_id: int, result) -> None:
        if isinstance(result, tuple) and result and result[0] == "photo":
            _, jpeg, caption = result
            if jpeg:
                self.send_photo(chat_id, jpeg, caption)
            else:
                self.send_message(chat_id, caption or "No image available.")
        elif isinstance(result, tuple) and result and result[0] == "video":
            _, mp4, caption = result
            if mp4:
                self.send_video(chat_id, mp4, caption)
            else:
                self.send_message(chat_id, caption or "No video available.")
        elif result:
            self.send_message(chat_id, str(result))

    # ------------------------------------------------------------------ #
    # Telegram API                                                         #
    # ------------------------------------------------------------------ #

    def _get_updates(self, timeout: int = 30) -> list:
        url = _API.format(token=self.token, method="getUpdates")
        data = urllib.parse.urlencode({"offset": self._offset, "timeout": timeout}).encode()
        # Read timeout must exceed the long-poll timeout.
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=timeout + 10) as resp:
            payload = json.load(resp)
        return payload.get("result", []) if payload.get("ok") else []

    def send_message(self, chat_id: int, text: str) -> None:
        url = _API.format(token=self.token, method="sendMessage")
        # Try Markdown first; if Telegram rejects the entities (free-form /ask
        # answers can contain stray markdown), resend as plain text.
        for params in ({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
                       {"chat_id": chat_id, "text": text}):
            try:
                data = urllib.parse.urlencode(params).encode()
                urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10).read()
                return
            except Exception:
                continue

    def send_photo(self, chat_id: int, jpeg: bytes, caption: str = "") -> None:
        try:
            url = _API.format(token=self.token, method="sendPhoto")
            boundary = "----E3M" + uuid.uuid4().hex
            crlf = b"\r\n"
            body = b""

            def field(name: str, value: str) -> bytes:
                return (f'--{boundary}\r\nContent-Disposition: form-data; '
                        f'name="{name}"\r\n\r\n{value}\r\n').encode()

            body += field("chat_id", str(chat_id))
            if caption:
                body += field("caption", caption)
            body += (f'--{boundary}\r\nContent-Disposition: form-data; '
                     f'name="photo"; filename="snapshot.jpg"\r\n'
                     f'Content-Type: image/jpeg\r\n\r\n').encode()
            body += jpeg + crlf
            body += f"--{boundary}--\r\n".encode()

            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            urllib.request.urlopen(req, timeout=20).read()
        except Exception:
            pass

    def send_video(self, chat_id: int, mp4: bytes, caption: str = "") -> None:
        try:
            url = _API.format(token=self.token, method="sendVideo")
            boundary = "----E3M" + uuid.uuid4().hex
            crlf = b"\r\n"
            body = b""

            def field(name: str, value: str) -> bytes:
                return (f'--{boundary}\r\nContent-Disposition: form-data; '
                        f'name="{name}"\r\n\r\n{value}\r\n').encode()

            body += field("chat_id", str(chat_id))
            if caption:
                body += field("caption", caption)
            body += (f'--{boundary}\r\nContent-Disposition: form-data; '
                     f'name="video"; filename="timelapse.mp4"\r\n'
                     f'Content-Type: video/mp4\r\n\r\n').encode()
            body += mp4 + crlf
            body += f"--{boundary}--\r\n".encode()

            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            # Large uploads over a slow uplink take a while; give it more room
            # than the photo/message calls.
            urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            pass
