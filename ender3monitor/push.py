"""Instant push notifications to your phone.

Supports three zero-/low-setup channels, any combination enabled via .env:

  • ntfy      — easiest: pick a topic, install the ntfy app, subscribe. No account.
                NTFY_TOPIC=ender3-myprinter   (or a full self-hosted URL)
  • Discord   — paste a channel webhook URL.  DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
  • Telegram  — create a bot via @BotFather.   TELEGRAM_BOT_TOKEN=...  TELEGRAM_CHAT_ID=...

Uses only the standard library (urllib), so there is no extra dependency.
Every send is best-effort and never raises — a notification failure must not
disrupt monitoring.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
import uuid
from typing import List, Optional


class PushNotifier:
    def __init__(self, ntfy_topic: str = "", discord_webhook: str = "",
                 telegram_bot_token: str = "", telegram_chat_id: str = "") -> None:
        self.ntfy_topic = (ntfy_topic or "").strip()
        self.discord_webhook = (discord_webhook or "").strip()
        self.telegram_bot_token = (telegram_bot_token or "").strip()
        self.telegram_chat_id = (telegram_chat_id or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.ntfy_topic or self.discord_webhook
                    or (self.telegram_bot_token and self.telegram_chat_id))

    def channels(self) -> List[str]:
        out = []
        if self.ntfy_topic:
            out.append("ntfy")
        if self.discord_webhook:
            out.append("discord")
        if self.telegram_bot_token and self.telegram_chat_id:
            out.append("telegram")
        return out

    # ------------------------------------------------------------------ #

    def send(self, title: str, message: str, priority: str = "default") -> None:
        """Fire the notification to every configured channel. Never raises."""
        if self.ntfy_topic:
            self._safe(self._send_ntfy, title, message, priority)
        if self.discord_webhook:
            self._safe(self._send_discord, title, message, priority)
        if self.telegram_bot_token and self.telegram_chat_id:
            self._safe(self._send_telegram, title, message, priority)

    @staticmethod
    def _safe(fn, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:
            print(f"  [PUSH] {fn.__name__} failed: {exc}")

    @staticmethod
    def _post(url: str, data: bytes, headers: Optional[dict] = None) -> None:
        req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST")
        with urllib.request.urlopen(req, timeout=8) as resp:
            resp.read()

    # ── channel implementations ──

    def _send_ntfy(self, title: str, message: str, priority: str) -> None:
        # Topic may be a bare name (ntfy.sh) or a full URL to a self-hosted server.
        url = self.ntfy_topic if self.ntfy_topic.startswith("http") \
            else f"https://ntfy.sh/{self.ntfy_topic}"
        prio = {"high": "urgent", "default": "default", "low": "low"}.get(priority, "default")
        headers = {
            "Title": title.encode("ascii", "ignore").decode(),
            "Priority": prio,
            "Tags": "warning" if priority == "high" else "printer",
        }
        self._post(url, message.encode("utf-8"), headers)

    def _send_discord(self, title: str, message: str, priority: str) -> None:
        color = 0xF87171 if priority == "high" else 0x4F8EF7
        payload = {"embeds": [{"title": title, "description": message, "color": color}]}
        self._post(self.discord_webhook, json.dumps(payload).encode("utf-8"),
                   {"Content-Type": "application/json"})

    def _send_telegram(self, title: str, message: str, priority: str) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        text = f"*{title}*\n{message}"
        data = urllib.parse.urlencode({
            "chat_id": self.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }).encode("utf-8")
        self._post(url, data, {"Content-Type": "application/x-www-form-urlencoded"})

    # ── media (Telegram only) — used by the completion report ──

    @property
    def _telegram_ready(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    def _telegram_multipart(self, method: str, fields: dict, file_field: str,
                            filename: str, content_type: str, blob: bytes) -> None:
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/{method}"
        boundary = "----E3M" + uuid.uuid4().hex
        body = b""
        for k, v in fields.items():
            body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"'
                     f'\r\n\r\n{v}\r\n').encode("utf-8")
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
                 f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n').encode("utf-8")
        body += blob + b"\r\n" + f"--{boundary}--\r\n".encode("utf-8")
        self._post(url, body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})

    def send_photo(self, jpeg: bytes, caption: str = "") -> None:
        """Send a photo to the Telegram channel (best-effort)."""
        if not self._telegram_ready:
            return
        self._safe(self._telegram_multipart, "sendPhoto",
                   {"chat_id": self.telegram_chat_id, "caption": caption[:1024]},
                   "photo", "snapshot.jpg", "image/jpeg", jpeg)

    def send_video(self, mp4_path: str, caption: str = "") -> None:
        """Send a video file to the Telegram channel (best-effort, ≤ ~50 MB)."""
        if not self._telegram_ready:
            return
        try:
            with open(mp4_path, "rb") as f:
                blob = f.read()
        except Exception:
            return
        if len(blob) > 49 * 1024 * 1024:
            return   # over Telegram's bot upload limit; caller falls back to a note
        self._safe(self._telegram_multipart, "sendVideo",
                   {"chat_id": self.telegram_chat_id, "caption": caption[:1024]},
                   "video", "timelapse.mp4", "video/mp4", blob)
