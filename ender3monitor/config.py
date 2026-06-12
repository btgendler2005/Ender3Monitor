import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Project root is one level up from this package directory, so .env loads
# correctly no matter which directory the app is launched from.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _parse_flip(value: str) -> Optional[int]:
    """Convert CAMERA_FLIP env string to cv2.flip() code.

    Values:
      none / off / 0  → no flip
      180             → rotate 180° (upside-down mount)  → cv2 code -1
      vertical / v    → flip top-bottom                  → cv2 code  0
      horizontal / h  → flip left-right                  → cv2 code  1
    """
    v = value.strip().lower()
    if v in ("none", "off", "0", ""):
        return None
    if v in ("180", "-1", "rotate180"):
        return -1
    if v in ("vertical", "v"):
        return 0
    if v in ("horizontal", "h", "1"):
        return 1
    return None


@dataclass
class Config:
    # AI backend
    analyzer_backend: str       # "anthropic" or "ollama"
    anthropic_api_key: str
    anthropic_model: str
    ollama_model: str
    ollama_host: str

    # Email alerts
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_recipient: str
    smtp_sender: str

    # Camera / capture
    camera_index: int           # -1 means prompt user to select
    camera_flip: int            # cv2.flip code: -1=180°, 0=vertical, 1=horizontal, None=off

    # Metrics / output
    metrics_port: int
    timelapse_dir: str
    confidence_threshold: float
    capture_interval: int        # seconds between AI analysis frames

    # First-layer inspection (more frequent, focused analysis early)
    first_layer_interval: int    # seconds between analyses while on the first layer
    first_layer_max_z: float     # Z height (mm) at/under which we treat it as first layer

    # Auto-start monitoring when the printer begins a print (USB)
    auto_start_on_print: bool

    # Timelapse
    timelapse_mode: str          # "auto" | "layer" | "time"
    timelapse_max_sessions: int
    timelapse_retention_days: int
    timelapse_delete_frames_after_compile: bool

    # Maintenance / health
    maintenance_reminder_hours: int

    # Printer USB control (optional)
    printer_port: str           # serial device path; "" = disabled, "auto" = autodetect
    printer_baud: int
    auto_pause_on_failure: bool
    auto_pause_action: str       # "pause" | "cooldown" | "estop"

    # Push notifications (optional)
    ntfy_topic: str              # ntfy.sh topic (or full URL); "" = disabled
    discord_webhook: str         # Discord webhook URL; "" = disabled
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_allowed_chats: str  # comma-separated chat IDs allowed to send commands

    @classmethod
    def from_env(cls) -> "Config":
        # Load the project's .env explicitly so it works from any cwd.
        load_dotenv(_ENV_PATH if _ENV_PATH.exists() else None)
        backend = os.getenv("ANALYZER_BACKEND", "anthropic").lower()

        # Only require the API key when using the Anthropic backend
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if backend == "anthropic" and not api_key:
            raise KeyError("ANTHROPIC_API_KEY")

        return cls(
            analyzer_backend=backend,
            anthropic_api_key=api_key,
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llava:7b"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=os.getenv("SMTP_USERNAME", ""),
            smtp_password=os.getenv("SMTP_PASSWORD", ""),
            smtp_recipient=os.getenv("SMTP_RECIPIENT", ""),
            smtp_sender=os.getenv("SMTP_SENDER", os.getenv("SMTP_USERNAME", "")),
            camera_index=int(os.getenv("CAMERA_INDEX", "-1")),
            camera_flip=_parse_flip(os.getenv("CAMERA_FLIP", "none")),
            metrics_port=int(os.getenv("METRICS_PORT", "8000")),
            timelapse_dir=os.getenv("TIMELAPSE_DIR", "timelapse_frames"),
            timelapse_mode=os.getenv("TIMELAPSE_MODE", "auto").strip().lower(),
            timelapse_max_sessions=int(os.getenv("TIMELAPSE_MAX_SESSIONS", "20")),
            timelapse_retention_days=int(os.getenv("TIMELAPSE_RETENTION_DAYS", "30")),
            timelapse_delete_frames_after_compile=os.getenv(
                "TIMELAPSE_DELETE_FRAMES_AFTER_COMPILE", "false"
            ).strip().lower() in ("1", "true", "yes", "on"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.70")),
            capture_interval=max(10, int(os.getenv("CAPTURE_INTERVAL_SECONDS", "300"))),
            first_layer_interval=max(10, int(os.getenv("FIRST_LAYER_INTERVAL_SECONDS", "60"))),
            first_layer_max_z=float(os.getenv("FIRST_LAYER_MAX_Z_MM", "0.6")),
            auto_start_on_print=os.getenv("AUTO_START_ON_PRINT", "true").strip().lower()
                in ("1", "true", "yes", "on"),
            maintenance_reminder_hours=int(os.getenv("MAINTENANCE_REMINDER_HOURS", "250")),
            printer_port=os.getenv("PRINTER_PORT", "").strip(),
            printer_baud=int(os.getenv("PRINTER_BAUD", "115200")),
            auto_pause_on_failure=os.getenv("AUTO_PAUSE_ON_FAILURE", "false").strip().lower()
                in ("1", "true", "yes", "on"),
            auto_pause_action=os.getenv("AUTO_PAUSE_ACTION", "pause").strip().lower(),
            ntfy_topic=os.getenv("NTFY_TOPIC", "").strip(),
            discord_webhook=os.getenv("DISCORD_WEBHOOK", "").strip(),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            telegram_allowed_chats=os.getenv("TELEGRAM_ALLOWED_CHATS", "").strip(),
        )
