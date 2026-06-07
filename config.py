import os
from dataclasses import dataclass
from dotenv import load_dotenv


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

    # Metrics / output
    metrics_port: int
    timelapse_dir: str
    confidence_threshold: float

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
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
            metrics_port=int(os.getenv("METRICS_PORT", "8000")),
            timelapse_dir=os.getenv("TIMELAPSE_DIR", "timelapse_frames"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.70")),
        )
