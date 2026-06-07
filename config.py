import os
from dataclasses import dataclass, field
from dotenv import load_dotenv


@dataclass
class Config:
    anthropic_api_key: str
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_recipient: str
    smtp_sender: str
    camera_index: int  # -1 means prompt user to select
    metrics_port: int
    timelapse_dir: str
    confidence_threshold: float

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        return cls(
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            smtp_host=os.getenv("SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=os.environ["SMTP_USERNAME"],
            smtp_password=os.environ["SMTP_PASSWORD"],
            smtp_recipient=os.environ["SMTP_RECIPIENT"],
            smtp_sender=os.getenv("SMTP_SENDER", os.environ.get("SMTP_USERNAME", "")),
            camera_index=int(os.getenv("CAMERA_INDEX", "-1")),
            metrics_port=int(os.getenv("METRICS_PORT", "8000")),
            timelapse_dir=os.getenv("TIMELAPSE_DIR", "timelapse_frames"),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.70")),
        )
