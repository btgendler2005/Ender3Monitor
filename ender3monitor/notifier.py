import smtplib
import io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

import cv2
import numpy as np

from ender3monitor.analyzer import AnalysisResult


class EmailNotifier:
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        sender: str,
        recipient: str,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender
        self.recipient = recipient

    @property
    def enabled(self) -> bool:
        """True only when SMTP is fully configured — callers skip silently otherwise."""
        return bool(self.username and self.password and self.recipient)

    def send_alert(self, result: AnalysisResult, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"[Ender3Monitor] 3D Print Failure Detected – {result.failure_type}"

        body = f"""3D Print Failure Alert

Time: {timestamp}
Failure Type: {result.failure_type}
Confidence: {result.confidence:.1%}
Details: {result.description}

Please check your printer immediately.
"""

        msg = MIMEMultipart()
        msg["From"] = self.sender
        msg["To"] = self.recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Attach the frame that triggered the alert
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_bytes = buf.tobytes()
        img_part = MIMEImage(img_bytes, name=f"failure_{timestamp.replace(' ', '_').replace(':', '')}.jpg")
        img_part.add_header("Content-Disposition", "attachment", filename=img_part.get_filename())
        msg.attach(img_part)

        # timeout: a hung SMTP connection must never stall the monitoring thread
        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.sender, self.recipient, msg.as_string())

    def send_completion(self, frame: np.ndarray, frames_analyzed: int) -> None:
        """Send a print-complete notification with the final frame attached."""
        if not self.enabled:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = "[Ender3Monitor] 3D Print Appears Complete"

        body = f"""3D Print Completion Notice

Time: {timestamp}
Frames Analyzed: {frames_analyzed}
Status: No change detected for 4 consecutive frames (≥ 2 minutes)

Your print appears to have finished. The monitor has been stopped automatically.
A snapshot from the final frame is attached.
"""

        msg = MIMEMultipart()
        msg["From"] = self.sender
        msg["To"] = self.recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_bytes = buf.tobytes()
        safe_ts = timestamp.replace(" ", "_").replace(":", "")
        img_part = MIMEImage(img_bytes, name=f"complete_{safe_ts}.jpg")
        img_part.add_header("Content-Disposition", "attachment", filename=img_part.get_filename())
        msg.attach(img_part)

        # timeout: a hung SMTP connection must never stall the monitoring thread
        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.sender, self.recipient, msg.as_string())
