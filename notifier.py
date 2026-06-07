import smtplib
import io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

import cv2
import numpy as np

from analyzer import AnalysisResult


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

    def send_alert(self, result: AnalysisResult, frame: np.ndarray) -> None:
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

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.sender, self.recipient, msg.as_string())
