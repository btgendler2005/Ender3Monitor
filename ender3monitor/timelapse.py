import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path


class TimelapseManager:
    def __init__(self, output_dir: str = "timelapse_frames") -> None:
        self.output_dir = Path(output_dir)
        self._session_dir: Path | None = None
        self._frame_count = 0

    def _ensure_session_dir(self) -> Path:
        if self._session_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._session_dir = self.output_dir / ts
            self._session_dir.mkdir(parents=True, exist_ok=True)
        return self._session_dir

    def save_frame(self, frame: np.ndarray) -> None:
        session = self._ensure_session_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = session / f"frame_{self._frame_count:06d}_{ts}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self._frame_count += 1

    def reset_session(self) -> None:
        self._session_dir = None
        self._frame_count = 0

    def compile(self, fps: int = 24, output_file: str | None = None) -> str | None:
        if self._session_dir is None:
            print("No timelapse session to compile.")
            return None

        frames = sorted(self._session_dir.glob("frame_*.jpg"))
        if not frames:
            print("No frames found to compile.")
            return None

        sample = cv2.imread(str(frames[0]))
        if sample is None:
            print("Cannot read frames.")
            return None

        h, w = sample.shape[:2]
        if output_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"timelapse_{ts}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

        for f in frames:
            img = cv2.imread(str(f))
            if img is not None:
                writer.write(img)

        writer.release()
        print(f"Timelapse compiled: {output_file} ({len(frames)} frames @ {fps} fps)")
        return output_file
