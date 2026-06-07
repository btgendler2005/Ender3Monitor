import re
import subprocess
import sys
import cv2
import numpy as np
from typing import Optional


def _camera_names_macos() -> list[str]:
    """Return camera names in system_profiler order (macOS only).

    system_profiler lists cameras in the same order AVFoundation (and OpenCV)
    enumerates them. Camera names are the lines indented with exactly 4 spaces
    that end with ':' — one level below the top-level 'Camera:' section header.
    """
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        # Regex: lines with exactly 4-space indent, no colon in the name, ending ':'
        return re.findall(r"^    ([^:\n]+):$", out, re.MULTILINE)
    except Exception:
        return []


class CameraManager:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None

    @staticmethod
    def list_available_cameras(max_check: int = 5) -> list[tuple[int, int, int]]:
        """Return [(index, width, height), ...] for every readable camera."""
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    available.append((i, w, h))
            cap.release()
        return available

    @staticmethod
    def select_camera() -> int:
        cameras = CameraManager.list_available_cameras()
        if not cameras:
            raise RuntimeError("No cameras detected.")
        if len(cameras) == 1:
            idx, w, h = cameras[0]
            print(f"  One camera detected (index {idx}, {w}×{h}). Using it.")
            return idx

        # Best-effort device names from system_profiler on macOS
        names: list[str] = _camera_names_macos() if sys.platform == "darwin" else []

        print("\nAvailable cameras:")
        for idx, w, h in cameras:
            name = names[idx] if idx < len(names) else f"Camera {idx}"
            print(f"  [{idx}] {name}  ({w}×{h})")

        indices = [idx for idx, _, _ in cameras]
        while True:
            try:
                choice = int(input(f"  Select camera index [{indices[0]}]: ").strip() or indices[0])
                if choice in indices:
                    return choice
                print(f"  Invalid choice. Pick from {indices}.")
            except ValueError:
                print("  Please enter a number.")

    def open(self, width: int = 1280, height: int = 720) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}.")
        # Cap resolution — no need to decode 4K for a failure-detection thumbnail.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Keep only the latest frame in the buffer so we never analyse a stale image.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def capture_frame(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
