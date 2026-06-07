import subprocess
import sys
import cv2
import numpy as np
from typing import Optional


def _camera_names_macos() -> dict[int, str]:
    """Return a best-effort {index: name} map using system_profiler on macOS."""
    names: dict[int, str] = {}
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPCameraDataType"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode()
        # Camera entries are indented with exactly 4 spaces and end with ":".
        # The top-level "Camera:" section header has no indent — skip it.
        # Sub-properties ("Model ID:", etc.) are indented 6+ spaces — skip them too.
        raw_names = [
            line.strip().rstrip(":")
            for line in out.splitlines()
            if line.startswith("    ")          # exactly 4-space indent
            and not line.startswith("     ")    # but not 5+ (sub-properties)
            and line.strip().endswith(":")
            and ":" not in line.strip()[:-1]    # no colon in the name itself
        ]
        for i, name in enumerate(raw_names):
            names[i] = name
    except Exception:
        pass
    return names


class CameraManager:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None

    @staticmethod
    def list_available_cameras(max_check: int = 5) -> list[int]:
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
            cap.release()
        return available

    @staticmethod
    def select_camera() -> int:
        cameras = CameraManager.list_available_cameras()
        if not cameras:
            raise RuntimeError("No cameras detected.")
        if len(cameras) == 1:
            print(f"One camera detected (index {cameras[0]}). Using it.")
            return cameras[0]

        # Try to get human-readable names on macOS
        names: dict[int, str] = {}
        if sys.platform == "darwin":
            names = _camera_names_macos()

        print("\nAvailable cameras:")
        for idx in cameras:
            label = names.get(idx, f"Camera {idx}")
            print(f"  [{idx}] {label}")
        while True:
            try:
                choice = int(input(f"Select camera index [{cameras[0]}]: ").strip() or cameras[0])
                if choice in cameras:
                    return choice
                print(f"  Invalid choice. Pick from {cameras}.")
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
