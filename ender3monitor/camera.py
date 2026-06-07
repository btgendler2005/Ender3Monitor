import contextlib
import os
import re
import subprocess
import sys
import tempfile
import cv2
import numpy as np
from pathlib import Path
from typing import Optional


@contextlib.contextmanager
def _quiet():
    """Suppress C-level stderr (OpenCV's 'out of bound' noise during camera scan)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


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
        return re.findall(r"^    ([^:\n]+):$", out, re.MULTILINE)
    except Exception:
        return []


def _snapshot(index: int, width: int = 1280, height: int = 720) -> Optional[np.ndarray]:
    """Open a camera, grab one frame, immediately release it.

    Opening and releasing on every capture eliminates the OpenCV internal
    capture thread that otherwise runs at 30 fps continuously in the
    background, even when no frames are being consumed.  On an 8 GB M2 this
    was the primary source of CPU / memory-bandwidth pressure.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Discard first frame — camera needs one read to finish initialising.
    cap.read()
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


class CameraManager:
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720,
                 flip: Optional[int] = None):
        self.camera_index = camera_index
        self._width = width
        self._height = height
        self._flip = flip   # cv2.flip code: -1=180°, 0=vertical, 1=horizontal

    # ------------------------------------------------------------------ #
    # Snapshot capture (preferred for monitoring — no persistent thread)  #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Optional[np.ndarray]:
        """Capture one frame and immediately release the camera.

        Call this from the monitoring loop instead of keeping the camera open.
        The camera is only active for the ~200 ms it takes to warm up and read
        one frame; the rest of the time no background thread is running.
        """
        frame = _snapshot(self.camera_index, self._width, self._height)
        if frame is not None and self._flip is not None:
            frame = cv2.flip(frame, self._flip)
        return frame

    # ------------------------------------------------------------------ #
    # Camera discovery and selection                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def list_available_cameras(max_check: int = 5) -> list[tuple[int, int, int]]:
        """Return [(index, width, height), ...] for every readable camera."""
        available = []
        with _quiet():
            for i in range(max_check):
                frame = _snapshot(i)
                if frame is not None:
                    h, w = frame.shape[:2]
                    available.append((i, w, h))
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

        # Capture a thumbnail from each camera and open them all in Preview
        # so the user can see exactly what each index sees before choosing.
        tmp = Path(tempfile.gettempdir())
        thumbs: dict[int, Path] = {}
        for idx, w, h in cameras:
            frame = _snapshot(idx)
            if frame is not None:
                path = tmp / f"ender3monitor_cam{idx}.jpg"
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                thumbs[idx] = path

        print("\nAvailable cameras:")
        for idx, w, h in cameras:
            print(f"  [{idx}] Camera {idx}  ({w}×{h})")

        if thumbs:
            print("\n  Opening preview snapshots so you can see which is which…")
            for idx, path in sorted(thumbs.items()):
                print(f"    Camera {idx} → {path}")
            if sys.platform == "darwin":
                subprocess.Popen(["open"] + [str(p) for p in sorted(thumbs.values())])
            else:
                print("  Open the files above to identify each camera.")

        print()
        indices = [idx for idx, _, _ in cameras]
        while True:
            try:
                choice = int(input(f"  Select camera index [{indices[0]}]: ").strip() or indices[0])
                if choice in indices:
                    return choice
                print(f"  Invalid choice. Pick from {indices}.")
            except ValueError:
                print("  Please enter a number.")

    # ------------------------------------------------------------------ #
    # Legacy persistent-capture API (kept for backward compatibility)      #
    # ------------------------------------------------------------------ #

    def open(self) -> None:
        """Verify the camera index is readable. Does not keep it open."""
        frame = _snapshot(self.camera_index, self._width, self._height)
        if frame is None:
            raise RuntimeError(f"Cannot open camera {self.camera_index}.")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Single-shot capture (alias for snapshot)."""
        return self.snapshot()

    def release(self) -> None:
        """No-op — camera is released after every snapshot."""
        pass
