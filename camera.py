import cv2
import numpy as np
from typing import Optional


class CameraManager:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self._cap: Optional[cv2.VideoCapture] = None

    @staticmethod
    def list_available_cameras(max_check: int = 10) -> list[int]:
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

        print("\nAvailable cameras:")
        for idx in cameras:
            print(f"  [{idx}] Camera {idx}")
        while True:
            try:
                choice = int(input(f"Select camera index [{cameras[0]}]: ").strip() or cameras[0])
                if choice in cameras:
                    return choice
                print(f"  Invalid choice. Pick from {cameras}.")
            except ValueError:
                print("  Please enter a number.")

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}.")

    def capture_frame(self) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
