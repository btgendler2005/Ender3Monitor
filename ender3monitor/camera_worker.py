"""Camera capture, isolated in its own OS process.

On macOS, once a USB camera drops mid-session (e.g. a docking station gets
unplugged), OpenCV's AVFoundation capture backend is left wedged for the rest
of that process's life — releasing and recreating a cv2.VideoCapture in the
same process does not reliably pick the device back up, even after it's
replugged. A fresh OS process gets a fresh AVFoundation session and finds it
immediately, which is why restarting the whole app "fixes" it.

This module runs the actual cv2.VideoCapture loop as a standalone target for
multiprocessing so the supervisor (StreamCapture in web.py) can kill and
respawn just this piece — a real fresh process each time — without dropping
the printer connection, web server, or any other app state. Deliberately has
no import-time side effects, since the ``spawn`` start method re-imports this
module in the child.
"""
from __future__ import annotations

import time
from multiprocessing.shared_memory import SharedMemory


def capture_worker(index: int, width: int, height: int,
                    shm_name: str, frame_ready, stop_event) -> None:
    """Capture frames from `index` into the shared-memory buffer `shm_name`.

    Exits on open failure or a few consecutive read failures instead of
    retrying internally — the supervisor treats any exit as "dead" and spawns
    a brand-new process, which is the only thing that reliably recovers a
    camera that dropped and came back (see module docstring).
    """
    import cv2       # imported lazily so importing this module has no hard cv2 dependency
    import numpy as np

    shm = SharedMemory(name=shm_name)
    try:
        frame_buf = np.ndarray((height, width, 3), dtype=np.uint8, buffer=shm.buf)

        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            consecutive_fails = 0
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_fails += 1
                    if consecutive_fails >= 3:
                        return
                    time.sleep(0.1)
                    continue
                consecutive_fails = 0
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                frame_buf[:] = frame
                frame_ready.set()
        finally:
            cap.release()
    finally:
        shm.close()


def scan_worker(max_check: int, width: int, height: int, result_queue) -> None:
    """One-shot enumeration of readable camera indices, isolated in its own process.

    Same rationale as capture_worker: opening cv2.VideoCapture from a
    background thread of the long-lived app process (as the old in-process
    scan did) can leave macOS's AVFoundation backend unable to open *any*
    camera for the rest of that process's life. Running the whole scan on a
    fresh process's main thread avoids that.
    """
    import cv2   # imported lazily — see capture_worker

    results = []
    for index in range(max_check):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.read()   # discard first frame — camera needs one read to finish initialising
            ok, frame = cap.read()
        finally:
            cap.release()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            results.append((index, w, h))
    result_queue.put(results)
