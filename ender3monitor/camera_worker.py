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

# ── shared-memory frame exchange ──
#
# Layout:  [header: 64 bytes][slot 0: H*W*3][slot 1: H*W*3]
# Header (uint64): [0] publish counter — the last completed frame lives in
#                      slot (counter & 1);
#                  [1], [2] per-slot sequence numbers, odd while that slot is
#                      being written (seqlock).
#
# The writer always fills the slot that is *not* published, bumping that
# slot's seq to odd before and back to even after, then advances the publish
# counter. The reader copies the published slot and re-checks its seq; the
# copy is only discarded and retried if the writer lapped it (finished a whole
# frame and started overwriting this slot mid-copy), which with two slots
# needs a reader slower than one camera frame. No lock on the hot path.
HEADER_BYTES = 64


def frame_shm_size(width: int, height: int) -> int:
    """Bytes of shared memory a FrameExchange of this size needs."""
    return HEADER_BYTES + 2 * width * height * 3


class FrameExchange:
    """Double-buffered, seqlock-checked frame hand-off over a shared buffer.

    One process writes (``write``), another reads (``read``). Holds numpy
    views on ``buf``; call ``release()`` (or drop it) before closing the
    SharedMemory, or ``close()`` raises BufferError.
    """

    MAX_READ_ATTEMPTS = 8

    def __init__(self, buf, width: int, height: int) -> None:
        import numpy as np   # lazy — keep module import side-effect free
        frame_bytes = width * height * 3
        self._hdr = np.ndarray((3,), dtype=np.uint64, buffer=buf)
        self._slots = [
            np.ndarray((height, width, 3), dtype=np.uint8, buffer=buf,
                       offset=HEADER_BYTES + i * frame_bytes)
            for i in (0, 1)
        ]

    def release(self) -> None:
        self._hdr = None
        self._slots = None

    def write(self, frame) -> int:
        """Copy `frame` into the free slot and publish it; returns its counter."""
        hdr = self._hdr
        nxt = int(hdr[0]) + 1
        slot = nxt & 1
        hdr[1 + slot] += 1          # odd: slot is being written
        self._slots[slot][:] = frame
        hdr[1 + slot] += 1          # even: slot complete
        hdr[0] = nxt                # publish
        return nxt

    def read(self, transform=None):
        """Return ``(frame, counter)`` for the latest published frame.

        `transform(view)` must return a new array built from the slot view
        (e.g. ``cv2.flip``); the default is ``view.copy()``. Returns
        ``(None, counter)`` if the writer kept lapping the reader.
        """
        hdr = self._hdr
        counter = 0
        for _ in range(self.MAX_READ_ATTEMPTS):
            counter = int(hdr[0])
            slot = counter & 1
            seq = int(hdr[1 + slot])
            if seq & 1:
                continue            # writer lapped us onto this slot; re-read counter
            view = self._slots[slot]
            frame = transform(view) if transform is not None else view.copy()
            if int(hdr[1 + slot]) == seq:
                return frame, counter
        return None, counter


def capture_worker(index: int, width: int, height: int,
                    shm_name: str, frame_ready, stop_event) -> None:
    """Capture frames from `index` into the FrameExchange in `shm_name`.

    Exits on open failure or a few consecutive read failures instead of
    retrying internally — the supervisor treats any exit as "dead" and spawns
    a brand-new process, which is the only thing that reliably recovers a
    camera that dropped and came back (see module docstring).
    """
    import cv2       # imported lazily so importing this module has no hard cv2 dependency

    shm = SharedMemory(name=shm_name)
    frames = FrameExchange(shm.buf, width, height)
    try:

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
                frames.write(frame)
                frame_ready.set()
        finally:
            cap.release()
    finally:
        frames.release()   # drop numpy views first or shm.close() raises BufferError
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
