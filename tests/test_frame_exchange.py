"""Torn-frame tests for the camera worker's shared-memory hand-off.

Every frame the writer produces is filled with a single value, so a frame the
reader gets back is torn exactly when it isn't uniform.

Run:  python3 -m pytest tests/ -q
"""
import multiprocessing as mp
import sys
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ender3monitor.camera_worker import FrameExchange, frame_shm_size

W, H = 640, 480


def _frame(value: int) -> np.ndarray:
    return np.full((H, W, 3), value % 256, dtype=np.uint8)


def _is_uniform(frame: np.ndarray) -> bool:
    return bool((frame == frame.flat[0]).all())


@pytest.fixture
def shm():
    s = SharedMemory(create=True, size=frame_shm_size(W, H))
    yield s
    s.close()
    s.unlink()


def test_read_returns_latest_published_frame(shm):
    fx = FrameExchange(shm.buf, W, H)
    for v in (7, 8, 9):
        fx.write(_frame(v))
    frame, counter = fx.read()
    assert counter == 3
    assert _is_uniform(frame) and frame[0, 0, 0] == 9
    fx.release()


def test_writer_never_touches_the_published_slot(shm):
    """The next write goes to the other slot, so a reader mid-copy of the
    published frame is unaffected by one more frame arriving."""
    fx = FrameExchange(shm.buf, W, H)
    fx.write(_frame(1))

    def transform(view):
        top = view[: H // 2].copy()
        fx.write(_frame(2))              # worker delivers a frame mid-copy
        bottom = view[H // 2:].copy()
        return np.concatenate([top, bottom])

    frame, counter = fx.read(transform)
    assert counter == 1
    assert _is_uniform(frame) and frame[0, 0, 0] == 1
    fx.release()


def test_reader_lapped_mid_copy_retries_instead_of_returning_torn_frame(shm):
    """Two writes during one copy overwrite the slot being read: the seqlock
    must reject that copy and the retry must return a whole frame."""
    fx = FrameExchange(shm.buf, W, H)
    fx.write(_frame(1))
    calls = 0

    def transform(view):
        nonlocal calls
        calls += 1
        top = view[: H // 2].copy()
        if calls == 1:
            fx.write(_frame(2))
            fx.write(_frame(3))          # lands in the slot we're copying
        bottom = view[H // 2:].copy()
        return np.concatenate([top, bottom])

    frame, counter = fx.read(transform)
    assert calls == 2
    assert counter == 3
    assert _is_uniform(frame) and frame[0, 0, 0] == 3
    fx.release()


def test_reader_gives_up_rather_than_return_torn_frame(shm):
    fx = FrameExchange(shm.buf, W, H)
    fx.write(_frame(0))
    n = 0

    def transform(view):
        nonlocal n
        top = view[: H // 2].copy()
        for _ in range(2):               # lap the reader on every attempt
            n += 1
            fx.write(_frame(n))
        return np.concatenate([top, view[H // 2:].copy()])

    frame, _ = fx.read(transform)
    assert frame is None
    fx.release()


def _writer_proc(shm_name, width, height, stop_event, started):
    """Free-running writer in a separate (spawned) process, like capture_worker."""
    shm = SharedMemory(name=shm_name)
    fx = FrameExchange(shm.buf, width, height)
    frames = [np.full((height, width, 3), v, dtype=np.uint8) for v in range(256)]
    i = 0
    started.set()
    try:
        while not stop_event.is_set():
            fx.write(frames[i & 0xFF])
            i += 1
    finally:
        fx.release()
        shm.close()


def test_cross_process_reader_never_sees_mixed_frame(shm):
    """A real writer process hammering frames as fast as it can, against a
    reader copying in a loop: no frame handed back may mix two captures."""
    ctx = mp.get_context("spawn")
    stop, started = ctx.Event(), ctx.Event()
    proc = ctx.Process(target=_writer_proc, args=(shm.name, W, H, stop, started),
                       daemon=True)
    proc.start()
    fx = FrameExchange(shm.buf, W, H)
    try:
        assert started.wait(30), "writer process failed to start"
        reads = torn = 0
        last_counter = -1
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            frame, counter = fx.read()
            if frame is None:
                continue
            reads += 1
            assert counter >= last_counter
            last_counter = counter
            if not _is_uniform(frame):
                torn += 1
        assert reads > 50
        assert last_counter > reads   # writer really was outrunning the reader
        assert torn == 0
    finally:
        fx.release()
        stop.set()
        proc.join(5)
        if proc.is_alive():
            proc.terminate()
