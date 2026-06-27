"""Aspect-ratio reframing and burn-in overlays — shared by the live stream
and the timelapse compiler so both honour the same Camera-ratio setting.

Why reframe instead of changing the capture resolution: the camera is a fixed
16:9 sensor and the AI failure detector must keep seeing the *full* frame, so
the ratio is applied only to what gets shown/saved, never to what gets analysed.

Reframing a wider source to a narrower target (e.g. 9:16 from 16:9) can never
add pixels, so each fit mode is a different trade-off:
  • pad_blur  — scale the whole frame to fit, fill the gaps with a blurred,
                scaled-up copy of itself. Nothing is cut (the Instagram look).
  • letterbox — same fit, but solid black bars instead of blur.
  • crop      — centre-crop to the target ratio. Fuller subject, cuts the sides.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

# UI choices → (width, height) ratio. "native" leaves the frame untouched.
ASPECTS = {
    "native": None,
    "9:16": (9, 16),   # Reels / Stories / TikTok
    "4:5": (4, 5),     # Instagram feed portrait
    "1:1": (1, 1),     # Square
}

FIT_MODES = ("pad_blur", "letterbox", "crop")


def _even(n: float) -> int:
    """Round to the nearest even int — H.264 requires even frame dimensions."""
    n = int(round(n))
    return n - (n % 2)


def reframe(frame: Optional[np.ndarray], aspect: str,
            mode: str = "pad_blur") -> Optional[np.ndarray]:
    """Return `frame` reframed to `aspect` using fit `mode`.

    No-ops (returns the input) for None frames, the "native" aspect, or a
    source that already matches the target ratio.
    """
    spec = ASPECTS.get(aspect)
    if frame is None or spec is None:
        return frame
    tw, th = spec
    target = tw / th
    h, w = frame.shape[:2]
    source = w / h
    if abs(target - source) < 1e-3:
        return frame

    if mode == "crop":
        if target < source:                       # crop the sides
            nw = _even(h * target)
            x = (w - nw) // 2
            return frame[:, x:x + nw].copy()
        nh = _even(w / target)                    # crop top/bottom
        y = (h - nh) // 2
        return frame[y:y + nh, :].copy()

    # Contain modes (pad_blur / letterbox): build a canvas at the target ratio
    # large enough to hold the whole frame at native size, then centre it.
    if target < source:
        cw, ch = _even(w), _even(w / target)
    else:
        cw, ch = _even(h * target), _even(h)

    scale = min(cw / w, ch / h)
    fw, fh = _even(w * scale), _even(h * scale)
    fg = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)

    if mode == "letterbox":
        canvas = np.zeros((ch, cw, 3), dtype=frame.dtype)
    else:  # pad_blur — background is a blurred copy scaled to cover the canvas
        bscale = max(cw / w, ch / h)
        bg = cv2.resize(frame, (_even(w * bscale), _even(h * bscale)),
                        interpolation=cv2.INTER_LINEAR)
        bx = (bg.shape[1] - cw) // 2
        by = (bg.shape[0] - ch) // 2
        bg = bg[by:by + ch, bx:bx + cw]
        sigma = max(cw, ch) / 30.0
        canvas = cv2.GaussianBlur(bg, (0, 0), sigmaX=sigma)

    x = (cw - fw) // 2
    y = (ch - fh) // 2
    canvas[y:y + fh, x:x + fw] = fg
    return canvas


def draw_overlay(frame: Optional[np.ndarray],
                 lines: List[str]) -> Optional[np.ndarray]:
    """Burn `lines` into the bottom-left of `frame` on a translucent bar.

    Font size scales with frame width so it reads the same on a cropped 9:16
    clip as on a full 16:9 one. Mutates and returns `frame`.
    """
    if frame is None or not lines:
        return frame
    h, w = frame.shape[:2]
    scale = max(0.5, w / 1600.0)
    thick = max(1, int(round(scale * 2)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = int(round(14 * scale))

    sized: List[Tuple[str, int, int]] = []
    for text in lines:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        sized.append((text, tw, th))
    line_h = max(th for _, _, th in sized) + int(round(10 * scale))
    box_w = max(tw for _, tw, _ in sized) + 2 * pad
    box_h = line_h * len(sized) + pad

    x0, y1 = pad, h - pad
    y0 = y1 - box_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = y0 + line_h
    for text, _, _ in sized:
        cv2.putText(frame, text, (x0 + pad, y), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)
        y += line_h
    return frame
