import shutil
import subprocess
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional

from ender3monitor.framing import reframe, draw_overlay


class TimelapseManager:
    """Saves timelapse frames and compiles them to MP4, with disk retention.

    Without cleanup the frame folders grow unbounded (≈ one JPEG per capture,
    every print, forever). Retention bounds that:
      • keep at most `max_sessions` most-recent session folders
      • delete session folders and MP4s older than `retention_days`
      • optionally delete a session's frames after it is compiled to MP4
    Pruning runs at the start of each new session.
    """

    def __init__(self, output_dir: str = "timelapse_frames",
                 max_sessions: int = 20, retention_days: int = 30,
                 delete_frames_after_compile: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.max_sessions = max(1, max_sessions)
        self.retention_days = max(0, retention_days)
        self.delete_frames_after_compile = delete_frames_after_compile
        self._session_dir: Optional[Path] = None
        self._frame_count = 0

    # ------------------------------------------------------------------ #
    # Capture                                                              #
    # ------------------------------------------------------------------ #

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
        self.prune()                 # bound disk use before starting a new print
        self._session_dir = None
        self._frame_count = 0

    # ------------------------------------------------------------------ #
    # Retention                                                            #
    # ------------------------------------------------------------------ #

    def _session_dirs(self) -> list:
        if not self.output_dir.exists():
            return []
        return sorted([p for p in self.output_dir.iterdir() if p.is_dir()],
                      key=lambda p: p.stat().st_mtime)

    def prune(self) -> int:
        """Delete old session folders / MP4s per the retention policy.

        Returns the number of items removed. Never raises.
        """
        removed = 0
        try:
            cutoff = time.time() - self.retention_days * 86400 if self.retention_days else None

            # 1. Age-based: drop session folders older than the cutoff.
            if cutoff is not None:
                for p in self._session_dirs():
                    if p.stat().st_mtime < cutoff:
                        shutil.rmtree(p, ignore_errors=True)
                        removed += 1

            # 2. Count-based: keep only the newest `max_sessions` folders.
            sessions = self._session_dirs()
            excess = len(sessions) - self.max_sessions
            for p in sessions[:max(0, excess)]:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1

            # 3. Age-based: drop old compiled MP4s too.
            if cutoff is not None:
                for f in self.output_dir.glob("*.mp4"):
                    if f.stat().st_mtime < cutoff:
                        f.unlink(missing_ok=True)
                        removed += 1
        except Exception as exc:
            print(f"  [TIMELAPSE] Prune error (ignored): {exc}")

        if removed:
            print(f"  [TIMELAPSE] Pruned {removed} old item(s) "
                  f"(keep ≤{self.max_sessions} sessions, ≤{self.retention_days} days).")
        return removed

    # ------------------------------------------------------------------ #
    # Compile                                                              #
    # ------------------------------------------------------------------ #

    def compile(self, fps: int = 24, output_file: Optional[str] = None,
                aspect: str = "native", fit: str = "pad_blur",
                overlay_lines: Optional[list] = None) -> Optional[str]:
        if self._session_dir is None:
            print("No timelapse session to compile.")
            return None
        return self.compile_session(self._session_dir, fps=fps, output_file=output_file,
                                     aspect=aspect, fit=fit, overlay_lines=overlay_lines)

    def compile_session(self, session_dir, fps: int = 24, output_file: Optional[str] = None,
                         aspect: str = "native", fit: str = "pad_blur",
                         overlay_lines: Optional[list] = None) -> Optional[str]:
        """Compile a specific session folder (as returned by list_sessions())."""
        session_dir = Path(session_dir)
        if not session_dir.is_dir():
            print(f"No such timelapse session: {session_dir}")
            return None

        frames = sorted(session_dir.glob("frame_*.jpg"))
        if not frames:
            print("No frames found to compile.")
            return None

        sample = cv2.imread(str(frames[0]))
        if sample is None:
            print("Cannot read frames.")
            return None

        # Reframe to the configured aspect ratio (e.g. 9:16 for Instagram).
        # Derive the output dimensions from a reframed sample so every frame
        # written matches the writer's expected size.
        sample = reframe(sample, aspect, fit)
        h, w = sample.shape[:2]
        if output_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.output_dir / f"timelapse_{ts}.mp4")

        # Prefer H.264 (avc1): Telegram/iOS/browsers play it inline, whereas
        # mp4v (MPEG-4 Part 2) usually shows up as a non-previewable file.
        # avc1 works on macOS via VideoToolbox; fall back to mp4v elsewhere.
        writer = None
        codec_used = None
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
            if writer.isOpened():
                codec_used = codec
                break
            writer.release()
        if writer is None or not writer.isOpened():
            print("Cannot open a video writer (no usable codec).")
            return None

        for f in frames:
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = reframe(img, aspect, fit)
            if overlay_lines:
                draw_overlay(img, overlay_lines)
            # Guard against any size drift from rounding so the writer accepts
            # every frame (a mismatched size is silently dropped by OpenCV).
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h))
            writer.write(img)

        writer.release()

        # Last-resort compatibility pass: if we were stuck with mp4v but
        # ffmpeg is available, transcode to H.264 so chat apps preview it.
        if codec_used == "mp4v" and shutil.which("ffmpeg"):
            tmp = output_file + ".h264.mp4"
            rc = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", output_file,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", tmp],
            ).returncode
            if rc == 0 and Path(tmp).exists():
                Path(tmp).replace(output_file)
                codec_used = "h264 (ffmpeg)"
            else:
                Path(tmp).unlink(missing_ok=True)

        print(f"Timelapse compiled: {output_file} "
              f"({len(frames)} frames @ {fps} fps, codec {codec_used})")

        # Reclaim the (now redundant) frames if configured to.
        if self.delete_frames_after_compile:
            shutil.rmtree(session_dir, ignore_errors=True)
            print("  [TIMELAPSE] Source frames deleted after compile (MP4 kept).")

        return output_file

    # ------------------------------------------------------------------ #
    # Session browsing                                                     #
    # ------------------------------------------------------------------ #

    def list_sessions(self, limit: int = 3) -> list:
        """Most-recent-first summary of on-disk session folders.

        Each entry: {"path", "name", "frame_count", "last_frame"} where
        `last_frame` is the Path to that session's final captured JPEG (or
        None if the folder has no frames yet).
        """
        dirs = list(reversed(self._session_dirs()))[:max(0, limit)]
        out = []
        for d in dirs:
            frames = sorted(d.glob("frame_*.jpg"))
            out.append({
                "path": d,
                "name": d.name,
                "frame_count": len(frames),
                "last_frame": frames[-1] if frames else None,
            })
        return out
