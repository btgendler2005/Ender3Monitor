"""Slicer-aware print status: index a G-code file, then follow it live.

WHY THIS EXISTS
---------------
The printer runs from its SD card, so the host never sees the G-code. Over USB
we only get what we ask for: temps (M105), an SD byte offset (M27), elapsed time
(M31) and a Z height (M114). That leaves the monitor guessing — "44%" is a byte
percentage, and bytes are not time, so the linear ETA in `PrinterController`
drifts badly on prints whose density varies with height (a lithophane's dense
top costs far more time per byte than its sparse base).

The file itself already carries the answers. Orca emits `;LAYER_CHANGE` / `;Z:`
markers, `;TYPE:<feature>` spans, `M600` color changes and — when the printer
profile's "Disable set remaining print time" is off — `M73 P.. R..` lines with
its own acceleration-aware time estimate. None of that needs "Verbose G-code";
it's all present in a default export.

THE LINK BETWEEN FILE AND PRINT
-------------------------------
The file is only on the host briefly, while the SD card is mounted at export
time. So we index it *then* and keep only the index — a few hundred KB standing
in for a 118 MB file. Hours later, `M27` replies:

    SD printing byte 5231227/11973483
                             ^^^^^^^^
The total is the file's exact byte size, which is a usable fingerprint: it
identifies which file the user picked on the printer's LCD without the host ever
being told. Matching is confirmed against Z (see `confirm_z`) before we trust it,
and every consumer degrades to the old byte-percentage behaviour on no match.

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
No G-code is executed, simulated, or re-emitted. This module only reads, and
only ever reports; nothing it produces can change what the printer does.
"""
from __future__ import annotations

import bisect
import gzip
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Index format version. Bumping this invalidates every cached index, which is
# the intended migration path — reindexing is cheap and the source files are
# usually gone, so a stale-but-parseable index is worse than no index.
INDEX_VERSION = 1

# Parsed straight out of the file. All of these appear in a stock Orca export
# with `gcode_comments = 0`; verbose mode adds nothing we use.
_RE_TIME = re.compile(rb"^;TIME:\s*([0-9.]+)")
_RE_FILAMENT = re.compile(rb"^;Filament used:\s*([0-9.]+)m")
_RE_TOTAL_LAYERS = re.compile(rb"^;\s*total layer number:\s*(\d+)")
_RE_Z = re.compile(rb"^;Z:\s*([0-9.]+)")
_RE_TYPE = re.compile(rb"^;TYPE:\s*(.+?)\s*$")
_RE_M73 = re.compile(rb"^M73\b.*?\bR(\d+)")
_RE_M73_P = re.compile(rb"^M73\b.*?\bP(\d+)")
_RE_LAYER_HEIGHT = re.compile(rb"^;\s*layer_height\s*=\s*([0-9.]+)")
_RE_FILAMENT_TYPE = re.compile(rb"^;\s*filament_type\s*=\s*([^;\s]+)")
_RE_PRINTER_MODEL = re.compile(rb"^;\s*printer_model\s*=\s*(.+?)\s*$")

# Feature names we care enough about to tell the vision model. Anything else is
# still recorded, just not called out as a reason for expected-looking sag.
BRIDGING_FEATURES = {"bridge", "internal bridge", "overhang wall"}


def _f(raw: bytes) -> float:
    return float(raw.decode("ascii", "ignore"))


@dataclass
class GcodeIndex:
    """A compact, queryable summary of one G-code file.

    Offsets are byte positions from the start of the file, matching what M27
    reports. Parallel arrays (rather than a list of objects) keep the JSON small
    and the bisect lookups direct.
    """
    size_bytes: int = 0
    source_name: str = ""
    indexed_at: float = 0.0

    # Layer table: layer_offsets[i] is where layer i+1 begins; layer_z[i] its Z.
    layer_offsets: List[int] = field(default_factory=list)
    layer_z: List[float] = field(default_factory=list)
    total_layers: int = 0

    # Feature spans: feature_offsets[i] is where feature_names[i] starts.
    feature_offsets: List[int] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)

    # Slicer time estimate, if M73 is enabled in the printer profile.
    m73_offsets: List[int] = field(default_factory=list)
    m73_remaining_min: List[int] = field(default_factory=list)

    # Fallback time model, computed here from move distances and feedrates.
    # layer_time_frac[i] is the fraction of total print time elapsed at the
    # START of layer i+1, so it runs 0.0 → <1.0.
    #
    # This exists because neither bytes nor layers track time. Measured against
    # this model over a real library: byte-fraction diverged by up to 9.1 points
    # and layer-fraction by up to 15.4 — layers are not the safer fallback they
    # look like. The model ignores acceleration, so its absolute total is a
    # sizeable underestimate; only its *shape* is used, scaled by the slicer's
    # own `;TIME:` and then re-scaled against live elapsed time.
    layer_time_frac: List[float] = field(default_factory=list)
    modelled_seconds: Optional[float] = None

    # Embedded color/filament changes.
    m600_offsets: List[int] = field(default_factory=list)

    # Header/config metadata, all optional.
    estimated_seconds: Optional[float] = None
    filament_meters: Optional[float] = None
    layer_height: Optional[float] = None
    filament_type: Optional[str] = None
    printer_model: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Lookups                                                              #
    # ------------------------------------------------------------------ #

    @property
    def has_m73(self) -> bool:
        return len(self.m73_offsets) > 1

    def layer_at_z(self, z: Optional[float]) -> Optional[int]:
        """1-based layer number for a measured nozzle Z, or None.

        Preferred over `layer_at_offset` for "what layer am I on": Marlin's SD
        read pointer runs ahead of the nozzle by however deep the planner queue
        is, so the byte offset is optimistic while M114's Z is what is actually
        being printed right now.

        Picks the closest layer Z rather than the last one below it — a nozzle
        parked mid-Z-hop should not read as the layer above.
        """
        if z is None or not self.layer_z:
            return None
        i = bisect.bisect_left(self.layer_z, z)
        best, best_d = None, None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(self.layer_z):
                d = abs(self.layer_z[j] - z)
                if best_d is None or d < best_d:
                    best, best_d = j, d
        if best is None:
            return None
        # Well outside any known layer (wrong file, or parked for a change).
        tol = max(1.0, (self.layer_height or 0.2) * 3)
        if best_d is not None and best_d > tol:
            return None
        return best + 1

    def layer_at_offset(self, pos: Optional[int]) -> Optional[int]:
        """1-based layer number for a byte offset, or None."""
        if pos is None or not self.layer_offsets:
            return None
        i = bisect.bisect_right(self.layer_offsets, pos)
        return i if i > 0 else None

    def feature_at_offset(self, pos: Optional[int]) -> Optional[str]:
        """The `;TYPE:` in effect at a byte offset (e.g. "Outer wall")."""
        if pos is None or not self.feature_offsets:
            return None
        i = bisect.bisect_right(self.feature_offsets, pos)
        return self.feature_names[i - 1] if i > 0 else None

    def remaining_seconds_at_offset(self, pos: Optional[int]) -> Optional[float]:
        """Slicer's own remaining-time estimate at a byte offset, interpolated.

        Returns None when the file carries no M73 lines — i.e. the printer
        profile has "Disable set remaining print time" checked. Callers fall
        back to a layer- or byte-based estimate and say so.
        """
        if pos is None or not self.has_m73:
            return None
        offs, rem = self.m73_offsets, self.m73_remaining_min
        i = bisect.bisect_right(offs, pos)
        if i <= 0:
            return rem[0] * 60.0
        if i >= len(offs):
            return rem[-1] * 60.0
        # Linear interpolation between the bracketing M73 stamps.
        x0, x1 = offs[i - 1], offs[i]
        y0, y1 = rem[i - 1], rem[i]
        if x1 == x0:
            return y1 * 60.0
        frac = (pos - x0) / (x1 - x0)
        return (y0 + (y1 - y0) * frac) * 60.0

    def time_frac_at_offset(self, pos: Optional[int]) -> Optional[float]:
        """Modelled fraction of print time elapsed at a byte offset (0..1).

        Interpolates linearly *within* the containing layer. Bytes track time
        poorly across a whole print but acceptably inside a single layer, which
        is all this interpolation has to cover.
        """
        if pos is None or not self.layer_time_frac or not self.layer_offsets:
            return None
        offs, fracs = self.layer_offsets, self.layer_time_frac
        i = bisect.bisect_right(offs, pos)
        if i <= 0:
            return 0.0
        if i >= len(offs):
            return fracs[-1] if fracs else None
        f0, f1 = fracs[i - 1], fracs[i]
        x0, x1 = offs[i - 1], offs[i]
        if x1 <= x0:
            return f0
        return f0 + (f1 - f0) * ((pos - x0) / (x1 - x0))

    def modelled_remaining_seconds(self, pos: Optional[int]) -> Optional[float]:
        """Remaining time from the built-in model, scaled by the slicer's total.

        Used when the file carries no M73 lines. The model supplies the shape
        (which parts of the print are slow) and `;TIME:` supplies the scale.
        """
        frac = self.time_frac_at_offset(pos)
        total = self.estimated_seconds or self.modelled_seconds
        if frac is None or not total:
            return None
        return max(0.0, total * (1.0 - frac))

    def next_color_change(self, pos: Optional[int]) -> Optional[int]:
        """Byte offset of the next M600 after `pos`, or None if none remain."""
        if pos is None or not self.m600_offsets:
            return None
        i = bisect.bisect_right(self.m600_offsets, pos)
        return self.m600_offsets[i] if i < len(self.m600_offsets) else None

    def confirm_z(self, pos: Optional[int], z: Optional[float],
                  tolerance_layers: float = 12.0) -> bool:
        """Sanity-check that this index really describes the running print.

        Byte size alone is a good fingerprint but not a proof — across a real
        33-file library the closest pair sat 4,871 bytes apart, so a collision
        is improbable rather than impossible. This is the second opinion: the
        index predicts a Z for the current byte offset, and M114 says what Z the
        nozzle is actually at. A wrong file disagrees wildly (Z profiles across
        that same library ranged 1.7mm to 75.8mm at the midpoint).

        The tolerance is deliberately loose — the SD read pointer leads the
        nozzle by the planner queue depth, so a few layers of disagreement is
        normal and only a gross mismatch should disqualify the index.
        """
        if z is None or pos is None or not self.layer_z:
            return True   # nothing to check against — don't reject on no evidence
        expected = self.layer_at_offset(pos)
        if expected is None:
            return True
        expected_z = self.layer_z[min(expected - 1, len(self.layer_z) - 1)]
        slack = max(1.0, (self.layer_height or 0.2) * tolerance_layers)
        return abs(expected_z - z) <= slack

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "v": INDEX_VERSION,
            "size_bytes": self.size_bytes,
            "source_name": self.source_name,
            "indexed_at": self.indexed_at,
            "layer_offsets": self.layer_offsets,
            "layer_z": self.layer_z,
            "total_layers": self.total_layers,
            "feature_offsets": self.feature_offsets,
            "feature_names": self.feature_names,
            "m73_offsets": self.m73_offsets,
            "m73_remaining_min": self.m73_remaining_min,
            "layer_time_frac": [round(f, 6) for f in self.layer_time_frac],
            "modelled_seconds": self.modelled_seconds,
            "m600_offsets": self.m600_offsets,
            "estimated_seconds": self.estimated_seconds,
            "filament_meters": self.filament_meters,
            "layer_height": self.layer_height,
            "filament_type": self.filament_type,
            "printer_model": self.printer_model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Optional["GcodeIndex"]:
        if not isinstance(d, dict) or d.get("v") != INDEX_VERSION:
            return None
        idx = cls()
        for k in ("size_bytes", "source_name", "indexed_at", "layer_offsets",
                  "layer_z", "total_layers", "feature_offsets", "feature_names",
                  "m73_offsets", "m73_remaining_min", "m600_offsets",
                  "layer_time_frac", "modelled_seconds",
                  "estimated_seconds", "filament_meters", "layer_height",
                  "filament_type", "printer_model"):
            if k in d:
                setattr(idx, k, d[k])
        return idx


def parse_gcode(path: Path, max_bytes: int = 0) -> Optional[GcodeIndex]:
    """Read a G-code file once and return its index. Never raises.

    Streams in binary and only decodes the handful of bytes that match a marker
    — a 118 MB lithophane is mostly G1 moves we have no reason to look at, so
    decoding every line would dominate the runtime for no benefit.

    `max_bytes` guards against indexing something enormous by accident; 0 = no
    limit.
    """
    try:
        path = Path(path)
        size = path.stat().st_size
        if size <= 0 or (max_bytes and size > max_bytes):
            return None

        idx = GcodeIndex(size_bytes=size, source_name=path.name,
                         indexed_at=time.time())
        pos = 0
        want_z = False   # set by ;LAYER_CHANGE, consumed by the next ;Z:
        last_feature: Optional[str] = None

        # Move-timing state for the fallback model. Feedrate is sticky across
        # lines, so it carries between iterations like the printer's own does.
        mx = my = mz = 0.0
        feed = 0.0                      # mm/s
        layer_times: List[float] = []
        cur_layer_time = 0.0
        seen_layer = False

        with open(path, "rb") as fh:
            for raw in fh:
                start = pos
                pos += len(raw)
                if not raw:
                    continue
                c = raw[0]

                # Comments carry every structural marker we want.
                if c == 0x3B:            # ';'
                    if raw.startswith(b";LAYER_CHANGE"):
                        idx.layer_offsets.append(start)
                        if seen_layer:
                            layer_times.append(cur_layer_time)
                        cur_layer_time = 0.0
                        seen_layer = True
                        want_z = True
                        continue
                    if want_z:
                        m = _RE_Z.match(raw)
                        if m:
                            idx.layer_z.append(_f(m.group(1)))
                            want_z = False
                            continue
                    if raw.startswith(b";TYPE:"):
                        m = _RE_TYPE.match(raw)
                        if m:
                            name = m.group(1).decode("ascii", "ignore")
                            # Collapse runs — a feature repeated back-to-back
                            # adds entries without adding information.
                            if name != last_feature:
                                idx.feature_offsets.append(start)
                                idx.feature_names.append(name)
                                last_feature = name
                        continue
                    # Header / config scalars. Cheap prefix test first so the
                    # ~23 KB trailing config block doesn't cost regex per line.
                    if idx.estimated_seconds is None and raw.startswith(b";TIME:"):
                        m = _RE_TIME.match(raw)
                        if m:
                            idx.estimated_seconds = _f(m.group(1))
                    elif idx.filament_meters is None and raw.startswith(b";Filament used:"):
                        m = _RE_FILAMENT.match(raw)
                        if m:
                            idx.filament_meters = _f(m.group(1))
                    elif not idx.total_layers and b"total layer number" in raw:
                        m = _RE_TOTAL_LAYERS.match(raw)
                        if m:
                            idx.total_layers = int(m.group(1))
                    elif idx.layer_height is None and b"layer_height" in raw:
                        m = _RE_LAYER_HEIGHT.match(raw)
                        if m:
                            idx.layer_height = _f(m.group(1))
                    elif idx.filament_type is None and b"filament_type" in raw:
                        m = _RE_FILAMENT_TYPE.match(raw)
                        if m:
                            idx.filament_type = m.group(1).decode("ascii", "ignore")
                    elif idx.printer_model is None and b"printer_model" in raw:
                        m = _RE_PRINTER_MODEL.match(raw)
                        if m:
                            idx.printer_model = m.group(1).decode("ascii", "ignore")
                    continue

                # Commands: only M600 and M73 matter to us.
                if c == 0x4D:            # 'M'
                    if raw.startswith(b"M600"):
                        idx.m600_offsets.append(start)
                    elif raw.startswith(b"M73"):
                        m = _RE_M73.match(raw)
                        if m:
                            idx.m73_offsets.append(start)
                            idx.m73_remaining_min.append(int(m.group(1)))
                    continue

                # Motion, for the fallback time model. Only G0/G1 — arcs (G2/G3)
                # are not emitted by Orca for this printer profile.
                if c == 0x47 and (raw.startswith(b"G1 ") or raw.startswith(b"G0 ")):
                    body = raw.split(b";", 1)[0] if 0x3B in raw else raw
                    nx, ny, nz, e = mx, my, mz, None
                    for tok in body.split()[1:]:
                        if len(tok) < 2:
                            continue
                        k = tok[0]
                        try:
                            v = float(tok[1:])
                        except ValueError:
                            continue
                        if k == 0x58:      # X
                            nx = v
                        elif k == 0x59:    # Y
                            ny = v
                        elif k == 0x5A:    # Z
                            nz = v
                        elif k == 0x45:    # E
                            e = v
                        elif k == 0x46:    # F (mm/min → mm/s)
                            feed = v / 60.0
                    dx, dy, dz = nx - mx, ny - my, nz - mz
                    dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                    if dist == 0.0 and e is not None:
                        dist = abs(e)      # retract/prime: extruder-only move
                    if feed > 0.0 and dist > 0.0:
                        cur_layer_time += dist / feed
                    mx, my, mz = nx, ny, nz

        if seen_layer:
            layer_times.append(cur_layer_time)
        total_modelled = sum(layer_times)
        if total_modelled > 0 and len(layer_times) == len(idx.layer_offsets):
            idx.modelled_seconds = total_modelled
            # Cumulative fraction at the START of each layer (so it opens at 0).
            running = 0.0
            fracs = []
            for t in layer_times:
                fracs.append(running / total_modelled)
                running += t
            idx.layer_time_frac = fracs

        if not idx.total_layers:
            idx.total_layers = len(idx.layer_offsets)
        # A file with no layer markers isn't something we can follow.
        if not idx.layer_offsets:
            return None
        # Guard the invariant every lookup assumes: the layer arrays are
        # index-aligned, so bisecting one is a valid index into the others.
        n = min(len(idx.layer_offsets), len(idx.layer_z))
        idx.layer_offsets = idx.layer_offsets[:n]
        idx.layer_z = idx.layer_z[:n]
        idx.layer_time_frac = idx.layer_time_frac[:n]
        # M73 R counts *down*; a non-monotonic table means we misparsed and
        # interpolation would produce nonsense, so drop it rather than lie.
        if any(idx.m73_remaining_min[i] < idx.m73_remaining_min[i + 1]
               for i in range(len(idx.m73_remaining_min) - 1)):
            idx.m73_offsets, idx.m73_remaining_min = [], []
        return idx
    except Exception:
        return None


class IndexStore:
    """Persistent, size-keyed collection of G-code indexes.

    Keyed by byte size because that is the only identifier `M27` gives us. The
    index outlives the file it describes by design: the G-code goes to the SD
    card and is typically deleted from the host, while this stays behind.
    """

    def __init__(self, cache_dir: Path, max_indexes: int = 200) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_indexes = max_indexes
        self._lock = threading.RLock()
        self._by_size: Dict[int, GcodeIndex] = {}
        self._seen: Dict[str, Tuple[int, float]] = {}   # path -> (size, mtime)
        self._load_all()

    # ── persistence ──
    def _path_for(self, size: int) -> Path:
        return self.cache_dir / f"{size}.json.gz"

    def _load_all(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            print(f"  [GCODE] cache dir unavailable (ignored): {exc}")
            return
        for p in sorted(self.cache_dir.glob("*.json.gz")):
            try:
                with gzip.open(p, "rt", encoding="utf-8") as fh:
                    idx = GcodeIndex.from_dict(json.load(fh))
                if idx is not None:
                    self._by_size[idx.size_bytes] = idx
                else:
                    p.unlink(missing_ok=True)   # stale format version
            except Exception:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

    def _save(self, idx: GcodeIndex) -> None:
        try:
            tmp = self._path_for(idx.size_bytes).with_suffix(".gz.tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump(idx.to_dict(), fh, separators=(",", ":"))
            tmp.replace(self._path_for(idx.size_bytes))
        except Exception as exc:
            print(f"  [GCODE] index save failed (ignored): {exc}")

    def _prune(self) -> None:
        """Keep the newest `max_indexes`. Indexes are small; this is a backstop."""
        if len(self._by_size) <= self.max_indexes:
            return
        for idx in sorted(self._by_size.values(), key=lambda i: i.indexed_at)[
                :len(self._by_size) - self.max_indexes]:
            self._by_size.pop(idx.size_bytes, None)
            try:
                self._path_for(idx.size_bytes).unlink(missing_ok=True)
            except Exception:
                pass

    # ── public API ──
    def match(self, size: Optional[int]) -> Optional[GcodeIndex]:
        """The index for a file of exactly this byte size, if we have one."""
        if not size:
            return None
        with self._lock:
            return self._by_size.get(int(size))

    def add_file(self, path: Path, force: bool = False) -> Optional[GcodeIndex]:
        """Index one G-code file (skipping unchanged repeats). Never raises."""
        try:
            path = Path(path)
            st = path.stat()
        except Exception:
            return None
        key = str(path)
        with self._lock:
            if not force and self._seen.get(key) == (st.st_size, st.st_mtime):
                return self._by_size.get(st.st_size)
            if not force and st.st_size in self._by_size:
                # Already known by fingerprint (e.g. re-inserted card).
                self._seen[key] = (st.st_size, st.st_mtime)
                return self._by_size[st.st_size]

        idx = parse_gcode(path)
        with self._lock:
            self._seen[key] = (st.st_size, st.st_mtime)
            if idx is None:
                return None
            self._by_size[idx.size_bytes] = idx
            self._save(idx)
            self._prune()
        return idx

    def scan_dir(self, directory: Path, recursive: bool = False) -> int:
        """Index every .gcode in a directory. Returns how many were newly added."""
        try:
            directory = Path(directory)
            if not directory.is_dir():
                return 0
            it = directory.rglob("*") if recursive else directory.glob("*")
            paths = [p for p in it
                     if p.is_file() and p.suffix.lower() in (".gcode", ".gco", ".g")]
        except Exception:
            return 0
        added = 0
        for p in paths:
            before = len(self._by_size)
            if self.add_file(p) is not None and len(self._by_size) > before:
                added += 1
        return added

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._by_size)


class VolumeWatcher:
    """Index G-code as it lands on the SD card, before the card leaves the host.

    The export → eject → print workflow means the file is reachable for only a
    few seconds. This watches for removable volumes appearing under /Volumes and
    indexes what it finds, so by the time the print starts the file can be gone
    and the index is still here.

    Cards are matched by *content*, not name — the same card comes back as
    NONAME or PRINTER depending on how it was last formatted, so keying on a
    label would silently stop working after a reformat. Any freshly-mounted
    volume with .gcode at its top level qualifies.
    """

    #: Volumes that are never removable media, skipped to avoid pointless work.
    _SKIP_NAMES = {"Macintosh HD", "Recovery", "Preboot", "VM", "Update", "xarts",
                   "iSCPreboot", "Hardware", ".timemachine"}

    def __init__(self, store: IndexStore, extra_paths: Optional[List[Path]] = None,
                 volume_root: Path = Path("/Volumes"), interval: float = 15.0) -> None:
        self.store = store
        self.extra_paths = [Path(p) for p in (extra_paths or [])]
        self.volume_root = Path(volume_root)
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._known_volumes: set = set()
        self.last_scan: Optional[float] = None
        self.last_added: int = 0

    def _candidate_volumes(self) -> List[Path]:
        try:
            if not self.volume_root.is_dir():
                return []
            out = []
            for p in self.volume_root.iterdir():
                if p.name in self._SKIP_NAMES or p.name.startswith("."):
                    continue
                try:
                    if p.is_dir():
                        out.append(p)
                except Exception:
                    continue    # unreadable mount — not ours to worry about
            return out
        except Exception:
            return []

    def scan_once(self, force: bool = False) -> int:
        """Index newly-appeared volumes plus any configured folders.

        Returns the number of newly indexed files. `force` re-scans volumes we
        have already seen this session (used for the manual "rescan" action).
        """
        added = 0
        current = set()
        for vol in self._candidate_volumes():
            current.add(str(vol))
            if not force and str(vol) in self._known_volumes:
                continue
            n = self.store.scan_dir(vol)
            if n:
                print(f"  [GCODE] indexed {n} file(s) from {vol}")
            added += n
        # Drop ejected volumes so re-inserting the same card rescans it.
        self._known_volumes = current
        for extra in self.extra_paths:
            added += self.store.scan_dir(extra)
        self.last_scan = time.time()
        self.last_added = added
        return added

    def start(self) -> None:
        """Begin watching. Idempotent."""
        if self._thread is not None:
            return
        self._stop.clear()

        def _loop() -> None:
            # First pass immediately so a card already inserted at startup, or a
            # configured folder, is picked up without waiting a full interval.
            while True:
                try:
                    self.scan_once()
                except Exception as exc:
                    print(f"  [GCODE] scan error (ignored): {exc}")
                if self._stop.wait(self.interval):
                    return

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=3)
        self._thread = None
