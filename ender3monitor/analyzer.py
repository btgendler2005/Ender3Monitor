import base64
import json
import re
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Protocol

# Anthropic model — claude-sonnet-4-6 is the current ID
# (formerly advertised as claude-sonnet-4-20250514, which is deprecated)
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Default Ollama model — llava:7b is the best fit for 8GB M2 MacBook Air.
# llama3.2-vision:11b is higher quality but will swap heavily on 8GB.
OLLAMA_DEFAULT_MODEL = "llava:7b"

SYSTEM_PROMPT = """You are a 3D-print failure detector reviewing webcam frames of a printer. Catch only SEVERE, unmistakable failures. A healthy print — even a messy- or odd-looking one — must be reported as "none". False alarms are disruptive; missing a subtle issue is acceptable.

First describe what you actually see, then decide. Respond with ONLY this JSON object (no markdown, no extra text):
{
  "observations": "one short sentence describing what is actually on the bed and the state of the print",
  "failure_type": "none",
  "confidence": 0.0,
  "description": "short reason for your verdict"
}

"failure_type" MUST be EXACTLY ONE of these single tokens — never a list, never multiple, never the word "or":
  none               print looks fine / in progress / you are not sure  (THIS IS THE DEFAULT)
  spaghetti          a large chaotic tangle or nest of loose filament strands in the air or piled across the bed
  detached           the whole print (or a large chunk) has clearly broken free of the bed — dragged around, flipped, or stuck to the nozzle
  layer_shift        the print body is visibly stepped/offset sideways partway up its height
  stopped_extrusion  a large region that should be solid is clearly empty / no material is coming out while it should be
  warping            corners have curled UP off the bed badly enough to risk a nozzle crash
  no_printer         no printer / bed / print is visible in the frame

HARD RULES to avoid false alarms:
- Default to "none". Choose a failure ONLY when it is obvious and severe.
- A print sitting on the bed is NOT "detached", even at a steep/upside-down camera angle, even if it looks small, short, or oddly shaped. "detached" requires the part to be visibly OFF the bed or being dragged around.
- Supports, infill grids, brims, skirts, and normal layer lines are NOT failures.
- A few thin strings/wisps are NOT spaghetti — spaghetti is a big chaotic mess of loose filament.
- The nozzle hovering above or beside the print (a travel move) is NOT stopped_extrusion.
- Minor first-layer roughness or slight corner lift is NOT warping.
- If unsure, answer "none".

"confidence" is how certain you are of the chosen failure_type. For "none", use 0.0–0.3.
Only choose a non-"none" failure_type when your confidence is >= 0.85."""


@dataclass
class AnalysisResult:
    failure_detected: bool
    failure_type: str
    confidence: float
    description: str
    backend: str = "unknown"

    @property
    def summary(self) -> str:
        if not self.failure_detected:
            return f"OK ({self.confidence:.0%} confidence) [{self.backend}]"
        return f"FAILURE: {self.failure_type} ({self.confidence:.0%} confidence) [{self.backend}]"


# ------------------------------------------------------------------ #
# Shared helpers                                                        #
# ------------------------------------------------------------------ #

# Pre-check thresholds — tune these if you get false rejections
_MIN_BRIGHTNESS = 18.0   # mean pixel value (0-255); below = too dark
_MIN_CONTRAST   = 10.0   # std-dev of pixel values; below = too uniform
_MIN_EDGE_FRAC  = 0.01   # fraction of pixels that are edges; below = featureless


def _precheck_frame(frame: np.ndarray, backend: str) -> Optional["AnalysisResult"]:
    """Fast OpenCV sanity checks run *before* sending to the LLM.

    Catches obvious non-usable frames (lens cap, lights off, blank wall)
    without spending an API call or GPU time.  Returns an AnalysisResult
    describing the problem if the frame should be skipped, or None if it
    looks usable.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Brightness — too dark to see anything
    brightness = float(gray.mean())
    if brightness < _MIN_BRIGHTNESS:
        return AnalysisResult(
            False, "no_printer", 0.0,
            f"Frame too dark (brightness {brightness:.0f}/255) — "
            "check lighting or camera connection.",
            backend,
        )

    # 2. Contrast — frame is nearly a solid colour (lens cap, blank wall)
    contrast = float(gray.std())
    if contrast < _MIN_CONTRAST:
        return AnalysisResult(
            False, "no_printer", 0.0,
            f"Frame has almost no detail (contrast {contrast:.0f}) — "
            "camera may be covered or aimed at a blank surface.",
            backend,
        )

    # 3. Edge density — so few edges the scene is almost certainly not a printer
    edges = cv2.Canny(gray, threshold1=40, threshold2=120)
    edge_frac = float(edges.mean()) / 255.0
    if edge_frac < _MIN_EDGE_FRAC:
        return AnalysisResult(
            False, "no_printer", 0.0,
            f"Frame has very few edges ({edge_frac:.3%}) — "
            "camera may not be aimed at the printer.",
            backend,
        )

    return None   # frame looks usable — proceed to LLM


# Minimum confidence for a non-"none" verdict to count as a real failure.
_FAILURE_FLOOR = 0.85

# Map the model's clean single-token failure_type to the display names the rest
# of the app expects. Anything not in this map (compound junk, the echoed option
# list, hallucinated values) is treated as "none" — no false alarm.
_TYPE_MAP = {
    "none": "none",
    "spaghetti": "spaghetti/stringing",
    "detached": "detached from bed",
    "layer_shift": "layer shift",
    "stopped_extrusion": "stopped extrusion",
    "warping": "warping",
    "no_printer": "no_printer",
}


def _encode_frame_b64(frame: np.ndarray, quality: int = 90) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def _normalize_type(raw: str) -> str:
    """Reduce the model's failure_type to a single known token, else 'none'.

    Defends against the model returning a pipe-separated list or the entire
    option list verbatim — we take the first recognized token, but only if the
    string is a clean single token; a list-like answer collapses to 'none'.
    """
    t = (raw or "").strip().lower()
    if t in _TYPE_MAP:
        return _TYPE_MAP[t]
    # If it looks like a list/multiple values, the model didn't pick one → none.
    if "|" in t or "," in t or " or " in t or t.count(" ") > 2:
        return "none"
    # Last resort: substring match to a single known failure word.
    for token, display in _TYPE_MAP.items():
        if token != "none" and token in t:
            return display
    return "none"


def _parse_response(text: str, backend: str) -> AnalysisResult:
    """Extract and parse the JSON blob from a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return AnalysisResult(False, "none", 0.0, f"No JSON in response: {text[:120]}", backend)
    try:
        data = json.loads(match.group())
        confidence = float(data.get("confidence", 0.0))
        ftype = _normalize_type(str(data.get("failure_type", "none")))

        # A real failure = a recognized failure type with high confidence.
        failure_detected = ftype not in ("none", "no_printer") and confidence >= _FAILURE_FLOOR
        resolved_type = ftype if (failure_detected or ftype == "no_printer") else "none"

        # Prefer the model's reason; fall back to its observations.
        desc = str(data.get("description") or data.get("observations") or "")
        return AnalysisResult(
            failure_detected=failure_detected,
            failure_type=resolved_type,
            confidence=confidence,
            description=desc,
            backend=backend,
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return AnalysisResult(False, "none", 0.0, f"Parse error ({exc}): {text[:120]}", backend)


# ------------------------------------------------------------------ #
# Anthropic backend                                                    #
# ------------------------------------------------------------------ #

class AnthropicAnalyzer:
    """Uses claude-sonnet-4-6 via the Anthropic API."""

    def __init__(self, api_key: str, model: str = ANTHROPIC_MODEL) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._cache_logged = False
        self._prev_frame: Optional[np.ndarray] = None   # for temporal comparison

    def _image_block(self, frame: np.ndarray) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _encode_frame_b64(frame, quality=90),
            },
        }

    def analyze_frame(self, frame: np.ndarray) -> AnalysisResult:
        early = _precheck_frame(frame, backend=f"anthropic/{self._model}")
        if early:
            self._prev_frame = frame.copy()
            return early

        # Temporal context: show the previous analysis frame (~30 s ago) before
        # the current one. A real failure persists or worsens between frames; a
        # print that looks normal in both is normal. This sharply cuts single-
        # frame misreads (e.g. an attached print called "detached" from one odd
        # angle).
        content = []
        if self._prev_frame is not None:
            content.append({"type": "text",
                            "text": "FRAME 1 — about 30 seconds ago (for comparison only):"})
            content.append(self._image_block(self._prev_frame))
            content.append({"type": "text",
                            "text": "FRAME 2 — NOW. Judge this current frame. A genuine "
                                    "failure usually persists or gets worse from FRAME 1 to "
                                    "FRAME 2; if the print looks normal in both, answer none."})
            content.append(self._image_block(frame))
        else:
            content.append({"type": "text", "text": "Current printer frame:"})
            content.append(self._image_block(frame))
        content.append({"type": "text",
                        "text": "Respond with the JSON object only."})

        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            # Prompt caching: the system prompt is identical on every call,
            # so we mark it cache-eligible. Anthropic charges full price to
            # write it to cache once (1.25×), then ~0.1× on every cache hit
            # for the 5-minute TTL — a large saving at one call per 30 s.
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": content}],
        )
        self._prev_frame = frame.copy()
        # One-time confirmation that prompt caching is active. After the
        # first cache write, subsequent calls should show cache_read > 0.
        if not self._cache_logged:
            u = getattr(response, "usage", None)
            if u is not None:
                write = getattr(u, "cache_creation_input_tokens", 0) or 0
                read = getattr(u, "cache_read_input_tokens", 0) or 0
                if write or read:
                    print(f"  [CACHE] prompt caching active "
                          f"(write={write}, read={read} tokens)")
                    self._cache_logged = True

        raw_text = next((b.text for b in response.content if b.type == "text"), "{}")
        return _parse_response(raw_text, backend=f"anthropic/{self._model}")


# ------------------------------------------------------------------ #
# Ollama backend                                                        #
# ------------------------------------------------------------------ #

class OllamaAnalyzer:
    """Uses a local vision model via Ollama.

    Recommended for 8GB M2 MacBook Air: llava:7b (~4.1 GB, runs on Metal).
    Avoid llama3.2-vision:11b on 8GB — it will swap heavily and be very slow.

    Pull the model first:  ollama pull llava:7b
    """

    def __init__(self, model: str = OLLAMA_DEFAULT_MODEL, host: str = "http://localhost:11434") -> None:
        try:
            import ollama
            self._ollama = ollama
        except ImportError as exc:
            raise ImportError(
                "Ollama Python package not installed. Run: pip install ollama"
            ) from exc

        self._model = model
        self._host = host
        # Override host if non-default
        if host != "http://localhost:11434":
            import os
            os.environ.setdefault("OLLAMA_HOST", host)

    def analyze_frame(self, frame: np.ndarray) -> AnalysisResult:
        early = _precheck_frame(frame, backend=f"ollama/{self._model}")
        if early:
            return early
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buf.tobytes()

        # Combine system prompt into the user message — many llava builds
        # ignore a separate system role, so embedding it is more reliable.
        prompt = f"{SYSTEM_PROMPT}\n\nAnalyze this 3D printer image for failures. Respond with the JSON object only."

        response = self._ollama.chat(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }
            ],
        )
        raw_text = response["message"]["content"]
        return _parse_response(raw_text, backend=f"ollama/{self._model}")


# ------------------------------------------------------------------ #
# Factory — call this instead of instantiating directly               #
# ------------------------------------------------------------------ #

def create_analyzer(
    backend: str = "anthropic",
    anthropic_api_key: str = "",
    anthropic_model: str = ANTHROPIC_MODEL,
    ollama_model: str = OLLAMA_DEFAULT_MODEL,
    ollama_host: str = "http://localhost:11434",
):
    """Return the right analyzer based on the configured backend.

    backend: "anthropic" | "ollama"
    """
    backend = backend.lower().strip()
    if backend == "ollama":
        return OllamaAnalyzer(model=ollama_model, host=ollama_host)
    if backend == "anthropic":
        if not anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when using the anthropic backend.")
        return AnthropicAnalyzer(api_key=anthropic_api_key, model=anthropic_model)
    raise ValueError(f"Unknown ANALYZER_BACKEND '{backend}'. Choose 'anthropic' or 'ollama'.")


# Legacy alias so existing code that does `PrintAnalyzer(api_key=...)` still works.
class PrintAnalyzer(AnthropicAnalyzer):
    pass
