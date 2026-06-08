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

SYSTEM_PROMPT = """You are a 3D print failure detection system. You ONLY alert on severe, print-ruining failures that require immediate action. Minor issues, surface imperfections, and anything that would still produce an acceptable final part are NOT failures.

STEP 1 — VALIDATE THE SCENE
Does the image show a 3D printer, print bed, extruder, or an object being printed?
If NO printer is visible, respond with:
{"failure_detected": false, "failure_type": "no_printer", "confidence": 0.0, "description": "No 3D printer detected in frame."}

STEP 2 — CHECK FOR SPAGHETTI (highest priority, most common catastrophic failure)
Spaghetti = the print has completely failed and the nozzle is extruding freely into empty space,
producing a large chaotic tangle or nest of loose filament strands. This fills the air or piles
up randomly over a large area of the build volume.

SPAGHETTI — YES, flag it:
- A large messy bird's nest or pile of loose strands covering a significant portion of the build area
- Filament hanging in mid-air in a chaotic tangle across the full frame
- The intended print object is completely gone and replaced by random strands

SPAGHETTI — NO, do NOT flag it:
- A few fine strings or hairs between features (normal with PETG/PLA)
- Support structures (intentional thin pillars or lattice — attached and orderly)
- Infill patterns (grid, honeycomb, lines — regular and inside walls)
- Any texture that looks structured or intentional, even if it looks messy up close

STEP 3 — OTHER CATASTROPHIC FAILURES ONLY
Only flag if the failure has already destroyed or will imminently destroy the print:
- layer shift: the ENTIRE print body is offset — looks like a staircase or snapped sideways (NOT a surface line)
- detached from bed: the whole print or a large chunk has physically lifted off and is no longer adhered
- stopped extrusion: a large section of the print has completely missing walls or is hollow where solid was expected
- warping: edges have lifted SO severely the print is peeling off or curling dramatically — not slight corner lift

DO NOT FLAG:
- Small blobs, zits, or surface imperfections
- Slight corner lifting or elephant foot
- Minor stringing or wisps between parts
- Normal-looking layer lines or texture
- Anything you are not highly certain about

Respond ONLY with this JSON:
{
  "failure_detected": true or false,
  "failure_type": "spaghetti/stringing | layer shift | detached from bed | stopped extrusion | warping | none | no_printer",
  "confidence": 0.0 to 1.0,
  "description": "one sentence: exactly what you see that led to this classification"
}

CRITICAL RULES:
- When in doubt, respond with failure_detected=false and failure_type="none". Always.
- A normal print in progress — even a complex or messy-looking one — should be classified as "none".
- Only set failure_detected=true when the failure is so obvious and severe that a human watching would immediately stop the print.
- Set confidence=1.0 only when absolutely certain. Be conservative with your confidence score.

Respond with the JSON object only. No markdown, no extra text."""


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


def _encode_frame_b64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def _parse_response(text: str, backend: str) -> AnalysisResult:
    """Extract and parse the JSON blob from a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return AnalysisResult(False, "none", 0.0, f"No JSON in response: {text[:120]}", backend)
    try:
        data = json.loads(match.group())
        confidence = float(data.get("confidence", 0.0))
        failure_type = str(data.get("failure_type", "none"))
        failure_detected = bool(data.get("failure_detected", False)) and confidence >= 0.82
        # Preserve "no_printer" even when failure_detected is False —
        # it's a special sentinel, not a real failure, but we need it
        # to show the right status message in the UI.
        resolved_type = failure_type if (failure_detected or failure_type == "no_printer") else "none"
        return AnalysisResult(
            failure_detected=failure_detected,
            failure_type=resolved_type,
            confidence=confidence,
            description=str(data.get("description", "")),
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

    def analyze_frame(self, frame: np.ndarray) -> AnalysisResult:
        early = _precheck_frame(frame, backend=f"anthropic/{self._model}")
        if early:
            return early
        image_data = _encode_frame_b64(frame)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this 3D printer image for failures. Respond with the JSON object only.",
                        },
                    ],
                }
            ],
        )
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
