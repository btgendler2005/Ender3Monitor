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

SYSTEM_PROMPT = """You are an expert 3D printing failure detection system. Analyze images from a 3D printer webcam and detect print failures.

STEP 1 — VALIDATE THE SCENE
First, determine whether the image actually shows a 3D printer or an active 3D print.
If you do NOT see a 3D printer, print bed, extruder, filament, or a part being printed,
respond immediately with:
{"failure_detected": false, "failure_type": "no_printer", "confidence": 0.0, "description": "No 3D printer detected in frame. Please aim the camera at your printer."}

STEP 2 — ANALYZE FOR FAILURES (only if a printer is visible)
You must respond ONLY with a valid JSON object in this exact format:
{
  "failure_detected": true or false,
  "failure_type": "one of: spaghetti/stringing, layer shift, detached from bed, stopped extrusion, nozzle collision, warping, no_printer, or none",
  "confidence": 0.0 to 1.0,
  "description": "brief description of what you see"
}

Failure types:
- spaghetti/stringing: filament extruded randomly creating messy strands
- layer shift: layers misaligned, print shifted in X or Y direction
- detached from bed: print has lifted or detached from the print bed
- stopped extrusion: print head moving but no filament being deposited (gaps/missing material)
- nozzle collision: evidence of nozzle hitting and knocking the print
- warping: corners/edges lifting from the bed
- none: print appears to be progressing normally
- no_printer: image does not show a 3D printer

Set failure_detected to true only when confidence >= 0.5. Be conservative — false negatives are better than false positives.
Respond with the JSON object only. No markdown, no explanation."""


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
        failure_detected = bool(data.get("failure_detected", False)) and confidence >= 0.5
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
