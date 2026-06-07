import base64
import json
import re
import cv2
import numpy as np
import anthropic
from dataclasses import dataclass
from typing import Optional

# claude-sonnet-4-6 is the current model ID for what was formerly called claude-sonnet-4-20250514
MODEL_ID = "claude-sonnet-4-6"

FAILURE_TYPES = [
    "spaghetti/stringing",
    "layer shift",
    "detached from bed",
    "stopped extrusion",
    "nozzle collision",
    "warping",
    "none",
]

SYSTEM_PROMPT = """You are an expert 3D printing failure detection system. Analyze images from a 3D printer webcam and detect print failures.

You must respond ONLY with a valid JSON object in this exact format:
{
  "failure_detected": true or false,
  "failure_type": "one of: spaghetti/stringing, layer shift, detached from bed, stopped extrusion, nozzle collision, warping, or none",
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

Set failure_detected to true only when confidence >= 0.5. Be conservative — false negatives are better than false positives.
"""


@dataclass
class AnalysisResult:
    failure_detected: bool
    failure_type: str
    confidence: float
    description: str

    @property
    def summary(self) -> str:
        if not self.failure_detected:
            return f"OK ({self.confidence:.0%} confidence)"
        return f"FAILURE: {self.failure_type} ({self.confidence:.0%} confidence)"


def _encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


class PrintAnalyzer:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def analyze_frame(self, frame: np.ndarray) -> AnalysisResult:
        image_data = _encode_frame(frame)

        response = self.client.messages.create(
            model=MODEL_ID,
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

        raw_text = next(
            (b.text for b in response.content if b.type == "text"), "{}"
        )
        return _parse_response(raw_text)


def _parse_response(text: str) -> AnalysisResult:
    # Extract JSON even if the model wraps it in markdown fences
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return AnalysisResult(False, "none", 0.0, "Failed to parse response")
    try:
        data = json.loads(match.group())
        confidence = float(data.get("confidence", 0.0))
        failure_type = str(data.get("failure_type", "none"))
        failure_detected = bool(data.get("failure_detected", False)) and confidence >= 0.5
        return AnalysisResult(
            failure_detected=failure_detected,
            failure_type=failure_type if failure_detected else "none",
            confidence=confidence,
            description=str(data.get("description", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return AnalysisResult(False, "none", 0.0, f"Parse error: {text[:120]}")
