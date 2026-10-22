
"""Visual analysis of classroom video frames using the Gemini API."""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Find the project root from this file: project_root/src/video_analyzer_gemini.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("GEMINI_VISION_MODEL", "gemini-3.8-flash")
MAX_IMAGE_DIMENSION = int(os.getenv("GEMINI_MAX_IMAGE_DIMENSION", "1024"))


VISUAL_ANALYSIS_PROMPT = """
You are analyzing sampled frames from a classroom lesson.

Your job is to report observable visual evidence, not to guess whether
the teacher is effective or whether students are learning.

Analyze the frames together, in the order provided. Each frame is labelled
with its timestamp in the original recording.

Return valid JSON with exactly these top-level keys:

{
  "visible_scene": "Brief description of what is visibly happening",
  "teacher_visibility": {
    "status": "visible | not_visible | uncertain",
    "evidence": "What can actually be seen"
  },
  "teaching_materials": {
    "status": "visible | not_visible | uncertain",
    "evidence": "Board, slides, books, demonstrations, or other materials"
  },
  "classroom_setup": {
    "evidence": "Observable layout, camera framing, and visibility"
  },
  "observable_activity": {
    "evidence": "Visible actions only; do not infer attention or understanding"
  },
  "technical_issues": [
    "Clearly visible issues such as a black frame, obstruction, or poor framing"
  ],
  "evidence_by_frame": [
    {
      "timestamp_seconds": 0,
      "observation": "Specific visual evidence from this frame"
    }
  ],
  "limitations": [
    "What cannot reliably be concluded from these sampled frames"
  ]
}

Rules:
- Do not infer learning, engagement, understanding, or teaching quality from
  appearance alone.
- Do not identify people or infer age, identity, emotions, or other attributes.
- Do not assume students are absent just because they are not visible.
- Distinguish "not visible in the sampled frames" from "not present".
- Do not invent timestamps or visual details.
- If evidence is unclear, say so.
- Return JSON only, without Markdown fences.
"""


def _resolve_frame_path(frame_path: str | Path) -> Path:
    """Resolve absolute paths and project-relative paths."""
    path = Path(frame_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(f"Frame not found: {path}")

    return path


def _prepare_image(frame_path: str | Path) -> bytes:
    """Resize a frame if needed and return JPEG bytes."""
    path = _resolve_frame_path(frame_path)

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail(
            (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
            Image.Resampling.LANCZOS,
        )

        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85, optimize=True)
        return buffer.getvalue()


def analyze_video_frames(frames: list[dict]) -> dict:
    """
    Analyze FFmpeg-extracted frames.

    Expected input:
        [
            {
                "path": "data/frames/lesson/frame_001.jpg",
                "timestamp_seconds": 1.0
            },
            ...
        ]

    Returns a dictionary containing structured visual observations.
    """
    if not frames:
        raise ValueError("No video frames were provided for visual analysis.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "put_your_actual_api_key_here":
        raise ValueError("Set a valid GEMINI_API_KEY in your .env file.")

    client = genai.Client(api_key=api_key)

    try:
        contents = [
            types.Part.from_text(
                text=VISUAL_ANALYSIS_PROMPT
                + "\nNumber of sampled frames: "
                + str(len(frames))
            )
        ]

        # Add each frame with its original timestamp.
        for index, frame in enumerate(frames, start=1):
            timestamp = float(
                frame.get("timestamp_seconds", frame.get("timestamp", 0))
            )

            contents.append(
                types.Part.from_text(
                    text=f"Frame {index}; original timestamp: "
                    f"{timestamp:.2f} seconds."
                )
            )

            image_bytes = _prepare_image(frame["path"])
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                )
            )

        logger.info(
            "Sending %d frames to Gemini model %s",
            len(frames),
            MODEL_NAME,
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        observations = json.loads(response.text)

        return {
            "status": "success",
            "provider": "gemini",
            "model": MODEL_NAME,
            "frame_count": len(frames),
            "observations": observations,
        }

    except Exception:
        logger.exception("Gemini visual analysis failed.")
        raise

    finally:
        client.close()
