
"""
Local speech-to-text transcription using faster-whisper.

Responsibilities:
1. Read local model settings from environment variables.
2. Load and cache the Whisper model.
3. Transcribe an audio file locally.
4. Return a consistent result structure for the transcription service.
5. Log processing status and errors without logging transcript content.
"""

import logging
import os
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from faster_whisper import WhisperModel


# Load settings from .env when running locally.
# Environment variables configured by a hosting platform also work.
load_dotenv()

# Create this module's logger.
# The application's logging configuration controls its output.
logger = logging.getLogger(__name__)


# Read local model configuration from environment variables.
# Defaults are chosen for CPU-based development on a typical computer.
LOCAL_WHISPER_MODEL = os.getenv(
    "LOCAL_WHISPER_MODEL", "tiny"
).strip()

LOCAL_WHISPER_DEVICE = os.getenv(
    "LOCAL_WHISPER_DEVICE", "cpu"
).strip().lower()

LOCAL_WHISPER_COMPUTE_TYPE = os.getenv(
    "LOCAL_WHISPER_COMPUTE_TYPE", "int8"
).strip().lower()

LOCAL_WHISPER_CPU_THREADS = int(
    os.getenv("LOCAL_WHISPER_CPU_THREADS", "4")
)

LOCAL_WHISPER_BEAM_SIZE = int(
    os.getenv("LOCAL_WHISPER_BEAM_SIZE", "3")
)

# Parse the VAD setting as a boolean.
# This avoids treating the string "false" as True.
LOCAL_WHISPER_VAD_FILTER = (
    os.getenv("LOCAL_WHISPER_VAD_FILTER", "true").strip().lower()
    in {"true", "1", "yes", "on"}
)


def validate_configuration() -> None:
    """Check local settings before loading the model."""

    if not LOCAL_WHISPER_MODEL:
        raise ValueError("LOCAL_WHISPER_MODEL cannot be empty.")

    if LOCAL_WHISPER_DEVICE not in {"cpu", "cuda", "auto"}:
        raise ValueError(
            "LOCAL_WHISPER_DEVICE must be cpu, cuda, or auto."
        )

    if LOCAL_WHISPER_CPU_THREADS < 1:
        raise ValueError("LOCAL_WHISPER_CPU_THREADS must be at least 1.")

    if LOCAL_WHISPER_BEAM_SIZE < 1:
        raise ValueError("LOCAL_WHISPER_BEAM_SIZE must be at least 1.")


@lru_cache(maxsize=1)
def load_model() -> WhisperModel:
    """
    Load the local Whisper model once and reuse it.

    The first load may download model files if they are not cached.
    Caching avoids reloading the model for every transcription request.
    """

    validate_configuration()

    logger.info(
        "Loading local Whisper model: model=%s device=%s "
        "compute_type=%s cpu_threads=%d",
        LOCAL_WHISPER_MODEL,
        LOCAL_WHISPER_DEVICE,
        LOCAL_WHISPER_COMPUTE_TYPE,
        LOCAL_WHISPER_CPU_THREADS,
    )

    start_time = time.perf_counter()

    try:
        # Initialise faster-whisper with the configured model and hardware.
        model = WhisperModel(
            LOCAL_WHISPER_MODEL,
            device=LOCAL_WHISPER_DEVICE,
            compute_type=LOCAL_WHISPER_COMPUTE_TYPE,
            cpu_threads=LOCAL_WHISPER_CPU_THREADS,
        )

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Local Whisper model loaded successfully: "
            "model=%s elapsed_seconds=%.2f",
            LOCAL_WHISPER_MODEL,
            elapsed,
        )

        return model

    except Exception:
        # Log the traceback for debugging, then let the caller handle failure.
        logger.exception(
            "Failed to load local Whisper model: model=%s device=%s",
            LOCAL_WHISPER_MODEL,
            LOCAL_WHISPER_DEVICE,
        )
        raise


def transcribe_locally(audio_path: str) -> dict:
    """
    Transcribe an audio file using the local faster-whisper model.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dictionary containing:
        provider, model, language, transcript, and segments.

        Each segment contains start time, end time, and recognised text.

    Raises:
        FileNotFoundError: The audio file does not exist.
        ValueError: The audio file is empty.
        RuntimeError: Local model loading or transcription fails.
    """

    path = Path(audio_path)

    # Validate the input before loading the model.
    if not path.is_file():
        logger.error("Local transcription file not found: %s", path.name)
        raise FileNotFoundError(f"Audio file not found: {path}")

    file_size = path.stat().st_size

    if file_size == 0:
        logger.error("Local transcription file is empty: %s", path.name)
        raise ValueError("The audio file is empty.")

    logger.info(
        "Starting local transcription: file=%s size_mb=%.2f model=%s",
        path.name,
        file_size / (1024 * 1024),
        LOCAL_WHISPER_MODEL,
    )

    start_time = time.perf_counter()

    try:
        # Retrieve the cached model or load it on the first request.
        model = load_model()

        # transcribe() returns an iterator of segments and audio metadata.
        # vad_filter can skip detected non-speech regions.
        segments_iterator, info = model.transcribe(
            str(path),
            beam_size=LOCAL_WHISPER_BEAM_SIZE,
            vad_filter=LOCAL_WHISPER_VAD_FILTER,
        )

        segments = []

        # Iterate through the result to actually run transcription.
        # The model processes the audio as this iterator is consumed.
        for segment in segments_iterator:
            segments.append({
                "start": round(float(segment.start), 2),
                "end": round(float(segment.end), 2),
                "text": segment.text.strip(),
            })

        # Combine individual segment texts into a single transcript.
        transcript = " ".join(
            item["text"] for item in segments
        ).strip()

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Local transcription succeeded: file=%s "
            "elapsed_seconds=%.2f segments=%d language=%s",
            path.name,
            elapsed,
            len(segments),
            info.language,
        )

        # Keep the output format consistent with the Groq module.
        return {
            "provider": "Local faster-whisper",
            "model": LOCAL_WHISPER_MODEL,
            "language": info.language,
            "transcript": transcript,
            "segments": segments,
        }

    except (FileNotFoundError, ValueError):
        # Input validation errors should retain their original type.
        raise

    except Exception as exc:
        logger.exception(
            "Local transcription failed: file=%s model=%s",
            path.name,
            LOCAL_WHISPER_MODEL,
        )

        # Raise a clearer error for the coordinator, preserving the cause.
        raise RuntimeError(
            f"Local faster-whisper transcription failed "
            f"({type(exc).__name__})."
        ) from exc
