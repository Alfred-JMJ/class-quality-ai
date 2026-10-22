
"""
Groq-based speech-to-text transcription.

Responsibilities:
1. Validate the input audio file.
2. Read the Groq API key from environment variables.
3. Send the audio to Groq Whisper.
4. Convert the response to a consistent dictionary.
5. Log processing status without logging private audio or transcript content.

"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from groq import APIConnectionError, APIStatusError


# Load variables from the project's .env file, if one exists.
# In deployment, environment variables can instead be configured by the host.
load_dotenv()

# Create a logger for this module.
# The application configures the logging format and destination.
logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL", "whisper-large-v3-turbo"
).strip()

GROQ_MAX_FILE_SIZE_MB = float(
    os.getenv("GROQ_MAX_FILE_SIZE_MB", "25")
)

GROQ_TIMEOUT_SECONDS = float(
    os.getenv("GROQ_TIMEOUT_SECONDS", "120")
)

# Convert MiB to bytes for comparison with the file's size.
MAX_GROQ_FILE_SIZE = int(
    GROQ_MAX_FILE_SIZE_MB * 1024 * 1024
)

def transcribe_with_groq(audio_path: str) -> dict:
    """
    Transcribe an audio file using Groq Whisper.

    Args:
        audio_path: Path to a local audio file.

    Returns:
        Dictionary containing provider, model, language, transcript,
        and timestamped segments.

    Raises:
        FileNotFoundError: The audio file does not exist.
        ValueError: The file is empty or exceeds the configured size limit.
        RuntimeError: The API key is missing or the API request fails.
    """

    start_time = time.perf_counter()
    path = Path(audio_path)

    # Validate the path before making an API request.
    if not path.is_file():
        logger.error("Transcription input file not found: %s", path.name)
        raise FileNotFoundError(f"Audio file not found: {path}")

    # An empty file cannot contain usable audio.
    file_size = path.stat().st_size
    if file_size == 0:
        logger.error("Audio file is empty: %s", path.name)
        raise ValueError("The audio file is empty.")

    # Reject oversized files instead of making a request that may fail.
    # Chunking large audio will be implemented in a separate step.
    if file_size > MAX_GROQ_FILE_SIZE:
        logger.warning(
            "Audio exceeds configured Groq limit: "
            "file=%s size_mb=%.2f limit_mb=%.2f",
            path.name,
            file_size / (1024 * 1024),
            GROQ_MAX_FILE_SIZE_MB,
        )
        raise ValueError(
            f"Audio exceeds the configured limit of "
            f"{GROQ_MAX_FILE_SIZE_MB:g} MiB."
        )

    # Read the API key securely from the environment.
    # Never write the key to application logs.
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("Groq API key is not configured.")
        raise RuntimeError(
            "GROQ_API_KEY is missing. Configure it in .env "
            "or in the deployment environment."
        )

    logger.info(
        "Starting Groq transcription: file=%s size_mb=%.2f model=%s",
        path.name,
        file_size / (1024 * 1024),
        GROQ_MODEL,
    )

    try:
        # Create the API client using the secret API key.
        client = Groq(api_key=api_key)

        # Open the audio file in binary mode and submit it for transcription.
        # verbose_json requests structured output, including timestamps when
        # supported by the model and API response.
        with path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=(path.name, audio_file),
                model=GROQ_MODEL,
                response_format="verbose_json",
                temperature=0.0,
                timeout=GROQ_TIMEOUT_SECONDS,
            )

        # Convert SDK response segments into plain dictionaries.
        # Some SDK response objects expose segment fields as attributes;
        # the helper below also supports dictionary-like segment values.
        segments = []

        for segment in (getattr(response, "segments", None) or []):
            if isinstance(segment, dict):
                segment_start = segment.get("start")
                segment_end = segment.get("end")
                segment_text = segment.get("text", "")
            else:
                segment_start = getattr(segment, "start", None)
                segment_end = getattr(segment, "end", None)
                segment_text = getattr(segment, "text", "")

            # Skip malformed segments rather than failing the whole response.
            if segment_start is None or segment_end is None:
                logger.warning("Skipping a segment with missing timestamps.")
                continue

            segments.append({
                "start": float(segment_start),
                "end": float(segment_end),
                "text": str(segment_text).strip(),
            })

        # Return the same field names as the local transcription module.
        # This lets the coordinator switch providers without changing the UI.
        result = {
            "provider": "Groq",
            "model": GROQ_MODEL,
            "language": getattr(response, "language", None),
            "transcript": (getattr(response, "text", "") or "").strip(),
            "segments": segments,
        }

        elapsed = time.perf_counter() - start_time

        # Log metadata only. Do not log the transcript or audio content.
        logger.info(
            "Groq transcription succeeded: file=%s elapsed_seconds=%.2f "
            "segments=%d",
            path.name,
            elapsed,
            len(segments),
        )

        return result

    except (APIConnectionError, APIStatusError) as exc:
        # Log the failure type and re-raise a safe message.
        # The coordinator can catch this exception and invoke local fallback.
        logger.exception(
            "Groq API request failed: file=%s error_type=%s",
            path.name,
            type(exc).__name__,
        )
        raise RuntimeError(
            f"Groq transcription request failed ({type(exc).__name__})."
        ) from exc

    except Exception:
        # Record unexpected failures with their traceback for debugging.
        # Avoid logging exception messages if they might contain sensitive data.
        logger.exception(
            "Unexpected error during Groq transcription: file=%s",
            path.name,
        )
        raise
