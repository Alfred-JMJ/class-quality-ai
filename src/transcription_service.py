
"""
Transcription service coordinator.

Responsibilities:
1. Try Groq Whisper as the primary transcription provider.
2. Fall back to local faster-whisper when Groq fails.
3. Allow fallback behaviour to be configured through .env.
4. Return a consistent result structure to the application.
5. Log provider selection, fallback events, and processing duration.

This module coordinates the providers; the actual transcription logic
remains in transcription_groq.py and transcription.py.
"""

import logging
import os
import time

from dotenv import load_dotenv

from src.transcription_groq import transcribe_with_groq
from src.transcription import transcribe_locally


# Load environment variables from .env during local development.
# Environment variables set by the deployment platform are also supported.
load_dotenv()

# Create a logger for this module.
# The application should configure the log format and handlers.
logger = logging.getLogger(__name__)


def read_boolean_setting(name: str, default: bool) -> bool:
    """
    Read a boolean environment variable safely.

    Accepted true values: true, 1, yes, on
    Accepted false values: false, 0, no, off

    Invalid values raise an error rather than silently choosing a behaviour.
    """

    value = os.getenv(name)

    # Use the supplied default when the variable is not configured.
    if value is None:
        return default

    value = value.strip().lower()

    if value in {"true", "1", "yes", "on"}:
        return True

    if value in {"false", "0", "no", "off"}:
        return False

    raise ValueError(
        f"Invalid value for {name}. Use true or false."
    )


# Read the fallback setting once when this module is imported.
# Restart the application after changing this setting in .env.
TRANSCRIPTION_FALLBACK_ENABLED = read_boolean_setting(
    "TRANSCRIPTION_FALLBACK_ENABLED",
    default=True,
)


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using Groq first and local Whisper as fallback.

    Args:
        audio_path: Path to the extracted audio file.

    Returns:
        A dictionary containing:
        - provider: transcription provider used
        - model: model name
        - language: detected language, if available
        - transcript: complete recognised text
        - segments: timestamped transcript segments
        - fallback_used: whether the local model was used
        - groq_error: Groq failure type, if fallback was needed

    Raises:
        FileNotFoundError: The input audio file does not exist.
        RuntimeError: Both providers fail, or fallback is disabled.
        ValueError: Input or configuration is invalid.
    """

    start_time = time.perf_counter()

    logger.info(
        "Transcription requested: fallback_enabled=%s",
        TRANSCRIPTION_FALLBACK_ENABLED,
    )

    # Attempt the primary provider first.
    try:
        logger.info("Attempting primary provider: Groq")

        result = transcribe_with_groq(audio_path)

        # Add common metadata so the UI can identify the selected provider.
        result["fallback_used"] = False
        result["groq_error"] = None

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Transcription completed successfully: provider=Groq "
            "elapsed_seconds=%.2f",
            elapsed,
        )

        return result

    except FileNotFoundError:
        # A missing input file is not a provider failure.
        # Trying another model cannot fix an invalid path.
        logger.exception("Transcription stopped: input file not found.")
        raise

    except Exception as groq_exc:
        # Record the failure type without logging audio or transcript content.
        groq_error_type = type(groq_exc).__name__

        logger.warning(
            "Primary transcription failed: provider=Groq "
            "error_type=%s",
            groq_error_type,
        )

        # If fallback is disabled, report the primary-provider failure.
        if not TRANSCRIPTION_FALLBACK_ENABLED:
            logger.error(
                "Local fallback is disabled; transcription cannot continue."
            )
            raise RuntimeError(
                "Groq transcription failed and local fallback is disabled. "
                f"Failure type: {groq_error_type}."
            ) from groq_exc

    # Run the local model only if Groq failed and fallback is enabled.
    try:
        logger.info("Starting fallback provider: local faster-whisper")

        result = transcribe_locally(audio_path)

        # Use the same output structure as the primary provider.
        result["fallback_used"] = True
        result["groq_error"] = groq_error_type

        elapsed = time.perf_counter() - start_time

        logger.info(
            "Transcription completed successfully: "
            "provider=Local faster-whisper fallback_used=true "
            "elapsed_seconds=%.2f",
            elapsed,
        )

        return result

    except Exception as local_exc:
        # Both providers have failed. Log the error types, not private content.
        local_error_type = type(local_exc).__name__

        logger.exception(
            "Transcription failed with both providers: "
            "groq_error_type=%s local_error_type=%s",
            groq_error_type,
            local_error_type,
        )

        raise RuntimeError(
            "Both transcription providers failed. "
            f"Groq failure type: {groq_error_type}; "
            f"local failure type: {local_error_type}."
        ) from local_exc
