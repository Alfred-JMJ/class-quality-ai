
"""
Audio quality analysis for the class-quality evaluator.

Measures basic recording characteristics:
- Duration and audio format
- Overall loudness
- Peak level
- Potential clipping
- Proportion of low-volume audio frames

Important:
Low-volume frames are NOT necessarily silence, and this module
does not identify speech or speakers. These measurements are
evidence about the recording, not a class-quality score.
"""

import logging
import os
import time
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv


# Load configuration from .env during local development.
load_dotenv()

logger = logging.getLogger(__name__)


# Audio analysis settings, configurable through .env.
FRAME_DURATION_MS = float(
    os.getenv("AUDIO_FRAME_DURATION_MS", "20")
)

LOW_LEVEL_THRESHOLD_DBFS = float(
    os.getenv("AUDIO_LOW_LEVEL_THRESHOLD_DBFS", "-45")
)

CLIPPING_THRESHOLD = float(
    os.getenv("AUDIO_CLIPPING_THRESHOLD", "0.999")
)


def dbfs(amplitude: float) -> float:
    """Convert a normalized amplitude into dBFS."""
    if amplitude <= 0:
        return float("-inf")

    return float(20 * np.log10(amplitude))


def analyze_audio(audio_path: str) -> dict:
    """
    Analyse a PCM WAV audio file.

    The expected input is the WAV produced by media_processor.py.
    Audio is analysed in small frames to measure changes over time.

    Returns basic audio metrics. It does not determine teaching quality.
    """

    path = Path(audio_path)

    # Validate the input before reading audio data.
    if not path.is_file():
        logger.error("Audio analysis input not found: %s", path.name)
        raise FileNotFoundError(f"Audio file not found: {path}")

    if FRAME_DURATION_MS <= 0:
        raise ValueError("AUDIO_FRAME_DURATION_MS must be positive.")

    if not -100 <= LOW_LEVEL_THRESHOLD_DBFS <= 0:
        raise ValueError(
            "AUDIO_LOW_LEVEL_THRESHOLD_DBFS must be between -100 and 0."
        )

    if not 0 < CLIPPING_THRESHOLD <= 1:
        raise ValueError(
            "AUDIO_CLIPPING_THRESHOLD must be greater than 0 and at most 1."
        )

    start_time = time.perf_counter()

    logger.info("Starting audio analysis: file=%s", path.name)

    try:
        # The media pipeline should provide PCM WAV audio.
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            audio_bytes = wav.readframes(frame_count)

        # This initial implementation expects 16-bit PCM WAV,
        # matching the audio extracted by media_processor.py.
        if sample_width != 2:
            raise ValueError(
                "Expected 16-bit PCM WAV. Extract audio using "
                "media_processor.py before analysis."
            )

        if frame_count == 0 or sample_rate <= 0:
            raise ValueError("Audio file contains no usable samples.")

        # Convert the PCM samples to normalized floating-point values
        # in the range approximately -1.0 to +1.0.
        samples = np.frombuffer(audio_bytes, dtype=np.int16)
        samples = samples.astype(np.float32) / 32768.0

        # Combine stereo channels into one mono signal for analysis.
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        duration_seconds = len(samples) / sample_rate

        # Calculate overall loudness and peak amplitude.
        overall_rms = float(np.sqrt(np.mean(np.square(samples))))
        peak_amplitude = float(np.max(np.abs(samples)))

        overall_rms_dbfs = dbfs(overall_rms)
        peak_dbfs = dbfs(peak_amplitude)

        # Split the signal into short frames to observe volume changes.
        samples_per_frame = max(
            1,
            int(sample_rate * FRAME_DURATION_MS / 1000),
        )

        frame_rms_values = []

        for start in range(0, len(samples), samples_per_frame):
            frame = samples[start:start + samples_per_frame]

            if len(frame) == 0:
                continue

            frame_rms = float(np.sqrt(np.mean(np.square(frame))))
            frame_rms_values.append(frame_rms)

        frame_rms_values = np.asarray(
            frame_rms_values,
            dtype=np.float32,
        )

        # Estimate the percentage of frames below a configured level.
        # This is a low-level audio estimate, NOT speech detection.
        low_level_threshold = 10 ** (
            LOW_LEVEL_THRESHOLD_DBFS / 20
        )

        low_level_frames = frame_rms_values < low_level_threshold
        low_level_ratio = float(np.mean(low_level_frames))

        # Estimate how much of the audio is close to full digital scale.
        # This indicates possible clipping, but does not prove distortion.
        clipping_ratio = float(
            np.mean(np.abs(samples) >= CLIPPING_THRESHOLD)
        )

        elapsed = time.perf_counter() - start_time

        result = {
            "duration_seconds": round(duration_seconds, 2),
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "overall_rms_dbfs": round(overall_rms_dbfs, 2),
            "peak_dbfs": round(peak_dbfs, 2),
            "low_level_audio_ratio": round(low_level_ratio, 4),
            "clipping_sample_ratio": round(clipping_ratio, 6),
            "frame_duration_ms": FRAME_DURATION_MS,
            "low_level_threshold_dbfs": LOW_LEVEL_THRESHOLD_DBFS,
        }

        logger.info(
            "Audio analysis completed: file=%s duration_seconds=%.2f "
            "rms_dbfs=%.2f peak_dbfs=%.2f elapsed_seconds=%.2f",
            path.name,
            duration_seconds,
            overall_rms_dbfs,
            peak_dbfs,
            elapsed,
        )

        return result

    except Exception:
        logger.exception("Audio analysis failed: file=%s", path.name)
        raise



