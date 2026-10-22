
"""
Media processing utilities for the Humaniti classroom assessment app.

Functions:
    - extract_audio(): Extract WAV audio and optionally create MP3.
    - extract_key_frames(): Extract a configurable number of video frames.
    - detect_silence(): Detect low-volume intervals without modifying audio.

Configuration:
    Processing settings are loaded from the project's .env file.
"""

import json
import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root, not from the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


# ============================================================
# Configuration helpers
# ============================================================

def _env_int(name: str, default: int) -> int:
    """Read an integer setting from .env."""
    try:
        return int(os.getenv(name, str(default)).strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def _env_float(name: str, default: float) -> float:
    """Read a floating-point setting from .env."""
    try:
        value = float(os.getenv(name, str(default)).strip())
        if not math.isfinite(value):
            raise ValueError
        return value
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc


def _env_bool(name: str, default: bool) -> bool:
    """Read a Boolean setting from .env."""
    value = os.getenv(name, str(default)).strip().lower()

    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be true/false, yes/no, on/off, or 1/0."
    )


def _env_string(name: str, default: str) -> str:
    """Read a non-empty string setting from .env."""
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} cannot be empty.")
    return value


def _get_executable(env_name: str, default: str) -> str:
    """Resolve an executable from .env or the system PATH."""
    configured_path = _env_string(env_name, default)

    # Accept either an executable name or an absolute path.
    if Path(configured_path).is_file():
        return str(Path(configured_path).resolve())

    resolved = shutil.which(configured_path)
    if resolved is None:
        raise RuntimeError(
            f"Cannot find {configured_path!r}, configured by {env_name}. "
            "Install FFmpeg or set the full executable path in .env."
        )

    return resolved


def _run_command(
    command: list[str],
    operation: str,
    timeout_seconds: int,
) -> subprocess.CompletedProcess:
    """Execute an external command with consistent logging/errors."""
    logger.info("Starting media operation: %s", operation)

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        logger.info("Completed media operation: %s", operation)
        return result

    except subprocess.TimeoutExpired as exc:
        logger.error(
            "%s timed out after %s seconds.",
            operation,
            timeout_seconds,
        )
        raise RuntimeError(
            f"{operation} timed out after {timeout_seconds} seconds."
        ) from exc

    except subprocess.CalledProcessError as exc:
        # FFmpeg diagnostics help identify codec and file problems.
        details = (exc.stderr or exc.stdout or "")[-3000:]
        logger.error("%s failed: %s", operation, details)
        raise RuntimeError(
            f"{operation} failed. FFmpeg output: {details}"
        ) from exc


def _validate_audio_configuration() -> dict:
    """Read and validate audio-related settings."""
    sample_rate = _env_int("AUDIO_OUTPUT_SAMPLE_RATE", 16000)
    channels = _env_int("AUDIO_OUTPUT_CHANNELS", 1)
    codec = _env_string("AUDIO_OUTPUT_CODEC", "pcm_s16le")

    if sample_rate <= 0:
        raise ValueError("AUDIO_OUTPUT_SAMPLE_RATE must be positive.")
    if channels not in {1, 2}:
        raise ValueError("AUDIO_OUTPUT_CHANNELS must be 1 or 2.")
    if codec != "pcm_s16le":
        raise ValueError(
            "This WAV extraction implementation supports "
            "AUDIO_OUTPUT_CODEC=pcm_s16le."
        )

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "codec": codec,
    }


# ============================================================
# Video metadata
# ============================================================

def _get_video_metadata(video_path: Path) -> dict:
    """Read duration and dimensions using FFprobe."""
    ffprobe = _get_executable("FFPROBE_BINARY", "ffprobe")
    timeout = _env_int("MEDIA_METADATA_TIMEOUT_SECONDS", 60)

    result = _run_command(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration:"
            "format=duration",
            "-of", "json",
            str(video_path),
        ],
        operation="reading video metadata",
        timeout_seconds=timeout,
    )

    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])

        if not streams:
            raise ValueError("No video stream was found.")

        stream = streams[0]
        duration_value = stream.get("duration")

        if duration_value is None:
            duration_value = data.get("format", {}).get("duration")

        duration = float(duration_value)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Video duration is invalid.")

        fps_value = stream.get("avg_frame_rate", "0/1")
        numerator, denominator = fps_value.split("/", maxsplit=1)
        denominator = float(denominator)
        fps = float(numerator) / denominator if denominator else 0.0

        return {
            "duration_seconds": round(duration, 3),
            "fps": round(fps, 3),
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
        }

    except (ValueError, TypeError, KeyError, ZeroDivisionError) as exc:
        raise RuntimeError(
            f"Could not interpret video metadata: {exc}"
        ) from exc


# ============================================================
# 1. Audio extraction
# ============================================================

def extract_audio(
    video_path: str,
    audio_path: str,
    compressed_audio_path: str | None = None,
) -> str:
    """
    Extract audio as WAV using settings from .env.

    If AUDIO_COMPRESSION_ENABLED=true, also create a compressed copy.
    If no compressed path is supplied, it is generated beside the WAV.

    Returns:
        Absolute path to the extracted WAV file.
    """
    video = Path(video_path).resolve()
    audio = Path(audio_path).resolve()

    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")

    if audio == video:
        raise ValueError("Audio output must not overwrite the video.")

    audio_config = _validate_audio_configuration()
    ffmpeg = _get_executable("FFMPEG_BINARY", "ffmpeg")
    timeout = _env_int("MEDIA_COMMAND_TIMEOUT_SECONDS", 600)

    audio.parent.mkdir(parents=True, exist_ok=True)

    if audio.suffix.lower() != ".wav":
        raise ValueError("audio_path must have a .wav extension.")

    logger.info("Extracting audio from video: %s", video.name)

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video),
        "-vn",
        "-map", "0:a:0",
        "-ac", str(audio_config["channels"]),
        "-ar", str(audio_config["sample_rate"]),
        "-c:a", audio_config["codec"],
        str(audio),
    ]

    _run_command(
        command,
        operation="extracting WAV audio",
        timeout_seconds=timeout,
    )

    if not audio.is_file() or audio.stat().st_size == 0:
        raise RuntimeError("FFmpeg produced an empty WAV file.")

    logger.info(
        "WAV created: %s (%.2f MB)",
        audio.name,
        audio.stat().st_size / (1024 * 1024),
    )

    # Compression is optional and never replaces the WAV.
    compression_enabled = _env_bool(
        "AUDIO_COMPRESSION_ENABLED", False
    )

    if compression_enabled:
        compressed_format = _env_string(
            "AUDIO_COMPRESSED_FORMAT", "mp3"
        ).lower()

        if compressed_format != "mp3":
            raise ValueError(
                "AUDIO_COMPRESSED_FORMAT currently supports only mp3."
            )

        bitrate = _env_string(
            "AUDIO_COMPRESSED_BITRATE", "48k"
        )
        if bitrate not in {"32k", "48k", "64k", "96k", "128k"}:
            raise ValueError(
                "AUDIO_COMPRESSED_BITRATE must be 32k, 48k, "
                "64k, 96k, or 128k."
            )

        compressed = (
            Path(compressed_audio_path).resolve()
            if compressed_audio_path
            else audio.with_suffix(".mp3")
        )

        if compressed in {video, audio}:
            raise ValueError(
                "Compressed audio must use a different output path."
            )

        if compressed.suffix.lower() != ".mp3":
            raise ValueError(
                "Compressed audio output path must end in .mp3."
            )

        compressed.parent.mkdir(parents=True, exist_ok=True)

        _run_command(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(audio),
                "-vn",
                "-c:a", "libmp3lame",
                "-b:a", bitrate,
                str(compressed),
            ],
            operation="creating compressed MP3 audio",
            timeout_seconds=timeout,
        )

        if not compressed.is_file() or compressed.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced an empty MP3 file.")

        logger.info(
            "Compressed MP3 created: %s (%.2f MB)",
            compressed.name,
            compressed.stat().st_size / (1024 * 1024),
        )

    return str(audio)


# ============================================================
# 2. Automatic representative frame extraction
# ============================================================

def extract_key_frames(
    video_path: str,
    output_dir: str | None = None,
) -> dict:
    """
    Extract up to VIDEO_MAX_FRAMES frames at evenly distributed times.

    Frame count, JPEG quality, output directory, and timeouts come
    from .env. These are time-based samples, not semantic keyframes.
    """
    video = Path(video_path).resolve()

    if not video.is_file():
        raise FileNotFoundError(f"Video file not found: {video}")

    max_frames = _env_int("VIDEO_MAX_FRAMES", 6)
    jpeg_quality = _env_int("VIDEO_JPEG_QUALITY", 3)
    margin_setting = _env_float(
        "VIDEO_FRAME_START_MARGIN_SECONDS", 1.0
    )

    if not 1 <= max_frames <= 8:
        raise ValueError("VIDEO_MAX_FRAMES must be between 1 and 8.")
    if not 2 <= jpeg_quality <= 10:
        raise ValueError("VIDEO_JPEG_QUALITY must be between 2 and 10.")
    if margin_setting < 0:
        raise ValueError(
            "VIDEO_FRAME_START_MARGIN_SECONDS cannot be negative."
        )

    ffmpeg = _get_executable("FFMPEG_BINARY", "ffmpeg")
    timeout = _env_int("MEDIA_FRAME_TIMEOUT_SECONDS", 120)
    metadata = _get_video_metadata(video)
    duration = metadata["duration_seconds"]

    # For a single frame, use the midpoint. Otherwise, distribute
    # timestamps between the configured margins.
    if max_frames == 1:
        timestamps = [duration / 2]
    else:
        margin = min(margin_setting, duration / (2 * max_frames))
        start = margin
        end = duration - margin

        timestamps = [
            start + i * (end - start) / (max_frames - 1)
            for i in range(max_frames)
        ]

    configured_dir = (
        output_dir
        if output_dir is not None
        else _env_string("VIDEO_FRAME_OUTPUT_DIR", "data/frames")
    )

    frames_dir = Path(configured_dir).resolve() / video.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    saved_frames = []

    logger.info(
        "Extracting up to %d frames from %s (%.1f seconds)",
        max_frames,
        video.name,
        duration,
    )

    for index, timestamp in enumerate(timestamps):
        # Index-based filenames avoid collisions for short videos.
        frame_path = frames_dir / (
            f"frame_{index:02d}_{timestamp:09.3f}s.jpg"
        )

        try:
            _run_command(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-ss", f"{timestamp:.3f}",
                    "-i", str(video),
                    "-map", "0:v:0",
                    "-frames:v", "1",
                    "-q:v", str(jpeg_quality),
                    str(frame_path),
                ],
                operation=f"extracting frame at {timestamp:.2f}s",
                timeout_seconds=timeout,
            )

            if frame_path.is_file() and frame_path.stat().st_size > 0:
                saved_frames.append({
                    "timestamp_seconds": round(timestamp, 3),
                    "path": str(frame_path),
                    "size_bytes": frame_path.stat().st_size,
                })
            else:
                logger.warning(
                    "No frame produced at %.2f seconds", timestamp
                )

        except RuntimeError:
            # Keep successfully extracted frames if another timestamp fails.
            logger.exception(
                "Frame extraction failed at %.2f seconds", timestamp
            )

    if not saved_frames:
        raise RuntimeError("No video frames could be extracted.")

    result = {
        **metadata,
        "requested_frame_count": max_frames,
        "frame_count": len(saved_frames),
        "frames": saved_frames,
    }

    logger.info(
        "Frame extraction complete: %d of %d frames saved",
        len(saved_frames),
        max_frames,
    )

    return result


# ============================================================
# 3. Silence detection
# ============================================================

def detect_silence(audio_path: str) -> dict:
    """
    Detect low-volume intervals using settings from .env.

    Does not trim or modify the source audio.
    """
    audio = Path(audio_path).resolve()

    if not audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    enabled = _env_bool("SILENCE_DETECTION_ENABLED", True)

    if not enabled:
        logger.info("Silence detection is disabled in .env.")
        return {
            "enabled": False,
            "audio_path": str(audio),
            "interval_count": 0,
            "total_detected_seconds": 0.0,
            "intervals": [],
        }

    threshold_db = _env_int("SILENCE_NOISE_THRESHOLD_DB", -40)
    minimum_duration = _env_float(
        "SILENCE_MIN_DURATION_SECONDS", 2.0
    )

    if not -90 <= threshold_db <= -10:
        raise ValueError(
            "SILENCE_NOISE_THRESHOLD_DB must be between -90 and -10."
        )
    if minimum_duration <= 0:
        raise ValueError(
            "SILENCE_MIN_DURATION_SECONDS must be greater than zero."
        )

    ffmpeg = _get_executable("FFMPEG_BINARY", "ffmpeg")
    timeout = _env_int("MEDIA_COMMAND_TIMEOUT_SECONDS", 600)

    logger.info(
        "Detecting low-volume intervals in %s", audio.name
    )

    result = _run_command(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-i", str(audio),
            "-af",
            (
                f"silencedetect=noise={threshold_db}dB:"
                f"d={minimum_duration}"
            ),
            "-f", "null",
            "-",
        ],
        operation="detecting silence",
        timeout_seconds=timeout,
    )

    start_pattern = re.compile(r"silence_start:\s*([0-9.]+)")
    end_pattern = re.compile(r"silence_end:\s*([0-9.]+)")

    intervals = []
    pending_start = None

    for line in result.stderr.splitlines():
        start_match = start_pattern.search(line)
        end_match = end_pattern.search(line)

        if start_match:
            pending_start = float(start_match.group(1))

        if end_match and pending_start is not None:
            end_time = float(end_match.group(1))

            if end_time >= pending_start:
                intervals.append({
                    "start_seconds": round(pending_start, 3),
                    "end_seconds": round(end_time, 3),
                    "duration_seconds": round(
                        end_time - pending_start, 3
                    ),
                })

            pending_start = None

    # A silent interval continuing to EOF may have no silence_end event.
    if pending_start is not None:
        ffprobe = _get_executable("FFPROBE_BINARY", "ffprobe")

        duration_result = _run_command(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio),
            ],
            operation="reading audio duration",
            timeout_seconds=_env_int(
                "MEDIA_METADATA_TIMEOUT_SECONDS", 60
            ),
        )

        try:
            audio_duration = float(duration_result.stdout.strip())
        except ValueError as exc:
            raise RuntimeError(
                "Could not read audio duration for silence detection."
            ) from exc

        if audio_duration >= pending_start:
            intervals.append({
                "start_seconds": round(pending_start, 3),
                "end_seconds": round(audio_duration, 3),
                "duration_seconds": round(
                    audio_duration - pending_start, 3
                ),
            })

    total_seconds = sum(
        interval["duration_seconds"] for interval in intervals
    )

    logger.info(
        "Silence detection complete: %d intervals, %.2f seconds",
        len(intervals),
        total_seconds,
    )

    return {
        "enabled": True,
        "audio_path": str(audio),
        "noise_threshold_db": threshold_db,
        "minimum_silence_seconds": minimum_duration,
        "interval_count": len(intervals),
        "total_detected_seconds": round(total_seconds, 3),
        "intervals": intervals,
    }
