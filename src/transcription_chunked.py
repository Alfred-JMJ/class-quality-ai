
"""
Chunked transcription for long classroom recordings.

Splits a WAV file into manageable chunks using FFmpeg,
transcribes each chunk using the existing transcription service,
and adjusts segment timestamps to the original recording timeline.
"""

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from src.transcription_service import transcribe_audio

load_dotenv()

logger = logging.getLogger(__name__)


def transcribe_audio_in_chunks(audio_path: str) -> dict:
    """Transcribe a long WAV file in chunks and merge the results."""

    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError("The audio file is empty.")

    ffmpeg = os.getenv("FFMPEG_BINARY", "ffmpeg")
    chunk_seconds = int(
        os.getenv("TRANSCRIPTION_CHUNK_SECONDS", "300")
    )

    if chunk_seconds <= 0:
        raise ValueError(
            "TRANSCRIPTION_CHUNK_SECONDS must be positive."
        )

    started = time.perf_counter()

    logger.info(
        "Starting chunked transcription: file=%s chunk_seconds=%d",
        path.name,
        chunk_seconds,
    )

    with tempfile.TemporaryDirectory(
        prefix="class_transcription_"
    ) as temporary_directory:

        temp_dir = Path(temporary_directory)
        output_pattern = temp_dir / "chunk_%03d.wav"

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", str(path),
            "-map", "0:a:0",
            "-vn",
            "-f", "segment",
            "-segment_time", str(chunk_seconds),
            "-reset_timestamps", "1",
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_pattern),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=int(
                    os.getenv(
                        "MEDIA_COMMAND_TIMEOUT_SECONDS", "600"
                    )
                ),
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"FFmpeg executable was not found: {ffmpeg}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Audio chunking exceeded its time limit."
            ) from exc
        except subprocess.CalledProcessError as exc:
            logger.error(
                "FFmpeg chunking failed with exit code %s",
                exc.returncode,
            )
            raise RuntimeError(
                "FFmpeg could not split the audio into chunks."
            ) from exc

        chunks = sorted(temp_dir.glob("chunk_*.wav"))

        if not chunks:
            raise RuntimeError(
                "FFmpeg did not produce any audio chunks."
            )

        all_segments = []
        chunk_results = []
        transcript_parts = []
        languages = []
        providers = []
        groq_errors = []

        for index, chunk_path in enumerate(chunks):
            offset_seconds = index * chunk_seconds

            logger.info(
                "Transcribing chunk %d/%d",
                index + 1,
                len(chunks),
            )

            result = transcribe_audio(str(chunk_path))

            chunk_text = (result.get("transcript") or "").strip()

            if chunk_text:
                transcript_parts.append(chunk_text)

            language = result.get("language")
            if language and language not in languages:
                languages.append(language)

            provider = result.get("provider")
            if provider and provider not in providers:
                providers.append(provider)

            groq_error = result.get("groq_error")
            if groq_error and groq_error not in groq_errors:
                groq_errors.append(groq_error)

            # Convert chunk-relative timestamps to video-relative
            # timestamps by adding the chunk's starting offset.
            for segment in result.get("segments", []):
                if (
                    segment.get("start") is None
                    or segment.get("end") is None
                ):
                    continue

                all_segments.append({
                    "start": round(
                        float(segment["start"]) + offset_seconds, 3
                    ),
                    "end": round(
                        float(segment["end"]) + offset_seconds, 3
                    ),
                    "text": str(segment.get("text", "")).strip(),
                })

            chunk_results.append({
                "chunk_index": index,
                "offset_seconds": offset_seconds,
                "provider": provider,
                "fallback_used": bool(
                    result.get("fallback_used", False)
                ),
                "segment_count": len(result.get("segments", [])),
            })

        elapsed = time.perf_counter() - started

        logger.info(
            "Chunked transcription completed: chunks=%d "
            "segments=%d elapsed_seconds=%.2f",
            len(chunks),
            len(all_segments),
            elapsed,
        )

        return {
            "provider": (
                providers[0]
                if len(providers) == 1
                else ", ".join(providers) if providers
                else None
            ),
            "model": "chunked transcription",
            "language": languages[0] if languages else None,
            "transcript": "\n".join(transcript_parts),
            "segments": all_segments,
            "fallback_used": any(
                item["fallback_used"] for item in chunk_results
            ),
            "groq_error": ", ".join(groq_errors) or None,
            "chunk_count": len(chunks),
            "chunk_results": chunk_results,
        }
