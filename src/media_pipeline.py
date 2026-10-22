"""
Central media-processing pipeline for the classroom-quality evaluator.

Responsibilities:
1. Validate the input classroom video.
2. Extract WAV audio and representative video frames.
3. Analyze extracted frames with Gemini visual analysis.
4. Analyze audio quality and detect silence intervals.
5. Transcribe audio in chunks.
6. Assess classroom quality using Groq -> Gemini -> regex fallback.
7. Combine outputs into one structured result.
8. Log progress, timings, and errors without logging private
   audio or transcript content.
9. Optionally notify a UI through a progress callback.

This module coordinates existing modules. It does not independently
decide whether teaching quality is Poor, Average, Good, or Excellent.
"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv


# ------------------------------------------------------------
# 1. Project paths and environment configuration
# ------------------------------------------------------------

# Resolve the project root from this file's location.
# This works even when the script is run from another directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load configuration before importing modules that read .env
# values during import.
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# Store generated audio and other processed outputs separately
# from the original classroom recording.
MEDIA_OUTPUT_DIR = Path(
    os.getenv(
        "MEDIA_PIPELINE_OUTPUT_DIR",
        "data/processed",
    )
)

# Use the existing frame directory setting from .env.
VIDEO_FRAME_OUTPUT_DIR = Path(
    os.getenv(
        "VIDEO_FRAME_OUTPUT_DIR",
        "data/frames",
    )
)

# Limit the number of seconds allowed for an individual processing
# stage where a timeout can be applied by the underlying module.
MEDIA_COMMAND_TIMEOUT_SECONDS = int(
    os.getenv("MEDIA_COMMAND_TIMEOUT_SECONDS", "600")
)


# ------------------------------------------------------------
# 2. Import existing processing modules
# ------------------------------------------------------------

# These imports reuse the modules you have already implemented.
from src.media_processor import (
    extract_audio,
    extract_key_frames,
    detect_silence,
)
from src.audio_analyzer import analyze_audio
from src.transcription_chunked import transcribe_audio_in_chunks
from src.video_analyzer_gemini import analyze_video_frames
from src.llm_assessment_engine import llm_assess_classroom_quality


# ------------------------------------------------------------
# 3. Helper functions
# ------------------------------------------------------------

def _resolve_project_path(path: str | Path) -> Path:
    """
    Resolve relative paths against the project root.

    Absolute paths are kept unchanged.
    """
    resolved_path = Path(path).expanduser()

    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path

    return resolved_path.resolve()


def _notify_progress(
    progress_callback,
    stage: str,
    message: str,
    percent: int,
) -> None:
    """
    Notify an optional UI callback about the current pipeline stage.

    The callback is deliberately best-effort: a UI update failure must not
    interrupt video processing. Callback signature:
        callback(stage: str, message: str, percent: int)
    """
    if progress_callback is None:
        return

    try:
        progress_callback(stage, message, percent)
    except Exception:
        logger.warning(
            "Progress callback failed; continuing pipeline: stage=%s",
            stage,
            exc_info=True,
        )


def _record_stage_error(
    errors: list[dict],
    stage: str,
    exc: Exception,
) -> None:
    """
    Record an error without including audio or transcript content.

    Keeping errors in the result allows the application to display
    a useful status even if some processing stages fail.
    """
    error_type = type(exc).__name__

    logger.exception(
        "Media pipeline stage failed: stage=%s error_type=%s",
        stage,
        error_type,
    )

    errors.append({
        "stage": stage,
        "error_type": error_type,
        "message": str(exc),
    })


# ------------------------------------------------------------
# 4. Main media-processing pipeline
# ------------------------------------------------------------

def process_classroom_video(
    video_path: str | Path,
    progress_callback=None,
) -> dict:
    """
    Process a classroom recording and return a combined result.

    Args:
        video_path:
            Path to the original classroom video.
        progress_callback:
            Optional callable receiving (stage, message, percent) updates.
            Intended for a UI such as Streamlit. If omitted, behavior is
            the same as the original pipeline.

    Returns:
        A dictionary containing:
        - video metadata
        - extracted audio path
        - audio analysis
        - silence detection results
        - extracted frame information
        - Gemini visual analysis
        - transcription results
        - classroom-quality assessment (Groq -> Gemini -> regex fallback)
        - processing status
        - stage errors and processing timings

    Raises:
        FileNotFoundError:
            The input video does not exist.
        ValueError:
            The input path is invalid or points to a non-file.
    """
    pipeline_started = time.perf_counter()

    video_file = _resolve_project_path(video_path)

    # Validate the input before creating any output files.
    if not video_file.exists():
        raise FileNotFoundError(
            f"Input video was not found: {video_file}"
        )

    if not video_file.is_file():
        raise ValueError(
            f"Input video path is not a file: {video_file}"
        )

    if video_file.stat().st_size == 0:
        raise ValueError("The input video file is empty.")

    # Use a separate output directory for each source video.
    # This prevents audio outputs from different videos from
    # overwriting each other's files.
    video_output_dir = (
        _resolve_project_path(MEDIA_OUTPUT_DIR)
        / video_file.stem
    )

    audio_output_dir = video_output_dir / "audio"
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    audio_file = audio_output_dir / "classroom_audio.wav"

    # The pipeline returns results even if some independent stages
    # fail. The status is updated after all possible stages finish.
    result = {
        "status": "processing",
        "video": {
            "file_name": video_file.name,
            "file_size_mb": round(
                video_file.stat().st_size / (1024 * 1024),
                2,
            ),
            "path": str(video_file),
        },
        "audio": {
            "wav_path": None,
            "analysis": None,
            "silence": None,
        },
        "frames": None,
        "visual_analysis": None,
        "transcription": None,
        # Final assessment result. The assessment module handles its own
        # Groq -> Gemini -> regex fallback chain.
        "assessment": None,
        "errors": [],
        "stage_timings_seconds": {},
        "total_processing_seconds": None,
    }

    logger.info(
        "Starting classroom media pipeline: file=%s size_mb=%.2f",
        video_file.name,
        result["video"]["file_size_mb"],
    )

    audio_ready = False

    # --------------------------------------------------------
    # Stage 1: Extract WAV audio
    # --------------------------------------------------------

    _notify_progress(
        progress_callback, "audio_extraction", "Extracting audio...", 5
    )
    stage_started = time.perf_counter()

    try:
        logger.info("Stage 1/6: Extracting audio.")

        extracted_audio = extract_audio(
            video_path=str(video_file),
            audio_path=str(audio_file),
        )

        # Use the returned path because the media processor is
        # responsible for creating the actual audio output.
        audio_file = Path(extracted_audio)

        if not audio_file.is_absolute():
            audio_file = PROJECT_ROOT / audio_file

        if not audio_file.is_file() or audio_file.stat().st_size == 0:
            raise RuntimeError(
                "Audio extraction did not produce a usable WAV file."
            )

        result["audio"]["wav_path"] = str(audio_file)
        audio_ready = True

        logger.info(
            "Audio extraction completed: file=%s size_mb=%.2f",
            audio_file.name,
            audio_file.stat().st_size / (1024 * 1024),
        )

    except Exception as exc:
        _record_stage_error(
            result["errors"],
            "audio_extraction",
            exc,
        )

    finally:
        result["stage_timings_seconds"]["audio_extraction"] = round(
            time.perf_counter() - stage_started,
            2,
        )

    # --------------------------------------------------------
    # Stage 2: Extract representative video frames
    # --------------------------------------------------------

    _notify_progress(
        progress_callback, "frame_extraction", "Extracting video frames...", 18
    )
    stage_started = time.perf_counter()

    try:
        logger.info("Stage 2/6: Extracting video frames.")

        frame_result = extract_key_frames(
            video_path=str(video_file),
            output_dir=str(
                _resolve_project_path(VIDEO_FRAME_OUTPUT_DIR)
            ),
        )

        result["frames"] = frame_result

        logger.info(
            "Frame extraction completed: file=%s frame_count=%d",
            video_file.name,
            len(frame_result.get("frames", [])),
        )

    except Exception as exc:
        _record_stage_error(
            result["errors"],
            "frame_extraction",
            exc,
        )

    finally:
        result["stage_timings_seconds"]["frame_extraction"] = round(
            time.perf_counter() - stage_started,
            2,
        )

    # --------------------------------------------------------
    # Stage 3: Analyze extracted frames with Gemini
    # --------------------------------------------------------
    # This stage is independent of audio processing. If audio
    # extraction or transcription fails, visual analysis can still run.

    _notify_progress(
        progress_callback, "visual_analysis", "Analyzing sampled video frames...", 32
    )
    stage_started = time.perf_counter()

    try:
        extracted_frames = (
            result["frames"].get("frames", [])
            if isinstance(result["frames"], dict)
            else []
        )

        if not extracted_frames:
            raise RuntimeError(
                "Visual analysis skipped because no frames were extracted."
            )

        logger.info(
            "Stage 3/6: Analyzing %d frames with Gemini.",
            len(extracted_frames),
        )

        visual_result = analyze_video_frames(extracted_frames)
        result["visual_analysis"] = visual_result

        logger.info(
            "Gemini visual analysis completed: provider=%s model=%s",
            visual_result.get("provider"),
            visual_result.get("model"),
        )

    except Exception as exc:
        _record_stage_error(
            result["errors"],
            "visual_analysis",
            exc,
        )

        # Preserve a predictable structure when visual analysis fails.
        result["visual_analysis"] = {
            "status": "failed",
            "provider": "gemini",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    finally:
        result["stage_timings_seconds"]["visual_analysis"] = round(
            time.perf_counter() - stage_started,
            2,
        )

    # The remaining stages require successfully extracted audio.
    if audio_ready:

        # ----------------------------------------------------
        # Stage 4: Analyze audio quality
        # ----------------------------------------------------

        _notify_progress(
            progress_callback, "audio_analysis", "Analyzing audio quality...", 45
        )
        stage_started = time.perf_counter()

        try:
            logger.info("Stage 4/6: Analyzing audio quality.")

            result["audio"]["analysis"] = analyze_audio(
                str(audio_file)
            )

            logger.info(
                "Audio analysis completed: file=%s",
                audio_file.name,
            )

        except Exception as exc:
            _record_stage_error(
                result["errors"],
                "audio_analysis",
                exc,
            )

        finally:
            result["stage_timings_seconds"]["audio_analysis"] = round(
                time.perf_counter() - stage_started,
                2,
            )

        # ----------------------------------------------------
        # Stage 5: Detect silence intervals
        # ----------------------------------------------------

        _notify_progress(
            progress_callback, "silence_detection", "Detecting silence intervals...", 55
        )
        stage_started = time.perf_counter()

        try:
            logger.info("Stage 5/6: Detecting silence intervals.")

            result["audio"]["silence"] = detect_silence(
                str(audio_file)
            )

            logger.info(
                "Silence detection completed: file=%s",
                audio_file.name,
            )

        except Exception as exc:
            _record_stage_error(
                result["errors"],
                "silence_detection",
                exc,
            )

        finally:
            result["stage_timings_seconds"]["silence_detection"] = round(
                time.perf_counter() - stage_started,
                2,
            )

        # ----------------------------------------------------
        # Stage 6: Transcribe audio in chunks
        # ----------------------------------------------------

        _notify_progress(
            progress_callback, "transcription", "Transcribing speech in chunks...", 68
        )
        stage_started = time.perf_counter()

        try:
            logger.info(
                "Stage 6/6: Starting chunked transcription."
            )

            transcription_result = transcribe_audio_in_chunks(
                str(audio_file)
            )

            result["transcription"] = transcription_result

            logger.info(
                "Transcription completed: provider=%s chunks=%s "
                "segments=%d",
                transcription_result.get("provider"),
                transcription_result.get("chunk_count"),
                len(transcription_result.get("segments", [])),
            )

        except Exception as exc:
            _record_stage_error(
                result["errors"],
                "transcription",
                exc,
            )

        finally:
            result["stage_timings_seconds"]["transcription"] = round(
                time.perf_counter() - stage_started,
                2,
            )

    else:
        # Do not run audio analysis, silence detection, or transcription
        # when audio extraction has failed.
        logger.warning(
            "Audio-dependent stages skipped: audio extraction failed."
        )

        result["errors"].append({
            "stage": "audio_dependent_stages",
            "error_type": "Skipped",
            "message": (
                "Audio analysis, silence detection, and transcription "
                "were skipped because audio extraction failed."
            ),
        })

    # --------------------------------------------------------
    # Stage 7: Assess classroom quality
    # --------------------------------------------------------
    # This function internally runs the regex baseline first, then tries
    # Groq, then Gemini. If both LLM calls fail or return invalid output,
    # it returns the regex result with assessment_source="regex".
    # This stage is attempted even when transcription failed so the
    # baseline engine can return an insufficient-evidence result instead
    # of inventing a low rating.

    _notify_progress(
        progress_callback, "assessment", "Assessing classroom quality...", 88
    )
    stage_started = time.perf_counter()

    try:
        logger.info(
            "Stage 7/7: Assessing classroom quality (Groq -> Gemini -> regex)."
        )

        result["assessment"] = llm_assess_classroom_quality(
            transcription=result["transcription"],
            visual_analysis=result["visual_analysis"],
            audio_analysis=result["audio"]["analysis"],
            silence_analysis=result["audio"]["silence"],
        )

        logger.info(
            "Classroom assessment completed: source=%s rating=%s",
            result["assessment"].get("assessment_source"),
            result["assessment"].get("provisional_rating"),
        )

    except Exception as exc:
        # A programming/configuration error in the assessment wrapper should
        # not erase the media processing results collected earlier.
        _record_stage_error(result["errors"], "assessment", exc)
        result["assessment"] = {
            "status": "failed",
            "assessment_source": None,
            "provisional_rating": None,
            "overall_score_0_to_4": None,
            "criteria": {},
            "specific_observations": [],
            "limitations": [
                "Assessment could not be completed because of an unexpected pipeline error."
            ],
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    finally:
        result["stage_timings_seconds"]["assessment"] = round(
            time.perf_counter() - stage_started,
            2,
        )

    # --------------------------------------------------------
    # Finalize pipeline status
    # --------------------------------------------------------

    _notify_progress(
        progress_callback, "finalizing", "Preparing assessment results...", 96
    )

    result["total_processing_seconds"] = round(
        time.perf_counter() - pipeline_started,
        2,
    )

    if not result["errors"]:
        result["status"] = "completed"
    elif (
        result["audio"]["wav_path"] is None
        and result["frames"] is None
    ):
        result["status"] = "failed"
    else:
        result["status"] = "partial"

    logger.info(
        "Classroom media pipeline finished: file=%s status=%s "
        "errors=%d elapsed_seconds=%.2f",
        video_file.name,
        result["status"],
        len(result["errors"]),
        result["total_processing_seconds"],
    )

    return result
