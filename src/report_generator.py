"""Generate a human-readable classroom assessment report from pipeline output.

This module does not perform assessment or change scores. It formats the
structured result produced by ``src.media_pipeline.process_classroom_video``
into a report dictionary and Markdown suitable for display or download.

Privacy note: the report contains selected transcript evidence. Store it in an
approved location and do not expose identifiable classroom recordings or
transcripts without appropriate permission.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env relative to the project root so configuration does not depend on
# the current working directory. This module intentionally does not configure
# the root logger; the application should configure logging once at startup.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = Path(
    os.getenv("CLASSROOM_REPORT_OUTPUT_DIR", "data/reports")
)
MAX_EVIDENCE_ITEMS_PER_CRITERION = max(
    1, int(os.getenv("REPORT_MAX_EVIDENCE_PER_CRITERION", "5"))
)
MAX_TRANSCRIPT_EXCERPT_CHARS = max(
    80, int(os.getenv("REPORT_MAX_TRANSCRIPT_EXCERPT_CHARS", "400"))
)

# Keep the labels readable while retaining the stable keys used by the
# assessment engine. Unknown criterion keys are still included in the report.
CRITERION_LABELS = {
    "explanation_and_examples": "Explanation and examples",
    "lesson_structure": "Lesson structure",
    "questioning": "Questioning",
    "feedback_and_guidance": "Feedback and guidance",
    "teaching_materials": "Teaching materials (visual evidence only)",
}

DISCLAIMER = (
    "This report is an AI-assisted, provisional review aid. It is not a "
    "validated measure of teaching quality and must not replace human review. "
    "A missing transcript indicator or unavailable visual analysis is not "
    "proof that a teaching practice was absent. Sampled video frames cannot "
    "establish student engagement, learning, or teaching effectiveness."
)


def _as_dict(value: Any) -> dict:
    """Return a dictionary or an empty dictionary for missing/malformed data."""
    return value if isinstance(value, dict) else {}


def _as_string_list(value: Any) -> list[str]:
    """Normalize a list-like field without turning dictionaries into strings."""
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _extract_transcript(transcription: Any) -> str:
    """Support the pipeline's transcription dict, segment lists, or plain text."""
    if isinstance(transcription, str):
        return transcription.strip()

    if not isinstance(transcription, dict):
        return ""

    text = transcription.get("transcript")
    if isinstance(text, str) and text.strip():
        return text.strip()

    segments = transcription.get("segments")
    if isinstance(segments, list):
        parts = []
        for segment in segments:
            if isinstance(segment, dict):
                segment_text = segment.get("text")
                if isinstance(segment_text, str) and segment_text.strip():
                    parts.append(segment_text.strip())
        return " ".join(parts)

    return ""


def _seconds_from_segment(segment: dict, keys: tuple[str, ...]) -> float | None:
    """Read a numeric timestamp from one of the common segment field names."""
    for key in keys:
        value = segment.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
    return None


def _format_timestamp(seconds: float | None) -> str | None:
    """Format seconds as HH:MM:SS; return None when no timestamp is known."""
    if seconds is None:
        return None
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _normalize_for_match(text: str) -> str:
    """Normalize punctuation/spacing for conservative excerpt-to-segment matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _find_timestamp_for_excerpt(excerpt: str, segments: Any) -> dict | None:
    """Find a segment containing an evidence excerpt and return its timestamps.

    Matching is deliberately conservative. If the exact excerpt cannot be
    matched to a segment, the report does not invent a timestamp.
    """
    if not excerpt or not isinstance(segments, list):
        return None

    target = _normalize_for_match(excerpt)
    if len(target) < 12:
        return None

    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            continue
        segment_text = _normalize_for_match(segment["text"])
        # Accept containment in either direction for providers that split a
        # sentence into multiple timestamped chunks.
        if target in segment_text or (segment_text and segment_text in target):
            start = _seconds_from_segment(segment, ("start", "start_time", "start_seconds"))
            end = _seconds_from_segment(segment, ("end", "end_time", "end_seconds"))
            return {
                "start_seconds": start,
                "end_seconds": end,
                "start_timestamp": _format_timestamp(start),
                "end_timestamp": _format_timestamp(end),
            }

    return None


def _build_transcript_evidence(assessment: dict, transcription: Any) -> list[dict]:
    """Collect assessment evidence quotes and attach timestamps when available."""
    transcript_data = _as_dict(transcription)
    segments = transcript_data.get("segments", [])
    criteria = _as_dict(assessment.get("criteria"))
    collected: list[dict] = []
    seen: set[str] = set()

    for criterion_key, criterion in criteria.items():
        criterion = _as_dict(criterion)
        evidence_items = _as_string_list(criterion.get("evidence"))

        for excerpt in evidence_items[:MAX_EVIDENCE_ITEMS_PER_CRITERION]:
            normalized = _normalize_for_match(excerpt)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)

            timestamp = _find_timestamp_for_excerpt(excerpt, segments)
            if len(excerpt) > MAX_TRANSCRIPT_EXCERPT_CHARS:
                excerpt = excerpt[: MAX_TRANSCRIPT_EXCERPT_CHARS - 3].rstrip() + "..."

            collected.append({
                "criterion": criterion_key,
                "criterion_label": CRITERION_LABELS.get(
                    criterion_key, criterion_key.replace("_", " ").title()
                ),
                "excerpt": excerpt,
                "timestamp": timestamp,
                "source": "transcript",
            })

    return collected


def _collect_visual_evidence(visual_analysis: Any) -> dict:
    """Return successful visual observations separately from quality scores."""
    visual = _as_dict(visual_analysis)
    if visual.get("status") != "success":
        return {
            "status": visual.get("status", "unavailable"),
            "provider": visual.get("provider", "gemini"),
            "model": visual.get("model"),
            "observations": {},
            "message": visual.get("message") or "Visual observations were unavailable.",
        }

    observations = _as_dict(visual.get("observations"))
    return {
        "status": "success",
        "provider": visual.get("provider", "gemini"),
        "model": visual.get("model"),
        "observations": observations,
        "message": None,
    }


def build_report(pipeline_result: dict) -> dict:
    """Build normalized report data from ``process_classroom_video`` output.

    Args:
        pipeline_result: Dictionary returned by the media pipeline.

    Returns:
        A JSON-serializable report dictionary. This function does not write
        files and does not modify the input dictionary.
    """
    if not isinstance(pipeline_result, dict):
        raise TypeError("pipeline_result must be a dictionary")

    video = _as_dict(pipeline_result.get("video"))
    assessment = _as_dict(pipeline_result.get("assessment"))
    transcription = pipeline_result.get("transcription")
    visual = _collect_visual_evidence(pipeline_result.get("visual_analysis"))
    audio = _as_dict(pipeline_result.get("audio"))
    errors = pipeline_result.get("errors", [])
    if not isinstance(errors, list):
        errors = []

    criteria_report = []
    criteria = _as_dict(assessment.get("criteria"))
    for criterion_key, raw_criterion in criteria.items():
        criterion = _as_dict(raw_criterion)
        score = criterion.get("score_0_to_4")
        criteria_report.append({
            "key": criterion_key,
            "label": CRITERION_LABELS.get(
                criterion_key, criterion_key.replace("_", " ").title()
            ),
            "score_0_to_4": score if isinstance(score, (int, float)) else None,
            "evidence": _as_string_list(criterion.get("evidence")),
            "interpretation": str(criterion.get("interpretation", "") or ""),
            "evidence_status": criterion.get("evidence_status"),
        })

    transcript_text = _extract_transcript(transcription)
    transcript_data = _as_dict(transcription)
    report = {
        "report_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_status": pipeline_result.get("status", "unknown"),
        "video": {
            # Include the file name, not the full local path, in the report.
            "file_name": video.get("file_name", "Unknown video"),
            "file_size_mb": video.get("file_size_mb"),
        },
        "assessment": {
            "status": assessment.get("status", "unavailable"),
            "provisional_rating": assessment.get("provisional_rating"),
            "overall_score_0_to_4": assessment.get("overall_score_0_to_4"),
            "assessment_source": assessment.get("assessment_source", "unknown"),
            "criteria": criteria_report,
            "specific_observations": _as_string_list(
                assessment.get("specific_observations")
            ),
            "evidence_summary": assessment.get("evidence_summary"),
            "limitations": _as_string_list(assessment.get("limitations")),
            "heuristic_backup_available": isinstance(
                assessment.get("heuristic_backup"), dict
            ),
        },
        "transcript": {
            "available": bool(transcript_text),
            "character_count": len(transcript_text),
            "segment_count": len(transcript_data.get("segments", []))
            if isinstance(transcript_data.get("segments"), list) else 0,
            "evidence_excerpts": _build_transcript_evidence(assessment, transcription),
        },
        "visual_evidence": visual,
        "audio_context_not_scored": _as_dict(assessment.get(
            "audio_context_not_scored", audio.get("analysis")
        )),
        "silence_context_not_scored": _as_dict(assessment.get(
            "silence_context_not_scored", audio.get("silence")
        )),
        "stage_timings_seconds": _as_dict(
            pipeline_result.get("stage_timings_seconds")
        ),
        "total_processing_seconds": pipeline_result.get("total_processing_seconds"),
        "processing_errors": [
            {
                "stage": item.get("stage", "unknown"),
                "error_type": item.get("error_type", "Error"),
                # Avoid printing full exception details in the user-facing report;
                # they may contain local paths or provider request details.
                "message": str(item.get("message", "Processing error"))[:300],
            }
            for item in errors if isinstance(item, dict)
        ],
        "disclaimer": DISCLAIMER,
    }

    logger.info(
        "Report data prepared: video=%s pipeline_status=%s rating=%s "
        "assessment_source=%s criteria=%d transcript_evidence=%d errors=%d",
        report["video"]["file_name"],
        report["pipeline_status"],
        report["assessment"]["provisional_rating"],
        report["assessment"]["assessment_source"],
        len(criteria_report),
        len(report["transcript"]["evidence_excerpts"]),
        len(report["processing_errors"]),
    )
    return report


def _format_score(score: Any) -> str:
    """Format scores without displaying missing scores as zero."""
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return f"{score:.2f}/4" if isinstance(score, float) else f"{score}/4"
    return "Not scored — insufficient or unavailable evidence"


def render_markdown_report(report: dict) -> str:
    """Render report data as Markdown for Streamlit display/download."""
    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")

    video = _as_dict(report.get("video"))
    assessment = _as_dict(report.get("assessment"))
    transcript = _as_dict(report.get("transcript"))
    visual = _as_dict(report.get("visual_evidence"))

    lines = [
        "# Classroom Quality Assessment Report",
        "",
        f"**Video:** {video.get('file_name', 'Unknown video')}",
        f"**Report generated (UTC):** {report.get('generated_at_utc', 'Unknown')}",
        f"**Pipeline status:** {report.get('pipeline_status', 'unknown')}",
        "",
        "## Overall assessment",
        "",
        f"- **Provisional rating:** {assessment.get('provisional_rating') or 'Not assigned'}",
        f"- **Overall score:** {_format_score(assessment.get('overall_score_0_to_4'))}",
        f"- **Assessment source:** {assessment.get('assessment_source') or 'Unknown'}",
        f"- **Assessment status:** {assessment.get('status') or 'Unknown'}",
    ]

    if assessment.get("evidence_summary"):
        lines.extend(["", str(assessment["evidence_summary"])])

    lines.extend(["", "## Criterion-level scores and evidence", ""])
    criteria = assessment.get("criteria", [])
    if not criteria:
        lines.append("No criterion-level assessment was available.")
    else:
        lines.extend([
            "| Criterion | Score | Evidence / interpretation |",
            "|---|---:|---|",
        ])
        for criterion in criteria:
            evidence = _as_string_list(criterion.get("evidence"))
            interpretation = str(criterion.get("interpretation", "") or "")
            detail_parts = evidence[:MAX_EVIDENCE_ITEMS_PER_CRITERION]
            if interpretation:
                detail_parts.append(interpretation)
            detail = "<br>".join(
                part.replace("|", "\\|").replace("\n", " ")
                for part in detail_parts
            ) or "No evidence excerpt supplied."
            lines.append(
                f"| {criterion.get('label', criterion.get('key', 'Criterion'))} "
                f"| {_format_score(criterion.get('score_0_to_4'))} | {detail} |"
            )

    lines.extend(["", "## Transcript evidence and timestamps", ""])
    excerpts = transcript.get("evidence_excerpts", [])
    if not transcript.get("available"):
        lines.append("Transcript unavailable; no transcript excerpts can be shown.")
    elif not excerpts:
        lines.append(
            "No criterion-linked transcript excerpts were returned by the assessment. "
            "This does not prove that the practices were absent."
        )
    else:
        for item in excerpts:
            timestamp = _as_dict(item.get("timestamp"))
            time_label = ""
            if timestamp.get("start_timestamp"):
                time_label = f" ({timestamp['start_timestamp']}"
                if timestamp.get("end_timestamp"):
                    time_label += f"–{timestamp['end_timestamp']}"
                time_label += ")"
            else:
                time_label = " (timestamp unavailable)"
            lines.append(
                f"- **{item.get('criterion_label', 'Evidence')}**{time_label}: "
                f"> {str(item.get('excerpt', '')).replace(chr(10), ' ')}"
            )

    lines.extend(["", "## Observable visual evidence", ""])
    lines.append(f"**Visual analysis status:** {visual.get('status', 'unavailable')}")
    if visual.get("provider") or visual.get("model"):
        lines.append(
            f"**Provider/model:** {visual.get('provider') or 'unknown'} / "
            f"{visual.get('model') or 'unknown'}"
        )
    observations = _as_dict(visual.get("observations"))
    if observations:
        lines.append("")
        lines.append(
            "The following are visual observations only; they are not, by themselves, "
            "proof of teaching effectiveness or student learning:"
        )
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(observations, indent=2, ensure_ascii=False))
        lines.append("```")
    else:
        lines.append(str(visual.get("message") or "No visual observations are available."))

    if assessment.get("specific_observations"):
        lines.extend(["", "## Additional evidence-based observations", ""])
        lines.extend(f"- {item}" for item in assessment["specific_observations"])

    lines.extend(["", "## Processing details", ""])
    lines.append(f"- **Total processing time:** {report.get('total_processing_seconds')} seconds")
    timings = _as_dict(report.get("stage_timings_seconds"))
    if timings:
        lines.append("- **Stage timings:**")
        lines.extend(
            f"  - {stage.replace('_', ' ').title()}: {seconds} seconds"
            for stage, seconds in timings.items()
        )

    errors = report.get("processing_errors", [])
    lines.extend(["", "## Missing evidence and processing errors", ""])
    if not errors:
        lines.append("No pipeline stage errors were recorded.")
    else:
        lines.append(
            "One or more stages did not complete. Interpret the report with this limitation:"
        )
        for error in errors:
            lines.append(
                f"- **{error.get('stage', 'unknown')}** — "
                f"{error.get('error_type', 'Error')}: {error.get('message', '')}"
            )

    limitations = _as_string_list(assessment.get("limitations"))
    lines.extend(["", "## Assessment limitations", ""])
    if limitations:
        lines.extend(f"- {item}" for item in limitations)
    else:
        lines.append("No additional limitations were returned by the assessment module.")

    lines.extend(["", "## Disclaimer", "", str(report.get("disclaimer", DISCLAIMER)), ""])
    return "\n".join(lines)


def generate_markdown_report(
    pipeline_result: dict,
    output_path: str | Path | None = None,
) -> dict:
    """Build and render a report, optionally saving it as a Markdown file.

    Returns a dictionary containing ``report_data``, ``markdown``, and the
    saved ``path`` (or None when no file was requested).
    """
    started = __import__("time").perf_counter()
    logger.info("Starting classroom report generation.")

    try:
        report_data = build_report(pipeline_result)
        markdown = render_markdown_report(report_data)
        saved_path: str | None = None

        if output_path is not None:
            destination = Path(output_path).expanduser()
            if not destination.is_absolute():
                destination = PROJECT_ROOT / destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(markdown, encoding="utf-8")
            saved_path = str(destination.resolve())
            logger.info(
                "Markdown report saved: path=%s size_bytes=%d",
                destination.name,
                destination.stat().st_size,
            )

        logger.info(
            "Report generation completed: rating=%s source=%s saved=%s "
            "elapsed_seconds=%.2f",
            report_data["assessment"]["provisional_rating"],
            report_data["assessment"]["assessment_source"],
            saved_path is not None,
            __import__("time").perf_counter() - started,
        )
        return {
            "report_data": report_data,
            "markdown": markdown,
            "path": saved_path,
        }

    except Exception:
        logger.exception("Report generation failed.")
        raise
