"""
Streamlit UI for the Humaniti classroom-quality assessment prototype.

Run from the project root:
    streamlit run app.py

This UI orchestrates existing modules; it does not implement assessment logic.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment settings before importing the processing modules.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

from src.media_pipeline import process_classroom_video
from src.report_generator import generate_markdown_report


# ---------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_VIDEO_UPLOAD_MB = max(
    1, int(os.getenv("MAX_VIDEO_UPLOAD_MB", "200"))
)
REPORT_OUTPUT_DIR = Path(
    os.getenv("CLASSROOM_REPORT_OUTPUT_DIR", "data/reports")
)

SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "mkv", "avi", "webm"]

st.set_page_config(
    page_title="Humaniti | Classroom Quality Assessment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------
# UI helper functions
# ---------------------------------------------------------------------

def _safe_filename(filename: str) -> str:
    """Remove unsafe path characters while preserving a readable filename."""
    name = Path(filename).name
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return safe or "classroom_video"


def _score_label(score) -> str:
    """Format a score while preserving missing values as 'Not scored'."""
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return f"{score:.2f}/4"
    return "Not scored"


def _render_assessment(report_data: dict) -> None:
    """Render the overall result and criterion-level evidence."""
    assessment = report_data.get("assessment", {})
    rating = assessment.get("provisional_rating") or "Not available"
    overall_score = assessment.get("overall_score_0_to_4")
    source = assessment.get("assessment_source") or "unknown"
    assessment_status = assessment.get("status") or "unavailable"

    st.subheader("Classroom-quality assessment")
    st.caption(
        "AI-assisted provisional assessment. Review the evidence before "
        "using this result to make decisions."
    )

    score_text = (
        f"{overall_score:.2f}/4"
        if isinstance(overall_score, (int, float))
        and not isinstance(overall_score, bool)
        else "Not available"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Provisional rating", rating)
    col2.metric("Overall score", score_text)
    col3.metric("Assessment source", source.upper())
    col4.metric("Assessment status", assessment_status.title())

    if source == "regex":
        st.warning(
            "Both LLM providers were unavailable or returned unusable output. "
            "This result uses the baseline regex/heuristic assessment."
        )

    criteria = assessment.get("criteria", [])
    if criteria:
        st.markdown("#### Criterion scores")
        table_rows = []
        for criterion in criteria:
            table_rows.append(
                {
                    "Criterion": criterion.get("label", criterion.get("key", "Unknown")),
                    "Score": _score_label(criterion.get("score_0_to_4")),
                    "Evidence status": criterion.get("evidence_status") or "Not specified",
                }
            )
        st.dataframe(
            table_rows,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("#### Criterion-level evidence")
        for criterion in criteria:
            label = criterion.get("label", criterion.get("key", "Criterion"))
            score = _score_label(criterion.get("score_0_to_4"))
            with st.expander(f"{label} — {score}"):
                interpretation = criterion.get("interpretation")
                if interpretation:
                    st.write(interpretation)

                evidence_items = criterion.get("evidence", [])
                if evidence_items:
                    st.markdown("**Assessment evidence**")
                    for item in evidence_items:
                        st.markdown(f"- {item}")
                else:
                    st.info(
                        "No explicit evidence excerpts were returned for this criterion."
                    )
    else:
        st.info("No criterion-level scores are available.")

    observations = assessment.get("specific_observations", [])
    if observations:
        with st.expander("Additional assessment observations"):
            for item in observations:
                st.markdown(f"- {item}")

    limitations = assessment.get("limitations", [])
    if limitations:
        with st.expander("Assessment limitations"):
            for item in limitations:
                st.markdown(f"- {item}")


def _render_transcript_evidence(report_data: dict) -> None:
    """Render transcript excerpts and reliable timestamps, if available."""
    transcript = report_data.get("transcript", {})
    excerpts = transcript.get("evidence_excerpts", [])

    st.subheader("Transcript evidence")
    if not transcript.get("available"):
        st.info("A transcript was not available for this recording.")
        return

    st.caption(
        f"{transcript.get('segment_count', 0)} timestamped segments; "
        f"{transcript.get('character_count', 0):,} transcript characters."
    )

    if not excerpts:
        st.info("No transcript excerpts were selected as assessment evidence.")
        return

    for item in excerpts:
        timestamp_data = item.get("timestamp")
        timestamp_label = "Timestamp unavailable"
        if isinstance(timestamp_data, dict):
            start = timestamp_data.get("start_timestamp")
            end = timestamp_data.get("end_timestamp")
            if start and end:
                timestamp_label = f"{start}–{end}"
            elif start:
                timestamp_label = start

        criterion = item.get("criterion_label", "Assessment evidence")
        with st.expander(f"{criterion} · {timestamp_label}"):
            st.write(item.get("excerpt", ""))


def _render_visual_evidence(report_data: dict) -> None:
    """Show visual observations separately from scored teaching criteria."""
    visual = report_data.get("visual_evidence", {})
    st.subheader("Visual observations")
    st.caption(
        "These describe what was visible in sampled frames. They are not, "
        "by themselves, proof of student engagement, learning, or teaching effectiveness."
    )

    if visual.get("status") != "success":
        st.warning(
            visual.get("message")
            or f"Visual analysis is {visual.get('status', 'unavailable')}."
        )
        return

    st.caption(
        f"Provider: {visual.get('provider', 'unknown')} · "
        f"Model: {visual.get('model', 'unknown')}"
    )
    observations = visual.get("observations", {})
    if observations:
        st.json(observations, expanded=False)
    else:
        st.info("No structured visual observations were returned.")


def _render_processing_details(report_data: dict) -> None:
    """Display timings, processing errors, and overall pipeline status."""
    st.subheader("Processing details")

    status = report_data.get("pipeline_status", "unknown")
    if status == "completed":
        st.success("Pipeline completed without recorded stage errors.")
    elif status == "partial":
        st.warning("Pipeline completed partially. Review the warnings below.")
    else:
        st.error(f"Pipeline status: {status}")

    total_seconds = report_data.get("total_processing_seconds")
    if isinstance(total_seconds, (int, float)):
        st.write(f"**Total processing time:** {total_seconds:.1f} seconds")

    timings = report_data.get("stage_timings_seconds", {})
    if timings:
        timing_rows = [
            {"Stage": stage.replace("_", " ").title(), "Seconds": seconds}
            for stage, seconds in timings.items()
        ]
        st.dataframe(timing_rows, hide_index=True, use_container_width=True)

    errors = report_data.get("processing_errors", [])
    if errors:
        with st.expander(f"Processing warnings/errors ({len(errors)})", expanded=True):
            for error in errors:
                st.warning(
                    f"**{error.get('stage', 'unknown')}** — "
                    f"{error.get('error_type', 'Error')}: "
                    f"{error.get('message', 'No details provided')}"
                )


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

st.title("🎓 Humaniti Classroom Quality Assessment")
st.write(
    "Upload an authorized classroom recording to generate a provisional, "
    "evidence-based assessment and a downloadable report."
)

with st.expander("Privacy and assessment information", expanded=True):
    st.info(
        "Only upload recordings you are authorized to process. This application "
        "may send audio/transcript data to Groq and sampled video frames to Google "
        "Gemini, depending on your configuration. Confirm that your data-handling "
        "permissions allow this before proceeding. The rating is provisional and "
        "must be reviewed by a human."
    )

uploaded_video = st.file_uploader(
    "Choose a classroom video",
    type=SUPPORTED_VIDEO_TYPES,
    help=f"Supported formats: {', '.join(SUPPORTED_VIDEO_TYPES)}. "
         f"Configured upload limit: {MAX_VIDEO_UPLOAD_MB} MB.",
)

if uploaded_video is not None:
    size_mb = uploaded_video.size / (1024 * 1024)
    meta1, meta2 = st.columns(2)
    meta1.write(f"**File:** {uploaded_video.name}")
    meta2.write(f"**Size:** {size_mb:.2f} MB")

    if size_mb > MAX_VIDEO_UPLOAD_MB:
        st.error(
            f"This file is {size_mb:.1f} MB. The configured application limit "
            f"is {MAX_VIDEO_UPLOAD_MB} MB."
        )
        st.stop()

    start_assessment = st.button(
        "Start classroom assessment",
        type="primary",
        use_container_width=True,
    )

    if start_assessment:
        safe_name = _safe_filename(uploaded_video.name)
        suffix = Path(safe_name).suffix.lower() or ".mp4"
        original_stem = Path(safe_name).stem
        unique_stem = f"{original_stem}_{uuid.uuid4().hex[:8]}"

        # Store the upload temporarily so FFmpeg and the pipeline can read it
        # from disk. The original temporary upload is deleted after processing.
        upload_dir = PROJECT_ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temporary_video_path = None

        progress_bar = st.progress(0, text="Preparing upload...")
        status_box = st.status("Preparing assessment...", expanded=True)
        stage_placeholder = st.empty()

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=suffix,
                prefix=f"{unique_stem}_",
                dir=upload_dir,
                delete=False,
            ) as temporary_file:
                temporary_video_path = Path(temporary_file.name)
                temporary_file.write(uploaded_video.getbuffer())

            logger.info(
                "Video upload saved temporarily: name=%s size_mb=%.2f",
                safe_name,
                size_mb,
            )

            def update_progress(stage: str, message: str, percent: int) -> None:
                """Update the Streamlit progress bar from pipeline callbacks."""
                progress_bar.progress(
                    min(99, max(0, int(percent))),
                    text=message,
                )
                stage_placeholder.caption(f"Current stage: {stage.replace('_', ' ').title()}")
                status_box.update(label=message, state="running", expanded=True)

            # Run the existing processing pipeline. The optional callback
            # reports stage-level progress; it does not simulate percentages
            # within long-running FFmpeg or API calls.
            pipeline_result = process_classroom_video(
                temporary_video_path,
                progress_callback=update_progress,
            )

            # Replace the temporary name in the user-facing report with the
            # uploaded filename. The internal unique name prevents collisions.
            pipeline_result.setdefault("video", {})["file_name"] = safe_name

            progress_bar.progress(98, text="Generating assessment report...")
            status_box.update(
                label="Generating assessment report...",
                state="running",
                expanded=True,
            )

            report_name = f"{unique_stem}_assessment_report.md"
            report_path = REPORT_OUTPUT_DIR / report_name
            report_result = generate_markdown_report(
                pipeline_result=pipeline_result,
                output_path=report_path,
            )

            st.session_state["classroom_assessment_result"] = {
                "report_data": report_result["report_data"],
                "markdown": report_result["markdown"],
                "report_path": report_result.get("path"),
                "original_file_name": safe_name,
            }

            progress_bar.progress(100, text="Assessment complete.")
            status_box.update(
                label="Assessment complete.",
                state="complete",
                expanded=False,
            )
            logger.info(
                "Streamlit assessment completed: file=%s pipeline_status=%s "
                "rating=%s source=%s",
                safe_name,
                pipeline_result.get("status"),
                report_result["report_data"]["assessment"].get("provisional_rating"),
                report_result["report_data"]["assessment"].get("assessment_source"),
            )

        except Exception as exc:
            status_box.update(
                label="Assessment failed.",
                state="error",
                expanded=True,
            )
            logger.exception(
                "Streamlit assessment failed: file=%s error_type=%s",
                safe_name,
                type(exc).__name__,
            )
            st.error(
                "The application could not complete the assessment. Check the "
                "terminal logs for the technical details. No transcript content "
                "is written to the application log by this UI."
            )
        finally:
            if temporary_video_path and temporary_video_path.exists():
                try:
                    temporary_video_path.unlink()
                    logger.info(
                        "Temporary uploaded video removed: name=%s",
                        temporary_video_path.name,
                    )
                except OSError:
                    logger.warning(
                        "Could not remove temporary uploaded video: name=%s",
                        temporary_video_path.name,
                        exc_info=True,
                    )

# Render the latest completed report after processing, including on reruns.
saved_result = st.session_state.get("classroom_assessment_result")
if saved_result:
    st.divider()
    st.caption(f"Report for: {saved_result['original_file_name']}")

    report_data = saved_result["report_data"]
    _render_assessment(report_data)

    tab_evidence, tab_visual, tab_processing, tab_report = st.tabs(
        [
            "Transcript evidence",
            "Visual observations",
            "Processing details",
            "Full report",
        ]
    )

    with tab_evidence:
        _render_transcript_evidence(report_data)

    with tab_visual:
        _render_visual_evidence(report_data)

    with tab_processing:
        _render_processing_details(report_data)

    with tab_report:
        st.markdown(saved_result["markdown"])

    st.download_button(
        label="Download Markdown report",
        data=saved_result["markdown"],
        file_name=(
            f"{Path(saved_result['original_file_name']).stem}"
            "_assessment_report.md"
        ),
        mime="text/markdown",
        use_container_width=True,
    )

    st.caption(
        "Generated intermediate audio, frames, and report files may remain in "
        "`data/processed`, `data/frames`, and `data/reports`. Review your retention "
        "policy and clean these locations when no longer needed."
    )
