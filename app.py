"""
Streamlit UI for the Humaniti classroom-quality assessment prototype.

Run from the project root:
    streamlit run app.py

This UI orchestrates existing modules; it does not implement assessment logic.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
import tempfile
import uuid
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests

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


def _friendly_processing_message(message: object, stage: str | None = None) -> str:
    """Return short user-facing wording without leaking provider payloads."""
    raw = str(message or "")
    normalized = raw.lower()
    stage_name = (stage or "").replace("_", " ").strip().lower()

    # Provider error bodies commonly contain JSON, status codes, request IDs,
    # and nested structures. Never render those details directly to users.
    provider_error_markers = (
        "503", "unavailable", "429", "resource_exhausted", "rate limit",
        "quota", "{'error':", '"error":', "{\"error\":", "traceback",
        "api key", "permission_denied", "internal server error",
    )
    if any(marker in normalized for marker in provider_error_markers):
        if "visual" in stage_name or "gemini" in normalized:
            return "Visual analysis is temporarily unavailable. Other results may still be available."
        if "assessment" in stage_name or "groq" in normalized or "llm" in normalized:
            return "AI assessment is temporarily unavailable. A fallback result may be used."
        return "A processing step is temporarily unavailable. The assessment may be incomplete."

    # Do not display arbitrarily long raw exception text even if it does not
    # match a known provider error format.
    if len(raw) > 180 or "{" in raw or "}" in raw:
        if "visual" in stage_name:
            return "Visual analysis could not be completed."
        if "transcri" in stage_name:
            return "Transcription could not be completed."
        if "assessment" in stage_name:
            return "Assessment could not be completed."
        return "This processing step could not be completed."

    return raw.strip() or "This processing step could not be completed."


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
            _friendly_processing_message(
                visual.get("message") or visual.get("status", "unavailable"),
                stage="visual_analysis",
            )
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
                stage = str(error.get("stage", "Processing step"))
                st.warning(
                    f"**{stage.replace('_', ' ').title()}** — "
                    f"{_friendly_processing_message(error.get('message'), stage=stage)}"
                )


# ---------------------------------------------------------------------
# Download, staging, and processing helpers
# ---------------------------------------------------------------------

def _validate_public_https_url(url: str) -> str:
    """Validate a direct public HTTPS URL and reduce SSRF risk.

    Only public HTTPS URLs are accepted. Redirects are validated one by one
    before following them. This is not a replacement for network-level egress
    controls, but it blocks common localhost/private-network URL attempts.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() != "https":
        raise ValueError("Please provide a direct HTTPS video URL.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("The URL is invalid or contains embedded credentials.")

    hostname = parsed.hostname.rstrip(".")
    if hostname.lower() in {"localhost", "localhost.localdomain"}:
        raise ValueError("Localhost URLs are not allowed.")

    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ValueError("The URL hostname could not be resolved.") from exc

    if not addresses:
        raise ValueError("The URL hostname could not be resolved.")

    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("URLs resolving to private or non-public IP addresses are not allowed.")

    return url.strip()


def _extract_google_drive_file_id(url: str) -> str | None:
    """Return the file ID from common Google Drive share-link formats."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host not in {"drive.google.com", "docs.google.com"}:
        return None

    # Examples: /file/d/FILE_ID/view, /uc?id=FILE_ID, /open?id=FILE_ID
    match = re.search(r"/(?:file/)?d/([A-Za-z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)
    query_id = re.search(r"(?:^|&)id=([A-Za-z0-9_-]+)(?:&|$)", parsed.query)
    if query_id:
        return query_id.group(1)
    return None


def _download_video_from_url(url: str, destination: Path, progress_bar, status_box) -> tuple[Path, str, int]:
    """Download a direct video URL or a Google Drive share link safely."""
    input_url = url.strip()
    drive_file_id = _extract_google_drive_file_id(input_url)
    is_drive = drive_file_id is not None

    if is_drive:
        # Google Drive share links are HTML pages, not direct video URLs.
        # Convert the common share-link format to Drive's download endpoint.
        current_url = "https://drive.google.com/uc?export=download&id=" + drive_file_id
    else:
        current_url = input_url

    max_bytes = MAX_VIDEO_UPLOAD_MB * 1024 * 1024
    session = requests.Session()
    session.headers.update({"User-Agent": "Humaniti-Classroom-Quality-App/1.0"})
    response = None

    try:
        # Follow redirects manually so every redirect target is validated.
        # A Google Drive large-file warning may require a confirm token.
        confirmed_drive_download = False
        while True:
            for _ in range(6):
                current_url = _validate_public_https_url(current_url)
                response = session.get(
                    current_url,
                    stream=True,
                    timeout=(10, 60),
                    allow_redirects=False,
                )
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("Location")
                    response.close()
                    response = None
                    if not location:
                        raise ValueError("The video server returned an invalid redirect.")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                break
            else:
                raise ValueError("The video URL redirected too many times.")

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()

            # Google Drive sometimes returns an HTML confirmation page for
            # large files instead of the video bytes. Extract its confirmation token.
            if is_drive and content_type in {"text/html", "application/xhtml+xml"}:
                html = response.text[:2_000_000]
                token = None
                for pattern in (
                    r'name="confirm"\s+value="([^" ]+)"',
                    r'[?&]confirm=([0-9A-Za-z_-]+)',
                ):
                    match = re.search(pattern, html)
                    if match:
                        token = match.group(1)
                        break
                if not token:
                    response.close()
                    response = None
                    raise ValueError(
                        "Google Drive returned a webpage instead of the video. "
                        "Check that General access is set to 'Anyone with the link' "
                        "and that the link points to a downloadable video file."
                    )
                response.close()
                response = None
                if confirmed_drive_download:
                    raise ValueError("Google Drive did not provide a downloadable video after confirmation.")
                confirmed_drive_download = True
                current_url = "https://drive.google.com/uc?export=download&id=" + drive_file_id + "&confirm=" + token
                continue
            break

        content_length = response.headers.get("Content-Length")
        expected_bytes = int(content_length) if content_length and content_length.isdigit() else None
        if expected_bytes is not None and expected_bytes > max_bytes:
            raise ValueError(
                f"The video is larger than the configured {MAX_VIDEO_UPLOAD_MB} MB limit."
            )

        # Prefer the filename supplied by the server, which Google Drive usually
        # includes in Content-Disposition; otherwise use the URL path.
        disposition = response.headers.get("Content-Disposition", "")
        filename_match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', disposition, re.IGNORECASE)
        server_name = unquote(filename_match.group(1)).strip() if filename_match else ""
        parsed_path = unquote(urlparse(current_url).path)
        url_name = server_name or Path(parsed_path).name or "video_from_url.mp4"
        safe_name = _safe_filename(url_name)
        suffix = Path(safe_name).suffix.lower()
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()

        mime_to_suffix = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/x-matroska": ".mkv",
            "video/x-msvideo": ".avi",
            "video/webm": ".webm",
            "application/octet-stream": suffix if suffix in {f".{ext}" for ext in SUPPORTED_VIDEO_TYPES} else "",
            "binary/octet-stream": suffix if suffix in {f".{ext}" for ext in SUPPORTED_VIDEO_TYPES} else "",
        }
        allowed_suffixes = {f".{ext}" for ext in SUPPORTED_VIDEO_TYPES}
        if suffix not in allowed_suffixes:
            inferred_suffix = mime_to_suffix.get(content_type, "")
            if not inferred_suffix and is_drive:
                # The download endpoint may omit both a filename extension and
                # a specific MIME type. Use .mp4 as a temporary suffix; FFmpeg
                # will still validate the actual media contents downstream.
                inferred_suffix = ".mp4"
            if not inferred_suffix:
                raise ValueError(
                    "The URL did not return a recognizable supported video file. "
                    "For Google Drive, use a share link to a video file and set "
                    "General access to 'Anyone with the link'."
                )
            suffix = inferred_suffix
            safe_name = f"video_from_url{suffix}"

        if content_type and not content_type.startswith("video/") and content_type not in {
            "application/octet-stream", "binary/octet-stream"
        }:
            raise ValueError(
                f"The URL returned '{content_type}', not a video file. "
                "For Google Drive, verify that the link is shared for download."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        with destination.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > max_bytes:
                    raise ValueError(
                        f"The downloaded video exceeded the configured {MAX_VIDEO_UPLOAD_MB} MB limit."
                    )
                output_file.write(chunk)
                if expected_bytes:
                    percent = min(100, int(downloaded * 100 / expected_bytes))
                    progress_bar.progress(percent, text=f"Downloading video: {percent}%")
                    status_box.update(
                        label=f"Downloading video: {downloaded / (1024 * 1024):.1f} / "
                              f"{expected_bytes / (1024 * 1024):.1f} MB",
                        state="running",
                        expanded=True,
                    )
                else:
                    progress_bar.progress(0, text="Downloading video (server did not provide file size)...")
                    status_box.update(
                        label=f"Downloaded {downloaded / (1024 * 1024):.1f} MB",
                        state="running",
                        expanded=True,
                    )

        if downloaded == 0:
            raise ValueError("The URL returned an empty file.")
        progress_bar.progress(100, text="Video download complete.")
        return destination, safe_name, downloaded
    finally:
        if response is not None:
            response.close()
        session.close()


def _copy_uploaded_video(uploaded_video, destination: Path, progress_bar, status_box) -> int:
    """Copy Streamlit's already-received upload to disk and show staging progress.

    Streamlit's standard file_uploader controls the browser-to-server upload UI;
    this progress bar tracks copying the received file into the app's working folder.
    """
    total_bytes = uploaded_video.size
    written = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    buffer = uploaded_video.getbuffer()
    chunk_size = 1024 * 1024
    with destination.open("wb") as output_file:
        for offset in range(0, total_bytes, chunk_size):
            chunk = buffer[offset:offset + chunk_size]
            output_file.write(chunk)
            written += len(chunk)
            percent = int(written * 100 / total_bytes) if total_bytes else 100
            progress_bar.progress(percent, text=f"Preparing selected video: {percent}%")
            status_box.update(
                label=f"Preparing video: {written / (1024 * 1024):.1f} / "
                      f"{total_bytes / (1024 * 1024):.1f} MB",
                state="running",
                expanded=True,
            )
    return written


def _run_assessment(video_path: Path, original_name: str, size_bytes: int) -> None:
    """Run the existing media pipeline and render the report when complete."""
    safe_name = _safe_filename(original_name)
    unique_stem = f"{Path(safe_name).stem}_{uuid.uuid4().hex[:8]}"
    temporary_video_path = video_path
    progress_bar = st.progress(0, text="Starting classroom assessment...")
    status_box = st.status("Preparing assessment...", expanded=True)
    stage_placeholder = st.empty()

    try:
        size_mb = size_bytes / (1024 * 1024)

        def update_progress(stage: str, message: str, percent: int) -> None:
            """Display stage-level processing progress reported by the pipeline."""
            progress_bar.progress(min(99, max(0, int(percent))), text=message)
            stage_placeholder.caption(f"Current stage: {stage.replace('_', ' ').title()}")
            status_box.update(label=message, state="running", expanded=True)

        pipeline_result = process_classroom_video(
            temporary_video_path,
            progress_callback=update_progress,
        )
        pipeline_result.setdefault("video", {})["file_name"] = safe_name

        progress_bar.progress(98, text="Generating assessment report...")
        status_box.update(label="Generating assessment report...", state="running", expanded=True)

        REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        status_box.update(label="Assessment complete.", state="complete", expanded=False)
        logger.info(
            "Assessment completed: file=%s size_mb=%.2f status=%s source=%s",
            safe_name,
            size_mb,
            pipeline_result.get("status"),
            report_result["report_data"]["assessment"].get("assessment_source"),
        )
    except Exception as exc:
        status_box.update(label="Assessment failed.", state="error", expanded=True)
        logger.exception("Assessment failed: file=%s error_type=%s", safe_name, type(exc).__name__)
        st.error(
            "The application could not complete the assessment. Check the Streamlit app logs "
            "for technical details."
        )
    finally:
        if temporary_video_path and temporary_video_path.exists():
            try:
                temporary_video_path.unlink()
            except OSError:
                logger.warning("Could not remove temporary video file.", exc_info=True)


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

st.title("🎓 Humaniti Classroom Quality Assessment")
st.write(
    "Upload an authorized classroom recording or provide a direct video URL "
    "to generate a provisional, evidence-based assessment and downloadable report."
)

with st.expander("Privacy and assessment information", expanded=True):
    st.info(
        "Only process recordings you are authorized to use. "
        "Media and transcript data may be processed by external services. "
        "Video links must be publicly accessible or shared with suitable permissions. "
        "Assessment results are provisional and should be reviewed by a human."
    )

upload_tab, url_tab = st.tabs(["Upload video", "Video from URL"])

with upload_tab:
    st.caption(
        f"Supported formats: {', '.join(SUPPORTED_VIDEO_TYPES)}. "
        f"Maximum size: {MAX_VIDEO_UPLOAD_MB} MB."
    )
    uploaded_video = st.file_uploader(
        "Choose a classroom video",
        type=SUPPORTED_VIDEO_TYPES,
        help=(
            "Select an MP4, MOV, MKV, AVI, or WEBM file. Streamlit shows its own "
            "browser upload indicator; after selection, this app shows progress while "
            "preparing the file and processing the assessment."
        ),
        key="classroom_video_upload",
    )

    if uploaded_video is not None:
        size_mb = uploaded_video.size / (1024 * 1024)
        col1, col2 = st.columns(2)
        col1.write(f"**File:** {uploaded_video.name}")
        col2.write(f"**Size:** {size_mb:.2f} MB")

        if size_mb > MAX_VIDEO_UPLOAD_MB:
            st.error(
                f"This file is {size_mb:.1f} MB. The configured application limit "
                f"is {MAX_VIDEO_UPLOAD_MB} MB."
            )
        elif st.button("Start classroom assessment", type="primary", key="assess_uploaded_video"):
            safe_name = _safe_filename(uploaded_video.name)
            suffix = Path(safe_name).suffix.lower() or ".mp4"
            upload_dir = PROJECT_ROOT / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            temp_path = upload_dir / f"{uuid.uuid4().hex}_{safe_name}"
            progress = st.progress(0, text="Preparing selected video...")
            status = st.status("Preparing video...", expanded=True)
            try:
                written = _copy_uploaded_video(uploaded_video, temp_path, progress, status)
                logger.info("Selected video staged: name=%s size_mb=%.2f", safe_name, written / (1024 * 1024))
                _run_assessment(temp_path, safe_name, written)
            except Exception as exc:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                logger.exception("Could not prepare uploaded video: %s", type(exc).__name__)
                st.error("Could not prepare the uploaded video. Please check the file and try again.")

with url_tab:
    st.write("Paste a direct video-download URL or a Google Drive share link to a video file.")
    video_url = st.text_input(
        "Video URL",
        placeholder="https://example.com/video.mp4 or https://drive.google.com/file/d/FILE_ID/view",
        key="classroom_video_url",
        help="Supports direct MP4/MOV/MKV/AVI/WEBM URLs and Google Drive share links with access set to Anyone with the link. YouTube/Vimeo pages are not supported.",
    )
    if st.button("Download video and assess", type="primary", key="assess_video_url"):
        if not video_url.strip():
            st.warning("Enter a video URL first.")
        else:
            upload_dir = PROJECT_ROOT / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            temp_path = upload_dir / f"{uuid.uuid4().hex}_video_from_url.mp4"
            progress = st.progress(0, text="Connecting to video URL...")
            status = st.status("Connecting to video URL...", expanded=True)
            try:
                downloaded_path, original_name, downloaded_bytes = _download_video_from_url(
                    video_url, temp_path, progress, status
                )
                logger.info(
                    "Video downloaded from URL: name=%s size_mb=%.2f",
                    original_name,
                    downloaded_bytes / (1024 * 1024),
                )
                _run_assessment(downloaded_path, original_name, downloaded_bytes)
            except Exception as exc:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                logger.exception("URL video download/assessment failed: %s", type(exc).__name__)
                st.error(
                    "Could not download the video. Check the link permissions and file format, "
                    "then try again."
                )

# Render the latest completed report after processing, including on reruns.
saved_result = st.session_state.get("classroom_assessment_result")
if saved_result:
    st.divider()
    st.caption(f"Report for: {saved_result['original_file_name']}")

    report_data = saved_result["report_data"]
    _render_assessment(report_data)

    tab_evidence, tab_visual, tab_processing, tab_report = st.tabs(
        ["Transcript evidence", "Visual observations", "Processing details", "Full report"]
    )
    with tab_evidence:
        _render_transcript_evidence(report_data)
    with tab_visual:
        _render_visual_evidence(report_data)
    with tab_processing:
        _render_processing_details(report_data)
    with tab_report:
        st.markdown(saved_result["markdown"])

    # Make the report download action visually distinct while keeping it below the tabs.
    st.markdown(
        """
        <style>
        div[data-testid="stDownloadButton"] > button {
            background: linear-gradient(90deg, #1769aa, #0d8b8b) !important;
            color: #ffffff !important;
            border: 0 !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            min-height: 3rem !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.18) !important;
            transition: filter 0.15s ease-in-out, transform 0.15s ease-in-out;
        }
        div[data-testid="stDownloadButton"] > button:hover {
            filter: brightness(1.08);
            transform: translateY(-1px);
        }
        div[data-testid="stDownloadButton"] > button p {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        label="⬇️  Download assessment report",
        data=saved_result["markdown"],
        file_name=f"{Path(saved_result['original_file_name']).stem}_assessment_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    st.caption(
        "Generated intermediate audio, frames, and reports may remain in the app's temporary "
        "working directories. Apply an appropriate retention and cleanup policy."
    )
