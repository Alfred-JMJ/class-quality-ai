"""
Evidence-based assessment prototype for the classroom-quality evaluator.

This module combines available transcript and visual observations into a
transparent, provisional assessment. It does NOT claim to measure actual
student learning, engagement, or teaching effectiveness with certainty.

The initial scoring is heuristic and should be validated against a human-rated
sample before being used for consequential decisions.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

# Load .env relative to the project root, regardless of the current folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """Read a float setting from .env and fall back safely if invalid."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid float for %s=%r; using default %.3f",
            name,
            raw_value,
            default,
        )
        return default


def _env_int(name: str, default: int) -> int:
    """Read an integer setting from .env and fall back safely if invalid."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid integer for %s=%r; using default %d",
            name,
            raw_value,
            default,
        )
        return default


# Criterion weights are configurable. Defaults are provisional examples,
# not a validated or externally mandated classroom-quality rubric.
CRITERION_WEIGHTS = {
    "explanation_and_examples": max(
        0.0, _env_float("ASSESSMENT_WEIGHT_EXPLANATION", 0.25)
    ),
    "lesson_structure": max(
        0.0, _env_float("ASSESSMENT_WEIGHT_STRUCTURE", 0.20)
    ),
    "questioning": max(
        0.0, _env_float("ASSESSMENT_WEIGHT_QUESTIONING", 0.20)
    ),
    "feedback_and_guidance": max(
        0.0, _env_float("ASSESSMENT_WEIGHT_FEEDBACK", 0.20)
    ),
    "teaching_materials": max(
        0.0, _env_float("ASSESSMENT_WEIGHT_MATERIALS", 0.15)
    ),
}

# Rating thresholds use the weighted average of criterion scores (0-4).
# These are provisional thresholds and should be calibrated with reviewers.
RATING_POOR_MAX = _env_float("ASSESSMENT_POOR_MAX", 1.0)
RATING_AVERAGE_MAX = _env_float("ASSESSMENT_AVERAGE_MAX", 2.0)
RATING_GOOD_MAX = _env_float("ASSESSMENT_GOOD_MAX", 3.0)

# Do not assign a rating if the transcript is too short to support a useful
# transcript-based assessment. A visual-only result is not enough here.
MIN_TRANSCRIPT_CHARACTERS = max(
    0, _env_int("ASSESSMENT_MIN_TRANSCRIPT_CHARACTERS", 200)
)

# Keep evidence excerpts concise in logs and output.
MAX_EVIDENCE_PER_CRITERION = max(
    1, _env_int("ASSESSMENT_MAX_EVIDENCE_PER_CRITERION", 4)
)
MAX_EVIDENCE_TEXT_LENGTH = max(
    80, _env_int("ASSESSMENT_MAX_EVIDENCE_TEXT_LENGTH", 240)
)


# ---------------------------------------------------------------------
# 2. Observable evidence indicators
# ---------------------------------------------------------------------
# These patterns identify possible evidence in a transcript. They do not
# prove that a teaching practice was effective. For example, a question in
# the transcript does not prove that students answered it or understood it.

CRITERION_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "explanation_and_examples": [
        (
            "example_language",
            r"\b(for example|for instance|let me show you an example)\b",
        ),
        (
            "stepwise_explanation",
            r"\b(first|second|third|step by step|the next step is)\b",
        ),
        (
            "definition_language",
            r"\b(means|is defined as|we call this|the idea is)\b",
        ),
        (
            "reasoning_language",
            r"\b(because|therefore|this is why|the reason is)\b",
        ),
    ],
    "lesson_structure": [
        (
            "opening_or_objective",
            r"\b(today we will|today we're going to|our objective|the goal today)\b",
        ),
        (
            "sequence_or_transition",
            r"\b(now let's move on|next we will|moving on to|let's recap|to summarize)\b",
        ),
        (
            "review_or_summary",
            r"\b(in summary|to recap|let's review|what we learned|key takeaway)\b",
        ),
        (
            "task_instruction",
            r"\b(please try|your task is|I want you to|let's practice)\b",
        ),
    ],
    "questioning": [
        (
            "open_question",
            r"\b(why do you think|how would you|what do you notice|can you explain why)\b",
        ),
        (
            "checking_question",
            r"\b(does that make sense|any questions|what is the answer|who can tell me)\b",
        ),
        (
            "question_mark",
            r"\?",
        ),
        (
            "reasoning_question",
            r"\b(what would happen if|how can we tell|what is your reasoning)\b",
        ),
    ],
    "feedback_and_guidance": [
        (
            "affirming_feedback",
            r"\b(well done|good job|that's correct|that is correct|excellent work)\b",
        ),
        (
            "corrective_guidance",
            r"\b(not quite|try again|check your answer|let's correct|the mistake is)\b",
        ),
        (
            "explanation_of_feedback",
            r"\b(that is correct because|this works because|you did this correctly because)\b",
        ),
        (
            "guided_support",
            r"\b(let's work through|let me help you|here is a hint|think about it this way)\b",
        ),
    ],
}


# ---------------------------------------------------------------------
# 3. Input normalization helpers
# ---------------------------------------------------------------------

def _extract_transcript(transcription: Any) -> str:
    """Accept the pipeline transcription dictionary or a plain string."""
    if isinstance(transcription, str):
        return transcription.strip()

    if isinstance(transcription, dict):
        transcript_text = transcription.get("transcript", "")
        if isinstance(transcript_text, str):
            return transcript_text.strip()

        # Some transcription providers may return text only inside segments.
        segments = transcription.get("segments", [])
        if isinstance(segments, list):
            pieces = [
                str(segment.get("text", "")).strip()
                for segment in segments
                if isinstance(segment, dict) and segment.get("text")
            ]
            return " ".join(pieces).strip()

    return ""


def _extract_visual_observations(visual_analysis: Any) -> dict:
    """Return Gemini's observations only when its result indicates success."""
    if not isinstance(visual_analysis, dict):
        return {}

    if visual_analysis.get("status") != "success":
        return {}

    observations = visual_analysis.get("observations", {})
    return observations if isinstance(observations, dict) else {}


def _find_evidence(
    transcript: str,
    patterns: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """
    Find distinct indicator types and concise transcript excerpts.

    Evidence excerpts are taken from transcript sentences around each match.
    Pattern matches are indicators only, not proof of quality.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n+", transcript)
    matched_indicators: list[str] = []
    excerpts: list[str] = []

    for indicator_name, pattern in patterns:
        found_sentence = None

        for sentence in sentences:
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                found_sentence = sentence.strip()
                break

        if found_sentence is not None:
            matched_indicators.append(indicator_name)
            if found_sentence not in excerpts:
                excerpt = re.sub(r"\s+", " ", found_sentence)
                if len(excerpt) > MAX_EVIDENCE_TEXT_LENGTH:
                    excerpt = excerpt[: MAX_EVIDENCE_TEXT_LENGTH - 3].rstrip() + "..."
                excerpts.append(excerpt)

    return matched_indicators, excerpts[:MAX_EVIDENCE_PER_CRITERION]


def _score_from_indicator_count(indicator_count: int) -> int:
    """
    Map distinct detected indicator types to a 0-4 evidence score.

    This intentionally simple mapping is a baseline, not a validated
    pedagogical scoring model. Zero means "not detected", not "did not occur".
    """
    return min(4, max(0, indicator_count))


def _visual_materials_evidence(observations: dict) -> dict:
    """Create a materials criterion from explicitly returned visual evidence."""
    materials = observations.get("teaching_materials", {})
    if not isinstance(materials, dict):
        materials = {}

    status = str(materials.get("status", "uncertain")).lower()
    evidence_text = str(materials.get("evidence", "")).strip()

    # Visibility alone is not evidence that materials were used effectively.
    # Therefore, this prototype reports the observation but does not assign
    # a numeric teaching-quality score to this criterion yet.
    if status == "visible" and evidence_text:
        return {
            "score_0_to_4": None,
            "detected_indicators": ["materials_visible"],
            "evidence": [evidence_text],
            "interpretation": (
                "Teaching materials were visible in sampled frames. "
                "Their relevance, accessibility, and instructional quality "
                "cannot be established from visibility alone."
            ),
            "evidence_status": "observed_not_scored",
        }

    if status == "not_visible":
        return {
            "score_0_to_4": None,
            "detected_indicators": [],
            "evidence": [],
            "interpretation": (
                "Materials were not visible in the sampled frames; this does "
                "not establish that materials were absent from the lesson."
            ),
            "evidence_status": "not_observed",
        }

    return {
        "score_0_to_4": None,
        "detected_indicators": [],
        "evidence": [],
        "interpretation": (
            "Visual evidence for teaching materials is unavailable or uncertain."
        ),
        "evidence_status": "unavailable",
    }


# ---------------------------------------------------------------------
# 4. Main assessment function
# ---------------------------------------------------------------------

def assess_classroom_quality(
    transcription: Any,
    visual_analysis: Any = None,
    audio_analysis: Any = None,
    silence_analysis: Any = None,
) -> dict:
    """
    Produce a transparent, provisional assessment from available evidence.

    Args:
        transcription:
            The pipeline's transcription result dictionary or transcript text.
        visual_analysis:
            The pipeline's visual_analysis result dictionary, if available.
        audio_analysis:
            Optional audio quality analysis. Used for context/limitations,
            not as a direct teaching-quality score.
        silence_analysis:
            Optional silence-detection result. Used for context/limitations,
            not as a direct teaching-quality score.

    Returns:
        A JSON-serializable dictionary with criterion-level evidence,
        a provisional rating where evidence is sufficient, and limitations.
    """
    logger.info("Starting classroom-quality assessment.")

    transcript = _extract_transcript(transcription)
    observations = _extract_visual_observations(visual_analysis)
    limitations: list[str] = [
        "This is a heuristic prototype, not a validated measure of teaching quality.",
        "Transcript indicators can miss paraphrases, language variation, and context.",
        "A detected question or example does not prove student understanding.",
        "Audio levels and detected silence are not direct measures of teaching quality.",
        "Sampled frames cannot establish continuous student engagement or learning.",
    ]

    # Insufficient transcript should not be converted into a low rating.
    if len(transcript) < MIN_TRANSCRIPT_CHARACTERS:
        logger.warning(
            "Assessment unavailable: transcript has %d characters; minimum is %d.",
            len(transcript),
            MIN_TRANSCRIPT_CHARACTERS,
        )
        return {
            "status": "insufficient_evidence",
            "provisional_rating": None,
            "overall_score_0_to_4": None,
            "criteria": {},
            "evidence_summary": (
                "There is not enough transcript text to generate a "
                "meaningful provisional assessment."
            ),
            "limitations": limitations + [
                "A transcript was missing or shorter than the configured minimum."
            ],
            "configuration": {
                "minimum_transcript_characters": MIN_TRANSCRIPT_CHARACTERS,
            },
        }

    criteria: dict[str, dict] = {}

    # Score transcript-based criteria using distinct detected indicator types.
    for criterion_name, patterns in CRITERION_PATTERNS.items():
        indicators, evidence = _find_evidence(transcript, patterns)

        # No detected phrase is not proof of poor teaching. Mark the criterion
        # as unscored when no configured indicator was found.
        score = (
            _score_from_indicator_count(len(indicators))
            if indicators
            else None
        )

        criteria[criterion_name] = {
            "score_0_to_4": score,
            "detected_indicators": indicators,
            "evidence": evidence,
            "evidence_status": "observed" if evidence else "not_detected_unscored",
            "interpretation": (
                "Indicators were found in the transcript. Review the excerpts "
                "in context; this score does not establish effectiveness."
                if evidence
                else
                "No configured indicators were detected. This does not prove "
                "the practice was absent."
            ),
        }

    # Visual materials are optional. Missing visual output is treated as
    # unavailable evidence, not as a negative score.
    criteria["teaching_materials"] = _visual_materials_evidence(observations)

    # Include only criteria with available scores in the weighted average.
    # Weights are re-normalized across available criteria.
    weighted_total = 0.0
    available_weight_total = 0.0

    for criterion_name, weight in CRITERION_WEIGHTS.items():
        criterion = criteria.get(criterion_name, {})
        score = criterion.get("score_0_to_4")

        if score is None or weight <= 0:
            continue

        weighted_total += score * weight
        available_weight_total += weight

    if available_weight_total <= 0:
        overall_score = None
        rating = None
        status = "insufficient_evidence"
        limitations.append(
            "No configured transcript indicators were detected, so no rating "
            "was assigned. This does not mean teaching quality was poor."
        )
    else:
        overall_score = round(weighted_total / available_weight_total, 2)

        if overall_score <= RATING_POOR_MAX:
            rating = "Poor"
        elif overall_score <= RATING_AVERAGE_MAX:
            rating = "Average"
        elif overall_score <= RATING_GOOD_MAX:
            rating = "Good"
        else:
            rating = "Excellent"

        status = "provisional"

    # Audio data is reported as context only. It does not change the score.
    audio_context = {}
    if isinstance(audio_analysis, dict):
        audio_context = {
            key: audio_analysis.get(key)
            for key in (
                "duration_seconds",
                "overall_rms_dbfs",
                "low_level_audio_ratio",
                "clipping_sample_ratio",
            )
            if key in audio_analysis
        }

    silence_context = {}
    if isinstance(silence_analysis, dict):
        silence_context = {
            key: silence_analysis.get(key)
            for key in (
                "total_detected_seconds",
                "interval_count",
                "enabled",
            )
            if key in silence_analysis
        }

    if not observations:
        limitations.append(
            "Visual analysis was unavailable; visual criteria were not scored."
        )

    logger.info(
        "Assessment finished: status=%s rating=%s score=%s",
        status,
        rating,
        overall_score,
    )

    return {
        "status": status,
        "provisional_rating": rating,
        "overall_score_0_to_4": overall_score,
        "criteria": criteria,
        "evidence_summary": (
            "The rating is a provisional heuristic summary of detected "
            "transcript indicators and any available visual evidence. "
            "Review the criterion evidence before interpreting the rating."
        ),
        "audio_context_not_scored": audio_context,
        "silence_context_not_scored": silence_context,
        "limitations": limitations,
        "configuration": {
            "criterion_weights": CRITERION_WEIGHTS,
            "rating_thresholds": {
                "poor_max": RATING_POOR_MAX,
                "average_max": RATING_AVERAGE_MAX,
                "good_max": RATING_GOOD_MAX,
            },
            "minimum_transcript_characters": MIN_TRANSCRIPT_CHARACTERS,
        },
    }
