"""
LLM-assisted classroom assessment with a deterministic fallback chain.

Priority:
    1. Groq LLM assessment
    2. Gemini LLM assessment
    3. Existing regex/heuristic assessment in src.assessment_engine

The regex engine is always run first to create transparent supporting evidence.
Its result is used as the final fallback if both LLM providers fail.

This is a provisional assessment aid, not a validated pedagogical instrument.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load project .env reliably regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

# Import the already implemented regex/heuristic engine.
# This is the actual regex fallback; it is not implemented in this file.
from src.assessment_engine import assess_classroom_quality


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Keep assessment models separate from vision-analysis model configuration.
# Set these to model IDs currently available to your accounts.
GROQ_ASSESSMENT_MODEL = os.getenv(
    "GROQ_ASSESSMENT_MODEL",
    "llama-3.1-8b-instant",
).strip()

GEMINI_ASSESSMENT_MODEL = os.getenv(
    "GEMINI_ASSESSMENT_MODEL",
    "gemini-2.5-flash",
).strip()

MAX_TRANSCRIPT_CHARS_FOR_LLM = max(
    1000,
    int(os.getenv("ASSESSMENT_MAX_TRANSCRIPT_CHARS", "12000")),
)
LLM_MAX_OUTPUT_TOKENS = max(
    500,
    int(os.getenv("ASSESSMENT_LLM_MAX_OUTPUT_TOKENS", "1800")),
)
LLM_TEMPERATURE = float(os.getenv("ASSESSMENT_LLM_TEMPERATURE", "0.2"))
MIN_TRANSCRIPT_CHARS_FOR_LLM = max(
    0,
    int(os.getenv("ASSESSMENT_LLM_MIN_TRANSCRIPT_CHARACTERS", "200")),
)

VALID_RATINGS = {"Poor", "Average", "Good", "Excellent"}
CRITERION_NAMES = (
    "explanation_and_examples",
    "lesson_structure",
    "questioning",
    "feedback_and_guidance",
    "teaching_materials",
)


# ---------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an evidence-based evaluator of a recorded teaching session.

Assess only practices supported by the provided transcript and optional
visual observations. Return one valid JSON object and no Markdown fences.

Use this exact top-level structure:
{
  "provisional_rating": "Poor | Average | Good | Excellent",
  "overall_score_0_to_4": 0.0,
  "criteria": {
    "explanation_and_examples": {
      "score_0_to_4": 0,
      "evidence": ["short transcript quote or observation"],
      "interpretation": "One concise evidence-based sentence"
    },
    "lesson_structure": {
      "score_0_to_4": 0,
      "evidence": [],
      "interpretation": "One concise sentence"
    },
    "questioning": {
      "score_0_to_4": 0,
      "evidence": [],
      "interpretation": "One concise sentence"
    },
    "feedback_and_guidance": {
      "score_0_to_4": 0,
      "evidence": [],
      "interpretation": "One concise sentence"
    },
    "teaching_materials": {
      "score_0_to_4": null,
      "evidence": [],
      "interpretation": "One concise sentence"
    }
  },
  "specific_observations": ["Concrete observation grounded in evidence"],
  "limitations": ["What cannot be concluded from the supplied evidence"]
}

Scoring guidance:
- 0 = evidence of absence only when the evidence genuinely supports that;
  do not use 0 merely because a phrase was not detected.
- 1 = limited evidence, 2 = basic evidence, 3 = strong evidence,
  4 = consistently strong evidence.
- Use null when there is not enough evidence to score a criterion.
- Quote only words present in the supplied transcript. Do not invent quotes.
- The regex/heuristic evidence is a search aid, not ground truth.
- A question in the transcript does not prove students answered or understood.
- Visible materials do not prove they were used effectively.
- Do not infer student engagement, learning, identity, age, or emotions
  from limited visual observations.
- Audio volume and silence duration are context only, not direct quality scores.
- Keep the overall score consistent with the available criterion scores.
- If evidence is limited, say so and avoid an overconfident rating.
"""


# ---------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------

def _extract_transcript(transcription: Any) -> str:
    """Accept a transcript string or the pipeline's transcription result."""
    if isinstance(transcription, str):
        return transcription.strip()

    if isinstance(transcription, dict):
        transcript = transcription.get("transcript", "")
        if isinstance(transcript, str) and transcript.strip():
            return transcript.strip()

        segments = transcription.get("segments", [])
        if isinstance(segments, list):
            return " ".join(
                str(segment.get("text", "")).strip()
                for segment in segments
                if isinstance(segment, dict) and segment.get("text")
            ).strip()

    return ""


def _prepare_transcript_for_llm(transcript: str) -> str:
    """
    Keep the beginning and end of a long transcript to fit within a
    configurable character budget. This may omit important middle evidence,
    so the final assessment must disclose that truncation occurred.
    """
    if len(transcript) <= MAX_TRANSCRIPT_CHARS_FOR_LLM:
        return transcript

    half = MAX_TRANSCRIPT_CHARS_FOR_LLM // 2
    logger.warning(
        "Transcript truncated for LLM assessment: original_chars=%d limit=%d",
        len(transcript),
        MAX_TRANSCRIPT_CHARS_FOR_LLM,
    )
    return (
        transcript[:half]
        + "\n\n...[middle portion omitted for input length]...\n\n"
        + transcript[-half:]
    )


def _get_visual_observations(visual_analysis: Any) -> dict:
    """Return observations only when the visual-analysis stage succeeded."""
    if not isinstance(visual_analysis, dict):
        return {}
    if visual_analysis.get("status") != "success":
        return {}

    observations = visual_analysis.get("observations", {})
    return observations if isinstance(observations, dict) else {}


def _select_audio_context(audio_analysis: Any) -> dict:
    """Include useful audio diagnostics as context, never as quality scores."""
    if not isinstance(audio_analysis, dict):
        return {}

    keys = (
        "duration_seconds",
        "overall_rms_dbfs",
        "low_level_audio_ratio",
        "clipping_sample_ratio",
    )
    return {key: audio_analysis[key] for key in keys if key in audio_analysis}


def _select_silence_context(silence_analysis: Any) -> dict:
    """Include silence diagnostics as context only."""
    if not isinstance(silence_analysis, dict):
        return {}

    keys = ("total_detected_seconds", "interval_count", "enabled")
    return {key: silence_analysis[key] for key in keys if key in silence_analysis}


def _build_user_prompt(
    transcript: str,
    heuristic_result: dict,
    visual_observations: dict,
    audio_context: dict,
    silence_context: dict,
) -> str:
    """Build the common prompt used by Groq and Gemini."""
    heuristic_summary = {
        "status": heuristic_result.get("status"),
        "criteria": heuristic_result.get("criteria", {}),
        "provisional_rating": heuristic_result.get("provisional_rating"),
        "overall_score_0_to_4": heuristic_result.get("overall_score_0_to_4"),
    }

    sections = [
        "=== TRANSCRIPT ===",
        transcript,
        "\n=== REGEX / HEURISTIC EVIDENCE (supporting only; may be incomplete) ===",
        json.dumps(heuristic_summary, ensure_ascii=False),
    ]

    if visual_observations:
        sections.extend([
            "\n=== VISUAL OBSERVATIONS ===",
            json.dumps(visual_observations, ensure_ascii=False),
        ])
    else:
        sections.extend([
            "\n=== VISUAL OBSERVATIONS ===",
            "Unavailable. Do not invent visual evidence.",
        ])

    if audio_context:
        sections.extend([
            "\n=== AUDIO CONTEXT (not scored directly) ===",
            json.dumps(audio_context, ensure_ascii=False),
        ])

    if silence_context:
        sections.extend([
            "\n=== SILENCE DETECTION CONTEXT (not scored directly) ===",
            json.dumps(silence_context, ensure_ascii=False),
        ])

    sections.append(
        "\nAssess the lesson using only the evidence above. "
        "Return the required JSON object."
    )
    return "\n".join(sections)


# ---------------------------------------------------------------------
# LLM response validation
# ---------------------------------------------------------------------

def _validate_llm_result(payload: Any) -> dict:
    """
    Validate essential fields before accepting an LLM response.
    Raise ValueError for malformed responses so the next provider is tried.
    """
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON must be an object.")

    rating = payload.get("provisional_rating")
    if rating not in VALID_RATINGS:
        raise ValueError(f"Invalid provisional_rating: {rating!r}")

    score = payload.get("overall_score_0_to_4")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("overall_score_0_to_4 must be numeric.")
    if not 0 <= float(score) <= 4:
        raise ValueError("overall_score_0_to_4 must be between 0 and 4.")

    criteria = payload.get("criteria")
    if not isinstance(criteria, dict):
        raise ValueError("criteria must be a JSON object.")

    normalized_criteria = {}
    for name in CRITERION_NAMES:
        item = criteria.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"Missing or invalid criterion: {name}")

        criterion_score = item.get("score_0_to_4")
        if criterion_score is not None:
            if (
                isinstance(criterion_score, bool)
                or not isinstance(criterion_score, (int, float))
                or not 0 <= float(criterion_score) <= 4
            ):
                raise ValueError(f"Invalid score for criterion {name}")

        evidence = item.get("evidence", [])
        if not isinstance(evidence, list) or not all(
            isinstance(entry, str) for entry in evidence
        ):
            raise ValueError(f"Invalid evidence list for criterion {name}")

        interpretation = item.get("interpretation", "")
        if not isinstance(interpretation, str):
            raise ValueError(f"Invalid interpretation for criterion {name}")

        normalized_criteria[name] = {
            "score_0_to_4": criterion_score,
            "evidence": evidence,
            "interpretation": interpretation,
        }

    observations = payload.get("specific_observations", [])
    limitations = payload.get("limitations", [])
    if not isinstance(observations, list) or not all(
        isinstance(item, str) for item in observations
    ):
        raise ValueError("specific_observations must be a list of strings.")
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) for item in limitations
    ):
        raise ValueError("limitations must be a list of strings.")

    return {
        "provisional_rating": rating,
        "overall_score_0_to_4": round(float(score), 2),
        "criteria": normalized_criteria,
        "specific_observations": observations,
        "limitations": limitations,
    }


def _parse_json_response(response_text: str) -> dict:
    """Parse and validate a provider response, including basic fence cleanup."""
    if not response_text or not response_text.strip():
        raise ValueError("LLM returned an empty response.")

    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    payload = json.loads(cleaned)
    return _validate_llm_result(payload)


# ---------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------

def _call_groq(prompt: str) -> dict | None:
    """Call Groq. Return None on any API/import/response validation failure."""
    if not GROQ_API_KEY:
        logger.info("Groq assessment skipped: GROQ_API_KEY is not configured.")
        return None

    try:
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_ASSESSMENT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        result = _parse_json_response(content or "")
        logger.info("LLM assessment succeeded with Groq.")
        return result

    except Exception as exc:
        # Log the provider and error type, not the transcript or API key.
        logger.warning(
            "Groq assessment failed; will try Gemini if configured. "
            "error_type=%s detail=%s",
            type(exc).__name__,
            str(exc)[:240],
        )
        return None


def _call_gemini(prompt: str) -> dict | None:
    """Call Gemini using the current google-genai SDK."""
    if not GEMINI_API_KEY:
        logger.info("Gemini assessment skipped: GEMINI_API_KEY is not configured.")
        return None

    client = None
    try:
        # This is the same SDK family used by video_analyzer_gemini.py.
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_ASSESSMENT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            ),
        )
        result = _parse_json_response(response.text or "")
        logger.info("LLM assessment succeeded with Gemini.")
        return result

    except Exception as exc:
        logger.warning(
            "Gemini assessment failed; will use regex fallback. "
            "error_type=%s detail=%s",
            type(exc).__name__,
            str(exc)[:240],
        )
        return None

    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("Could not close Gemini client cleanly.")


# ---------------------------------------------------------------------
# Public assessment function: Groq -> Gemini -> regex
# ---------------------------------------------------------------------

def llm_assess_classroom_quality(
    transcription: Any,
    visual_analysis: Any = None,
    audio_analysis: Any = None,
    silence_analysis: Any = None,
    heuristic_result: dict | None = None,
) -> dict:
    """
    Assess classroom quality using Groq, then Gemini, then regex fallback.

    The regex assessment is run first to provide a transparent backup and
    supporting evidence. If both LLMs fail or return invalid JSON, that
    regex result is returned with assessment_source='regex'.

    This function is designed to be called by media_pipeline.py.
    """
    logger.info("Starting assessment with provider priority Groq -> Gemini -> regex.")

    # Always generate the deterministic fallback first.
    if heuristic_result is None:
        heuristic_result = assess_classroom_quality(
            transcription=transcription,
            visual_analysis=visual_analysis,
            audio_analysis=audio_analysis,
            silence_analysis=silence_analysis,
        )

    transcript = _extract_transcript(transcription)
    if len(transcript) < MIN_TRANSCRIPT_CHARS_FOR_LLM:
        logger.warning(
            "Transcript too short for LLM assessment (%d characters); "
            "using regex fallback.",
            len(transcript),
        )
        result = dict(heuristic_result)
        result["assessment_source"] = "regex"
        result.setdefault("limitations", [])
        result["limitations"] = list(result["limitations"]) + [
            "LLM assessment was skipped because the transcript was too short."
        ]
        return result

    was_truncated = len(transcript) > MAX_TRANSCRIPT_CHARS_FOR_LLM
    transcript_for_llm = _prepare_transcript_for_llm(transcript)

    visual_observations = _get_visual_observations(visual_analysis)
    audio_context = _select_audio_context(audio_analysis)
    silence_context = _select_silence_context(silence_analysis)

    prompt = _build_user_prompt(
        transcript=transcript_for_llm,
        heuristic_result=heuristic_result,
        visual_observations=visual_observations,
        audio_context=audio_context,
        silence_context=silence_context,
    )

    # Provider order is intentional: Groq first, Gemini second.
    llm_result = _call_groq(prompt)
    source = "groq"

    if llm_result is None:
        llm_result = _call_gemini(prompt)
        source = "gemini"

    if llm_result is None:
        logger.warning("Both LLM providers failed; returning regex assessment.")
        result = dict(heuristic_result)
        result["assessment_source"] = "regex"
        result.setdefault("limitations", [])
        result["limitations"] = list(result["limitations"]) + [
            "Groq and Gemini assessment were unavailable or returned invalid output; "
            "the regex/heuristic result was used."
        ]
        return result

    limitations = list(llm_result.get("limitations", []))
    limitations.append(
        "This is an AI-assisted provisional assessment, not a validated pedagogical instrument."
    )
    if not visual_observations:
        limitations.append(
            "Visual observations were unavailable or unsuccessful; no visual evidence was supplied."
        )
    if was_truncated:
        limitations.append(
            "The transcript was truncated to fit the configured LLM input length; "
            "important evidence from the middle may have been omitted."
        )

    final_result = {
        "status": "provisional",
        "provisional_rating": llm_result["provisional_rating"],
        "overall_score_0_to_4": llm_result["overall_score_0_to_4"],
        "criteria": llm_result["criteria"],
        "specific_observations": llm_result.get("specific_observations", []),
        "evidence_summary": (
            "LLM-based provisional assessment grounded in the supplied transcript "
            "and any available visual observations. Review the criterion evidence."
        ),
        "audio_context_not_scored": audio_context,
        "silence_context_not_scored": silence_context,
        "limitations": limitations,
        "assessment_source": source,
        "heuristic_backup": {
            "status": heuristic_result.get("status"),
            "provisional_rating": heuristic_result.get("provisional_rating"),
            "overall_score_0_to_4": heuristic_result.get("overall_score_0_to_4"),
            "criteria": heuristic_result.get("criteria", {}),
        },
    }

    logger.info("Assessment completed: source=%s rating=%s", source, final_result["provisional_rating"])
    return final_result
