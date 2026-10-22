"""Integration test: run the full classroom pipeline and generate a report.

Run from the project root:
    python tests/test_report_generation.py

This test invokes the real media pipeline, including transcription and any
configured LLM calls. It can take several minutes and may incur API usage.
"""

import logging
from pathlib import Path
from pprint import pprint

from src.media_pipeline import process_classroom_video
from src.report_generator import generate_markdown_report


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Specify the classroom video path here.
video_path = (
    r"D:\JMJ_code\JMJ_humaniti\B435 intermediate Part  - 2026-07-25.mp4"
)


def main():
    """Run the media pipeline, then build and save its assessment report."""
    print("\nStarting classroom media pipeline...")
    logger.info("Starting report-generation integration test.")

    # Step 1: Process the video and obtain the combined pipeline result.
    # This calls the existing stages: frame extraction, visual analysis,
    # audio analysis, silence detection, transcription, and assessment.
    pipeline_result = process_classroom_video(video_path)

    print("\nMedia pipeline summary:")
    pprint(
        {
            "status": pipeline_result.get("status"),
            "video": pipeline_result.get("video"),
            "stage_timings_seconds": pipeline_result.get(
                "stage_timings_seconds"
            ),
            "errors": pipeline_result.get("errors", []),
        },
        sort_dicts=False,
    )

    # Step 2: Generate the Markdown report.
    # A relative output path is resolved against the project root by
    # generate_markdown_report().
    report_path = Path("data/reports") / (
        f"{Path(video_path).stem}_assessment_report.md"
    )

    print("\nGenerating classroom assessment report...")
    report_result = generate_markdown_report(
        pipeline_result=pipeline_result,
        output_path=report_path,
    )

    report_data = report_result["report_data"]
    assessment = report_data.get("assessment", {})

    print("\nAssessment summary:")
    pprint(
        {
            "rating": assessment.get("provisional_rating"),
            "overall_score_0_to_4": assessment.get("overall_score_0_to_4"),
            "assessment_source": assessment.get("assessment_source"),
            "assessment_status": assessment.get("status"),
            "criterion_count": len(assessment.get("criteria", [])),
            "transcript_evidence_count": len(
                report_data.get("transcript", {}).get(
                    "evidence_excerpts", []
                )
            ),
            "visual_analysis_status": report_data.get(
                "visual_evidence", {}
            ).get("status"),
            "pipeline_status": report_data.get("pipeline_status"),
            "processing_error_count": len(
                report_data.get("processing_errors", [])
            ),
        },
        sort_dicts=False,
    )

    print("\nReport saved to:")
    print(report_result.get("path"))

    print("\nGenerated Markdown report preview:")
    print(report_result["markdown"])

    logger.info(
        "Integration test completed: pipeline_status=%s rating=%s "
        "assessment_source=%s report_path=%s",
        report_data.get("pipeline_status"),
        assessment.get("provisional_rating"),
        assessment.get("assessment_source"),
        report_result.get("path"),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Log the full traceback for debugging, without printing transcript
        # contents or other potentially sensitive classroom data.
        logger.exception("Report-generation integration test failed.")
        raise
