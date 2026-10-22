
import logging
from pprint import pprint

from src.media_pipeline import process_classroom_video


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


video_path = (
    r"D:\JMJ_code\JMJ_humaniti\B435 intermediate Part  - 2026-07-25.mp4"
)

result = process_classroom_video(video_path)

print("\nPipeline status:")
print(result["status"])

print("\nVideo details:")
pprint(result["video"])

print("\nAudio analysis:")
pprint(result["audio"]["analysis"])

print("\nSilence detection:")
pprint(result["audio"]["silence"])

print("\nFrame extraction:")
pprint(result["frames"])

print("\nTranscription summary:")
if result["transcription"]:
    pprint({
        "provider": result["transcription"].get("provider"),
        "language": result["transcription"].get("language"),
        "chunk_count": result["transcription"].get("chunk_count"),
        "segment_count": len(
            result["transcription"].get("segments", [])
        ),
        "fallback_used": result["transcription"].get(
            "fallback_used"
        ),
    })
else:
    print("Transcription unavailable.")

print("\nErrors:")
pprint(result["errors"])

print("\nStage timings:")
pprint(result["stage_timings_seconds"])

print("\nTotal processing time:")
print(result["total_processing_seconds"], "seconds")
