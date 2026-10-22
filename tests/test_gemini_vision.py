
import logging
from pprint import pprint

from src.media_processor import extract_key_frames
from src.video_analyzer_gemini import analyze_video_frames


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


# Specify the classroom video path here.
video_path = (
    r"D:\JMJ_code\JMJ_humaniti\B435 intermediate Part  - 2026-07-25.mp4"
)


def main():
    print("\nExtracting frames from video...")

    frame_result = extract_key_frames(video_path)
    frames = frame_result.get("frames", [])

    print("\nFrame extraction summary:")
    pprint({
        "frame_count": len(frames),
        "frames": frames,
    })

    if not frames:
        print("\nNo frames were extracted. Cannot run visual analysis.")
        return

    print("\nSending frames to Gemini for visual analysis...")

    result = analyze_video_frames(frames)

    print("\nVisual analysis status:")
    print(result["status"])

    print("\nProvider and model:")
    pprint({
        "provider": result.get("provider"),
        "model": result.get("model"),
        "frame_count": result.get("frame_count"),
    })

    print("\nVisual observations:")
    pprint(result.get("observations"), sort_dicts=False)


if __name__ == "__main__":
    main()
