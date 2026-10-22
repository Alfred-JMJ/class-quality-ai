import logging
from pprint import pprint

from src.media_processor import (
    extract_audio,
    extract_key_frames,
    detect_silence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

video = r"D:\JMJ_code\JMJ_humaniti\B435 intermediate Part  - 2026-07-25.mp4"

audio = extract_audio(
    video_path=video,
    audio_path="data/recordings/test_sample.wav",
)

frames = extract_key_frames(video_path=video)
silence = detect_silence(audio_path=audio)

print("\nFrames:")
pprint(frames)

print("\nSilence:")
pprint(silence)