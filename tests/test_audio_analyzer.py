
from pathlib import Path
from pprint import pprint

from src.audio_analyzer import analyze_audio

# Use the short WAV extracted for your transcription test.
audio_path = Path(
    r"D:\JMJ_code\JMJ_humaniti\test_sample.wav"
)

result = analyze_audio(str(audio_path))

pprint(result)
