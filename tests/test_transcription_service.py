from pathlib import Path
from src.transcription_service import transcribe_audio

audio_path = Path("D:/JMJ_code/JMJ_humaniti/test_sample.wav")

result = transcribe_audio(str(audio_path))

print("Provider:", result["provider"])
print("Model:", result["model"])
print("Language:", result["language"])
print("Fallback used:", result["fallback_used"])
print("Transcript:", result["transcript"])
print("Segment count:", len(result["segments"]))