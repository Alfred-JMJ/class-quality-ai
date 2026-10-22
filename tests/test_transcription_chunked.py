
import logging
from pprint import pprint

from src.transcription_chunked import transcribe_audio_in_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

audio_path = "data/recordings/test_sample.wav"

result = transcribe_audio_in_chunks(audio_path)

print("\nTranscription summary:")
pprint({
    "provider": result["provider"],
    "language": result["language"],
    "chunk_count": result["chunk_count"],
    "segment_count": len(result["segments"]),
    "fallback_used": result["fallback_used"],
    "groq_error": result["groq_error"],
})

print("\nFirst five transcript segments:")
pprint(result["segments"][:5])

print("\nTranscript preview:")
print(result["transcript"][:1000])

