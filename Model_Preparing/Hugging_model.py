import os
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from pydub import AudioSegment
import torchaudio
import subprocess



"""subprocess.call(["ffmpeg", "-i", "data/17 Apr, 12.02 am​(2).m4a", "-ar", "16000", "-ac", "1", "output_audio.wav"])"""
# Path to your audio file (WAV)
AUDIO_PATH = "Lokesh_Intro_3Voices.wav"  # Replace with your file
# Set your Hugging Face token
HF_TOKEN = "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"

# Load speaker diarization pipeline
speaker_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

# Load Whisper ASR (small model)
asr_pipeline = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small")


print("Processing....")
# Run diarization
diarization = speaker_pipeline(AUDIO_PATH)

# Load full audio for slicing
full_audio = AudioSegment.from_wav(AUDIO_PATH)

print("\n--- Speaker Diarization with Transcription ---\n")

results = []
speakers = set()

# Process each diarized segment
for i, (turn, _, speaker_label) in enumerate(diarization.itertracks(yield_label=True)):
    start_ms = int(turn.start * 1000)
    end_ms = int(turn.end * 1000)
    speakers.add(speaker_label)

    # Extract audio chunk
    chunk = full_audio[start_ms:end_ms]
    chunk_path = f"chunk_{i}.wav"
    chunk.export(chunk_path, format="wav")

    # Transcribe chunk
    text = asr_pipeline(chunk_path)["text"]

    # Output
    start_sec = round(turn.start, 2)
    end_sec = round(turn.end, 2)
    print(f"Speaker {speaker_label} [{start_sec}s - {end_sec}s]: {text}")

    results.append((speaker_label, start_sec, end_sec, text))

    # Clean up temp file
    os.remove(chunk_path)

# Final summary
print(f"\n🧠 Total unique speakers detected: {len(speakers)}")
