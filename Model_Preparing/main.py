from audio_preprocessing import load_wav, normalize_audio, resample_audio, save_wav
import os

input_path = "data/test_audio.wav"
output_path = "data/preprocessed.wav"

# Load
audio, original_rate = load_wav(input_path)
print(f"Original Sample Rate: {original_rate}, Length: {len(audio)}")

# Normalize
audio = normalize_audio(audio)

# Resample
audio = resample_audio(audio, original_rate, target_rate=16000)

# Save
save_wav(audio, 16000, output_path)
print(f"Preprocessed audio saved at {output_path}")
