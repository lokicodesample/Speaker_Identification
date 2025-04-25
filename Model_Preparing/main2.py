from audio_preprocessing import load_wav, normalize_audio, resample_audio, save_wav
from vad import frame_audio, vad_by_energy
from features import compute_mfcc
import matplotlib.pyplot as plt
import os

# --------- Phase 1: Preprocessing ----------
input_path = "data/test_audio.wav"
output_path = "data/preprocessed.wav"

audio, original_rate = load_wav(input_path)
print(f"Original Sample Rate: {original_rate}, Length: {len(audio)}")

audio = normalize_audio(audio)
audio = resample_audio(audio, original_rate, target_rate=16000)
save_wav(audio, 16000, output_path)
print(f"Preprocessed audio saved at {output_path}")

# --------- Phase 2: Voice Activity Detection ----------
frame_size = int(0.025 * 16000)  # 25ms
hop_size = int(0.010 * 16000)    # 10ms

frames = frame_audio(audio, frame_size, hop_size)
flags, energy = vad_by_energy(frames, energy_threshold=0.02)

# Optional: Plot VAD energy and speech detection
plt.figure(figsize=(10, 4))
plt.plot(energy, label="Short-Time Energy")
plt.plot(flags.astype(int) * max(energy), label="Speech Detected")
plt.title("VAD - Energy Based")
plt.xlabel("Frame Index")
plt.ylabel("Energy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# --------- Phase 3: MFCC Feature Extraction ----------
speech_frames = frames[flags == True]
mfccs = compute_mfcc(speech_frames, sr=16000)
print("MFCC shape:", mfccs.shape)

# Optional: Visualize MFCC
plt.imshow(mfccs.T[:3], aspect='auto', origin='lower', cmap='viridis')
plt.title("MFCCs (first 3 coefficients)")
plt.xlabel("Frame")
plt.ylabel("Coefficient")
plt.colorbar()
plt.tight_layout()
plt.show()
