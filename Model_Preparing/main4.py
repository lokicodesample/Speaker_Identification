from audio_preprocessing import load_wav, normalize_audio, resample_audio, save_wav
from vad import frame_audio, vad_by_energy
from features import compute_mfcc
from clustering import average_mfccs, cluster_embeddings
from diarization import generate_timestamps, save_diarization_output
import matplotlib.pyplot as plt
import os

# --------- Phase 1: Preprocessing ----------
input_path = "data/test_audio.wav"
output_path = "data/preprocessed.wav"

audio, original_rate = load_wav(input_path)
audio = normalize_audio(audio)
audio = resample_audio(audio, original_rate, target_rate=16000)
save_wav(audio, 16000, output_path)
print(f"Preprocessed audio saved at {output_path}")

# --------- Phase 2: VAD ----------
frame_size = int(0.025 * 16000)  # 25ms
hop_size = int(0.010 * 16000)    # 10ms

frames = frame_audio(audio, frame_size, hop_size)
flags, energy = vad_by_energy(frames, energy_threshold=0.02)

# --------- Phase 3: MFCC ----------
speech_frames = frames[flags == True]
mfccs = compute_mfcc(speech_frames, sr=16000)

# --------- Phase 4: Clustering ----------
group_size = 5
grouped_mfccs = average_mfccs(mfccs, group_size=group_size)
num_speakers = 2
labels = cluster_embeddings(grouped_mfccs, num_speakers=num_speakers)

# --------- Phase 5: Diarization Output ----------
segments = generate_timestamps(labels, group_size, hop_size, sample_rate=16000)
save_diarization_output(segments)

# Optional: Plot clustering
plt.figure(figsize=(10, 2))
plt.plot(labels, marker='o', linestyle='-', color='purple')
plt.title(f"Speaker Clustering (K={num_speakers})")
plt.xlabel("Grouped Segment Index")
plt.ylabel("Speaker ID")
plt.grid()
plt.tight_layout()
plt.show()
