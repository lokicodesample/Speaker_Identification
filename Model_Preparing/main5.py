from audio_preprocessing import load_wav, normalize_audio, resample_audio, save_wav
from vad import frame_audio, vad_by_energy
from features import compute_mfcc
from clustering import average_mfccs, cluster_embeddings
from diarization import generate_timestamps, save_diarization_output
from asr import transcribe_segment

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend


# --------- PHASE 1 ----------
input_path = "data/test_audio.wav"
output_path = "data/preprocessed.wav"

audio, original_rate = load_wav(input_path)
audio = normalize_audio(audio)
audio = resample_audio(audio, original_rate, target_rate=16000)
save_wav(audio, 16000, output_path)

# --------- PHASE 2 ----------
frame_size = int(0.025 * 16000)
hop_size = int(0.010 * 16000)

frames = frame_audio(audio, frame_size, hop_size)
flags, energy = vad_by_energy(frames, energy_threshold=0.02)

# --------- PHASE 3 ----------
speech_frames = frames[flags == True]
mfccs = compute_mfcc(speech_frames, sr=16000)

# --------- PHASE 4 ----------
group_size = 15
grouped_mfccs = average_mfccs(mfccs, group_size=group_size)
num_speakers = 2
labels = cluster_embeddings(grouped_mfccs, num_speakers=num_speakers)

# --------- PHASE 5 ----------
segments = generate_timestamps(labels, group_size, hop_size, sample_rate=16000)
save_diarization_output(segments)

# --------- PHASE 6 ----------
final_output_path = "results/diarization_transcript.txt"
with open(final_output_path, "w", encoding="utf-8") as f:
    for speaker, start, end in segments:
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        segment_audio = audio[start_sample:end_sample]
        text = transcribe_segment(segment_audio, 16000)
        line = f"Speaker {speaker} [{start:.2f}s - {end:.2f}s]: {text}"
        print(line)
        f.write(line + "\n")

print(f"\n✅ Transcript saved to: {final_output_path}")

# Optional: Clustering plot
plt.figure(figsize=(10, 2))
plt.plot(labels, marker='o', linestyle='-', color='purple')
plt.title(f"Speaker Clustering (K={num_speakers})")
plt.xlabel("Segment Index")
plt.ylabel("Speaker ID")
plt.grid()
plt.tight_layout()
# Save instead of show
plt.savefig("results/speaker_clustering.png")
print("✅ Clustering plot saved as results/speaker_clustering.png")
#plt.show()
