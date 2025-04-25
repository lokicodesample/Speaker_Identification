import os
import matplotlib
matplotlib.use('Agg')  # Avoid GUI errors
import matplotlib.pyplot as plt
import numpy as np

from audio_preprocessing import load_wav, normalize_audio, resample_audio, save_wav
from vad import frame_audio, vad_by_energy
from features import compute_mfcc
from clustering import average_mfccs, cluster_embeddings
from diarization import generate_timestamps, merge_short_segments, save_diarization_output
from asr import transcribe_segment, save_segment_audio
from export_to_csv import export_segments_to_csv

# --------- PHASE 1: Load & Preprocess ----------
input_path = "Lokesh_Intro_3Voices.wav"
output_path = "data/preprocessed.wav"
audio, original_rate = load_wav(input_path)
audio = normalize_audio(audio)
audio = resample_audio(audio, original_rate, target_rate=16000)
save_wav(audio, 16000, output_path)
print(f"✅ Preprocessed audio saved at {output_path}")

# --------- PHASE 2: VAD ----------
frame_size = int(0.025 * 16000)
hop_size = int(0.010 * 16000)
frames = frame_audio(audio, frame_size, hop_size)
flags, energy = vad_by_energy(frames, energy_threshold=0.02)

# --------- PHASE 3: MFCC ----------
speech_frames = frames[flags == True]
mfccs = compute_mfcc(speech_frames, sr=16000)

# --------- PHASE 4: Clustering ----------
group_size = 15
grouped_mfccs = average_mfccs(mfccs, group_size=group_size)
num_speakers = 2
labels = cluster_embeddings(grouped_mfccs, num_speakers=num_speakers)

# --------- PHASE 5: Timestamps & Merging ----------
segments = generate_timestamps(labels, group_size, hop_size, sample_rate=16000)
segments = merge_short_segments(segments, min_duration=0.8)
save_diarization_output(segments)

# --------- PHASE 6: Transcription & Save Audio ----------
final_output_path = "results/diarization_transcript.txt"
all_texts = []

with open(final_output_path, "w", encoding="utf-8") as f:
    for idx, (speaker, start, end) in enumerate(segments):
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)
        segment_audio = audio[start_sample:end_sample]

        text = transcribe_segment(segment_audio, 16000)
        line = f"Speaker {speaker} [{start:.2f}s - {end:.2f}s]: {text}"
        print(line)
        f.write(line + "\n")
        all_texts.append(text)

        save_segment_audio(audio, 16000, speaker, idx, start, end)

print(f"\n✅ Transcript saved to: {final_output_path}")

# --------- PHASE 7A: Export to CSV ----------
export_segments_to_csv(segments, all_texts)

# --------- PHASE 7B: Plot Clustering ----------
plt.figure(figsize=(10, 2))
plt.plot(labels, marker='o', linestyle='-', color='purple')
plt.title(f"Speaker Clustering (K={num_speakers})")
plt.xlabel("Segment Index")
plt.ylabel("Speaker ID")
plt.grid()
plt.tight_layout()
plt.savefig("results/speaker_clustering.png")
print("✅ Clustering plot saved as results/speaker_clustering.png")
