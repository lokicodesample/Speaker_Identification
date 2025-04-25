import numpy as np

def frame_audio(audio, frame_size, hop_size):
    """Split audio into overlapping frames"""
    num_samples = len(audio)
    frames = []
    for start in range(0, num_samples - frame_size + 1, hop_size):
        end = start + frame_size
        frames.append(audio[start:end])
    return np.array(frames)

def compute_short_time_energy(frames):
    """Compute energy of each frame"""
    return np.sum(frames ** 2, axis=1)

def vad_by_energy(frames, energy_threshold=0.02):
    """Simple VAD based on short-time energy"""
    energy = compute_short_time_energy(frames)
    speech_flags = energy > energy_threshold
    return speech_flags, energy
