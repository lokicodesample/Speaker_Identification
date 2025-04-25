import wave
import numpy as np
import os
from scipy.signal import resample

def load_wav(file_path):
    """Loads a .wav file and returns audio data and sample rate"""
    with wave.open(file_path, 'rb') as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        num_frames = wav_file.getnframes()

        raw_data = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        if num_channels == 2:
            audio_data = audio_data[::2]  # Convert to mono

        return audio_data.astype(np.float32), framerate

def normalize_audio(audio):
    """Normalize audio amplitude to range [-1, 1]"""
    return audio / np.max(np.abs(audio))

def resample_audio(audio, original_rate, target_rate=16000):
    """Resample audio to target rate"""
    duration = len(audio) / original_rate
    target_length = int(duration * target_rate)
    resampled_audio = resample(audio, target_length)
    return resampled_audio.astype(np.float32)

def save_wav(audio, sample_rate, output_path):
    """Save numpy audio array to .wav file"""
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes = 16 bits
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
