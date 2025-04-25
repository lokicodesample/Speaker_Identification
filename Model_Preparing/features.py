import numpy as np
from scipy.fftpack import dct

def hamming_window(frame):
    return frame * np.hamming(len(frame))

def power_spectrum(frame, NFFT):
    mag = np.fft.rfft(frame, n=NFFT)
    return (1.0 / NFFT) * np.abs(mag) ** 2

def mel_filterbank(sr, NFFT, n_filters=26):
    low_freq = 0
    high_freq = sr / 2
    mel_low = 2595 * np.log10(1 + low_freq / 700)
    mel_high = 2595 * np.log10(1 + high_freq / 700)

    mel_points = np.linspace(mel_low, mel_high, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_filters, int(NFFT / 2 + 1)))
    for m in range(1, n_filters + 1):
        f_m_minus = bin[m - 1]
        f_m = bin[m]
        f_m_plus = bin[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank

def compute_mfcc(frames, sr=16000, NFFT=512, n_filters=26, n_coeffs=13):
    """Compute MFCC for each frame"""
    fbank = mel_filterbank(sr, NFFT, n_filters)
    mfcc_features = []

    for frame in frames:
        frame = hamming_window(frame)
        pow_spec = power_spectrum(frame, NFFT)
        mel_energy = np.dot(fbank, pow_spec)
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
        log_mel_energy = np.log(mel_energy)
        mfcc = dct(log_mel_energy, type=2, norm='ortho')[:n_coeffs]
        mfcc_features.append(mfcc)

    return np.array(mfcc_features)
