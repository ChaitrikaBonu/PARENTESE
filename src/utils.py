# src/utils.py — audio feature extraction and augmentation utilities
import os
import numpy as np
import librosa
from config import SAMPLE_RATE, SAMPLES, N_MFCC, N_FFT, HOP_LENGTH
import soundfile as sf

def load_audio(path, sr=SAMPLE_RATE, duration=None):
    y, _ = librosa.load(path, sr=sr)
    target = int(sr * (duration or 3.0))
    if len(y) > target:
        y = y[:target]
    else:
        y = np.pad(y, (0, max(0, target - len(y))), mode='constant')
    return y

def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    return mfcc

def extract_pitch(y, sr=SAMPLE_RATE):
    try:
        f0 = librosa.yin(y, fmin=50, fmax=700, sr=sr, frame_length=2048, hop_length=HOP_LENGTH)
        f0 = f0[np.isfinite(f0)]
        return float(np.median(f0)) if len(f0) > 0 else 0.0
    except Exception:
        return 0.0

def add_noise(y, noise_factor=0.005):
    return y + noise_factor * np.random.randn(len(y))

def pitch_shift(y, sr, n_steps):
    try:
        return librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
    except Exception:
        return y

def save_wav(y, path, sr=SAMPLE_RATE):
    sf.write(path, y, sr)
