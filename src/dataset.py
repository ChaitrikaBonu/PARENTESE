# src/dataset.py — dataset scanning and feature preparation
import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import CATEGORIES, DURATION
from utils import load_audio, extract_mfcc, extract_pitch, add_noise, pitch_shift

def scan_dataset(raw_dir='data/raw'):
    rows = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(raw_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for f in os.listdir(cat_dir):
            if f.lower().endswith('.wav'):
                rows.append((os.path.join(cat_dir, f), cat))
    return rows

def prepare_features(raw_dir='data/raw', out_npz='data/processed/features.npz', augment=True):
    rows = scan_dataset(raw_dir)
    X_mfcc, X_pitch, y = [], [], []

    for path, cat in tqdm(rows, desc="Extracting features"):
        y_raw = load_audio(path, duration=DURATION)
        mfcc = extract_mfcc(y_raw)
        pitch = extract_pitch(y_raw)
        X_mfcc.append(mfcc)
        X_pitch.append([pitch])
        y.append(CATEGORIES.index(cat))

        if augment:
            y_aug = add_noise(y_raw)
            X_mfcc.append(extract_mfcc(y_aug))
            X_pitch.append([extract_pitch(y_aug)])
            y.append(CATEGORIES.index(cat))

            y_ps = pitch_shift(y_raw, 22050, 2)
            X_mfcc.append(extract_mfcc(y_ps))
            X_pitch.append([extract_pitch(y_ps)])
            y.append(CATEGORIES.index(cat))

    np.savez_compressed(out_npz, X_mfcc=X_mfcc, X_pitch=X_pitch, y=y)
    with open('label_map.json','w') as f:
        json.dump({i:cat for i,cat in enumerate(CATEGORIES)}, f)
    print(f"Saved features to {out_npz}")
    return out_npz

def train_val_test_split(feature_npz='data/processed/features.npz'):
    data = np.load(feature_npz, allow_pickle=True)
    X_mfcc, X_pitch, y = data['X_mfcc'], data['X_pitch'], data['y']
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X_mfcc, X_pitch, y = X_mfcc[idx], X_pitch[idx], y[idx]

    X_train_m, X_test_m, X_train_p, X_test_p, y_train, y_test = train_test_split(
        X_mfcc, X_pitch, y, test_size=0.15, stratify=y, random_state=42)
    X_train_m, X_val_m, X_train_p, X_val_p, y_train, y_val = train_test_split(
        X_train_m, X_train_p, y_train, test_size=0.1, stratify=y_train, random_state=42)
    return (X_train_m, X_train_p, y_train), (X_val_m, X_val_p, y_val), (X_test_m, X_test_p, y_test)
