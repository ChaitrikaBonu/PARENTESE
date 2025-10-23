import sys
import numpy as np
import tensorflow as tf
import librosa

MODEL_PATH = 'saved_models/final_model.h5'
STATS_PATH = 'data/processed/feature_stats.npz'
CATEGORIES = ['hungry', 'discomfort', 'hot_cold', 'belly_pain', 
              'burping', 'lonely', 'scared', 'tired']
SR = 22050
N_MFCC = 40
MAX_LEN = 130
WINDOW_LEN_SEC = 2.0  
HOP_LEN_SEC = 1.0     

if len(sys.argv) < 2:
    print("Usage: python predict.py <audio_file>")
    sys.exit(1)

audio_path = sys.argv[1]

model = tf.keras.models.load_model(MODEL_PATH)
stats = np.load(STATS_PATH)
mfcc_mean, mfcc_std = stats['mfcc_mean'], stats['mfcc_std']
pitch_mean, pitch_std = stats['pitch_mean'], stats['pitch_std']

def extract_features_segment(y_segment):

    mfcc = librosa.feature.mfcc(y=y_segment, sr=SR, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]

    pitch_arr = librosa.yin(y_segment, fmin=80, fmax=400)
    pitch = np.mean(pitch_arr)

    mfcc = (mfcc - mfcc_mean) / mfcc_std
    pitch = (pitch - pitch_mean) / pitch_std

    return mfcc, pitch

def predict_audio(audio_path):
    y, _ = librosa.load(audio_path, sr=SR)
    total_len_sec = len(y) / SR
    start = 0.0
    preds = []

    while start < total_len_sec:
        start_sample = int(start * SR)
        end_sample = int(min((start + WINDOW_LEN_SEC) * SR, len(y)))
        segment = y[start_sample:end_sample]

        mfcc, pitch = extract_features_segment(segment)
        mfcc_input = np.expand_dims(mfcc, axis=(0,-1))
        pitch_input = np.expand_dims([pitch], axis=0)

        pred = model.predict({'mfcc_input': mfcc_input, 'pitch_input': pitch_input}, verbose=0)
        preds.append(pred[0])
        start += HOP_LEN_SEC

    avg_pred = np.mean(preds, axis=0) 
    percentages = avg_pred * 100      

    results = {cat: round(percent, 2) for cat, percent in zip(CATEGORIES, percentages)}
    top_label = max(results, key=results.get)

    return top_label, results

top_label, results = predict_audio(audio_path)
print(f"\nPredicted Top Emotion: {top_label}\n")
print("Predicted Percentages for All Categories:")
for cat, pct in results.items():
    print(f"  {cat:12}: {pct:.2f}%")
