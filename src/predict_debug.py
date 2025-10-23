import sys
import numpy as np
import librosa
from tensorflow import keras

def extract_features(audio_path, sr=22050, n_mfcc=13):
    print(f"\nLoading audio file: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches, axis=1)
    
    print(f"MFCC shape: {mfcc.shape}")
    print(f"Pitch shape: {pitch.shape}")
    print(f"MFCC first row example: {mfcc[0] if len(mfcc) > 0 else 'Empty'}")
    print(f"Pitch example: {pitch[0] if len(pitch) > 0 else 'Empty'}")
    
    return mfcc, pitch

if len(sys.argv) < 3:
    print("Usage: python predict_debug.py <audio_file> <model_path>")
    sys.exit(1)

audio_path = sys.argv[1]
model_path = sys.argv[2]

print(f"\nLoading model from: {model_path}")
model = keras.models.load_model(model_path)
model.summary() 
mfcc, pitch = extract_features(audio_path)

mfcc_input = mfcc[np.newaxis, ..., np.newaxis]  # example: (1, frames, n_mfcc, 1)

print("\nRunning prediction...")
preds = model.predict(mfcc_input)
print("Raw model predictions (softmax probabilities):", preds)

pred_class_index = np.argmax(preds)
print(f"Predicted class index: {pred_class_index}")

labels = ['hungry', 'discomfort', 'hot_cold', 'belly_pain', 'burping', 'lonely', 'scared', 'tired']
pred_label = labels[pred_class_index]
print(f"Predicted Emotion: {pred_label} ({preds[0][pred_class_index]*100:.2f}% confidence)")
