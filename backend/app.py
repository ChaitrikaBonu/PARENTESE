# backend/app.py
import os
import io
import json
import tempfile
import traceback
from typing import Dict

import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---- Config ----
MODEL_PATH = os.path.join('..', 'saved_models', 'final_model.h5')
STATS_PATH = os.path.join('..', 'data', 'processed', 'feature_stats.npz')
SR = 22050
N_MFCC = 40
MAX_LEN = 130
WINDOW_LEN_SEC = 2.0
HOP_LEN_SEC = 1.0
CATEGORIES = ['hungry', 'discomfort', 'hot_cold', 'belly_pain',
              'burping', 'lonely', 'scared', 'tired']

# ---- Load model & stats ----
app = FastAPI(title="PARENTESE Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
FRONTEND_DIR = os.path.abspath(FRONTEND_DIR)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
else:
    print("WARNING: frontend directory not found:", FRONTEND_DIR)

# ---- Root endpoint serves home.html ----
@app.get("/")
async def root():
    home_path = os.path.join(FRONTEND_DIR, "home.html")
    if os.path.exists(home_path):
        return FileResponse(home_path)
    else:
        return JSONResponse({"error": "home.html not found"}, status_code=404)

# ---- Safe model loading ----
def safe_load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
        return model
    except Exception as e:
        print("Could not load model:", e)
        return None

def safe_load_stats():
    try:
        stats = np.load(STATS_PATH, allow_pickle=True)
        print("Loaded stats:", STATS_PATH)
        return stats
    except Exception as e:
        print("Could not load stats:", e)
        return None

MODEL = safe_load_model()
STATS = safe_load_stats()
if STATS is not None:
    mfcc_mean = STATS['mfcc_mean']
    mfcc_std = STATS['mfcc_std']
    pitch_mean = STATS['pitch_mean']
    pitch_std = STATS['pitch_std']
else:
    mfcc_mean = mfcc_std = pitch_mean = pitch_std = None

# ---- Prediction helpers ----
def extract_features_segment(y_segment):
    mfcc = librosa.feature.mfcc(y=y_segment, sr=SR, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]

    try:
        pitch_arr = librosa.yin(y_segment, fmin=80, fmax=400)
        pitch = float(np.mean(pitch_arr))
    except Exception:
        pitch = 0.0

    if mfcc_mean is not None and mfcc_std is not None:
        mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-8)
    if pitch_mean is not None and pitch_std is not None:
        pitch = (pitch - pitch_mean) / (pitch_std + 1e-8)

    return mfcc, pitch

def predict_audio_file(file_path: str):
    if MODEL is None:
        raise RuntimeError("Model not loaded on server.")

    y, _ = librosa.load(file_path, sr=SR)
    total_len_sec = len(y) / SR
    start = 0.0
    preds = []

    while start < total_len_sec:
        start_sample = int(start * SR)
        end_sample = int(min((start + WINDOW_LEN_SEC) * SR, len(y)))
        segment = y[start_sample:end_sample]

        if len(segment) < 256:
            pad_len = int(WINDOW_LEN_SEC * SR) - len(segment)
            segment = np.pad(segment, (0, pad_len), mode='constant')

        mfcc, pitch = extract_features_segment(segment)
        mfcc_input = np.expand_dims(mfcc, axis=(0,-1))
        pitch_input = np.expand_dims([pitch], axis=0)

        pred = MODEL.predict({'mfcc_input': mfcc_input, 'pitch_input': pitch_input}, verbose=0)
        preds.append(pred[0])
        start += HOP_LEN_SEC

    if len(preds) == 0:
        raise RuntimeError("No audio frames to predict.")

    avg_pred = np.mean(preds, axis=0)
    percentages = (avg_pred * 100).tolist()
    results = {cat: round(float(p),2) for cat,p in zip(CATEGORIES, percentages)}
    top_label = max(results, key=results.get)
    return top_label, results

# ---- API Endpoints ----
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        top_label, results = predict_audio_file(tmp_path)
        os.remove(tmp_path)
        return {"top_reason": top_label, "percentages": results}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/index.html")
async def index_page():
    path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "index.html not found"}

@app.get("/home.html")
async def home_page():
    path = os.path.join(FRONTEND_DIR, "home.html")
    if os.path.exists(path):
        return FileResponse(path)
    else:
        return {"error": "home.html not found"}

@app.get("/contact.html")
async def contact_page():
    contact_path = os.path.join(FRONTEND_DIR, "contact.html")
    if os.path.exists(contact_path):
        return FileResponse(contact_path)
    else:
        return {"error": "contact.html not found"}
    
@app.post("/contact")
async def contact_endpoint(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}  # fallback if empty or invalid
    messages_file = os.path.join(os.path.dirname(__file__), 'messages.json')
    try:
        messages = []
        if os.path.exists(messages_file):
            with open(messages_file,'r',encoding='utf-8') as f:
                messages = json.load(f)
        messages.append(data)
        with open(messages_file,'w',encoding='utf-8') as f:
            json.dump(messages,f,indent=2)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/profile")
async def profile_endpoint(req: Request):
    data = await req.json()
    file_path = os.path.join(os.path.dirname(__file__), 'profile.json')
    try:
        with open(file_path,'w',encoding='utf-8') as f:
            json.dump(data,f,indent=2)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/settings")
async def settings_endpoint(req: Request):
    data = await req.json()
    file_path = os.path.join(os.path.dirname(__file__), 'settings.json')
    try:
        with open(file_path,'w',encoding='utf-8') as f:
            json.dump(data,f,indent=2)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}
