# src/config.py — Configuration for PARENTESE

SAMPLE_RATE = 22050
DURATION = 3.0            # seconds: trim/pad each clip
SAMPLES = int(SAMPLE_RATE * DURATION)
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 60
RANDOM_SEED = 42

CATEGORIES = [
    "hungry",
    "discomfort",
    "hot_cold",
    "belly_pain",
    "burping",
    "lonely",
    "scared",
    "tired"
]
