# src/train.py — training pipeline for PARENTESE
import os
import numpy as np
import tensorflow as tf
from dataset import prepare_features, train_val_test_split
from model import build_model
from config import BATCH_SIZE, EPOCHS

def main():
   
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    feat = prepare_features(raw_dir='data/raw', out_npz='data/processed/features.npz')
    (X_train_m, X_train_p, y_train), (X_val_m, X_val_p, y_val), (X_test_m, X_test_p, y_test) = train_val_test_split(feat)

    X_train_m = X_train_m[..., np.newaxis]
    X_val_m = X_val_m[..., np.newaxis]
    X_test_m = X_test_m[..., np.newaxis]

    model = build_model(X_train_m.shape[1:3], X_train_p.shape[1:])
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if y_train is one-hot
        metrics=['accuracy']
    )

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        'saved_models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True
    )

    model.fit(
        {'mfcc_input': X_train_m, 'pitch_input': X_train_p},
        y_train,
        validation_data=({'mfcc_input': X_val_m, 'pitch_input': X_val_p}, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[ckpt, early]
    )

    print("Evaluating on test data...")
    loss, acc = model.evaluate({'mfcc_input': X_test_m, 'pitch_input': X_test_p}, y_test)
    print(f"Test accuracy: {acc:.4f}")

    model.save('saved_models/final_model.h5')
    print("Model saved successfully.")

if __name__ == '__main__':
    main()
