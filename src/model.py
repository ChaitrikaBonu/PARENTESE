# src/model.py
from tensorflow.keras import layers, models

def build_model(mfcc_shape, pitch_shape):

    mfcc_input = layers.Input(shape=(*mfcc_shape, 1), name='mfcc_input')
    
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(mfcc_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    shape_before_lstm = x.shape
    x = layers.Reshape((shape_before_lstm[1], shape_before_lstm[2]*shape_before_lstm[3]))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    pitch_input = layers.Input(shape=pitch_shape, name='pitch_input')
    p = layers.Dense(32, activation='relu')(pitch_input)

    concat = layers.concatenate([x, p])
    concat = layers.Dense(128, activation='relu')(concat)
    concat = layers.Dropout(0.3)(concat)
    output = layers.Dense(8, activation='softmax')(concat)

    model = models.Model([mfcc_input, pitch_input], output)
    return model
