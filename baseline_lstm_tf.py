"""
baseline_lstm_tf.py
Baseline LSTM model (TensorFlow Keras). Contains build and train functions.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_baseline(input_shape, units=64, dropout=0.1):
    """
    Build a simple LSTM model.
    input_shape: (timesteps, features)
    """
    model = Sequential([
        LSTM(units, input_shape=input_shape),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, verbose=2)
    return history
