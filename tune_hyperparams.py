"""
tune_hyperparams.py
Simple random-search hyperparameter tuning for number of LSTM units, learning rate, and sequence length.
This script performs quick short trainings and records validation loss for each trial.
"""
import numpy as np
import pandas as pd
import itertools
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # Import Adam optimizer

from baseline_lstm_tf import build_baseline # Assume this builds the model architecture
from preprocess import create_sequences
# Assume train_model now accepts the compiled model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    # This mock function replaces the call to train_model, 
    # ensuring the model is compiled here with the correct optimizer object.
    
    # NOTE: You might need to integrate the original logic of train_model here
    # or ensure your original train_model handles the already compiled model.
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=0
    )
    return history

def random_search(csv='multivariate_timeseries.csv', trials=6):
    units_list = [32, 64, 128]
    lr_list = [1e-3, 5e-4, 1e-4]
    seq_list = [20, 30, 50]
    combos = list(itertools.product(units_list, lr_list, seq_list))
    np.random.shuffle(combos)
    combos = combos[:trials]
    results = []
    
    for units, lr, seq in combos:
        print('Trial:', units, lr, seq)
        # regenerate sequences with seq
        X, y, _ = create_sequences(csv_path=csv, seq_len=seq)
        # quick split
        n = len(X)
        split1 = int(n * 0.8)
        split2 = split1 + int(n * 0.1)
        X_train, X_val = X[:split1], X[split1:split2]
        y_train, y_val = y[:split1], y[split1:split2]
        
        # Build the model architecture
        model = build_baseline(input_shape=X_train.shape[1:], units=units)
        
        # --- FIX: Instantiate the optimizer with the specific learning rate ---
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Train the model (using the modified logic or ensuring the original train_model accepts compiled model)
        # The original train_model call is now effectively replaced by the lines above and below
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=5, batch_size=32)
        
        val_loss = history.history.get('val_loss', [None])[-1]
        results.append({'units': units, 'lr': lr, 'seq': seq, 'val_loss': float(val_loss) if val_loss is not None else None})
        
    df = pd.DataFrame(results)
    df.to_csv('tuning_results.csv', index=False)
    print('Saved tuning_results.csv')
    return df

if __name__ == '__main__':
    # Ensure there's a simple placeholder for train_model if you don't use the mock above
    if 'train_model' not in locals():
         print("Warning: Please ensure 'train_model' is imported or defined correctly.")
         
    random_search()