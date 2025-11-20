"""
train_and_evaluate_tf.py
Orchestrates training of baseline and attention models, evaluation, and saving outputs.
Produces metrics_comparison.csv and plots in results/.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from baseline_lstm_tf import build_baseline, train_model
from attention_lstm_tf import build_attention_lstm

def load_sequences():
    X = np.load('X.npy')
    y = np.load('y.npy')
    return X, y

def split_data(X, y, test_size=0.2, val_size=0.1, seed=42):
    # Do not shuffle for time series
    n = len(X)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    X_train = X[:-(n_test + n_val)] if (n_test + n_val) > 0 else X
    X_val = X[-(n_test + n_val):-n_test] if n_test > 0 else X[-n_val:]
    X_test = X[-n_test:] if n_test > 0 else X[-n_test:]
    y_train = y[:-(n_test + n_val)] if (n_test + n_val) > 0 else y
    y_val = y[-(n_test + n_val):-n_test] if n_test > 0 else y[-n_val:]
    y_test = y[-n_test:] if n_test > 0 else y[-n_test:]
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_and_save(model, X, y, name, is_attention=False):
    results = {}
    os.makedirs('results', exist_ok=True)
    if is_attention:
        pred, attn = model.predict(X, verbose=0)
    else:
        pred = model.predict(X, verbose=0)
        attn = None

    # Ensure shapes
    pred = pred.reshape(-1)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, pred)
    results['rmse'] = float(rmse)
    results['mae'] = float(mae)

    # save prediction plot (first 200 points or fewer)
    n = min(200, len(y))
    plt.figure(figsize=(10,4))
    plt.plot(y[:n], label='True')
    plt.plot(pred[:n], label='Pred')
    plt.title(f'{name} - Actual vs Pred (first {n})')
    plt.legend()
    plt.savefig(f'results/{name}_forecast.png')
    plt.close()

    # save attention heatmap if available
    if attn is not None:
        try:
            attn_map = np.squeeze(attn)  # (samples, time)
            if attn_map.ndim == 1:
                attn_map = attn_map[np.newaxis, :]
            # plot first 100 samples x timesteps
            to_plot = attn_map[:100, :].T
            plt.figure(figsize=(8,6))
            plt.imshow(to_plot, aspect='auto', origin='lower')
            plt.colorbar()
            plt.title(f'{name} - Attention weights (time x sample)')
            plt.xlabel('sample index')
            plt.ylabel('timestep')
            plt.savefig(f'results/{name}_attention_heatmap.png')
            plt.close()
        except Exception as e:
            print('Could not save attention plot:', e)

    return results

def main():
    X, y = load_sequences()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print('Shapes:', X_train.shape, X_val.shape, X_test.shape)

    # Baseline
    baseline = build_baseline(input_shape=X_train.shape[1:], units=64)
    history_b = train_model(baseline, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    b_results = evaluate_and_save(baseline, X_test, y_test, 'baseline', is_attention=False)

    # Attention model
    attn = build_attention_lstm(input_shape=X_train.shape[1:], units=64)
    
    # --- FIX APPLIED HERE ---
    # Create dummy labels for the attention weights output
    # The validation data must also be structured as a list of labels
    y_dummy_train = np.zeros_like(y_train)
    y_dummy_val = np.zeros_like(y_val)
    
    attn.fit(
        X_train, 
        # Pass a list of labels: [true labels, dummy labels]
        [y_train, y_dummy_train], 
        validation_data=(X_val, [y_val, y_dummy_val]), 
        epochs=10, 
        batch_size=32, 
        verbose=2
    )
    # ------------------------
    
    a_results = evaluate_and_save(attn, X_test, y_test, 'attention', is_attention=True)

    # Save metrics comparison
    metrics = pd.DataFrame([
        {'model': 'baseline', 'rmse': b_results['rmse'], 'mae': b_results['mae']},
        {'model': 'attention', 'rmse': a_results['rmse'], 'mae': a_results['mae']}
    ])
    metrics.to_csv('metrics_comparison.csv', index=False)
    print('Saved metrics_comparison.csv')
    print(metrics)

if __name__ == '__main__':
    main()