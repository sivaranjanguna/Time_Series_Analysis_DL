"""
preprocess.py
Loads multivariate_timeseries.csv, handles missing values, scales features, and prepares sequences.
Saves sequences as numpy files for training scripts: X.npy, y.npy, and saves scaler as scaler.pkl
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

def create_sequences(csv_path='multivariate_timeseries.csv', seq_len=30, target_col='Close', dropna=True):
    """
    Read CSV, select numeric features (prefer Open, High, Low, Close, Volume),
    scale them with MinMaxScaler, and produce sliding windows of length seq_len.
    """
    df = pd.read_csv(csv_path)
    if dropna:
        df = df.dropna().reset_index(drop=True)

    # Choose features
    preferred = ['Open', 'High', 'Low', 'Close', 'Volume']
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if set(preferred).issubset(set(df.columns)):
        features = preferred
    else:
        # fallback to numeric columns
        features = numeric_cols

    if len(features) < 1:
        raise ValueError('No numeric columns available in CSV.')

    data = df[features].values.astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    target_idx = features.index(target_col) if target_col in features else 0

    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len, target_idx])

    X = np.array(X)
    y = np.array(y)

    # Save artifacts
    np.save('X.npy', X)
    np.save('y.npy', y)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print('Created sequences X.npy and y.npy. Shapes:', X.shape, y.shape)
    print('Features used:', features)
    return X, y, features

if __name__ == '__main__':
    create_sequences()
