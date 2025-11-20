📘 Time Series Forecasting Using LSTM and Attention (TensorFlow)

This project focuses on multivariate time series forecasting using the S&P 500 Index (GSPC) dataset.
Two models are built and compared:

Baseline LSTM Model

Attention-based LSTM Model

Both models are trained, evaluated, and compared using RMSE and MAE metrics.
The project also includes hyperparameter tuning, attention visualization, and a fully reproducible pipeline.


📁 Project Structure
project/
│
├── data_acquisition.py
├── preprocess.py
├── baseline_lstm_tf.py
├── attention_lstm_tf.py
├── train_and_evaluate_tf.py
├── tune_hyperparams.py
├── visualize.py
│
├── multivariate_timeseries.csv        # Generated after running data_acquisition.py
├── X.npy, y.npy                       # Generated after running preprocess.py
├── scaler.pkl                         # Feature scaler
│
├── results/
│      ├── baseline_forecast.png
│      ├── attention_forecast.png
│      ├── attention_attention_heatmap.png
│      ├── baseline_forecast.png
│      ├── training_curve.png
│      └── (others)
│
├── metrics_comparison.csv
├── tuning_results.csv
│
└── README.md

🧠 Project Objective

To forecast future closing prices of S&P 500 using deep learning models and evaluate if adding an attention mechanism improves predictive performance.

📥 1. Data Acquisition

Run:
python data_acquisition.py


This script:

Downloads S&P 500 index data using yfinance

If internet is unavailable, generates a synthetic placeholder dataset

Saves data as multivariate_timeseries.csv

🛠️ 2. Preprocessing

python preprocess.py

This will:

Load the data

Remove missing values

Scale numeric features using MinMaxScaler

Create sequence windows (default 30 timesteps)

Save processed arrays X.npy and y.npy

🤖 3. Models Included
A. Baseline LSTM Model

Single LSTM layer

Dropout

Dense output

B. Attention-Based LSTM Model

LSTM with return_sequences

Custom Attention layer

Output + attention weights

Both models use Mean Squared Error (MSE) as the loss function.

🏋️ 4. Training & Evaluation

Run:
