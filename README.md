Project Title

Time Series Forecasting Using LSTM and Attention Mechanisms

Overview

This project implements multivariate time series forecasting using deep learning. Two primary models are developed and evaluated:

Baseline LSTM model

Attention-based LSTM model

Both models are trained on historical S&P 500 data and evaluated using RMSE and MAE. Hyperparameter tuning is also performed to identify optimal model settings.

The goal is to determine whether incorporating an attention mechanism improves forecast accuracy and interpretability.

Project Structure
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
├── multivariate_timeseries.csv
├── X.npy
├── y.npy
├── scaler.pkl
│
├── results/
│      baseline_forecast.png
│      attention_forecast.png
│      attention_heatmap.png
│      training_curve.png
│
├── metrics_comparison.csv
├── tuning_results.csv
│
└── README.md

Requirements

Install dependencies using:

pip install numpy pandas matplotlib scikit-learn tensorflow yfinance

How to Run the Project
Step 1: Acquire Data
python data_acquisition.py


This downloads S&P 500 data and saves it as multivariate_timeseries.csv.
If internet is not available, synthetic data is generated.

Step 2: Preprocess Data
python preprocess.py


This generates X.npy, y.npy, and scaler.pkl.

Step 3: Train and Evaluate Models
python train_and_evaluate_tf.py


This script:

Trains the baseline LSTM

Trains the attention-based LSTM

Evaluates models using RMSE and MAE

Saves metrics_comparison.csv

Generates forecast plots

Generates attention heatmaps

Step 4: Hyperparameter Tuning (Optional)
python tune_hyperparams.py


This performs random search over selected hyperparameters and saves tuning_results.csv.

Outputs
Generated Files

multivariate_timeseries.csv

X.npy

y.npy

scaler.pkl

Evaluation Results

metrics_comparison.csv

tuning_results.csv

Visualizations

Saved in the results directory:

baseline_forecast.png

attention_forecast.png

attention_heatmap.png

training_curve.png

Model Summary
Baseline LSTM

A single-layer LSTM model that establishes base performance.

Attention-Based LSTM

An extended version of the LSTM model with an attention mechanism to highlight important timesteps.
Improves interpretability and often performance.
