"""
visualize.py
Utility functions to plot training curves, forecasts, and attention heatmaps.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_training(history, out='results/training_curve.png'):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.figure(figsize=(8,4))
    if hasattr(history, 'history'):
        plt.plot(history.history.get('loss', []), label='loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='val_loss')
    else:
        # history could be a dict
        plt.plot(history.get('loss', []), label='loss')
    plt.legend()
    plt.savefig(out)
    plt.close()

def plot_forecast(true, pred, out='results/forecast.png', n=200):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = min(n, len(true), len(pred))
    plt.figure(figsize=(10,4))
    plt.plot(true[:n], label='True')
    plt.plot(pred[:n], label='Pred')
    plt.legend()
    plt.savefig(out)
    plt.close()

def plot_attention_heatmap(attn_array, out='results/attention_heatmap.png'):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # attn_array: (samples, time)
    plt.figure(figsize=(8,6))
    plt.imshow(attn_array.T, aspect='auto', origin='lower')
    plt.colorbar()
    plt.savefig(out)
    plt.close()
