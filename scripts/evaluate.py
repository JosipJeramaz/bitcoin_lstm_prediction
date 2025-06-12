"""
Evaluates the trained LSTM model on the prepared dataset.
- Loads the trained model and scaler.
- Generates predictions for signals and prices.
- Saves predictions to the outputs directory.
"""

import torch
import pandas as pd
import numpy as np
from scripts.model import LSTMPredictor
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def evaluate_model():
    df = pd.read_csv("data/final_dataset.csv", index_col="timestamp", parse_dates=True)
    # Drop non-numeric columns
    if "timeframe" in df.columns:
        df = df.drop(columns=["timeframe"])
    data = df.values
    # Load the scaler fitted during training
    scaler = joblib.load("models/scaler.save")
    data_scaled = scaler.transform(data)

    X = []
    pred_timestamps = []
    for i in range(len(data_scaled) - 10):
        X.append(data_scaled[i:i+10])
        pred_timestamps.append(df.index[i+10])  # Save the timestamp for the prediction
    X = torch.tensor(X, dtype=torch.float32)

    model = LSTMPredictor(input_size=X.shape[2], hidden_size=128, num_layers=3, dropout=0.3, bidirectional=True)
    model.load_state_dict(torch.load("models/lstm.pth"))
    model.eval()

    with torch.no_grad():
        price_out, signal_prob = model(X)
        signal_pred = (signal_prob.numpy() > 0.5).astype(int)
        prices_pred = scaler.inverse_transform(np.hstack([price_out.numpy(), np.zeros((price_out.shape[0], data.shape[1]-2))]))[:, :2]

    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    pred_open = prices_pred[:, 0]
    pred_close = prices_pred[:, 1]
    # Add synthetic high/low for visualization
    wick = np.abs(pred_open - pred_close) * 0.1
    pred_high = np.maximum(pred_open, pred_close) + wick
    pred_low = np.minimum(pred_open, pred_close) - wick

    pd.DataFrame({
        "timestamp": pred_timestamps,
        "pred_signal": signal_pred.flatten(),
        "pred_open": pred_open,
        "pred_close": pred_close,
        "pred_high": pred_high,
        "pred_low": pred_low
    }).to_csv("outputs/predictions.csv", index=False)

if __name__ == "__main__":
    evaluate_model()
