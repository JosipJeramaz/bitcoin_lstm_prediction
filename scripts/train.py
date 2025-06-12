import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from scripts.model import LSTMPredictor
from torch.utils.data import TensorDataset, DataLoader

def train_model():
    df = pd.read_csv("data/final_dataset.csv", index_col="timestamp", parse_dates=True)
    # Drop non-numeric columns
    if "timeframe" in df.columns:
        df = df.drop(columns=["timeframe"])
    data = df.values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y_signal, y_price = [], [], []
    for i in range(len(data_scaled) - 10):
        X.append(data_scaled[i:i+10])
        y_signal.append(int(data_scaled[i+10][3] > data_scaled[i+9][3]))  # 1 if next close > prev close
        y_price.append(data_scaled[i+10][:2])  # open, close

    X = torch.tensor(X, dtype=torch.float32)
    y_signal = torch.tensor(y_signal, dtype=torch.float32).view(-1, 1)
    y_price = torch.tensor(y_price, dtype=torch.float32)

    # Split into train and validation sets
    X_train, X_val, y_signal_train, y_signal_val, y_price_train, y_price_val = train_test_split(
        X, y_signal, y_price, test_size=0.2, random_state=42, shuffle=True
    )

    # Create TensorDatasets and DataLoaders for batching
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_signal_train, y_price_train)
    val_dataset = TensorDataset(X_val, y_signal_val, y_price_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMPredictor(input_size=X.shape[2], hidden_size=128, num_layers=3, dropout=0.3, bidirectional=True)
    criterion_signal = nn.BCELoss()
    criterion_price = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 50  # Increased epochs for more realistic training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb_signal, yb_price in train_loader:
            price_out, signal_prob = model(xb)
            loss = criterion_signal(signal_prob, yb_signal) + criterion_price(price_out, yb_price)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb_signal, yb_price in val_loader:
                val_price_out, val_signal_prob = model(xb)
                loss = criterion_signal(val_signal_prob, yb_signal) + criterion_price(val_price_out, yb_price)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save model and scaler for reproducibility
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm.pth")
    joblib.dump(scaler, "models/scaler.save")

if __name__ == "__main__":
    train_model()