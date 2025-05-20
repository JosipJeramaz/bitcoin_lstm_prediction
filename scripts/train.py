import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_signal = nn.Linear(hidden_size, 1)
        self.fc_price = nn.Linear(hidden_size, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        signal = torch.sigmoid(self.fc_signal(last))
        price = self.fc_price(last)
        return signal, price

def train_model():
    df = pd.read_csv("data/final_dataset.csv", index_col="timestamp", parse_dates=True)
    data = df.values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y_signal, y_price = [], [], []
    for i in range(len(data_scaled) - 10):
        X.append(data_scaled[i:i+10])
        y_signal.append(int(data_scaled[i+10][3] > data_scaled[i+9][3]))  # simple rule
        y_price.append(data_scaled[i+10][:2])  # open, close

    X = torch.tensor(X, dtype=torch.float32)
    y_signal = torch.tensor(y_signal, dtype=torch.float32).view(-1, 1)
    y_price = torch.tensor(y_price, dtype=torch.float32)

    model = LSTMModel(input_size=X.shape[2])
    criterion_signal = nn.BCELoss()
    criterion_price = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        signal_out, price_out = model(X)
        loss = criterion_signal(signal_out, y_signal) + criterion_price(price_out, y_price)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/lstm.pth")
