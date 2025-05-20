import torch
import pandas as pd
from scripts.train import LSTMModel
from sklearn.preprocessing import MinMaxScaler

def evaluate_model():
    df = pd.read_csv("data/final_dataset.csv", index_col="timestamp", parse_dates=True)
    data = df.values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X = []
    for i in range(len(data_scaled) - 10):
        X.append(data_scaled[i:i+10])
    X = torch.tensor(X, dtype=torch.float32)

    model = LSTMModel(input_size=X.shape[2])
    model.load_state_dict(torch.load("models/lstm.pth"))
    model.eval()

    with torch.no_grad():
        signal_out, price_out = model(X)
        signal_pred = (signal_out.numpy() > 0.5).astype(int)
        prices_pred = scaler.inverse_transform(np.hstack([price_out.numpy(), np.zeros((price_out.shape[0], data.shape[1]-2))]))[:, :2]

    pd.DataFrame({
        "pred_signal": signal_pred.flatten(),
        "pred_open": prices_pred[:, 0],
        "pred_close": prices_pred[:, 1]
    }).to_csv("outputs/predictions.csv", index=False)
