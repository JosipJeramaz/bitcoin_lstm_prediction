import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, bidirectional=True):
        super(LSTMPredictor, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        direction_factor = 2 if bidirectional else 1
        self.regression_head = nn.Linear(hidden_size * direction_factor, 2)  # open_t+1, close_t+1
        self.classification_head = nn.Linear(hidden_size * direction_factor, 1)  # long/short signal

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        price_pred = self.regression_head(last_hidden)
        signal_logit = self.classification_head(last_hidden)
        signal_prob = torch.sigmoid(signal_logit)
        return price_pred, signal_prob
