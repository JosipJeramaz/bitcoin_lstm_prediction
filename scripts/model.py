import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import get_model_config, get_training_config

class AttentionPooling(nn.Module):
    """Attention-based pooling for better sequence representation"""
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)  
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)  
        return attended_output

class PositionalEncoding(nn.Module):
    """Add positional encoding to help the model understand sequence order"""
    def __init__(self, d_model, max_len=None):
        super(PositionalEncoding, self).__init__()
        if max_len is None:
            config = get_model_config()
            max_len = config['positional_encoding_max_len']
            
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        config = get_model_config()
        log_constant = config['log_constant']
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(log_constant) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class DualLSTMPredictor(nn.Module):
    """
    Enhanced Dual LSTM architecture that processes 1d and 4h timeframes separately
    and combines their outputs for final predictions
    """
    def __init__(self, input_size):
        super(DualLSTMPredictor, self).__init__()
        
        # Load all configuration from config.py
        config = get_model_config()
        training_config = get_training_config()
        
        # Use config values
        hidden_size = training_config['hidden_size']
        num_layers = training_config['num_layers']
        dropout = training_config['dropout']
        
        self.use_attention = config['use_attention']
        self.use_cross_attention = config['use_cross_attention']
        self.use_positional_encoding = config['use_positional_encoding']
        self.use_layer_norm = config['use_layer_norm']
        self.use_input_projection = config['use_input_projection']
        
        # Optional positional encoding
        if self.use_positional_encoding:
            self.pos_encoding_1d = PositionalEncoding(input_size)
            self.pos_encoding_4h = PositionalEncoding(input_size)
        
        # Optional input projection layers
        if self.use_input_projection:
            if self.use_layer_norm:
                self.input_proj_1d = nn.Sequential(
                    nn.Linear(input_size, input_size),
                    nn.LayerNorm(input_size),
                    nn.ReLU()
                )
                self.input_proj_4h = nn.Sequential(
                    nn.Linear(input_size, input_size),
                    nn.LayerNorm(input_size),
                    nn.ReLU()
                )
            else:
                self.input_proj_1d = nn.Sequential(
                    nn.Linear(input_size, input_size),
                    nn.ReLU()
                )
                self.input_proj_4h = nn.Sequential(
                    nn.Linear(input_size, input_size),
                    nn.ReLU()
                )
        
        # Two separate LSTMs for different timeframes
        self.lstm_1d = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False
        )
        
        self.lstm_4h = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False
        )
        
        # Optional attention pooling
        if self.use_attention:
            self.attention_1d = AttentionPooling(hidden_size)
            self.attention_4h = AttentionPooling(hidden_size)
        
        # Optional cross-attention between timeframes
        if self.use_cross_attention:
            config = get_model_config()
            num_heads = config['num_attention_heads']
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            if self.use_layer_norm:
                self.norm_1d = nn.LayerNorm(hidden_size)
                self.norm_4h = nn.LayerNorm(hidden_size)
        
        # Combine outputs from both LSTMs
        combined_size = hidden_size * 2  # concatenate outputs
        
        # Dense layers for price prediction only
        config = get_model_config()
        regression_hidden = config['regression_hidden_size']
        
        if self.use_layer_norm:
            self.regression_head = nn.Sequential(
                nn.Linear(combined_size, regression_hidden),
                nn.LayerNorm(regression_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(regression_hidden, 2)  # open, close
            )
        else:
            # Keep your original structure
            self.regression_head = nn.Sequential(
                nn.Linear(combined_size, regression_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(regression_hidden, 2)  # open, close
            )
        
        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x_1d, x_4h):
        # Optional input projections
        if self.use_input_projection:
            x_1d = self.input_proj_1d(x_1d)
            x_4h = self.input_proj_4h(x_4h)
        
        # Optional positional encoding
        if self.use_positional_encoding:
            x_1d = self.pos_encoding_1d(x_1d)
            x_4h = self.pos_encoding_4h(x_4h)
        
        # Process 1d data
        lstm_out_1d, _ = self.lstm_1d(x_1d)
        
        # Process 4h data
        lstm_out_4h, _ = self.lstm_4h(x_4h)
        
        # For price prediction - use attention if enabled
        if self.use_attention:
            last_hidden_1d_price = self.attention_1d(lstm_out_1d)
            last_hidden_4h_price = self.attention_4h(lstm_out_4h)
        else:
            last_hidden_1d_price = lstm_out_1d[:, -1, :]  # (batch, hidden)
            last_hidden_4h_price = lstm_out_4h[:, -1, :]  # (batch, hidden)
        
        # Optional cross-attention between timeframes
        if self.use_cross_attention:
            # Apply cross-attention to price features
            # Expand dimensions for multi-head attention
            feat_1d_price = last_hidden_1d_price.unsqueeze(1)  # (batch, 1, hidden)
            feat_4h_price = last_hidden_4h_price.unsqueeze(1)  # (batch, 1, hidden)
            
            # 4h queries 1d context (4h learns from 1d trends)
            attended_4h_price, _ = self.cross_attention(feat_4h_price, feat_1d_price, feat_1d_price)
            if self.use_layer_norm:
                attended_4h_price = self.norm_4h(attended_4h_price.squeeze(1) + last_hidden_4h_price)  # Residual connection
            else:
                attended_4h_price = attended_4h_price.squeeze(1) + last_hidden_4h_price  # Residual connection
            
            # 1d queries 4h context (1d learns from 4h patterns)
            attended_1d_price, _ = self.cross_attention(feat_1d_price, feat_4h_price, feat_4h_price)
            if self.use_layer_norm:
                attended_1d_price = self.norm_1d(attended_1d_price.squeeze(1) + last_hidden_1d_price)  # Residual connection
            else:
                attended_1d_price = attended_1d_price.squeeze(1) + last_hidden_1d_price  # Residual connection
            
            # Use attended features for price
            last_hidden_1d_price = attended_1d_price
            last_hidden_4h_price = attended_4h_price
        
        # Combine outputs for price prediction only
        combined_features = torch.cat([last_hidden_1d_price, last_hidden_4h_price], dim=1)  # (batch, hidden*2)
        
        # Final prediction - price only
        price_pred = self.regression_head(combined_features)  # (batch, 2)
        
        return price_pred

