# Bitcoin Dual LSTM Prediction System

An advanced cryptocurrency price forecasting system using a sophisticated Dual LSTM architecture that processes multiple timeframes to predict Bitcoin prices with high accuracy.

## 🎯 Key Features

- **Single-Step Price Forecasting**: Predicts next candle (4 hours) with high accuracy
- **Dual Timeframe Processing**: Combines 1-day and 4-hour data for comprehensive market analysis
- **Advanced LSTM Architecture**: Cross-attention mechanisms between timeframes
- **Market Continuity**: Uses realistic price modeling where next_open = current_close
- **Comprehensive Evaluation**: Multiple validation metrics and visualization tools
- **Production Ready**: Complete pipeline from data fetching to model deployment

## 🚀 Quick Start

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run Complete Pipeline:**
```bash
python run_all.py
```

This will:
- Fetch latest Bitcoin data from Binance
- Process and prepare datasets with technical indicators
- Train the dual LSTM model for next-candle prediction
- Generate comprehensive evaluations and visualizations
- Save predictions to `outputs/predictions.csv`

## 📁 Project Structure

```
bitcoin_lstm_prediction/
├── data/                          # Dataset files
│   ├── final_dataset.csv          # Combined 1d+4h dataset with indicators
│   ├── processed_1d.csv           # Daily timeframe data
│   ├── processed_4h.csv           # 4-hour timeframe data
│   ├── raw_1d.csv                 # Raw daily OHLCV data
│   └── raw_4h.csv                 # Raw 4-hour OHLCV data
├── models/                        # Trained models and artifacts
│   ├── dual_lstm_final.pth        # Main trained model
│   ├── scaler.save                # Data normalization scaler (StandardScaler)
│   ├── training_history.save      # Training metrics history
│   └── model_info.save            # Model configuration and metadata
├── outputs/                       # Results and visualizations
│   ├── predictions.csv            # Price predictions
│   ├── actual_vs_predicted.png    # Performance comparison plots
│   ├── prediction_plot.png        # Price forecast visualization
│   └── training_val_loss_lr.png   # Training loss curves
├── scripts/                       # Core implementation
│   ├── config.py                  # Centralized configuration
│   ├── model.py                   # DualLSTMPredictor architecture
│   ├── train.py                   # Single-step training pipeline
│   ├── evaluate.py                # Model evaluation and prediction
│   ├── visualize.py               # Comprehensive visualization suite
│   ├── fetch_data.py              # Binance data collection
│   ├── indicators.py              # Technical indicator calculations
│   ├── prepare_dataset.py         # Data preprocessing pipeline
│   ├── multi_step_forecast_1d.py  # Optional multi-day forecasting
│   └── historical_backtest.py     # Backtesting utilities
├── utils/                         # Helper utilities
│   └── helpers.py                 # Common utility functions
├── requirements.txt               # Python dependencies
└── run_all.py                    # Complete pipeline executor
```

## 🧠 Model Architecture

### DualLSTMPredictor Components:

1. **Input Processing**
   - Optional input projection layers with layer normalization
   - Handles multi-dimensional feature vectors from both timeframes

2. **Dual Timeframe LSTMs**
   - Separate LSTM networks for 1-day and 4-hour data
   - Multi-layer architecture with dropout regularization
   - Bidirectional processing for enhanced pattern recognition

3. **Attention Mechanisms**
   - **Self-Attention**: Weighted pooling of LSTM sequence outputs
   - **Cross-Attention**: Information exchange between timeframes
   - **Multi-Head Attention**: 6 parallel attention heads for diverse pattern capture

4. **Feature Fusion & Prediction**
   - Concatenation of timeframe representations
   - Multi-layer dense network for price regression
   - **Market Continuity**: Models next_open = current_close principle
   - **Model Output**: Single step (open, close) for next 4-hour candle

### Key Technical Specifications:
- **Input Size**: 20+ features (OHLCV + 15+ technical indicators)
- **Hidden Size**: 180 neurons per LSTM layer (configurable in config.py)
- **Sequence Length**: 15 timesteps for pattern recognition
- **Prediction Horizon**: Next 4-hour candle
- **Model Output Format**: [batch, 2] for (open, close) of next candle
- **Training Approach**: Single-step prediction with market continuity

## ⚙️ Configuration

All parameters are centralized in `scripts/config.py` for easy experimentation:

### Training Configuration
```python
TRAINING_CONFIG = {
    'num_epochs': 100,              # Max training epochs (early stopping applied)
    'batch_size': 32,               # Training batch size
    'learning_rate': 0.0015,        # Initial learning rate
    'hidden_size': 180,             # LSTM hidden dimensions
    'sequence_length': 15,          # Input sequence length
    'num_layers': 3,                # LSTM layers per timeframe
    'dropout': 0.2,                 # Dropout for regularization
    'early_stopping_patience': 25,  # Early stopping patience
    'lr_scheduler_patience': 6,     # Learning rate scheduler patience
    'lr_scheduler_factor': 0.7      # LR reduction factor
}
```

### Model Architecture Options
```python
MODEL_CONFIG = {
    'use_attention': True,                # Enable self-attention pooling
    'use_cross_attention': True,          # Cross-timeframe attention
    'use_positional_encoding': True,      # Positional encoding for sequences
    'use_layer_norm': True,               # Layer normalization
    'use_input_projection': True,         # Input projection layers
    'num_attention_heads': 6,             # Multi-head attention heads
    'regression_hidden_size': 120         # Final regression layer size
}
```

### Data Configuration
```python
DATA_CONFIG = {
    'train_split': 0.7,            # Training data percentage
    'val_split': 0.15,             # Validation data percentage  
    'test_split': 0.15,            # Test data percentage
    'random_state': 42             # Reproducibility seed
}
```

## 📊 Performance Metrics

The model achieves excellent accuracy on single-step price prediction:

### Overall Performance
- **RMSE**: Varies with market volatility
- **MAE**: Low average absolute error on price predictions
- **Close Price MAPE**: Typically 2-4% on test data
- **Open Price MAPE**: Typically 1-3% (benefits from market continuity)

### Evaluation Metrics Included:
- **Per-feature metrics**: Separate tracking for open/close prices
- **Percentage errors**: MAPE for relative performance assessment
- **Statistical validation**: Comprehensive error analysis
- **Market continuity accuracy**: How well next_open = current_close is maintained

## 🛠️ Individual Component Usage

### Train New Model
```bash
python scripts/train.py
```
- Trains single-step prediction model from scratch
- Saves best model with early stopping
- Outputs training history and metrics

### Evaluate Existing Model  
```bash
python scripts/evaluate.py
```
- Loads trained model and generates predictions
- Creates detailed evaluation metrics
- Saves predictions to CSV format

### Generate Visualizations
```bash
python scripts/visualize.py
```
- Creates comprehensive visualization suite
- Training history plots, prediction comparisons
- Single-step forecast analysis

### Fetch Fresh Data
```bash
python scripts/fetch_data.py
```
- Downloads latest Bitcoin data from Binance
- Supports both 1d and 4h timeframes  
- Automatically calculates technical indicators

## 📈 Data Processing Pipeline

1. **Data Collection**
   - Binance API integration for real-time OHLCV data
   - Automatic handling of rate limits and data validation

2. **Technical Indicators** (15+ indicators calculated)
   - **Trend**: SMA, EMA, MACD, ADX
   - **Momentum**: RSI, Stochastic, Williams %R
   - **Volatility**: Bollinger Bands, ATR
   - **Volume**: Volume SMA, Price-Volume Trend

3. **Data Preprocessing**
   - Timestamp-based alignment between timeframes
   - StandardScaler normalization with log transformation
   - Sequence generation with sliding windows
   - Market continuity target creation (next_open = current_close)

4. **Quality Assurance**
   - NaN value handling and data cleaning
   - Chronological ordering validation
   - Feature correlation analysis

## 🔮 Understanding "Future" Prediction Accuracy

### The Validation Challenge
**Question**: How can we know prediction accuracy if we don't have real future data?

**Answer**: We use several validation approaches, each with different strengths:

### 1. **Historical Backtesting** (Current Approach)
- **How it works**: Train on 2022-2023 data, test on 2024-2025 data
- **Pros**: Large dataset, statistical significance, reproducible results
- **Cons**: Past performance doesn't guarantee future results
- **Our Results**: 2-3% MAPE on historical "future" data

### 2. **Walk-Forward Validation** (Production Recommendation)
```python
# Example implementation concept:
for test_date in recent_dates:
    # Train on data up to test_date
    # Predict next 10 steps
    # Wait for actual results
    # Calculate real accuracy
```

### 3. **Real-Time Monitoring** (True Validation)
- Deploy model in production
- Make predictions and track actual outcomes
- Calculate rolling accuracy metrics
- Adapt model based on performance drift

### Current Model Confidence Level
- **High confidence (next 4h)**: 2-4% error typical
- **Market continuity**: Open price predictions very accurate due to continuity assumption
- **Close price**: More challenging but still reliable predictions
- **Market conditions matter**: Bull/bear markets affect accuracy

## 🚀 Advanced Usage Examples

### Custom Model Training
```python
from scripts.model import DualLSTMPredictor
from scripts.config import get_model_config, get_training_config

# Initialize model with actual parameters
model = DualLSTMPredictor(
    input_size=20  # Number of features (OHLCV + technical indicators)
)

# Single-step prediction for next 4-hour candle
# Training is handled automatically by train.py
```

### Loading Pre-trained Model for Predictions
```python
import torch
import joblib
import pandas as pd
from scripts.model import DualLSTMPredictor

# Load trained model
model = DualLSTMPredictor(input_size=20)  # Initialize with correct input size
model.load_state_dict(torch.load('models/dual_lstm_final.pth'))

# Load scaler
scaler = joblib.load('models/scaler.save')

# Make predictions on new data
model.eval()
with torch.no_grad():
    predictions = model(new_1d_data, new_4h_data)
    # predictions shape: [batch, 2] for (open, close) of next candle
```

### Visualization Outputs

The system generates comprehensive visualizations in `outputs/`:

1. **actual_vs_predicted.png**
   - Side-by-side comparison of model predictions vs real prices
   - Color-coded accuracy indicators
   - Single-step prediction analysis

2. **prediction_plot.png** 
   - Time series plot showing historical and predicted prices
   - Confidence intervals for prediction uncertainty
   - Clear visual separation of training/validation/test periods

3. **training_val_loss_lr.png**
   - Training and validation loss curves
   - Learning rate schedule visualization
   - Early stopping point identification

## 🔧 Technical Requirements

### System Requirements
- **Python**: 3.8+ (tested on 3.11, 3.13)
- **RAM**: 8GB+ recommended for training
- **Storage**: 2GB+ for data and models
- **GPU**: Optional (CUDA-compatible), CPU training supported

### Dependencies
```txt
torch>=2.0.0              # PyTorch for deep learning
pandas>=1.5.0             # Data manipulation
numpy>=1.24.0             # Numerical computing  
scikit-learn>=1.3.0       # Machine learning utilities
matplotlib>=3.6.0         # Plotting and visualization
seaborn>=0.12.0           # Statistical visualizations
python-binance>=1.0.0     # Binance API client
joblib>=1.3.0             # Model serialization
tqdm>=4.65.0              # Progress bars
```

## 🎯 Production Deployment Considerations

### Model Retraining Schedule
- **Recommended**: Weekly retraining with latest data
- **Minimum**: Monthly retraining to maintain accuracy
- **Trigger**: If accuracy drops below acceptable threshold (>5% MAPE)

### Real-Time Prediction Pipeline
1. **Data Ingestion**: Automated Binance data fetching
2. **Preprocessing**: Apply same normalization as training (StandardScaler + log transform)
3. **Prediction**: Generate next 4-hour candle forecasts
4. **Monitoring**: Track prediction accuracy over time
5. **Alerting**: Notify when model performance degrades

### Risk Management
- **Never use for automated trading without human oversight**
- **Combine with fundamental analysis and market sentiment**
- **Implement position sizing and stop-loss mechanisms**
- **Monitor for black swan events and market regime changes**

## 📚 Research Background

This implementation is based on advanced time series forecasting research:

- **Dual Timeframe Processing**: Captures both short-term and long-term patterns
- **Attention Mechanisms**: Focuses on relevant historical patterns
- **Market Continuity Modeling**: Realistic price prediction approach
- **Cross-Attention**: Enables information flow between timeframes

### Citation
If using this work for research, please cite:
```
Bitcoin Dual LSTM Prediction System
Advanced cryptocurrency forecasting with dual timeframe processing
https://github.com/JosipJeramaz/bitcoin_lstm_prediction
```

## ⚠️ Disclaimer

**Important**: This system is for educational and research purposes only.

- **Not financial advice**: Do not use for real trading without extensive testing
- **No guarantees**: Past performance does not predict future results  
- **Market risks**: Cryptocurrency markets are highly volatile and unpredictable
- **Model limitations**: AI predictions can be wrong, especially during market disruptions

**Always conduct your own research and risk assessment before making investment decisions.**

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **Alternative architectures**: Transformer models, hybrid approaches
- **Additional features**: Sentiment analysis, on-chain metrics
- **Real-time deployment**: Production pipeline implementation  
- **Validation methods**: Walk-forward testing, cross-validation
- **Visualization enhancements**: Interactive plots, real-time dashboards

## 📞 Support

For questions or issues:
1. Check the outputs in `outputs/` directory for error analysis
2. Review configuration in `scripts/config.py`
3. Examine training logs and model metrics
4. Open an issue with detailed error information

---
