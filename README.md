# Bitcoin LSTM Prediction

A sophisticated cryptocurrency price prediction system using Dual LSTM architecture that processes multiple timeframes (1d and 4h) to predict Bitcoin prices with enhanced accuracy.

## Setup

1. **Clone the repository**
2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure the model in `scripts/config.py`:**

```python
# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 150,
    'batch_size': 32,
    'learning_rate': 0.002,
    'hidden_size': 156,
    'sequence_length': 35,
    # ... other parameters
}

# Model Architecture
MODEL_CONFIG = {
    'use_attention': True,
    'use_cross_attention': True,
    'use_positional_encoding': True,
    'num_attention_heads': 6,
    # ... other parameters
}
```

4. **Run the complete pipeline:**

```bash
python run_all.py
```

## Directory Structure

```
bitcoin_lstm_prediction/
├── data/                    # Dataset files
│   ├── final_dataset.csv    # Combined dataset
│   ├── processed_1d.csv     # 1-day timeframe data
│   └── processed_4h.csv     # 4-hour timeframe data
├── models/                  # Trained models and artifacts
│   ├── dual_lstm_final.pth  # Best trained model
│   ├── scaler.save          # Data scaler
│   └── model_info.save      # Model metadata
├── outputs/                 # Results and visualizations
│   ├── predictions.csv      # Price predictions
│   ├── prediction_plot.png  # Visual predictions
│   └── residual_analysis.png # Model validation plots
├── scripts/                 # Source code
│   ├── config.py           # Centralized configuration
│   ├── model.py            # Dual LSTM architecture
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Model evaluation
│   └── visualize.py        # Visualization tools
├── utils/                  # Helper functions
└── requirements.txt        # Python dependencies
```

## Development

- **Train model:** `python scripts/train.py`
- **Evaluate model:** `python scripts/evaluate.py` 
- **Generate visualizations:** `python scripts/visualize.py`
- **Fetch new data:** `python scripts/fetch_data.py`

## Features

- **Dual timeframe processing** - Combines 1d and 4h data for comprehensive analysis
- **Advanced LSTM architecture** - Multiple layers with attention mechanisms
- **Cross-timeframe attention** - Learning patterns across different time horizons
- **Positional encoding** - Enhanced sequence understanding
- **Real-time predictions** - Generate future price forecasts
- **Comprehensive evaluation** - Multiple metrics and residual analysis
- **Interactive visualizations** - Detailed charts and performance plots
- **Configurable parameters** - Easy experimentation via config.py

## Model Architecture

The system uses a sophisticated **Dual LSTM Predictor** with:

- **Input Processing:** Optional input projections with layer normalization
- **Timeframe LSTMs:** Separate processing for 1d and 4h data
- **Attention Mechanisms:** 
  - Self-attention pooling for sequence summarization
  - Cross-attention between timeframes for pattern correlation
- **Feature Fusion:** Concatenation of timeframe representations
- **Price Regression:** Multi-layer dense network for open/close prediction

## Configuration

All model parameters are centralized in `scripts/config.py`:

### Training Parameters
- **Epochs:** 150 (with early stopping)
- **Batch Size:** 32
- **Learning Rate:** 0.002 (with scheduler)
- **Hidden Size:** 156 neurons
- **Sequence Length:** 35 timesteps
- **Dropout:** 0.25 for regularization

### Architecture Options
- **Attention Heads:** 6 (multi-head attention)
- **Regression Hidden:** 96 neurons
- **Layer Normalization:** Enabled
- **Positional Encoding:** Enabled
- **Cross-attention:** Enabled between timeframes

## Performance Metrics

The model is evaluated using multiple regression metrics:

- **MSE/RMSE** - Overall prediction accuracy
- **MAE** - Mean absolute error
- **MAPE** - Mean absolute percentage error
- **Feature-specific metrics** - Separate tracking for open/close prices
- **Residual analysis** - Homoscedasticity and variance checks

## Data Processing

1. **Multi-timeframe alignment** - Timestamp-based synchronization
2. **Log transformation** - For price stability
3. **StandardScaler normalization** - Zero mean, unit variance
4. **Sequence generation** - Sliding window approach
5. **Train/validation/test splits** - 70/15/15 with stratification

## Visualization Outputs

- **Prediction plots** - Actual vs predicted prices with candlestick charts
- **Performance metrics** - Comprehensive evaluation dashboard
- **Residual analysis** - Statistical validation plots
- **Training history** - Loss curves and learning rate progression

## Notes

- **Data requirements:** CSV files with OHLCV data and technical indicators
- **Memory usage:** Optimized for efficient training on standard hardware
- **Reproducibility:** Fixed random seeds for consistent results
- **Extensibility:** Modular design for easy feature additions
- **Configuration-driven:** No hardcoded parameters in the model code

## Quick Start Example

```python
# Import the model
from scripts.model import DualLSTMPredictor
from scripts.config import get_training_config

# Create model (automatically uses config.py settings)
model = DualLSTMPredictor(input_size=20)

# Train the model
python scripts/train.py

# Generate predictions
python scripts/evaluate.py

# View results
# Check outputs/ directory for predictions and visualizations
```

## License

This project is designed for educational and research purposes in cryptocurrency price prediction and machine learning applications.
