# 1D Bitcoin Price Forecasting

This implementation uses your existing Dual LSTM model to make multi-step daily predictions using an autoregressive approach.

## How It Works

1. **Single-Step Model**: Uses your existing trained model that predicts the next daily candle
2. **Autoregressive Extension**: Feeds predictions back as inputs to predict further into the future
3. **1D-Only System**: Uses daily timeframe data for recursive prediction with technical indicators

## Key Features

- **Flexible Horizon**: Predict from 1 day to multiple days ahead
- **Confidence Scoring**: Each prediction includes a confidence score
- **Directional Analysis**: Predicts both price and direction (up/down)
- **Historical Backtesting**: Test accuracy on historical periods
- **Comprehensive Visualization**: Charts and plots for all results

## Quick Start

### 1. Test the Implementation
```bash
python test_1d_forecast.py
```
This runs a quick test with 5 days of predictions to verify everything works.

### 2. Generate 5-Day Forecast
```bash
python run_1d_forecast.py forecast --days 5
```
Creates a comprehensive 5-day forecast with visualizations.

### 3. Run Historical Backtest
```bash
python test_historical.py
```
Tests accuracy on multiple historical periods using real data.

### 4. Run Complete Forecast
```bash
python run_1d_forecast.py all --days 5
```
Runs comprehensive forecasting with all features enabled.

## Individual Scripts

### Test 1D Forecast Functionality
```bash
python test_1d_forecast.py
```

### 1D Daily Forecasting
```bash
python -m scripts.multi_step_forecast_1d
```

### Historical Backtesting
```bash
python -m scripts.historical_backtest
```

## Output Files

All results are saved in the `predictions/` directory:

- `multi_step_forecast_1d.csv` - Full daily forecast data
- `multi_step_forecast_1d.png` - Forecast visualization
- `historical_backtest_results.csv` - Historical backtest performance
- `historical_backtest.png` - Backtest visualization
- `historical_backtest_raw.save` - Raw backtest data for analysis

## Understanding the Results

### Prediction DataFrame Columns
- `predicted_open/close` - Predicted prices in USD
- `confidence` - Confidence score (0-1, decreases with distance)
- `direction_label` - UP/DOWN prediction
- `predicted_change_pct` - Expected percentage change
- `days_ahead` - Prediction horizon in days

### Historical Backtest Metrics
- **RMSE** - Root Mean Square Error in USD
- **MAE** - Mean Absolute Error in USD  
- **MAPE** - Mean Absolute Percentage Error
- **Directional Accuracy** - Percentage of correct direction predictions
- **Open/Close Error** - Separate errors for open vs close prices

## Important Notes

1. **Error Accumulation**: Accuracy decreases the further you predict due to compounding errors
2. **Confidence Degradation**: Confidence scores decrease linearly with prediction steps
3. **Data Requirements**: Needs your trained model files and 1D dataset
4. **Market Reality**: Long-term predictions should be used with caution
5. **1D Focus**: This system now focuses exclusively on daily predictions

## Prerequisites

Ensure you have:
- Trained model: `models/dual_lstm_final.pth`
- Fitted scaler: `models/scaler.save`
- Dataset: `data/final_dataset.csv`

If missing, first run:
```bash
python -m scripts.train
```

## Customization

You can modify prediction horizons, confidence calculations, and visualization settings in the respective script files:

- `scripts/multi_step_forecast_1d.py` - Core autoregressive logic and forecasting
- `scripts/historical_backtest.py` - Historical backtesting functionality  
- `scripts/visualize.py` - Visualization and plotting utilities

## Example Output

```
Creating 5-step forecast (5 days ahead)...

Forecast Summary:
Prediction period: 2025-09-20 00:00:00 to 2025-09-25 00:00:00
Number of predictions: 5
Average confidence: 0.85
Bullish predictions: 3/5
Bearish predictions: 2/5

✅ Forecast completed successfully!
📊 Check 'predictions/' directory for detailed results
```
