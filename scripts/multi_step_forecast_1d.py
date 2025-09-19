"""
Multi-step Bitcoin price forecasting using 1D data for enhanced long-term predictions.

This module implements autoregressive forecasting capabilities that:
- Uses trained Dual LSTM model for recursive daily price predictions
- Calculates real-time technical indicators for each prediction step
- Provides confidence scoring that degrades naturally with prediction distance
- Generates comprehensive visualizations and detailed prediction analysis
- Maintains market continuity and realistic price movements

The autoregressive approach extends single-step model predictions to multi-day
forecasts while preserving the temporal dependencies and technical indicator
relationships learned during training.
"""

import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from scripts.model import DualLSTMPredictor
from scripts.config import get_model_config, get_training_config, get_data_config

# Suppress sklearn warnings about feature names (functionality works correctly)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def calculate_real_indicators(price_history, current_features, feature_columns):
    """
    Calculate real-time technical indicators based on predicted price history.
    
    This function computes moving averages and Bollinger Bands dynamically as new
    price predictions are generated, ensuring that technical indicators remain
    consistent with the evolving price series during autoregressive forecasting.
    
    Args:
        price_history (list): Historical OHLCV data including recent predictions
        current_features (torch.Tensor): Current scaled feature tensor
        feature_columns (list): Column names for feature identification
        
    Returns:
        torch.Tensor: Updated feature tensor with recalculated technical indicators
        
    Note:
        Expected feature order: [open, high, low, close, volume, ma5, ma10, ma30, 
        ma60, bb_bbm, bb_bbh, bb_bbl] as defined in indicators.py
    """
    if len(price_history) < 2:
        return current_features.clone()
        
    # Extract price data
    closes = [candle[3] for candle in price_history[-60:]]  # Last 60 for MA60
    highs = [candle[1] for candle in price_history[-60:]]
    lows = [candle[2] for candle in price_history[-60:]]
    volumes = [candle[4] for candle in price_history[-60:]]
    
    # Start with current values
    new_features = current_features.clone()
    
    try:
        # Expected feature order based on indicators.py:
        # 0: open, 1: high, 2: low, 3: close, 4: volume
        # 5: ma5, 6: ma10, 7: ma30, 8: ma60 
        # 9: bb_bbm, 10: bb_bbh, 11: bb_bbl
        
        current_price = closes[-1]
        
        # Update volume (simple trend based on price movement)
        if len(closes) >= 2:
            price_change = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0
            volume_multiplier = 1.0 + (abs(price_change) * 2)  # Higher volume on big moves
            if len(new_features) > 4:
                new_features[4] = max(1000.0, volumes[-1] * volume_multiplier)  # Ensure positive volume
        
        # Calculate Moving Averages
        if len(closes) >= 5:
            ma5 = sum(closes[-5:]) / 5
            if len(new_features) > 5:
                new_features[5] = max(1.0, ma5)  # Ensure positive
        
        if len(closes) >= 10:
            ma10 = sum(closes[-10:]) / 10
            if len(new_features) > 6:
                new_features[6] = max(1.0, ma10)  # Ensure positive
        
        if len(closes) >= 30:
            ma30 = sum(closes[-30:]) / 30
            if len(new_features) > 7:
                new_features[7] = max(1.0, ma30)  # Ensure positive
        
        if len(closes) >= 60:
            ma60 = sum(closes[-60:]) / 60
            if len(new_features) > 8:
                new_features[8] = max(1.0, ma60)  # Ensure positive
                
        # Calculate Bollinger Bands (20-period)
        if len(closes) >= 20:
            bb_period = min(20, len(closes))
            bb_closes = closes[-bb_period:]
            
            # Middle band (SMA20)
            bb_middle = sum(bb_closes) / len(bb_closes)
            
            # Standard deviation
            variance = sum([(price - bb_middle) ** 2 for price in bb_closes]) / len(bb_closes)
            bb_std = max(1.0, variance ** 0.5)  # Ensure positive std
            
            # Upper and lower bands
            bb_upper = bb_middle + (2 * bb_std)
            bb_lower = max(1.0, bb_middle - (2 * bb_std))  # Ensure positive
            
            # Update Bollinger Band features
            if len(new_features) > 9:
                new_features[9] = max(1.0, bb_middle)  # bb_bbm
            if len(new_features) > 10:
                new_features[10] = max(1.0, bb_upper)  # bb_bbh  
            if len(new_features) > 11:
                new_features[11] = max(1.0, bb_lower)  # bb_bbl
                
        print(f"Updated indicators - MA5: {new_features[5]:.2f}, BB_mid: {new_features[9]:.2f}")
        
    except Exception as e:
        print(f"Warning: Error calculating indicators: {e}")
        # Fallback - keep current values with slight decay
        for i in range(5, len(new_features)):
            new_features[i] = new_features[i] * 0.995
    
    return new_features


def predict_recursive_1d(model, initial_1d_data, scaler, steps=5, device='cpu'):
    """
    Make recursive 1-day predictions for multi-step forecasting with proper indicator calculation.
    
    Parameters:
        model: Trained DualLSTMPredictor model
        initial_1d_data: Initial 1d timeframe data tensor [sequence_length, features]
        scaler: Fitted StandardScaler
        steps: Number of days to predict
        device: Device for inference
        
    Returns:
        predictions: Array of shape [steps, 2] with daily predictions (open, close)
        raw_predictions: Array of predictions in standardized form
        confidence_scores: Array of confidence scores
    """
    print(f"Making {steps}-day recursive predictions using 1d data with real indicators...")
    
    model.eval()
    model = model.to(device)
    
    # Create copies of input data
    current_1d = initial_1d_data.clone().unsqueeze(0).to(device)  # [1, seq_len, features]
    # Create dummy 4H data for model compatibility (same shape as 1D)
    current_4h = current_1d.clone()  # [1, seq_len, features] (for compatibility)
    
    # Store predictions
    raw_predictions = []
    actual_predictions = []
    confidence_scores = []
    
    # Keep track of price history for indicator calculation (in actual prices, not scaled)
    price_history = []
    
    # Feature indices
    open_idx = 0
    close_idx = 1
    
    # Load data configuration
    data_config = get_data_config()
    log_epsilon = data_config['log_epsilon']
    
    # Initialize price history from recent actual data
    try:
        # Get last few actual price points for initial indicator calculation
        seq_len = current_1d.shape[1]
        for i in range(-min(10, seq_len), 0):  # Last 10 points or all available
            scaled_data = current_1d[0, i, [2, 8, 5, 7]].cpu().numpy()  # OHLC from scaler order
            
            # Create dummy array for inverse transform
            dummy_data = np.zeros((1, scaler.n_features_in_))
            dummy_data[0, 2] = scaled_data[0]  # open
            dummy_data[0, 8] = scaled_data[1]  # high  
            dummy_data[0, 5] = scaled_data[2]  # low
            dummy_data[0, 7] = scaled_data[3]  # close
            
            try:
                inv_scaled = scaler.inverse_transform(dummy_data)
                actual_ohlc = np.exp(inv_scaled[0, [2, 8, 5, 7]]) - log_epsilon  # OHLC
                actual_ohlc = np.maximum(actual_ohlc, 1.0)  # Ensure positive prices
                
                # Add volume estimate
                volume_estimate = 50000  # Placeholder
                price_history.append([
                    actual_ohlc[0],  # open
                    actual_ohlc[1],  # high
                    actual_ohlc[2],  # low
                    actual_ohlc[3],  # close
                    volume_estimate  # volume
                ])
            except:
                # Fallback for problematic transforms
                price_history.append([50000, 50500, 49500, 50000, 50000])
                
    except Exception as e:
        print(f"Warning: Could not initialize price history: {e}")
        # Initialize with dummy data
        for i in range(10):
            price_history.append([50000, 50500, 49500, 50000, 50000])
    
    with torch.no_grad():
        for step in range(steps):
            print(f"Predicting step {step + 1}/{steps}...")
            
            # Get prediction from model
            next_pred = model(current_1d, current_4h)
            raw_pred = next_pred.squeeze().cpu().numpy()
            raw_predictions.append(raw_pred)
            
            # Transform prediction to actual prices
            dummy_data = np.zeros((1, scaler.n_features_in_))
            dummy_data[0, 2] = raw_pred[0]  # open is column 2 in scaler order
            dummy_data[0, 7] = raw_pred[1]  # close is column 7 in scaler order
            
            try:
                inv_scaled = scaler.inverse_transform(dummy_data)
                pred_open = np.exp(inv_scaled[0, 2]) - log_epsilon  # from column 2 (open)
                pred_close = np.exp(inv_scaled[0, 7]) - log_epsilon  # from column 7 (close)
                pred_open = max(1.0, pred_open)  # Ensure positive
                pred_close = max(1.0, pred_close)  # Ensure positive
                
                actual_predictions.append([pred_open, pred_close])
                
                # Calculate high/low based on open/close without artificial volatility
                # Let the model's natural prediction variance drive the range
                high_price = max(pred_open, pred_close)
                low_price = min(pred_open, pred_close)
                
                # Volume estimate based on price movement (realistic correlation)
                price_change_pct = abs(pred_close - pred_open) / pred_open if pred_open > 0 else 0
                base_volume = 45000  # Base daily volume
                volume_multiplier = 1 + (price_change_pct * 2)  # Higher volume on big moves (realistic)
                estimated_volume = base_volume * volume_multiplier
                
                # Add to price history
                new_candle = [pred_open, high_price, low_price, pred_close, estimated_volume]
                price_history.append(new_candle)
                
                # Keep only recent history (60 periods for MA60)
                if len(price_history) > 60:
                    price_history = price_history[-60:]
                
                # Calculate confidence (decreases with prediction distance)
                base_confidence = 0.85
                distance_decay = step * 0.08  # 8% decay per step
                confidence = max(0.2, base_confidence - distance_decay)
                confidence_scores.append(confidence)
                
                print(f"  Predicted prices: Open=${pred_open:.0f}, Close=${pred_close:.0f}")
                
            except Exception as e:
                print(f"Warning: Error in price transformation at step {step}: {e}")
                # Use previous prediction or fallback
                if actual_predictions:
                    actual_predictions.append(actual_predictions[-1])
                else:
                    actual_predictions.append([50000.0, 50500.0])
                confidence_scores.append(0.1)
                
                # Add fallback to price history
                price_history.append([50000, 50500, 49500, 50000, 50000])
            
            # Update sequence with new prediction and calculated indicators
            new_1d_data = current_1d.clone()
            new_1d_data[0, :-1] = current_1d[0, 1:]  # Shift left
            
            # Get current scaled features template
            current_features = current_1d[0, -1].clone()
            
            # Calculate updated indicators using price history
            updated_features = calculate_real_indicators(
                price_history, 
                current_features, 
                feature_columns=None  # We know the order from indicators.py
            )
            
            # Override OHLC with model predictions (in scaled space, scaler column order)
            updated_features[2] = next_pred[0, 0]  # open (scaled) - index 2 in scaler order  
            updated_features[7] = next_pred[0, 1]  # close (scaled) - index 7 in scaler order
            
            # Calculate high/low in scaled space
            try:
                # Transform high/low to scaled space
                high_low_dummy = np.zeros((1, scaler.n_features_in_))
                high_low_dummy[0, 8] = high_price  # high is at index 8
                high_low_dummy[0, 5] = low_price   # low is at index 5
                
                high_low_log = np.log(high_low_dummy + log_epsilon)
                high_low_scaled = scaler.transform(high_low_log)
                
                updated_features[8] = high_low_scaled[0, 8]  # high (scaled)
                updated_features[5] = high_low_scaled[0, 5]  # low (scaled)
                
                # Volume in scaled space
                if len(updated_features) > 10:
                    volume_dummy = np.zeros((1, scaler.n_features_in_))
                    volume_dummy[0, 10] = estimated_volume  # volume is at index 10
                    volume_log = np.log(volume_dummy + log_epsilon)
                    volume_scaled = scaler.transform(volume_log)
                    updated_features[10] = volume_scaled[0, 10]
                
            except Exception as e:
                print(f"Warning: Error scaling high/low/volume: {e}")
                # Fallback - simple calculation
                updated_features[1] = max(updated_features[0], updated_features[3]) + 0.01  # high
                updated_features[2] = min(updated_features[0], updated_features[3]) - 0.01  # low
            
            # Transform calculated MA and BB indicators to scaled space
            try:
                # Scale MA features in scaler order: ma5=0, ma10=3, ma30=9, ma60=11
                ma_indices = [0, 3, 9, 11]  # ma5, ma10, ma30, ma60 in scaler order
                ma_dummy = np.zeros((1, scaler.n_features_in_))
                for ma_idx in ma_indices:
                    if len(updated_features) > ma_idx:
                        ma_value = max(1.0, abs(updated_features[ma_idx].item()))  # Ensure positive
                        ma_dummy[0, 7] = ma_value  # Use close column (index 7) for MA scaling
                        ma_log = np.log(ma_dummy + log_epsilon)
                        ma_scaled = scaler.transform(ma_log)
                        updated_features[ma_idx] = ma_scaled[0, 7]
                
                # Scale BB features in scaler order: bb_bbm=4, bb_bbh=6, bb_bbl=1
                bb_indices = [4, 6, 1]  # bb_bbm, bb_bbh, bb_bbl in scaler order
                bb_dummy = np.zeros((1, scaler.n_features_in_))
                for bb_idx in bb_indices:
                    if len(updated_features) > bb_idx:
                        bb_value = max(1.0, abs(updated_features[bb_idx].item()))  # Ensure positive
                        bb_dummy[0, 7] = bb_value  # Use close column (index 7) for BB scaling
                        bb_log = np.log(bb_dummy + log_epsilon)
                        bb_scaled = scaler.transform(bb_log)
                        updated_features[bb_idx] = bb_scaled[0, 7]
                            
            except Exception as e:
                print(f"Warning: Error scaling indicators: {e}")
                # Fallback: use decay approach for indicators
                for i in [0, 1, 3, 4, 6, 9, 11]:  # All indicator indices in scaler order
                    if i < len(updated_features):
                        updated_features[i] = updated_features[i] * 0.995
            
            # Set the updated features as the new data point
            new_1d_data[0, -1] = updated_features
            current_1d = new_1d_data
    
    print(f"Generated {len(actual_predictions)} daily predictions with calculated indicators")
    return np.array(actual_predictions), np.array(raw_predictions), np.array(confidence_scores)


def create_1d_multi_step_forecast(days=5, data_path="data/final_dataset.csv", save_results=True):
    """
    Create multi-day forecast using 1d data for better long-term predictions.
    
    Parameters:
        days: Number of days to predict
        data_path: Path to dataset
        save_results: Whether to save results
        
    Returns:
        predictions_df: DataFrame with daily predictions
    """
    days = int(days)  # Ensure days is an integer
    print(f"Creating {days}-day forecast using daily data...")
    print("="*60)
    
    try:
        # Load model and scaler
        print("Loading model and scaler...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler = joblib.load("models/scaler.save")
        input_size = scaler.n_features_in_
        
        model = DualLSTMPredictor(input_size=input_size)
        model.load_state_dict(torch.load("models/dual_lstm_final.pth", map_location=device))
        model = model.to(device)
        model.eval()
        
        # Prepare latest data
        print("Preparing latest data...")
        training_config = get_training_config()
        data_config = get_data_config()
        sequence_length = training_config['sequence_length']
        log_epsilon = data_config['log_epsilon']
        
        # Load data
        df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
        df_1d = df[df['timeframe'] == '1d'].drop(columns=['timeframe']).sort_index()
        
        # Reorder columns to match scaler's expected order
        scaler_column_order = ['ma5', 'bb_bbl', 'open', 'ma10', 'bb_bbm', 'low', 'bb_bbh', 'close', 'high', 'ma30', 'volume', 'ma60']
        df_1d = df_1d[scaler_column_order]
        
        print(f"Loaded 1D data: {len(df_1d)} rows")
        print(f"Column order: {df_1d.columns.tolist()}")
        
        # Log transform and scale
        df_1d_log = np.log(df_1d + log_epsilon)
        
        df_1d_scaled = pd.DataFrame(
            scaler.transform(df_1d_log), 
            index=df_1d.index, 
            columns=df_1d.columns
        )
        
        # Get latest sequences
        latest_1d = df_1d_scaled.values[-sequence_length:]
        
        latest_1d_tensor = torch.tensor(latest_1d, dtype=torch.float32)
        
        last_1d_timestamp = df_1d.index[-1]
        
        # Make daily predictions
        print(f"\nMaking {days}-day predictions...")
        predictions, raw_predictions, confidence_scores = predict_recursive_1d(
            model=model,
            initial_1d_data=latest_1d_tensor,
            scaler=scaler,
            steps=days,
            device=device
        )
        
        # Create future timestamps (daily)
        future_timestamps = pd.date_range(
            start=last_1d_timestamp + pd.Timedelta(days=1), 
            periods=days, 
            freq='D'
        )
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'predicted_open': predictions[:, 0],
            'predicted_close': predictions[:, 1],
            'confidence': confidence_scores,
            'days_ahead': list(range(1, days + 1))
        }, index=future_timestamps)
        
        # Add analysis columns
        # Calculate day-to-day changes (close-to-close)
        prev_close = df_1d['close'].iloc[-1]  # Last historical close price
        predictions_df['prev_close'] = 0.0
        predictions_df.loc[predictions_df.index[0], 'prev_close'] = prev_close
        
        # For subsequent days, previous close is the close of the previous predicted day
        for i in range(1, len(predictions_df)):
            predictions_df.loc[predictions_df.index[i], 'prev_close'] = predictions_df.loc[predictions_df.index[i-1], 'predicted_close']
        
        # Calculate proper day-to-day changes
        predictions_df['predicted_change'] = predictions_df['predicted_close'] - predictions_df['prev_close']
        predictions_df['predicted_change_pct'] = (predictions_df['predicted_change'] / predictions_df['prev_close']) * 100
        predictions_df['direction'] = (predictions_df['predicted_change'] > 0).astype(int)
        predictions_df['direction_label'] = predictions_df['direction'].map({1: 'UP', 0: 'DOWN'})
        
        # Print summary
        print(f"\n1D Forecast Summary:")
        print(f"Prediction period: {future_timestamps[0]} to {future_timestamps[-1]}")
        print(f"Starting price: ${predictions_df['predicted_open'].iloc[0]:,.2f}")
        print(f"Ending price: ${predictions_df['predicted_close'].iloc[-1]:,.2f}")
        
        total_change = ((predictions_df['predicted_close'].iloc[-1] - predictions_df['predicted_open'].iloc[0]) / 
                       predictions_df['predicted_open'].iloc[0]) * 100
        print(f"Total predicted change: {total_change:+.2f}%")
        
        bullish_days = (predictions_df['direction'] == 1).sum()
        print(f"Bullish days: {bullish_days}/{days}")
        print(f"Average confidence: {predictions_df['confidence'].mean():.3f}")
        
        if save_results:
            import os
            os.makedirs("predictions", exist_ok=True)
            
            # Save predictions
            predictions_file = "predictions/multi_step_forecast_1d.csv"
            predictions_df.to_csv(predictions_file)
            print(f"\nPredictions saved to: {predictions_file}")
            
            # Create visualization
            create_1d_forecast_plot(predictions_df, df_1d, save_path="predictions/multi_step_forecast_1d.png")
        
        return predictions_df
        
    except Exception as e:
        print(f"Error creating 1d forecast: {str(e)}")
        raise


def create_1d_forecast_plot(predictions_df, historical_1d, save_path=None):
    """Create visualization for 1d forecast"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    print("Creating 1d forecast visualization...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Bitcoin Daily Price Forecast (1D Data)', fontsize=16, fontweight='bold')
    
    # Plot 1: Price predictions with historical context
    historical_recent = historical_1d.tail(30)  # Last 30 days
    
    ax1.plot(historical_recent.index, historical_recent['close'], 'b-', label='Historical Close', linewidth=2)
    ax1.plot(historical_recent.index, historical_recent['open'], 'b:', label='Historical Open', alpha=0.7)
    
    # Plot predictions
    ax1.plot(predictions_df.index, predictions_df['predicted_close'], 'r-', label='Predicted Close', linewidth=2)
    ax1.plot(predictions_df.index, predictions_df['predicted_open'], 'r:', label='Predicted Open', alpha=0.8)
    
    # Add confidence bands
    confidence = predictions_df['confidence']
    upper_bound = predictions_df['predicted_close'] * (1 + (1 - confidence) * 0.15)
    lower_bound = predictions_df['predicted_close'] * (1 - (1 - confidence) * 0.15)
    ax1.fill_between(predictions_df.index, lower_bound, upper_bound, alpha=0.2, color='red', label='Confidence Band')
    
    # Add vertical line at prediction start
    last_hist_date = historical_recent.index[-1]
    ax1.axvline(x=last_hist_date, color='blue', linestyle='--', alpha=0.7, label='Prediction Start')
    
    ax1.set_title('Daily Price Forecast with Historical Context')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Plot 2: Daily changes and confidence
    colors = ['red' if d == 0 else 'green' for d in predictions_df['direction']]
    bars = ax2.bar(predictions_df.index, predictions_df['predicted_change_pct'], 
                   color=colors, alpha=0.7, width=0.8)
    
    # Add confidence line
    ax2_twin = ax2.twinx()
    ax2_twin.plot(predictions_df.index, predictions_df['confidence'], 'k--', label='Confidence', alpha=0.8)
    ax2_twin.set_ylabel('Confidence')
    ax2_twin.set_ylim(0, 1)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_title('Daily Price Changes and Confidence')
    ax2.set_ylabel('Daily Change (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Add direction labels
    for i, (idx, row) in enumerate(predictions_df.iterrows()):
        ax2.text(idx, row['predicted_change_pct'] + 0.3 if row['predicted_change_pct'] > 0 else row['predicted_change_pct'] - 0.3,
                row['direction_label'], ha='center', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"1D forecast plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function for 1d multi-step forecasting"""
    print("Bitcoin Multi-Day Forecasting (Using 1D Data)")
    print("="*50)
    
    try:
        # Create 5-day forecast using 1d data
        predictions_df = create_1d_multi_step_forecast(days=5, save_results=True)
        
        print(f"\n✅ 1D forecast completed successfully!")
        print(f"📊 Generated {len(predictions_df)} daily predictions")
        print(f"📁 Check 'predictions/' directory for results")
        
        return predictions_df
        
    except Exception as e:
        print(f"❌ Error in 1d forecasting: {str(e)}")
        return None


if __name__ == "__main__":
    main()
