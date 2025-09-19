"""
Evaluates the trained Dual LSTM model on the prepared dataset.

This module provides comprehensive evaluation capabilities for the Bitcoin price
prediction model, including:
- Loading trained model and preprocessing components
- Processing both 1D and 4H timeframe data
- Generating accurate price predictions with proper inverse transformation
- Saving predictions for visualization and analysis

The evaluation process correctly handles the market continuity assumption
where next_open = current_close, ensuring realistic price predictions.
"""

import torch
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.model import DualLSTMPredictor
from scripts.config import get_model_config, get_training_config, get_data_config
from sklearn.preprocessing import StandardScaler
import joblib

def prepare_evaluation_data(df, timeframe_label, sequence_length):
    """
    Prepare evaluation data for a specific timeframe.
    
    Args:
        df (pd.DataFrame): Combined dataset with timeframe labels
        timeframe_label (str): Target timeframe ('1d' or '4h')
        sequence_length (int): Number of historical periods for sequence
        
    Returns:
        tuple: (sequences, timestamps, timeframe_data)
            - sequences: numpy array of shape (n_samples, sequence_length, n_features)
            - timestamps: list of prediction timestamps
            - timeframe_data: filtered dataframe for the timeframe
    """
    timeframe_data = df[df['timeframe'] == timeframe_label].drop(columns=['timeframe'])
    data = timeframe_data.values
    
    X = []
    timestamps = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        timestamps.append(timeframe_data.index[i+sequence_length])
    
    return np.array(X), timestamps, timeframe_data

def evaluate_model():
    """
    Evaluate the Dual LSTM model and generate Bitcoin price predictions.
    
    This function performs comprehensive model evaluation by:
    1. Loading the trained model, scaler, and configuration
    2. Preparing evaluation data for both 1D and 4H timeframes
    3. Aligning timeframes using timestamp matching or approximation
    4. Generating predictions using the trained model
    5. Applying correct inverse transformation to get actual price values
    6. Saving predictions to CSV file for analysis
    
    The evaluation correctly handles the market continuity principle where
    the opening price of a new candle equals the closing price of the previous candle.
    
    Returns:
        None: Results are saved to 'outputs/predictions.csv'
    """
    print("Starting Dual LSTM evaluation...")
    
    # Get base directory (parent of scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    # Load dataset with timestamp index
    df = pd.read_csv(os.path.join(base_dir, "data", "final_dataset.csv"), index_col="timestamp", parse_dates=True)
    
    # Load trained model components
    scaler = joblib.load(os.path.join(base_dir, "models", "scaler.save"))
    model_info = joblib.load(os.path.join(base_dir, "models", "model_info.save"))
    
    sequence_length = model_info['sequence_length']
    input_size = model_info['input_size']
    
    print(f"Model info - Input size: {input_size}, Sequence length: {sequence_length}")
    
    # Prepare data normalization (using the same scaler from training)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    data_config = get_data_config()
    log_epsilon = data_config['log_epsilon']
    
    # Apply log transformation and scaling as used during training
    df_log = np.log(df[numeric_cols] + log_epsilon)
    df[numeric_cols] = scaler.transform(df_log)
    
    print("Preparing evaluation data...")
    
    # Extract data by timeframes  
    X_1d, timestamps_1d, df_1d = prepare_evaluation_data(df, '1d', sequence_length)
    X_4h, timestamps_4h, df_4h = prepare_evaluation_data(df, '4h', sequence_length)
    
    print(f"1d evaluation sequences: {len(X_1d)}")
    print(f"4h evaluation sequences: {len(X_4h)}")
    
    # Align data by timestamps
    common_timestamps = list(set(timestamps_1d) & set(timestamps_4h))
    common_timestamps.sort()
    
    print(f"Common timestamps for evaluation: {len(common_timestamps)}")
    
    if len(common_timestamps) < 10:
        print("WARNING: Very few common timestamps for evaluation!")
        print("Using timestamp approximation...")
        
        # Fallback: match by closest timestamps
        X_1d_aligned = []
        X_4h_aligned = []
        pred_timestamps = []
        
        for i, ts_1d in enumerate(timestamps_1d):
            # Find closest 4h timestamp
            closest_4h_idx = None
            min_diff = float('inf')
            
            for j, ts_4h in enumerate(timestamps_4h):
                diff = abs((ts_1d - ts_4h).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    closest_4h_idx = j
            
            # Only use if timestamps are reasonably close (within 4 hours)
            if closest_4h_idx is not None and min_diff <= 4 * 3600:  # 4 hours in seconds
                X_1d_aligned.append(X_1d[i])
                X_4h_aligned.append(X_4h[closest_4h_idx])
                pred_timestamps.append(ts_1d)
    else:
        # Exact timestamp matching
        X_1d_aligned = []
        X_4h_aligned = []
        pred_timestamps = []
        
        for ts in common_timestamps:
            try:
                idx_1d = timestamps_1d.index(ts)
                idx_4h = timestamps_4h.index(ts)
                
                X_1d_aligned.append(X_1d[idx_1d])
                X_4h_aligned.append(X_4h[idx_4h])
                pred_timestamps.append(ts)
                
            except ValueError:
                continue
    
    if not X_1d_aligned:
        print("ERROR: No aligned data for evaluation!")
        return
    
    print(f"Final aligned evaluation samples: {len(X_1d_aligned)}")
    
    # Convert lists to numpy arrays first for efficient tensor creation
    X_1d_array = np.array(X_1d_aligned)
    X_4h_array = np.array(X_4h_aligned)
    
    X_1d_tensor = torch.tensor(X_1d_array, dtype=torch.float32)
    X_4h_tensor = torch.tensor(X_4h_array, dtype=torch.float32)
    
    # Load configurations to match training
    model_config = get_model_config()
    training_config = get_training_config()
    
    # Load model with same architecture as training (automatically from config)
    model = DualLSTMPredictor(input_size=input_size)
    model.load_state_dict(torch.load(os.path.join(base_dir, "models", "dual_lstm_final.pth")))
    model.eval()
    
    print("Generating predictions...")
    
    # Generate predictions 
    with torch.no_grad():
        price_pred = model(X_1d_tensor, X_4h_tensor)  
        
        # Convert to numpy
        price_pred = price_pred.numpy()
    
    print(f"Generated {len(price_pred)} predictions")
    
    # Inverse transform prices with market continuity fix
    predictions = []
    for i in range(len(price_pred)):
        # Create dummy array for inverse transformation with correct column positioning
        # The scaler expects features in the original dataset order:
        # [volume, open, high, low, close, bb_bbm, ma60, bb_bbh, ma10, bb_bbl, ma5, ma30]
        dummy_data = np.zeros((1, len(numeric_cols)))
        dummy_data[0, 1] = price_pred[i][0]  # Place predicted open in column 1 (open position)
        dummy_data[0, 4] = price_pred[i][1]  # Place predicted close in column 4 (close position)
        
        try:
            # Apply inverse transformations to recover original price values
            # Step 1: Inverse StandardScaler transformation
            prices_std_inv = scaler.inverse_transform(dummy_data)[0]
            
            data_config = get_data_config()
            log_epsilon = data_config['log_epsilon']
            
            # Step 2: Inverse log transformation to get actual price values
            pred_open = np.exp(prices_std_inv[1]) - log_epsilon   # Extract from column 1 (open)
            pred_close = np.exp(prices_std_inv[4]) - log_epsilon  # Extract from column 4 (close)
            
            # Store prediction results
            predictions.append({
                "timestamp": pred_timestamps[i],
                "pred_open": pred_open,
                "pred_close": pred_close
            })
        except Exception as e:
            print(f"Error processing prediction {i}: {e}")
            continue
    
    if not predictions:
        print("ERROR: No valid predictions generated!")
        return
    
    # Save predictions (current only for now)
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(os.path.join(outputs_dir, "predictions.csv"), index=False)
    
    print(f"Saved {len(predictions)} predictions to outputs/predictions.csv")
    print(f"Date range: {pred_df['timestamp'].min()} to {pred_df['timestamp'].max()}")
    print(f"Average predicted close: ${pred_df['pred_close'].mean():.2f}")

if __name__ == "__main__":
    evaluate_model()
