"""
Price-focused Dual LSTM model training for Bitcoin price prediction.

This module implements a sophisticated dual-timeframe LSTM architecture that:
- Processes both 1D and 4H timeframe data with proper timestamp alignment
- Trains exclusively on price prediction (pure regression approach)
- Implements market continuity logic (next_open = current_close)
- Uses comprehensive early stopping and learning rate scheduling
- Provides detailed performance metrics and model persistence

The training process ensures temporal consistency and realistic price predictions
by modeling the continuous nature of financial markets.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib
from scripts.model import DualLSTMPredictor
from scripts.config import get_model_config, get_training_config, get_data_config, print_config
import warnings
warnings.filterwarnings('ignore')

def prepare_timeframe_data(df, timeframe_label, sequence_length):
    """
    Prepare sequential data for a specific timeframe.
    
    Args:
        df (pd.DataFrame): Combined dataset with timeframe labels
        timeframe_label (str): Target timeframe ('1d' or '4h')
        sequence_length (int): Number of historical periods for LSTM sequences
        
    Returns:
        tuple: (sequences, timestamps, timeframe_data)
            - sequences: numpy array of shape (n_samples, sequence_length, n_features)
            - timestamps: list of prediction timestamps
            - timeframe_data: filtered and sorted dataframe for the timeframe
    """
    timeframe_data = df[df['timeframe'] == timeframe_label].drop(columns=['timeframe'])
    timeframe_data = timeframe_data.sort_index()  # Ensure chronological order
    data = timeframe_data.values
    
    X = []
    timestamps = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        timestamps.append(timeframe_data.index[i+sequence_length])
    
    return np.array(X), timestamps, timeframe_data

def align_timeframes_by_timestamp(timestamps_1d, timestamps_4h, X_1d, X_4h, df_4h):
    print("Aligning timeframes based on timestamps...")
    
    # Convert to pandas datetime if not already
    ts_1d = pd.to_datetime(timestamps_1d)
    ts_4h = pd.to_datetime(timestamps_4h)
    
    aligned_pairs = []
    
    for i, ts_4h_current in enumerate(ts_4h):
        # Find the most recent 1d timestamp that is <= current 4h timestamp  
        valid_1d_mask = ts_1d <= ts_4h_current
        
        if valid_1d_mask.any():
            # Get the most recent valid 1d index
            latest_1d_idx = np.where(valid_1d_mask)[0][-1]
            
            # Ensure we have enough future data for price target creation
            if i + 1 < len(df_4h):  # Just need next candle for price target
                aligned_pairs.append({
                    '1d_idx': latest_1d_idx,
                    '4h_idx': i,
                    'timestamp': ts_4h_current
                })
    
    print(f"Successfully aligned {len(aligned_pairs)} timestamp pairs")
    return aligned_pairs

def create_price_targets(df_4h, aligned_pairs):
    """
    Create realistic price targets using market continuity principles.
    
    This function implements the market continuity assumption where the opening
    price of a new candle equals the closing price of the previous candle.
    This approach significantly improves open price prediction accuracy.
    
    Args:
        df_4h (pd.DataFrame): 4H timeframe data
        aligned_pairs (list): List of aligned timestamp pairs between timeframes
        
    Returns:
        np.array: Price targets of shape (n_samples, 2) where:
            - Column 0: current_close (becomes next_open due to market continuity)
            - Column 1: next_close (actual target price to predict)
    """
    print(f"Creating price targets for {len(aligned_pairs)} samples...")
    
    prices = []
    
    for pair in aligned_pairs:
        try:
            idx = pair['4h_idx']
            
            # Implement market continuity: next_open = current_close
            if idx + 1 < len(df_4h):
                current_close = df_4h.iloc[idx]['close']  # Current candle's close price
                next_close = df_4h.iloc[idx + 1]['close']  # Next candle's close price
                # Target structure: [current_close, next_close] where current_close becomes next_open
                prices.append([current_close, next_close])
        
        except (IndexError, KeyError):
            continue

    print(f"Generated {len(prices)} price targets")
    return np.array(prices)

class EarlyStopping:
    """Early stopping with best weight restoration"""
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        improved = val_loss < self.best_loss - self.min_delta
        
        if improved:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def restore_best_weights(self, model):
        if self.best_weights is not None:
            device = next(model.parameters()).device
            model.load_state_dict({k: v.to(device) for k, v in self.best_weights.items()})

def calculate_price_metrics(y_true, y_pred, scaler=None):
    """Calculate comprehensive metrics for price prediction"""
    metrics = {}
    
    # If we have a scaler, inverse transform to get actual prices
    if scaler is not None:
        # Create dummy data for inverse transform
        dummy_data = np.zeros((len(y_true), scaler.n_features_in_))
        # Assume open/close are the first two features (adjust if needed)
        dummy_data[:, :2] = y_true
        # Inverse: StandardScaler -> exp -> original
        y_true_std_inv = scaler.inverse_transform(dummy_data)[:, :2]
        
        data_config = get_data_config()
        log_epsilon = data_config['log_epsilon']
        y_true_actual = np.exp(y_true_std_inv) - log_epsilon
        
        dummy_data[:, :2] = y_pred
        y_pred_std_inv = scaler.inverse_transform(dummy_data)[:, :2]
        y_pred_actual = np.exp(y_pred_std_inv) - log_epsilon
    else:
        y_true_actual = y_true
        y_pred_actual = y_pred
    
    # Overall metrics
    metrics['mse'] = mean_squared_error(y_true_actual, y_pred_actual)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true_actual, y_pred_actual)
    
    # Per-feature metrics (open, close)
    for i, feature in enumerate(['open', 'close']):
        metrics[f'{feature}_mse'] = mean_squared_error(y_true_actual[:, i], y_pred_actual[:, i])
        metrics[f'{feature}_rmse'] = np.sqrt(metrics[f'{feature}_mse'])
        metrics[f'{feature}_mae'] = mean_absolute_error(y_true_actual[:, i], y_pred_actual[:, i])
        
        # Percentage error
        metrics[f'{feature}_mape'] = np.mean(np.abs((y_true_actual[:, i] - y_pred_actual[:, i]) / y_true_actual[:, i])) * 100
    
    return metrics

def train_model():
    """
    Train the Price-focused Dual LSTM model using centralized configuration.
    
    This function orchestrates the complete training pipeline:
    1. Load and validate configuration parameters
    2. Prepare and align multi-timeframe data
    3. Create realistic price targets with market continuity
    4. Split data using random sampling for robust evaluation
    5. Train model with early stopping and learning rate scheduling
    6. Evaluate performance with comprehensive metrics
    7. Save model artifacts and training history
    
    The training process focuses exclusively on price prediction using a dual
    LSTM architecture that processes both 1D and 4H timeframe information.
    
    Returns:
        dict: Training results containing model, metrics, and training history
    """
    # Load configurations
    model_config = get_model_config()
    training_config = get_training_config()
    data_config = get_data_config()
    
    print("Starting Price-Focused Dual LSTM training...")
    print_config()
    
    # Extract training parameters
    sequence_length = training_config['sequence_length']
    num_epochs = training_config['num_epochs']
    batch_size = training_config['batch_size']
    learning_rate = training_config['learning_rate']

    
    # Load data with timeframe column
    df = pd.read_csv("data/final_dataset.csv", index_col="timestamp", parse_dates=True)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Timeframes available: {df['timeframe'].unique()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Data quality check
    if df.isnull().any().any():
        print("Warning: Dataset contains NaN values - cleaning...")
        df = df.dropna()
        print(f"Cleaned dataset shape: {df.shape}")
    
    # Prepare scaler only on numeric columns (without timeframe)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    
    # DON'T fit scaler yet - will fit only on train data later
    df_numeric = df[numeric_cols].copy()
    
    # Use sequence_length from config
    sequence_length = training_config['sequence_length']
    print(f"Using sequence length: {sequence_length}")
    
    print("Preparing timeframe-specific data...")
    
    # Prepare data for both timeframes
    X_1d, timestamps_1d, df_1d = prepare_timeframe_data(df, '1d', sequence_length)
    X_4h, timestamps_4h, df_4h = prepare_timeframe_data(df, '4h', sequence_length)
    
    print(f"1d sequences: {len(X_1d)}")
    print(f"4h sequences: {len(X_4h)}")
    
    if len(X_1d) == 0 or len(X_4h) == 0:
        print("ERROR: Insufficient data for one or both timeframes!")
        return
    
    # Proper timestamp-based alignment
    aligned_pairs = align_timeframes_by_timestamp(
        timestamps_1d, timestamps_4h, X_1d, X_4h, df_4h
    )
    
    if len(aligned_pairs) < 50:
        print(f"ERROR: Not enough aligned samples for training: {len(aligned_pairs)}")
        return
    
    # Extract aligned sequences
    aligned_1d_indices = [pair['1d_idx'] for pair in aligned_pairs]
    aligned_4h_indices = [pair['4h_idx'] for pair in aligned_pairs]
    
    X_1d_aligned = X_1d[aligned_1d_indices]
    X_4h_aligned = X_4h[aligned_4h_indices]
    
    # Create price targets (no signal filtering)
    y_price = create_price_targets(df_4h, aligned_pairs)
    
    # Ensure consistent sample count
    min_samples = min(len(X_1d_aligned), len(X_4h_aligned), len(y_price))
    print(f"Adjusting to {min_samples} samples for consistency...")
    
    X_1d_aligned = X_1d_aligned[:min_samples]
    X_4h_aligned = X_4h_aligned[:min_samples] 
    y_price = y_price[:min_samples]
    
    print(f"Final dataset: {min_samples} samples")
    print(f"Data utilization: {min_samples/len(X_4h)*100:.1f}% of 4H data used")
    
    # Convert to tensors
    X_1d_tensor = torch.tensor(X_1d_aligned, dtype=torch.float32)
    X_4h_tensor = torch.tensor(X_4h_aligned, dtype=torch.float32)
    y_price_tensor = torch.tensor(y_price, dtype=torch.float32)

    print(f"Tensor shapes:")
    print(f"X_1d: {X_1d_tensor.shape}")
    print(f"X_4h: {X_4h_tensor.shape}")
    print(f"y_price: {y_price_tensor.shape}")

    # Data splitting (using config values)
    indices = list(range(len(X_1d_tensor)))
    
    print("Using random splitting for price regression...")
    # Calculate split sizes from config
    train_size = data_config['train_split']  
    val_size = data_config['val_split']      
    test_size = data_config['test_split']    
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices, 
        train_size=train_size, 
        random_state=data_config['random_state']
    )
    
    # Second split: val vs test from the temp portion
    val_ratio_in_temp = val_size / (val_size + test_size) 
    val_idx, test_idx = train_test_split(
        temp_idx, 
        train_size=val_ratio_in_temp, 
        random_state=data_config['random_state']
    )
    
    # Fit scaler ONLY on training data indices from original df  
    print("Fitting scaler on training data only...")
    df_train = df_numeric.iloc[train_idx]
    
    data_config = get_data_config()
    log_epsilon = data_config['log_epsilon']
    
    df_log = np.log(df_train + log_epsilon)
    scaler.fit(df_log)
    
    # Apply scaling to full dataset
    df_all_log = np.log(df_numeric + log_epsilon)
    df[numeric_cols] = scaler.transform(df_all_log)
    
    # Re-prepare sequences with scaled data
    print("Re-preparing scaled timeframe data...")
    X_1d, timestamps_1d, df_1d = prepare_timeframe_data(df, '1d', sequence_length)
    X_4h, timestamps_4h, df_4h = prepare_timeframe_data(df, '4h', sequence_length)
    
    # Re-align and create tensors
    aligned_pairs = align_timeframes_by_timestamp(timestamps_1d, timestamps_4h, X_1d, X_4h, df_4h)
    aligned_1d_indices = [pair['1d_idx'] for pair in aligned_pairs]
    aligned_4h_indices = [pair['4h_idx'] for pair in aligned_pairs]
    
    X_1d_aligned = X_1d[aligned_1d_indices]
    X_4h_aligned = X_4h[aligned_4h_indices]
    y_price = create_price_targets(df_4h, aligned_pairs)
    
    min_samples = min(len(X_1d_aligned), len(X_4h_aligned), len(y_price))
    X_1d_aligned = X_1d_aligned[:min_samples]
    X_4h_aligned = X_4h_aligned[:min_samples] 
    y_price = y_price[:min_samples]
    
    X_1d_tensor = torch.tensor(X_1d_aligned, dtype=torch.float32)
    X_4h_tensor = torch.tensor(X_4h_aligned, dtype=torch.float32)
    y_price_tensor = torch.tensor(y_price, dtype=torch.float32)
    
    # Create training sets from updated indices
    indices = list(range(len(X_1d_tensor))) 
    
    # Use the same split configuration as before
    train_idx, temp_idx = train_test_split(
        indices, 
        train_size=train_size, 
        random_state=data_config['random_state']
    )
    val_idx, test_idx = train_test_split(
        temp_idx, 
        train_size=val_ratio_in_temp, 
        random_state=data_config['random_state']
    )
    
    X_1d_train, X_1d_val, X_1d_test = X_1d_tensor[train_idx], X_1d_tensor[val_idx], X_1d_tensor[test_idx]
    X_4h_train, X_4h_val, X_4h_test = X_4h_tensor[train_idx], X_4h_tensor[val_idx], X_4h_tensor[test_idx]
    y_price_train, y_price_val, y_price_test = y_price_tensor[train_idx], y_price_tensor[val_idx], y_price_tensor[test_idx]
    
    print(f"Data split:")
    print(f"  Training: {len(train_idx)} samples ({len(train_idx)/len(indices)*100:.1f}%)")
    print(f"  Validation: {len(val_idx)} samples ({len(val_idx)/len(indices)*100:.1f}%)")
    print(f"  Test: {len(test_idx)} samples ({len(test_idx)/len(indices)*100:.1f}%)")

    # Model and optimization
    input_size = X_1d_tensor.shape[2]
    model = DualLSTMPredictor(input_size=input_size)
    
    # Loss function - only MSE for price prediction
    criterion_price = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=training_config['lr_scheduler_factor'], 
        patience=training_config['lr_scheduler_patience']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=training_config['early_stopping_patience'], 
        min_delta=training_config['early_stopping_min_delta']
    )

    print("Starting training (price prediction only)...")
    
    # Training
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        # Mini-batch training
        for i in range(0, len(X_1d_train), batch_size):
            end_idx = min(i + batch_size, len(X_1d_train))
            
            batch_1d = X_1d_train[i:end_idx]
            batch_4h = X_4h_train[i:end_idx]
            batch_price = y_price_train[i:end_idx]
            
            optimizer.zero_grad()
            
            # Get price prediction only
            price_pred = model(batch_1d, batch_4h)
            
            loss = criterion_price(price_pred, batch_price)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_config['gradient_clip_max_norm'])
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for i in range(0, len(X_1d_val), batch_size):
                end_idx = min(i + batch_size, len(X_1d_val))
                
                batch_1d = X_1d_val[i:end_idx]
                batch_4h = X_4h_val[i:end_idx]
                batch_price = y_price_val[i:end_idx]
                
                price_pred = model(batch_1d, batch_4h)
                
                loss = criterion_price(price_pred, batch_price)
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(range(0, len(X_1d_train), batch_size))
        avg_val_loss = val_loss / len(range(0, len(X_1d_val), batch_size))
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/dual_lstm_best.pth")
        
        # Record training history
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.6f}")
            print(f"Val Loss: {avg_val_loss:.6f}")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if early_stopping(avg_val_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Restore best weights
    early_stopping.restore_best_weights(model)

    # Final evaluation on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET - PRICE PREDICTION")
    print("="*50)
    
    model.eval()
    test_loss = 0.0
    test_price_preds = []
    test_price_targets = []
    
    with torch.no_grad():
        for i in range(0, len(X_1d_test), batch_size):
            end_idx = min(i + batch_size, len(X_1d_test))
            
            batch_1d = X_1d_test[i:end_idx]
            batch_4h = X_4h_test[i:end_idx]
            batch_price = y_price_test[i:end_idx]
            
            price_pred = model(batch_1d, batch_4h)
            
            loss = criterion_price(price_pred, batch_price)
            test_loss += loss.item()
            
            test_price_preds.extend(price_pred.cpu().numpy())
            test_price_targets.extend(batch_price.cpu().numpy())
    
    avg_test_loss = test_loss / len(range(0, len(X_1d_test), batch_size))
    test_price_preds = np.array(test_price_preds)
    test_price_targets = np.array(test_price_targets)
    
    # Calculate comprehensive price metrics
    test_metrics = calculate_price_metrics(test_price_targets, test_price_preds, scaler)
    
    # Print comprehensive results
    print(f"Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    
    print(f"\nPer-feature metrics:")
    for feature in ['open', 'close']:
        print(f"  {feature.upper()}:")
        print(f"    RMSE: {test_metrics[f'{feature}_rmse']:.6f}")
        print(f"    MAE: {test_metrics[f'{feature}_mae']:.6f}")
        print(f"    MAPE: {test_metrics[f'{feature}_mape']:.2f}%")

    # Save model and artifacts
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/dual_lstm_final.pth")
    joblib.dump(scaler, "models/scaler.save")
    joblib.dump(training_history, "models/training_history.save")
    
    # Save comprehensive model info
    model_info = {
        'model_architecture': 'Price-Focused DualLSTMPredictor',
        'training_mode': 'price_prediction_only',
        'input_size': input_size,
        'sequence_length': sequence_length,
        'num_samples': min_samples,
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_idx),
        'data_utilization': min_samples/len(X_4h)*100,
        'test_metrics': test_metrics,
        'training_epochs': epoch + 1,
        'best_val_loss': best_val_loss,
        'model_config': {
            **model_config,  # Include all model config
            **training_config  # Include all training config
        },
        'training_notes': 'Trained only on price prediction (pure regression). Uses all available data samples for better price prediction performance.'
    }
    
    joblib.dump(model_info, "models/model_info.save")
    
    # Save test set predictions for visualization
    test_results = {
        'test_predictions': test_price_preds,
        'test_targets': test_price_targets,
        'test_metrics': test_metrics
    }
    joblib.dump(test_results, "models/test_results.save")
    
    print("\n" + "="*50)
    print("PRICE-FOCUSED TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final test RMSE: {test_metrics['rmse']:.6f}")
    print(f"Final test MAE: {test_metrics['mae']:.6f}")
    print(f"Models saved in 'models/' directory")
    print("="*50)
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'model_info': model_info
    }

if __name__ == "__main__":
    results = train_model()
    print("Training completed successfully!")