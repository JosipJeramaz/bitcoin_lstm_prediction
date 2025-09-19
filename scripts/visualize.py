"""
Comprehensive visualization suite for Bitcoin LSTM price prediction analysis.

This module provides professional-grade visualization capabilities including:
- Actual vs predicted price comparisons with timestamp alignment
- Training history and convergence analysis
- Rigorous test set performance evaluation
- Statistical residual analysis for model validation
- Correlation analysis and prediction accuracy metrics

The visualization suite generates publication-ready charts that demonstrate
model performance, prediction accuracy, and statistical validity of the
Bitcoin price forecasting system.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import numpy as np
import os
from config import get_data_config

def plot_predictions():
    """
    Generate comprehensive prediction visualization with current and future forecasts.
    
    Creates a professional line chart showing Bitcoin price predictions over time,
    distinguishing between historical predictions and future forecasts with
    appropriate styling and annotations.
    """
    # Get base directory (parent of scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    df = pd.read_csv(os.path.join(base_dir, "outputs", "predictions.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Separate current and future predictions
    if 'prediction_type' in df.columns:
        current_df = df[df['prediction_type'] == 'current'].copy()
        future_df = df[df['prediction_type'] == 'future'].copy()
    else:
        current_df = df.copy()
        future_df = pd.DataFrame()
    
    plt.figure(figsize=(18, 10))
    
    # Plot current predictions
    if not current_df.empty:
        plt.plot(current_df["timestamp"], current_df["pred_open"], 
                label="Current Predicted Open", alpha=0.7, color='blue', linewidth=2)
        plt.plot(current_df["timestamp"], current_df["pred_close"], 
                label="Current Predicted Close", alpha=0.7, color='red', linewidth=2)
    
    # Plot future predictions with different style
    if not future_df.empty:
        plt.plot(future_df["timestamp"], future_df["pred_open"], 
                label="Future Predicted Open (+10 days)", alpha=0.8, color='lightblue', 
                linewidth=3, linestyle='--', marker='o', markersize=4)
        plt.plot(future_df["timestamp"], future_df["pred_close"], 
                label="Future Predicted Close (+10 days)", alpha=0.8, color='orange', 
                linewidth=3, linestyle='--', marker='s', markersize=4)
        
        # Add vertical line to separate current from future
        if not current_df.empty:
            separation_line = current_df["timestamp"].max()
            plt.axvline(x=separation_line, color='gray', linestyle=':', 
                       alpha=0.7, linewidth=2, label='Current → Future')
    
    plt.title("Bitcoin Price Predictions - Current & Future", fontsize=16, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Improve date formatting
    from matplotlib.dates import DateFormatter, MonthLocator, WeekdayLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "outputs", "prediction_plot.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    if not future_df.empty:
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"Current predictions: {len(current_df)} days")
        print(f"Future predictions: {len(future_df)} days")
        print(f"Future price range: ${future_df['pred_close'].min():.2f} - ${future_df['pred_close'].max():.2f}")
        print(f"Expected trend: {'📈 Bullish' if future_df['pred_close'].iloc[-1] > future_df['pred_close'].iloc[0] else '📉 Bearish'}")
    else:
        print(f"📊 Showing {len(current_df)} current predictions only")

def plot_actual_vs_predicted():
    """Compare actual historical prices with model predictions using proper timestamp alignment"""
    # Get base directory (parent of scripts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    # Load predictions
    pred_df = pd.read_csv(os.path.join(base_dir, "outputs", "predictions.csv"))
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    
    # Load actual historical data
    actual_df = pd.read_csv(os.path.join(base_dir, "data", "final_dataset.csv"), 
                           parse_dates=["timestamp"])
    
    # Filter only 1d data for comparison (since predictions are daily)
    actual_df = actual_df[actual_df['timeframe'] == '1d'].copy()
    
    # CRITICAL: Merge by exact timestamp match
    merged_df = pd.merge(
        pred_df[['timestamp', 'pred_open', 'pred_close']], 
        actual_df[['timestamp', 'open', 'close']], 
        on='timestamp', 
        how='inner'
    )
    
    if len(merged_df) == 0:
        print("❌ No matching timestamps found between predictions and actual data!")
        print("Prediction range:", pred_df['timestamp'].min(), "to", pred_df['timestamp'].max())
        print("Actual data range:", actual_df['timestamp'].min(), "to", actual_df['timestamp'].max())
        return
    
    print(f"✅ Found {len(merged_df)} matching timestamps for comparison")
    print(f"Comparison range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Actual vs Predicted Bitcoin Prices (Timestamp-Aligned)', fontsize=16, fontweight='bold')
    
    # 1. Close Price Comparison
    axes[0,0].plot(merged_df['timestamp'], merged_df['close'], 
                   label='Actual Close', color='blue', linewidth=2, alpha=0.8)
    axes[0,0].plot(merged_df['timestamp'], merged_df['pred_close'], 
                   label='Predicted Close', color='red', linewidth=2, alpha=0.8)
    axes[0,0].set_title('Close Price: Actual vs Predicted')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Price (USD)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Open Price Comparison
    axes[0,1].plot(merged_df['timestamp'], merged_df['open'], 
                   label='Actual Open', color='green', linewidth=2, alpha=0.8)
    axes[0,1].plot(merged_df['timestamp'], merged_df['pred_open'], 
                   label='Predicted Open', color='orange', linewidth=2, alpha=0.8)
    axes[0,1].set_title('Open Price: Actual vs Predicted')
    axes[0,1].set_xlabel('Date')
    axes[0,1].set_ylabel('Price (USD)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Price Difference Analysis (in percentage)
    close_diff = merged_df['pred_close'] - merged_df['close']
    open_diff = merged_df['pred_open'] - merged_df['open']
    close_diff_pct = (close_diff / merged_df['close']) * 100
    open_diff_pct = (open_diff / merged_df['open']) * 100
    
    axes[1,0].plot(merged_df['timestamp'], close_diff_pct, 
                   label='Close Price Error (%)', color='purple', linewidth=1.5)
    axes[1,0].plot(merged_df['timestamp'], open_diff_pct, 
                   label='Open Price Error (%)', color='brown', linewidth=1.5)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].set_title('Price Prediction Errors (Percentage)')
    axes[1,0].set_xlabel('Date')
    axes[1,0].set_ylabel('Error (%)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Scatter Plot - Correlation
    axes[1,1].scatter(merged_df['close'], merged_df['pred_close'], 
                     alpha=0.6, color='blue', s=20, label='Close Price')
    axes[1,1].scatter(merged_df['open'], merged_df['pred_open'], 
                     alpha=0.6, color='red', s=20, label='Open Price')
    
    # Add perfect prediction line (y=x)
    min_price = min(merged_df[['close', 'open', 'pred_close', 'pred_open']].min())
    max_price = max(merged_df[['close', 'open', 'pred_close', 'pred_open']].max())
    axes[1,1].plot([min_price, max_price], [min_price, max_price], 
                   'k--', alpha=0.5, label='Perfect Prediction')
    
    axes[1,1].set_title('Actual vs Predicted Correlation')
    axes[1,1].set_xlabel('Actual Price (USD)')
    axes[1,1].set_ylabel('Predicted Price (USD)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "outputs", "actual_vs_predicted.png"), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and print CORRECTED accuracy metrics
    print("\n" + "="*60)
    print("TIMESTAMP-ALIGNED PREDICTION ACCURACY ANALYSIS")
    print("="*60)
    
    # Mean Absolute Error (MAE)
    mae_close = np.mean(np.abs(close_diff))
    mae_open = np.mean(np.abs(open_diff))
    
    # Mean Absolute Percentage Error (MAPE)
    mape_close = np.mean(np.abs(close_diff_pct))
    mape_open = np.mean(np.abs(open_diff_pct))
    
    # Root Mean Square Error (RMSE)
    rmse_close = np.sqrt(np.mean(close_diff**2))
    rmse_open = np.sqrt(np.mean(open_diff**2))
    
    # Correlation coefficient
    corr_close = np.corrcoef(merged_df['close'], merged_df['pred_close'])[0,1]
    corr_open = np.corrcoef(merged_df['open'], merged_df['pred_open'])[0,1]
    
    print(f"📅 Timestamp-aligned comparison covers {len(merged_df)} data points")
    print(f"📆 Date range: {merged_df['timestamp'].min().strftime('%Y-%m-%d')} to {merged_df['timestamp'].max().strftime('%Y-%m-%d')}")
    
    print(f"\nClose Price Metrics:")
    print(f"  MAE:  ${mae_close:.2f}")
    print(f"  MAPE: {mape_close:.2f}%")
    print(f"  RMSE: ${rmse_close:.2f}")
    print(f"  Correlation: {corr_close:.3f}")
    
    print(f"\nOpen Price Metrics:")
    print(f"  MAE:  ${mae_open:.2f}")
    print(f"  MAPE: {mape_open:.2f}%") 
    print(f"  RMSE: ${rmse_open:.2f}")
    print(f"  Correlation: {corr_open:.3f}")
    
    # Show data range issues if any
    if len(merged_df) < len(pred_df):
        missing = len(pred_df) - len(merged_df)
        print(f"\n⚠️  {missing} predictions couldn't be matched with actual data")
        print("This suggests the model is predicting beyond available historical data")

def plot_training_history():
    """Plot learning rate, train loss, and val loss from training history CSV"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_path = os.path.join(base_dir, "models", "multistep_training_history.csv")
    
    if not os.path.exists(history_path):
        print(f"Training history file not found: {history_path}")
        return
    
    try:
        history_df = pd.read_csv(history_path)
        print(f"Loaded training history with {len(history_df)} epochs")

        plt.figure(figsize=(10, 6))
        plt.plot(history_df['train_loss'], label='Train Loss', color='blue')
        plt.plot(history_df['val_loss'], label='Val Loss', color='orange')
        plt.plot(history_df['learning_rate'], label='Learning Rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Learning Rate')
        plt.title('Training & Validation Loss and Learning Rate Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        output_dir = os.path.join(base_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "training_val_loss_lr.png"))
        plt.show()
        print("Training history plot saved and displayed")
    except Exception as e:
        print(f"Error plotting training history: {e}")

def plot_test_set_performance():
    """Plot rigorous test set performance from train.py - the most accurate metrics"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_results_path = os.path.join(base_dir, "models", "test_results.save")
    
    if not os.path.exists(test_results_path):
        print(f"Test results file not found: {test_results_path}")
        print("Run training first to generate test results.")
        return
    
    try:
        import joblib
        from sklearn.preprocessing import StandardScaler
        
        # Load test results
        test_results = joblib.load(test_results_path)
        test_preds = test_results['test_predictions']  # Shape: (n_samples, 2) - [open, close]
        test_targets = test_results['test_targets']    # Shape: (n_samples, 2) - [open, close]
        test_metrics = test_results['test_metrics']
        
        # Load scaler to denormalize
        scaler = joblib.load(os.path.join(base_dir, "models", "scaler.save"))
        
        print(f"Loaded test set results: {len(test_preds)} samples")
        
        # Denormalize predictions and targets 
        # Create dummy arrays with zeros for other features, then denormalize
        dummy_shape = scaler.scale_.shape[0]  # Number of features scaler expects
        
        dummy_preds = np.zeros((len(test_preds), dummy_shape))
        dummy_targets = np.zeros((len(test_targets), dummy_shape))
        
        dummy_preds[:, :2] = test_preds  # Put open, close predictions in first 2 columns
        dummy_targets[:, :2] = test_targets  # Put open, close targets in first 2 columns
        
        # Denormalize: StandardScaler -> exp -> original
        denorm_preds_std = scaler.inverse_transform(dummy_preds)[:, :2]  
        denorm_targets_std = scaler.inverse_transform(dummy_targets)[:, :2]  
        
        data_config = get_data_config()
        log_epsilon = data_config['log_epsilon']
        
        denorm_preds = np.exp(denorm_preds_std) - log_epsilon
        denorm_targets = np.exp(denorm_targets_std) - log_epsilon
        
        pred_open, pred_close = denorm_preds[:, 0], denorm_preds[:, 1]
        true_open, true_close = denorm_targets[:, 0], denorm_targets[:, 1]
        
        # Create test set visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Rigorous Test Set Performance (Train.py Metrics)', fontsize=16, fontweight='bold')
        
        # Sample indices for x-axis (since we don't have timestamps for test set)
        sample_indices = np.arange(len(pred_open))
        
        # 1. Actual vs Predicted Close Prices
        axes[0,0].scatter(true_close, pred_close, alpha=0.6, s=20, color='blue')
        
        # Perfect prediction line
        min_price = min(np.min(true_close), np.min(pred_close))
        max_price = max(np.max(true_close), np.max(pred_close))
        axes[0,0].plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.7, label='Perfect Prediction')
        
        # Calculate correlation for close
        close_corr = np.corrcoef(true_close, pred_close)[0,1]
        axes[0,0].set_title(f'Close Price Accuracy (Correlation: {close_corr:.3f})')
        axes[0,0].set_xlabel('Actual Close Price ($)')
        axes[0,0].set_ylabel('Predicted Close Price ($)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted Open Prices  
        axes[0,1].scatter(true_open, pred_open, alpha=0.6, s=20, color='green')
        axes[0,1].plot([min_price, max_price], [min_price, max_price], 'r--', alpha=0.7, label='Perfect Prediction')
        
        # Calculate correlation for open
        open_corr = np.corrcoef(true_open, pred_open)[0,1]
        axes[0,1].set_title(f'Open Price Accuracy (Correlation: {open_corr:.3f})')
        axes[0,1].set_xlabel('Actual Open Price ($)')
        axes[0,1].set_ylabel('Predicted Open Price ($)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Error Distribution - Close
        close_errors = pred_close - true_close
        close_error_pct = (close_errors / true_close) * 100
        
        axes[1,0].hist(close_error_pct, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1,0].axvline(x=0, color='red', linestyle='--', label='Zero Error')
        axes[1,0].axvline(x=np.mean(close_error_pct), color='orange', linestyle='-', label=f'Mean: {np.mean(close_error_pct):.2f}%')
        axes[1,0].set_title(f'Close Price Error Distribution (MAPE: {test_metrics["close_mape"]:.2f}%)')
        axes[1,0].set_xlabel('Prediction Error (%)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Error Distribution - Open
        open_errors = pred_open - true_open
        open_error_pct = (open_errors / true_open) * 100
        
        axes[1,1].hist(open_error_pct, bins=30, alpha=0.7, color='brown', edgecolor='black')
        axes[1,1].axvline(x=0, color='red', linestyle='--', label='Zero Error')
        axes[1,1].axvline(x=np.mean(open_error_pct), color='orange', linestyle='-', label=f'Mean: {np.mean(open_error_pct):.2f}%')
        axes[1,1].set_title(f'Open Price Error Distribution (MAPE: {test_metrics["open_mape"]:.2f}%)')
        axes[1,1].set_xlabel('Prediction Error (%)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(base_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "test_set_performance.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed metrics
        print("="*60)
        print("🎯 RIGOROUS TEST SET PERFORMANCE (Train.py Metrics)")
        print("="*60)
        print(f"📊 Test set size: {len(test_preds)} samples")
        print(f"📈 Most accurate metrics - same scaler, proper train/test split")
        print("")
        print("📍 CLOSE PRICE PERFORMANCE:")
        print(f"   RMSE: ${test_metrics['close_rmse']:.2f}")
        print(f"   MAE:  ${test_metrics['close_mae']:.2f}") 
        print(f"   MAPE: {test_metrics['close_mape']:.2f}%")
        print(f"   Correlation: {close_corr:.3f}")
        print("")
        print("📍 OPEN PRICE PERFORMANCE:")
        print(f"   RMSE: ${test_metrics['open_rmse']:.2f}")
        print(f"   MAE:  ${test_metrics['open_mae']:.2f}")
        print(f"   MAPE: {test_metrics['open_mape']:.2f}%")
        print(f"   Correlation: {open_corr:.3f}")
        print("")
        print("✅ These are the most reliable performance metrics!")
        print("="*60)
        
    except Exception as e:
        print(f"Error plotting test set performance: {e}")
        import traceback
        traceback.print_exc()

def plot_residual_analysis():
    """Detailed residual analysis for model validation"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_results_path = os.path.join(base_dir, "models", "test_results.save")
    
    if not os.path.exists(test_results_path):
        print(f"Test results file not found: {test_results_path}")
        print("Run training first to generate test results.")
        return
    
    try:
        import joblib
        from sklearn.preprocessing import StandardScaler
        from scipy import stats
        
        # Load test results
        test_results = joblib.load(test_results_path)
        test_preds = test_results['test_predictions']
        test_targets = test_results['test_targets'] 
        test_metrics = test_results['test_metrics']
        
        # Load scaler to denormalize
        scaler = joblib.load(os.path.join(base_dir, "models", "scaler.save"))
        
        # Denormalize predictions and targets
        dummy_shape = scaler.scale_.shape[0]
        
        dummy_preds = np.zeros((len(test_preds), dummy_shape))
        dummy_targets = np.zeros((len(test_targets), dummy_shape))
        
        dummy_preds[:, :2] = test_preds
        dummy_targets[:, :2] = test_targets
        
        denorm_preds_std = scaler.inverse_transform(dummy_preds)[:, :2]
        denorm_targets_std = scaler.inverse_transform(dummy_targets)[:, :2]
        
        data_config = get_data_config()
        log_epsilon = data_config['log_epsilon']
        
        denorm_preds = np.exp(denorm_preds_std) - log_epsilon
        denorm_targets = np.exp(denorm_targets_std) - log_epsilon
        
        pred_open, pred_close = denorm_preds[:, 0], denorm_preds[:, 1]
        true_open, true_close = denorm_targets[:, 0], denorm_targets[:, 1]
        
        # Calculate residuals
        residuals_open = pred_open - true_open
        residuals_close = pred_close - true_close
        
        # Calculate percentage residuals
        residuals_open_pct = (residuals_open / true_open) * 100
        residuals_close_pct = (residuals_close / true_close) * 100
        
        # Create comprehensive residual analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Residual Analysis - Model Validation', fontsize=16, fontweight='bold')
        
        # 1. Residuals vs Predicted Values (Heteroscedasticity check)
        axes[0,0].scatter(pred_close, residuals_close, alpha=0.6, s=20, color='blue')
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0,0].set_title('Close: Residuals vs Predicted')
        axes[0,0].set_xlabel('Predicted Close Price ($)')
        axes[0,0].set_ylabel('Residuals ($)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add trend line to check for patterns
        z = np.polyfit(pred_close, residuals_close, 1)
        p = np.poly1d(z)
        axes[0,0].plot(pred_close, p(pred_close), "g--", alpha=0.7, label=f'Trend (slope: {z[0]:.2f})')
        axes[0,0].legend()
        
        # 2. Residuals vs Predicted Values for Open
        axes[0,1].scatter(pred_open, residuals_open, alpha=0.6, s=20, color='green')
        axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0,1].set_title('Open: Residuals vs Predicted')
        axes[0,1].set_xlabel('Predicted Open Price ($)')
        axes[0,1].set_ylabel('Residuals ($)')
        axes[0,1].grid(True, alpha=0.3)
        
        z_open = np.polyfit(pred_open, residuals_open, 1)
        p_open = np.poly1d(z_open)
        axes[0,1].plot(pred_open, p_open(pred_open), "g--", alpha=0.7, label=f'Trend (slope: {z_open[0]:.2f})')
        axes[0,1].legend()
        
        # 3. Residual Distribution (Normality check)
        axes[0,2].hist(residuals_close_pct, bins=30, alpha=0.7, color='purple', edgecolor='black', density=True)
        
        # Overlay normal distribution
        mu, sigma = stats.norm.fit(residuals_close_pct)
        x = np.linspace(residuals_close_pct.min(), residuals_close_pct.max(), 100)
        axes[0,2].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        
        axes[0,2].set_title('Close: Residual Distribution')
        axes[0,2].set_xlabel('Residuals (%)')
        axes[0,2].set_ylabel('Density')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Q-Q Plot for normality (Close)
        stats.probplot(residuals_close_pct, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('Close: Q-Q Plot (Normality Test)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Residuals Time Series (if we had time index)
        sample_indices = np.arange(len(residuals_close))
        axes[1,1].plot(sample_indices, residuals_close_pct, alpha=0.7, color='blue', linewidth=1)
        axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Add rolling mean and std
        window = min(50, len(residuals_close_pct)//10)
        if window > 1:
            rolling_mean = pd.Series(residuals_close_pct).rolling(window=window).mean()
            rolling_std = pd.Series(residuals_close_pct).rolling(window=window).std()
            
            axes[1,1].plot(sample_indices, rolling_mean, 'g-', alpha=0.8, label=f'Rolling Mean ({window})')
            axes[1,1].fill_between(sample_indices, rolling_mean - rolling_std, rolling_mean + rolling_std, 
                                  alpha=0.2, color='gray', label='±1 Std')
        
        axes[1,1].set_title('Close: Residuals Over Time')
        axes[1,1].set_xlabel('Sample Index')
        axes[1,1].set_ylabel('Residuals (%)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Absolute Residuals vs Predicted (Variance check)
        axes[1,2].scatter(pred_close, np.abs(residuals_close_pct), alpha=0.6, s=20, color='orange')
        axes[1,2].set_title('Close: Absolute Residuals vs Predicted')
        axes[1,2].set_xlabel('Predicted Close Price ($)')
        axes[1,2].set_ylabel('|Residuals| (%)')
        axes[1,2].grid(True, alpha=0.3)
        
        # Add variance trend line
        z_var = np.polyfit(pred_close, np.abs(residuals_close_pct), 1)
        p_var = np.poly1d(z_var)
        axes[1,2].plot(pred_close, p_var(pred_close), "r--", alpha=0.7, 
                      label=f'Variance trend (slope: {z_var[0]:.4f})')
        axes[1,2].legend()
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(base_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "residual_analysis.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical tests and interpretation
        print("="*70)
        print("📊 RESIDUAL ANALYSIS - MODEL VALIDATION")
        print("="*70)
        
        # Normality tests
        from scipy.stats import shapiro, jarque_bera
        
        # Shapiro-Wilk test (works better for small samples)
        if len(residuals_close_pct) < 5000:
            shapiro_stat, shapiro_p = shapiro(residuals_close_pct)
            print(f"🔍 Shapiro-Wilk Normality Test (Close):")
            print(f"   Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4e}")
            print(f"   Result: {'✅ Residuals appear normal' if shapiro_p > 0.05 else '⚠️ Residuals may not be normal'}")
        
        # Jarque-Bera test
        jb_stat, jb_p = jarque_bera(residuals_close_pct)
        print(f"\n🔍 Jarque-Bera Normality Test (Close):")
        print(f"   Statistic: {jb_stat:.4f}, p-value: {jb_p:.4e}")
        print(f"   Result: {'✅ Residuals appear normal' if jb_p > 0.05 else '⚠️ Residuals may not be normal'}")
        
        # Residual statistics
        print(f"\n📈 RESIDUAL STATISTICS:")
        print(f"   Mean residual (Close): {np.mean(residuals_close_pct):.3f}% (should be ~0)")
        print(f"   Std residual (Close): {np.std(residuals_close_pct):.3f}%")
        print(f"   Mean residual (Open): {np.mean(residuals_open_pct):.3f}% (should be ~0)")
        print(f"   Std residual (Open): {np.std(residuals_open_pct):.3f}%")
        
        # Heteroscedasticity interpretation
        data_config = get_data_config()
        slope_threshold = data_config['slope_threshold']
        variance_threshold = data_config['variance_threshold']
        
        print(f"\n🔍 HETEROSCEDASTICITY CHECK:")
        print(f"   Close trend slope: {z[0]:.4f}")
        print(f"   Open trend slope: {z_open[0]:.4f}")
        print(f"   Result: {'✅ Homoscedastic' if abs(z[0]) < slope_threshold and abs(z_open[0]) < slope_threshold else '⚠️ May have heteroscedasticity'}")
        
        # Variance trend interpretation
        print(f"\n📊 VARIANCE ANALYSIS:")
        print(f"   Variance trend slope: {z_var[0]:.6f}")
        print(f"   Result: {'✅ Constant variance' if abs(z_var[0]) < variance_threshold else '⚠️ Variance may increase with predicted values'}")
        
        print(f"\n✅ Model validation complete - check residual_analysis.png for visual details")
        print("="*70)
        
    except Exception as e:
        print(f"Error in residual analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Creating visualizations...")
    print("1. Plotting predictions with future projections...")
    plot_predictions()
    print("2. Plotting actual vs predicted comparison...")
    plot_actual_vs_predicted()
    print("3. Plotting training history...")
    plot_training_history()
    print("4. Plotting rigorous test set performance...")
    plot_test_set_performance()
    print("5. Plotting residual analysis...")
    plot_residual_analysis()
    print("All visualizations completed!")
