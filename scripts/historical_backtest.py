"""
Historical backtesting script for 1D Bitcoin LSTM predictions.
Tests the model on historical daily data to evaluate real performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import joblib
import os
from datetime import datetime, timedelta
from scripts.multi_step_forecast_1d import predict_recursive_1d
from scripts.config import get_model_config, get_training_config, get_data_config
from scripts.model import DualLSTMPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_historical_data(data_path="data/final_dataset.csv"):
    """Load and prepare historical data for backtesting"""
    print("Loading historical data...")
    
    df = pd.read_csv(data_path, index_col="timestamp", parse_dates=True)
    
    # Separate timeframes
    df_1d = df[df['timeframe'] == '1d'].drop(columns=['timeframe']).sort_index()
    df_4h = df[df['timeframe'] == '4h'].drop(columns=['timeframe']).sort_index()
    
    print(f"Historical data loaded:")
    print(f"  1d data: {len(df_1d)} points from {df_1d.index[0]} to {df_1d.index[-1]}")
    print(f"  4h data: {len(df_4h)} points from {df_4h.index[0]} to {df_4h.index[-1]}")
    
    return df_1d, df_4h


def prepare_scaled_historical_data(df_1d, df_4h, scaler):
    """Apply same scaling as used in training"""
    data_config = get_data_config()
    log_epsilon = data_config['log_epsilon']
    
    # Log transform
    df_1d_log = np.log(df_1d + log_epsilon)
    df_4h_log = np.log(df_4h + log_epsilon)
    
    # Scale using the trained scaler
    df_1d_scaled = pd.DataFrame(
        scaler.transform(df_1d_log), 
        index=df_1d.index, 
        columns=df_1d.columns
    )
    df_4h_scaled = pd.DataFrame(
        scaler.transform(df_4h_log),
        index=df_4h.index,
        columns=df_4h.columns
    )
    
    return df_1d_scaled, df_4h_scaled


def select_test_periods(df_1d, num_tests=8, forecast_days=5):
    """
    Select historical periods for 1D backtesting that have enough future data
    
    Parameters:
        df_1d: Daily historical data
        num_tests: Number of test periods to select
        forecast_days: Days to forecast ahead
        
    Returns:
        list: List of test start indices
    """
    training_config = get_training_config()
    sequence_length = training_config['sequence_length']
    
    # Need sequence_length for input + forecast_days for output
    min_data_needed = sequence_length + forecast_days
    
    # Find valid start positions (need enough data before and after)
    valid_starts = []
    for i in range(sequence_length, len(df_1d) - forecast_days):
        valid_starts.append(i)
    
    if len(valid_starts) < num_tests:
        print(f"Warning: Only {len(valid_starts)} valid test periods available, requested {num_tests}")
        num_tests = len(valid_starts)
    
    # Select evenly distributed test periods
    if num_tests > 1:
        step = len(valid_starts) // num_tests
        selected_indices = [valid_starts[i * step] for i in range(num_tests)]
    else:
        selected_indices = [valid_starts[len(valid_starts) // 2]]  # Middle period
    
    print(f"Selected {len(selected_indices)} test periods:")
    for i, idx in enumerate(selected_indices):
        test_date = df_1d.index[idx]
        end_date = df_1d.index[idx + forecast_days - 1]
        print(f"  Test {i+1}: {test_date.date()} → {end_date.date()}")
    
    return selected_indices


def run_single_backtest(model, scaler, df_1d_scaled, df_1d_actual, 
                       start_idx, forecast_days, device):
    """
    Run a single 1D backtest starting from start_idx
    
    Returns:
        dict: Backtest results with predictions and actual values
    """
    training_config = get_training_config()
    sequence_length = training_config['sequence_length']
    
    # Ensure we have enough data
    if start_idx < sequence_length or start_idx + forecast_days >= len(df_1d_actual):
        print(f"Warning: Not enough data for backtest at index {start_idx}")
        return None
    
    # Get input sequences (use the data up to start_idx)
    input_1d = df_1d_scaled.iloc[start_idx-sequence_length:start_idx].values
    
    # For 1D-only prediction, use the same 1D data for both inputs
    # This allows us to use the dual model architecture with only daily data
    input_4h = input_1d  # Use the same 1D data as "4H" input
    
    # Validate input shapes
    if len(input_1d) != sequence_length:
        print(f"Warning: Invalid sequence length at index {start_idx}")
        return None
    
    # Convert to tensors (remove extra batch dimension)
    input_1d_tensor = torch.tensor(input_1d, dtype=torch.float32)  # Shape: (sequence_length, features)
    input_4h_tensor = torch.tensor(input_4h, dtype=torch.float32)  # Shape: (sequence_length, features)
    
    # Get actual future values (the days we want to predict)
    actual_future = df_1d_actual.iloc[start_idx:start_idx + forecast_days]
    
    if len(actual_future) < forecast_days:
        print(f"Warning: Not enough future data at index {start_idx}")
        return None
    
    # Make predictions using the 1D prediction function
    try:
        predictions, raw_predictions, confidence_scores = predict_recursive_1d(
            model=model,
            initial_1d_data=input_1d_tensor,
            initial_4h_data=input_4h_tensor,
            scaler=scaler,
            steps=forecast_days,
            device=device
        )
        
        return {
            'start_timestamp': df_1d_actual.index[start_idx],
            'predictions': predictions,
            'actual': actual_future[['open', 'close']].values,
            'timestamps': actual_future.index,
            'confidence': confidence_scores
        }
        
    except Exception as e:
        print(f"Error in backtest at {df_1d_actual.index[start_idx]}: {e}")
        return None


def calculate_backtest_metrics(results):
    """Calculate comprehensive metrics from backtest results"""
    all_metrics = []
    
    for i, result in enumerate(results):
        if result is None:
            continue
            
        pred = result['predictions']
        actual = result['actual']
        
        # Price metrics
        rmse_open = np.sqrt(mean_squared_error(actual[:, 0], pred[:, 0]))
        rmse_close = np.sqrt(mean_squared_error(actual[:, 1], pred[:, 1]))
        
        mae_open = mean_absolute_error(actual[:, 0], pred[:, 0])
        mae_close = mean_absolute_error(actual[:, 1], pred[:, 1])
        
        # Directional accuracy
        actual_direction = (actual[:, 1] > actual[:, 0]).astype(int)
        pred_direction = (pred[:, 1] > pred[:, 0]).astype(int)
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # MAPE
        mape_open = np.mean(np.abs((actual[:, 0] - pred[:, 0]) / actual[:, 0])) * 100
        mape_close = np.mean(np.abs((actual[:, 1] - pred[:, 1]) / actual[:, 1])) * 100
        
        # Price change accuracy
        actual_change = ((actual[:, 1] - actual[:, 0]) / actual[:, 0]) * 100
        pred_change = ((pred[:, 1] - pred[:, 0]) / pred[:, 0]) * 100
        change_error = np.mean(np.abs(actual_change - pred_change))
        
        metrics = {
            'test_period': i + 1,
            'start_date': result['start_timestamp'],
            'rmse_open': rmse_open,
            'rmse_close': rmse_close,
            'mae_open': mae_open,
            'mae_close': mae_close,
            'mape_open': mape_open,
            'mape_close': mape_close,
            'directional_accuracy': directional_accuracy,
            'change_error': change_error,
            'avg_confidence': np.mean(result['confidence'])
        }
        
        all_metrics.append(metrics)
    
    return all_metrics


def create_backtest_visualization(results, save_path="predictions/historical_backtest.png"):
    """Create comprehensive visualization of backtest results"""
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("No valid results to visualize")
        return
    
    num_tests = len(valid_results)
    if num_tests == 0:
        print("No valid test results to plot")
        return
    
    # Create subplots - 2 columns, rows as needed
    rows = max(1, (num_tests + 1) // 2)
    fig, axes = plt.subplots(rows, 2, figsize=(20, 6 * rows))
    
    # Handle single subplot case
    if num_tests == 1:
        axes = [axes] if rows == 1 else [axes[0]]
    elif rows == 1:
        axes = [axes[0], axes[1]] if num_tests > 1 else [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Historical Backtest Results - Bitcoin LSTM Predictions', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(valid_results):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Plot actual vs predicted prices
        timestamps = result['timestamps']
        actual = result['actual']
        pred = result['predictions']
        
        ax.plot(timestamps, actual[:, 1], 'b-', label='Actual Close', linewidth=2)
        ax.plot(timestamps, pred[:, 1], 'r--', label='Predicted Close', linewidth=2, alpha=0.8)
        ax.plot(timestamps, actual[:, 0], 'b:', label='Actual Open', alpha=0.7)
        ax.plot(timestamps, pred[:, 0], 'r:', label='Predicted Open', alpha=0.7)
        
        # Add confidence band
        confidence = result['confidence']
        upper_bound = pred[:, 1] * (1 + (1 - confidence) * 0.05)
        lower_bound = pred[:, 1] * (1 - (1 - confidence) * 0.05)
        ax.fill_between(timestamps, lower_bound, upper_bound, alpha=0.2, color='red', label='Confidence Band')
        
        ax.set_title(f'Test {i+1}: {result["start_timestamp"].strftime("%Y-%m-%d")}')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    # Hide empty subplots
    for i in range(len(valid_results), len(axes)):
        if i < len(axes):
            axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs("predictions", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Backtest visualization saved to: {save_path}")
    plt.show()


def print_backtest_summary(metrics):
    """Print comprehensive summary of backtest results"""
    
    if not metrics:
        print("No valid backtest results to summarize")
        return
    
    print("\n" + "="*80)
    print("HISTORICAL BACKTEST SUMMARY")
    print("="*80)
    
    # Overall statistics
    avg_rmse_close = np.mean([m['rmse_close'] for m in metrics])
    avg_mae_close = np.mean([m['mae_close'] for m in metrics])
    avg_mape_close = np.mean([m['mape_close'] for m in metrics])
    avg_dir_acc = np.mean([m['directional_accuracy'] for m in metrics])
    avg_change_error = np.mean([m['change_error'] for m in metrics])
    
    print(f"\nOVERALL PERFORMANCE ({len(metrics)} test periods):")
    print(f"  Average RMSE (Close): ${avg_rmse_close:,.2f}")
    print(f"  Average MAE (Close):  ${avg_mae_close:,.2f}")
    print(f"  Average MAPE (Close): {avg_mape_close:.2f}%")
    print(f"  Average Directional Accuracy: {avg_dir_acc:.1f}%")
    print(f"  Average Change Error: {avg_change_error:.2f}%")
    
    # Best and worst performance
    best_rmse = min(metrics, key=lambda x: x['rmse_close'])
    worst_rmse = max(metrics, key=lambda x: x['rmse_close'])
    best_dir_acc = max(metrics, key=lambda x: x['directional_accuracy'])
    worst_dir_acc = min(metrics, key=lambda x: x['directional_accuracy'])
    
    print(f"\nBEST PERFORMANCE:")
    print(f"  Lowest RMSE: ${best_rmse['rmse_close']:.2f} on {best_rmse['start_date'].strftime('%Y-%m-%d')}")
    print(f"  Highest Dir. Acc: {best_dir_acc['directional_accuracy']:.1f}% on {best_dir_acc['start_date'].strftime('%Y-%m-%d')}")
    
    print(f"\nWORST PERFORMANCE:")
    print(f"  Highest RMSE: ${worst_rmse['rmse_close']:.2f} on {worst_rmse['start_date'].strftime('%Y-%m-%d')}")
    print(f"  Lowest Dir. Acc: {worst_dir_acc['directional_accuracy']:.1f}% on {worst_dir_acc['start_date'].strftime('%Y-%m-%d')}")
    
    # Performance consistency
    rmse_std = np.std([m['rmse_close'] for m in metrics])
    dir_acc_std = np.std([m['directional_accuracy'] for m in metrics])
    
    print(f"\nCONSISTENCY:")
    print(f"  RMSE Std Deviation: ${rmse_std:.2f}")
    print(f"  Dir. Acc Std Deviation: {dir_acc_std:.1f}%")
    
    # Market condition analysis
    bull_periods = [m for m in metrics if m['directional_accuracy'] > 60]
    bear_periods = [m for m in metrics if m['directional_accuracy'] < 40]
    
    print(f"\nMARKET CONDITION ANALYSIS:")
    print(f"  Strong performance periods (>60% dir acc): {len(bull_periods)}")
    print(f"  Weak performance periods (<40% dir acc): {len(bear_periods)}")
    print(f"  Neutral performance periods: {len(metrics) - len(bull_periods) - len(bear_periods)}")
    
    # Detailed results
    print(f"\nDETAILED RESULTS BY TEST PERIOD:")
    print("-" * 120)
    print(f"{'Period':<8} {'Date':<12} {'RMSE':<10} {'MAE':<10} {'MAPE':<8} {'Dir Acc':<8} {'Change Err':<10} {'Confidence':<10}")
    print("-" * 120)
    
    for m in metrics:
        print(f"{m['test_period']:<8} "
              f"{m['start_date'].strftime('%Y-%m-%d'):<12} "
              f"${m['rmse_close']:<9,.0f} "
              f"${m['mae_close']:<9,.0f} "
              f"{m['mape_close']:<7.2f}% "
              f"{m['directional_accuracy']:<7.1f}% "
              f"{m['change_error']:<9.2f}% "
              f"{m['avg_confidence']:<10.3f}")


def load_model_and_scaler():
    """Load the trained model and scaler"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load("models/scaler.save")
    input_size = scaler.n_features_in_
    
    model = DualLSTMPredictor(input_size=input_size)
    model.load_state_dict(torch.load("models/dual_lstm_final.pth", map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, scaler, device


def run_historical_backtest(num_tests=8, forecast_days=5, save_results=True):
    """
    Main function to run historical backtesting
    
    Parameters:
        num_tests: Number of historical periods to test
        forecast_days: Days to forecast ahead for each test
        save_results: Whether to save results to files
    """
    
    print("="*80)
    print("HISTORICAL BACKTESTING - BITCOIN 1D LSTM PREDICTIONS")
    print("="*80)
    print(f"Configuration:")
    print(f"  Number of test periods: {num_tests}")
    print(f"  Forecast horizon: {forecast_days} days")
    print("="*80)
    
    try:
        # Load model and data
        print("\n1. Loading model and historical data...")
        model, scaler, device = load_model_and_scaler()
        df_1d, df_4h = load_historical_data()
        df_1d_scaled, df_4h_scaled = prepare_scaled_historical_data(df_1d, df_4h, scaler)
        
        # Select test periods
        print("\n2. Selecting test periods...")
        test_indices = select_test_periods(df_1d, num_tests, forecast_days)
        
        # Run backtests
        print(f"\n3. Running {len(test_indices)} backtests...")
        results = []
        
        for i, start_idx in enumerate(test_indices):
            print(f"   Running backtest {i+1}/{len(test_indices)} "
                  f"(starting {df_1d.index[start_idx].strftime('%Y-%m-%d')})")
            
            result = run_single_backtest(
                model, scaler, df_1d_scaled, df_1d,
                start_idx, forecast_days, device
            )
            results.append(result)
        
        # Calculate metrics
        print("\n4. Calculating performance metrics...")
        metrics = calculate_backtest_metrics(results)
        
        # Create visualization
        print("\n5. Creating visualizations...")
        create_backtest_visualization(results)
        
        # Print summary
        print_backtest_summary(metrics)
        
        # Save results
        if save_results and metrics:
            os.makedirs("predictions", exist_ok=True)
            
            # Save detailed results
            results_df = pd.DataFrame(metrics)
            results_df.to_csv("predictions/historical_backtest_results.csv", index=False)
            
            # Save raw results for further analysis
            backtest_data = {
                'results': results,
                'metrics': metrics,
                'config': {
                    'num_tests': num_tests,
                    'forecast_days': forecast_days,
                    'test_date': datetime.now().isoformat()
                }
            }
            joblib.dump(backtest_data, "predictions/historical_backtest_raw.save")
            
            print(f"\nResults saved:")
            print(f"  - predictions/historical_backtest_results.csv")
            print(f"  - predictions/historical_backtest_raw.save")
            print(f"  - predictions/historical_backtest.png")
        
        print(f"\n✅ Historical backtesting completed successfully!")
        
        # Return summary for further use
        if metrics:
            return {
                'avg_rmse': np.mean([m['rmse_close'] for m in metrics]),
                'avg_directional_accuracy': np.mean([m['directional_accuracy'] for m in metrics]),
                'num_tests': len(metrics),
                'best_rmse': min(m['rmse_close'] for m in metrics),
                'worst_rmse': max(m['rmse_close'] for m in metrics)
            }
        else:
            return None
            
    except Exception as e:
        print(f"\n❌ Error during historical backtesting: {str(e)}")
        raise


def main():
    """Main function to run historical backtesting"""
    
    # Configuration
    num_test_periods = 8  # Test on 8 different historical periods
    forecast_horizon = 5  # 5 days ahead (same as your current predictions)
    
    try:
        summary = run_historical_backtest(
            num_tests=num_test_periods,
            forecast_days=forecast_horizon,
            save_results=True
        )
        
        if summary:
            print(f"\n🎯 QUICK SUMMARY:")
            print(f"   Average RMSE: ${summary['avg_rmse']:,.2f}")
            print(f"   Average Directional Accuracy: {summary['avg_directional_accuracy']:.1f}%")
            print(f"   Best RMSE: ${summary['best_rmse']:,.2f}")
            print(f"   Worst RMSE: ${summary['worst_rmse']:,.2f}")
            
            # Interpretation
            if summary['avg_directional_accuracy'] > 60:
                print("   🟢 Good directional prediction capability!")
            elif summary['avg_directional_accuracy'] > 50:
                print("   🟡 Moderate directional prediction capability")
            else:
                print("   🔴 Poor directional prediction - needs improvement")
                
    except Exception as e:
        print(f"❌ Backtesting failed: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    main()
