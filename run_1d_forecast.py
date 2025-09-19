"""
Main runner script for Bitcoin 1D price forecasting.
Provides a simple interface to run daily forecasting tasks.
"""

import os
import sys
import argparse
from datetime import datetime


def print_banner():
    """Print a nice banner"""
    print("="*60)
    print("    Bitcoin Daily Price Forecasting System")
    print("="*60)
    print(f"    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)


def check_prerequisites():
    """Check if all required files exist"""
    required_files = [
        "models/dual_lstm_final.pth",
        "models/scaler.save", 
        "data/final_dataset.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease train your model first using:")
        print("   python -m scripts.train")
        return False
    
    print("✅ All prerequisites found")
    return True


def run_forecast(days=5):
    """Run 1D multi-step forecast"""
    days = int(days)  # Ensure days is an integer
    print(f"\n📊 Running {days}-day forecast using daily data...")
    try:
        from scripts.multi_step_forecast_1d import create_1d_multi_step_forecast
        predictions_df = create_1d_multi_step_forecast(days=days)
        print(f"✅ 1D Forecast completed! {len(predictions_df)} daily predictions generated.")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ 1D Forecast error: {e}")
        return False


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Bitcoin 1D Price Forecasting")
    parser.add_argument("--task", choices=["forecast", "all"], 
                       default="forecast", help="Task to run")
    parser.add_argument("--days", type=float, default=5.0, 
                       help="Number of days to forecast")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    success = True
    
    if args.task == "forecast" or args.task == "all":
        success &= run_forecast(days=args.days)
    
    if success:
        print(f"\n🎉 All tasks completed successfully!")
        print(f"📁 Check the 'predictions/' directory for results")
    else:
        print(f"\n⚠️  Some tasks failed. Check the error messages above.")


if __name__ == "__main__":
    main()
