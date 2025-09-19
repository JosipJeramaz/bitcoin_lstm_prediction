"""
Simple launcher for historical backtesting.
Tests your Bitcoin LSTM model on historical data without changing existing code.
"""

import sys
import os

def main():
    print("🕐 Historical Backtesting for Bitcoin LSTM")
    print("=" * 50)
    
    try:
        from scripts.historical_backtest import run_historical_backtest
        
        # Run backtest with default settings
        # 8 test periods, 5 days forecast each
        summary = run_historical_backtest(
            num_tests=8,
            forecast_days=5,
            save_results=True
        )
        
        if summary:
            print(f"\n🎉 Backtesting Results:")
            print(f"   📊 Tested on {summary['num_tests']} historical periods")
            print(f"   💰 Average Price Error (RMSE): ${summary['avg_rmse']:,.2f}")
            print(f"   🎯 Average Direction Accuracy: {summary['avg_directional_accuracy']:.1f}%")
            
            # Simple interpretation
            if summary['avg_directional_accuracy'] >= 65:
                print("   ✅ Excellent directional predictions!")
            elif summary['avg_directional_accuracy'] >= 55:
                print("   👍 Good directional predictions")
            elif summary['avg_directional_accuracy'] >= 50:
                print("   ⚖️  Moderate directional predictions")
            else:
                print("   ⚠️  Poor directional predictions - model needs improvement")
            
            print(f"\n📁 Results saved in 'predictions/' directory")
            print(f"   - historical_backtest_results.csv")
            print(f"   - historical_backtest.png")
            
        else:
            print("❌ No results generated - check for errors above")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct directory")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
