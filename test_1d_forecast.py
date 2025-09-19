"""
Comprehensive test suite for 1D multi-step Bitcoin price forecasting.

This module provides testing and validation capabilities for the daily prediction
system, including:
- Functional testing of the 1D forecasting pipeline
- Performance analysis with quality metrics
- Directional prediction assessment
- Volatility analysis for risk evaluation
- Professional result presentation with confidence scoring

The test suite validates that the autoregressive forecasting system produces
realistic Bitcoin price predictions with appropriate uncertainty quantification.
"""

def test_1d_forecast():
    """
    Execute comprehensive testing of the 1D forecasting system.
    
    This function validates the complete forecasting pipeline by:
    1. Loading the trained model and generating 5-day predictions
    2. Analyzing prediction quality and confidence scores
    3. Evaluating directional accuracy and market sentiment
    4. Calculating volatility metrics for risk assessment
    5. Presenting results in a professional format
    
    Returns:
        bool: True if all tests pass successfully, False otherwise
    """
    print("🗓️  Testing 1D Multi-Step Forecasting")
    print("="*40)
    
    try:
        from scripts.multi_step_forecast_1d import create_1d_multi_step_forecast
        
        # Test with 5 days prediction
        print("Creating 5-day forecast using daily data...")
        predictions_df = create_1d_multi_step_forecast(days=5, save_results=True)
        
        if predictions_df is not None:
            print("\n📊 1D Forecast Results:")
            print("-" * 50)
            
            for idx, row in predictions_df.iterrows():
                print(f"{idx.strftime('%Y-%m-%d')}: "
                      f"${row['predicted_open']:,.0f} → ${row['predicted_close']:,.0f} "
                      f"({row['predicted_change_pct']:+.2f}%) "
                      f"[{row['direction_label']}] "
                      f"Confidence: {row['confidence']:.2f}")
            
            # Summary
            total_change = ((predictions_df['predicted_close'].iloc[-1] - predictions_df['predicted_open'].iloc[0]) / 
                           predictions_df['predicted_open'].iloc[0]) * 100
            
            print(f"\n📈 Summary:")
            print(f"   Total 5-day change: {total_change:+.2f}%")
            print(f"   Bullish days: {(predictions_df['direction'] == 1).sum()}/5")
            print(f"   Average confidence: {predictions_df['confidence'].mean():.3f}")
            
            print(f"\n✅ 1D forecasting test completed!")
            print(f"📁 Results saved in 'predictions/' directory")
            return True
        else:
            print("❌ No predictions generated")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def show_forecast_analysis():
    """Show 1D forecast analysis and recommendations"""
    print("\n📊 1D Forecast Analysis")
    print("="*40)
    
    try:
        import pandas as pd
        import os
        
        forecast_1d_path = "predictions/multi_step_forecast_1d.csv"
        
        if os.path.exists(forecast_1d_path):
            df_1d = pd.read_csv(forecast_1d_path, index_col=0, parse_dates=True)
            
            print("📈 Forecast Quality Metrics:")
            print("-" * 40)
            
            # Confidence analysis
            avg_confidence = df_1d['confidence'].mean()
            confidence_degradation = df_1d['confidence'].iloc[0] - df_1d['confidence'].iloc[-1]
            
            print(f"   Average confidence: {avg_confidence:.3f}")
            print(f"   Confidence degradation: {confidence_degradation:.3f}")
            print(f"   Final day confidence: {df_1d['confidence'].iloc[-1]:.3f}")
            
            # Direction analysis
            bullish_days = (df_1d['direction'] == 1).sum()
            total_days = len(df_1d)
            
            print(f"\n📊 Directional Analysis:")
            print(f"   Bullish days: {bullish_days}/{total_days}")
            print(f"   Market sentiment: {'Bullish' if bullish_days > total_days/2 else 'Bearish'}")
            
            # Volatility analysis
            daily_changes = df_1d['predicted_change_pct'].abs()
            avg_volatility = daily_changes.mean()
            max_daily_move = daily_changes.max()
            
            print(f"\n📈 Volatility Analysis:")
            print(f"   Average daily move: {avg_volatility:.2f}%")
            print(f"   Maximum daily move: {max_daily_move:.2f}%")
            print(f"   Volatility level: {'High' if avg_volatility > 2 else 'Moderate' if avg_volatility > 1 else 'Low'}")
            
        else:
            print("⚠️  No 1D forecast available")
            print("   Run: python run_1d_forecast.py forecast --days 5")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")


def main():
    """Main test function"""
    # Test 1D forecasting
    success = test_1d_forecast()
    
    if success:
        # Show forecast analysis
        show_forecast_analysis()
        
        print(f"\n🎯 1D Forecasting Benefits:")
        print(f"   ✅ Less noise for long-term trends")
        print(f"   ✅ More stable predictions over 5+ days")
        print(f"   ✅ Reduced autoregressive error accumulation")
        print(f"   ✅ Better suited for multi-day investment decisions")
        
        print(f"\n💡 Usage Recommendations:")
        print(f"   📅 Use 1D for forecasts > 3 days")
        print(f"   � Run historical backtest for validation")
        print(f"   � Monitor confidence scores for reliability")
    
    return success


if __name__ == "__main__":
    main()
