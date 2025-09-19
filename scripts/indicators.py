"""
Adds technical indicators (moving averages, Bollinger Bands) to raw OHLCV data.
- Processes multiple timeframes (1d, 4h).
- Handles missing files and validates output.
- Includes verification utility for processed data.
"""

import pandas as pd
import ta
import os

def add_indicators():
    """
    Add technical indicators (MA and Bollinger Bands) to raw OHLCV data
    """
    print("Starting technical indicators calculation...")
    print("=" * 50)
    
    for tf in ["1d", "4h"]:
        raw_file = f"data/raw_{tf}.csv"
        processed_file = f"data/processed_{tf}.csv"
        
        # Check if raw file exists
        if not os.path.exists(raw_file):
            print(f"Error: {raw_file} not found. Run fetch_data.py first.")
            continue
            
        # Skip if processed file already exists
        if os.path.exists(processed_file):
            print(f"{processed_file} already exists. Skipping calculation.")
            continue
        
        try:
            print(f"Processing {tf} timeframe...")
            
            # Load raw data
            df = pd.read_csv(raw_file, index_col="timestamp", parse_dates=True)
            
            if df.empty:
                print(f"Warning: Empty dataset for {tf} timeframe")
                continue
            
            print(f"  Loaded {len(df)} records from {df.index[0]} to {df.index[-1]}")
            
            # Add Moving Averages
            print("  Calculating Moving Averages...")
            for ma_period in [5, 10, 30, 60]:
                df[f"ma{ma_period}"] = df["close"].rolling(window=ma_period, min_periods=ma_period).mean()
                print(f"    MA{ma_period} calculated")
            
            # Add Bollinger Bands
            print("  Calculating Bollinger Bands...")
            try:
                bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
                df["bb_bbm"] = bb.bollinger_mavg()  # Middle band (SMA)
                df["bb_bbh"] = bb.bollinger_hband() # Upper band
                df["bb_bbl"] = bb.bollinger_lband() # Lower band
                print("    Bollinger Bands calculated")
            except Exception as e:
                print(f"    Error calculating Bollinger Bands: {e}")
                # Add fallback manual calculation
                df["bb_bbm"] = df["close"].rolling(window=20).mean()
                bb_std = df["close"].rolling(window=20).std()
                df["bb_bbh"] = df["bb_bbm"] + (bb_std * 2)
                df["bb_bbl"] = df["bb_bbm"] - (bb_std * 2)
                print("    Bollinger Bands calculated (manual fallback)")
            
            # Check data before dropping NaN
            initial_count = len(df)
            df.dropna(inplace=True)
            final_count = len(df)
            dropped_count = initial_count - final_count
            
            if dropped_count > 0:
                print(f"  Dropped {dropped_count} rows with NaN values")
            
            if df.empty:
                print(f"Warning: No data left after dropping NaN for {tf} timeframe")
                continue
            
            # Validate indicators
            required_columns = ["open", "high", "low", "close", "volume", 
                              "ma5", "ma10", "ma30", "ma60", 
                              "bb_bbm", "bb_bbh", "bb_bbl"]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns: {missing_columns}")
            
            # Save processed data
            df.to_csv(processed_file)
            print(f"  Successfully saved {len(df)} records to {processed_file}")
            print(f"  Final date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Columns: {list(df.columns)}")
            
        except Exception as e:
            print(f"Error processing {tf} timeframe: {e}")
        
        print("-" * 30)
    
    print("=" * 50)
    print("Technical indicators calculation completed!")

def verify_indicators():
    """
    Verify that indicators were calculated correctly
    """
    print("\nVerifying indicators...")
    print("=" * 30)
    
    for tf in ["1d", "4h"]:
        processed_file = f"data/processed_{tf}.csv"
        
        if not os.path.exists(processed_file):
            print(f"❌ {processed_file} not found")
            continue
            
        try:
            df = pd.read_csv(processed_file, index_col="timestamp", parse_dates=True)
            
            print(f"\n{tf.upper()} Timeframe:")
            print(f"  Records: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            
            # Check for NaN values
            nan_counts = df.isnull().sum()
            if nan_counts.sum() > 0:
                print(f"  ⚠️  NaN values found:")
                for col, count in nan_counts[nan_counts > 0].items():
                    print(f"    {col}: {count}")
            else:
                print(f"  ✅ No NaN values")
            
            # Show sample of indicators
            print(f"  Sample indicators:")
            sample_cols = ["close", "ma5", "ma30", "bb_bbm", "bb_bbh", "bb_bbl"]
            available_cols = [col for col in sample_cols if col in df.columns]
            if available_cols:
                print(df[available_cols].tail(3).round(2))
                
        except Exception as e:
            print(f"❌ Error verifying {tf}: {e}")
    
    print("=" * 30)

if __name__ == "__main__":
    add_indicators()
    verify_indicators()