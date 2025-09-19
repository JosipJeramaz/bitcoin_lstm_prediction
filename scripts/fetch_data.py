"""
Fetches raw OHLCV data from Binance and saves it as CSV files for further processing in the pipeline.
- Handles both small and large data requests.
- Ensures data directory exists.
- Validates and cleans data before saving.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from binance.um_futures import UMFutures
import time

def fetch_binance_data_large(symbol, interval, total_limit, filename):
    """
    Fetch large amounts of historical data using multiple API calls
    """
    client = UMFutures()
    
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return
    
    try:
        all_data = []
        max_limit_per_call = 1000  # Binance API limit
        calls_needed = (total_limit + max_limit_per_call - 1) // max_limit_per_call
        
        print(f"Fetching {total_limit} {interval} candlesticks for {symbol}...")
        print(f"This requires {calls_needed} API calls...")
        
        for i in range(calls_needed):
            current_limit = min(max_limit_per_call, total_limit - i * max_limit_per_call)
            
            if current_limit <= 0:
                break
                
            print(f"API call {i+1}/{calls_needed}: fetching {current_limit} candlesticks...")
            
            # Calculate endTime for this batch (working backwards from most recent)
            if i == 0:
                # First call - get most recent data
                klines = client.klines(symbol=symbol, interval=interval, limit=current_limit)
            else:
                # Subsequent calls - get older data
                last_timestamp = all_data[0][0]  # First timestamp from previous batch
                klines = client.klines(
                    symbol=symbol, 
                    interval=interval, 
                    limit=current_limit,
                    endTime=last_timestamp - 1  # Get data before previous batch
                )
            
            if not klines:
                print(f"No more data available for {symbol} {interval}")
                break
                
            # Insert at beginning to maintain chronological order
            all_data = klines + all_data
            
            # Be nice to the API - small delay between calls
            if i < calls_needed - 1:
                time.sleep(0.1)
        
        if not all_data:
            print(f"No data received for {symbol} {interval}")
            return
        
        df = pd.DataFrame(all_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", 
            "close_time", "quote_asset_volume", "number_of_trades", 
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        
        # Convert timestamp and set as index
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        
        # Keep only OHLCV data and convert to float
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        
        # Remove duplicates that might occur from overlapping API calls
        df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        
        # Validate data
        if df.empty:
            print(f"Empty dataframe for {symbol} {interval}")
            return
            
        if df.isnull().any().any():
            print(f"Warning: NULL values found in {symbol} {interval} data")
            df.dropna(inplace=True)
        
        # Save to CSV
        df.to_csv(filename)
        print(f"Successfully saved {len(df)} records to {filename}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
    except Exception as e:
        print(f"Error fetching data for {symbol} {interval}: {e}")

def fetch_binance_data(symbol, interval, limit, filename):
    """
    Fetch historical kline data from Binance (for smaller datasets)
    """
    if limit <= 1000:
        # Use simple single API call for small datasets
        client = UMFutures()
        
        if os.path.exists(filename):
            print(f"{filename} already exists. Skipping download.")
            return
        
        try:
            print(f"Fetching {limit} {interval} candlesticks for {symbol}...")
            klines = client.klines(symbol=symbol, interval=interval, limit=limit)
            
            if not klines:
                print(f"No data received for {symbol} {interval}")
                return
            
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume", 
                "close_time", "quote_asset_volume", "number_of_trades", 
                "taker_buy_base", "taker_buy_quote", "ignore"
            ])
            
            # Convert timestamp and set as index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            df.set_index("timestamp", inplace=True)
            
            # Keep only OHLCV data and convert to float
            df = df[["open", "high", "low", "close", "volume"]].astype(float)
            
            # Save to CSV
            df.to_csv(filename)
            print(f"Successfully saved {len(df)} records to {filename}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
        except Exception as e:
            print(f"Error fetching data for {symbol} {interval}: {e}")
    else:
        # Use multiple API calls for large datasets
        fetch_binance_data_large(symbol, interval, limit, filename)

def fetch_all_data():
    """
    Fetch all required datasets for the project
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    print("Starting data fetch process...")
    print("=" * 50)
    
    # Fetch 1D data (1000 days ≈ 2.7 years)
    fetch_binance_data("BTCUSDT", "1d", 1000, "data/raw_1d.csv")
    
    print("-" * 50)
    
    # Fetch 4H data (6000 × 4h = 1000 days)
    fetch_binance_data("BTCUSDT", "4h", 6000, "data/raw_4h.csv")
    
    print("=" * 50)
    print("Data fetch completed!")

if __name__ == "__main__":
    fetch_all_data()