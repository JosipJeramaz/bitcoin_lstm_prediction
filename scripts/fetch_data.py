import os
import pandas as pd
from datetime import datetime, timedelta
from binance.um_futures import UMFutures

def fetch_binance_data(symbol, interval, limit, filename):
    client = UMFutures()
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return
    klines = client.klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume", 
        "close_time", "quote_asset_volume", "number_of_trades", 
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df.to_csv(filename)
    print(f"Saved {filename}")

def fetch_all_data():
    os.makedirs("data", exist_ok=True)
    fetch_binance_data("BTCUSDT", "1d", 120, "data/raw_1d.csv")
    fetch_binance_data("BTCUSDT", "4h", 120, "data/raw_4h.csv")
