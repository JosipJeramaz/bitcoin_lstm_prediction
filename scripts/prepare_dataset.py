"""
Prepares the final dataset by combining processed data from different timeframes.
- Loads processed 1d and 4h data.
- Adds timeframe labels for dual LSTM architecture.
- Combines datasets while preserving temporal information.
- Saves the final dataset for model training.
"""

import pandas as pd
import os

def prepare_dataset():
    """
    Prepare the final dataset by combining processed timeframes
    """
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    print("Preparing dataset for Dual LSTM...")
    
    # Load processed data
    df_1d = pd.read_csv("data/processed_1d.csv", index_col="timestamp", parse_dates=True)
    df_4h = pd.read_csv("data/processed_4h.csv", index_col="timestamp", parse_dates=True)

    print(f"1d data shape: {df_1d.shape}")
    print(f"4h data shape: {df_4h.shape}")

    # Add timeframe column 
    df_1d["timeframe"] = "1d"
    df_4h["timeframe"] = "4h"

    # Ensure both datasets have the same columns (
    common_cols = list(set(df_1d.columns) & set(df_4h.columns))
    common_cols.remove("timeframe")  # Remove timeframe from common cols
    
    # Keep only common columns plus timeframe
    df_1d = df_1d[common_cols + ["timeframe"]]
    df_4h = df_4h[common_cols + ["timeframe"]]

    print(f"Common columns: {len(common_cols)}")
    print(f"Columns: {common_cols[:5]}...")  # Show first 5 columns

    # Concatenate vertically, keeping all rows
    df_all = pd.concat([df_1d, df_4h], axis=0)
    df_all.sort_index(inplace=True)

    # Remove duplicates that might exist
    df_all = df_all[~df_all.index.duplicated(keep='first')]
    
    # Check for overlapping timestamps between timeframes
    timestamps_1d = set(df_1d.index)
    timestamps_4h = set(df_4h.index)
    common_timestamps = timestamps_1d & timestamps_4h
    
    print(f"Total 1d timestamps: {len(timestamps_1d)}")
    print(f"Total 4h timestamps: {len(timestamps_4h)}")
    print(f"Common timestamps: {len(common_timestamps)}")
    print(f"This gives us {len(common_timestamps)} aligned data points for dual LSTM")

    # Save final dataset with timestamp as a column
    df_all.to_csv("data/final_dataset.csv", index=True, index_label="timestamp")
    print(f"Final dataset saved: {df_all.shape[0]} rows, {df_all.shape[1]} columns.")
    print("Timeframe column preserved for Dual LSTM architecture.")
    
    # Save alignment info for training
    alignment_info = {
        'common_timestamps': len(common_timestamps),
        'total_1d': len(timestamps_1d),
        'total_4h': len(timestamps_4h),
        'alignment_ratio': len(common_timestamps) / min(len(timestamps_1d), len(timestamps_4h))
    }
    
    print(f"Alignment ratio: {alignment_info['alignment_ratio']:.2%}")
    if alignment_info['alignment_ratio'] < 0.1:
        print("WARNING: Low alignment ratio! Consider checking timestamp alignment.")
    
    return alignment_info

if __name__ == "__main__":
    prepare_dataset()
