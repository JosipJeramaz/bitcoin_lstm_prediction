import pandas as pd
import os

"""
Prepares the final dataset for LSTM modeling by stacking all 1d and 4h data, keeping all rows and marking their timeframe.
- No data is lost; duplicate timestamps are allowed.
- Adds a 'timeframe' column to distinguish between 1d and 4h rows.
- Outputs a single CSV for model training.
"""

def prepare_dataset():
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Load processed data
    df_1d = pd.read_csv("data/processed_1d.csv", index_col="timestamp", parse_dates=True)
    df_4h = pd.read_csv("data/processed_4h.csv", index_col="timestamp", parse_dates=True)

    # Add timeframe column
    df_1d["timeframe"] = "1d"
    df_4h["timeframe"] = "4h"

    # Concatenate vertically, keeping all rows
    df_all = pd.concat([df_1d, df_4h], axis=0)
    df_all.sort_index(inplace=True)

    # Save final dataset with timestamp as a column
    df_all.to_csv("data/final_dataset.csv", index=True, index_label="timestamp")
    print(f"Final dataset saved: {df_all.shape[0]} rows, {df_all.shape[1]} columns. Duplicate timestamps are allowed.")

if __name__ == "__main__":
    prepare_dataset()
