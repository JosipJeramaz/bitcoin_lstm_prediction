import pandas as pd

def prepare_dataset():
    df_1d = pd.read_csv("data/processed_1d.csv", index_col="timestamp", parse_dates=True)
    df_4h = pd.read_csv("data/processed_4h.csv", index_col="timestamp", parse_dates=True)
    df_combined = pd.concat([df_1d, df_4h], axis=1, join="inner", keys=["1d", "4h"])
    df_combined.dropna(inplace=True)
    df_combined.to_csv("data/final_dataset.csv")
