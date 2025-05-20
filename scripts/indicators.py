import pandas as pd
import ta

def add_indicators():
    for tf in ["1d", "4h"]:
        df = pd.read_csv(f"data/raw_{tf}.csv", index_col="timestamp", parse_dates=True)
        for ma in [5, 10, 30, 60]:
            df[f"ma{ma}"] = df["close"].rolling(ma).mean()
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_bbm"] = bb.bollinger_mavg()
        df["bb_bbh"] = bb.bollinger_hband()
        df["bb_bbl"] = bb.bollinger_lband()
        df.dropna(inplace=True)
        df.to_csv(f"data/processed_{tf}.csv")
