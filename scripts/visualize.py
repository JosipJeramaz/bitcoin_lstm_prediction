import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

def plot_predictions():
    df = pd.read_csv("outputs/predictions.csv")
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(df["timestamp"]), df["pred_open"], label="Predicted Open")
    plt.plot(pd.to_datetime(df["timestamp"]), df["pred_close"], label="Predicted Close")
    plt.title("Predicted Open/Close Prices")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/prediction_plot.png")
    plt.show()

def plot_predictions_candles():
    df = pd.read_csv("outputs/predictions.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig, ax = plt.subplots(figsize=(14, 7))

    # Candlestick: use pred_open as open, pred_close as close, and set high/low as max/min of open/close
    for idx, row in df.iterrows():
        color = '#26a69a' if row['pred_close'] >= row['pred_open'] else '#ef5350'  # vivid green/red
        # Candle body
        ax.plot([row['timestamp'], row['timestamp']], [row['pred_open'], row['pred_close']], color=color, linewidth=6, solid_capstyle='butt')
        # Candle wick
        ax.plot([row['timestamp'], row['timestamp']], [row['pred_low'], row['pred_high']], color=color, linewidth=1)

    ax.set_title("Predicted Candlestick Chart (Open/Close/High/Low)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/prediction_candles.png")
    plt.show()

if __name__ == "__main__":
    plot_predictions()
    plot_predictions_candles()
