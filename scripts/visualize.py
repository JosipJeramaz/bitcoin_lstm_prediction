import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions():
    df = pd.read_csv("outputs/predictions.csv")
    plt.plot(df["pred_open"], label="Predicted Open")
    plt.plot(df["pred_close"], label="Predicted Close")
    plt.title("Predicted Open/Close Prices")
    plt.legend()
    plt.savefig("outputs/prediction_plot.png")
    plt.show()
