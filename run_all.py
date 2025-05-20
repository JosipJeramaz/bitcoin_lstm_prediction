from scripts.fetch_data import fetch_all_data
from scripts.indicators import add_indicators
from scripts.prepare_dataset import prepare_dataset
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.visualize import plot_predictions

print("1. Fetching data...")
fetch_all_data()

print("2. Adding indicators...")
add_indicators()

print("3. Preparing dataset...")
prepare_dataset()

print("4. Training model...")
train_model()

print("5. Evaluating model...")
evaluate_model()

print("6. Plotting results...")
plot_predictions()
