"""
Main execution script for the Bitcoin LSTM Prediction Pipeline
Runs all steps of the pipeline in sequence:
1. Fetch data from Binance API
2. Add technical indicators  
3. Prepare dataset for Dual LSTM
4. Train Dual LSTM model
5. Evaluate model and generate predictions
6. Create visualizations
"""

from scripts.fetch_data import fetch_all_data
from scripts.indicators import add_indicators
from scripts.prepare_dataset import prepare_dataset
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.visualize import plot_predictions, plot_actual_vs_predicted, plot_training_history, plot_test_set_performance, plot_residual_analysis
import time

def main():
    """Execute the complete Bitcoin LSTM prediction pipeline"""
    start_time = time.time()
    
    print("=" * 60)
    print(" BITCOIN DUAL LSTM PREDICTION PIPELINE")
    print("=" * 60)
    
    try:
        print("\n1️⃣  Fetching data from Binance...")
        fetch_all_data()
        
        print("\n2️⃣  Adding technical indicators...")
        add_indicators()
        
        print("\n3️⃣  Preparing dataset for Dual LSTM...")
        prepare_dataset()
        
        print("\n4️⃣  Training Dual LSTM model...")
        train_model()
        
        print("\n5️⃣  Evaluating model and generating predictions...")
        evaluate_model()
        
        print("\n6️⃣  Creating visualizations...")
        plot_predictions()
        plot_actual_vs_predicted()
        plot_training_history()
        plot_test_set_performance()
        plot_residual_analysis()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("=" * 60)
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"⏱  Total execution time: {duration:.1f} seconds")
        print(" Check the outputs/ folder for results:")
        print("   - predictions.csv (model predictions)")
        print("   - prediction_plot.png (line chart)")
        print("   - actual_vs_predicted.png (comparison chart)")
        print("   - training_val_loss_lr.png (training history)")
        print("   - test_set_performance.png (rigorous test metrics)")
        print("   - residual_analysis.png (model validation)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed with error: {e}")
        print("Please check the error message above and try again.")
        raise

if __name__ == "__main__":
    main()