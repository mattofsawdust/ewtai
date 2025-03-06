"""
Simple example of using the Enhanced Market Analyzer
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_analyzer import EnhancedMarketAnalyzer
import utils

def run_simple_analysis():
    # Initialize the analyzer with a single symbol
    symbol = 'AAPL'
    analyzer = EnhancedMarketAnalyzer(
        symbols=[symbol],
        lookback_period=365,
        models_dir='../models'
    )
    
    # Fetch data
    print(f"Fetching data for {symbol}...")
    analyzer.fetch_data(extended_data=True)
    
    # Analyze Elliott Waves
    print("Analyzing Elliott Wave patterns...")
    patterns = analyzer.analyze_elliott_waves(symbol)
    if patterns:
        print(f"Found {len(patterns)} Elliott Wave patterns")
    else:
        print("No Elliott Wave patterns found")
    
    # Train AI models
    print("Training machine learning models...")
    
    # Try LSTM model if TensorFlow is available
    try:
        lstm_success = analyzer.train_lstm_model(symbol)
        if not lstm_success:
            print("Could not train LSTM model, will use XGBoost only")
    except Exception as e:
        print(f"Error training LSTM model: {e}")
        print("Will use XGBoost only")
        
    # Train XGBoost model
    analyzer.train_xgboost_model(symbol)
    
    # Make prediction
    print("Making price prediction...")
    prediction = analyzer.predict_price_movement(symbol)
    
    if prediction:
        direction = "UP" if prediction['direction'] > 0 else "DOWN"
        print(f"Prediction for {symbol}: {direction} with {prediction['confidence']:.2f} confidence")
        print(f"Current price: ${prediction['current_price']:.2f}")
        print(f"Price target: ${prediction['price_target']:.2f}")
        
        # Save prediction to file
        output_file = utils.save_prediction_results(prediction)
        print(f"Prediction saved to {output_file}")
        
        # Create visualization
        print("Creating visualization...")
        analyzer.visualize_prediction(symbol, save_path=f'../results/{symbol}_prediction.png')
        print(f"Visualization saved to ../results/{symbol}_prediction.png")
    else:
        print("Could not make prediction")

if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs('../results', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # Run the analysis
    run_simple_analysis()