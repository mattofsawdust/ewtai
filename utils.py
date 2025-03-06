"""
Utility functions for the Enhanced Market Analyzer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta
import yfinance as yf

def fetch_market_data(symbols, start_date, end_date, interval='1d'):
    """
    Fetch market data for a list of symbols
    
    Parameters:
    symbols (list): List of symbols to fetch data for
    start_date (str or datetime): Start date for data
    end_date (str or datetime): End date for data
    interval (str): Data interval ('1d', '1h', etc.)
    
    Returns:
    dict: Dictionary of DataFrames with symbol as key
    """
    data = {}
    
    for symbol in symbols:
        try:
            ticker_data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            if not ticker_data.empty:
                data[symbol] = ticker_data
            else:
                print(f"No data available for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            
    return data

def save_prediction_results(results, output_dir='results'):
    """
    Save prediction results to a JSON file
    
    Parameters:
    results (dict): Prediction results
    output_dir (str): Directory to save results
    
    Returns:
    str: Path to saved file
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create filename
    symbol = results.get('symbol', 'unknown')
    filename = f"{symbol}_prediction_{timestamp}.json"
    file_path = os.path.join(output_dir, filename)
    
    # Convert non-serializable objects to strings or floats
    def json_serializable(obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    
    # Save results to file
    with open(file_path, 'w') as f:
        json.dump(results, f, default=json_serializable, indent=4)
        
    return file_path

def load_prediction_results(file_path):
    """
    Load prediction results from a JSON file
    
    Parameters:
    file_path (str): Path to JSON file
    
    Returns:
    dict: Prediction results
    """
    with open(file_path, 'r') as f:
        return json.load(f)
        
def generate_summary_report(predictions, output_file=None):
    """
    Generate a summary report of multiple predictions
    
    Parameters:
    predictions (list): List of prediction dictionaries
    output_file (str): Optional file to save report to
    
    Returns:
    str: Summary report text
    """
    if not predictions:
        return "No predictions to summarize"
        
    report = "# Market Prediction Summary Report\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Sort predictions by confidence
    sorted_predictions = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
    
    # Add table header
    report += "| Symbol | Direction | Confidence | Current Price | Target Price | Horizon |\n"
    report += "|--------|-----------|------------|---------------|--------------|--------|\n"
    
    # Add table rows
    for pred in sorted_predictions:
        symbol = pred.get('symbol', 'N/A')
        direction = "UP" if pred.get('direction', 0) > 0 else "DOWN" if pred.get('direction', 0) < 0 else "NEUTRAL"
        confidence = f"{pred.get('confidence', 0):.2f}"
        current_price = f"{pred.get('current_price', 0):.2f}"
        target_price = f"{pred.get('price_target', 0):.2f}" if pred.get('price_target') else "N/A"
        horizon = pred.get('prediction_horizon', 'N/A')
        
        report += f"| {symbol} | {direction} | {confidence} | {current_price} | {target_price} | {horizon} |\n"
    
    # Add detailed sections
    report += "\n## Detailed Analysis\n\n"
    
    for pred in sorted_predictions:
        symbol = pred.get('symbol', 'N/A')
        report += f"### {symbol}\n\n"
        
        # Add technical indicators
        if 'model_details' in pred and 'technical_indicators' in pred['model_details']:
            tech = pred['model_details']['technical_indicators']
            report += "#### Technical Indicators\n\n"
            
            if 'bullish_signals' in tech and tech['bullish_signals']:
                report += "Bullish Signals:\n"
                for signal in tech['bullish_signals']:
                    report += f"- {signal}\n"
                report += "\n"
                
            if 'bearish_signals' in tech and tech['bearish_signals']:
                report += "Bearish Signals:\n"
                for signal in tech['bearish_signals']:
                    report += f"- {signal}\n"
                report += "\n"
        
        # Add Elliott Wave analysis
        if 'model_details' in pred and 'elliott_wave' in pred['model_details']:
            ew = pred['model_details']['elliott_wave']
            report += "#### Elliott Wave Analysis\n\n"
            pattern_type = ew.get('pattern_type', 'N/A')
            wave_direction = "UP" if ew.get('direction', 0) > 0 else "DOWN"
            wave_confidence = ew.get('confidence', 0)
            
            report += f"Pattern Type: {pattern_type}\n"
            report += f"Expected Direction: {wave_direction}\n"
            report += f"Wave Confidence: {wave_confidence:.2f}\n\n"
            
        report += "\n"
        
    # Save report if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
            
    return report