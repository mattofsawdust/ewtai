"""
Minimal example for testing basic functionality
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Add parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def simple_technical_analysis(symbol, lookback_days=30):
    """
    Perform simple technical analysis on a stock
    """
    # Fetch data
    print(f"Fetching data for {symbol}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Download data
    df = yf.download(symbol, start=start_date, end=end_date, interval='1d')
    
    if df.empty:
        print(f"No data available for {symbol}")
        return None
    
    print(f"Data fetched successfully for {symbol}")
    
    # Calculate basic indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Determine trend
    current_price = float(df['Close'].iloc[-1].iloc[0]) if isinstance(df['Close'].iloc[-1], pd.Series) else float(df['Close'].iloc[-1])
    sma_20_val = df['SMA_20'].iloc[-1].iloc[0] if isinstance(df['SMA_20'].iloc[-1], pd.Series) else df['SMA_20'].iloc[-1]
    sma_20 = float(sma_20_val) if not pd.isna(sma_20_val) else 0
    rsi_val = df['RSI'].iloc[-1].iloc[0] if isinstance(df['RSI'].iloc[-1], pd.Series) else df['RSI'].iloc[-1]
    rsi = float(rsi_val) if not pd.isna(rsi_val) else 50
    
    # Simple analysis
    trend = "Bullish" if current_price > sma_20 else "Bearish"
    overbought = rsi > 70
    oversold = rsi < 30
    
    # Print results
    print(f"\nBasic Analysis for {symbol}:")
    print(f"Current Price: ${current_price:.2f}")
    print(f"20-day SMA: ${sma_20:.2f}")
    print(f"RSI: {rsi:.2f}")
    print(f"Trend: {trend}")
    
    if overbought:
        print("Warning: Stock appears overbought (RSI > 70)")
    elif oversold:
        print("Warning: Stock appears oversold (RSI < 30)")
    
    # Save a simple chart
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['SMA_20'], label='20-day SMA')
    
    if 'SMA_50' in df.columns:
        plt.plot(df.index, df['SMA_50'], label='50-day SMA')
    
    plt.title(f'{symbol} Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Create results directory if it doesn't exist
    os.makedirs('../results', exist_ok=True)
    
    # Save chart
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    chart_path = os.path.join(results_dir, f'{symbol}_basic_chart.png')
    plt.savefig(chart_path)
    plt.close()
    
    print(f"Chart saved to {chart_path}")
    
    return df

if __name__ == "__main__":
    # Create output directories if they don't exist
    os.makedirs('../results', exist_ok=True)
    
    # Run analysis on a few symbols
    symbols = ['AAPL', 'MSFT', 'GOOG']
    for symbol in symbols:
        simple_technical_analysis(symbol)
        print("\n" + "-"*50 + "\n")