import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Try to import tensorflow, but continue if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    tensorflow_available = True
except ImportError:
    tensorflow_available = False
    print("TensorFlow not available. LSTM models will not be available.")
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
from scipy.signal import find_peaks
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import requests
# Try to import talib, but continue if not available
try:
    import talib
    talib_available = True
except ImportError:
    talib_available = False
    print("TA-Lib not available. Some technical indicators may not be calculated.")
import numpy as np
import warnings
import os
# Try to import transformers, but continue if not available
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    transformers_available = False
    print("Transformers not available. NLP sentiment analysis will not be available.")
from scipy.stats import linregress
import pickle
import joblib

warnings.filterwarnings('ignore')

class EnhancedMarketAnalyzer:
    def __init__(self, symbols=None, timeframe='1d', lookback_period=365, models_dir='models'):
        """
        Initialize the Enhanced Market Analyzer with AI capabilities
        
        Parameters:
        symbols (list): List of stock/crypto symbols to analyze
        timeframe (str): Data timeframe ('1d', '1h', etc.)
        lookback_period (int): Number of days to look back
        models_dir (str): Directory to save trained models
        """
        self.symbols = symbols or []
        self.timeframe = timeframe
        self.lookback_period = lookback_period
        self.data = {}
        self.wave_patterns = {}
        self.retracement_levels = {}
        self.models_dir = models_dir
        self.lstm_models = {}
        self.xgb_models = {}
        self.sentiment_analyzer = None
        self.feature_scalers = {}
        self.target_scalers = {}
        self.market_indicators = {}
        self.indicators_ready = False
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Initialize sentiment analyzer if transformers is available
        if transformers_available:
            try:
                self.sentiment_analyzer = pipeline('sentiment-analysis')
                print("NLP sentiment analyzer initialized successfully")
            except:
                print("Could not initialize sentiment analyzer. Using alternative methods.")
        else:
            print("Transformers not available, sentiment analysis will use alternative methods.")
    
    def add_symbol(self, symbol):
        """Add a symbol to the watchlist"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
    
    def remove_symbol(self, symbol):
        """Remove a symbol from the watchlist"""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
    
    def fetch_data(self, extended_data=True):
        """
        Fetch market data for all symbols with additional features
        
        Parameters:
        extended_data (bool): Whether to fetch additional market indicators
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period)
        
        for symbol in self.symbols:
            try:
                ticker_data = yf.download(symbol, start=start_date, end=end_date, interval=self.timeframe)
                if not ticker_data.empty:
                    self.data[symbol] = ticker_data
                    print(f"Data fetched for {symbol}")
                    
                    # Calculate technical indicators
                    if extended_data:
                        self.calculate_technical_indicators(symbol)
                else:
                    print(f"No data available for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
        
        # Fetch market sentiment data if available
        if extended_data:
            self.fetch_sentiment_data()
            self.fetch_market_indicators()
            self.indicators_ready = True
                
    def calculate_technical_indicators(self, symbol):
        """
        Calculate various technical indicators for the given symbol
        
        Parameters:
        symbol (str): Symbol to calculate indicators for
        """
        if symbol not in self.data or self.data[symbol].empty:
            return
            
        df = self.data[symbol].copy()
        
        # Basic indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        try:
            if len(df) > 14:  # Need at least 14 data points for RSI
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df['RSI'] = 100 - (100 / (1 + rs))
        except:
            print(f"Error calculating RSI for {symbol}")
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
        
        # Stochastic Oscillator
        try:
            if len(df) > 14:
                low_14 = df['Low'].rolling(window=14).min()
                high_14 = df['High'].rolling(window=14).max()
                df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
                df['%D'] = df['%K'].rolling(window=3).mean()
        except:
            print(f"Error calculating Stochastic Oscillator for {symbol}")
        
        # Average True Range (ATR)
        try:
            if len(df) > 14:
                tr1 = df['High'] - df['Low']
                tr2 = abs(df['High'] - df['Close'].shift())
                tr3 = abs(df['Low'] - df['Close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                df['ATR'] = tr.rolling(window=14).mean()
        except:
            print(f"Error calculating ATR for {symbol}")
        
        # On-Balance Volume (OBV)
        try:
            obv = np.zeros(len(df))
            for i in range(1, len(df)):
                if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] + df['Volume'].iloc[i]
                elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                    obv[i] = obv[i-1] - df['Volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
            df['OBV'] = obv
        except:
            print(f"Error calculating OBV for {symbol}")
        
        # Rate of Change (ROC)
        df['ROC'] = df['Close'].pct_change(periods=12) * 100
        
        # Williams %R
        try:
            if len(df) > 14:
                high_14 = df['High'].rolling(window=14).max()
                low_14 = df['Low'].rolling(window=14).min()
                df['Williams_%R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        except:
            print(f"Error calculating Williams %R for {symbol}")
        
        # Chaikin Money Flow (CMF)
        try:
            if len(df) > 20:
                mf_volume = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
                df['CMF'] = mf_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        except:
            print(f"Error calculating CMF for {symbol}")
        
        # ADX - Average Directional Index
        try:
            if 'Close' in df and len(df) > 14:
                df['TR'] = np.maximum(df['High'] - df['Low'], 
                                     np.maximum(abs(df['High'] - df['Close'].shift()), 
                                              abs(df['Low'] - df['Close'].shift())))
                df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']),
                                    np.maximum(df['High'] - df['High'].shift(), 0), 0)
                df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()),
                                    np.maximum(df['Low'].shift() - df['Low'], 0), 0)
                df['ATR_14'] = df['TR'].rolling(window=14).mean()
                df['+DI_14'] = 100 * (df['+DM'].rolling(window=14).mean() / df['ATR_14'])
                df['-DI_14'] = 100 * (df['-DM'].rolling(window=14).mean() / df['ATR_14'])
                df['DX'] = 100 * abs(df['+DI_14'] - df['-DI_14']) / (df['+DI_14'] + df['-DI_14'])
                df['ADX'] = df['DX'].rolling(window=14).mean()
        except:
            print(f"Error calculating ADX for {symbol}")
        
        # Calculate additional wave-specific features for Elliott Wave
        self.calculate_elliott_wave_features(symbol, df)
        
        # Add price momentum features
        df['price_momentum_1d'] = df['Close'].pct_change(periods=1)
        df['price_momentum_5d'] = df['Close'].pct_change(periods=5)
        df['price_momentum_10d'] = df['Close'].pct_change(periods=10)
        df['price_momentum_20d'] = df['Close'].pct_change(periods=20)
        
        # Add volatility features
        df['volatility_5d'] = df['Close'].rolling(window=5).std() / df['Close'].rolling(window=5).mean()
        df['volatility_10d'] = df['Close'].rolling(window=10).std() / df['Close'].rolling(window=10).mean()
        df['volatility_20d'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        # Update the dataframe
        self.data[symbol] = df
        
    def calculate_elliott_wave_features(self, symbol, df):
        """
        Calculate features specific to Elliott Wave analysis
        
        Parameters:
        symbol (str): Symbol to calculate features for
        df (DataFrame): Dataframe containing price data
        """
        # Find potential pivots (peaks and troughs)
        try:
            prices = df['Close'].values
            normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
            
            peaks, _ = find_peaks(normalized_prices, distance=10, prominence=0.05)
            troughs, _ = find_peaks(-normalized_prices, distance=10, prominence=0.05)
            
            # Create a pivot point series (1 for peak, -1 for trough, 0 for neither)
            pivot_series = np.zeros(len(df))
            pivot_series[peaks] = 1
            pivot_series[troughs] = -1
            df['pivot_points'] = pivot_series
            
            # Count recent peaks and troughs (within last 20 days)
            df['recent_peaks'] = df['pivot_points'].rolling(window=20).apply(lambda x: np.sum(x == 1))
            df['recent_troughs'] = df['pivot_points'].rolling(window=20).apply(lambda x: np.sum(x == -1))
            
            # Calculate peak-to-trough ratios (measure of wave symmetry)
            # For each window, count pattern occurrences similar to Elliott Waves
            pattern_windows = [50, 100, 200]
            for window in pattern_windows:
                if len(df) > window:
                    df[f'pattern_strength_{window}'] = df['pivot_points'].rolling(window=window).apply(
                        lambda x: self._calculate_pattern_strength(x)
                    )
        except Exception as e:
            print(f"Error calculating Elliott Wave features: {str(e)}")
            
    def _calculate_pattern_strength(self, pivot_series):
        """
        Calculate a measure of how closely the pivot series matches Elliott Wave patterns
        
        Parameters:
        pivot_series (array): Series of pivot points (1 for peak, -1 for trough, 0 for neither)
        
        Returns:
        float: Pattern strength score (higher is better match)
        """
        # Find all non-zero points (actual pivots)
        pivot_indices = np.where(pivot_series != 0)[0]
        pivot_values = pivot_series[pivot_indices]
        
        if len(pivot_indices) < 5:
            return 0
            
        # Look for 5-3 wave patterns (5 impulse waves + 3 corrective waves)
        # Ideal pattern would be: -1, 1, -1, 1, -1, 1, -1, 1
        # (trough, peak, trough, peak, trough, peak, trough, peak)
        
        pattern_strength = 0
        
        # Check for segments of 8 consecutive alternating pivot points
        for i in range(len(pivot_values) - 7):
            segment = pivot_values[i:i+8]
            
            # Check if the segment alternates properly
            alternating = True
            for j in range(1, len(segment)):
                if segment[j] == segment[j-1]:
                    alternating = False
                    break
                    
            if alternating:
                pattern_strength += 1
                
        return pattern_strength
    
    def fetch_sentiment_data(self):
        """Fetch market sentiment data from external sources"""
        # In a real implementation, this would connect to news APIs, social media, etc.
        # For this example, we'll generate synthetic sentiment data
        print("Fetching market sentiment data...")
        
        for symbol in self.symbols:
            if symbol not in self.data or self.data[symbol].empty:
                continue
                
            df = self.data[symbol]
            
            # Generate synthetic sentiment scores (-1 to 1 range)
            # In reality, these would come from NLP analysis of news, social media, etc.
            np.random.seed(42)  # For reproducibility
            try:
                # Get exact data length to avoid shape mismatches
                data_length = len(df.index)
                
                # Generate 1D array of random values
                sentiment_scores = np.random.normal(0, 0.3, size=data_length).flatten()
                
                # Make sentiment somewhat correlated with price movements
                # Ensure price_changes is also a 1D array
                price_changes = df['Close'].pct_change().fillna(0).values.flatten()
                
                # Check that arrays have compatible shapes
                if len(sentiment_scores) != len(price_changes):
                    # If lengths don't match, recreate arrays with correct length
                    sentiment_scores = np.random.normal(0, 0.3, size=len(price_changes)).flatten()
                
                # Calculate sentiment with correlation to price movements
                sentiment_scores = 0.7 * sentiment_scores + 0.3 * np.sign(price_changes)
                
                # Clip to reasonable range
                sentiment_scores = np.clip(sentiment_scores, -1, 1)
                
                # Add to dataframe
                df['sentiment_score'] = sentiment_scores
                
                # Create 5-day rolling average of sentiment
                df['sentiment_ma5'] = df['sentiment_score'].rolling(window=5).mean()
                
                # Update data
                self.data[symbol] = df
            except Exception as e:
                print(f"Error generating sentiment data for {symbol}: {str(e)}")
    
    def fetch_market_indicators(self):
        """Fetch broader market indicators"""
        try:
            # Market indices (S&P 500, NASDAQ, etc.)
            market_indices = ['^GSPC', '^IXIC', '^DJI']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_period)
            
            for index in market_indices:
                index_data = yf.download(index, start=start_date, end=end_date, interval=self.timeframe)
                if not index_data.empty:
                    # Calculate daily returns
                    index_data['returns'] = index_data['Close'].pct_change()
                    
                    # Calculate volatility
                    index_data['volatility_20d'] = index_data['returns'].rolling(window=20).std()
                    
                    self.market_indicators[index] = index_data
                    
            # Add market indicators to each symbol's data
            for symbol in self.symbols:
                if symbol not in self.data or self.data[symbol].empty:
                    continue
                    
                df = self.data[symbol]
                
                # Add market index returns and volatility for each date
                for index, index_data in self.market_indicators.items():
                    index_name = index.replace('^', '')
                    df[f'{index_name}_return'] = index_data['returns']
                    df[f'{index_name}_volatility'] = index_data['volatility_20d']
                
                # Update data
                self.data[symbol] = df
                
        except Exception as e:
            print(f"Error fetching market indicators: {str(e)}")
    
    def identify_pivots(self, symbol, window=10, prominence=0.05):
        """
        Identify potential wave pivots (highs and lows)
        
        Parameters:
        symbol (str): The symbol to analyze
        window (int): Window size for peak detection
        prominence (float): Prominence threshold for peak detection
        
        Returns:
        tuple: Arrays of peak and trough indices
        """
        if symbol not in self.data:
            print(f"No data available for {symbol}")
            return None, None
            
        # Get the price data and ensure it's a flat array
        try:
            prices = self.data[symbol]['Close'].values
            
            # Check if prices is a 1D array, if not, flatten it
            if len(prices.shape) > 1:
                prices = prices.flatten()
                
            # Normalize prices for better peak detection
            if len(prices) > 0:
                min_price = np.min(prices)
                max_price = np.max(prices)
                if max_price > min_price:  # Avoid division by zero
                    normalized_prices = (prices - min_price) / (max_price - min_price)
                else:
                    # If all prices are the same, we can't find peaks
                    return np.array([]), np.array([])
            else:
                return np.array([]), np.array([])
            
            # Find peaks (potential wave tops)
            peaks, _ = find_peaks(normalized_prices, distance=window, prominence=prominence)
            
            # Find troughs (potential wave bottoms)
            troughs, _ = find_peaks(-normalized_prices, distance=window, prominence=prominence)
            
            return peaks, troughs
        except Exception as e:
            print(f"Error finding pivots: {e}")
            return np.array([]), np.array([])
    
    def analyze_elliott_waves(self, symbol):
        """
        Perform Elliott Wave analysis on the given symbol
        
        Parameters:
        symbol (str): The symbol to analyze
        
        Returns:
        dict: Elliott Wave pattern details if found
        """
        if symbol not in self.data:
            print(f"No data available for {symbol}")
            return None
            
        try:
            peaks, troughs = self.identify_pivots(symbol)
            if peaks is None or troughs is None or len(peaks) == 0 or len(troughs) == 0:
                print(f"Insufficient pivot points for {symbol}")
                return None
                
            # Combine peaks and troughs into pivots with labels
            pivot_data = []
            for peak in peaks:
                # Ensure we're not accessing out of bounds
                if peak < len(self.data[symbol]['Close']):
                    pivot_data.append((peak, 'peak', float(self.data[symbol]['Close'].iloc[peak])))
            
            for trough in troughs:
                # Ensure we're not accessing out of bounds
                if trough < len(self.data[symbol]['Close']):
                    pivot_data.append((trough, 'trough', float(self.data[symbol]['Close'].iloc[trough])))
            
            # Make sure we have enough pivot points to form patterns (at least 8 for a 5-3 pattern)
            if len(pivot_data) < 8:
                print(f"Not enough valid pivot points for {symbol} to form wave patterns")
                return None
                
            # Sort pivots by their position in the time series
            pivot_data.sort(key=lambda x: x[0])
            
            # Try to identify 5-wave and 3-wave patterns
            waves = self._identify_wave_patterns(symbol, pivot_data)
            
            if waves:
                # Calculate Fibonacci retracement levels for potential corrections
                self._calculate_retracement_levels(symbol, waves)
                
                # Calculate key support and resistance levels
                self._calculate_support_resistance_levels(symbol, waves)
                
                return waves
            else:
                print(f"No wave patterns identified for {symbol}")
                return None
        except Exception as e:
            print(f"Error in Elliott Wave analysis for {symbol}: {str(e)}")
            return None
    
    def _identify_wave_patterns(self, symbol, pivots):
        """
        Attempt to identify Elliott Wave patterns from the pivot points
        
        This is a simplified implementation focusing on finding potential 5-wave impulse patterns
        followed by 3-wave corrective patterns. A more sophisticated implementation would
        check additional Elliott Wave rules and guidelines.
        """
        if len(pivots) < 8:  # Need at least 8 points for a 5+3 wave pattern
            print(f"Not enough pivot points to identify wave patterns for {symbol}")
            return None
            
        try:
            patterns = []
            prices = self.data[symbol]['Close']
            
            # For debugging - print the sequence we're searching for
            print(f"Seeking pattern sequence for {symbol}. Found {len(pivots)} pivot points.")
            
            # Look for potential 5-wave patterns with more flexible criteria
            for i in range(len(pivots) - 7):
                try:
                    # Get the sequence of points
                    seq = [p[1] for p in pivots[i:i+8]]
                    ideal_seq = ['trough', 'peak', 'trough', 'peak', 'trough', 'peak', 'trough', 'peak']
                    
                    # For debugging
                    if i == 0:
                        print(f"First sequence found: {seq}")
                        print(f"Compared to ideal: {ideal_seq}")
                        
                    # Check if we have the right sequence - be permissive at first
                    if seq != ideal_seq:
                        # Either continue to next iteration (strict) or use a 
                        # more flexible approach (create artificial pattern)
                        
                        # For now, if no patterns are found with strict criteria,
                        # let's create a single artificial pattern for demo purposes
                        if i == len(pivots) - 8 and not patterns and 'trough' in seq and 'peak' in seq:
                            print(f"Creating demo pattern for visualization from available pivots")
                            
                            # Get the price values
                            wave_points = [float(p[2]) for p in pivots[i:i+8]]
                            positions = [p[0] for p in pivots[i:i+8]]
                            
                            # Create a synthetic pattern for demonstration
                            pattern = {
                                'type': 'demo-pattern',
                                'wave_points': wave_points,
                                'positions': positions,
                                'dates': [prices.index[pos] for pos in positions],
                                'impulse_end': positions[5],  # End of wave 5
                                'correction_end': positions[7],  # End of correction
                                'confidence': 0.5,  # Medium confidence for demo
                                'current_wave': '5',  # Assume we're in wave 5 for demo
                                'is_demo': True
                            }
                            patterns.append(pattern)
                        continue
                        
                    # Get the price values
                    wave_points = [float(p[2]) for p in pivots[i:i+8]]
                    positions = [p[0] for p in pivots[i:i+8]]
                    
                    # Check basic Elliott Wave rules (simplified)
                    # Wave 3 should not be the shortest among waves 1, 3, and 5
                    wave1_move = abs(wave_points[1] - wave_points[0])
                    wave3_move = abs(wave_points[3] - wave_points[2])
                    wave5_move = abs(wave_points[5] - wave_points[4])
                    
                    rule1_ok = not (wave3_move < wave1_move and wave3_move < wave5_move)
                    rule2_ok = not (wave_points[2] <= wave_points[0])  # Wave 2 shouldn't retrace >100% of wave 1
                    rule3_ok = not (min(wave_points[3], wave_points[4]) <= max(wave_points[0], wave_points[1]))  # Wave 4 shouldn't overlap wave 1
                    
                    # Determine if we're in a bullish or bearish pattern
                    is_bullish = wave_points[5] > wave_points[0]
                    
                    # Determine the current wave we're in (assume wave 5 or correction)
                    current_wave = '5'  # Default to wave 5
                    
                    # Check the rules
                    if rule1_ok and rule2_ok and rule3_ok:
                        # If all checks pass, we have a potential Elliott Wave pattern
                        pattern = {
                            'type': 'impulse+correction',
                            'wave_points': wave_points,
                            'positions': positions,
                            'dates': [prices.index[pos] for pos in positions],
                            'impulse_end': positions[5],  # End of wave 5
                            'correction_end': positions[7],  # End of correction
                            'confidence': self._calculate_pattern_confidence(symbol, positions),
                            'current_wave': current_wave,
                            'is_bullish': is_bullish
                        }
                        patterns.append(pattern)
                    else:
                        # For debugging
                        if i < 3:
                            print(f"Rules check: Wave 3 shortest: {not rule1_ok}, Wave 2 retraces >100%: {not rule2_ok}, Wave 4 overlaps Wave 1: {not rule3_ok}")
                except Exception as e:
                    print(f"Error processing pattern at index {i}: {e}")
            
            # If we didn't find any patterns with strict rules, create a demo pattern
            if not patterns and len(pivots) >= 8:
                print(f"Creating demo pattern for visualization")
                
                # Use the first 8 pivot points to create a pattern
                wave_points = [float(p[2]) for p in pivots[:8]]
                positions = [p[0] for p in pivots[:8]]
                
                # Create a synthetic pattern for demonstration
                pattern = {
                    'type': 'demo-pattern',
                    'wave_points': wave_points,
                    'positions': positions,
                    'dates': [prices.index[pos] for pos in positions],
                    'impulse_end': positions[5],  # End of wave 5
                    'correction_end': positions[7],  # End of correction
                    'confidence': 0.5,  # Medium confidence for demo
                    'current_wave': '5',  # Assume we're in wave 5 for demo
                    'is_demo': True,
                    'is_bullish': wave_points[5] > wave_points[0]
                }
                patterns.append(pattern)
            
            # Store the patterns for this symbol
            self.wave_patterns[symbol] = patterns
            
            if patterns:
                print(f"Found {len(patterns)} wave patterns for {symbol}")
                return patterns
            else:
                print(f"No valid Elliott Wave patterns identified for {symbol}")
                return None
                
        except Exception as e:
            print(f"Error identifying wave patterns for {symbol}: {e}")
            return None
    
    def _calculate_pattern_confidence(self, symbol, positions):
        """
        Calculate a confidence score for the identified pattern
        
        Parameters:
        symbol (str): Symbol being analyzed
        positions (list): List of position indices for the pattern
        
        Returns:
        float: Confidence score from 0 to 1
        """
        if symbol not in self.data:
            return 0.5
            
        df = self.data[symbol]
        
        # Base confidence score
        confidence = 0.7
        
        # Adjust confidence based on technical indicators at the last position
        try:
            last_pos = positions[-1]
            
            # Check if RSI suggests a reversal
            if 'RSI' in df.columns:
                rsi = df['RSI'].iloc[last_pos]
                if rsi < 30 or rsi > 70:
                    confidence += 0.05
            
            # Check for MACD crossover
            if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
                macd = df['MACD'].iloc[last_pos]
                signal = df['MACD_Signal'].iloc[last_pos]
                prev_macd = df['MACD'].iloc[last_pos-1] if last_pos > 0 else macd
                prev_signal = df['MACD_Signal'].iloc[last_pos-1] if last_pos > 0 else signal
                
                if (prev_macd < prev_signal and macd > signal) or (prev_macd > prev_signal and macd < signal):
                    confidence += 0.05
            
            # Check for volume confirmation
            if 'Volume' in df.columns:
                avg_volume = df['Volume'].iloc[max(0, last_pos-5):last_pos+1].mean()
                last_volume = df['Volume'].iloc[last_pos]
                
                if last_volume > avg_volume * 1.5:
                    confidence += 0.05
                    
            # Sentiment influence
            if 'sentiment_score' in df.columns:
                sentiment = df['sentiment_score'].iloc[last_pos]
                if abs(sentiment) > 0.5:  # Strong sentiment
                    confidence += 0.05 * np.sign(sentiment)
                    
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            
        # Ensure confidence is between 0 and 1
        return max(0, min(1, confidence))
    
    def _calculate_retracement_levels(self, symbol, wave_patterns):
        """Calculate Fibonacci retracement levels for the identified patterns"""
        if not wave_patterns:
            return
            
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        retracements = []
        
        for pattern in wave_patterns:
            impulse_start = pattern['wave_points'][0]
            impulse_end = pattern['wave_points'][5]
            price_range = impulse_end - impulse_start
            
            levels = {}
            for fib in fib_levels:
                if price_range > 0:  # Upward impulse
                    levels[fib] = impulse_end - (price_range * fib)
                else:  # Downward impulse
                    levels[fib] = impulse_end + (abs(price_range) * fib)
            
            retracements.append({
                'pattern_idx': wave_patterns.index(pattern),
                'impulse_range': (impulse_start, impulse_end),
                'retracement_levels': levels
            })
        
        self.retracement_levels[symbol] = retracements
        
    def _calculate_support_resistance_levels(self, symbol, wave_patterns):
        """Calculate key support and resistance levels based on the wave patterns"""
        if not wave_patterns:
            return
            
        support_resistance = {
            'key_resistance': [],
            'key_support': [],
            'invalidation_levels': []
        }
        
        for pattern in wave_patterns:
            wave_points = pattern['wave_points']
            positions = pattern['positions']
            dates = pattern['dates']
            
            # Check if this is a bullish or bearish pattern
            is_bullish = wave_points[5] > wave_points[0]
            
            # Determine key levels based on wave structure
            if is_bullish:
                # Key support levels (potential buy zones)
                # Wave 2 low is a key support
                support_resistance['key_support'].append({
                    'price': wave_points[2],
                    'date': dates[2],
                    'description': 'Wave 2 Low',
                    'confidence': 0.8
                })
                
                # Wave 4 low is another key support
                support_resistance['key_support'].append({
                    'price': wave_points[4],
                    'date': dates[4],
                    'description': 'Wave 4 Low',
                    'confidence': 0.7
                })
                
                # Key resistance levels
                # Wave 3 high is often a key resistance
                support_resistance['key_resistance'].append({
                    'price': wave_points[3],
                    'date': dates[3],
                    'description': 'Wave 3 High',
                    'confidence': 0.75
                })
                
                # Wave 5 high is a major resistance
                support_resistance['key_resistance'].append({
                    'price': wave_points[5],
                    'date': dates[5],
                    'description': 'Wave 5 High',
                    'confidence': 0.85
                })
                
                # Invalidation level - if price falls below wave 1 low
                support_resistance['invalidation_levels'].append({
                    'price': wave_points[0],
                    'date': dates[0],
                    'description': 'Below Wave 1 Low (Pattern Invalidation)',
                    'type': 'bearish_invalidation'
                })
            else:
                # Bearish pattern - reverse the logic
                # Key resistance levels
                support_resistance['key_resistance'].append({
                    'price': wave_points[2],
                    'date': dates[2],
                    'description': 'Wave 2 High',
                    'confidence': 0.8
                })
                
                support_resistance['key_resistance'].append({
                    'price': wave_points[4],
                    'date': dates[4],
                    'description': 'Wave 4 High',
                    'confidence': 0.7
                })
                
                # Key support levels
                support_resistance['key_support'].append({
                    'price': wave_points[3],
                    'date': dates[3],
                    'description': 'Wave 3 Low',
                    'confidence': 0.75
                })
                
                support_resistance['key_support'].append({
                    'price': wave_points[5],
                    'date': dates[5],
                    'description': 'Wave 5 Low',
                    'confidence': 0.85
                })
                
                # Invalidation level - if price rises above wave 1 high
                support_resistance['invalidation_levels'].append({
                    'price': wave_points[0],
                    'date': dates[0],
                    'description': 'Above Wave 1 High (Pattern Invalidation)',
                    'type': 'bullish_invalidation'
                })
            
            # Add wave points as potential stop-loss levels
            if is_bullish:
                # For bullish pattern, add stop-loss below key support levels
                support_resistance['stop_loss_levels'] = [
                    {
                        'price': wave_points[0] * 0.98,  # Slightly below Wave 1 start
                        'description': 'Stop-Loss below Wave 1 start',
                        'for_entry': 'Wave 1-2 entry'
                    },
                    {
                        'price': wave_points[2] * 0.98,  # Slightly below Wave 2 low
                        'description': 'Stop-Loss below Wave 2 low',
                        'for_entry': 'Wave 3 entry'
                    },
                    {
                        'price': wave_points[4] * 0.98,  # Slightly below Wave 4 low
                        'description': 'Stop-Loss below Wave 4 low',
                        'for_entry': 'Wave 5 entry'
                    }
                ]
            else:
                # For bearish pattern, add stop-loss above key resistance levels
                support_resistance['stop_loss_levels'] = [
                    {
                        'price': wave_points[0] * 1.02,  # Slightly above Wave 1 start
                        'description': 'Stop-Loss above Wave 1 start',
                        'for_entry': 'Wave 1-2 entry'
                    },
                    {
                        'price': wave_points[2] * 1.02,  # Slightly above Wave 2 high
                        'description': 'Stop-Loss above Wave 2 high',
                        'for_entry': 'Wave 3 entry'
                    },
                    {
                        'price': wave_points[4] * 1.02,  # Slightly above Wave 4 high
                        'description': 'Stop-Loss above Wave 4 high',
                        'for_entry': 'Wave 5 entry'
                    }
                ]
        
        # Store support and resistance levels for this symbol
        if not hasattr(self, 'support_resistance_levels'):
            self.support_resistance_levels = {}
        
        self.support_resistance_levels[symbol] = support_resistance
    
    def prepare_model_features(self, symbol, prediction_days=5, include_market_data=True):
        """
        Prepare features for AI models
        
        Parameters:
        symbol (str): Symbol to prepare features for
        prediction_days (int): Number of days ahead to predict
        include_market_data (bool): Whether to include market indicators
        
        Returns:
        tuple: X (features), y (target), feature_names
        """
        if symbol not in self.data or self.data[symbol].empty:
            print(f"No data available for {symbol}")
            return None, None, None
            
        df = self.data[symbol].copy()
        
        # Create target variable (future price change)
        df['target'] = df['Close'].pct_change(periods=prediction_days).shift(-prediction_days)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < 50:
            print(f"Not enough data for {symbol} after preprocessing")
            return None, None, None
            
        # Select features (technical indicators and other relevant columns)
        feature_columns = [
            'SMA_20', 'SMA_50', 'SMA_200', 
            'EMA_12', 'EMA_26', 
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Std',
            '%K', '%D', 'ATR', 'OBV', 'ROC', 'Williams_%R', 'CMF', 'ADX',
            'pivot_points', 'recent_peaks', 'recent_troughs',
            'price_momentum_1d', 'price_momentum_5d', 'price_momentum_10d', 'price_momentum_20d',
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'sentiment_score', 'sentiment_ma5'
        ]
        
        # Add market index features if available
        if include_market_data and self.indicators_ready:
            for index in self.market_indicators:
                index_name = index.replace('^', '')
                feature_columns.extend([f'{index_name}_return', f'{index_name}_volatility'])
            
        # Keep only available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Check if we have enough features
        if len(available_features) < 5:
            print(f"Not enough features available for {symbol}")
            return None, None, None
            
        # Prepare feature matrix and target vector
        X = df[available_features].values
        y = df['target'].values
        
        return X, y, available_features
    
    def train_lstm_model(self, symbol, epochs=50, prediction_days=5, sequence_length=20):
        """
        Train an LSTM model for price prediction
        
        Parameters:
        symbol (str): Symbol to train model for
        epochs (int): Number of training epochs
        prediction_days (int): Number of days ahead to predict
        sequence_length (int): Length of input sequences
        
        Returns:
        bool: True if training was successful, False otherwise
        """
        if not tensorflow_available:
            print("TensorFlow not available. Cannot train LSTM model.")
            return False
            
        print(f"Training LSTM model for {symbol}...")
        
        X, y, feature_names = self.prepare_model_features(symbol, prediction_days)
        if X is None or len(X) < sequence_length + prediction_days + 10:
            print(f"Not enough data for {symbol} to train LSTM model")
            return False
            
        # Scale features
        feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = feature_scaler.fit_transform(X)
        
        # Scale target
        target_scaler = StandardScaler()
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Save scalers
        self.feature_scalers[symbol] = feature_scaler
        self.target_scalers[symbol] = target_scaler
        
        # Prepare sequences
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - sequence_length - prediction_days + 1):
            X_seq.append(X_scaled[i:i+sequence_length])
            y_seq.append(y_scaled[i+sequence_length+prediction_days-1])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        try:
            # Build LSTM model
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Setup early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Save model
            model_path = os.path.join(self.models_dir, f'lstm_{symbol}.h5')
            model.save(model_path)
            
            # Save model metadata
            metadata = {
                'feature_names': feature_names,
                'sequence_length': sequence_length,
                'prediction_days': prediction_days,
                'trained_date': datetime.now().strftime('%Y-%m-%d')
            }
            with open(os.path.join(self.models_dir, f'lstm_{symbol}_metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            # Store model in memory
            self.lstm_models[symbol] = {
                'model': model,
                'metadata': metadata
            }
            
            print(f"LSTM model for {symbol} trained successfully")
            return True
                
        except Exception as e:
            print(f"Error training LSTM model for {symbol}: {str(e)}")
            return False

    def train_xgboost_model(self, symbol, prediction_days=5):
        """
        Train an XGBoost model for price movement classification
        
        Parameters:
        symbol (str): Symbol to train model for
        prediction_days (int): Number of days ahead to predict
        
        Returns:
        bool: True if training was successful, False otherwise
        """
        print(f"Training XGBoost model for {symbol}...")
        
        X, y, feature_names = self.prepare_model_features(symbol, prediction_days)
        if X is None:
            print(f"Not enough data for {symbol} to train XGBoost model")
            return False
            
        # Convert target to classification: 1 if price goes up, 0 if down
        y_class = (y > 0).astype(int)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_class, test_size=0.2, random_state=42
        )
        
        # Train XGBoost model
        try:
            xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                early_stopping_rounds=20,
                verbose=0
            )
            
            # Calculate accuracy
            y_pred = xgb_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"XGBoost model for {symbol} achieved {accuracy:.2f} accuracy")
            
            # Save model
            model_path = os.path.join(self.models_dir, f'xgb_{symbol}.pkl')
            joblib.dump(xgb_model, model_path)
            
            # Save model metadata
            metadata = {
                'feature_names': feature_names,
                'prediction_days': prediction_days,
                'accuracy': float(accuracy),
                'trained_date': datetime.now().strftime('%Y-%m-%d')
            }
            with open(os.path.join(self.models_dir, f'xgb_{symbol}_metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
            # Calculate feature importance
            importance = xgb_model.feature_importances_
            feature_importance = [(feature_names[i], float(importance[i])) 
                                 for i in range(len(feature_names))]
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Add feature importance to metadata
            metadata['feature_importance'] = feature_importance
            
            # Store model in memory
            self.xgb_models[symbol] = {
                'model': xgb_model,
                'metadata': metadata
            }
            
            print(f"XGBoost model for {symbol} trained successfully")
            return True
            
        except Exception as e:
            print(f"Error training XGBoost model for {symbol}: {str(e)}")
            return False
        
    def predict_price_movement(self, symbol, prediction_days=5, confidence_threshold=0.6):
        """
        Predict price movement using the ensemble of trained models
        
        Parameters:
        symbol (str): Symbol to predict for
        prediction_days (int): Number of days ahead to predict
        confidence_threshold (float): Minimum confidence for predictions
        
        Returns:
        dict: Prediction details
        """
        if symbol not in self.data or self.data[symbol].empty:
            print(f"No data available for {symbol}")
            return None
            
        # Prepare the latest data for prediction
        df = self.data[symbol].copy()
        
        # Check if we have models for this symbol
        lstm_available = symbol in self.lstm_models
        xgb_available = symbol in self.xgb_models
        
        if not lstm_available and not xgb_available:
            print(f"No trained models available for {symbol}")
            return self._generate_baseline_prediction(symbol, prediction_days)
            
        results = {}
        confidence = 0.5  # Starting neutral confidence
        
        # Get latest Elliott Wave patterns
        wave_patterns = self.analyze_elliott_waves(symbol)
        if wave_patterns and len(wave_patterns) > 0:
            # Get the most recent pattern confidence
            pattern_confidence = wave_patterns[-1].get('confidence', 0.5)
            
            # Determine expected direction from the pattern
            latest_position = wave_patterns[-1]['positions'][-1]
            latest_type = 'peak' if df['pivot_points'].iloc[latest_position] > 0 else 'trough'
            
            if latest_type == 'peak':
                wave_direction = -1  # Expected to go down after a peak
            else:
                wave_direction = 1  # Expected to go up after a trough
                
            # Include this in results
            results['elliott_wave'] = {
                'direction': wave_direction,
                'confidence': pattern_confidence,
                'pattern_type': wave_patterns[-1]['type']
            }
            
            # Adjust overall confidence
            confidence = 0.3 * confidence + 0.7 * (0.5 + (pattern_confidence - 0.5) * wave_direction)
        
        # Make LSTM prediction if available
        if lstm_available:
            try:
                # Get model and metadata
                lstm_model = self.lstm_models[symbol]['model']
                metadata = self.lstm_models[symbol]['metadata']
                sequence_length = metadata['sequence_length']
                feature_names = metadata['feature_names']
                
                # Prepare features
                available_features = [col for col in feature_names if col in df.columns]
                if len(available_features) == len(feature_names):
                    # Extract and scale the latest sequence
                    latest_data = df[available_features].values
                    scaled_data = self.feature_scalers[symbol].transform(latest_data)
                    
                    # Create sequence
                    sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, len(feature_names))
                    
                    # Make prediction
                    scaled_pred = lstm_model.predict(sequence)[0][0]
                    pred = self.target_scalers[symbol].inverse_transform([[scaled_pred]])[0][0]
                    
                    # Store in results
                    results['lstm'] = {
                        'predicted_change': float(pred),
                        'direction': 1 if pred > 0 else -1,
                        'confidence': min(1.0, abs(pred) * 10)  # Scale confidence based on magnitude
                    }
                    
                    # Adjust overall confidence
                    direction = results['lstm']['direction']
                    lstm_confidence = results['lstm']['confidence']
                    confidence = 0.6 * confidence + 0.4 * (0.5 + (lstm_confidence - 0.5) * direction)
            except Exception as e:
                print(f"Error making LSTM prediction for {symbol}: {str(e)}")
        
        # Make XGBoost prediction if available
        if xgb_available:
            try:
                # Get model and metadata
                xgb_model = self.xgb_models[symbol]['model']
                metadata = self.xgb_models[symbol]['metadata']
                feature_names = metadata['feature_names']
                
                # Prepare features
                available_features = [col for col in feature_names if col in df.columns]
                if len(available_features) == len(feature_names):
                    # Extract latest data
                    latest_data = df[available_features].values[-1:, :]
                    
                    # Make prediction
                    pred_proba = xgb_model.predict_proba(latest_data)[0]
                    pred_class = int(pred_proba[1] > 0.5)
                    direction = 1 if pred_class == 1 else -1
                    
                    # Get confidence from probability
                    confidence_level = max(pred_proba)
                    
                    # Store in results
                    results['xgboost'] = {
                        'probability_up': float(pred_proba[1]),
                        'predicted_class': pred_class,
                        'direction': direction,
                        'confidence': float(confidence_level)
                    }
                    
                    # Adjust overall confidence
                    xgb_confidence = results['xgboost']['confidence']
                    confidence = 0.5 * confidence + 0.5 * (0.5 + (xgb_confidence - 0.5) * direction)
            except Exception as e:
                print(f"Error making XGBoost prediction for {symbol}: {str(e)}")
                
        # Get technical indicator signals
        tech_signals = self._analyze_technical_indicators(symbol)
        if tech_signals:
            results['technical_indicators'] = tech_signals
            
            # Adjust overall confidence based on technical signals
            tech_direction = tech_signals['overall_direction']
            tech_strength = tech_signals['signal_strength']
            confidence = 0.7 * confidence + 0.3 * (0.5 + (tech_strength - 0.5) * tech_direction)
            
        # Add sentiment analysis if available
        if 'sentiment_score' in df.columns:
            latest_sentiment = df['sentiment_score'].iloc[-1]
            sentiment_ma5 = df['sentiment_ma5'].iloc[-1] if 'sentiment_ma5' in df.columns else latest_sentiment
            
            results['sentiment'] = {
                'current': float(latest_sentiment),
                'moving_avg': float(sentiment_ma5),
                'direction': 1 if sentiment_ma5 > 0 else -1,
                'strength': abs(sentiment_ma5)
            }
            
            # Adjust overall confidence based on sentiment
            sentiment_direction = results['sentiment']['direction']
            sentiment_strength = results['sentiment']['strength']
            confidence = 0.8 * confidence + 0.2 * (0.5 + (sentiment_strength - 0.5) * sentiment_direction)
            
        # Determine overall direction and confidence
        overall_direction = 1 if confidence > 0.5 else (-1 if confidence < 0.5 else 0)
        overall_confidence = abs(confidence - 0.5) * 2  # Scale to 0-1 range
        
        # Calculate estimated price target
        current_price = df['Close'].iloc[-1]
        price_target = None
        
        # If we have a confident prediction, calculate price target
        if overall_confidence > confidence_threshold:
            # Estimate price movement magnitude
            avg_daily_volatility = df['Close'].pct_change().std()
            estimated_move = avg_daily_volatility * prediction_days * overall_direction * overall_confidence * 2
            price_target = current_price * (1 + estimated_move)
        
        # Compile final prediction
        prediction = {
            'symbol': symbol,
            'current_price': float(current_price),
            'prediction_horizon': f"{prediction_days} days",
            'direction': overall_direction,
            'confidence': float(overall_confidence),
            'price_target': float(price_target) if price_target is not None else None,
            'model_details': results
        }
        
        return prediction

    def _analyze_technical_indicators(self, symbol):
        """
        Analyze technical indicators for trading signals
        
        Parameters:
        symbol (str): Symbol to analyze
        
        Returns:
        dict: Analysis results
        """
        if symbol not in self.data or self.data[symbol].empty:
            return None
            
        df = self.data[symbol].copy()
        
        # Dictionary to track signals
        signals = {
            'bullish': [],
            'bearish': []
        }
        
        # Check for moving average crossovers
        if all(col in df.columns for col in ['SMA_20', 'SMA_50']):
            if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] and df['SMA_20'].iloc[-2] <= df['SMA_50'].iloc[-2]:
                signals['bullish'].append('SMA_20 crossed above SMA_50')
            elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] and df['SMA_20'].iloc[-2] >= df['SMA_50'].iloc[-2]:
                signals['bearish'].append('SMA_20 crossed below SMA_50')
        
        # Check RSI
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if rsi < 30:
                signals['bullish'].append(f'RSI oversold ({rsi:.1f})')
            elif rsi > 70:
                signals['bearish'].append(f'RSI overbought ({rsi:.1f})')
        
        # Check MACD
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
                signals['bullish'].append('MACD crossed above signal line')
            elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
                signals['bearish'].append('MACD crossed below signal line')
        
        # Check Bollinger Bands
        if all(col in df.columns for col in ['Close', 'BB_Lower', 'BB_Upper']):
            if df['Close'].iloc[-1] < df['BB_Lower'].iloc[-1]:
                signals['bullish'].append('Price below lower Bollinger Band')
            elif df['Close'].iloc[-1] > df['BB_Upper'].iloc[-1]:
                signals['bearish'].append('Price above upper Bollinger Band')
        
        # Check stochastic oscillator
        if all(col in df.columns for col in ['%K', '%D']):
            if df['%K'].iloc[-1] < 20 and df['%D'].iloc[-1] < 20:
                signals['bullish'].append('Stochastic oscillator in oversold territory')
            elif df['%K'].iloc[-1] > 80 and df['%D'].iloc[-1] > 80:
                signals['bearish'].append('Stochastic oscillator in overbought territory')
            elif df['%K'].iloc[-1] > df['%D'].iloc[-1] and df['%K'].iloc[-2] <= df['%D'].iloc[-2]:
                signals['bullish'].append('Stochastic %K crossed above %D')
            elif df['%K'].iloc[-1] < df['%D'].iloc[-1] and df['%K'].iloc[-2] >= df['%D'].iloc[-2]:
                signals['bearish'].append('Stochastic %K crossed below %D')
        
        # Determine overall signal
        bullish_count = len(signals['bullish'])
        bearish_count = len(signals['bearish'])
        total_signals = bullish_count + bearish_count
        
        if total_signals == 0:
            overall_direction = 0
            signal_strength = 0
        else:
            overall_direction = 1 if bullish_count > bearish_count else (-1 if bearish_count > bullish_count else 0)
            signal_strength = abs(bullish_count - bearish_count) / total_signals
            
        return {
            'bullish_signals': signals['bullish'],
            'bearish_signals': signals['bearish'],
            'overall_direction': overall_direction,
            'signal_strength': signal_strength
        }

    def _generate_baseline_prediction(self, symbol, prediction_days):
        """
        Generate a baseline prediction when ML models aren't available
        
        Parameters:
        symbol (str): Symbol to predict for
        prediction_days (int): Number of days ahead to predict
        
        Returns:
        dict: Basic prediction based on technical indicators and wave analysis
        """
        if symbol not in self.data or self.data[symbol].empty:
            return None
            
        df = self.data[symbol].copy()
        current_price = df['Close'].iloc[-1]
        
        # Analyze Elliott Waves
        wave_patterns = self.analyze_elliott_waves(symbol)
        
        # Analyze technical indicators
        tech_signals = self._analyze_technical_indicators(symbol)
        
        # Initialize prediction components
        direction = 0
        confidence = 0.5
        
        # Use wave patterns if available
        if wave_patterns and len(wave_patterns) > 0:
            latest_pattern = wave_patterns[-1]
            pattern_confidence = latest_pattern.get('confidence', 0.5)
            
            latest_position = latest_pattern['positions'][-1]
            latest_type = 'peak' if df['pivot_points'].iloc[latest_position] > 0 else 'trough'
            
            if latest_type == 'peak':
                direction = -1  # Expected to go down after a peak
            else:
                direction = 1  # Expected to go up after a trough
                
            confidence = 0.3 + (pattern_confidence * 0.4)  # Scale to 0.3-0.7 range
        
        # Incorporate technical signals if available
        if tech_signals:
            tech_direction = tech_signals['overall_direction']
            tech_strength = tech_signals['signal_strength']
            
            # If tech signals agree with wave patterns, increase confidence
            if tech_direction == direction:
                confidence = min(0.9, confidence + 0.2)
            # If tech signals disagree, decrease confidence
            elif tech_direction != 0 and direction != 0:
                confidence = max(0.1, confidence - 0.2)
                
            # If no direction from wave patterns, use tech signals
            if direction == 0:
                direction = tech_direction
                confidence = 0.3 + (tech_strength * 0.4)  # Scale to 0.3-0.7 range
        
        # Estimate price target
        avg_daily_volatility = df['Close'].pct_change().std()
        price_move = avg_daily_volatility * prediction_days * direction * (confidence - 0.5) * 4
        price_target = current_price * (1 + price_move)
        
        # Create prediction object
        prediction = {
            'symbol': symbol,
            'current_price': float(current_price),
            'prediction_horizon': f"{prediction_days} days",
            'direction': direction,
            'confidence': float(confidence),
            'price_target': float(price_target) if price_target is not None else None,
            'model_details': {
                'method': 'baseline',
                'elliott_wave': bool(wave_patterns),
                'technical_indicators': tech_signals
            }
        }
        
        return prediction

    def find_trade_opportunities(self, confidence_threshold=0.7):
        """
        Find potential trade opportunities based on all available analysis
        
        Parameters:
        confidence_threshold (float): Minimum confidence level for opportunities
        
        Returns:
        list: Trading opportunities
        """
        opportunities = []
        
        for symbol in self.symbols:
            # Get price prediction
            prediction = self.predict_price_movement(symbol, confidence_threshold=confidence_threshold)
            if not prediction:
                continue
                
            # Check if we have a high-confidence prediction
            if prediction['confidence'] >= confidence_threshold:
                # Get Elliott Wave patterns and retracement levels
                wave_patterns = self.wave_patterns.get(symbol, [])
                retracements = self.retracement_levels.get(symbol, [])
                
                # Additional data about the opportunity
                opp_data = {
                    'symbol': symbol,
                    'current_price': prediction['current_price'],
                    'predicted_direction': prediction['direction'],
                    'confidence': prediction['confidence'],
                    'price_target': prediction['price_target'],
                    'time_horizon': prediction['prediction_horizon'],
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M')
                }
                
                # Add Elliott Wave information if available
                if 'elliott_wave' in prediction['model_details']:
                    opp_data['elliott_wave'] = prediction['model_details']['elliott_wave']
                    
                # Add technical indicator signals if available
                if 'technical_indicators' in prediction['model_details']:
                    opp_data['technical_signals'] = prediction['model_details']['technical_indicators']
                
                # If we have a lot of confidence, recommend an action
                if prediction['confidence'] > 0.8:
                    if prediction['direction'] > 0:
                        opp_data['recommended_action'] = 'BUY'
                    else:
                        opp_data['recommended_action'] = 'SELL'
                else:
                    opp_data['recommended_action'] = 'WATCH'
                
                opportunities.append(opp_data)
        
        return opportunities

    def visualize_waves(self, symbol, save_path=None):
        """
        Create a visualization of the detected Elliott Waves
        
        Parameters:
        symbol (str): Symbol to visualize
        save_path (str): Optional path to save the visualization
        """
        if symbol not in self.data or symbol not in self.wave_patterns:
            print(f"No data or wave patterns available for {symbol}")
            return
            
        plt.figure(figsize=(14, 7))
        
        # Plot price data
        plt.plot(self.data[symbol].index, self.data[symbol]['Close'], 'k-', alpha=0.7)
        
        # Plot detected waves
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, pattern in enumerate(self.wave_patterns[symbol]):
            dates = pattern['dates']
            points = pattern['wave_points']
            
            # Plot impulse wave (0-5)
            plt.plot(dates[:6], points[:6], f'{colors[i % len(colors)]}-', linewidth=2, 
                     label=f'Impulse {i+1}')
            
            # Plot correction wave (5-7)
            plt.plot(dates[5:8], points[5:8], f'{colors[i % len(colors)]}--', linewidth=2)
            
            # Add wave numbers
            for j in range(6):
                plt.annotate(f'W{j+1}', (dates[j], points[j]), 
                            textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.annotate('A', (dates[6], points[6]), textcoords="offset points", 
                        xytext=(0,10), ha='center')
            plt.annotate('B', (dates[7], points[7]), textcoords="offset points", 
                        xytext=(0,10), ha='center')
            
            # Plot retracement levels if available
            if symbol in self.retracement_levels:
                for retracement in self.retracement_levels[symbol]:
                    if retracement['pattern_idx'] == i:
                        for fib, level in retracement['retracement_levels'].items():
                            plt.axhline(y=level, color=colors[i % len(colors)], 
                                       linestyle=':', alpha=0.5)
                            plt.annotate(f'{fib*100:.1f}%', 
                                       (dates[-1], level), 
                                       textcoords="offset points", 
                                       xytext=(5, 0), ha='left')
        
        # Plot prediction if available
        prediction = self.predict_price_movement(symbol)
        if prediction and prediction['price_target']:
            target_date = self.data[symbol].index[-1] + pd.Timedelta(days=5)
            current_price = prediction['current_price']
            target_price = prediction['price_target']
            
            plt.plot([self.data[symbol].index[-1], target_date], 
                    [current_price, target_price], 
                    'm--', linewidth=2, label='AI Prediction')
            
            plt.annotate(f"Target: {target_price:.2f}", 
                        (target_date, target_price),
                        textcoords="offset points",
                        xytext=(5, 0), ha='left',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
        
        plt.title(f'Enhanced Elliott Wave Analysis for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def visualize_prediction(self, symbol, save_path=None):
        """
        Create a visualization of the price prediction with all indicators
        
        Parameters:
        symbol (str): Symbol to visualize
        save_path (str): Optional path to save the visualization
        """
        if symbol not in self.data:
            print(f"No data available for {symbol}")
            return
            
        # Make prediction
        prediction = self.predict_price_movement(symbol)
        if not prediction:
            print(f"Could not make prediction for {symbol}")
            return
            
        df = self.data[symbol].copy()
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot price and moving averages
        axs[0].plot(df.index, df['Close'], 'k-', label='Close Price')
        if 'SMA_20' in df.columns:
            axs[0].plot(df.index, df['SMA_20'], 'b-', label='SMA 20')
        if 'SMA_50' in df.columns:
            axs[0].plot(df.index, df['SMA_50'], 'g-', label='SMA 50')
        if 'SMA_200' in df.columns:
            axs[0].plot(df.index, df['SMA_200'], 'r-', label='SMA 200')
            
        # Plot Bollinger Bands if available
        if all(col in df.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            axs[0].plot(df.index, df['BB_Upper'], 'c--', alpha=0.5)
            axs[0].plot(df.index, df['BB_Middle'], 'c-', alpha=0.5)
            axs[0].plot(df.index, df['BB_Lower'], 'c--', alpha=0.5)
            axs[0].fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='cyan', alpha=0.1)
        
        # Plot Elliott Wave patterns if available
        if symbol in self.wave_patterns:
            for i, pattern in enumerate(self.wave_patterns[symbol]):
                dates = pattern['dates']
                points = pattern['wave_points']
                
                # Plot impulse wave (0-5)
                axs[0].plot(dates[:6], points[:6], 'm-', linewidth=2, label=f'Elliott Wave' if i == 0 else "")
                
                # Plot correction wave (5-7)
                axs[0].plot(dates[5:8], points[5:8], 'm--', linewidth=2)
        
        # Plot future prediction
        if prediction and prediction['price_target']:
            target_date = df.index[-1] + pd.Timedelta(days=5)
            current_price = prediction['current_price']
            target_price = prediction['price_target']
            
            axs[0].plot([df.index[-1], target_date], 
                      [current_price, target_price], 
                      'g--' if prediction['direction'] > 0 else 'r--', 
                      linewidth=3, 
                      label='AI Prediction')
            
            # Add confidence level as text
            confidence_text = f"Confidence: {prediction['confidence']:.2f}"
            axs[0].annotate(confidence_text, 
                          (target_date, target_price),
                          textcoords="offset points",
                          xytext=(5, 10), ha='left',
                          bbox=dict(boxstyle="round,pad=0.3", 
                                  fc="yellow" if prediction['confidence'] > 0.7 else "white", 
                                  alpha=0.5))
        
        axs[0].set_title(f'Price Prediction for {symbol}')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='upper left')
    
        # Plot indicators - RSI
        if 'RSI' in df.columns:
            axs[1].plot(df.index, df['RSI'], 'b-', label='RSI')
            axs[1].axhline(y=70, color='r', linestyle='--', alpha=0.5)
            axs[1].axhline(y=30, color='g', linestyle='--', alpha=0.5)
            axs[1].fill_between(df.index, 70, 30, color='gray', alpha=0.1)
            axs[1].set_ylim(0, 100)
            axs[1].set_title('RSI')
            axs[1].grid(True, alpha=0.3)
            
            # Plot MACD
            if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
                axs[2].plot(df.index, df['MACD'], 'b-', label='MACD')
                axs[2].plot(df.index, df['MACD_Signal'], 'r-', label='Signal')
                
                # Color MACD histogram
                if 'MACD_Hist' in df.columns:
                    for i in range(1, len(df)):
                        if df['MACD_Hist'].iloc[i] >= 0:
                            axs[2].bar(df.index[i], df['MACD_Hist'].iloc[i], color='g', alpha=0.5)
                        else:
                            axs[2].bar(df.index[i], df['MACD_Hist'].iloc[i], color='r', alpha=0.5)
                
                axs[2].set_title('MACD')
                axs[2].grid(True, alpha=0.3)
                axs[2].legend(loc='upper left')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
    
    def send_alert(self, opportunity, email_to, email_from, password, smtp_server='smtp.gmail.com', port=587):
        """Send email alert for trading opportunity"""
        subject = f"AI Trading Signal: {opportunity['symbol']}"
        
        body = f"""
        AI-Enhanced Trading Opportunity Detected
        
        Symbol: {opportunity['symbol']}
        Current Price: {opportunity['current_price']:.2f}
        Predicted Direction: {'UP' if opportunity['predicted_direction'] > 0 else 'DOWN'}
        Confidence: {opportunity['confidence']:.2f}
        Price Target: {opportunity['price_target']:.2f}
        Time Horizon: {opportunity['time_horizon']}
        Recommended Action: {opportunity['recommended_action']}
        Analysis Date: {opportunity['analysis_date']}
        
        """
        
        # Add Elliott Wave information if available
        if 'elliott_wave' in opportunity:
            body += f"""
        Elliott Wave Analysis:
        Pattern Type: {opportunity['elliott_wave'].get('pattern_type', 'N/A')}
        Wave Direction: {'UP' if opportunity['elliott_wave'].get('direction', 0) > 0 else 'DOWN'}
        Wave Confidence: {opportunity['elliott_wave'].get('confidence', 0):.2f}
        """
        
        # Add technical signals if available
        if 'technical_signals' in opportunity:
            tech = opportunity['technical_signals']
            body += f"""
        Technical Signals:
        Bullish Signals: {len(tech.get('bullish_signals', []))}
        Bearish Signals: {len(tech.get('bearish_signals', []))}
        Signal Strength: {tech.get('signal_strength', 0):.2f}
        """
            
            # List bullish signals
            if tech.get('bullish_signals'):
                body += "\nBullish signals:\n"
                for signal in tech.get('bullish_signals', []):
                    body += f"- {signal}\n"
                    
            # List bearish signals
            if tech.get('bearish_signals'):
                body += "\nBearish signals:\n"
                for signal in tech.get('bearish_signals', []):
                    body += f"- {signal}\n"
        
        body += """
        This is an automated alert generated by your AI-Enhanced Market Analyzer.
        """
        
        message = MIMEMultipart()
        message['From'] = email_from
        message['To'] = email_to
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))
        
        try:
            server = smtplib.SMTP(smtp_server, port)
            server.starttls()
            server.login(email_from, password)
            server.send_message(message)
            server.quit()
            print(f"Alert sent for {opportunity['symbol']}")
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            
    def run_monitoring_loop(self, interval_hours=6, email_to=None, email_from=None, email_password=None, confidence_threshold=0.7):
        """
        Run continuous monitoring for trading opportunities
        
        Parameters:
        interval_hours (int): Hours between checks
        email_to (str): Email to send alerts to
        email_from (str): Email to send alerts from
        email_password (str): Password for email_from
        confidence_threshold (float): Minimum confidence for alerts
        """
        print(f"Starting enhanced market monitoring for {self.symbols}")
        
        while True:
            print(f"\nUpdating data at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            self.fetch_data(extended_data=True)
            
            # Train or update models periodically
            for symbol in self.symbols:
                if symbol not in self.lstm_models:
                    self.train_lstm_model(symbol)
                if symbol not in self.xgb_models:
                    self.train_xgboost_model(symbol)
            
            # Find trading opportunities
            opportunities = self.find_trade_opportunities(confidence_threshold=confidence_threshold)
            
            if opportunities:
                print(f"Found {len(opportunities)} high-confidence trading opportunities:")
                for opp in opportunities:
                    direction = "UP" if opp['predicted_direction'] > 0 else "DOWN"
                    print(f" - {opp['symbol']}: {direction} with {opp['confidence']:.2f} confidence, target: {opp['price_target']:.2f}")
                    
                    # Send email alerts if configured
                    if email_to and email_from and email_password:
                        self.send_alert(opp, email_to, email_from, email_password)
                    
                    # Generate and save visualization
                    try:
                        save_path = os.path.join(self.models_dir, f"{opp['symbol']}_prediction.png")
                        self.visualize_prediction(opp['symbol'], save_path=save_path)
                    except Exception as e:
                        print(f"Error creating visualization: {str(e)}")
            else:
                print("No high-confidence trading opportunities found at this time.")
                
            # Wait for the next check
            next_check = datetime.now() + timedelta(hours=interval_hours)
            print(f"Next check scheduled for {next_check.strftime('%Y-%m-%d %H:%M')}")
            time.sleep(interval_hours * 3600)
    
    def trade_simulation(self, symbol, start_date=None, end_date=None, initial_capital=10000, confidence_threshold=0.7):
        """
        Run a trading simulation to test the strategy
        
        Parameters:
        symbol (str): Symbol to simulate trading for
        start_date (str): Starting date for simulation (format: 'YYYY-MM-DD')
        end_date (str): Ending date for simulation (format: 'YYYY-MM-DD')
        initial_capital (float): Initial capital amount
        confidence_threshold (float): Minimum confidence for trades
        
        Returns:
        dict: Simulation results
        """
        if symbol not in self.data or self.data[symbol].empty:
            print(f"No data available for {symbol}")
            return None
            
        df = self.data[symbol].copy()
        
        # Set date range for simulation
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        if len(df) < 30:
            print(f"Not enough data for simulation. Need at least 30 days.")
            return None
            
        # Initialize simulation variables
        capital = initial_capital
        shares = 0
        trades = []
        equity_curve = []
        position_history = []
        
        # Train models if not already trained
        if symbol not in self.lstm_models:
            self.train_lstm_model(symbol)
        if symbol not in self.xgb_models:
            self.train_xgboost_model(symbol)
            
        # Loop through each day in the simulation
        for i in range(20, len(df) - 5):  # Start after enough data for indicators
            date = df.index[i]
            current_price = df['Close'].iloc[i]
            
            # Record current equity
            equity = capital + (shares * current_price)
            equity_curve.append({
                'date': date,
                'equity': equity,
                'cash': capital,
                'shares': shares,
                'price': current_price
            })
            position_history.append(1 if shares > 0 else (0 if shares == 0 else -1))
            
            # Only make trading decisions every 5 days
            if i % 5 != 0:
                continue
                
            # Use historical data up to current date
            historical_data = df.iloc[:i+1].copy()
            
            # Temporarily replace self.data with historical data
            original_data = self.data.copy()
            self.data[symbol] = historical_data
            
            # Generate prediction
            prediction = self.predict_price_movement(symbol, confidence_threshold=confidence_threshold)
            
            # Restore original data
            self.data = original_data
            
            # Skip if no prediction
            if not prediction:
                continue
                
            # Execute trades based on prediction
            if prediction['confidence'] >= confidence_threshold:
                trade_direction = prediction['direction']
                
                # Buy signal
                if trade_direction > 0 and shares == 0:
                    # Calculate number of shares to buy
                    max_shares = capital // current_price
                    shares_to_buy = int(max_shares * 0.95)  # Invest 95% of capital
                    
                    if shares_to_buy > 0:
                        trade_cost = shares_to_buy * current_price
                        capital -= trade_cost
                        shares += shares_to_buy
                        
                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'cost': trade_cost,
                            'confidence': prediction['confidence'],
                            'remaining_capital': capital
                        })
                
                # Sell signal
                elif trade_direction < 0 and shares > 0:
                    trade_proceeds = shares * current_price
                    capital += trade_proceeds
                    
                    trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': shares,
                        'proceeds': trade_proceeds,
                        'confidence': prediction['confidence'],
                        'remaining_capital': capital
                    })
                    
                    shares = 0
            
        # Sell any remaining shares at the end
        if shares > 0:
            final_price = df['Close'].iloc[-1]
            trade_proceeds = shares * final_price
            capital += trade_proceeds
            
            trades.append({
                'date': df.index[-1],
                'action': 'SELL',
                'price': final_price,
                'shares': shares,
                'proceeds': trade_proceeds,
                'confidence': 1.0,  # Forced sell
                'remaining_capital': capital
            })
            
            shares = 0
        
        # Calculate final equity and performance metrics
        final_equity = capital
        total_return = (final_equity / initial_capital - 1) * 100
        
        # Calculate annualized return
        days = (df.index[-1] - df.index[20]).days
        annualized_return = ((final_equity / initial_capital) ** (365 / days) - 1) * 100 if days > 0 else 0
        
        # Calculate max drawdown
        peak = 0
        max_drawdown = 0
        for point in equity_curve:
            equity = point['equity']
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            
        # Calculate Sharpe ratio (assuming risk-free rate of 0%)
        if len(equity_curve) > 1:
            equity_values = [point['equity'] for point in equity_curve]
            equity_returns = [equity_values[i] / equity_values[i-1] - 1 for i in range(1, len(equity_values))]
            avg_return = np.mean(equity_returns)
            std_return = np.std(equity_returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Calculate win rate
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        wins = 0
        for i in range(min(len(buy_trades), len(sell_trades))):
            if sell_trades[i]['price'] > buy_trades[i]['price']:
                wins += 1
                
        win_rate = (wins / len(buy_trades)) * 100 if len(buy_trades) > 0 else 0
        
        # Compile results
        results = {
            'symbol': symbol,
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': len(buy_trades),
            'trades': trades,
            'equity_curve': equity_curve,
            'position_history': position_history
        }
        
        return results
    
    def visualize_simulation(self, results, save_path=None):
        """
        Visualize the results of a trading simulation
        
        Parameters:
        results (dict): Results from trade_simulation method
        save_path (str): Optional path to save the visualization
        """
        if not results:
            print("No simulation results to visualize")
            return
            
        symbol = results['symbol']
        equity_curve = results['equity_curve']
        trades = results['trades']
        position_history = results['position_history']
        
        # Create dates and equity values for plotting
        dates = [point['date'] for point in equity_curve]
        equity_values = [point['equity'] for point in equity_curve]
        prices = [point['price'] for point in equity_curve]
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Plot equity curve
        axs[0].plot(dates, equity_values, 'b-', label='Portfolio Value')
        axs[0].set_title(f'Trading Simulation for {symbol}')
        axs[0].set_ylabel('Portfolio Value ($)')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc='upper left')
        
        # Add buy/sell markers
        for trade in trades:
            if trade['action'] == 'BUY':
                axs[0].plot(trade['date'], trade['remaining_capital'] + trade['cost'], 'g^', markersize=10)
            else:  # SELL
                axs[0].plot(trade['date'], trade['remaining_capital'], 'rv', markersize=10)
        
        # Plot price with buy/sell points
        axs[1].plot(dates, prices, 'k-', label='Price')
        axs[1].set_ylabel('Price ($)')
        axs[1].grid(True, alpha=0.3)
        
        # Add buy/sell markers to price chart
        for trade in trades:
            if trade['action'] == 'BUY':
                axs[1].plot(trade['date'], trade['price'], 'g^', markersize=10)
            else:  # SELL
                axs[1].plot(trade['date'], trade['price'], 'rv', markersize=10)
        
        # Plot position history (1 for long, 0 for cash, -1 for short)
        axs[2].plot(dates, position_history, 'b-', drawstyle='steps-post')
        axs[2].set_ylabel('Position')
        axs[2].set_yticks([-1, 0, 1])
        axs[2].set_yticklabels(['Short', 'Cash', 'Long'])
        axs[2].grid(True, alpha=0.3)
        axs[2].set_xlabel('Date')
        
        # Add performance metrics as text
        stats_text = (
            f"Initial Capital: ${results['initial_capital']:,.2f}\n"
            f"Final Equity: ${results['final_equity']:,.2f}\n"
            f"Total Return: {results['total_return']:.2f}%\n"
            f"Annualized Return: {results['annualized_return']:.2f}%\n"
            f"Max Drawdown: {results['max_drawdown']:.2f}%\n"
            f"Sharpe Ratio: {results['sharpe_ratio']:.2f}\n"
            f"Win Rate: {results['win_rate']:.2f}%\n"
            f"Number of Trades: {results['num_trades']}"
        )
        
        # Add text box with stats
        axs[0].text(0.02, 0.05, stats_text, transform=axs[0].transAxes,
                  bbox=dict(facecolor='white', alpha=0.8),
                  fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize the analyzer with some popular stocks and cryptocurrencies
    symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'BTC-USD', 'ETH-USD', 'ADA-USD']
    
    analyzer = EnhancedMarketAnalyzer(symbols=symbols, lookback_period=365)
    
    # Fetch initial data with all indicators
    analyzer.fetch_data(extended_data=True)
    
    # Analyze each symbol
    for symbol in symbols:
        # Analyze Elliott Waves
        patterns = analyzer.analyze_elliott_waves(symbol)
        if patterns:
            print(f"Found {len(patterns)} Elliott Wave patterns for {symbol}")
            
        # Train AI models
        analyzer.train_lstm_model(symbol)
        analyzer.train_xgboost_model(symbol)
        
        # Make predictions
        prediction = analyzer.predict_price_movement(symbol)
        if prediction:
            direction = "UP" if prediction['direction'] > 0 else "DOWN"
            print(f"Prediction for {symbol}: {direction} with {prediction['confidence']:.2f} confidence")
            print(f"Price target: {prediction['price_target']:.2f}")
            
        # Visualize analysis
        analyzer.visualize_prediction(symbol)
        
    # Find current opportunities
    opportunities = analyzer.find_trade_opportunities(confidence_threshold=0.7)
    if opportunities:
        for opp in opportunities:
            print(f"High-confidence opportunity for {opp['symbol']} - Action: {opp['recommended_action']}")
    
    # Run a backtest simulation
    symbol = 'AAPL'  # Example symbol
    simulation_results = analyzer.trade_simulation(
        symbol=symbol,
        initial_capital=10000,
        confidence_threshold=0.65
    )
    
    if simulation_results:
        print(f"Simulation Results for {symbol}:")
        print(f"Total Return: {simulation_results['total_return']:.2f}%")
        print(f"Annualized Return: {simulation_results['annualized_return']:.2f}%")
        print(f"Sharpe Ratio: {simulation_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {simulation_results['max_drawdown']:.2f}%")
        
        # Visualize simulation results
        analyzer.visualize_simulation(simulation_results)
    
    # Start continuous monitoring (uncomment and configure email to use)
    # analyzer.run_monitoring_loop(
    #     interval_hours=6,
    #     email_to="your_email@example.com",
    #     email_from="your_sender_email@example.com",
    #     email_password="your_email_password",
    #     confidence_threshold=0.7
    # )