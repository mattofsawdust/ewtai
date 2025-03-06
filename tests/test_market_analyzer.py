import unittest
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to the path to import market_analyzer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from market_analyzer import EnhancedMarketAnalyzer

class TestEnhancedMarketAnalyzer(unittest.TestCase):
    """Test cases for the EnhancedMarketAnalyzer class"""
    
    def setUp(self):
        """Setup for tests"""
        # Create a test instance with minimal setup
        self.analyzer = EnhancedMarketAnalyzer(
            symbols=['AAPL', 'MSFT'],
            lookback_period=30,
            models_dir='tests/test_models'
        )
        
        # Create test directory if it doesn't exist
        if not os.path.exists('tests/test_models'):
            os.makedirs('tests/test_models')
            
    def tearDown(self):
        """Teardown after tests"""
        # Clean up any test files
        import shutil
        if os.path.exists('tests/test_models'):
            shutil.rmtree('tests/test_models')
            
    def test_initialization(self):
        """Test that the analyzer initializes correctly"""
        self.assertEqual(self.analyzer.symbols, ['AAPL', 'MSFT'])
        self.assertEqual(self.analyzer.lookback_period, 30)
        self.assertEqual(self.analyzer.models_dir, 'tests/test_models')
        self.assertEqual(self.analyzer.timeframe, '1d')
        
    def test_add_symbol(self):
        """Test adding a symbol"""
        self.analyzer.add_symbol('GOOG')
        self.assertIn('GOOG', self.analyzer.symbols)
        
    def test_remove_symbol(self):
        """Test removing a symbol"""
        self.analyzer.remove_symbol('MSFT')
        self.assertNotIn('MSFT', self.analyzer.symbols)
        
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation with mock data"""
        # Create mock data
        dates = pd.date_range(start='2023-01-01', periods=50)
        mock_data = pd.DataFrame({
            'Open': np.random.randn(50) * 10 + 100,
            'High': np.random.randn(50) * 10 + 105,
            'Low': np.random.randn(50) * 10 + 95,
            'Close': np.random.randn(50) * 10 + 100,
            'Volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # Add data to analyzer
        symbol = 'AAPL'
        self.analyzer.data[symbol] = mock_data
        
        # Calculate indicators
        self.analyzer.calculate_technical_indicators(symbol)
        
        # Check that indicators were calculated
        self.assertIn('SMA_20', self.analyzer.data[symbol].columns)
        self.assertIn('EMA_12', self.analyzer.data[symbol].columns)
        self.assertIn('MACD', self.analyzer.data[symbol].columns)
        
    def test_identify_pivots(self):
        """Test pivot point identification with mock data"""
        # Create mock data with clear peaks and troughs
        dates = pd.date_range(start='2023-01-01', periods=100)
        x = np.linspace(0, 4*np.pi, 100)
        prices = 100 + 10 * np.sin(x)
        
        mock_data = pd.DataFrame({
            'Open': prices,
            'High': prices + 2,
            'Low': prices - 2,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Add data to analyzer
        symbol = 'AAPL'
        self.analyzer.data[symbol] = mock_data
        
        # Identify pivots
        peaks, troughs = self.analyzer.identify_pivots(symbol, window=5, prominence=0.05)
        
        # Check that peaks and troughs were identified
        self.assertTrue(len(peaks) > 0)
        self.assertTrue(len(troughs) > 0)
        
    def test_prepare_model_features(self):
        """Test feature preparation with mock data"""
        # Create mock data
        dates = pd.date_range(start='2023-01-01', periods=100)
        mock_data = pd.DataFrame({
            'Open': np.random.randn(100) * 10 + 100,
            'High': np.random.randn(100) * 10 + 105,
            'Low': np.random.randn(100) * 10 + 95,
            'Close': np.random.randn(100) * 10 + 100,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Add synthetic indicators
        mock_data['SMA_20'] = mock_data['Close'].rolling(window=20).mean()
        mock_data['SMA_50'] = mock_data['Close'].rolling(window=50).mean()
        mock_data['RSI'] = np.random.randn(100) * 20 + 50  # Mock RSI
        
        # Add data to analyzer
        symbol = 'AAPL'
        self.analyzer.data[symbol] = mock_data
        
        # Prepare features
        X, y, feature_names = self.analyzer.prepare_model_features(symbol, prediction_days=5)
        
        # Check that features were prepared correctly
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(feature_names)
        self.assertEqual(len(feature_names), X.shape[1])
        self.assertEqual(len(y), X.shape[0])

if __name__ == '__main__':
    unittest.main()