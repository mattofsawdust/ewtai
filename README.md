# EWTai - Enhanced Market Analyzer with AI and Elliott Wave Theory

EWTai is an advanced market analysis tool that combines traditional technical analysis with artificial intelligence and Elliott Wave Theory to predict price movements and identify trading opportunities.

## Features

- **Elliott Wave Analysis**: Automatically detects wave patterns in price data
- **Machine Learning Models**: Uses LSTM and XGBoost models for price prediction
- **Technical Indicators**: Calculates and analyzes over 15 technical indicators
- **Sentiment Analysis**: Incorporates sentiment data for enhanced predictions
- **Backtesting**: Simulates trading strategies to evaluate performance
- **Real-time Monitoring**: Continuous monitoring with email alerts for trading opportunities
- **Web Interface**: Interactive Streamlit UI for easy analysis and visualization

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/EWTai.git
   cd EWTai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   Note: TA-Lib may require additional installation steps depending on your platform. See [TA-Lib Installation Guide](https://github.com/mrjbq7/ta-lib).

## Project Structure

```
EWTai/
├── app.py               # Streamlit web interface
├── config/              # Configuration files
│   └── default_settings.json
├── data/                # Data storage
├── examples/            # Example scripts
├── logs/                # Log files
├── models/              # Trained ML models
├── results/             # Analysis results and visualizations
├── market_analyzer.py   # Main analyzer class
├── run_analyzer.py      # Script to run the analyzer
├── tests/               # Unit tests
│   └── test_market_analyzer.py
└── requirements.txt     # Dependencies
```

## Usage

### Web Interface

The easiest way to use EWTai is through the Streamlit web interface:

```
streamlit run app.py
```

This will launch a web app at http://localhost:8501 where you can:
- Enter or select stock symbols to analyze
- View interactive technical charts
- Get analysis summaries and predictions
- Switch between basic and enhanced analysis modes

### Command Line Usage

#### Basic Analysis

To analyze a set of symbols:

```
python run_analyzer.py --mode analyze
```

This will analyze all symbols in the configuration file and generate predictions.

#### Analyze a Specific Symbol

```
python run_analyzer.py --mode analyze --symbol AAPL
```

#### Backtest a Trading Strategy

```
python run_analyzer.py --mode backtest --symbol AAPL
```

This will simulate trading based on the AI predictions and generate performance metrics.

#### Continuous Monitoring

```
python run_analyzer.py --mode monitor
```

This will start continuous monitoring for trading opportunities and send alerts when configured.

### Simple Examples

For quick analysis without requiring all dependencies:

```
python examples/minimal_example.py
```

This runs a basic technical analysis on a few popular stocks and generates charts.

## Configuration

Edit `config/default_settings.json` to customize:

- Symbols to analyze
- Timeframe and lookback period
- Confidence threshold for trading signals
- Email notification settings

## Optional Dependencies

For full functionality, you may want to install:

- **TensorFlow**: Required for LSTM models (`pip install tensorflow>=2.6.0`)
- **TA-Lib**: For additional technical indicators (`pip install ta-lib`)
- **Transformers**: For NLP-based sentiment analysis (`pip install transformers>=4.8.0`)

The basic functionality will work without these, but with reduced capabilities.

## License

This project is licensed under the MIT License - see the LICENSE file for details.