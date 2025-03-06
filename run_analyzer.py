import json
import os
import argparse
from market_analyzer import EnhancedMarketAnalyzer

def load_config(config_file):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Market Analyzer')
    parser.add_argument('--config', type=str, default='config/default_settings.json', 
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['analyze', 'backtest', 'monitor'], 
                        default='analyze', help='Operation mode')
    parser.add_argument('--symbol', type=str, help='Specific symbol to analyze (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize analyzer
    symbols = [args.symbol] if args.symbol else config['symbols']
    analyzer = EnhancedMarketAnalyzer(
        symbols=symbols,
        timeframe=config['timeframe'],
        lookback_period=config['lookback_period'],
        models_dir='models'
    )
    
    # Fetch data
    print(f"Fetching data for {', '.join(symbols)}...")
    analyzer.fetch_data(extended_data=True)
    
    if args.mode == 'analyze':
        # Analyze each symbol
        for symbol in symbols:
            print(f"\nAnalyzing {symbol}...")
            
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
                
            # Save visualization
            save_path = os.path.join('results', f"{symbol}_prediction.png")
            analyzer.visualize_prediction(symbol, save_path=save_path)
            print(f"Visualization saved to {save_path}")
            
        # Find current opportunities
        opportunities = analyzer.find_trade_opportunities(
            confidence_threshold=config['confidence_threshold']
        )
        if opportunities:
            print("\nHigh-confidence trading opportunities:")
            for opp in opportunities:
                print(f" - {opp['symbol']}: {opp['recommended_action']} (Confidence: {opp['confidence']:.2f})")
                
    elif args.mode == 'backtest':
        if not args.symbol:
            print("Error: Symbol is required for backtest mode")
            return
            
        symbol = args.symbol
        print(f"Running backtest simulation for {symbol}...")
        
        simulation_results = analyzer.trade_simulation(
            symbol=symbol,
            initial_capital=10000,
            confidence_threshold=config['confidence_threshold']
        )
        
        if simulation_results:
            print(f"Simulation Results for {symbol}:")
            print(f"Total Return: {simulation_results['total_return']:.2f}%")
            print(f"Annualized Return: {simulation_results['annualized_return']:.2f}%")
            print(f"Sharpe Ratio: {simulation_results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {simulation_results['max_drawdown']:.2f}%")
            print(f"Win Rate: {simulation_results['win_rate']:.2f}%")
            print(f"Number of Trades: {simulation_results['num_trades']}")
            
            # Save visualization
            save_path = os.path.join('results', f"{symbol}_backtest.png")
            analyzer.visualize_simulation(simulation_results, save_path=save_path)
            print(f"Backtest visualization saved to {save_path}")
            
    elif args.mode == 'monitor':
        print("Starting continuous monitoring...")
        email_settings = config['email_settings']
        
        email_to = email_settings['email_to'] if email_settings['enabled'] else None
        email_from = email_settings['email_from'] if email_settings['enabled'] else None
        email_password = email_settings['password'] if email_settings['enabled'] else None
        
        analyzer.run_monitoring_loop(
            interval_hours=config['monitoring_interval_hours'],
            email_to=email_to,
            email_from=email_from,
            email_password=email_password,
            confidence_threshold=config['confidence_threshold']
        )

if __name__ == "__main__":
    main()