#!/usr/bin/env python3
"""
Quick Start Guide for Crypto Trading System
This script provides a step-by-step guide to get the system up and running
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("quickstart")

def check_environment():
    """Check if the environment is properly set up"""
    logger.info("Checking environment...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python version must be 3.8 or higher")
        return False
    
    # Check required directories
    required_dirs = ['data', 'data/models', 'data/historical', 'data/processed']
    for directory in required_dirs:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    # Check config file
    if not os.path.exists('config/config.ini'):
        logger.error("Config file not found: config/config.ini")
        return False
    
    logger.info("Environment check completed successfully")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'alpaca-py', 'pandas', 'numpy', 'matplotlib', 
        'scikit-learn', 'tensorflow', 'xgboost'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            logger.error(f"✗ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("All dependencies are installed")
    return True

def setup_database():
    """Initialize the time series database"""
    logger.info("Setting up database...")
    
    try:
        from src.time_series_db import TimeSeriesDatabase
        db = TimeSeriesDatabase()
        db.initialize_database()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def test_alpaca_connection():
    """Test connection to Alpaca API"""
    logger.info("Testing Alpaca API connection...")
    
    try:
        from src.config_manager import ConfigManager
        config = ConfigManager()
        api_key = config.get('alpaca', 'api_key')
        api_secret = config.get('alpaca', 'api_secret')
        
        if api_key == "YOUR_ALPACA_API_KEY" or api_secret == "YOUR_ALPACA_API_SECRET":
            logger.error("API keys not configured. Please update config/config.ini with your Alpaca API keys")
            return False
        
        from alpaca.data import CryptoHistoricalDataClient
        from alpaca.data.timeframe import TimeFrame
        
        client = CryptoHistoricalDataClient(api_key, api_secret)
        
        # Test fetching some data
        end = datetime.now()
        start = end - timedelta(days=1)
        
        logger.info(f"Fetching BTC/USD data from {start} to {end}")
        bars = client.get_crypto_bars(
            "BTC/USD", 
            TimeFrame.Hour, 
            start, 
            end
        ).df
        
        logger.info(f"Successfully fetched {len(bars)} bars")
        logger.info(f"Sample data:\n{bars.head()}")
        return True
    except Exception as e:
        logger.error(f"Error connecting to Alpaca API: {e}")
        return False

def download_historical_data():
    """Download historical data for backtesting"""
    logger.info("Downloading historical data...")
    
    try:
        from src.alpaca_data_client import AlpacaDataClient
        client = AlpacaDataClient()
        
        # Download data for BTC/USD, ETH/USD, and SOL/USD
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        timeframes = ["1H", "4H", "1D"]
        
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"Downloading {symbol} {timeframe} data from {start_date} to {end_date}")
                client.download_historical_data(symbol, timeframe, start_date, end_date)
        
        logger.info("Historical data downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading historical data: {e}")
        return False

def run_backtest():
    """Run a simple backtest"""
    logger.info("Running backtest...")
    
    try:
        from src.backtest_engine import BacktestEngine
        from src.trading_strategy import AdaptiveTradingStrategy
        
        # Create strategy
        strategy = AdaptiveTradingStrategy()
        
        # Create backtest engine
        engine = BacktestEngine(
            start_date="2023-01-01",
            end_date="2023-02-28",
            symbols=["BTC/USD"],
            strategy=strategy,
            initial_capital=10000
        )
        
        # Run backtest
        results = engine.run()
        
        # Print results
        logger.info(f"Backtest results: {results}")
        return True
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return False

def test_streaming():
    """Test streaming data"""
    logger.info("Testing streaming data...")
    
    try:
        from src.crypto_stream import CryptoStreamManager
        
        # Create stream manager
        stream_manager = CryptoStreamManager()
        
        # Define handler
        def handle_bar(bar):
            logger.info(f"Received bar: {bar}")
            # Stop after receiving one bar
            stream_manager.stop_streaming()
        
        # Set handler
        stream_manager.set_bar_handler(handle_bar)
        
        # Start streaming
        logger.info("Starting streaming for BTC/USD...")
        stream_manager.start_streaming(["BTC/USD"])
        
        logger.info("Streaming test completed")
        return True
    except Exception as e:
        logger.error(f"Error testing streaming: {e}")
        return False

def main():
    """Main function to run all tests"""
    logger.info("Starting quickstart guide...")
    
    steps = [
        ("Check environment", check_environment),
        ("Check dependencies", check_dependencies),
        ("Setup database", setup_database),
        ("Test Alpaca connection", test_alpaca_connection),
        ("Download historical data", download_historical_data),
        ("Run backtest", run_backtest),
        ("Test streaming", test_streaming)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        logger.info(f"\n{'='*50}\nStep: {step_name}\n{'='*50}")
        try:
            result = step_func()
            results[step_name] = "Success" if result else "Failed"
        except Exception as e:
            logger.error(f"Unexpected error in {step_name}: {e}")
            results[step_name] = "Error"
    
    # Print summary
    logger.info("\n\n" + "="*50)
    logger.info("Quickstart Guide Summary")
    logger.info("="*50)
    
    for step_name, result in results.items():
        status = "✓" if result == "Success" else "✗"
        logger.info(f"{status} {step_name}: {result}")
    
    # Provide next steps
    logger.info("\nNext Steps:")
    if all(result == "Success" for result in results.values()):
        logger.info("All tests passed! You're ready to use the system.")
        logger.info("- To run a full backtest: python run_performance_evaluation.py")
        logger.info("- To start live trading: python live_trading.py")
    else:
        logger.info("Some tests failed. Please fix the issues before proceeding.")
        logger.info("- Check the logs above for error details")
        logger.info("- Make sure your config.ini file is properly set up")
        logger.info("- Verify your Alpaca API keys have the correct permissions")

if __name__ == "__main__":
    main()
