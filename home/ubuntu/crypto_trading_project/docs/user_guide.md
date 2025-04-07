# Cryptocurrency Trading System User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Data Integration](#data-integration)
6. [Feature Engineering](#feature-engineering)
7. [Machine Learning Models](#machine-learning-models)
8. [Trading Strategies](#trading-strategies)
9. [Backtesting](#backtesting)
10. [Performance Evaluation](#performance-evaluation)
11. [Risk Management](#risk-management)
12. [Troubleshooting](#troubleshooting)
13. [Future Enhancements](#future-enhancements)

## Introduction

The Cryptocurrency Trading System is a comprehensive platform for developing, testing, and evaluating algorithmic trading strategies for cryptocurrencies. The system integrates with Alpaca Markets for real-time and historical data, implements advanced feature engineering, utilizes machine learning for signal generation, and provides robust backtesting capabilities.

This user guide provides detailed instructions on how to use the system, from installation to performance evaluation.

## System Overview

The Cryptocurrency Trading System consists of several integrated components:

1. **Data Integration**: Connects to Alpaca Markets API for real-time and historical cryptocurrency data.
2. **Data Processing**: Validates, normalizes, and stores time-series data in an optimized database.
3. **Feature Engineering**: Calculates technical indicators and generates custom features for analysis.
4. **Machine Learning**: Trains and evaluates models for price prediction and spike detection.
5. **Trading Strategies**: Implements various strategies with comprehensive risk management.
6. **Backtesting**: Simulates trading on historical data to evaluate strategy performance.
7. **Performance Evaluation**: Calculates metrics and generates reports on strategy performance.

## Installation

### Prerequisites

- Python 3.10 or higher
- Alpaca Markets API credentials
- Internet connection for data access

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/adamanx/crypto-trading-system.git
   cd crypto-trading-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up configuration:
   ```bash
   cp config/config.ini.example config/config.ini
   # Edit config.ini with your Alpaca API credentials
   ```

## Configuration

The system is configured through the `config/config.ini` file. This file contains settings for:

- API credentials
- Database configuration
- Trading parameters
- Logging settings

Example configuration:

```ini
[alpaca]
api_key = your_api_key
api_secret = your_api_secret
base_url = https://paper-api.alpaca.markets
data_url = https://data.alpaca.markets

[database]
path = data/crypto_database.db

[trading]
default_symbols = BTC/USD,ETH/USD,SOL/USD
default_timeframes = 1Min,5Min,15Min,1H,4H,1D

[logging]
level = INFO
file = logs/trading_system.log
```

## Data Integration

The system integrates with Alpaca Markets for cryptocurrency data through two main components:

1. **CryptoStreamManager**: Handles real-time data streaming via WebSockets.
2. **AlpacaDataClient**: Fetches historical data for backtesting and model training.

### Fetching Historical Data

To download historical data for a specific cryptocurrency:

```python
from src.alpaca_data_client import AlpacaDataClient

# Initialize client
client = AlpacaDataClient()

# Download historical data
client.download_historical_data(
    symbol="BTC/USD",
    timeframe="1H",
    start_date="2023-01-01",
    end_date="2023-02-01"
)
```

### Streaming Real-time Data

To stream real-time cryptocurrency data:

```python
from src.crypto_stream import CryptoStreamManager

# Initialize stream manager
stream_manager = CryptoStreamManager()

# Start streaming
stream_manager.start_streaming(symbols=["BTC/USD", "ETH/USD"])
```

## Feature Engineering

The system includes a comprehensive feature engineering pipeline that calculates technical indicators and generates custom features for analysis.

### Available Technical Indicators

- Moving Averages (SMA, EMA, WMA)
- Oscillators (RSI, Stochastic, MACD)
- Volatility Indicators (Bollinger Bands, ATR)
- Volume Indicators (OBV, VWAP)
- Trend Indicators (ADX, Ichimoku Cloud)
- And many more

### Generating Features

To generate features for a specific cryptocurrency:

```python
from src.feature_engineering_pipeline import FeatureEngineeringPipeline

# Initialize pipeline
pipeline = FeatureEngineeringPipeline()

# Generate features
features_df = pipeline.process_symbol(
    symbol="BTC/USD",
    start_timestamp=1672531200000,  # 2023-01-01
    end_timestamp=1675209600000,    # 2023-02-01
    timeframe="1H",
    include_indicators=["sma", "rsi", "bbands", "macd"]
)
```

## Machine Learning Models

The system implements machine learning models for price prediction and spike detection.

### Available Models

- Price Direction Classification
- Price Change Regression
- Spike Detection (Binary and Multi-class)

### Training Models

To train models for a specific cryptocurrency:

```python
from src.ml_model_manager import MLModelManager

# Initialize model manager
model_manager = MLModelManager()

# Train models
results = model_manager.train_models_for_symbol(
    symbol="BTC/USD",
    days_back=30,
    timeframe="1H"
)
```

### Making Predictions

To make predictions with trained models:

```python
# Get best models
models = model_manager.get_best_models(
    symbol="BTC/USD",
    timeframe="1H"
)

# Make predictions
predictions = model_manager.predict_with_models(features_df, models)
```

## Trading Strategies

The system includes several trading strategies with comprehensive risk management.

### Available Strategies

1. **SpikeDetectionStrategy**: Uses machine learning models to detect and trade price spikes.
2. **MovingAverageCrossoverStrategy**: Generates signals based on moving average crossovers.
3. **AdvancedTradingStrategy**: Combines multiple signals with robust risk management.
4. **AdaptiveTradingStrategy**: Adjusts parameters based on market conditions.

### Creating a Custom Strategy

To create a custom trading strategy, extend the `TradingStrategy` base class:

```python
from src.backtest_engine import TradingStrategy

class MyCustomStrategy(TradingStrategy):
    def __init__(self, name="MyStrategy", param1=0.1, param2=0.2):
        super().__init__(name)
        self.param1 = param1
        self.param2 = param2
        
    def on_bar(self, timestamp, bar):
        # Process new bar data
        super().on_bar(timestamp, bar)
        
    def generate_signals(self, timestamp, bar):
        # Generate trading signals
        signals = []
        
        # Your signal generation logic here
        
        return signals
```

## Backtesting

The system provides a comprehensive backtesting framework to evaluate trading strategies on historical data.

### Running a Backtest

To run a backtest for a specific strategy:

```python
from src.backtest_engine import BacktestEngine
from src.trading_strategy import AdvancedTradingStrategy

# Initialize backtest engine
backtest_engine = BacktestEngine()

# Create strategy
strategy = AdvancedTradingStrategy(risk_per_trade=0.02)

# Run backtest
results = backtest_engine.run_backtest(
    strategy=strategy,
    symbol="BTC/USD",
    start_date="2023-01-01",
    end_date="2023-02-01",
    timeframe="1H",
    initial_capital=10000.0
)
```

### Comparing Strategies

To compare multiple trading strategies:

```python
# Create strategies
strategies = [
    SpikeDetectionStrategy(),
    MovingAverageCrossoverStrategy(),
    AdvancedTradingStrategy(),
    AdaptiveTradingStrategy()
]

# Compare strategies
comparison = backtest_engine.compare_strategies(
    strategies=strategies,
    symbol="BTC/USD",
    start_date="2023-01-01",
    end_date="2023-02-01",
    timeframe="1H"
)
```

## Performance Evaluation

The system includes a comprehensive performance evaluation framework to assess trading strategy performance.

### Running Performance Evaluation

To run a comprehensive performance evaluation:

```python
from src.performance_evaluator import PerformanceEvaluator

# Initialize performance evaluator
evaluator = PerformanceEvaluator()

# Run strategy evaluation
evaluation_results = evaluator.run_strategy_evaluation(
    strategy=strategy,
    symbol="BTC/USD",
    start_date="2023-01-01",
    end_date="2023-02-01",
    timeframe="1H"
)
```

### Optimizing Strategy Parameters

To optimize strategy parameters:

```python
# Define parameter grid
parameter_grid = {
    'threshold': [0.6, 0.7, 0.8],
    'holding_period': [3, 5, 7]
}

# Optimize parameters
optimization_results = evaluator.optimize_strategy_parameters(
    strategy_class=SpikeDetectionStrategy,
    parameter_grid=parameter_grid,
    symbol="BTC/USD",
    start_date="2023-01-01",
    end_date="2023-02-01",
    timeframe="1H"
)
```

## Risk Management

The system implements comprehensive risk management through the `RiskManager` class.

### Risk Management Features

- Position sizing based on risk percentage
- Kelly Criterion position sizing
- Volatility-adjusted stop-loss calculation
- Portfolio heat monitoring
- Maximum drawdown protection

### Using Risk Management

To use risk management in a trading strategy:

```python
from src.trading_strategy import RiskManager

# Initialize risk manager
risk_manager = RiskManager()

# Calculate position size
position_size = risk_manager.calculate_position_size(
    capital=10000.0,
    risk_per_trade=0.02,
    entry_price=50000.0,
    stop_price=48000.0
)

# Calculate volatility-adjusted stop
stop_price = risk_manager.calculate_volatility_adjusted_stop(
    price=50000.0,
    atr=1000.0,
    multiplier=2.0
)
```

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Check your API credentials in the configuration file
   - Verify your internet connection
   - Ensure you have access to Alpaca Markets API

2. **Database Errors**
   - Check if the database file exists and is accessible
   - Ensure you have write permissions to the database directory
   - Verify the database schema is up to date

3. **Performance Issues**
   - Reduce the number of symbols or timeframes for faster processing
   - Use smaller date ranges for backtesting
   - Optimize feature calculation by selecting only necessary indicators

### Logging

The system uses Python's logging module to log information, warnings, and errors. Logs are written to the file specified in the configuration.

To view logs:

```bash
tail -f logs/trading_system.log
```

## Future Enhancements

The Cryptocurrency Trading System is designed to be extensible. Here are some potential future enhancements:

1. **Additional Data Sources**
   - Integration with other cryptocurrency exchanges
   - On-chain data integration
   - Sentiment analysis from social media

2. **Advanced Machine Learning**
   - Deep learning models (LSTM, Transformer)
   - Reinforcement learning for strategy optimization
   - Ensemble methods for improved predictions

3. **Real-time Trading**
   - Live trading execution
   - Portfolio management
   - Risk monitoring dashboard

4. **User Interface**
   - Web-based dashboard
   - Interactive strategy builder
   - Real-time performance monitoring
