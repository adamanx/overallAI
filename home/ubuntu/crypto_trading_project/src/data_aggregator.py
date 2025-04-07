import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# List of supported cryptocurrency trading pairs
CRYPTO_PAIRS = [
    # USD pairs
    "AAVE/USD", "AVAX/USD", "BAT/USD", "BCH/USD", "BTC/USD",
    "CRV/USD", "DOGE/USD", "DOT/USD", "ETH/USD", "GRT/USD",
    "LINK/USD", "LTC/USD", "MKR/USD", "PEPE/USD", "SHIB/USD",
    "SOL/USD", "SUSHI/USD", "TRUMP/USD", "UNI/USD", "USDC/USD",
    "USDT/USD", "XRP/USD", "XTZ/USD", "YFI/USD",
    
    # USDC pairs
    "AAVE/USDC", "AVAX/USDC", "BAT/USDC", "BCH/USDC", "BTC/USDC",
    "CRV/USDC", "DOGE/USDC", "DOT/USDC", "ETH/USDC", "GRT/USDC",
    "LINK/USDC", "LTC/USDC", "MKR/USDC", "SHIB/USDC", "SUSHI/USDC",
    "UNI/USDC", "XTZ/USDC", "YFI/USDC",
    
    # USDT pairs
    "AAVE/USDT", "AVAX/USDT", "BAT/USDT", "BCH/USDT", "BTC/USDT",
    "CRV/USDT", "DOGE/USDT", "DOT/USDT", "ETH/USDT", "GRT/USDT",
    "LINK/USDT", "LTC/USDT", "MKR/USDT", "SHIB/USDT", "SUSHI/USDT",
    "UNI/USDT", "XTZ/USDT", "YFI/USDT",
    
    # BTC pairs
    "ETH/BTC"
]

# Supported timeframes
TIMEFRAMES = ['1Min', '5Min', '15Min', '30Min', '1H', '2H', '4H', '1D']

class DataAggregator:
    """
    Aggregates cryptocurrency market data across different timeframes.
    Handles data resampling, consolidation, and transformation.
    """
    
    def __init__(self, db_instance=None):
        """
        Initialize the DataAggregator.
        
        Args:
            db_instance: Instance of TimeSeriesDatabase
        """
        self.logger = logging.getLogger("data_aggregator")
        
        # Import here to avoid circular imports
        if db_instance is None:
            from src.time_series_db import TimeSeriesDatabase
            self.db = TimeSeriesDatabase()
        else:
            self.db = db_instance
        
        # Create aggregated data directory if it doesn't exist
        os.makedirs("data/aggregated", exist_ok=True)
        
        self.logger.info("DataAggregator initialized")
    
    def aggregate_all_timeframes(self, symbol, start_timestamp, end_timestamp, base_timeframe='1Min'):
        """
        Aggregate data for all supported timeframes.
        
        Args:
            symbol (str): Cryptocurrency symbol
            start_timestamp (int): Start timestamp in milliseconds
            end_timestamp (int): End timestamp in milliseconds
            base_timeframe (str): Base timeframe to aggregate from
            
        Returns:
            dict: Dictionary of DataFrames for each timeframe
        """
        try:
            self.logger.info(f"Aggregating all timeframes for {symbol}")
            
            # Get base timeframe data
            base_df = self.db.get_bars(symbol, start_timestamp, end_timestamp, base_timeframe)
            
            if base_df.empty:
                self.logger.warning(f"No data found for {symbol} in {base_timeframe} timeframe")
                return {}
            
            # Map timeframe string to pandas resample rule
            timeframe_map = {
                '1Min': '1T',
                '5Min': '5T',
                '15Min': '15T',
                '30Min': '30T',
                '1H': '1H',
                '2H': '2H',
                '4H': '4H',
                '1D': '1D'
            }
            
            # Aggregate to each timeframe
            results = {base_timeframe: base_df}
            
            for timeframe in TIMEFRAMES:
                if timeframe == base_timeframe:
                    continue
                
                if timeframe not in timeframe_map:
                    self.logger.warning(f"Unsupported timeframe: {timeframe}")
                    continue
                
                # Resample to target timeframe
                resampled = base_df.resample(timeframe_map[timeframe]).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'trade_count': 'sum'
                })
                
                # Drop rows with NaN values
                resampled = resampled.dropna()
                
                if not resampled.empty:
                    results[timeframe] = resampled
                    self.logger.info(f"Aggregated {len(base_df)} {base_timeframe} bars to {len(resampled)} {timeframe} bars")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error aggregating timeframes: {e}", exc_info=True)
            return {}
    
    def save_aggregated_data(self, symbol, aggregated_data, output_dir=None):
        """
        Save aggregated data to CSV files.
        
        Args:
            symbol (str): Cryptocurrency symbol
            aggregated_data (dict): Dictionary of DataFrames for each timeframe
            output_dir (str): Output directory
            
        Returns:
            dict: Dictionary of saved file paths
        """
        try:
            if not aggregated_data:
                self.logger.warning(f"No aggregated data to save for {symbol}")
                return {}
            
            # Use default output directory if not provided
            if output_dir is None:
                output_dir = "data/aggregated"
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save each timeframe to a separate file
            saved_files = {}
            
            for timeframe, df in aggregated_data.items():
                if df.empty:
                    continue
                
                # Generate filename
                filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
                filepath = os.path.join(output_dir, filename)
                
                # Save to CSV
                df.to_csv(filepath)
                
                saved_files[timeframe] = filepath
                self.logger.info(f"Saved {len(df)} {timeframe} bars to {filepath}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving aggregated data: {e}", exc_info=True)
            return {}
    
    def process_all_symbols(self, symbols=None, days_back=30, base_timeframe='1Min'):
        """
        Process and aggregate data for all symbols.
        
        Args:
            symbols (list): List of cryptocurrency symbols
            days_back (int): Number of days of historical data to process
            base_timeframe (str): Base timeframe to aggregate from
            
        Returns:
            dict: Dictionary of results for each symbol
        """
        try:
            # Use default symbols if not provided
            if symbols is None:
                symbols = CRYPTO_PAIRS
            
            # Calculate time range
            end_timestamp = int(datetime.now().timestamp() * 1000)
            start_timestamp = end_timestamp - (days_back * 24 * 60 * 60 * 1000)
            
            self.logger.info(f"Processing {len(symbols)} symbols for the last {days_back} days")
            
            # Process each symbol
            results = {}
            
            for symbol in symbols:
                self.logger.info(f"Processing {symbol}")
                
                # Aggregate data
                aggregated_data = self.aggregate_all_timeframes(
                    symbol, start_timestamp, end_timestamp, base_timeframe
                )
                
                # Save aggregated data
                saved_files = self.save_aggregated_data(symbol, aggregated_data)
                
                results[symbol] = {
                    'aggregated_data': aggregated_data,
                    'saved_files': saved_files
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing symbols: {e}", exc_info=True)
            return {}

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create data aggregator
    aggregator = DataAggregator()
    
    # Process a subset of symbols
    test_symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    results = aggregator.process_all_symbols(test_symbols, days_back=7)
    
    # Print results
    for symbol, result in results.items():
        print(f"{symbol}:")
        for timeframe, filepath in result['saved_files'].items():
            print(f"  {timeframe}: {filepath}")
