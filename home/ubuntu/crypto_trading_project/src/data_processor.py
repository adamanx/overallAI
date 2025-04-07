import os
import logging
import json
from datetime import datetime
from src.time_series_db import TimeSeriesDatabase

class DataProcessor:
    """
    Processes and transforms cryptocurrency market data.
    Handles data validation, normalization, and aggregation.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the DataProcessor.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.logger = logging.getLogger("data_processor")
        self.db = TimeSeriesDatabase(db_path)
        
        # Create processed data directory if it doesn't exist
        os.makedirs("data/processed", exist_ok=True)
        
        self.logger.info("DataProcessor initialized")
    
    async def process_bar(self, bar_data):
        """
        Process bar data from real-time stream.
        
        Args:
            bar_data (dict): Bar data from Alpaca
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Processing bar: {bar_data['S']} - {bar_data['t']}")
            
            # Validate data
            if not self._validate_bar_data(bar_data):
                return False
            
            # Store in database
            success = self.db.store_bar(bar_data)
            if not success:
                return False
            
            # Save processed data
            timestamp = datetime.fromtimestamp(bar_data['t'] / 1000).strftime('%Y%m%d_%H%M%S')
            filename = f"data/processed/bar_{bar_data['S'].replace('/', '_')}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(bar_data, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing bar data: {e}", exc_info=True)
            return False
    
    async def process_trade(self, trade_data):
        """
        Process trade data from real-time stream.
        
        Args:
            trade_data (dict): Trade data from Alpaca
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Processing trade: {trade_data['S']} - {trade_data['t']}")
            
            # Validate data
            if not self._validate_trade_data(trade_data):
                return False
            
            # Store in database
            success = self.db.store_trade(trade_data)
            if not success:
                return False
            
            # Save processed data
            timestamp = datetime.fromtimestamp(trade_data['t'] / 1000).strftime('%Y%m%d_%H%M%S')
            filename = f"data/processed/trade_{trade_data['S'].replace('/', '_')}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(trade_data, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing trade data: {e}", exc_info=True)
            return False
    
    async def process_quote(self, quote_data):
        """
        Process quote data from real-time stream.
        
        Args:
            quote_data (dict): Quote data from Alpaca
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Processing quote: {quote_data['S']} - {quote_data['t']}")
            
            # Validate data
            if not self._validate_quote_data(quote_data):
                return False
            
            # Store in database
            success = self.db.store_quote(quote_data)
            if not success:
                return False
            
            # Save processed data
            timestamp = datetime.fromtimestamp(quote_data['t'] / 1000).strftime('%Y%m%d_%H%M%S')
            filename = f"data/processed/quote_{quote_data['S'].replace('/', '_')}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(quote_data, f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing quote data: {e}", exc_info=True)
            return False
    
    def _validate_bar_data(self, bar_data):
        """
        Validate bar data.
        
        Args:
            bar_data (dict): Bar data from Alpaca
            
        Returns:
            bool: Validation result
        """
        try:
            # Check required fields
            required_fields = ['S', 't', 'o', 'h', 'l', 'c', 'v']
            for field in required_fields:
                if field not in bar_data:
                    self.logger.error(f"Missing required field in bar data: {field}")
                    return False
            
            # Check data types
            if not isinstance(bar_data['t'], int):
                self.logger.error(f"Invalid timestamp type: {type(bar_data['t'])}")
                return False
            
            for field in ['o', 'h', 'l', 'c', 'v']:
                if not isinstance(bar_data[field], (int, float)):
                    self.logger.error(f"Invalid {field} type: {type(bar_data[field])}")
                    return False
            
            # Check logical constraints
            if bar_data['h'] < bar_data['l']:
                self.logger.error(f"High price less than low price: {bar_data['h']} < {bar_data['l']}")
                return False
            
            if bar_data['o'] < 0 or bar_data['h'] < 0 or bar_data['l'] < 0 or bar_data['c'] < 0 or bar_data['v'] < 0:
                self.logger.error(f"Negative values in bar data: {bar_data}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating bar data: {e}", exc_info=True)
            return False
    
    def _validate_trade_data(self, trade_data):
        """
        Validate trade data.
        
        Args:
            trade_data (dict): Trade data from Alpaca
            
        Returns:
            bool: Validation result
        """
        try:
            # Check required fields
            required_fields = ['S', 't', 'p', 's']
            for field in required_fields:
                if field not in trade_data:
                    self.logger.error(f"Missing required field in trade data: {field}")
                    return False
            
            # Check data types
            if not isinstance(trade_data['t'], int):
                self.logger.error(f"Invalid timestamp type: {type(trade_data['t'])}")
                return False
            
            for field in ['p', 's']:
                if not isinstance(trade_data[field], (int, float)):
                    self.logger.error(f"Invalid {field} type: {type(trade_data[field])}")
                    return False
            
            # Check logical constraints
            if trade_data['p'] < 0 or trade_data['s'] < 0:
                self.logger.error(f"Negative values in trade data: {trade_data}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating trade data: {e}", exc_info=True)
            return False
    
    def _validate_quote_data(self, quote_data):
        """
        Validate quote data.
        
        Args:
            quote_data (dict): Quote data from Alpaca
            
        Returns:
            bool: Validation result
        """
        try:
            # Check required fields
            required_fields = ['S', 't', 'bp', 'bs', 'ap', 'as']
            for field in required_fields:
                if field not in quote_data:
                    self.logger.error(f"Missing required field in quote data: {field}")
                    return False
            
            # Check data types
            if not isinstance(quote_data['t'], int):
                self.logger.error(f"Invalid timestamp type: {type(quote_data['t'])}")
                return False
            
            for field in ['bp', 'bs', 'ap', 'as']:
                if not isinstance(quote_data[field], (int, float)):
                    self.logger.error(f"Invalid {field} type: {type(quote_data[field])}")
                    return False
            
            # Check logical constraints
            if quote_data['bp'] < 0 or quote_data['bs'] < 0 or quote_data['ap'] < 0 or quote_data['as'] < 0:
                self.logger.error(f"Negative values in quote data: {quote_data}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating quote data: {e}", exc_info=True)
            return False
    
    def aggregate_bars(self, symbol, start_timestamp, end_timestamp, source_timeframe='1Min', target_timeframe='5Min'):
        """
        Aggregate bars to a larger timeframe.
        
        Args:
            symbol (str): Cryptocurrency symbol
            start_timestamp (int): Start timestamp in milliseconds
            end_timestamp (int): End timestamp in milliseconds
            source_timeframe (str): Source timeframe
            target_timeframe (str): Target timeframe
            
        Returns:
            pandas.DataFrame: Aggregated bars
        """
        try:
            self.logger.info(f"Aggregating {symbol} bars from {source_timeframe} to {target_timeframe}")
            
            # Get source bars
            df = self.db.get_bars(symbol, start_timestamp, end_timestamp, source_timeframe)
            
            if df.empty:
                self.logger.warning(f"No bars found for {symbol}")
                return df
            
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
            
            if target_timeframe not in timeframe_map:
                self.logger.error(f"Unsupported target timeframe: {target_timeframe}")
                return df
            
            # Resample to target timeframe
            resampled = df.resample(timeframe_map[target_timeframe]).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'trade_count': 'sum'
            })
            
            # Drop rows with NaN values
            resampled = resampled.dropna()
            
            self.logger.info(f"Aggregated {len(df)} bars to {len(resampled)} {target_timeframe} bars")
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error aggregating bars: {e}", exc_info=True)
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create data processor
    processor = DataProcessor("data/test_market_data.db")
    
    # Test processing bar data
    bar_data = {
        'S': 'BTC/USD',
        't': int(datetime.now().timestamp() * 1000),
        'o': 50000.0,
        'h': 51000.0,
        'l': 49000.0,
        'c': 50500.0,
        'v': 10.5,
        'n': 100,
        'vw': 50200.0
    }
    
    import asyncio
    success = asyncio.run(processor.process_bar(bar_data))
    print(f"Process bar data: {success}")
