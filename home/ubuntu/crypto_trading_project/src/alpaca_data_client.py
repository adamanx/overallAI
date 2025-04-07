import os
import logging
import asyncio
import pandas as pd
from alpaca.data import CryptoHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import CryptoBarsRequest
from datetime import datetime, timedelta

class AlpacaDataClient:
    """
    Client for fetching historical cryptocurrency data from Alpaca Markets.
    """
    
    def __init__(self, api_key=None, api_secret=None):
        """
        Initialize the AlpacaDataClient.
        
        Args:
            api_key (str, optional): Alpaca API key. If None, will try to load from config.
            api_secret (str, optional): Alpaca API secret. If None, will try to load from config.
        """
        self.logger = logging.getLogger("alpaca_data_client")
        
        # If API keys not provided, try to load from config
        if api_key is None or api_secret is None:
            try:
                from src.config_manager import ConfigManager
                config = ConfigManager()
                api_key, api_secret, _ = config.get_alpaca_credentials()
                self.logger.info("Loaded API credentials from config file")
            except Exception as e:
                self.logger.error(f"Error loading API credentials from config: {e}")
                raise ValueError("API credentials not provided and could not be loaded from config")
        
        self.client = CryptoHistoricalDataClient(api_key, api_secret)
        
        # Create data directory if it doesn't exist
        os.makedirs("data/historical", exist_ok=True)
        
        self.logger.info("AlpacaDataClient initialized")
    
    async def get_historical_bars(self, symbols, start_date, end_date=None, timeframe=TimeFrame.Minute):
        """
        Fetch historical bar data for specified symbols.
        
        Args:
            symbols (list): List of cryptocurrency symbols
            start_date (datetime): Start date for historical data
            end_date (datetime, optional): End date for historical data (defaults to now)
            timeframe (TimeFrame): Timeframe for the bars
            
        Returns:
            dict: Historical bar data
        """
        try:
            if end_date is None:
                end_date = datetime.now()
            
            self.logger.info(f"Fetching historical bars for {symbols} from {start_date} to {end_date}")
            
            request_params = CryptoBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            
            bars = self.client.get_crypto_bars(request_params)
            
            # Check if bars is a BarSet object and handle it appropriately
            if hasattr(bars, 'df'):
                # For newer Alpaca API versions that return a BarSet object
                self.logger.info(f"Fetched bars for {len(bars.df) if not bars.df.empty else 0} timestamps")
            else:
                # For older Alpaca API versions that return a dict
                self.logger.info(f"Fetched bars for {sum(len(bars[symbol].df) for symbol in bars if symbol in bars)}")
            
            return bars
            
        except Exception as e:
            self.logger.error(f"Error fetching historical bars: {e}", exc_info=True)
            return None
    
    async def download_historical_data(self, symbols, days_back=30, timeframe=TimeFrame.Minute):
        """
        Download and save historical data for specified symbols.
        
        Args:
            symbols (list): List of cryptocurrency symbols
            days_back (int): Number of days of historical data to fetch
            timeframe (TimeFrame): Timeframe for the bars
            
        Returns:
            bool: Success status
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            self.logger.info(f"Downloading {days_back} days of historical data for {symbols}")
            
            bars = await self.get_historical_bars(symbols, start_date, end_date, timeframe)
            
            if bars is None:
                return False
            
            # Handle different return types from Alpaca API
            if hasattr(bars, 'df'):
                # For newer Alpaca API versions that return a BarSet object
                df = bars.df
                
                # If it's a multi-index DataFrame with symbols as the first level
                if isinstance(df.index, pd.MultiIndex) and df.index.names[0] == 'symbol':
                    for symbol in symbols:
                        if symbol in df.index.get_level_values('symbol'):
                            symbol_df = df.xs(symbol, level='symbol')
                            filename = f"data/historical/{symbol.replace('/', '_')}_{timeframe.value}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
                            symbol_df.to_csv(filename)
                            self.logger.info(f"Saved historical data for {symbol} to {filename}")
                else:
                    # If it's a single symbol request
                    symbol = symbols[0] if isinstance(symbols, list) else symbols
                    filename = f"data/historical/{symbol.replace('/', '_')}_{timeframe.value}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
                    df.to_csv(filename)
                    self.logger.info(f"Saved historical data for {symbol} to {filename}")
            else:
                # For older Alpaca API versions that return a dict
                for symbol in symbols:
                    if symbol in bars:
                        symbol_bars = bars[symbol]
                        filename = f"data/historical/{symbol.replace('/', '_')}_{timeframe.value}_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
                        
                        # Convert to DataFrame and save to CSV
                        df = symbol_bars.df
                        df.to_csv(filename)
                        
                        self.logger.info(f"Saved historical data for {symbol} to {filename}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading historical data: {e}", exc_info=True)
            return False

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create data client - will load API keys from config
    data_client = AlpacaDataClient()
    
    # List of symbols to fetch
    SYMBOLS = ["BTC/USD", "ETH/USD"]
    
    # Run the download
    asyncio.run(data_client.download_historical_data(SYMBOLS, days_back=7))
