import os
import logging
from alpaca.data import CryptoDataStream
import asyncio
import json
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_stream.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("crypto_stream")

class CryptoStreamManager:
    """
    Manages WebSocket connections to Alpaca's crypto data streams.
    Handles subscription to different data types and processes incoming data.
    """
    
    def __init__(self, api_key, api_secret, symbols=None):
        """
        Initialize the CryptoStreamManager.
        
        Args:
            api_key (str): Alpaca API key
            api_secret (str): Alpaca API secret
            symbols (list): List of cryptocurrency symbols to subscribe to (e.g., ["BTC/USD", "ETH/USD"])
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols or ["BTC/USD", "ETH/USD"]
        self.crypto_stream = CryptoDataStream(api_key, api_secret)
        self.logger = logger
        
        # Create data directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
        self.logger.info(f"CryptoStreamManager initialized with symbols: {self.symbols}")
    
    async def handle_crypto_bars(self, bar):
        """
        Process incoming bar data.
        
        Args:
            bar (dict): Bar data from Alpaca
        """
        try:
            self.logger.info(f"Received bar: {bar['S']} - {bar['t']}")
            
            # Save raw data
            timestamp = datetime.datetime.fromtimestamp(bar['t'] / 1000).strftime('%Y%m%d_%H%M%S')
            filename = f"data/raw/bar_{bar['S'].replace('/', '_')}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(bar, f)
                
            # Process data (to be implemented)
            # await self.process_bar(bar)
            
        except Exception as e:
            self.logger.error(f"Error handling bar: {e}", exc_info=True)
    
    async def handle_crypto_trades(self, trade):
        """
        Process incoming trade data.
        
        Args:
            trade (dict): Trade data from Alpaca
        """
        try:
            self.logger.info(f"Received trade: {trade['S']} - {trade['t']}")
            
            # Save raw data
            timestamp = datetime.datetime.fromtimestamp(trade['t'] / 1000).strftime('%Y%m%d_%H%M%S')
            filename = f"data/raw/trade_{trade['S'].replace('/', '_')}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(trade, f)
                
            # Process data (to be implemented)
            # await self.process_trade(trade)
            
        except Exception as e:
            self.logger.error(f"Error handling trade: {e}", exc_info=True)
    
    async def handle_crypto_quotes(self, quote):
        """
        Process incoming quote data.
        
        Args:
            quote (dict): Quote data from Alpaca
        """
        try:
            self.logger.info(f"Received quote: {quote['S']} - {quote['t']}")
            
            # Save raw data
            timestamp = datetime.datetime.fromtimestamp(quote['t'] / 1000).strftime('%Y%m%d_%H%M%S')
            filename = f"data/raw/quote_{quote['S'].replace('/', '_')}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(quote, f)
                
            # Process data (to be implemented)
            # await self.process_quote(quote)
            
        except Exception as e:
            self.logger.error(f"Error handling quote: {e}", exc_info=True)
    
    def start_streaming(self):
        """
        Start streaming data from Alpaca.
        """
        try:
            self.logger.info("Starting crypto data stream...")
            
            # Subscribe to different data types
            self.crypto_stream.subscribe_bars(self.handle_crypto_bars, *self.symbols)
            self.crypto_stream.subscribe_trades(self.handle_crypto_trades, *self.symbols)
            self.crypto_stream.subscribe_quotes(self.handle_crypto_quotes, *self.symbols)
            
            # Start the WebSocket connection
            self.crypto_stream.run()
            
        except Exception as e:
            self.logger.error(f"Error in streaming: {e}", exc_info=True)
            # Implement reconnection logic
            self.reconnect()
    
    def reconnect(self, max_retries=5, backoff_factor=2):
        """
        Reconnect to Alpaca's WebSocket with exponential backoff.
        
        Args:
            max_retries (int): Maximum number of reconnection attempts
            backoff_factor (int): Factor to increase wait time between retries
        """
        retries = 0
        wait_time = 1  # Initial wait time in seconds
        
        while retries < max_retries:
            try:
                self.logger.info(f"Attempting to reconnect (attempt {retries+1}/{max_retries})...")
                
                # Create a new crypto stream instance
                self.crypto_stream = CryptoDataStream(self.api_key, self.api_secret)
                
                # Resubscribe to data types
                self.crypto_stream.subscribe_bars(self.handle_crypto_bars, *self.symbols)
                self.crypto_stream.subscribe_trades(self.handle_crypto_trades, *self.symbols)
                self.crypto_stream.subscribe_quotes(self.handle_crypto_quotes, *self.symbols)
                
                # Start the WebSocket connection
                self.crypto_stream.run()
                
                self.logger.info("Reconnection successful!")
                return
                
            except Exception as e:
                retries += 1
                self.logger.error(f"Reconnection attempt {retries} failed: {e}", exc_info=True)
                
                if retries < max_retries:
                    wait_time *= backoff_factor
                    self.logger.info(f"Waiting {wait_time} seconds before next attempt...")
                    time.sleep(wait_time)
        
        self.logger.critical(f"Failed to reconnect after {max_retries} attempts. Manual intervention required.")

# Example usage
if __name__ == "__main__":
    # Replace with your Alpaca API credentials
    API_KEY = "your_api_key"
    API_SECRET = "your_api_secret"
    
    # List of symbols to subscribe to
    SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    # Create and start the stream manager
    stream_manager = CryptoStreamManager(API_KEY, API_SECRET, SYMBOLS)
    stream_manager.start_streaming()
