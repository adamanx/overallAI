import os
import logging
import asyncio
from src.config_manager import ConfigManager
from src.crypto_stream import CryptoStreamManager
from src.data_processor import DataProcessor
from src.time_series_db import TimeSeriesDatabase
from src.data_aggregator import DataAggregator, CRYPTO_PAIRS

class DataIntegrationManager:
    """
    Integrates data streaming with processing and storage components.
    Manages the flow of data from Alpaca Markets to the time-series database.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the DataIntegrationManager.
        
        Args:
            config_file (str): Path to the configuration file
        """
        self.logger = logging.getLogger("data_integration")
        
        # Load configuration
        self.config = ConfigManager(config_file)
        
        # Initialize database
        self.db = TimeSeriesDatabase()
        
        # Initialize data processor
        self.processor = DataProcessor(db_path=self.db.db_path)
        
        # Initialize data aggregator
        self.aggregator = DataAggregator(db_instance=self.db)
        
        # Get API credentials
        self.api_key, self.api_secret, _ = self.config.get_alpaca_credentials()
        
        # Get symbols from configuration or use default list
        self.symbols = self.config.get_list("DATA", "symbols")
        if not self.symbols:
            self.symbols = CRYPTO_PAIRS
            # Update configuration with default symbols
            self.config.set("DATA", "symbols", ",".join(CRYPTO_PAIRS))
            self.config.save_config()
        
        self.logger.info(f"DataIntegrationManager initialized with {len(self.symbols)} symbols")
    
    async def initialize_stream_handlers(self):
        """
        Initialize stream handlers with data processor callbacks.
        
        Returns:
            CryptoStreamManager: Configured stream manager
        """
        try:
            if not self.api_key or not self.api_secret:
                self.logger.error("Alpaca API credentials not configured")
                return None
            
            # Create stream manager
            stream_manager = CryptoStreamManager(self.api_key, self.api_secret, self.symbols)
            
            # Override handlers to use data processor
            stream_manager.handle_crypto_bars = self.processor.process_bar
            stream_manager.handle_crypto_trades = self.processor.process_trade
            stream_manager.handle_crypto_quotes = self.processor.process_quote
            
            self.logger.info("Stream handlers initialized with data processor callbacks")
            return stream_manager
            
        except Exception as e:
            self.logger.error(f"Error initializing stream handlers: {e}", exc_info=True)
            return None
    
    async def start_data_integration(self):
        """
        Start the data integration process.
        
        Returns:
            bool: Success status
        """
        try:
            # Initialize stream handlers
            stream_manager = await self.initialize_stream_handlers()
            if not stream_manager:
                return False
            
            # Start streaming
            self.logger.info("Starting data integration process")
            stream_manager.start_streaming()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting data integration: {e}", exc_info=True)
            return False
    
    async def process_historical_data(self, days_back=30):
        """
        Process and aggregate historical data.
        
        Args:
            days_back (int): Number of days of historical data to process
            
        Returns:
            dict: Processing results
        """
        try:
            self.logger.info(f"Processing historical data for the last {days_back} days")
            
            # Process all symbols
            results = self.aggregator.process_all_symbols(self.symbols, days_back)
            
            self.logger.info(f"Historical data processing completed for {len(results)} symbols")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing historical data: {e}", exc_info=True)
            return {}
    
    async def run_scheduled_aggregation(self, interval_hours=6):
        """
        Run scheduled data aggregation at regular intervals.
        
        Args:
            interval_hours (int): Interval between aggregations in hours
            
        Returns:
            None
        """
        try:
            self.logger.info(f"Starting scheduled aggregation every {interval_hours} hours")
            
            while True:
                # Process historical data
                await self.process_historical_data(days_back=7)
                
                # Wait for next interval
                self.logger.info(f"Waiting {interval_hours} hours until next aggregation")
                await asyncio.sleep(interval_hours * 3600)
                
        except Exception as e:
            self.logger.error(f"Error in scheduled aggregation: {e}", exc_info=True)
    
    async def run_integration_services(self):
        """
        Run all integration services.
        
        Returns:
            bool: Success status
        """
        try:
            # Start data streaming
            streaming_task = asyncio.create_task(self.start_data_integration())
            
            # Start scheduled aggregation
            aggregation_task = asyncio.create_task(self.run_scheduled_aggregation())
            
            # Wait for tasks to complete (they should run indefinitely)
            await asyncio.gather(streaming_task, aggregation_task)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error running integration services: {e}", exc_info=True)
            return False

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("data_integration.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create integration manager
    integration_manager = DataIntegrationManager()
    
    # Run integration services
    asyncio.run(integration_manager.run_integration_services())
