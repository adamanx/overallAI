import os
import logging
import asyncio
import argparse
from src.config_manager import ConfigManager
from src.crypto_stream import CryptoStreamManager
from src.alpaca_data_client import AlpacaDataClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("main")

async def download_historical_data(config):
    """
    Download historical data for backtesting.
    
    Args:
        config (ConfigManager): Configuration manager
    """
    api_key, api_secret, _ = config.get_alpaca_credentials()
    symbols = config.get_symbols()
    days_back = config.get_int("DATA", "historical_days", fallback=30)
    
    if not api_key or not api_secret:
        logger.error("Alpaca API credentials not configured")
        return False
    
    data_client = AlpacaDataClient(api_key, api_secret)
    success = await data_client.download_historical_data(symbols, days_back)
    
    return success

def start_streaming(config):
    """
    Start real-time data streaming.
    
    Args:
        config (ConfigManager): Configuration manager
    """
    api_key, api_secret, _ = config.get_alpaca_credentials()
    symbols = config.get_symbols()
    
    if not api_key or not api_secret:
        logger.error("Alpaca API credentials not configured")
        return False
    
    stream_manager = CryptoStreamManager(api_key, api_secret, symbols)
    stream_manager.start_streaming()
    
    return True

async def main():
    """
    Main entry point for the application.
    """
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading System')
    parser.add_argument('--config', type=str, default='config/config.ini', help='Path to configuration file')
    parser.add_argument('--download', action='store_true', help='Download historical data')
    parser.add_argument('--stream', action='store_true', help='Start real-time data streaming')
    parser.add_argument('--setup', action='store_true', help='Setup configuration')
    
    args = parser.parse_args()
    
    # Create config manager
    config = ConfigManager(args.config)
    
    # Check if API credentials are already configured
    api_key, api_secret, _ = config.get_alpaca_credentials()
    
    if args.setup or (not api_key or not api_secret):
        # Only prompt for API credentials if explicitly requested with --setup
        # or if credentials are not already configured
        api_key = input("Enter Alpaca API Key: ") if not api_key or args.setup else api_key
        api_secret = input("Enter Alpaca API Secret: ") if not api_secret or args.setup else api_secret
        
        config.set("API", "alpaca_api_key", api_key)
        config.set("API", "alpaca_api_secret", api_secret)
        
        # Prompt for symbols only if setup is explicitly requested
        if args.setup:
            symbols = input("Enter cryptocurrency symbols (comma-separated, e.g., BTC/USD,ETH/USD): ")
            if symbols:
                config.set("DATA", "symbols", symbols)
        
        # Save configuration
        config.save_config()
        logger.info("Configuration saved")
    
    if args.download:
        # Download historical data
        success = await download_historical_data(config)
        if success:
            logger.info("Historical data download completed successfully")
        else:
            logger.error("Historical data download failed")
    
    if args.stream:
        # Start real-time data streaming
        success = start_streaming(config)
        if success:
            logger.info("Real-time data streaming started")
        else:
            logger.error("Failed to start real-time data streaming")
    
    if not (args.setup or args.download or args.stream):
        # Show help if no arguments provided
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())