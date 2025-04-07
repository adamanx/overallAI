import logging
import asyncio
from alpaca.data import CryptoDataStream
from src.config_manager import ConfigManager

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

async def main():
    # Get credentials from config
    config = ConfigManager()
    api_key, api_secret, _ = config.get_alpaca_credentials()
    
    logger.debug(f"Using API key: {api_key[:4]}...")
    
    # Create crypto stream
    crypto_stream = CryptoDataStream(api_key, api_secret)
    
    async def bar_callback(bar):
        logger.info(f"Received bar: {bar}")
    
    # Subscribe to BTC/USD bars
    logger.debug("Subscribing to BTC/USD bars...")
    crypto_stream.subscribe_bars(bar_callback, "BTC/USD")
    
    # Run the stream
    logger.debug("Starting stream...")
    try:
        await crypto_stream.run()
    except Exception as e:
        logger.error(f"Error running stream: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
