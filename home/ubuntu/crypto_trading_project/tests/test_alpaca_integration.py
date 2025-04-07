import unittest
import os
import sys
import logging
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config_manager import ConfigManager
from src.crypto_stream import CryptoStreamManager
from src.alpaca_data_client import AlpacaDataClient

# Disable logging during tests
logging.disable(logging.CRITICAL)

class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Use a test config file
        self.test_config_file = "test_config.ini"
        
        # Remove test config file if it exists
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test config file
        if os.path.exists(self.test_config_file):
            os.remove(self.test_config_file)
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        config = ConfigManager(self.test_config_file)
        
        # Check if default sections exist
        self.assertTrue(config.config.has_section("API"))
        self.assertTrue(config.config.has_section("DATA"))
        self.assertTrue(config.config.has_section("TRADING"))
        self.assertTrue(config.config.has_section("SYSTEM"))
        
        # Check if default values exist
        self.assertEqual(config.get("DATA", "symbols"), "BTC/USD,ETH/USD")
        self.assertEqual(config.get_float("TRADING", "initial_capital"), 10000.0)
    
    def test_get_set_config(self):
        """Test getting and setting configuration values."""
        config = ConfigManager(self.test_config_file)
        
        # Set values
        config.set("API", "alpaca_api_key", "test_key")
        config.set("API", "alpaca_api_secret", "test_secret")
        config.set("DATA", "symbols", "BTC/USD,ETH/USD,SOL/USD")
        
        # Save and reload
        config.save_config()
        config = ConfigManager(self.test_config_file)
        
        # Check values
        self.assertEqual(config.get("API", "alpaca_api_key"), "test_key")
        self.assertEqual(config.get("API", "alpaca_api_secret"), "test_secret")
        self.assertEqual(config.get("DATA", "symbols"), "BTC/USD,ETH/USD,SOL/USD")
    
    def test_get_list(self):
        """Test getting list values."""
        config = ConfigManager(self.test_config_file)
        
        # Set list value
        config.set("DATA", "symbols", "BTC/USD,ETH/USD,SOL/USD")
        
        # Get list
        symbols = config.get_list("DATA", "symbols")
        
        # Check list
        self.assertEqual(symbols, ["BTC/USD", "ETH/USD", "SOL/USD"])
    
    def test_get_alpaca_credentials(self):
        """Test getting Alpaca credentials."""
        config = ConfigManager(self.test_config_file)
        
        # Set credentials
        config.set("API", "alpaca_api_key", "test_key")
        config.set("API", "alpaca_api_secret", "test_secret")
        config.set("API", "alpaca_base_url", "test_url")
        
        # Get credentials
        api_key, api_secret, base_url = config.get_alpaca_credentials()
        
        # Check credentials
        self.assertEqual(api_key, "test_key")
        self.assertEqual(api_secret, "test_secret")
        self.assertEqual(base_url, "test_url")

class TestCryptoStreamManager(unittest.TestCase):
    """Test cases for CryptoStreamManager class."""
    
    @patch('src.crypto_stream.CryptoDataStream')
    def test_initialization(self, mock_crypto_stream):
        """Test initialization of CryptoStreamManager."""
        # Create manager
        manager = CryptoStreamManager("test_key", "test_secret", ["BTC/USD", "ETH/USD"])
        
        # Check if CryptoDataStream was initialized with correct parameters
        mock_crypto_stream.assert_called_once_with("test_key", "test_secret")
        
        # Check if symbols were set correctly
        self.assertEqual(manager.symbols, ["BTC/USD", "ETH/USD"])
    
    @patch('src.crypto_stream.CryptoDataStream')
    def test_start_streaming(self, mock_crypto_stream):
        """Test starting streaming."""
        # Create mock instance
        mock_instance = MagicMock()
        mock_crypto_stream.return_value = mock_instance
        
        # Create manager
        manager = CryptoStreamManager("test_key", "test_secret", ["BTC/USD", "ETH/USD"])
        
        # Start streaming
        manager.start_streaming()
        
        # Check if subscribe methods were called
        mock_instance.subscribe_bars.assert_called_once()
        mock_instance.subscribe_trades.assert_called_once()
        mock_instance.subscribe_quotes.assert_called_once()
        
        # Check if run method was called
        mock_instance.run.assert_called_once()

class TestAlpacaDataClient(unittest.TestCase):
    """Test cases for AlpacaDataClient class."""
    
    @patch('src.alpaca_data_client.CryptoHistoricalDataClient')
    def test_initialization(self, mock_historical_client):
        """Test initialization of AlpacaDataClient."""
        # Create client
        client = AlpacaDataClient("test_key", "test_secret")
        
        # Check if CryptoHistoricalDataClient was initialized with correct parameters
        mock_historical_client.assert_called_once_with("test_key", "test_secret")

if __name__ == '__main__':
    unittest.main()
