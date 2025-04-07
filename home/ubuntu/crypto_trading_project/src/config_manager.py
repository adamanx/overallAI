import os
import json
import logging
from configparser import ConfigParser

class ConfigManager:
    """
    Manages configuration settings for the cryptocurrency trading system.
    Handles API keys, trading parameters, and system settings.
    """
    
    def __init__(self, config_file=None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_file (str): Path to the configuration file
        """
        self.logger = logging.getLogger("config_manager")
        self.config_file = config_file or "config/config.ini"
        self.config = ConfigParser()
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        # Load configuration if file exists, otherwise create default
        if os.path.exists(self.config_file):
            self.load_config()
        else:
            self.create_default_config()
            self.save_config()
    
    def create_default_config(self):
        """
        Create default configuration settings.
        """
        self.config["API"] = {
            "alpaca_api_key": "",
            "alpaca_api_secret": "",
            "alpaca_base_url": "https://paper-api.alpaca.markets",  # Paper trading by default
        }
        
        self.config["DATA"] = {
            "symbols": "BTC/USD,ETH/USD",
            "data_dir": "data",
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
        }
        
        self.config["TRADING"] = {
            "initial_capital": "10000",
            "max_position_size": "0.1",  # 10% of portfolio
            "stop_loss_pct": "0.02",     # 2% stop loss
            "take_profit_pct": "0.05",   # 5% take profit
        }
        
        self.config["SYSTEM"] = {
            "log_level": "INFO",
            "reconnect_attempts": "5",
            "backoff_factor": "2",
        }
        
        self.logger.info("Created default configuration")
    
    def load_config(self):
        """
        Load configuration from file.
        """
        try:
            self.config.read(self.config_file)
            self.logger.info(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}", exc_info=True)
            self.create_default_config()
    
    def save_config(self):
        """
        Save configuration to file.
        """
        try:
            with open(self.config_file, 'w') as f:
                self.config.write(f)
            self.logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}", exc_info=True)
    
    def get(self, section, key, fallback=None):
        """
        Get a configuration value.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            fallback: Default value if key doesn't exist
            
        Returns:
            The configuration value
        """
        return self.config.get(section, key, fallback=fallback)
    
    def get_int(self, section, key, fallback=None):
        """
        Get a configuration value as integer.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            fallback: Default value if key doesn't exist
            
        Returns:
            The configuration value as integer
        """
        return self.config.getint(section, key, fallback=fallback)
    
    def get_float(self, section, key, fallback=None):
        """
        Get a configuration value as float.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            fallback: Default value if key doesn't exist
            
        Returns:
            The configuration value as float
        """
        return self.config.getfloat(section, key, fallback=fallback)
    
    def get_boolean(self, section, key, fallback=None):
        """
        Get a configuration value as boolean.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            fallback: Default value if key doesn't exist
            
        Returns:
            The configuration value as boolean
        """
        return self.config.getboolean(section, key, fallback=fallback)
    
    def get_list(self, section, key, fallback=None):
        """
        Get a configuration value as list (comma-separated).
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            fallback: Default value if key doesn't exist
            
        Returns:
            The configuration value as list
        """
        value = self.get(section, key, fallback=fallback)
        if value:
            return [item.strip() for item in value.split(',')]
        return []
    
    def set(self, section, key, value):
        """
        Set a configuration value.
        
        Args:
            section (str): Configuration section
            key (str): Configuration key
            value: Value to set
        """
        if not self.config.has_section(section):
            self.config.add_section(section)
        
        self.config.set(section, key, str(value))
    
    def get_alpaca_credentials(self):
        """
        Get Alpaca API credentials.
        
        Returns:
            tuple: (api_key, api_secret, base_url)
        """
        api_key = self.get("API", "alpaca_api_key")
        api_secret = self.get("API", "alpaca_api_secret")
        base_url = self.get("API", "alpaca_base_url")
        
        return api_key, api_secret, base_url
    
    def get_symbols(self):
        """
        Get list of cryptocurrency symbols to trade.
        
        Returns:
            list: List of symbols
        """
        return self.get_list("DATA", "symbols")
    
    def get_trading_parameters(self):
        """
        Get trading parameters.
        
        Returns:
            dict: Trading parameters
        """
        return {
            "initial_capital": self.get_float("TRADING", "initial_capital"),
            "max_position_size": self.get_float("TRADING", "max_position_size"),
            "stop_loss_pct": self.get_float("TRADING", "stop_loss_pct"),
            "take_profit_pct": self.get_float("TRADING", "take_profit_pct"),
        }

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create config manager
    config_manager = ConfigManager()
    
    # Set API credentials (in a real application, these would be securely provided)
    config_manager.set("API", "alpaca_api_key", "your_api_key")
    config_manager.set("API", "alpaca_api_secret", "your_api_secret")
    
    # Save configuration
    config_manager.save_config()
    
    # Get configuration values
    api_key, api_secret, base_url = config_manager.get_alpaca_credentials()
    symbols = config_manager.get_symbols()
    trading_params = config_manager.get_trading_parameters()
    
    print(f"API Key: {api_key}")
    print(f"Symbols: {symbols}")
    print(f"Trading Parameters: {trading_params}")
