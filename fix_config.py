from src.config_manager import ConfigManager

# Load the current config
config = ConfigManager()

# Get current credentials
api_key, api_secret, _ = config.get_alpaca_credentials()

# Set the correct endpoints
config.set("ENDPOINTS", "alpaca_base_url", "https://paper-api.alpaca.markets")
config.set("ENDPOINTS", "alpaca_data_url", "https://data.alpaca.markets")

# Save the updated config
config.save_config()

print("Config updated with correct Alpaca API endpoints")
