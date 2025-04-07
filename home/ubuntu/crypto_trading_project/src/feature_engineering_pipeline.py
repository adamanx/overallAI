import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from src.time_series_db import TimeSeriesDatabase
from src.technical_indicators import TechnicalIndicators
from src.feature_generator import FeatureGenerator
from src.data_aggregator import CRYPTO_PAIRS, TIMEFRAMES

class FeatureEngineeringPipeline:
    """
    Pipeline for feature engineering on cryptocurrency market data.
    Integrates technical indicators and feature generation with data storage.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the FeatureEngineeringPipeline.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.logger = logging.getLogger("feature_engineering_pipeline")
        
        # Initialize database
        self.db = TimeSeriesDatabase(db_path)
        
        # Initialize technical indicators
        self.indicators = TechnicalIndicators()
        
        # Initialize feature generator
        self.feature_gen = FeatureGenerator()
        
        # Create features directory if it doesn't exist
        os.makedirs("data/features", exist_ok=True)
        
        self.logger.info("FeatureEngineeringPipeline initialized")
    
    def process_symbol(self, symbol, start_timestamp, end_timestamp, timeframe='1Min',
                      include_indicators=None, normalize=True, save_to_file=True):
        """
        Process a single symbol to generate features.
        
        Args:
            symbol (str): Cryptocurrency symbol
            start_timestamp (int): Start timestamp in milliseconds
            end_timestamp (int): End timestamp in milliseconds
            timeframe (str): Timeframe of the data
            include_indicators (list): List of indicators to include
            normalize (bool): Whether to normalize features
            save_to_file (bool): Whether to save features to file
            
        Returns:
            DataFrame with generated features
        """
        try:
            self.logger.info(f"Processing {symbol} for feature engineering")
            
            # Get OHLCV data from database
            df = self.db.get_bars(symbol, start_timestamp, end_timestamp, timeframe)
            
            if df.empty:
                self.logger.warning(f"No data found for {symbol}")
                return df
            
            # Generate features
            features_df = self.feature_gen.generate_features(df, include_indicators=include_indicators)
            
            # Normalize features if requested
            if normalize:
                # Exclude OHLCV columns from normalization
                exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
                
                # Normalize features
                normalized_df, scalers = self.feature_gen.normalize_features(
                    features_df, method='standard', exclude_columns=exclude_columns
                )
                
                features_df = normalized_df
            
            # Save to file if requested
            if save_to_file:
                # Generate filename
                start_date = datetime.fromtimestamp(start_timestamp / 1000).strftime('%Y%m%d')
                end_date = datetime.fromtimestamp(end_timestamp / 1000).strftime('%Y%m%d')
                filename = f"data/features/{symbol.replace('/', '_')}_{timeframe}_{start_date}_{end_date}.csv"
                
                # Save to CSV
                features_df.to_csv(filename)
                self.logger.info(f"Saved features to {filename}")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error processing symbol: {e}", exc_info=True)
            return pd.DataFrame()
    
    def process_all_symbols(self, symbols=None, days_back=30, timeframe='1Min',
                           include_indicators=None, normalize=True, save_to_file=True):
        """
        Process all symbols to generate features.
        
        Args:
            symbols (list): List of cryptocurrency symbols
            days_back (int): Number of days of historical data to process
            timeframe (str): Timeframe of the data
            include_indicators (list): List of indicators to include
            normalize (bool): Whether to normalize features
            save_to_file (bool): Whether to save features to file
            
        Returns:
            Dict of DataFrames with generated features for each symbol
        """
        try:
            # Use default symbols if not provided
            if symbols is None:
                symbols = CRYPTO_PAIRS
            
            # Calculate time range
            end_timestamp = int(datetime.now().timestamp() * 1000)
            start_timestamp = end_timestamp - (days_back * 24 * 60 * 60 * 1000)
            
            self.logger.info(f"Processing {len(symbols)} symbols for feature engineering")
            
            # Process each symbol
            results = {}
            
            for symbol in symbols:
                features_df = self.process_symbol(
                    symbol, start_timestamp, end_timestamp, timeframe,
                    include_indicators, normalize, save_to_file
                )
                
                if not features_df.empty:
                    results[symbol] = features_df
            
            self.logger.info(f"Processed {len(results)} symbols successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing symbols: {e}", exc_info=True)
            return {}
    
    def create_feature_dataset(self, symbols=None, days_back=30, timeframe='1Min',
                              target_column='close', target_shift=-1, normalize=True):
        """
        Create a dataset for machine learning with features and target.
        
        Args:
            symbols (list): List of cryptocurrency symbols
            days_back (int): Number of days of historical data to process
            timeframe (str): Timeframe of the data
            target_column (str): Column to use as target
            target_shift (int): Number of periods to shift target (negative for future)
            normalize (bool): Whether to normalize features
            
        Returns:
            Dict with X (features) and y (target) for each symbol
        """
        try:
            # Process all symbols
            features_dict = self.process_all_symbols(
                symbols, days_back, timeframe, normalize=normalize, save_to_file=False
            )
            
            # Create datasets for each symbol
            datasets = {}
            
            for symbol, features_df in features_dict.items():
                if features_df.empty:
                    continue
                
                # Create target variable (future price)
                if target_shift < 0:
                    # Negative shift means future values (e.g., -1 is the next period's value)
                    features_df['target'] = features_df[target_column].shift(target_shift)
                else:
                    # Positive shift means past values
                    features_df['target'] = features_df[target_column].shift(target_shift)
                
                # Drop rows with NaN in target
                features_df = features_df.dropna(subset=['target'])
                
                if features_df.empty:
                    continue
                
                # Create target for classification (price direction)
                features_df['target_direction'] = np.where(
                    features_df['target'] > features_df[target_column], 1, 0
                )
                
                # Create target for regression (percent change)
                features_df['target_pct_change'] = (
                    (features_df['target'] - features_df[target_column]) / features_df[target_column] * 100
                )
                
                # Separate features and targets
                X = features_df.drop(['target', 'target_direction', 'target_pct_change'], axis=1)
                y_reg = features_df['target_pct_change']
                y_cls = features_df['target_direction']
                
                datasets[symbol] = {
                    'X': X,
                    'y_regression': y_reg,
                    'y_classification': y_cls,
                    'full_data': features_df
                }
            
            self.logger.info(f"Created datasets for {len(datasets)} symbols")
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error creating feature dataset: {e}", exc_info=True)
            return {}
    
    def generate_spike_detection_features(self, df, price_col='close', volume_col='volume', 
                                         window_sizes=[5, 10, 20], std_threshold=2.0):
        """
        Generate features specifically for spike detection.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Column name for price
            volume_col: Column name for volume
            window_sizes: List of window sizes for rolling calculations
            std_threshold: Threshold for standard deviation to detect spikes
            
        Returns:
            DataFrame with spike detection features
        """
        try:
            result_df = df.copy()
            
            # Calculate price changes
            result_df['price_change'] = result_df[price_col].diff()
            result_df['price_change_pct'] = result_df[price_col].pct_change() * 100
            
            # Calculate volume changes
            result_df['volume_change'] = result_df[volume_col].diff()
            result_df['volume_change_pct'] = result_df[volume_col].pct_change() * 100
            
            # Calculate features for each window size
            for window in window_sizes:
                # Price volatility
                result_df[f'price_std_{window}'] = result_df[price_col].rolling(window=window).std()
                result_df[f'price_zscore_{window}'] = (
                    (result_df[price_col] - result_df[price_col].rolling(window=window).mean()) / 
                    result_df[price_col].rolling(window=window).std()
                )
                
                # Volume volatility
                result_df[f'volume_std_{window}'] = result_df[volume_col].rolling(window=window).std()
                result_df[f'volume_zscore_{window}'] = (
                    (result_df[volume_col] - result_df[volume_col].rolling(window=window).mean()) / 
                    result_df[volume_col].rolling(window=window).std()
                )
                
                # Price change volatility
                result_df[f'price_change_std_{window}'] = result_df['price_change'].rolling(window=window).std()
                result_df[f'price_change_zscore_{window}'] = (
                    (result_df['price_change'] - result_df['price_change'].rolling(window=window).mean()) / 
                    result_df['price_change'].rolling(window=window).std()
                )
                
                # Detect price spikes
                result_df[f'price_spike_{window}'] = np.where(
                    result_df[f'price_zscore_{window}'] > std_threshold, 1,
                    np.where(result_df[f'price_zscore_{window}'] < -std_threshold, -1, 0)
                )
                
                # Detect volume spikes
                result_df[f'volume_spike_{window}'] = np.where(
                    result_df[f'volume_zscore_{window}'] > std_threshold, 1, 0
                )
                
                # Combined price and volume spikes
                result_df[f'combined_spike_{window}'] = np.where(
                    (result_df[f'price_spike_{window}'] != 0) & 
                    (result_df[f'volume_spike_{window}'] == 1),
                    result_df[f'price_spike_{window}'], 0
                )
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating spike detection features: {e}", exc_info=True)
            return df
    
    def detect_spikes(self, symbol, start_timestamp, end_timestamp, timeframe='1Min',
                     window_sizes=[5, 10, 20], std_threshold=2.5, min_pct_change=1.0):
        """
        Detect price spikes in cryptocurrency data.
        
        Args:
            symbol (str): Cryptocurrency symbol
            start_timestamp (int): Start timestamp in milliseconds
            end_timestamp (int): End timestamp in milliseconds
            timeframe (str): Timeframe of the data
            window_sizes (list): List of window sizes for rolling calculations
            std_threshold (float): Threshold for standard deviation to detect spikes
            min_pct_change (float): Minimum percent change to consider as spike
            
        Returns:
            DataFrame with detected spikes
        """
        try:
            self.logger.info(f"Detecting spikes for {symbol}")
            
            # Get OHLCV data from database
            df = self.db.get_bars(symbol, start_timestamp, end_timestamp, timeframe)
            
            if df.empty:
                self.logger.warning(f"No data found for {symbol}")
                return df
            
            # Generate spike detection features
            spike_df = self.generate_spike_detection_features(
                df, window_sizes=window_sizes, std_threshold=std_threshold
            )
            
            # Filter for significant spikes
            spike_results = []
            
            for window in window_sizes:
                # Find positive spikes
                positive_spikes = spike_df[
                    (spike_df[f'price_spike_{window}'] == 1) & 
                    (spike_df['price_change_pct'] > min_pct_change) &
                    (spike_df[f'volume_spike_{window}'] == 1)
                ]
                
                # Find negative spikes
                negative_spikes = spike_df[
                    (spike_df[f'price_spike_{window}'] == -1) & 
                    (spike_df['price_change_pct'] < -min_pct_change) &
                    (spike_df[f'volume_spike_{window}'] == 1)
                ]
                
                # Combine results
                if not positive_spikes.empty:
                    for idx, row in positive_spikes.iterrows():
                        spike_results.append({
                            'timestamp': idx,
                            'symbol': symbol,
                            'price': row['close'],
                            'volume': row['volume'],
                            'price_change_pct': row['price_change_pct'],
                            'volume_change_pct': row['volume_change_pct'],
                            'window': window,
                            'direction': 'positive',
                            'zscore': row[f'price_zscore_{window}']
                        })
                
                if not negative_spikes.empty:
                    for idx, row in negative_spikes.iterrows():
                        spike_results.append({
                            'timestamp': idx,
                            'symbol': symbol,
                            'price': row['close'],
                            'volume': row['volume'],
                            'price_change_pct': row['price_change_pct'],
                            'volume_change_pct': row['volume_change_pct'],
                            'window': window,
                            'direction': 'negative',
                            'zscore': row[f'price_zscore_{window}']
                        })
  <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>