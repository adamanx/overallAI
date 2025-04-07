import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from src.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.ml_pipeline import MLPipeline
from src.data_aggregator import CRYPTO_PAIRS, TIMEFRAMES

class MLModelManager:
    """
    Manager for machine learning models in the cryptocurrency trading system.
    Integrates feature engineering with model training and evaluation.
    """
    
    def __init__(self):
        """Initialize the MLModelManager class."""
        self.logger = logging.getLogger("ml_model_manager")
        
        # Initialize feature engineering pipeline
        self.feature_pipeline = FeatureEngineeringPipeline()
        
        # Initialize ML pipeline
        self.ml_pipeline = MLPipeline()
        
        # Create directories if they don't exist
        os.makedirs("models/results", exist_ok=True)
        
        self.logger.info("MLModelManager initialized")
    
    def train_models_for_symbol(self, symbol, days_back=30, timeframe='1Min',
                               include_indicators=None, train_spike_models=True):
        """
        Train models for a specific cryptocurrency symbol.
        
        Args:
            symbol (str): Cryptocurrency symbol
            days_back (int): Number of days of historical data to use
            timeframe (str): Timeframe of the data
            include_indicators (list): List of indicators to include
            train_spike_models (bool): Whether to train spike detection models
            
        Returns:
            Dict with training results
        """
        try:
            self.logger.info(f"Training models for {symbol} with {days_back} days of data")
            
            # Calculate time range
            end_timestamp = int(datetime.now().timestamp() * 1000)
            start_timestamp = end_timestamp - (days_back * 24 * 60 * 60 * 1000)
            
            # Generate features
            features_df = self.feature_pipeline.process_symbol(
                symbol, start_timestamp, end_timestamp, timeframe,
                include_indicators, normalize=False, save_to_file=True
            )
            
            if features_df.empty:
                self.logger.warning(f"No features generated for {symbol}")
                return {}
            
            # Create datasets for different prediction targets
            datasets = {}
            
            # Price direction prediction (classification)
            features_df['target_direction'] = np.where(
                features_df['close'].shift(-1) > features_df['close'], 1, 0
            )
            
            # Price change prediction (regression)
            features_df['target_change'] = features_df['close'].pct_change(periods=-1) * 100
            
            # Volatility prediction (regression)
            if 'atr_14' in features_df.columns:
                features_df['target_volatility'] = features_df['atr_14'].shift(-1)
            
            # Drop rows with NaN values
            features_df = features_df.dropna()
            
            if features_df.empty:
                self.logger.warning(f"No data after creating targets for {symbol}")
                return {}
            
            # Exclude price and target columns from features
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
            exclude_cols += ['target_direction', 'target_change', 'target_volatility']
            
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            # Create datasets
            X = features_df[feature_cols]
            y_direction = features_df['target_direction']
            y_change = features_df['target_change']
            
            # Prepare data for each target
            data_direction = self.ml_pipeline.prepare_data(X, y_direction, test_size=0.2, time_series_split=True)
            data_change = self.ml_pipeline.prepare_data(X, y_change, test_size=0.2, time_series_split=True)
            
            # Train models for price direction prediction
            direction_results = self.ml_pipeline.train_multiple_models(
                data_direction['X_train'], data_direction['y_train'],
                data_direction['X_test'], data_direction['y_test'],
                'classification', f"{symbol}_direction", timeframe
            )
            
            # Train models for price change prediction
            change_results = self.ml_pipeline.train_multiple_models(
                data_change['X_train'], data_change['y_train'],
                data_change['X_test'], data_change['y_test'],
                'regression', f"{symbol}_change", timeframe
            )
            
            # Train spike detection models if requested
            spike_results = {}
            if train_spike_models:
                spike_results = self.ml_pipeline.train_spike_detection_model(
                    features_df, symbol=symbol, timeframe=timeframe
                )
            
            # Combine results
            results = {
                'direction': direction_results,
                'change': change_results,
                'spike': spike_results,
                'feature_cols': feature_cols,
                'training_samples': len(features_df)
            }
            
            # Save results
            results_file = f"models/results/{symbol.replace('/', '_')}_{timeframe}_results.json"
            
            # Convert results to JSON-serializable format
            json_results = {
                'direction': {
                    'best_model_type': direction_results.get('best_model_type', ''),
                    'best_model_id': direction_results.get('best_model_id', '')
                },
                'change': {
                    'best_model_type': change_results.get('best_model_type', ''),
                    'best_model_id': change_results.get('best_model_id', '')
                },
                'spike': {
                    'binary_best_model_id': spike_results.get('binary_results', {}).get('best_model_id', ''),
                    'multi_best_model_id': spike_results.get('multi_results', {}).get('best_model_id', '')
                },
                'feature_cols': feature_cols,
                'training_samples': len(features_df),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            import json
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=4)
            
            self.logger.info(f"Saved training results to {results_file}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training models for {symbol}: {e}", exc_info=True)
            return {}
    
    def train_models_for_all_symbols(self, symbols=None, days_back=30, timeframe='1Min',
                                    include_indicators=None, train_spike_models=True):
        """
        Train models for all specified cryptocurrency symbols.
        
        Args:
            symbols (list): List of cryptocurrency symbols
            days_back (int): Number of days of historical data to use
            timeframe (str): Timeframe of the data
            include_indicators (list): List of indicators to include
            train_spike_models (bool): Whether to train spike detection models
            
        Returns:
            Dict with training results for all symbols
        """
        try:
            # Use default symbols if not provided
            if symbols is None:
                symbols = CRYPTO_PAIRS
            
            self.logger.info(f"Training models for {len(symbols)} symbols")
            
            # Train models for each symbol
            results = {}
            
            for symbol in symbols:
                symbol_results = self.train_models_for_symbol(
                    symbol, days_back, timeframe, include_indicators, train_spike_models
                )
                
                if symbol_results:
                    results[symbol] = symbol_results
            
            self.logger.info(f"Completed training for {len(results)} symbols")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training models for all symbols: {e}", exc_info=True)
            return {}
    
    def get_best_models(self, symbol, timeframe='1Min'):
        """
        Get the best models for a specific cryptocurrency symbol.
        
        Args:
            symbol (str): Cryptocurrency symbol
            timeframe (str): Timeframe of the data
            
        Returns:
            Dict with best models
        """
        try:
            self.logger.info(f"Getting best models for {symbol} {timeframe}")
            
            # Get best model for price direction prediction
            direction_model_id = self.ml_pipeline.get_best_model(
                f"{symbol}_direction", timeframe, 'classification', 'f1'
            )
            
            # Get best model for price change prediction
            change_model_id = self.ml_pipeline.get_best_model(
                f"{symbol}_change", timeframe, 'regression', 'rmse'
            )
            
            # Get best model for spike detection (binary)
            spike_binary_model_id = self.ml_pipeline.get_best_model(
                f"{symbol}_binary", timeframe, 'classification', 'f1'
            )
            
            # Get best model for spike detection (multi-class)
            spike_multi_model_id = self.ml_pipeline.get_best_model(
                f"{symbol}_multi", timeframe, 'classification', 'f1'
            )
            
            # Load models
            models = {}
            
            if direction_model_id:
                direction_model = self.ml_pipeline.load_model(direction_model_id)
                if direction_model:
                    models['direction'] = direction_model
            
            if change_model_id:
                change_model = self.ml_pipeline.load_model(change_model_id)
                if change_model:
                    models['change'] = change_model
            
            if spike_binary_model_id:
                spike_binary_model = self.ml_pipeline.load_model(spike_binary_model_id)
                if spike_binary_model:
                    models['spike_binary'] = spike_binary_model
            
            if spike_multi_model_id:
                spike_multi_model = self.ml_pipeline.load_model(spike_multi_model_id)
                if spike_multi_model:
                    models['spike_multi'] = spike_multi_model
            
            self.logger.info(f"Loaded {len(models)} models for {symbol} {timeframe}")
            
            return models
            
        except Exception as e:
            self.logger.error(f"Error getting best models for {symbol}: {e}", exc_info=True)
            return {}
    
    def predict_with_models(self, features_df, models):
        """
        Make predictions with trained models.
        
        Args:
            features_df (DataFrame): DataFrame with features
            models (dict): Dict with loaded models
            
        Returns:
            Dict with predictions
        """
        try:
            self.logger.info(f"Making predictions with {len(models)} models")
            
            # Prepare features
            # Exclude price columns from features
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            X = features_df[feature_cols]
            
            # Make predictions with each model
            predictions = {}
            
            if 'direction' in models:
                direction_model = models['direction']['model']
                direction_metadata = models['direction']['metadata']
                
                # Check if feature columns match
                model_features = direction_metadata.get('feature_names', [])
                common_features = [col for col in model_features if col in X.columns]
                
                if len(common_features) == len(model_features):
                    # All features are available
                    direction_pred = self.ml_pipeline.predict(
                        direction_model, X[model_features], 'classification'
                    )
                    predictions['direction'] = direction_pred
                else:
                    self.logger.warning(f"Feature mismatch for direction model: {len(common_features)}/{len(model_features)} features available")
            
            if 'change' in models:
                change_model = models['change']['model']
                change_metadata = models['change']['metadata']
                
                # Check if feature columns match
                model_features = change_metadata.get('feature_names', [])
                common_features = [col for col in model_features if col in X.columns]
                
                if len(common_features) == len(model_features):
                    # All features are available
                    change_pred = self.ml_pipeline.predict(
                        change_model, X[model_features], 'regression'
                    )
                    predictions['change'] = change_pred
                else:
                    self.logger.warning(f"Feature mismatch for change model: {len(common_features)}/{len(model_features)} features available")
            
            if 'spike_binary' in models:
                spike_binary_model = models['spike_binary']['model']
                spike_binary_metadata = models['spike_binary']['metadata']
                
                # Check if feature columns match
                model_features = spike_binary_metadata.get('feature_names', [])
                common_features = [col for col in model_features if col in X.columns]
                
                if len(common_features) == len(model_features):
                    # All features are available
                    spike_binary_pred = self.ml_pipeline.predict(
                        spike_binary_model, X[model_features], 'classification'
                    )
                    predictions['spike_binary'] = spike_binary_pred
                else:
                    self.logger.warning(f"Feature mismatch for spike_binary model: {len(common_features)}/{len(model_features)} features available")
            
            if 'spike_multi' in models:
                spike_multi_model = models['spike_multi']['model']
                spike_multi_metadata = models['spike_multi']['metadata']
                
                # Check if feature columns match
                model_features = spike_multi_metadata.get('feature_names', [])
                common_features = [col for col in model_features if col in X.columns]
                
                if len(common_features) == len(model_features):
                    # All features are available
                    spike_multi_pred = self.ml_pipeline.predict(
                        spike_multi_model, X[model_features], 'classification'
                    )
                    predictions['spike_multi'] = spike_multi_pred
                else:
                    self.logger.warning(f"Feature mismatch for spike_multi model: {len(common_features)}/{len(model_features)} features available")
            
            self.logger.info(f"Made predictions with {len(predictions)} models")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}", exc_info=True)
  <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>