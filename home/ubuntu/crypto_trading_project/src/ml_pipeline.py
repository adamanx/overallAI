import os
import logging
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Dict, List, Union, Optional, Tuple, Any

# Machine learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

class MLPipeline:
    """
    Machine Learning Pipeline for cryptocurrency trading.
    Handles data preparation, model training, evaluation, and prediction.
    """
    
    def __init__(self):
        """Initialize the MLPipeline class."""
        self.logger = logging.getLogger("ml_pipeline")
        
        # Create directories if they don't exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("models/trained", exist_ok=True)
        os.makedirs("models/metadata", exist_ok=True)
        os.makedirs("models/predictions", exist_ok=True)
        
        # Initialize model registry
        self.model_registry = {}
        self._load_model_registry()
        
        self.logger.info("MLPipeline initialized")
    
    def _load_model_registry(self):
        """Load model registry from file."""
        try:
            registry_file = "models/model_registry.json"
            if os.path.exists(registry_file):
                with open(registry_file, 'r') as f:
                    self.model_registry = json.load(f)
                self.logger.info(f"Loaded model registry with {len(self.model_registry)} entries")
            else:
                self.model_registry = {}
                self.logger.info("Created new model registry")
        except Exception as e:
            self.logger.error(f"Error loading model registry: {e}", exc_info=True)
            self.model_registry = {}
    
    def _save_model_registry(self):
        """Save model registry to file."""
        try:
            registry_file = "models/model_registry.json"
            with open(registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=4)
            self.logger.info(f"Saved model registry with {len(self.model_registry)} entries")
        except Exception as e:
            self.logger.error(f"Error saving model registry: {e}", exc_info=True)
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                    time_series_split: bool = True, n_splits: int = 5) -> Dict:
        """
        Prepare data for machine learning.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            time_series_split: Whether to use time series cross-validation
            n_splits: Number of splits for time series cross-validation
            
        Returns:
            Dict with prepared data
        """
        try:
            # Handle time series data
            if time_series_split:
                # Use the last test_size portion for final testing
                train_size = int(len(X) * (1 - test_size))
                X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
                y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
                
                # Create time series cross-validation splits for training
                tscv = TimeSeriesSplit(n_splits=n_splits)
                cv_splits = []
                
                for train_idx, val_idx in tscv.split(X_train):
                    cv_splits.append((train_idx, val_idx))
                
                self.logger.info(f"Prepared data with time series split: {len(X_train)} train, {len(X_test)} test samples")
                
                return {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'cv_splits': cv_splits
                }
            else:
                # Regular train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, shuffle=False
                )
                
                self.logger.info(f"Prepared data with regular split: {len(X_train)} train, {len(X_test)} test samples")
                
                return {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test
                }
                
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}", exc_info=True)
            return {}
    
    def get_model(self, model_type: str, task: str, params: Optional[Dict] = None) -> Any:
        """
        Get a model instance based on type and task.
        
        Args:
            model_type: Type of model ('lr', 'rf', 'gb', 'svm', 'xgb')
            task: Type of task ('classification', 'regression')
            params: Model parameters
            
        Returns:
            Model instance
        """
        try:
            # Default parameters if none provided
            if params is None:
                params = {}
            
            # Classification models
            if task == 'classification':
                if model_type == 'lr':
                    return LogisticRegression(**params)
                elif model_type == 'rf':
                    return RandomForestClassifier(**params)
                elif model_type == 'gb':
                    return GradientBoostingClassifier(**params)
                elif model_type == 'svm':
                    return SVC(**params)
                elif model_type == 'xgb':
                    return XGBClassifier(**params)
                else:
                    self.logger.warning(f"Unknown model type: {model_type}, using RandomForestClassifier")
                    return RandomForestClassifier()
            
            # Regression models
            elif task == 'regression':
                if model_type == 'lr':
                    return LinearRegression(**params)
                elif model_type == 'rf':
                    return RandomForestRegressor(**params)
                elif model_type == 'gb':
                    return GradientBoostingRegressor(**params)
                elif model_type == 'svm':
                    return SVR(**params)
                elif model_type == 'xgb':
                    return XGBRegressor(**params)
                else:
                    self.logger.warning(f"Unknown model type: {model_type}, using RandomForestRegressor")
                    return RandomForestRegressor()
            
            else:
                self.logger.warning(f"Unknown task: {task}, using classification")
                return RandomForestClassifier()
                
        except Exception as e:
            self.logger.error(f"Error getting model: {e}", exc_info=True)
            # Return a default model
            if task == 'classification':
                return RandomForestClassifier()
            else:
                return RandomForestRegressor()
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   model_type: str, task: str, params: Optional[Dict] = None,
                   cv_splits: Optional[List] = None, grid_search: bool = False,
                   param_grid: Optional[Dict] = None) -> Dict:
        """
        Train a machine learning model.
        
        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model ('lr', 'rf', 'gb', 'svm', 'xgb')
            task: Type of task ('classification', 'regression')
            params: Model parameters
            cv_splits: Cross-validation splits
            grid_search: Whether to use grid search for hyperparameter tuning
            param_grid: Parameter grid for grid search
            
        Returns:
            Dict with trained model and metadata
        """
        try:
            self.logger.info(f"Training {model_type} model for {task}")
            
            # Get model instance
            model = self.get_model(model_type, task, params)
            
            # Perform grid search if requested
            if grid_search and param_grid is not None:
                self.logger.info(f"Performing grid search with {len(param_grid)} parameters")
                
                # Create grid search
                if cv_splits is not None:
                    # Use custom CV splits
                    grid = GridSearchCV(model, param_grid, cv=cv_splits, scoring='f1' if task == 'classification' else 'neg_mean_squared_error')
                else:
                    # Use default CV
                    grid = GridSearchCV(model, param_grid, scoring='f1' if task == 'classification' else 'neg_mean_squared_error')
                
                # Fit grid search
                grid.fit(X_train, y_train)
                
                # Get best model
                model = grid.best_estimator_
                best_params = grid.best_params_
                best_score = grid.best_score_
                
                self.logger.info(f"Grid search complete. Best score: {best_score}, Best params: {best_params}")
            else:
                # Train model directly
                model.fit(X_train, y_train)
                best_params = params
                best_score = None
            
            # Get feature importances if available
            feature_importances = None
            if hasattr(model, 'feature_importances_'):
                feature_importances = dict(zip(X_train.columns, model.feature_importances_))
                # Sort by importance
                feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
            
            # Create model metadata
            metadata = {
                'model_type': model_type,
                'task': task,
                'params': best_params,
                'feature_names': list(X_train.columns),
                'feature_importances': feature_importances,
                'training_samples': len(X_train),
                'grid_search': grid_search,
                'best_score': best_score,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return {
                'model': model,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}", exc_info=True)
            return {}
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      task: str) -> Dict:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            task: Type of task ('classification', 'regression')
            
        Returns:
            Dict with evaluation metrics
        """
        try:
            self.logger.info(f"Evaluating model for {task}")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics based on task
            if task == 'classification':
                # Classification metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='binary'),
                    'recall': recall_score(y_test, y_pred, average='binary'),
                    'f1': f1_score(y_test, y_pred, average='binary')
                }
                
                # Add probability predictions if available
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                    # Add probabilities to results
                    metrics['probabilities'] = y_prob.tolist()
            else:
                # Regression metrics
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
            
            # Add predictions to results
            metrics['predictions'] = y_pred.tolist()
            metrics['actual'] = y_test.tolist()
            
            self.logger.info(f"Evaluation complete: {metrics}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}", exc_info=True)
            return {}
    
    def save_model(self, model: Any, metadata: Dict, metrics: Dict, 
                  symbol: str, timeframe: str) -> str:
        """
        Save a trained model and its metadata.
        
        Args:
            model: Trained model
            metadata: Model metadata
            metrics: Evaluation metrics
            symbol: Cryptocurrency symbol
            timeframe: Timeframe of the data
            
        Returns:
            Model ID
        """
        try:
            # Generate model ID
            model_id = f"{symbol.replace('/', '_')}_{timeframe}_{metadata['model_type']}_{metadata['task']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save model
            model_path = f"models/trained/{model_id}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Combine metadata and metrics
            full_metadata = {
                **metadata,
                'metrics': metrics,
                'model_id': model_id,
                'model_path': model_path,
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            # Save metadata
            metadata_path = f"models/metadata/{model_id}.json"
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=4)
            
            # Update model registry
            self.model_registry[model_id] = {
                'model_id': model_id,
                'symbol': symbol,
                'timeframe': timeframe,
                'model_type': metadata['model_type'],
                'task': metadata['task'],
                'metrics': {k: v for k, v in metrics.items() if k not in ['predictions', 'actual', 'probabilities']},
                'model_path': model_path,
                'metadata_path': metadata_path,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save model registry
            self._save_model_registry()
            
            self.logger.info(f"Saved model {model_id}")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}", exc_info=True)
            return ""
    
    def load_model(self, model_id: str) -> Dict:
        """
        Load a trained model and its metadata.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dict with model and metadata
        """
        try:
            # Check if model exists in registry
            if model_id not in self.model_registry:
                self.logger.<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>