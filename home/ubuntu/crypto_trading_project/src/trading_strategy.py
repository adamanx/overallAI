import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Union, Optional, Tuple, Any
from src.backtest_engine import TradingStrategy
from src.ml_model_manager import MLModelManager
from src.feature_engineering_pipeline import FeatureEngineeringPipeline

class RiskManager:
    """
    Risk management system for cryptocurrency trading strategies.
    Handles position sizing, stop-loss, take-profit, and risk allocation.
    """
    
    def __init__(self):
        """Initialize the RiskManager."""
        self.logger = logging.getLogger("risk_manager")
        self.logger.info("RiskManager initialized")
    
    def calculate_position_size(self, capital: float, risk_per_trade: float, 
                              entry_price: float, stop_price: float) -> float:
        """
        Calculate position size based on risk per trade.
        
        Args:
            capital: Available capital
            risk_per_trade: Percentage of capital to risk per trade
            entry_price: Entry price
            stop_price: Stop-loss price
            
        Returns:
            Position size
        """
        try:
            # Calculate risk amount in currency
            risk_amount = capital * risk_per_trade
            
            # Calculate price difference
            price_diff = abs(entry_price - stop_price)
            
            # Calculate position size
            if price_diff > 0:
                position_size = risk_amount / price_diff
            else:
                position_size = 0
                self.logger.warning("Invalid price difference for position sizing")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}", exc_info=True)
            return 0
    
    def calculate_kelly_position_size(self, capital: float, win_rate: float, 
                                    win_loss_ratio: float, max_risk: float = 0.2) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            capital: Available capital
            win_rate: Probability of winning
            win_loss_ratio: Ratio of average win to average loss
            max_risk: Maximum risk percentage
            
        Returns:
            Position size as percentage of capital
        """
        try:
            # Calculate Kelly percentage
            kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
            
            # Apply maximum risk constraint
            position_pct = min(kelly_pct, max_risk)
            
            # Ensure position percentage is positive
            position_pct = max(position_pct, 0)
            
            # Calculate position size
            position_size = capital * position_pct
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size: {e}", exc_info=True)
            return 0
    
    def calculate_volatility_adjusted_stop(self, price: float, atr: float, 
                                         multiplier: float = 2.0) -> float:
        """
        Calculate volatility-adjusted stop-loss price.
        
        Args:
            price: Current price
            atr: Average True Range
            multiplier: ATR multiplier
            
        Returns:
            Stop-loss price
        """
        try:
            # Calculate stop distance
            stop_distance = atr * multiplier
            
            # Calculate stop price
            stop_price = price - stop_distance
            
            return stop_price
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility-adjusted stop: {e}", exc_info=True)
            return price * 0.95  # Default to 5% below price
    
    def calculate_risk_reward_ratio(self, entry_price: float, stop_price: float, 
                                  target_price: float) -> float:
        """
        Calculate risk-reward ratio.
        
        Args:
            entry_price: Entry price
            stop_price: Stop-loss price
            target_price: Take-profit price
            
        Returns:
            Risk-reward ratio
        """
        try:
            # Calculate risk and reward
            risk = abs(entry_price - stop_price)
            reward = abs(target_price - entry_price)
            
            # Calculate ratio
            if risk > 0:
                ratio = reward / risk
            else:
                ratio = 0
                self.logger.warning("Invalid risk for risk-reward calculation")
            
            return ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-reward ratio: {e}", exc_info=True)
            return 0
    
    def calculate_max_drawdown_stop(self, equity: float, peak_equity: float, 
                                  max_drawdown: float = 0.1) -> bool:
        """
        Check if maximum drawdown stop is triggered.
        
        Args:
            equity: Current equity
            peak_equity: Peak equity
            max_drawdown: Maximum allowed drawdown
            
        Returns:
            True if stop is triggered, False otherwise
        """
        try:
            # Calculate current drawdown
            if peak_equity > 0:
                drawdown = (peak_equity - equity) / peak_equity
            else:
                drawdown = 0
            
            # Check if drawdown exceeds maximum
            if drawdown >= max_drawdown:
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown stop: {e}", exc_info=True)
            return False
    
    def calculate_portfolio_heat(self, positions: List[Dict], capital: float) -> float:
        """
        Calculate portfolio heat (total risk exposure).
        
        Args:
            positions: List of position dictionaries
            capital: Total capital
            
        Returns:
            Portfolio heat as percentage of capital
        """
        try:
            # Calculate total risk
            total_risk = sum(position.get('risk_amount', 0) for position in positions)
            
            # Calculate portfolio heat
            if capital > 0:
                portfolio_heat = total_risk / capital
            else:
                portfolio_heat = 0
                self.logger.warning("Invalid capital for portfolio heat calculation")
            
            return portfolio_heat
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio heat: {e}", exc_info=True)
            return 0

class AdvancedTradingStrategy(TradingStrategy):
    """
    Advanced trading strategy with comprehensive risk management.
    Combines multiple signals and risk management techniques.
    """
    
    def __init__(self, name="AdvancedTrading", risk_per_trade=0.02, max_portfolio_risk=0.1,
                max_drawdown=0.15, atr_multiplier=2.0, min_risk_reward=2.0):
        """
        Initialize the AdvancedTradingStrategy.
        
        Args:
            name: Strategy name
            risk_per_trade: Percentage of capital to risk per trade
            max_portfolio_risk: Maximum portfolio risk
            max_drawdown: Maximum allowed drawdown
            atr_multiplier: ATR multiplier for stop-loss
            min_risk_reward: Minimum risk-reward ratio for trades
        """
        super().__init__(name)
        
        # Risk parameters
        self.risk_per_trade = risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown = max_drawdown
        self.atr_multiplier = atr_multiplier
        self.min_risk_reward = min_risk_reward
        
        # Initialize risk manager
        self.risk_manager = RiskManager()
        
        # Initialize ML model manager
        self.model_manager = MLModelManager()
        
        # Initialize feature engineering pipeline
        self.feature_pipeline = FeatureEngineeringPipeline()
        
        # Strategy state
        self.models = {}
        self.peak_equity = 0
        self.positions = []
        self.trade_history = []
        self.win_rate = 0.5  # Initial estimate
        self.win_loss_ratio = 1.0  # Initial estimate
        
        self.logger.info(f"AdvancedTradingStrategy initialized with risk_per_trade={risk_per_trade}, max_portfolio_risk={max_portfolio_risk}")
    
    def initialize(self, symbol, timeframe, initial_capital=10000.0, commission=0.001, slippage=0.0005):
        """
        Initialize the strategy with parameters.
        
        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe of the data
            initial_capital: Initial capital for the backtest
            commission: Commission rate per trade
            slippage: Slippage rate per trade
        """
        super().initialize(symbol, timeframe, initial_capital, commission, slippage)
        
        # Reset strategy state
        self.peak_equity = initial_capital
        self.positions = []
        self.trade_history = []
        
        # Load models
        self.models = self.model_manager.get_best_models(symbol, timeframe)
        
        if not self.models:
            self.logger.warning(f"No models found for {symbol} {timeframe}")
        else:
            self.logger.info(f"Loaded {len(self.models)} models for {symbol} {timeframe}")
    
    def on_bar(self, timestamp, bar):
        """
        Process a new bar.
        
        Args:
            timestamp: Bar timestamp
            bar: Bar data
        """
        super().on_bar(timestamp, bar)
        
        # Update peak equity
        current_equity = self.cash + self.position * bar['close']
        self.peak_equity = max(self.peak_equity, current_equity)
        
        # Check for maximum drawdown stop
        if self.risk_manager.calculate_max_drawdown_stop(current_equity, self.peak_equity, self.max_drawdown):
            self.logger.warning(f"Maximum drawdown of {self.max_drawdown:.1%} reached, exiting all positions")
            # This will be handled in generate_signals
    
    def generate_signals(self, timestamp, bar):
        """
        Generate trading signals.
        
        Args:
            timestamp: Bar timestamp
            bar: Bar data
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Get current price
        price = bar['close']
        
        # Check for maximum drawdown stop
        current_equity = self.cash + self.position * price
        if self.risk_manager.calculate_max_drawdown_stop(current_equity, self.peak_equity, self.max_drawdown):
            if self.position > 0:
                signals.append({
                    'type': 'exit',
                    'reason': 'max_drawdown',
                    'size': self.position
                })
                self.logger.info(f"Maximum drawdown exit at {price}")
                return signals
        
        # Check for exit signals if we have a position
        if self.position > 0:
            # Check for exit signals from models
            features_df = pd.DataFrame([bar])
            predictions = self.model_manager.predict_with_models(features_df, self.models)
            
            # Check direction prediction
            if 'direction' in predictions and len(predictions['direction']) > 0:
                direction_prob = predictions['direction'][0]
                
                # Exit if direction probability is low
                if direction_prob < 0.3:
                    signals.append({
                        'type': 'exit',
                        'reason': 'model_signal',
                        'size': self.position
                    })
                    self.logger.info(f"Model-based exit at {price}")
                    return signals
            
            # Check for trailing stop
            if hasattr(self, 'highest_price') and price < self.highest_price * 0.95:
                signals.append({
                    'type': 'exit',
                    'reason': 'trailing_stop',
                    'size': self.position
                })
                self.logger.info(f"Trailing stop exit at {price}")
                return signals
            
            # Update highest price
            if not hasattr(self, 'highest_price') or price > self.highest_price:
                self.highest_price = price
        
        # Check for entry signals if we don't have a position
        if self.position == 0:
            # Check portfolio heat
            portfolio_heat = self.risk_manager.calculate_portfolio_heat(self.positions, self.cash)
            if portfolio_heat >= self.max_portfolio_risk:
                self.logger.info(f"Maximum portfolio risk reached ({portfolio_heat:.1%}), skipping entry")
                return signals
            
            # Make predictions
            features_df = pd.DataFrame([bar])
            predictions = self.model_manager.predict_with_models(features_df, self.models)
            
            # Check for spike signals
            spike_signal = False
            spike_prob = 0
            
            if 'spike_binary' in predictions and len(predictions['spike_binary']) > 0:
                spike_prob = predictions['spike_binary'][0]
                
                # Check if probability exceeds threshold
                if spike_prob > 0.7:
                    spike_signal = True
            
            # Check direction prediction
            direction_signal = False
            
            if 'direction' in predictions and len(predictions['direction']) > 0:
                direction_prob = predictions['direction'][0]
                
                # Check if probability exceeds threshold
                if direction_prob > 0.7:
                    direction_signal = True
            
            # Check price change prediction
            change_signal = False
            
            if 'change' in predictions and len(predictions['change']) > 0:
                change_pred = predictions['change'][0]
                
                # Check if predicted change is positive and significant
                if change_pred > 1.0:
                    change_signal = True
            
            # Combine signals
            entry_signal = (spike_signal and direction_signal) or (direction_signal and change_signal)
            
            if entry_signal:
                # Calculate ATR if available
                atr = bar.get('atr_14', price * 0.02)  # Default to 2% of price
                
                # Calculate stop price
                stop_price = self.risk_manager.calculate_volatility_adjusted_stop(price, atr, self.atr_multiplier)
                
                # Calculate target price
                price_diff = price - stop_price
                target_price = price + (price_diff * self.min_risk_reward)
                
                # Check risk-reward ratio
                risk_reward = self.risk_manager.calculate_risk_reward_ratio(price, stop_price, target_price)
                
                if risk_reward >= self.min_risk_reward:
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        self.cash, self.risk_per_trade, price, stop_price
                    )
                    
                    # Generate buy signal
                    signals.append({
                        'type': 'buy',
                        'reason': 'combined_signal',
                        'size': position_size,
                   <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>