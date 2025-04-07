import logging
import pandas as pd
import numpy as np
from src.trading_strategy import AdaptiveTradingStrategy
from src.market_regime_detector import MarketRegimeDetector
from src.on_chain_metrics_collector import OnChainMetricsCollector
from src.enhanced_feature_generator import EnhancedFeatureGenerator

class EnhancedAdaptiveTradingStrategy(AdaptiveTradingStrategy):
    """
    Enhanced adaptive trading strategy that incorporates market regime detection
    and on-chain metrics for improved signal generation and risk management.
    """
    
    def __init__(self, base_risk_per_trade=0.02, max_risk_per_trade=0.04, 
                 min_risk_per_trade=0.0075, atr_multiplier=2.0, max_drawdown=0.15,
                 etherscan_api_key=None, solscan_api_key=None):
        """
        Initialize the EnhancedAdaptiveTradingStrategy.
        
        Args:
            base_risk_per_trade: Base risk per trade as a fraction of portfolio
            max_risk_per_trade: Maximum risk per trade
            min_risk_per_trade: Minimum risk per trade
            atr_multiplier: Multiplier for ATR to set stop loss
            max_drawdown: Maximum allowed drawdown
            etherscan_api_key: API key for Etherscan
            solscan_api_key: API key for Solscan
        """
        super().__init__(base_risk_per_trade, max_risk_per_trade, 
                         min_risk_per_trade, atr_multiplier, max_drawdown)
        
        self.logger = logging.getLogger("enhanced_adaptive_strategy")
        self.feature_generator = EnhancedFeatureGenerator(
            etherscan_api_key=etherscan_api_key,
            solscan_api_key=solscan_api_key
        )
        
        # Additional parameters for enhanced strategy
        self.regime_weight = 0.4  # Weight for regime-based signals
        self.on_chain_weight = 0.3  # Weight for on-chain signals
        self.price_action_weight = 0.3  # Weight for price action signals
        
        # Regime-specific parameters
        self.regime_params = {
            'bullish': {
                'risk_multiplier': 1.2,
                'profit_target_multiplier': 1.5,
                'trailing_stop_activation': 0.5
            },
            'bearish': {
                'risk_multiplier': 0.8,
                'profit_target_multiplier': 1.2,
                'trailing_stop_activation': 0.3
            },
            'volatile': {
                'risk_multiplier': 0.7,
                'profit_target_multiplier': 1.8,
                'trailing_stop_activation': 0.4
            },
            'low_volatility': {
                'risk_multiplier': 0.9,
                'profit_target_multiplier': 1.3,
                'trailing_stop_activation': 0.6
            },
            'mean_reverting': {
                'risk_multiplier': 1.0,
                'profit_target_multiplier': 1.4,
                'trailing_stop_activation': 0.5
            },
            'support_resistance': {
                'risk_multiplier': 1.1,
                'profit_target_multiplier': 1.6,
                'trailing_stop_activation': 0.7
            },
            'neutral': {
                'risk_multiplier': 1.0,
                'profit_target_multiplier': 1.5,
                'trailing_stop_activation': 0.5
            },
            'unknown': {
                'risk_multiplier': 0.9,
                'profit_target_multiplier': 1.4,
                'trailing_stop_activation': 0.5
            }
        }
        
        self.logger.info("EnhancedAdaptiveTradingStrategy initialized")
    
    def on_bar(self, timestamp, bar_data):
        """
        Process a new bar of data.
        
        Args:
            timestamp: Bar timestamp
            bar_data: Bar data including OHLCV and predictions
            
        Returns:
            None
        """
        try:
            # Call parent method first
            super().on_bar(timestamp, bar_data)
            
            # Get enhanced features
            if hasattr(self, 'data') and len(self.data) > 0:
                # Create a DataFrame from stored data
                df = pd.DataFrame(self.data)
                
                # Generate enhanced features
                enhanced_df = self.feature_generator.generate_features(df, self.symbol)
                
                # Store enhanced features
                self.enhanced_features = enhanced_df.iloc[-1].to_dict() if not enhanced_df.empty else {}
                
                # Update strategy parameters based on market regime
                self._adapt_to_market_regime()
                
                # Update strategy parameters based on on-chain metrics
                self._adapt_to_on_chain_metrics()
            
        except Exception as e:
            self.logger.error(f"Error in on_bar: {e}", exc_info=True)
    
    def generate_signals(self, timestamp, bar_data):
        """
        Generate trading signals based on current data and enhanced features.
        
        Args:
            timestamp: Current timestamp
            bar_data: Current bar data
            
        Returns:
            list: Trading signals
        """
        try:
            # Get base signals from parent class
            base_signals = super().generate_signals(timestamp, bar_data)
            
            # If we don't have enhanced features yet, return base signals
            if not hasattr(self, 'enhanced_features') or not self.enhanced_features:
                return base_signals
            
            # Generate enhanced signals
            enhanced_signals = self._generate_enhanced_signals(timestamp, bar_data)
            
            # Combine base and enhanced signals
            combined_signals = self._combine_signals(base_signals, enhanced_signals)
            
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}", exc_info=True)
            return []
    
    def _adapt_to_market_regime(self):
        """
        Adapt strategy parameters based on detected market regime.
        
        Returns:
            None
        """
        try:
            # Check if we have regime information
            if 'market_regime' not in self.enhanced_features:
                return
            
            regime = self.enhanced_features['market_regime']
            confidence = self.enhanced_features.get('regime_confidence', 0.5)
            
            # Get regime parameters
            if regime in self.regime_params:
                params = self.regime_params[regime]
                
                # Adjust parameters based on regime and confidence
                # The higher the confidence, the more we adjust
                base_risk = self.base_risk_per_trade
                adjusted_risk = base_risk * (1 + (params['risk_multiplier'] - 1) * confidence)
                self.current_risk_per_trade = max(self.min_risk_per_trade, 
                                                min(self.max_risk_per_trade, adjusted_risk))
                
                # Adjust profit target
                self.profit_target_multiplier = params['profit_target_multiplier']
                
                # Adjust trailing stop activation
                self.trailing_stop_activation = params['trailing_stop_activation']
                
                self.logger.info(f"Adapted to {regime} regime (confidence: {confidence:.2f}): "
                                f"risk={self.current_risk_per_trade:.4f}, "
                                f"profit_target={self.profit_target_multiplier:.2f}, "
                                f"trailing_stop={self.trailing_stop_activation:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error adapting to market regime: {e}", exc_info=True)
    
    def _adapt_to_on_chain_metrics(self):
        """
        Adapt strategy parameters based on on-chain metrics.
        
        Returns:
            None
        """
        try:
            # Check if we have on-chain metrics
            on_chain_cols = [col for col in self.enhanced_features.keys() 
                            if col.startswith('on_chain_')]
            
            if not on_chain_cols:
                return
            
            # Get on-chain momentum and sentiment
            momentum = self.enhanced_features.get('on_chain_on_chain_momentum', 0)
            sentiment = self.enhanced_features.get('on_chain_on_chain_sentiment', 0)
            
            # Adjust risk based on on-chain metrics
            # Positive momentum and sentiment increase risk, negative decrease it
            on_chain_factor = (momentum + sentiment) / 2
            risk_adjustment = 1 + (on_chain_factor * 0.2)  # Max ±20% adjustment
            
            self.current_risk_per_trade *= risk_adjustment
            self.current_risk_per_trade = max(self.min_risk_per_trade, 
                                            min(self.max_risk_per_trade, self.current_risk_per_trade))
            
            # Adjust position sizing based on on-chain metrics
            if hasattr(self, 'position_sizer'):
                self.position_sizer.on_chain_factor = on_chain_factor
            
            self.logger.info(f"Adapted to on-chain metrics: momentum={momentum:.2f}, "
                            f"sentiment={sentiment:.2f}, risk_adjustment={risk_adjustment:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error adapting to on-chain metrics: {e}", exc_info=True)
    
    def _generate_enhanced_signals(self, timestamp, bar_data):
        """
        Generate signals based on enhanced features.
        
        Args:
            timestamp: Current timestamp
            bar_data: Current bar data
            
        Returns:
            list: Enhanced signals
        """
        try:
            signals = []
            
            # Check if we have the necessary enhanced features
            if 'signal_strength' not in self.enhanced_features:
                return signals
            
            # Get signal strength and category
            signal_strength = self.enhanced_features['signal_strength']
            signal_category = self.enhanced_features.get('signal_category', 'neutral')
            
            # Generate signals based on signal strength and category
            if signal_category in ['strong_buy', 'buy'] and signal_strength > 0.2:
                # Generate buy signal
                entry_price = bar_data['close']
                stop_loss = self._calculate_stop_loss(entry_price, 'long')
                position_size = self._calculate_position_size(entry_price, stop_loss, 'long')
                
                signals.append({
                    'type': 'buy',
                    'timestamp': timestamp,
                    'price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'profit_target': entry_price + (entry_price - stop_loss) * self.profit_target_multiplier,
                    'source': 'enhanced_features',
                    'signal_strength': signal_strength,
                    'signal_category': signal_category
                })
                
            elif signal_category in ['strong_sell', 'sell'] and signal_strength < -0.2:
                # Generate sell signal
                entry_price = bar_data['close']
                stop_loss = self._calculate_stop_loss(entry_price, 'short')
                position_size = self._calculate_position_size(entry_price, stop_loss, 'short')
                
                signals.append({
                    'type': 'sell',
                    'timestamp': timestamp,
                    'price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'profit_target': entry_price - (stop_loss - entry_price) * self.profit_target_multiplier,
                    'source': 'enhanced_features',
                    'signal_strength': signal_strength,
                    'signal_category': signal_category
                })
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced signals: {e}", exc_info=True)
            return []
    
    def _combine_signals(self, base_signals, enhanced_signals):
        """
        Combine base and enhanced signals.
        
        Args:
            base_signals: Signals from base strategy
            enhanced_signals: Signals from enhanced features
            
        Returns:
            list: Combined signals
        """
        try:
            # If either list is empty, return the other
            if not base_signals:
                return enhanced_signals
            if not enhanced_signals:
                return base_signals
            
            # Combine signals based on weights
            combined_signals = []
            
            # Process buy signals
            base_buys = [s for s in base_signals if s['type'] == 'buy']
            enhanced_buys = [s for s in enhanced_signals if s['type'] == 'buy']
            
            if base_buys and enhanced_buys:
                # Both strategies want to buy, combine them
                base_buy = base_buys[0]
                enhanced_buy = enhanced_buys[0]
                
                # Calculate weighted average for position size
                combined_size = (base_buy['size'] * self.price_action_weight + 
                                enhanced_buy['size'] * (self.regime_weight + self.on_chain_weight))
                
                # Use the more conservative stop loss
                combined_stop = max(base_buy['stop_loss'], enhanced_buy['stop_loss'])
                
                # Use the more conservative profit target
                combined_target = min(base_buy['profit_target'], enhanced_buy['profit_target'])
                
                combined_signals.append({
                    'type': 'buy',
                    'timestamp': base_buy['timestamp'],
                    'price': base_buy['price'],
                    'size': combined_size,
                    'stop_loss': combined_stop,
                    'profit_target': combined_target,
                    'source': 'combined',
                    'base_weight': self.price_action_weight,
                    'enhanced_weight': self.regime_weight + self.on_chain_weight
                })
                
            elif base_buys:
                # Only base strategy wants to buy
                combined_signals.extend(base_buys)
                
            elif enhanced_buys:
                # Only enhanced strategy wants to buy
                combined_signals.extend(enhanced_buys)
            
            # Process sell signals (similar logic)
            base_sells = [s for s in base_signals if s['type'] == 'sell']
            enhanced_sells = [s for s in enhanced_signals if s['type'] == 'sell']
            
            if base_sells and enhanced_sells:
                # Both strategies want to sell, combine them
                base_sell = base_sells[0]
                enhanced_sell = enhanced_sells[0]
                
                # Calculate weighted average for position size
                combined_size = (base_sell['size'] * self.price_action_weight + 
                                enhanced_sell['size'] * (self.regime_weight + self.on_chain_weight))
                
                # Use the more conservative stop loss
                combined_stop = min(base_sell['stop_loss'], enhanced_sell['stop_loss'])
                
                # Use the more conservative profit target
                combined_target = max(base_sell['profit_target'], enhanced_sell['profit_target'])
                
                combined_signals.append({
                    'type': 'sell',<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>