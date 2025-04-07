import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.technical_indicators import TechnicalIndicators

class FeatureGenerator:
    """
    Class for generating and transforming features for machine learning models.
    Handles feature creation, selection, and normalization.
    """
    
    def __init__(self):
        """Initialize the FeatureGenerator class."""
        self.logger = logging.getLogger("feature_generator")
        self.indicators = TechnicalIndicators()
        self.logger.info("FeatureGenerator initialized")
    
    def generate_features(self, df: pd.DataFrame, include_indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            include_indicators: List of indicators to include (None for all)
            
        Returns:
            DataFrame with generated features
        """
        try:
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate technical indicators
            result_df = self.indicators.calculate_all(result_df, include=include_indicators)
            
            # Generate price-based features
            result_df = self.generate_price_features(result_df)
            
            # Generate volume-based features
            result_df = self.generate_volume_features(result_df)
            
            # Generate volatility features
            result_df = self.generate_volatility_features(result_df)
            
            # Generate momentum features
            result_df = self.generate_momentum_features(result_df)
            
            # Generate pattern recognition features
            result_df = self.generate_pattern_features(result_df)
            
            # Generate market regime features
            result_df = self.generate_market_regime_features(result_df)
            
            # Drop rows with NaN values
            result_df = result_df.dropna()
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}", exc_info=True)
            return df
    
    def generate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added price features
        """
        try:
            result_df = df.copy()
            
            # Calculate price changes
            result_df['price_change'] = result_df['close'].diff()
            result_df['price_change_pct'] = result_df['close'].pct_change() * 100
            
            # Calculate log returns
            result_df['log_return'] = np.log(result_df['close'] / result_df['close'].shift(1))
            
            # Calculate price ratios
            result_df['high_low_ratio'] = result_df['high'] / result_df['low']
            result_df['close_open_ratio'] = result_df['close'] / result_df['open']
            
            # Calculate price position
            result_df['price_position'] = (result_df['close'] - result_df['low']) / (result_df['high'] - result_df['low'])
            
            # Calculate moving averages crossovers
            if 'sma_20' in result_df.columns and 'sma_50' in result_df.columns:
                result_df['sma_20_50_cross'] = np.where(result_df['sma_20'] > result_df['sma_50'], 1, -1)
            
            if 'ema_12' in result_df.columns and 'ema_26' in result_df.columns:
                result_df['ema_12_26_cross'] = np.where(result_df['ema_12'] > result_df['ema_26'], 1, -1)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating price features: {e}", exc_info=True)
            return df
    
    def generate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volume-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volume features
        """
        try:
            result_df = df.copy()
            
            # Calculate volume changes
            result_df['volume_change'] = result_df['volume'].diff()
            result_df['volume_change_pct'] = result_df['volume'].pct_change() * 100
            
            # Calculate volume moving averages
            result_df['volume_sma_5'] = result_df['volume'].rolling(window=5).mean()
            result_df['volume_sma_20'] = result_df['volume'].rolling(window=20).mean()
            
            # Calculate volume ratios
            result_df['volume_ratio_5_20'] = result_df['volume_sma_5'] / result_df['volume_sma_20']
            
            # Calculate price-volume relationship
            result_df['price_volume_corr_5'] = result_df['close'].rolling(window=5).corr(result_df['volume'])
            result_df['price_volume_corr_20'] = result_df['close'].rolling(window=20).corr(result_df['volume'])
            
            # Calculate volume relative to average
            result_df['volume_relative_to_avg_20'] = result_df['volume'] / result_df['volume_sma_20']
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating volume features: {e}", exc_info=True)
            return df
    
    def generate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate volatility-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added volatility features
        """
        try:
            result_df = df.copy()
            
            # Calculate daily range
            result_df['daily_range'] = result_df['high'] - result_df['low']
            result_df['daily_range_pct'] = result_df['daily_range'] / result_df['close'] * 100
            
            # Calculate historical volatility
            result_df['volatility_5'] = result_df['log_return'].rolling(window=5).std() * np.sqrt(252)
            result_df['volatility_20'] = result_df['log_return'].rolling(window=20).std() * np.sqrt(252)
            
            # Calculate Garman-Klass volatility
            result_df['garman_klass_vol'] = np.sqrt(
                0.5 * np.log(result_df['high'] / result_df['low'])**2 -
                (2 * np.log(2) - 1) * np.log(result_df['close'] / result_df['open'])**2
            )
            
            # Calculate volatility ratios
            result_df['volatility_ratio_5_20'] = result_df['volatility_5'] / result_df['volatility_20']
            
            # Calculate Bollinger Band width if available
            if 'bb_bandwidth' in result_df.columns:
                result_df['bb_width_change'] = result_df['bb_bandwidth'].diff()
                result_df['bb_width_z_score'] = (result_df['bb_bandwidth'] - result_df['bb_bandwidth'].rolling(window=20).mean()) / result_df['bb_bandwidth'].rolling(window=20).std()
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating volatility features: {e}", exc_info=True)
            return df
    
    def generate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added momentum features
        """
        try:
            result_df = df.copy()
            
            # Calculate rate of change
            result_df['roc_5'] = (result_df['close'] / result_df['close'].shift(5) - 1) * 100
            result_df['roc_10'] = (result_df['close'] / result_df['close'].shift(10) - 1) * 100
            result_df['roc_20'] = (result_df['close'] / result_df['close'].shift(20) - 1) * 100
            
            # Calculate momentum
            result_df['momentum_5'] = result_df['close'] - result_df['close'].shift(5)
            result_df['momentum_10'] = result_df['close'] - result_df['close'].shift(10)
            
            # Calculate acceleration
            result_df['acceleration'] = result_df['momentum_5'].diff()
            
            # Calculate RSI divergence if RSI is available
            if 'rsi_14' in result_df.columns:
                result_df['rsi_divergence'] = np.where(
                    (result_df['close'] > result_df['close'].shift(1)) & (result_df['rsi_14'] < result_df['rsi_14'].shift(1)),
                    -1,  # Bearish divergence
                    np.where(
                        (result_df['close'] < result_df['close'].shift(1)) & (result_df['rsi_14'] > result_df['rsi_14'].shift(1)),
                        1,   # Bullish divergence
                        0    # No divergence
                    )
                )
            
            # Calculate MACD divergence if MACD is available
            if 'macd_line' in result_df.columns and 'macd_signal' in result_df.columns:
                result_df['macd_divergence'] = np.where(
                    (result_df['close'] > result_df['close'].shift(1)) & (result_df['macd_line'] < result_df['macd_line'].shift(1)),
                    -1,  # Bearish divergence
                    np.where(
                        (result_df['close'] < result_df['close'].shift(1)) & (result_df['macd_line'] > result_df['macd_line'].shift(1)),
                        1,   # Bullish divergence
                        0    # No divergence
                    )
                )
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating momentum features: {e}", exc_info=True)
            return df
    
    def generate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate pattern recognition features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added pattern features
        """
        try:
            result_df = df.copy()
            
            # Calculate candlestick patterns
            
            # Doji
            result_df['doji'] = np.where(
                abs(result_df['close'] - result_df['open']) <= 0.1 * (result_df['high'] - result_df['low']),
                1, 0
            )
            
            # Hammer
            result_df['hammer'] = np.where(
                (result_df['high'] - result_df['low'] > 3 * (result_df['open'] - result_df['close'])) &
                ((result_df['close'] - result_df['low']) / (0.001 + result_df['high'] - result_df['low']) > 0.6) &
                ((result_df['open'] - result_df['low']) / (0.001 + result_df['high'] - result_df['low']) > 0.6),
                1, 0
            )
            
            # Shooting Star
            result_df['shooting_star'] = np.where(
                (result_df['high'] - result_df['low'] > 3 * (result_df['open'] - result_df['close'])) &
                ((result_df['high'] - result_df['close']) / (0.001 + result_df['high'] - result_df['low']) > 0.6) &
                ((result_df['high'] - result_df['open']) / (0.001 + result_df['high'] - result_df['low']) > 0.6),
                1, 0
            )
            
            # Engulfing patterns
            result_df['bullish_engulfing'] = np.where(
                (result_df['close'].shift(1) < result_df['open'].shift(1)) &  # Previous candle is bearish
                (result_df['close'] > result_df['open']) &                    # Current candle is bullish
                (result_df['open'] <= result_df['close'].shift(1)) &          # Current open <= previous close
                (result_df['close'] >= result_df['open'].shift(1)),           # Current close >= previous open
                1, 0
            )
            
            result_df['bearish_engulfing'] = np.where(
                (result_df['close'].shift(1) > result_df['open'].shift(1)) &  # Previous candle is bullish
                (result_df['close'] < result_df['open']) &                    # Current candle is bearish
                (result_df['open'] >= result_df['close'].shift(1)) &          # Current open >= previous close
                (result_df['close'] <= result_df['open'].shift(1)),           # Current close <= previous open
                1, 0
            )
            
            # Three white soldiers
            result_df['three_white_soldiers'] = np.where(
                (result_df['close'] > result_df['open']) &                    # Current candle is bullish
                (result_df['close'].shift(1) > result_df['open'].shift(1)) &  # Previous candle is bullish
                (result_df['close'].shift(2) > result_df['open'].shift(2)) &  # Candle before previous is bullish
                (result_df['open'] > result_df['open'].shift(1)) &            # Current open > previous open
                (result_df['open'].shift(1) > result_df['open'].shift(2)) &   # Previous open > before previous open
                (result_df['close'] > result_df['close'].shift(1)) &          # Current close > previous close
                (result_df['close'].shift(1) > result_df['close'].shift(2)),  # Previous close > before previous close
                1, 0
            )
            
            # Three black crows
            result_df['three_black_crows'] = np.where(
                (result_df['close'] < result_df['open']) &                    # Current candle is bearish
                (result_df['close'].shift(1) < result_df['open'].shift(1)) &  # Previous candle is bearish
                (result_df['close'].shift(2) < result_df['open'].shift(2)) &  # Candle before previous is bearish
                (result_df['open'] < result_df['open'].shift(1)) &            # Current open < previous open
                (result_df['open'].shift(1) < result_df['open'].shift(2)) &   # Previous open < before previous open
                (result_df['close'] < result_df['close'].shift(1)) &          # Current close < previous close
                (result_df['close'].shift(1) < result_df['close'].shift(2)),  # Previous close < before previous close
                1, 0
            )
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating pattern features: {e}", exc_info=True)
            return df
    
    def generate_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate market regime features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added market regime features
        """
        try:
            result_df = df.copy()
            
            # Calculate trend strength
            if 'adx' not in result_df.columns and 'atr_14' in result_df.columns:
                # Simple trend strength based on price movement and ATR
                result_df['trend_strength'] = abs(result_df['close'] - result_df['close'].shift(14)) / (result_df['atr_14'] * 14)
            
            # Calculate market regime based on SMA crossovers
            if 'sma_20' in result_df.columns and 'sma_50' in result_df.columns and 'sma_200' in result_df.columns:
                # Bullish market: 20 SMA > 50 SMA > 200 SMA
                # Bearish market: 20 SMA < 50 SMA < 200 SMA
                # Transition market: other conditions
                
                result_df['market_regime'] = np.where(
                    (result_df['sma_20'] > result_df['sma_50']) & (result_df['sma_50'] > result_df['sma_200']),
                    1,  # Bulli<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>