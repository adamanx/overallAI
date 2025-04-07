import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple

class TechnicalIndicators:
    """
    Class for calculating technical indicators on cryptocurrency price data.
    Implements common indicators used in trading strategies.
    """
    
    def __init__(self):
        """Initialize the TechnicalIndicators class."""
        self.logger = logging.getLogger("technical_indicators")
        self.logger.info("TechnicalIndicators initialized")
    
    def calculate_all(self, df: pd.DataFrame, include: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate all technical indicators on the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            include: List of indicators to include (None for all)
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Define all available indicators
            all_indicators = {
                'sma': self.sma,
                'ema': self.ema,
                'rsi': self.rsi,
                'macd': self.macd,
                'bollinger_bands': self.bollinger_bands,
                'atr': self.atr,
                'stochastic': self.stochastic,
                'obv': self.obv,
                'vwap': self.vwap,
                'ichimoku': self.ichimoku,
                'fibonacci_retracement': self.fibonacci_retracement
            }
            
            # Determine which indicators to calculate
            indicators_to_calculate = all_indicators
            if include is not None:
                indicators_to_calculate = {k: v for k, v in all_indicators.items() if k in include}
            
            # Calculate each indicator
            for name, func in indicators_to_calculate.items():
                try:
                    self.logger.info(f"Calculating {name}")
                    result_df = func(result_df)
                except Exception as e:
                    self.logger.error(f"Error calculating {name}: {e}", exc_info=True)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}", exc_info=True)
            return df
    
    def sma(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Simple Moving Average for multiple periods.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of periods to calculate SMA for
            
        Returns:
            DataFrame with added SMA columns
        """
        try:
            result_df = df.copy()
            
            for period in periods:
                result_df[f'sma_{period}'] = result_df['close'].rolling(window=period).mean()
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}", exc_info=True)
            return df
    
    def ema(self, df: pd.DataFrame, periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average for multiple periods.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of periods to calculate EMA for
            
        Returns:
            DataFrame with added EMA columns
        """
        try:
            result_df = df.copy()
            
            for period in periods:
                result_df[f'ema_{period}'] = result_df['close'].ewm(span=period, adjust=False).mean()
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}", exc_info=True)
            return df
    
    def rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for RSI calculation
            
        Returns:
            DataFrame with added RSI column
        """
        try:
            result_df = df.copy()
            
            # Calculate price changes
            delta = result_df['close'].diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            result_df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}", exc_info=True)
            return df
    
    def macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate Moving Average Convergence Divergence.
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            DataFrame with added MACD columns
        """
        try:
            result_df = df.copy()
            
            # Calculate fast and slow EMAs
            fast_ema = result_df['close'].ewm(span=fast_period, adjust=False).mean()
            slow_ema = result_df['close'].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            result_df['macd_line'] = fast_ema - slow_ema
            
            # Calculate signal line
            result_df['macd_signal'] = result_df['macd_line'].ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            result_df['macd_histogram'] = result_df['macd_line'] - result_df['macd_signal']
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}", exc_info=True)
            return df
    
    def bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for moving average
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        try:
            result_df = df.copy()
            
            # Calculate middle band (SMA)
            result_df['bb_middle'] = result_df['close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            result_df['bb_std'] = result_df['close'].rolling(window=period).std()
            
            # Calculate upper and lower bands
            result_df['bb_upper'] = result_df['bb_middle'] + (result_df['bb_std'] * std_dev)
            result_df['bb_lower'] = result_df['bb_middle'] - (result_df['bb_std'] * std_dev)
            
            # Calculate bandwidth and %B
            result_df['bb_bandwidth'] = (result_df['bb_upper'] - result_df['bb_lower']) / result_df['bb_middle']
            result_df['bb_percent_b'] = (result_df['close'] - result_df['bb_lower']) / (result_df['bb_upper'] - result_df['bb_lower'])
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}", exc_info=True)
            return df
    
    def atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for ATR calculation
            
        Returns:
            DataFrame with added ATR column
        """
        try:
            result_df = df.copy()
            
            # Calculate true range
            result_df['tr1'] = abs(result_df['high'] - result_df['low'])
            result_df['tr2'] = abs(result_df['high'] - result_df['close'].shift())
            result_df['tr3'] = abs(result_df['low'] - result_df['close'].shift())
            result_df['tr'] = result_df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Calculate ATR
            result_df[f'atr_{period}'] = result_df['tr'].rolling(window=period).mean()
            
            # Drop temporary columns
            result_df = result_df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}", exc_info=True)
            return df
    
    def stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            df: DataFrame with OHLCV data
            k_period: Period for %K
            d_period: Period for %D
            
        Returns:
            DataFrame with added Stochastic columns
        """
        try:
            result_df = df.copy()
            
            # Calculate %K
            result_df['stoch_lowest_low'] = result_df['low'].rolling(window=k_period).min()
            result_df['stoch_highest_high'] = result_df['high'].rolling(window=k_period).max()
            result_df['stoch_k'] = 100 * ((result_df['close'] - result_df['stoch_lowest_low']) / 
                                         (result_df['stoch_highest_high'] - result_df['stoch_lowest_low']))
            
            # Calculate %D
            result_df['stoch_d'] = result_df['stoch_k'].rolling(window=d_period).mean()
            
            # Drop temporary columns
            result_df = result_df.drop(['stoch_lowest_low', 'stoch_highest_high'], axis=1)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}", exc_info=True)
            return df
    
    def obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On-Balance Volume.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added OBV column
        """
        try:
            result_df = df.copy()
            
            # Calculate price direction
            result_df['price_direction'] = np.where(result_df['close'] > result_df['close'].shift(1), 1,
                                                  np.where(result_df['close'] < result_df['close'].shift(1), -1, 0))
            
            # Calculate OBV
            result_df['obv'] = (result_df['price_direction'] * result_df['volume']).cumsum()
            
            # Drop temporary columns
            result_df = result_df.drop(['price_direction'], axis=1)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}", exc_info=True)
            return df
    
    def vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume-Weighted Average Price.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added VWAP column
        """
        try:
            result_df = df.copy()
            
            # Calculate typical price
            result_df['typical_price'] = (result_df['high'] + result_df['low'] + result_df['close']) / 3
            
            # Calculate cumulative values
            result_df['tp_volume'] = result_df['typical_price'] * result_df['volume']
            result_df['cumulative_tp_volume'] = result_df['tp_volume'].cumsum()
            result_df['cumulative_volume'] = result_df['volume'].cumsum()
            
            # Calculate VWAP
            result_df['vwap'] = result_df['cumulative_tp_volume'] / result_df['cumulative_volume']
            
            # Drop temporary columns
            result_df = result_df.drop(['typical_price', 'tp_volume', 'cumulative_tp_volume', 'cumulative_volume'], axis=1)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}", exc_info=True)
            return df
    
    def ichimoku(self, df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, 
                senkou_b_period: int = 52, displacement: int = 26) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud.
        
        Args:
            df: DataFrame with OHLCV data
            tenkan_period: Period for Tenkan-sen (Conversion Line)
            kijun_period: Period for Kijun-sen (Base Line)
            senkou_b_period: Period for Senkou Span B
            displacement: Displacement for Senkou Span A and B
            
        Returns:
            DataFrame with added Ichimoku columns
        """
        try:
            result_df = df.copy()
            
            # Calculate Tenkan-sen (Conversion Line)
            result_df['ichimoku_tenkan'] = (result_df['high'].rolling(window=tenkan_period).max() + 
                                          result_df['low'].rolling(window=tenkan_period).min()) / 2
            
            # Calculate Kijun-sen (Base Line)
            result_df['ichimoku_kijun'] = (result_df['high'].rolling(window=kijun_period).max() + 
                                         result_df['low'].rolling(window=kijun_period).min()) / 2
            
            # Calculate Senkou Span A (Leading Span A)
            result_df['ichimoku_senkou_a'] = ((result_df['ichimoku_tenkan'] + result_df['ichimoku_kijun']) / 2).shift(displacement)
            
            # Calculate Senkou Span B (Leading Span B)
            result_df['ichimoku_senkou_b'] = ((result_df['high'].rolling(window=senkou_b_period).max() + 
                                             result_df['low'].rolling(window=senkou_b_period).min()) / 2).shift(displacement)
            
            # Calculate Chikou Span (Lagging Span)
            result_df['ichimoku_chikou'] = result_df['close'].shift(-displacement)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku: {e}", exc_info=True)
            return df
    
    def fibonacci_retracement(self, df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
        """
        Calculate Fibonacci Retracement levels based on recent high and low.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period to look back for high and low
            
        Returns:
            DataFrame with added Fibonacci Retracement columns
        """
        try:
            result_df = df.copy()
            
            # Calculate rolling high and low
            result_df['fib_high'] = result_df['high'].rolling(window=period).max()
            result_df['fib_low'] = result_df['low'].rolling(window=period).min()
            
            # Calculate Fibonacci levels
            result_df['fib_diff'] = result_df['fib_high'] - result_df['fib_low']
            result_df['fib_0'] = result_df['fib_low']
            result_df['fib_0.236'] = result_df['fib_low'] + 0.236 * result_df['fib_diff']
            result_df['fib_0.382'] = result_df['fib_low'] + 0.382 * result_df['fib_diff']
            result_df['fib_0.5'] = result_df['fib_low'] + 0.5 * result_df['fib_diff']
            result_df['fib_0.618'] = result_df['fib_low'] + 0.618 * result_df['fib_diff']
            result_df['fib_0.786'] = result_df['fib_low'] + 0.786 * result_df['fib_diff']
       <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>