import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from scipy import stats

class MarketRegimeDetector:
    """
    Enhanced market regime detection system that identifies different market conditions
    using multiple methods and indicators.
    """
    
    def __init__(self):
        """Initialize the MarketRegimeDetector."""
        self.logger = logging.getLogger("market_regime_detector")
        self.regime_history = {}
        
    def detect_regime(self, df, symbol, methods=None):
        """
        Detect the current market regime using multiple methods.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol to detect regime for
            methods: List of methods to use for detection (default: all)
            
        Returns:
            dict: Market regime information
        """
        if df is None or df.empty:
            self.logger.warning(f"Empty data provided for {symbol}")
            return {"regime": "unknown", "confidence": 0.0, "details": {}}
        
        # Default to all methods if none specified
        if methods is None:
            methods = [
                "trend_analysis", 
                "volatility_analysis", 
                "statistical_analysis",
                "pattern_recognition",
                "volume_analysis",
                "clustering"
            ]
        
        results = {}
        
        # Apply each method
        for method in methods:
            if method == "trend_analysis":
                results[method] = self._detect_trend(df)
            elif method == "volatility_analysis":
                results[method] = self._detect_volatility_regime(df)
            elif method == "statistical_analysis":
                results[method] = self._detect_statistical_regime(df)
            elif method == "pattern_recognition":
                results[method] = self._detect_patterns(df)
            elif method == "volume_analysis":
                results[method] = self._detect_volume_regime(df)
            elif method == "clustering":
                results[method] = self._detect_regime_clustering(df)
        
        # Combine results from all methods
        combined_result = self._combine_regime_results(results)
        
        # Store history
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
        
        self.regime_history[symbol].append({
            "timestamp": df.index[-1],
            "regime": combined_result["regime"],
            "confidence": combined_result["confidence"],
            "details": combined_result["details"]
        })
        
        # Limit history size
        if len(self.regime_history[symbol]) > 100:
            self.regime_history[symbol] = self.regime_history[symbol][-100:]
        
        return combined_result
    
    def _detect_trend(self, df):
        """
        Detect market trend using multiple indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            dict: Trend analysis results
        """
        try:
            # Calculate indicators
            df = df.copy()
            
            # Moving averages
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['sma50'] = df['close'].rolling(window=50).mean()
            df['sma200'] = df['close'].rolling(window=200).mean()
            
            # ADX (Average Directional Index) for trend strength
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                                    np.maximum(df['high'] - df['high'].shift(1), 0), 0)
            df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                                     np.maximum(df['low'].shift(1) - df['low'], 0), 0)
            
            # Calculate smoothed values
            period = 14
            df['tr14'] = df['tr'].rolling(window=period).sum()
            df['plus_di14'] = 100 * (df['plus_dm'].rolling(window=period).sum() / df['tr14'])
            df['minus_di14'] = 100 * (df['minus_dm'].rolling(window=period).sum() / df['tr14'])
            df['dx'] = 100 * abs(df['plus_di14'] - df['minus_di14']) / (df['plus_di14'] + df['minus_di14'])
            df['adx'] = df['dx'].rolling(window=period).mean()
            
            # Linear regression slope
            df['price_change'] = df['close'].pct_change()
            df['slope20'] = df['close'].rolling(window=20).apply(
                lambda x: stats.linregress(np.arange(len(x)), x)[0] / x.mean(), raw=True)
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Determine trend based on moving averages
            ma_trend = 0
            if latest['close'] > latest['sma20'] > latest['sma50']:
                ma_trend += 1
            if latest['close'] < latest['sma20'] < latest['sma50']:
                ma_trend -= 1
            if latest['sma20'] > latest['sma50'] > latest['sma200']:
                ma_trend += 1
            if latest['sma20'] < latest['sma50'] < latest['sma200']:
                ma_trend -= 1
                
            # Determine trend strength based on ADX
            adx_value = latest['adx']
            adx_trend = 0
            if adx_value > 25:  # Strong trend
                if latest['plus_di14'] > latest['minus_di14']:
                    adx_trend = 1
                else:
                    adx_trend = -1
            
            # Determine trend based on slope
            slope_trend = 0
            if latest['slope20'] > 0.001:
                slope_trend = 1
            elif latest['slope20'] < -0.001:
                slope_trend = -1
            
            # Combine trend indicators
            trend_score = ma_trend + adx_trend + slope_trend
            
            if trend_score >= 2:
                trend = "uptrend"
                strength = min(abs(trend_score) / 3, 1.0)
            elif trend_score <= -2:
                trend = "downtrend"
                strength = min(abs(trend_score) / 3, 1.0)
            else:
                trend = "sideways"
                strength = 1.0 - min(abs(trend_score) / 3, 1.0)
            
            return {
                "regime": trend,
                "confidence": strength,
                "details": {
                    "adx": adx_value,
                    "ma_trend": ma_trend,
                    "adx_trend": adx_trend,
                    "slope_trend": slope_trend,
                    "trend_score": trend_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in trend detection: {e}", exc_info=True)
            return {"regime": "unknown", "confidence": 0.0, "details": {}}
    
    def _detect_volatility_regime(self, df):
        """
        Detect volatility regime.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            dict: Volatility regime results
        """
        try:
            # Calculate indicators
            df = df.copy()
            
            # Calculate volatility metrics
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
            
            # ATR (Average True Range)
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr14'] = df['tr'].rolling(window=14).mean()
            df['atr_pct'] = df['atr14'] / df['close']
            
            # Bollinger Bands width
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['stddev'] = df['close'].rolling(window=20).std()
            df['bb_width'] = (df['sma20'] + 2 * df['stddev'] - (df['sma20'] - 2 * df['stddev'])) / df['sma20']
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Historical volatility percentiles
            vol_percentile = stats.percentileofscore(
                df['volatility'].dropna(), latest['volatility']) / 100
            atr_percentile = stats.percentileofscore(
                df['atr_pct'].dropna(), latest['atr_pct']) / 100
            bb_percentile = stats.percentileofscore(
                df['bb_width'].dropna(), latest['bb_width']) / 100
            
            # Combine volatility metrics
            vol_score = (vol_percentile + atr_percentile + bb_percentile) / 3
            
            if vol_score > 0.7:
                regime = "high_volatility"
                confidence = (vol_score - 0.7) / 0.3  # Scale 0.7-1.0 to 0-1
            elif vol_score < 0.3:
                regime = "low_volatility"
                confidence = (0.3 - vol_score) / 0.3  # Scale 0-0.3 to 0-1
            else:
                regime = "normal_volatility"
                confidence = 1.0 - abs(vol_score - 0.5) / 0.2  # Highest at 0.5
            
            return {
                "regime": regime,
                "confidence": confidence,
                "details": {
                    "volatility": latest['volatility'],
                    "atr_pct": latest['atr_pct'],
                    "bb_width": latest['bb_width'],
                    "vol_percentile": vol_percentile,
                    "atr_percentile": atr_percentile,
                    "bb_percentile": bb_percentile,
                    "vol_score": vol_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in volatility regime detection: {e}", exc_info=True)
            return {"regime": "unknown", "confidence": 0.0, "details": {}}
    
    def _detect_statistical_regime(self, df):
        """
        Detect statistical properties of the market.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            dict: Statistical regime results
        """
        try:
            # Calculate returns
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            
            # Get recent returns (last 30 periods)
            recent_returns = df['returns'].dropna().tail(30)
            
            if len(recent_returns) < 30:
                return {"regime": "unknown", "confidence": 0.0, "details": {}}
            
            # Test for stationarity (Augmented Dickey-Fuller test)
            adf_result = adfuller(df['close'].dropna().tail(60))
            is_stationary = adf_result[1] < 0.05  # p-value < 0.05 means stationary
            
            # Test for normality (Shapiro-Wilk test)
            _, normality_pvalue = stats.shapiro(recent_returns)
            is_normal = normality_pvalue > 0.05  # p-value > 0.05 means normal
            
            # Calculate skewness and kurtosis
            skewness = stats.skew(recent_returns)
            kurtosis = stats.kurtosis(recent_returns)
            
            # Determine regime based on statistical properties
            if is_stationary:
                if abs(skewness) < 0.5 and abs(kurtosis) < 1:
                    regime = "mean_reverting_normal"
                    confidence = 0.7 + (0.3 * (1 - abs(skewness) / 0.5))
                else:
                    regime = "mean_reverting_nonnormal"
                    confidence = 0.6 + (0.4 * (1 - min(abs(skewness) / 2, 1)))
            else:
                if abs(skewness) > 1 or kurtosis > 3:
                    regime = "trending_fat_tails"
                    confidence = 0.6 + (0.4 * min(max(abs(skewness) / 2, kurtosis / 6), 1))
                else:
                    regime = "random_walk"
                    confidence = 0.5 + (0.5 * (1 - normality_pvalue))
            
            return {
                "regime": regime,
                "confidence": confidence,
                "details": {
                    "is_stationary": is_stationary,
                    "adf_pvalue": adf_result[1],
                    "is_normal": is_normal,
                    "normality_pvalue": normality_pvalue,
                    "skewness": skewness,
                    "kurtosis": kurtosis
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in statistical regime detection: {e}", exc_info=True)
            return {"regime": "unknown", "confidence": 0.0, "details": {}}
    
    def _detect_patterns(self, df):
        """
        Detect chart patterns in the market.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            dict: Pattern recognition results
        """
        try:
            # Calculate indicators
            df = df.copy()
            
            # Detect support and resistance levels
            window = 20
            df['rolling_high'] = df['high'].rolling(window=window).max()
            df['rolling_low'] = df['low'].rolling(window=window).min()
            
            # Detect if price is near support or resistance
            latest = df.iloc[-1]
            price = latest['close']
            
            # Calculate distance to support/resistance as percentage
            support_distance = (price - latest['rolling_low']) / price
            resistance_distance = (latest['rolling_high'] - price) / price
            
            # Detect breakouts
            breakout_threshold = 0.02  # 2%
            prev_resistance = df['rolling_high'].shift(1).iloc[-1]
            prev_support = df['rolling_low'].shift(1).iloc[-1]
            
            resistance_breakout = price > prev_resistance * (1 + breakout_threshold)
            support_breakdown = price < prev_support * (1 - breakout_threshold)
            
            # Detect consolidation (narrowing Bollinger Bands)
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['stddev'] = df['close'].rolling(window=20).std()
            df['bb_width'] = (df['sma20'] + 2 * df['stddev'] - (df['sma20'] - 2 * df['stddev'])) / df['sma20']
            
            # Check if BB width is narrowing
            bb_width_change = df['bb_width'].pct_change(periods=5).iloc[-1]
            is_consolidating = bb_width_change < -0.1  # BB width decreased by 10%
            
            # Determine pattern-based regime
            if resistance_breakout:
                regime = "breakout"
                confidence = 0.7 + (0.3 * min(price / prev_resistance - 1, 0.1) / 0.1)
            elif support_breakdown:
                regime = "breakdown"
                confidence = 0.7 + (0.3 * min(1 - price / prev_support, 0.1) / 0.1)
            elif is_consolidating:
                regime = "consolidation"
                confidence = 0.6 + (0.4 * min(abs(bb_width_change) / 0.2, 1))
            elif support_distance < 0.03:  # Within 3% of support
                regime = "near_support"
                confidence = 0.6 + (0.4 * (1 - support_distance / 0.03))
            elif resistance_distance < 0.03:  # Within 3% of resistance
                regime = "near_resistance"
                confidence = 0.6 + (0.4 * (1 - resistance_distance / 0.03))
            else:
                regime = "range_bound"
                confidence = 0.5
            
            return {
                "regime": regime,
                "confidence": confidence,
                "details": {
    <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>