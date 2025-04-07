import logging
import pandas as pd
import numpy as np
from src.market_regime_detector import MarketRegimeDetector
from src.on_chain_metrics_collector import OnChainMetricsCollector

class EnhancedFeatureGenerator:
    """
    Enhanced feature generator that incorporates market regime detection and on-chain metrics
    for improved trading signal generation.
    """
    
    def __init__(self, etherscan_api_key=None, solscan_api_key=None):
        """
        Initialize the EnhancedFeatureGenerator.
        
        Args:
            etherscan_api_key: API key for Etherscan
            solscan_api_key: API key for Solscan
        """
        self.logger = logging.getLogger("enhanced_feature_generator")
        self.regime_detector = MarketRegimeDetector()
        self.on_chain_collector = OnChainMetricsCollector(
            etherscan_api_key=etherscan_api_key,
            solscan_api_key=solscan_api_key
        )
        
        self.logger.info("EnhancedFeatureGenerator initialized")
    
    def generate_features(self, df, symbol):
        """
        Generate enhanced features including market regime and on-chain metrics.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol to generate features for
            
        Returns:
            DataFrame: Enhanced features
        """
        if df is None or df.empty:
            self.logger.warning(f"Empty data provided for {symbol}")
            return pd.DataFrame()
        
        try:
            # Create a copy of the dataframe
            enhanced_df = df.copy()
            
            # Detect market regime
            regime_result = self.regime_detector.detect_regime(enhanced_df, symbol)
            
            # Add regime features
            self._add_regime_features(enhanced_df, regime_result)
            
            # Add on-chain metrics
            self._add_on_chain_features(enhanced_df, symbol)
            
            # Add combined features
            self._add_combined_features(enhanced_df)
            
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced features: {e}", exc_info=True)
            return df
    
    def _add_regime_features(self, df, regime_result):
        """
        Add market regime features to the dataframe.
        
        Args:
            df: DataFrame to add features to
            regime_result: Result from regime detector
            
        Returns:
            None (modifies df in-place)
        """
        try:
            # Add regime as categorical feature
            df['market_regime'] = regime_result['regime']
            
            # Add regime confidence
            df['regime_confidence'] = regime_result['confidence']
            
            # Add regime details
            if 'details' in regime_result and 'method_results' in regime_result['details']:
                for method, result in regime_result['details']['method_results'].items():
                    df[f'regime_{method}'] = result['regime']
                    df[f'regime_{method}_confidence'] = result['confidence']
            
            # Add regime transition probabilities
            # This requires historical data, so it might not be available for all symbols
            transition_probs = self.regime_detector.get_regime_transition_probability(
                df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'unknown'
            )
            
            if transition_probs:
                # Flatten transition probabilities
                for from_regime, to_regimes in transition_probs.items():
                    for to_regime, prob in to_regimes.items():
                        df[f'transition_prob_{from_regime}_to_{to_regime}'] = prob
            
            # Add next regime prediction
            next_regime = self.regime_detector.predict_next_regime(
                df.iloc[-1]['symbol'] if 'symbol' in df.columns else 'unknown'
            )
            
            if next_regime['next_regime'] != 'unknown':
                df['predicted_next_regime'] = next_regime['next_regime']
                df['predicted_next_regime_probability'] = next_regime['probability']
            
            # Create numerical encoding for regime
            regime_map = {
                'bullish': 1.0,
                'bearish': -1.0,
                'neutral': 0.0,
                'volatile': 0.5,
                'low_volatility': -0.5,
                'mean_reverting': 0.25,
                'support_resistance': 0.0,
                'unknown': 0.0
            }
            
            df['regime_numeric'] = df['market_regime'].map(regime_map)
            
            # Create regime strength feature (regime * confidence)
            df['regime_strength'] = df['regime_numeric'] * df['regime_confidence']
            
        except Exception as e:
            self.logger.error(f"Error adding regime features: {e}", exc_info=True)
    
    def _add_on_chain_features(self, df, symbol):
        """
        Add on-chain metrics features to the dataframe.
        
        Args:
            df: DataFrame to add features to
            symbol: Symbol to get on-chain metrics for
            
        Returns:
            None (modifies df in-place)
        """
        try:
            # Determine which asset to get metrics for
            asset = None
            if 'ETH' in symbol:
                asset = 'ETH'
            elif 'SOL' in symbol:
                asset = 'SOL'
            else:
                self.logger.warning(f"Unsupported asset for on-chain metrics: {symbol}")
                return
            
            # Get on-chain metrics
            metrics = self.on_chain_collector.get_metrics_for_trading(asset)
            
            if not metrics:
                self.logger.warning(f"No on-chain metrics available for {asset}")
                return
            
            # Add metrics to dataframe
            for key, value in metrics.items():
                df[f'on_chain_{key}'] = value
            
        except Exception as e:
            self.logger.error(f"Error adding on-chain features: {e}", exc_info=True)
    
    def _add_combined_features(self, df):
        """
        Add combined features that integrate market regime and on-chain metrics.
        
        Args:
            df: DataFrame to add features to
            
        Returns:
            None (modifies df in-place)
        """
        try:
            # Check if we have both regime and on-chain features
            if 'regime_strength' not in df.columns or 'on_chain_on_chain_momentum' not in df.columns:
                return
            
            # Create combined momentum feature
            if 'on_chain_on_chain_momentum' in df.columns:
                df['combined_momentum'] = (df['regime_strength'] + df['on_chain_on_chain_momentum']) / 2
            
            # Create combined sentiment feature
            if 'on_chain_on_chain_sentiment' in df.columns:
                df['combined_sentiment'] = (df['regime_strength'] + df['on_chain_on_chain_sentiment']) / 2
            
            # Create regime-adjusted on-chain features
            on_chain_cols = [col for col in df.columns if col.startswith('on_chain_')]
            for col in on_chain_cols:
                if col in df.columns and 'regime_confidence' in df.columns:
                    df[f'regime_adjusted_{col}'] = df[col] * df['regime_confidence']
            
            # Create trading signal strength based on combined features
            if 'combined_momentum' in df.columns and 'combined_sentiment' in df.columns:
                df['signal_strength'] = (df['combined_momentum'] + df['combined_sentiment']) / 2
                
                # Categorize signal strength
                df['signal_category'] = pd.cut(
                    df['signal_strength'],
                    bins=[-1.0, -0.5, -0.2, 0.2, 0.5, 1.0],
                    labels=['strong_sell', 'sell', 'neutral', 'buy', 'strong_buy']
                )
            
        except Exception as e:
            self.logger.error(f"Error adding combined features: {e}", exc_info=True)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create generator
    generator = EnhancedFeatureGenerator(
        etherscan_api_key="YOUR_ETHERSCAN_API_KEY",
        solscan_api_key="YOUR_SOLSCAN_API_KEY"
    )
    
    # Example data
    import yfinance as yf
    eth_data = yf.download("ETH-USD", start="2023-01-01", end="2023-03-01")
    
    # Generate enhanced features
    enhanced_df = generator.generate_features(eth_data, "ETH/USD")
    
    print("Enhanced Features:")
    print(enhanced_df.columns.tolist())
    
    # Show signal features
    if 'signal_strength' in enhanced_df.columns:
        print("\nSignal Strength:")
        print(enhanced_df['signal_strength'].describe())
        
        print("\nSignal Categories:")
        print(enhanced_df['signal_category'].value_counts())
