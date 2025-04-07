import logging
import requests
import pandas as pd
import time
from datetime import datetime, timedelta

class OnChainMetricsCollector:
    """
    Collects on-chain metrics from Etherscan and Solscan for enhanced trading signals.
    """
    
    def __init__(self, etherscan_api_key=None, solscan_api_key=None):
        """
        Initialize the OnChainMetricsCollector.
        
        Args:
            etherscan_api_key: API key for Etherscan
            solscan_api_key: API key for Solscan
        """
        self.logger = logging.getLogger("on_chain_metrics_collector")
        self.etherscan_api_key = etherscan_api_key
        self.solscan_api_key = solscan_api_key
        
        # Base URLs for API endpoints
        self.etherscan_base_url = "https://api.etherscan.io/api"
        self.solscan_base_url = "https://public-api.solscan.io"
        
        # Cache for metrics to reduce API calls
        self.metrics_cache = {
            "eth": {},
            "sol": {}
        }
        
        self.logger.info("OnChainMetricsCollector initialized")
    
    def get_eth_metrics(self, metric_types=None, days=7):
        """
        Get Ethereum on-chain metrics.
        
        Args:
            metric_types: List of metric types to fetch (default: all)
            days: Number of days of historical data to fetch
            
        Returns:
            dict: Ethereum on-chain metrics
        """
        if metric_types is None:
            metric_types = [
                "transactions", 
                "gas", 
                "addresses", 
                "defi", 
                "whales",
                "network"
            ]
        
        results = {}
        
        for metric_type in metric_types:
            try:
                if metric_type == "transactions":
                    results[metric_type] = self._get_eth_transaction_metrics(days)
                elif metric_type == "gas":
                    results[metric_type] = self._get_eth_gas_metrics(days)
                elif metric_type == "addresses":
                    results[metric_type] = self._get_eth_address_metrics(days)
                elif metric_type == "defi":
                    results[metric_type] = self._get_eth_defi_metrics(days)
                elif metric_type == "whales":
                    results[metric_type] = self._get_eth_whale_metrics(days)
                elif metric_type == "network":
                    results[metric_type] = self._get_eth_network_metrics(days)
            except Exception as e:
                self.logger.error(f"Error fetching ETH {metric_type} metrics: {e}", exc_info=True)
                results[metric_type] = {}
        
        # Cache results
        cache_key = f"all_{days}"
        self.metrics_cache["eth"][cache_key] = {
            "timestamp": datetime.now(),
            "data": results
        }
        
        return results
    
    def get_sol_metrics(self, metric_types=None, days=7):
        """
        Get Solana on-chain metrics.
        
        Args:
            metric_types: List of metric types to fetch (default: all)
            days: Number of days of historical data to fetch
            
        Returns:
            dict: Solana on-chain metrics
        """
        if metric_types is None:
            metric_types = [
                "transactions", 
                "accounts", 
                "staking", 
                "nft", 
                "defi",
                "network"
            ]
        
        results = {}
        
        for metric_type in metric_types:
            try:
                if metric_type == "transactions":
                    results[metric_type] = self._get_sol_transaction_metrics(days)
                elif metric_type == "accounts":
                    results[metric_type] = self._get_sol_account_metrics(days)
                elif metric_type == "staking":
                    results[metric_type] = self._get_sol_staking_metrics(days)
                elif metric_type == "nft":
                    results[metric_type] = self._get_sol_nft_metrics(days)
                elif metric_type == "defi":
                    results[metric_type] = self._get_sol_defi_metrics(days)
                elif metric_type == "network":
                    results[metric_type] = self._get_sol_network_metrics(days)
            except Exception as e:
                self.logger.error(f"Error fetching SOL {metric_type} metrics: {e}", exc_info=True)
                results[metric_type] = {}
        
        # Cache results
        cache_key = f"all_{days}"
        self.metrics_cache["sol"][cache_key] = {
            "timestamp": datetime.now(),
            "data": results
        }
        
        return results
    
    def get_metrics_for_trading(self, asset):
        """
        Get on-chain metrics formatted for trading signals.
        
        Args:
            asset: Asset to get metrics for ("ETH" or "SOL")
            
        Returns:
            dict: On-chain metrics for trading
        """
        asset = asset.lower()
        
        if asset not in ["eth", "sol"]:
            self.logger.error(f"Unsupported asset: {asset}")
            return {}
        
        # Check cache first
        cache_key = "all_7"  # 7 days of data
        if asset in self.metrics_cache and cache_key in self.metrics_cache[asset]:
            cache_entry = self.metrics_cache[asset][cache_key]
            cache_age = datetime.now() - cache_entry["timestamp"]
            
            # Use cache if less than 1 hour old
            if cache_age < timedelta(hours=1):
                self.logger.info(f"Using cached {asset} metrics")
                metrics = cache_entry["data"]
            else:
                # Refresh cache
                self.logger.info(f"Refreshing {asset} metrics cache")
                if asset == "eth":
                    metrics = self.get_eth_metrics()
                else:
                    metrics = self.get_sol_metrics()
        else:
            # No cache, fetch metrics
            self.logger.info(f"Fetching {asset} metrics (no cache)")
            if asset == "eth":
                metrics = self.get_eth_metrics()
            else:
                metrics = self.get_sol_metrics()
        
        # Extract and format metrics for trading
        trading_metrics = self._extract_trading_metrics(metrics, asset)
        
        return trading_metrics
    
    def _extract_trading_metrics(self, metrics, asset):
        """
        Extract and format metrics for trading signals.
        
        Args:
            metrics: Raw metrics data
            asset: Asset type ("eth" or "sol")
            
        Returns:
            dict: Formatted metrics for trading
        """
        trading_metrics = {}
        
        try:
            # Common metrics for both ETH and SOL
            if "transactions" in metrics:
                tx_metrics = metrics["transactions"]
                if "daily_count" in tx_metrics:
                    # Calculate transaction growth rate
                    daily_counts = tx_metrics["daily_count"]
                    if len(daily_counts) >= 2:
                        current = daily_counts[-1]
                        previous = daily_counts[-2]
                        tx_growth = (current - previous) / previous if previous > 0 else 0
                        trading_metrics["tx_growth"] = tx_growth
                
                if "avg_value" in tx_metrics:
                    trading_metrics["avg_tx_value"] = tx_metrics["avg_value"]
            
            # Network metrics
            if "network" in metrics:
                network_metrics = metrics["network"]
                if "active_addresses" in network_metrics:
                    trading_metrics["active_addresses"] = network_metrics["active_addresses"][-1]
                
                if "new_addresses" in network_metrics:
                    trading_metrics["new_addresses"] = network_metrics["new_addresses"][-1]
            
            # Asset-specific metrics
            if asset == "eth":
                # ETH-specific metrics
                if "gas" in metrics:
                    gas_metrics = metrics["gas"]
                    if "avg_gas_price" in gas_metrics:
                        trading_metrics["avg_gas_price"] = gas_metrics["avg_gas_price"][-1]
                
                if "defi" in metrics:
                    defi_metrics = metrics["defi"]
                    if "total_value_locked" in defi_metrics:
                        trading_metrics["tvl"] = defi_metrics["total_value_locked"][-1]
                
                if "whales" in metrics:
                    whale_metrics = metrics["whales"]
                    if "whale_transaction_count" in whale_metrics:
                        trading_metrics["whale_txs"] = whale_metrics["whale_transaction_count"][-1]
            
            elif asset == "sol":
                # SOL-specific metrics
                if "staking" in metrics:
                    staking_metrics = metrics["staking"]
                    if "total_staked" in staking_metrics:
                        trading_metrics["total_staked"] = staking_metrics["total_staked"][-1]
                    
                    if "staking_yield" in staking_metrics:
                        trading_metrics["staking_yield"] = staking_metrics["staking_yield"][-1]
                
                if "nft" in metrics:
                    nft_metrics = metrics["nft"]
                    if "daily_volume" in nft_metrics:
                        trading_metrics["nft_volume"] = nft_metrics["daily_volume"][-1]
            
            # Calculate on-chain momentum
            trading_metrics["on_chain_momentum"] = self._calculate_on_chain_momentum(metrics, asset)
            
            # Calculate on-chain sentiment
            trading_metrics["on_chain_sentiment"] = self._calculate_on_chain_sentiment(metrics, asset)
            
        except Exception as e:
            self.logger.error(f"Error extracting trading metrics: {e}", exc_info=True)
        
        return trading_metrics
    
    def _calculate_on_chain_momentum(self, metrics, asset):
        """
        Calculate on-chain momentum score.
        
        Args:
            metrics: Raw metrics data
            asset: Asset type ("eth" or "sol")
            
        Returns:
            float: Momentum score (-1 to 1)
        """
        try:
            momentum_factors = []
            
            # Transaction growth
            if "transactions" in metrics and "daily_count" in metrics["transactions"]:
                daily_counts = metrics["transactions"]["daily_count"]
                if len(daily_counts) >= 7:
                    recent_avg = sum(daily_counts[-3:]) / 3
                    previous_avg = sum(daily_counts[-7:-3]) / 4
                    tx_momentum = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                    # Normalize to -1 to 1
                    tx_momentum = max(min(tx_momentum * 5, 1), -1)
                    momentum_factors.append(tx_momentum)
            
            # Network growth
            if "network" in metrics and "active_addresses" in metrics["network"]:
                active_addresses = metrics["network"]["active_addresses"]
                if len(active_addresses) >= 7:
                    recent_avg = sum(active_addresses[-3:]) / 3
                    previous_avg = sum(active_addresses[-7:-3]) / 4
                    address_momentum = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                    # Normalize to -1 to 1
                    address_momentum = max(min(address_momentum * 5, 1), -1)
                    momentum_factors.append(address_momentum)
            
            # Asset-specific factors
            if asset == "eth":
                # DeFi TVL growth
                if "defi" in metrics and "total_value_locked" in metrics["defi"]:
                    tvl = metrics["defi"]["total_value_locked"]
                    if len(tvl) >= 7:
                        recent_avg = sum(tvl[-3:]) / 3
                        previous_avg = sum(tvl[-7:-3]) / 4
                        tvl_momentum = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                        # Normalize to -1 to 1
                        tvl_momentum = max(min(tvl_momentum * 5, 1), -1)
                        momentum_factors.append(tvl_momentum)
            
            elif asset == "sol":
                # Staking growth
                if "staking" in metrics and "total_staked" in metrics["staking"]:
                    staked = metrics["staking"]["total_staked"]
                    if len(staked) >= 7:
                        recent_avg = sum(staked[-3:]) / 3
                        previous_avg = sum(staked[-7:-3]) / 4
                        staking_momentum = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                        # Normalize to -1 to 1
                        staking_momentum = max(min(staking_momentum * 5, 1), -1)
                        momentum_factors.append(staking_momentum)
            
            # Calculate overall momentum
            if momentum_factors:
                return sum(momentum_factors) / len(momentum_factors)
            else:
                return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating on-chain momentum: {e}", exc_info=True)
            return 0.0
    
    def _calculate_on_chain_sentiment(self, metrics, asset):
        """
        Calculate on-chain sentiment score.
        
        Args:
            metrics: Raw metrics data
            asset: Asset type ("eth" or "sol")
            
        Returns:
            float: Sentiment score (-1 to 1)
        """
        try:
            sentiment_factors = []
            
            # Common factors
            
            # Transaction value trend
            if "transactions" in metrics and "avg_value" in metrics["transactions"]:
                avg_values = metrics["transactions"]["avg_value"]
                if len(avg_values) >= 7:
                    recent_avg = sum(avg_values[-3:]) / 3
                    previous_avg = sum(avg_values[-7:-3]) / 4
                    value_sentiment = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                    # Normalize to -1 to 1
                    value_sentiment = max(min(value_sentiment * 5, 1), -1)
                    sentiment_factors.append(value_sentiment)
            
            # New addresses trend
            if "network" in metrics and "new_addresses" in metrics["network"]:
                new_addresses = metrics["network"]["new_addresses"]
                if len(new_addresses) >= 7:
                    recent_avg = sum(new_addresses[-3:]) / 3
                    previous_avg = sum(new_addresses[-7:-3]) / 4
                    address_sentiment = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                    # Normalize to -1 to 1
                    address_sentiment = max(min(address_sentiment * 5, 1), -1)
                    sentiment_factors.append(address_sentiment)
            
            # Asset-specific factors
            if asset == "eth":
                # Whale activity
                if "whales" in metrics and "whale_transaction_count" in metrics["whales"]:
                    whale_txs = metrics["whales"]["whale_transaction_count"]
                    if len(whale_txs) >= 7:
                        recent_avg = sum(whale_txs[-3:]) / 3
                        previous_avg = sum(whale_txs[-7:-3]) / 4
                        whale_sentiment = (recent_avg - previous_avg) / previous_avg if previous_avg > 0 else 0
                        # Normalize to -1 to 1
                        whale_sentiment = max(min(whale_sentiment * 5, 1), -1)
  <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>