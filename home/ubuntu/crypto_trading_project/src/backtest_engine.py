import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from src.time_series_db import TimeSeriesDatabase
from src.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.ml_model_manager import MLModelManager

class BacktestEngine:
    """
    Backtesting engine for cryptocurrency trading strategies.
    Simulates trading on historical data to evaluate strategy performance.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the BacktestEngine.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.logger = logging.getLogger("backtest_engine")
        
        # Initialize database
        self.db = TimeSeriesDatabase(db_path)
        
        # Initialize feature engineering pipeline
        self.feature_pipeline = FeatureEngineeringPipeline(db_path)
        
        # Initialize ML model manager
        self.model_manager = MLModelManager()
        
        # Create directories if they don't exist
        os.makedirs("backtest_results", exist_ok=True)
        
        self.logger.info("BacktestEngine initialized")
    
    def run_backtest(self, strategy, symbol, start_date, end_date, timeframe='1Min',
                    initial_capital=10000.0, commission=0.001, slippage=0.0005):
        """
        Run a backtest for a trading strategy.
        
        Args:
            strategy: Trading strategy instance
            symbol (str): Cryptocurrency symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            timeframe (str): Timeframe of the data
            initial_capital (float): Initial capital for the backtest
            commission (float): Commission rate per trade
            slippage (float): Slippage rate per trade
            
        Returns:
            Dict with backtest results
        """
        try:
            self.logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
            
            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Get historical data
            df = self.db.get_bars(symbol, start_timestamp, end_timestamp, timeframe)
            
            if df.empty:
                self.logger.warning(f"No data found for {symbol}")
                return {}
            
            # Generate features
            features_df = self.feature_pipeline.process_symbol(
                symbol, start_timestamp, end_timestamp, timeframe,
                normalize=False, save_to_file=False
            )
            
            if features_df.empty:
                self.logger.warning(f"No features generated for {symbol}")
                return {}
            
            # Initialize strategy
            strategy.initialize(
                symbol=symbol,
                timeframe=timeframe,
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage
            )
            
            # Run backtest
            results = self._run_backtest_simulation(strategy, features_df)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(results)
            
            # Save results
            self._save_backtest_results(strategy, symbol, timeframe, results, performance)
            
            return {
                'results': results,
                'performance': performance
            }
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}", exc_info=True)
            return {}
    
    def _run_backtest_simulation(self, strategy, features_df):
        """
        Run the backtest simulation.
        
        Args:
            strategy: Trading strategy instance
            features_df: DataFrame with features
            
        Returns:
            Dict with simulation results
        """
        try:
            self.logger.info("Running backtest simulation")
            
            # Initialize results
            results = {
                'timestamp': [],
                'price': [],
                'position': [],
                'cash': [],
                'equity': [],
                'returns': [],
                'trade_type': [],
                'trade_price': [],
                'trade_size': [],
                'trade_cost': [],
                'trade_pnl': []
            }
            
            # Initialize variables
            position = 0
            cash = strategy.initial_capital
            equity = cash
            last_equity = equity
            
            # Iterate through each bar
            for idx, row in features_df.iterrows():
                # Update strategy with current bar
                strategy.on_bar(idx, row)
                
                # Get current price
                price = row['close']
                
                # Get strategy signals
                signals = strategy.generate_signals(idx, row)
                
                # Process signals
                for signal in signals:
                    # Get signal details
                    signal_type = signal['type']  # 'buy', 'sell', 'exit'
                    signal_size = signal['size']  # Number of units
                    
                    # Calculate trade details
                    trade_price = price * (1 + strategy.slippage if signal_type == 'buy' else 1 - strategy.slippage)
                    trade_cost = trade_price * signal_size * strategy.commission
                    
                    # Execute trade
                    if signal_type == 'buy':
                        # Check if we have enough cash
                        max_size = cash / (trade_price * (1 + strategy.commission))
                        actual_size = min(signal_size, max_size)
                        
                        # Update position and cash
                        position += actual_size
                        cash -= (trade_price * actual_size + trade_cost)
                        
                        # Record trade
                        results['trade_type'].append('buy')
                        results['trade_price'].append(trade_price)
                        results['trade_size'].append(actual_size)
                        results['trade_cost'].append(trade_cost)
                        results['trade_pnl'].append(0)
                        
                    elif signal_type == 'sell':
                        # Check if we have enough position
                        actual_size = min(signal_size, position)
                        
                        # Calculate PnL
                        trade_pnl = (trade_price - strategy.avg_entry_price) * actual_size - trade_cost
                        
                        # Update position and cash
                        position -= actual_size
                        cash += (trade_price * actual_size - trade_cost)
                        
                        # Record trade
                        results['trade_type'].append('sell')
                        results['trade_price'].append(trade_price)
                        results['trade_size'].append(actual_size)
                        results['trade_cost'].append(trade_cost)
                        results['trade_pnl'].append(trade_pnl)
                        
                    elif signal_type == 'exit':
                        # Exit all position
                        if position > 0:
                            # Calculate PnL
                            trade_pnl = (trade_price - strategy.avg_entry_price) * position - trade_cost
                            
                            # Update position and cash
                            cash += (trade_price * position - trade_cost)
                            position = 0
                            
                            # Record trade
                            results['trade_type'].append('exit')
                            results['trade_price'].append(trade_price)
                            results['trade_size'].append(position)
                            results['trade_cost'].append(trade_cost)
                            results['trade_pnl'].append(trade_pnl)
                
                # Calculate equity
                equity = cash + position * price
                
                # Calculate returns
                returns = (equity / last_equity) - 1 if last_equity > 0 else 0
                last_equity = equity
                
                # Record results
                results['timestamp'].append(idx)
                results['price'].append(price)
                results['position'].append(position)
                results['cash'].append(cash)
                results['equity'].append(equity)
                results['returns'].append(returns)
            
            # Convert results to DataFrame
            results_df = pd.DataFrame({
                'timestamp': results['timestamp'],
                'price': results['price'],
                'position': results['position'],
                'cash': results['cash'],
                'equity': results['equity'],
                'returns': results['returns']
            })
            
            # Create trades DataFrame
            if results['trade_type']:
                trades_df = pd.DataFrame({
                    'type': results['trade_type'],
                    'price': results['trade_price'],
                    'size': results['trade_size'],
                    'cost': results['trade_cost'],
                    'pnl': results['trade_pnl']
                })
            else:
                trades_df = pd.DataFrame(columns=['type', 'price', 'size', 'cost', 'pnl'])
            
            return {
                'results_df': results_df,
                'trades_df': trades_df,
                'final_equity': equity,
                'final_position': position,
                'final_cash': cash
            }
            
        except Exception as e:
            self.logger.error(f"Error running backtest simulation: {e}", exc_info=True)
            return {}
    
    def _calculate_performance_metrics(self, results):
        """
        Calculate performance metrics for the backtest.
        
        Args:
            results: Dict with simulation results
            
        Returns:
            Dict with performance metrics
        """
        try:
            self.logger.info("Calculating performance metrics")
            
            # Get results DataFrame
            results_df = results.get('results_df')
            trades_df = results.get('trades_df')
            
            if results_df is None or results_df.empty:
                self.logger.warning("No results to calculate performance metrics")
                return {}
            
            # Calculate returns
            initial_equity = results_df['equity'].iloc[0]
            final_equity = results_df['equity'].iloc[-1]
            total_return = (final_equity / initial_equity) - 1
            
            # Calculate annualized return
            days = (results_df['timestamp'].iloc[-1] - results_df['timestamp'].iloc[0]).days
            if days > 0:
                annual_return = (1 + total_return) ** (365 / days) - 1
            else:
                annual_return = 0
            
            # Calculate drawdown
            results_df['peak'] = results_df['equity'].cummax()
            results_df['drawdown'] = (results_df['equity'] / results_df['peak']) - 1
            max_drawdown = results_df['drawdown'].min()
            
            # Calculate Sharpe ratio
            if results_df['returns'].std() > 0:
                sharpe_ratio = results_df['returns'].mean() / results_df['returns'].std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Calculate trade statistics
            if not trades_df.empty:
                num_trades = len(trades_df)
                profitable_trades = len(trades_df[trades_df['pnl'] > 0])
                win_rate = profitable_trades / num_trades if num_trades > 0 else 0
                avg_profit = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if profitable_trades > 0 else 0
                avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if num_trades - profitable_trades > 0 else 0
                profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if trades_df[trades_df['pnl'] <= 0]['pnl'].sum() != 0 else float('inf')
            else:
                num_trades = 0
                win_rate = 0
                avg_profit = 0
                avg_loss = 0
                profit_factor = 0
            
            # Calculate other metrics
            volatility = results_df['returns'].std() * np.sqrt(252)
            sortino_ratio = results_df['returns'].mean() / results_df[results_df['returns'] < 0]['returns'].std() * np.sqrt(252) if len(results_df[results_df['returns'] < 0]) > 0 and results_df[results_df['returns'] < 0]['returns'].std() > 0 else 0
            
            # Compile performance metrics
            performance = {
                'initial_equity': initial_equity,
                'final_equity': final_equity,
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'volatility': volatility,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            }
            
            self.logger.info(f"Performance metrics: {performance}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}", exc_info=True)
            return {}
    
    def _save_backtest_results(self, strategy, symbol, timeframe, results, performance):
        """
        Save backtest results to files.
        
        Args:
            strategy: Trading strategy instance
            symbol (str): Cryptocurrency symbol
            timeframe (str): Timeframe of the data
            results: Dict with simulation results
            performance: Dict with performance metrics
            
        Returns:
            Dict with saved file paths
        """
        try:
            self.logger.info("Saving backtest results")
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Generate base filename
            base_filename = f"{symbol.replace('/', '_')}_{timeframe}_{strategy.name}_{timestamp}"
            
            # Save results DataFrame
            results_file = f"backtest_results/{base_filename}_results.csv"
            results['results_df'].to_csv(results_file)
            
            # Save trades DataFrame
            trades_file = f"backtest_results/{base_filename}_trades.csv"
            results['trades_df'].to_csv(trades_file)
            
            # Save performance metrics
            import json
            performance_file = f"backtest_results/{base_filename}_performance.json"
            with open(performance_file, 'w') as f:
                json.dump(performance, f, indent=4)
            
            # Save strategy configu<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>