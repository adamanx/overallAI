import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from src.backtest_engine import BacktestEngine
from src.trading_strategy import AdvancedTradingStrategy, AdaptiveTradingStrategy
from src.backtest_engine import SpikeDetectionStrategy, MovingAverageCrossoverStrategy
from src.time_series_db import TimeSeriesDatabase
from src.feature_engineering_pipeline import FeatureEngineeringPipeline
from src.ml_model_manager import MLModelManager

class PerformanceEvaluator:
    """
    Performance evaluation system for cryptocurrency trading strategies.
    Runs backtests, calculates metrics, and generates reports.
    """
    
    def __init__(self, output_dir="performance_results"):
        """
        Initialize the PerformanceEvaluator.
        
        Args:
            output_dir: Directory to save performance results
        """
        self.logger = logging.getLogger("performance_evaluator")
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.backtest_engine = BacktestEngine()
        self.db = TimeSeriesDatabase()
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.model_manager = MLModelManager()
        
        self.logger.info("PerformanceEvaluator initialized")
    
    def run_strategy_evaluation(self, strategy, symbol, start_date, end_date, 
                              timeframe='1Min', initial_capital=10000.0,
                              commission=0.001, slippage=0.0005):
        """
        Run a comprehensive evaluation of a trading strategy.
        
        Args:
            strategy: Trading strategy instance
            symbol: Cryptocurrency symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe of the data
            initial_capital: Initial capital for the backtest
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            
        Returns:
            Dict with evaluation results
        """
        try:
            self.logger.info(f"Evaluating {strategy.name} on {symbol} from {start_date} to {end_date}")
            
            # Run backtest
            backtest_results = self.backtest_engine.run_backtest(
                strategy, symbol, start_date, end_date, timeframe,
                initial_capital, commission, slippage
            )
            
            if not backtest_results:
                self.logger.warning(f"No backtest results for {strategy.name} on {symbol}")
                return {}
            
            # Extract performance metrics
            performance = backtest_results.get('performance', {})
            
            # Generate performance report
            report = self._generate_performance_report(strategy, symbol, timeframe, performance)
            
            # Generate equity curve chart
            results_df = backtest_results.get('results', {}).get('results_df')
            if results_df is not None and not results_df.empty:
                chart_path = self._generate_equity_curve(
                    results_df, strategy.name, symbol, timeframe
                )
                report['equity_curve_chart'] = chart_path
            
            # Save report
            report_path = self._save_performance_report(report, strategy.name, symbol, timeframe)
            
            return {
                'backtest_results': backtest_results,
                'report': report,
                'report_path': report_path
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategy: {e}", exc_info=True)
            return {}
    
    def _generate_performance_report(self, strategy, symbol, timeframe, performance):
        """
        Generate a performance report for a strategy.
        
        Args:
            strategy: Trading strategy instance
            symbol: Cryptocurrency symbol
            timeframe: Timeframe of the data
            performance: Dict with performance metrics
            
        Returns:
            Dict with performance report
        """
        try:
            # Format metrics for report
            report = {
                'strategy_name': strategy.name,
                'symbol': symbol,
                'timeframe': timeframe,
                'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics': {
                    'returns': {
                        'total_return': f"{performance.get('total_return', 0) * 100:.2f}%",
                        'annual_return': f"{performance.get('annual_return', 0) * 100:.2f}%",
                        'initial_equity': f"${performance.get('initial_equity', 0):.2f}",
                        'final_equity': f"${performance.get('final_equity', 0):.2f}"
                    },
                    'risk': {
                        'max_drawdown': f"{performance.get('max_drawdown', 0) * 100:.2f}%",
                        'volatility': f"{performance.get('volatility', 0) * 100:.2f}%",
                        'sharpe_ratio': f"{performance.get('sharpe_ratio', 0):.2f}",
                        'sortino_ratio': f"{performance.get('sortino_ratio', 0):.2f}"
                    },
                    'trading': {
                        'num_trades': performance.get('num_trades', 0),
                        'win_rate': f"{performance.get('win_rate', 0) * 100:.2f}%",
                        'profit_factor': f"{performance.get('profit_factor', 0):.2f}",
                        'avg_profit': f"${performance.get('avg_profit', 0):.2f}",
                        'avg_loss': f"${performance.get('avg_loss', 0):.2f}"
                    }
                },
                'strategy_config': strategy.get_config()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}", exc_info=True)
            return {}
    
    def _generate_equity_curve(self, results_df, strategy_name, symbol, timeframe):
        """
        Generate an equity curve chart.
        
        Args:
            results_df: DataFrame with backtest results
            strategy_name: Strategy name
            symbol: Cryptocurrency symbol
            timeframe: Timeframe of the data
            
        Returns:
            Path to the saved chart
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot equity curve
            plt.subplot(2, 1, 1)
            plt.plot(results_df['equity'], label='Equity')
            plt.title(f"Equity Curve - {strategy_name} on {symbol} ({timeframe})")
            plt.xlabel('Time')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.legend()
            
            # Plot drawdown
            plt.subplot(2, 1, 2)
            plt.plot(results_df['drawdown'] * 100, color='red', label='Drawdown')
            plt.fill_between(results_df.index, results_df['drawdown'] * 100, 0, color='red', alpha=0.3)
            plt.title('Drawdown')
            plt.xlabel('Time')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = f"{self.output_dir}/{symbol.replace('/', '_')}_{timeframe}_{strategy_name}_{timestamp}_equity.png"
            plt.savefig(chart_path)
            plt.close()
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error generating equity curve: {e}", exc_info=True)
            return None
    
    def _save_performance_report(self, report, strategy_name, symbol, timeframe):
        """
        Save performance report to file.
        
        Args:
            report: Dict with performance report
            strategy_name: Strategy name
            symbol: Cryptocurrency symbol
            timeframe: Timeframe of the data
            
        Returns:
            Path to the saved report
        """
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Generate filename
            filename = f"{self.output_dir}/{symbol.replace('/', '_')}_{timeframe}_{strategy_name}_{timestamp}_report.json"
            
            # Save report
            with open(filename, 'w') as f:
                json.dump(report, f, indent=4)
            
            self.logger.info(f"Saved performance report to {filename}")
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving performance report: {e}", exc_info=True)
            return None
    
    def compare_strategies(self, strategies, symbol, start_date, end_date, 
                         timeframe='1Min', initial_capital=10000.0,
                         commission=0.001, slippage=0.0005):
        """
        Compare multiple trading strategies.
        
        Args:
            strategies: List of trading strategy instances
            symbol: Cryptocurrency symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe of the data
            initial_capital: Initial capital for the backtest
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            
        Returns:
            Dict with comparison results
        """
        try:
            self.logger.info(f"Comparing {len(strategies)} strategies on {symbol}")
            
            # Run backtest for each strategy
            results = {}
            
            for strategy in strategies:
                strategy_name = strategy.name
                self.logger.info(f"Evaluating strategy: {strategy_name}")
                
                # Run evaluation
                evaluation_results = self.run_strategy_evaluation(
                    strategy, symbol, start_date, end_date, timeframe,
                    initial_capital, commission, slippage
                )
                
                if evaluation_results:
                    results[strategy_name] = evaluation_results
            
            # Compare strategies
            comparison = self._compare_strategies(results)
            
            # Generate comparison chart
            chart_path = self._generate_comparison_chart(results, symbol, timeframe)
            if chart_path:
                comparison['comparison_chart'] = chart_path
            
            # Save comparison results
            comparison_path = self._save_comparison_results(comparison, symbol, timeframe)
            
            return {
                'comparison': comparison,
                'comparison_path': comparison_path,
                'strategy_results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing strategies: {e}", exc_info=True)
            return {}
    
    def _compare_strategies(self, results):
        """
        Compare performance metrics for multiple strategies.
        
        Args:
            results: Dict with evaluation results for each strategy
            
        Returns:
            Dict with comparison results
        """
        try:
            # Initialize comparison
            comparison = {
                'strategies': list(results.keys()),
                'metrics': {}
            }
            
            # Define metrics to compare
            metrics = [
                'total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio',
                'sortino_ratio', 'volatility', 'num_trades', 'win_rate',
                'profit_factor'
            ]
            
            # Compare each metric
            for metric in metrics:
                comparison['metrics'][metric] = {}
                
                for strategy_name, evaluation_results in results.items():
                    backtest_results = evaluation_results.get('backtest_results', {})
                    performance = backtest_results.get('performance', {})
                    comparison['metrics'][metric][strategy_name] = performance.get(metric, 0)
            
            # Determine best strategy for each metric
            comparison['best_strategy'] = {}
            
            for metric in metrics:
                if metric in ['max_drawdown', 'volatility']:
                    # Lower is better
                    best_strategy = min(
                        comparison['metrics'][metric].items(),
                        key=lambda x: x[1]
                    )[0]
                else:
                    # Higher is better
                    best_strategy = max(
                        comparison['metrics'][metric].items(),
                        key=lambda x: x[1]
                    )[0]
                
                comparison['best_strategy'][metric] = best_strategy
            
            # Determine overall best strategy
            # Simple scoring: count how many times each strategy is the best
            scores = {}
            for strategy_name in comparison['strategies']:
                scores[strategy_name] = sum(
                    1 for metric, best in comparison['best_strategy'].items()
                    if best == strategy_name
                )
            
            comparison['overall_best_strategy'] = max(
                scores.items(),
                key=lambda x: x[1]
            )[0]
            
            self.logger.info(f"Overall best strategy: {comparison['overall_best_strategy']}")
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing strategies: {e}", exc_info=True)
            return {}
    
    def _generate_comparison_chart(self, results, symbol, timeframe):
        """
        Generate a comparison chart for multiple strategies.
        
        Args:
            results: Dict with evaluation results for each strategy
            symbol: Cryptocurrency symbol
            timeframe: Timeframe of the data
            
        Returns:
            Path to the saved chart
        """
        try:
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot equity curves
            for strategy_name, evaluation_results in results.items():
                backtest_results = evaluation_results.get('backtest_results', {})
                results_df = backtest_results.get('results', {}).get('results_df')
                
                if results_df is not None and not results_df.empty:
                    plt.plot(results_df['equity'], label=strategy_name)
            
            # Add labels and legend
            plt.title(f"Strategy Comparison - {symbol} ({timeframe})")
            plt.xlabel('Time')
            plt.ylabel('Equity ($)')
            plt.grid(True)
            plt.legend()
            
            # Save chart
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_path = f"{self.output_dir}/{symbol.replace('/', '_')}_{timeframe}_comparison_{timestamp}.png"
            plt.savefig(chart_path)
            plt.close()
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error generating comparison chart: {e}", exc_info=True)
            return None
    
    def _save_comparison_results(self, comparison, symbol, timefr<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>