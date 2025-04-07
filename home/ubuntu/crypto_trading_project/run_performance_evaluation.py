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
from src.performance_evaluator import PerformanceEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("performance_evaluation")

def run_performance_evaluation():
    """
    Run comprehensive performance evaluation of all trading strategies.
    """
    logger.info("Starting performance evaluation")
    
    # Create output directory
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Create performance evaluator
    evaluator = PerformanceEvaluator(output_dir="evaluation_results")
    
    # Create strategies
    strategies = [
        SpikeDetectionStrategy(threshold=0.7, holding_period=5),
        MovingAverageCrossoverStrategy(fast_period=20, slow_period=50),
        AdvancedTradingStrategy(risk_per_trade=0.02, max_portfolio_risk=0.1),
        AdaptiveTradingStrategy(base_risk_per_trade=0.02)
    ]
    
    # Define symbols to test
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    # Define timeframes to test
    timeframes = ["1H", "4H", "1D"]
    
    # Define test periods
    test_periods = [
        {
            "name": "recent_month",
            "start_date": "2023-02-01",
            "end_date": "2023-02-28"
        },
        {
            "name": "bull_market",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31"
        },
        {
            "name": "bear_market",
            "start_date": "2022-11-01",
            "end_date": "2022-11-30"
        }
    ]
    
    # Run strategy comparison for each symbol, timeframe, and period
    results = {}
    
    for symbol in symbols:
        results[symbol] = {}
        
        for timeframe in timeframes:
            results[symbol][timeframe] = {}
            
            for period in test_periods:
                period_name = period["name"]
                start_date = period["start_date"]
                end_date = period["end_date"]
                
                logger.info(f"Evaluating strategies for {symbol} {timeframe} during {period_name}")
                
                # Compare strategies
                comparison_results = evaluator.compare_strategies(
                    strategies, symbol, start_date, end_date, timeframe
                )
                
                if comparison_results:
                    results[symbol][timeframe][period_name] = comparison_results
                    
                    # Log best strategy
                    comparison = comparison_results.get('comparison', {})
                    best_strategy = comparison.get('overall_best_strategy', '')
                    
                    logger.info(f"Best strategy for {symbol} {timeframe} during {period_name}: {best_strategy}")
    
    # Optimize parameters for the best strategy
    best_strategy_counts = {}
    
    # Count which strategy performed best most often
    for symbol in results:
        for timeframe in results[symbol]:
            for period in results[symbol][timeframe]:
                comparison = results[symbol][timeframe][period].get('comparison', {})
                best_strategy = comparison.get('overall_best_strategy', '')
                
                if best_strategy:
                    best_strategy_counts[best_strategy] = best_strategy_counts.get(best_strategy, 0) + 1
    
    # Find the overall best strategy
    overall_best_strategy = max(best_strategy_counts.items(), key=lambda x: x[1])[0] if best_strategy_counts else None
    
    logger.info(f"Overall best strategy across all tests: {overall_best_strategy}")
    
    # Optimize parameters for the best strategy
    if overall_best_strategy == "SpikeDetection":
        parameter_grid = {
            'threshold': [0.6, 0.65, 0.7, 0.75, 0.8],
            'holding_period': [3, 4, 5, 6, 7],
            'stop_loss': [0.01, 0.02, 0.03],
            'take_profit': [0.03, 0.05, 0.07]
        }
        strategy_class = SpikeDetectionStrategy
    elif overall_best_strategy == "MACrossover":
        parameter_grid = {
            'fast_period': [10, 15, 20, 25, 30],
            'slow_period': [40, 50, 60, 70, 80],
            'stop_loss': [0.01, 0.02, 0.03],
            'take_profit': [0.03, 0.05, 0.07]
        }
        strategy_class = MovingAverageCrossoverStrategy
    elif overall_best_strategy == "AdvancedTrading":
        parameter_grid = {
            'risk_per_trade': [0.01, 0.015, 0.02, 0.025, 0.03],
            'max_portfolio_risk': [0.05, 0.075, 0.1, 0.125, 0.15],
            'max_drawdown': [0.1, 0.15, 0.2],
            'atr_multiplier': [1.5, 2.0, 2.5],
            'min_risk_reward': [1.5, 2.0, 2.5]
        }
        strategy_class = AdvancedTradingStrategy
    elif overall_best_strategy == "AdaptiveTrading":
        parameter_grid = {
            'base_risk_per_trade': [0.01, 0.015, 0.02, 0.025, 0.03],
            'max_risk_per_trade': [0.03, 0.04, 0.05],
            'min_risk_per_trade': [0.005, 0.0075, 0.01]
        }
        strategy_class = AdaptiveTradingStrategy
    else:
        parameter_grid = None
        strategy_class = None
    
    # Run optimization if we have a best strategy
    if parameter_grid and strategy_class:
        # Select a representative symbol, timeframe, and period for optimization
        symbol = "BTC/USD"
        timeframe = "1H"
        start_date = "2023-01-01"
        end_date = "2023-02-28"
        
        logger.info(f"Optimizing parameters for {strategy_class.__name__}")
        
        # Simplify parameter grid for faster optimization
        simplified_grid = {}
        for param, values in parameter_grid.items():
            simplified_grid[param] = values[:3]  # Take first 3 values
        
        # Run optimization
        optimization_results = evaluator.optimize_strategy_parameters(
            strategy_class, simplified_grid, symbol, start_date, end_date, timeframe
        )
        
        if optimization_results:
            optimization_report = optimization_results.get('optimization_report', {})
            best_parameters = optimization_report.get('best_parameters', {})
            
            logger.info(f"Best parameters for {strategy_class.__name__}: {best_parameters}")
            
            # Create strategy with optimized parameters
            optimized_strategy = strategy_class(**best_parameters)
            
            # Run final evaluation with optimized strategy
            logger.info(f"Running final evaluation with optimized strategy")
            
            final_evaluation = evaluator.run_strategy_evaluation(
                optimized_strategy, symbol, start_date, end_date, timeframe
            )
            
            if final_evaluation:
                report = final_evaluation.get('report', {})
                metrics = report.get('metrics', {})
                
                logger.info(f"Final evaluation results:")
                logger.info(f"Total return: {metrics.get('returns', {}).get('total_return', '0%')}")
                logger.info(f"Sharpe ratio: {metrics.get('risk', {}).get('sharpe_ratio', '0.0')}")
                logger.info(f"Max drawdown: {metrics.get('risk', {}).get('max_drawdown', '0%')}")
                logger.info(f"Win rate: {metrics.get('trading', {}).get('win_rate', '0%')}")
    
    # Generate summary report
    summary = {
        "evaluation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "symbols_tested": symbols,
        "timeframes_tested": timeframes,
        "periods_tested": [p["name"] for p in test_periods],
        "strategies_tested": [s.name for s in strategies],
        "best_strategy_counts": best_strategy_counts,
        "overall_best_strategy": overall_best_strategy
    }
    
    # Add best strategy for each symbol and timeframe
    summary["best_by_symbol_timeframe"] = {}
    
    for symbol in results:
        summary["best_by_symbol_timeframe"][symbol] = {}
        
        for timeframe in results[symbol]:
            best_strategies = {}
            
            for period in results[symbol][timeframe]:
                comparison = results[symbol][timeframe][period].get('comparison', {})
                best_strategy = comparison.get('overall_best_strategy', '')
                
                if best_strategy:
                    best_strategies[period] = best_strategy
            
            # Find most common best strategy for this symbol and timeframe
            if best_strategies:
                strategy_counts = {}
                for strategy in best_strategies.values():
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                
                most_common = max(strategy_counts.items(), key=lambda x: x[1])[0]
                summary["best_by_symbol_timeframe"][symbol][timeframe] = most_common
    
    # Save summary report
    summary_path = f"evaluation_results/performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Saved performance summary to {summary_path}")
    
    return {
        "results": results,
        "summary": summary,
        "summary_path": summary_path
    }

if __name__ == "__main__":
    run_performance_evaluation()
