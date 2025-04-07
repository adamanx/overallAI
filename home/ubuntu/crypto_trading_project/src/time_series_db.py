import os
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
import sqlite3
from pathlib import Path

class TimeSeriesDatabase:
    """
    Time-series database for storing and retrieving cryptocurrency market data.
    Implements efficient storage and querying for both real-time and historical data.
    """
    
    def __init__(self, db_path=None):
        """
        Initialize the TimeSeriesDatabase.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.logger = logging.getLogger("time_series_db")
        self.db_path = db_path or "data/market_data.db"
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"TimeSeriesDatabase initialized at {self.db_path}")
    
    def _init_database(self):
        """
        Initialize the database schema.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            
            # Bars table (OHLCV data)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                timeframe TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                trade_count INTEGER,
                vwap REAL,
                source TEXT DEFAULT 'alpaca',
                UNIQUE(symbol, timestamp, timeframe)
            )
            ''')
            
            # Trades table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                trade_id TEXT,
                exchange TEXT,
                source TEXT DEFAULT 'alpaca',
                UNIQUE(symbol, timestamp, trade_id)
            )
            ''')
            
            # Quotes table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                bid_price REAL NOT NULL,
                bid_size REAL NOT NULL,
                ask_price REAL NOT NULL,
                ask_size REAL NOT NULL,
                exchange TEXT,
                source TEXT DEFAULT 'alpaca',
                UNIQUE(symbol, timestamp)
            )
            ''')
            
            # Create indexes for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bars_symbol_timestamp ON bars (symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades (symbol, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_quotes_symbol_timestamp ON quotes (symbol, timestamp)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database schema initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}", exc_info=True)
            raise
    
    def store_bar(self, bar_data):
        """
        Store bar data in the database.
        
        Args:
            bar_data (dict): Bar data from Alpaca or other sources
            
        Returns:
            bool: Success status
        """
        try:
            # Normalize data format
            normalized_data = self._normalize_bar_data(bar_data)
            if not normalized_data:
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert data
            cursor.execute('''
            INSERT OR REPLACE INTO bars 
            (symbol, timestamp, timeframe, open, high, low, close, volume, trade_count, vwap, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                normalized_data['symbol'],
                normalized_data['timestamp'],
                normalized_data['timeframe'],
                normalized_data['open'],
                normalized_data['high'],
                normalized_data['low'],
                normalized_data['close'],
                normalized_data['volume'],
                normalized_data.get('trade_count'),
                normalized_data.get('vwap'),
                normalized_data.get('source', 'alpaca')
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing bar data: {e}", exc_info=True)
            return False
    
    def store_trade(self, trade_data):
        """
        Store trade data in the database.
        
        Args:
            trade_data (dict): Trade data from Alpaca or other sources
            
        Returns:
            bool: Success status
        """
        try:
            # Normalize data format
            normalized_data = self._normalize_trade_data(trade_data)
            if not normalized_data:
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert data
            cursor.execute('''
            INSERT OR REPLACE INTO trades 
            (symbol, timestamp, price, size, trade_id, exchange, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                normalized_data['symbol'],
                normalized_data['timestamp'],
                normalized_data['price'],
                normalized_data['size'],
                normalized_data.get('trade_id'),
                normalized_data.get('exchange'),
                normalized_data.get('source', 'alpaca')
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing trade data: {e}", exc_info=True)
            return False
    
    def store_quote(self, quote_data):
        """
        Store quote data in the database.
        
        Args:
            quote_data (dict): Quote data from Alpaca or other sources
            
        Returns:
            bool: Success status
        """
        try:
            # Normalize data format
            normalized_data = self._normalize_quote_data(quote_data)
            if not normalized_data:
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert data
            cursor.execute('''
            INSERT OR REPLACE INTO quotes 
            (symbol, timestamp, bid_price, bid_size, ask_price, ask_size, exchange, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                normalized_data['symbol'],
                normalized_data['timestamp'],
                normalized_data['bid_price'],
                normalized_data['bid_size'],
                normalized_data['ask_price'],
                normalized_data['ask_size'],
                normalized_data.get('exchange'),
                normalized_data.get('source', 'alpaca')
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing quote data: {e}", exc_info=True)
            return False
    
    def get_bars(self, symbol, start_timestamp, end_timestamp, timeframe='1Min'):
        """
        Retrieve bar data from the database.
        
        Args:
            symbol (str): Cryptocurrency symbol
            start_timestamp (int): Start timestamp in milliseconds
            end_timestamp (int): End timestamp in milliseconds
            timeframe (str): Timeframe of the bars
            
        Returns:
            pandas.DataFrame: Bar data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
            SELECT timestamp, open, high, low, close, volume, trade_count, vwap
            FROM bars
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ? AND timeframe = ?
            ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(symbol, start_timestamp, end_timestamp, timeframe)
            )
            
            conn.close()
            
            if not df.empty:
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving bar data: {e}", exc_info=True)
            return pd.DataFrame()
    
    def get_trades(self, symbol, start_timestamp, end_timestamp):
        """
        Retrieve trade data from the database.
        
        Args:
            symbol (str): Cryptocurrency symbol
            start_timestamp (int): Start timestamp in milliseconds
            end_timestamp (int): End timestamp in milliseconds
            
        Returns:
            pandas.DataFrame: Trade data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
            SELECT timestamp, price, size, trade_id, exchange
            FROM trades
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(symbol, start_timestamp, end_timestamp)
            )
            
            conn.close()
            
            if not df.empty:
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving trade data: {e}", exc_info=True)
            return pd.DataFrame()
    
    def get_quotes(self, symbol, start_timestamp, end_timestamp):
        """
        Retrieve quote data from the database.
        
        Args:
            symbol (str): Cryptocurrency symbol
            start_timestamp (int): Start timestamp in milliseconds
            end_timestamp (int): End timestamp in milliseconds
            
        Returns:
            pandas.DataFrame: Quote data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
            SELECT timestamp, bid_price, bid_size, ask_price, ask_size, exchange
            FROM quotes
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            '''
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(symbol, start_timestamp, end_timestamp)
            )
            
            conn.close()
            
            if not df.empty:
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving quote data: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _normalize_bar_data(self, bar_data):
        """
        Normalize bar data to a standard format.
        
        Args:
            bar_data (dict): Bar data from Alpaca or other sources
            
        Returns:
            dict: Normalized bar data
        """
        try:
            # Handle Alpaca format
            if 'S' in bar_data and 't' in bar_data:
                return {
                    'symbol': bar_data['S'],
                    'timestamp': bar_data['t'],
                    'timeframe': '1Min',  # Default for streaming data
                    'open': bar_data['o'],
                    'high': bar_data['h'],
                    'low': bar_data['l'],
                    'close': bar_data['c'],
                    'volume': bar_data['v'],
                    'trade_count': bar_data.get('n'),
                    'vwap': bar_data.get('vw'),
                    'source': 'alpaca'
                }
            
            # Handle already normalized format
            elif all(k in bar_data for k in ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']):
                return {
                    'symbol': bar_data['symbol'],
                    'timestamp': bar_data['timestamp'],
                    'timeframe': bar_data.get('timeframe', '1Min'),
                    'open': bar_data['open'],
                    'high': bar_data['high'],
                    'low': bar_data['low'],
                    'close': bar_data['close'],
                    'volume': bar_data['volume'],
                    'trade_count': bar_data.get('trade_count'),
                    'vwap': bar_data.get('vwap'),
                    'source': bar_data.get('source', 'unknown')
                }
            
            else:
                self.logger.error(f"Unknown bar data format: {bar_data}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error normalizing bar data: {e}", exc_info=True)
            return None
    
    def _normalize_trade_data(self, trade_data):
        """
        Normalize trade data to a standard format.
        
        Args:
            trade_data (dict): Trade data from Alpaca or other sources
            
        Returns:
            dict: Normalized trade data
        """
        try:
            # Handle Alpaca format
            if 'S' in trade_data and 't' in trade_data:
                return {
                    'symbol': trade_data['S'],
                    'timestamp': trade_data['t'],
                    'price': trade_data['p'],
                    'size': trade_data['s'],
                    'trade_id': trade_data.get('i'),
                    'exchange': trade_data.get('x'),
                    'source': 'alpaca'
                }
            
            # Handle already normalized format
            elif all(k in trade_data for k in ['symbol', 'timestamp', 'price', 'size']):
                return {
                    'symbol': trade_data['symbol'],
                    'timestamp': trade_data['timestamp'],
                    'price': trade_data['price'],
                    'size': trade_data['size'],
                    'trade_id': trade_data.get('trade_id'),
                    'exchange': trade_data.get('exchange'),
                    'source': trade_data.get('source', 'unknown')
                }
            
            else:
                self.logger.error(f"Unknown trade data format: {trade_data}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error normalizing trade data: {e}", exc_info=True)
            return None
    
    def _normalize_quote_data(self, quote_data):
        """
        Normalize quote data to a sta<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>