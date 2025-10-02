"""
Stock Price Fetching Service
Handles real-time stock price fetching with caching
"""

import logging
import time
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import concurrent.futures

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

# Set yfinance cache location to /tmp to avoid permission issues
try:
    yf.set_tz_cache_location("/tmp/yf_cache")
except Exception as e:
    logger.debug(f"Could not set yfinance cache location: {e}")


class PriceCache:
    """Simple in-memory cache for stock prices"""
    
    def __init__(self, cache_duration_seconds: int = 60):
        self.cache = {}
        self.cache_duration = cache_duration_seconds
        
    def get(self, symbol: str) -> Optional[Tuple[float, datetime]]:
        """Get cached price and timestamp"""
        if symbol in self.cache:
            price, timestamp = self.cache[symbol]
            if (datetime.now() - timestamp).total_seconds() < self.cache_duration:
                return price, timestamp
        return None
        
    def set(self, symbol: str, price: float):
        """Cache price with current timestamp"""
        self.cache[symbol] = (price, datetime.now())
        
    def clear(self):
        """Clear all cached prices"""
        self.cache.clear()


class StockPriceService:
    """Service for fetching current stock prices"""
    
    def __init__(self, cache_duration_seconds: int = 60):
        self.cache = PriceCache(cache_duration_seconds)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        
    async def get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Get current prices for multiple symbols
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL', 'TSLA'])
            
        Returns:
            Dictionary mapping symbol to current price (or None if unavailable)
        """
        if not symbols:
            return {}
            
        # Check cache first
        prices = {}
        symbols_to_fetch = []
        
        for symbol in symbols:
            cached_result = self.cache.get(symbol.upper())
            if cached_result:
                price, timestamp = cached_result
                prices[symbol.upper()] = price
                logger.debug(f"Using cached price for {symbol}: ${price} (cached at {timestamp})")
            else:
                symbols_to_fetch.append(symbol.upper())
        
        # Fetch uncached symbols
        if symbols_to_fetch:
            logger.info("Fetching prices for %d symbols: %s", len(symbols_to_fetch), symbols_to_fetch)
            fetched_prices = await self._fetch_prices_batch(symbols_to_fetch)
            
            # Update cache and results
            for symbol, price in fetched_prices.items():
                if price is not None:
                    self.cache.set(symbol, price)
                prices[symbol] = price
        
        return prices
    
    async def _fetch_prices_batch(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Fetch prices for multiple symbols in batch
        
        Args:
            symbols: List of symbols to fetch
            
        Returns:
            Dictionary mapping symbol to price (or None if failed)
        """
        def _fetch_sync():
            try:
                # Join symbols for batch request
                symbols_str = " ".join(symbols)
                logger.debug(f"Fetching batch prices for: {symbols_str}")
                
                # Download price data
                data = yf.download(
                    symbols_str,
                    period="1d",
                    interval="1m",
                    progress=False,
                    auto_adjust=True
                )
                
                prices = {}
                
                if data.empty:
                    logger.warning("No price data returned from yfinance")
                    return {symbol: None for symbol in symbols}
                
                # Handle single symbol vs multiple symbols
                if len(symbols) == 1:
                    symbol = symbols[0]
                    if not data.empty and 'Close' in data.columns:
                        # Get the latest available price
                        latest_price = data['Close'].dropna().iloc[-1] if not data['Close'].dropna().empty else None
                        prices[symbol] = float(latest_price) if latest_price is not None else None
                        logger.debug(f"Fetched price for {symbol}: ${prices[symbol]}")
                    else:
                        prices[symbol] = None
                        logger.warning(f"No close price data for {symbol}")
                else:
                    # Multiple symbols
                    for symbol in symbols:
                        try:
                            if ('Close', symbol) in data.columns:
                                close_prices = data[('Close', symbol)].dropna()
                                if not close_prices.empty:
                                    latest_price = close_prices.iloc[-1]
                                    prices[symbol] = float(latest_price)
                                    logger.debug(f"Fetched price for {symbol}: ${prices[symbol]}")
                                else:
                                    prices[symbol] = None
                                    logger.warning(f"No close price data for {symbol}")
                            else:
                                prices[symbol] = None
                                logger.warning(f"Symbol {symbol} not found in response")
                        except Exception as e:
                            logger.error(f"Error processing price for {symbol}: {e}")
                            prices[symbol] = None
                
                return prices
                
            except Exception as e:
                logger.error(f"Error fetching prices for {symbols}: {e}")
                return {symbol: None for symbol in symbols}
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _fetch_sync)
    
    async def get_single_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a single symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')

        Returns:
            Current price or None if unavailable
        """
        prices = await self.get_current_prices([symbol])
        return prices.get(symbol.upper())

    def clear_cache(self):
        """Clear the price cache"""
        self.cache.clear()
        logger.info("Price cache cleared")


# Global instance
price_service = StockPriceService()


async def get_stock_prices(symbols: List[str]) -> Dict[str, Optional[float]]:
    """
    Convenience function to get stock prices
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        Dictionary mapping symbol to current price
    """
    return await price_service.get_current_prices(symbols)


async def get_stock_price(symbol: str) -> Optional[float]:
    """
    Convenience function to get a single stock price
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Current price or None if unavailable
    """
    return await price_service.get_single_price(symbol)