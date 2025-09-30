"""
Stock Split Detection and Compensation Module
Handles fetching and applying stock split adjustments to historical trades
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
from functools import lru_cache
import json

logger = logging.getLogger(__name__)

# Known major stock splits (as a fallback/cache)
KNOWN_SPLITS = {
    'NVDA': [
        {'date': '2024-06-07', 'ratio': 10.0, 'description': '10-for-1 split'},
        {'date': '2021-07-20', 'ratio': 4.0, 'description': '4-for-1 split'}
    ],
    'AAPL': [
        {'date': '2020-08-31', 'ratio': 4.0, 'description': '4-for-1 split'},
        {'date': '2014-06-09', 'ratio': 7.0, 'description': '7-for-1 split'}
    ],
    'TSLA': [
        {'date': '2022-08-25', 'ratio': 3.0, 'description': '3-for-1 split'},
        {'date': '2020-08-31', 'ratio': 5.0, 'description': '5-for-1 split'}
    ],
    'AMZN': [
        {'date': '2022-06-06', 'ratio': 20.0, 'description': '20-for-1 split'}
    ],
    'GOOGL': [
        {'date': '2022-07-18', 'ratio': 20.0, 'description': '20-for-1 split'}
    ],
    'GOOG': [
        {'date': '2022-07-18', 'ratio': 20.0, 'description': '20-for-1 split'}
    ],
    'SHOP': [
        {'date': '2022-06-29', 'ratio': 10.0, 'description': '10-for-1 split'}
    ]
}


class StockSplitDetector:
    """Detect and compensate for stock splits in trading data"""

    def __init__(self, use_api: bool = True):
        """
        Initialize the stock split detector

        Args:
            use_api: Whether to fetch split data from external API (requires API key)
        """
        self.use_api = use_api
        self.split_cache = {}

    @lru_cache(maxsize=100)
    def get_stock_splits(self, symbol: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Get stock split history for a symbol

        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of split events with date and ratio
        """
        # First check known splits
        if symbol in KNOWN_SPLITS:
            splits = KNOWN_SPLITS[symbol]
            logger.info(f"Using cached split data for {symbol}: {len(splits)} splits")
            return splits

        # If API is enabled, try to fetch from external source
        if self.use_api:
            try:
                splits = self._fetch_splits_from_api(symbol, start_date, end_date)
                if splits:
                    return splits
            except Exception as e:
                logger.warning(f"Failed to fetch split data from API for {symbol}: {e}")

        # Return empty list if no splits found
        logger.info(f"No split data found for {symbol}")
        return []

    def _fetch_splits_from_api(self, symbol: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """
        Fetch split data from external API (placeholder for actual implementation)

        Note: In production, you would use a service like:
        - Alpha Vantage API
        - Yahoo Finance API
        - Polygon.io
        - IEX Cloud
        """
        # Placeholder - would need actual API implementation
        logger.info(f"API fetch not implemented, using known splits for {symbol}")
        return []

    def detect_potential_splits(self, trades_df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """
        Detect potential stock splits by analyzing price jumps in trading data

        Args:
            trades_df: DataFrame with columns ['Symbol', 'Date', 'Price']

        Returns:
            Dictionary of symbols with potential split dates and ratios
        """
        potential_splits = {}

        for symbol in trades_df['Symbol'].unique():
            symbol_trades = trades_df[trades_df['Symbol'] == symbol].copy()
            symbol_trades = symbol_trades.sort_values('Date')

            if len(symbol_trades) < 2:
                continue

            # Calculate price ratios between consecutive trades
            prices = symbol_trades['Price'].values
            dates = symbol_trades['Date'].values

            for i in range(1, len(prices)):
                ratio = prices[i-1] / prices[i]

                # Check for significant price drops that might indicate splits
                # Common split ratios: 2, 3, 4, 5, 7, 10, 20
                for split_ratio in [2, 3, 4, 5, 7, 10, 20]:
                    if abs(ratio - split_ratio) < 0.1:  # Within 10% of expected ratio
                        if symbol not in potential_splits:
                            potential_splits[symbol] = []
                        potential_splits[symbol].append((str(dates[i]), split_ratio))
                        logger.warning(f"Potential {split_ratio}-for-1 split detected for {symbol} around {dates[i]}")

        return potential_splits

    def adjust_trades_for_splits(self, trades_df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Adjust trade prices and quantities for stock splits

        Args:
            trades_df: DataFrame with trades to adjust
            symbol: Specific symbol to adjust (None for all)

        Returns:
            DataFrame with split-adjusted prices and quantities
        """
        adjusted_df = trades_df.copy()

        # Get unique symbols to process
        symbols = [symbol] if symbol else adjusted_df['Symbol'].unique()

        for sym in symbols:
            # Get splits for this symbol
            splits = self.get_stock_splits(sym)

            if not splits:
                continue

            # Apply each split adjustment
            for split_info in splits:
                split_date = pd.to_datetime(split_info['date'])
                split_ratio = split_info['ratio']

                # Find trades before the split date
                mask = (adjusted_df['Symbol'] == sym) & (adjusted_df['Date'] < split_date)

                # Adjust prices (divide by split ratio)
                adjusted_df.loc[mask, 'Price'] = adjusted_df.loc[mask, 'Price'] / split_ratio

                # Adjust quantities (multiply by split ratio)
                if 'Quantity' in adjusted_df.columns:
                    adjusted_df.loc[mask, 'Quantity'] = adjusted_df.loc[mask, 'Quantity'] * split_ratio

                logger.info(f"Applied {split_ratio}-for-1 split adjustment for {sym} on {split_date.date()}")

        return adjusted_df

    def adjust_roundtrips_for_splits(self, roundtrips_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust roundtrip entry/exit prices for stock splits

        Args:
            roundtrips_df: DataFrame with roundtrip trades

        Returns:
            DataFrame with split-adjusted roundtrips
        """
        adjusted_df = roundtrips_df.copy()

        for symbol in adjusted_df['Symbol'].unique():
            splits = self.get_stock_splits(symbol)

            if not splits:
                continue

            symbol_mask = adjusted_df['Symbol'] == symbol
            symbol_trades = adjusted_df[symbol_mask]

            for split_info in splits:
                split_date = pd.to_datetime(split_info['date'])
                split_ratio = split_info['ratio']

                # Adjust entry prices if entry date is before split but exit is after
                mask = (
                    (adjusted_df['Symbol'] == symbol) &
                    (adjusted_df['Entry Date'] < split_date) &
                    (adjusted_df['Exit Date'] >= split_date)
                )

                if mask.any():
                    # Adjust entry price for trades that span the split
                    adjusted_df.loc[mask, 'Entry Price'] = adjusted_df.loc[mask, 'Entry Price'] / split_ratio
                    adjusted_df.loc[mask, 'Shares'] = adjusted_df.loc[mask, 'Shares'] * split_ratio

                    logger.info(f"Adjusted {mask.sum()} {symbol} roundtrips for {split_ratio}-for-1 split on {split_date.date()}")

        return adjusted_df


def apply_split_adjustments(trades_df: pd.DataFrame, detect_unknown: bool = True) -> pd.DataFrame:
    """
    Convenience function to apply split adjustments to trades

    Args:
        trades_df: DataFrame with trade data
        detect_unknown: Whether to detect potential unknown splits

    Returns:
        DataFrame with split-adjusted trades
    """
    detector = StockSplitDetector()

    # Detect potential unknown splits if requested
    if detect_unknown:
        potential = detector.detect_potential_splits(trades_df)
        if potential:
            logger.warning(f"Potential unknown splits detected: {potential}")

    # Apply known split adjustments
    return detector.adjust_trades_for_splits(trades_df)


def apply_split_adjustments_to_roundtrips(roundtrips_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to apply split adjustments to roundtrips

    Args:
        roundtrips_df: DataFrame with roundtrip data

    Returns:
        DataFrame with split-adjusted roundtrips
    """
    detector = StockSplitDetector()
    return detector.adjust_roundtrips_for_splits(roundtrips_df)