"""
Unified data manager for handling multiple data sources (CSV, IBKR API)
"""
import logging
from typing import List, Dict, Any, Optional, Union
from io import BytesIO

import pandas as pd
from fastapi import UploadFile

from core.processor import process_ibkr_csv
from core.ibkr_client import FlexQueryClient
from core.calculator import calculate_performance

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data acquisition from multiple sources"""

    def __init__(self):
        """Initialize data manager"""
        pass

    async def get_trades(
        self,
        source: str = "csv",
        files: Optional[List[UploadFile]] = None,
        token: Optional[str] = None,
        query_id: Optional[str] = None,
        historical: bool = False,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get trades from specified source

        Args:
            source: Data source type ('csv' or 'ibkr')
            files: List of uploaded CSV files (for CSV source)
            token: IBKR FlexQuery token (for IBKR source)
            query_id: IBKR FlexQuery ID (for IBKR source)
            historical: Whether to fetch historical data (IBKR only)
            start_year: Starting year for historical data (e.g., 2020)
            end_year: Ending year for historical data (optional)

        Returns:
            Dictionary containing processed trades and roundtrips
        """
        try:
            if source == "csv":
                return await self._process_csv_files(files)
            elif source == "ibkr":
                return await self._fetch_from_ibkr(token, query_id, historical, start_year, end_year)
            elif source == "hybrid":
                return await self._process_hybrid_data(files, token, query_id)
            else:
                raise ValueError(f"Unknown data source: {source}")

        except Exception as e:
            logger.error(f"Error getting trades from {source}: {e}")
            raise

    async def _process_csv_files(self, files: List[UploadFile]) -> Dict[str, Any]:
        """
        Process uploaded CSV files

        Args:
            files: List of uploaded CSV files

        Returns:
            Dictionary containing processed trades and roundtrips
        """
        if not files:
            raise ValueError("No files provided")

        all_dataframes = []

        for file in files:
            try:
                # Read file content
                content = await file.read()
                df = pd.read_csv(BytesIO(content))
                all_dataframes.append(df)
                logger.info(f"Read {len(df)} rows from {file.filename}")
            except Exception as e:
                logger.error(f"Error reading file {file.filename}: {e}")
                raise ValueError(f"Error processing {file.filename}: {str(e)}")

        # Combine all dataframes
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            logger.info(f"Combined {len(combined_df)} total trades from {len(files)} files")

            # Process trades using existing processor
            # Convert DataFrame to CSV string for processing
            csv_content = combined_df.to_csv(index=False)
            roundtrips_df, open_long, open_short = process_ibkr_csv(csv_content)

            # Apply stock split adjustments
            from core.stock_splits import apply_split_adjustments_to_roundtrips
            try:
                roundtrips_df = apply_split_adjustments_to_roundtrips(roundtrips_df)
                logger.info("Applied stock split adjustments to roundtrips")
            except Exception as e:
                logger.warning(f"Could not apply split adjustments: {e}")

            # Calculate performance
            performance_data = calculate_performance(roundtrips_df)

            # Process open positions
            from core.open_positions import process_open_positions
            try:
                open_positions_data = await process_open_positions(open_long, open_short, roundtrips_df)
                logger.info(f"Processed {len(open_positions_data['positions'])} open positions")
            except Exception as e:
                logger.warning(f"Could not process open positions: {e}")
                open_positions_data = {"positions": [], "summary": {}}

            return {
                "roundtrips": roundtrips_df,
                "performance": performance_data,
                "open_positions": open_positions_data
            }
        else:
            raise ValueError("No valid data found in uploaded files")

    async def _fetch_from_ibkr(self, token: str, query_id: str, historical: bool = False,
                               start_year: Optional[int] = None, end_year: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch trades from IBKR FlexQuery API

        Args:
            token: IBKR FlexQuery token
            query_id: IBKR FlexQuery ID
            historical: Whether to fetch historical data across multiple years
            start_year: Starting year for historical data
            end_year: Ending year for historical data

        Returns:
            Dictionary containing processed trades and roundtrips
        """
        if not token or not query_id:
            raise ValueError("IBKR token and query ID are required")

        try:
            if historical and start_year:
                # Fetch historical data across multiple years
                logger.info(f"Fetching historical data from {start_year} to {end_year or 'current year'}")
                trades_df = FlexQueryClient.fetch_historical_data(token, query_id, start_year, end_year)
            else:
                # Standard fetch (current year or default date range)
                client = FlexQueryClient(token, query_id)
                logger.info("Fetching trades from IBKR FlexQuery...")
                trades_df = client.fetch_and_process_trades()

            if trades_df.empty:
                raise ValueError("No trades found in IBKR FlexQuery response")

            logger.info(f"Fetched {len(trades_df)} trades from IBKR")

            # Process ALL trades at once using existing processor
            # Convert DataFrame to CSV string for processing
            csv_content = trades_df.to_csv(index=False)
            roundtrips_df, open_long, open_short = process_ibkr_csv(csv_content)

            logger.info(f"Processed into {len(roundtrips_df)} roundtrips")

            # Apply stock split adjustments
            from core.stock_splits import apply_split_adjustments_to_roundtrips
            try:
                roundtrips_df = apply_split_adjustments_to_roundtrips(roundtrips_df)
                logger.info("Applied stock split adjustments to roundtrips")
            except Exception as e:
                logger.warning(f"Could not apply split adjustments: {e}")

            # Calculate performance on ALL historical data combined
            performance_data = calculate_performance(roundtrips_df)

            # Process open positions
            from core.open_positions import process_open_positions
            try:
                open_positions_data = await process_open_positions(open_long, open_short, roundtrips_df)
                logger.info(f"Processed {len(open_positions_data['positions'])} open positions")
            except Exception as e:
                logger.warning(f"Could not process open positions: {e}")
                open_positions_data = {"positions": [], "summary": {}}

            return {
                "roundtrips": roundtrips_df,
                "performance": performance_data,
                "open_positions": open_positions_data
            }

        except Exception as e:
            logger.error(f"Error fetching from IBKR: {e}")
            raise ValueError(f"Failed to fetch IBKR data: {str(e)}")

    async def test_ibkr_connection(self, token: str, query_id: str) -> Dict[str, Any]:
        """
        Test IBKR FlexQuery connection

        Args:
            token: IBKR FlexQuery token
            query_id: IBKR FlexQuery ID

        Returns:
            Dictionary with test results
        """
        try:
            from core.ibkr_client import test_connection

            success, message = test_connection(token, query_id)

            return {
                "success": success,
                "message": message
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection test failed: {str(e)}"
            }

    async def _process_hybrid_data(self, files: List[UploadFile], token: str, query_id: str) -> Dict[str, Any]:
        """
        Process hybrid data: CSV files (historical) + IBKR current year

        Args:
            files: List of uploaded CSV files (historical data)
            token: IBKR FlexQuery token
            query_id: IBKR FlexQuery ID

        Returns:
            Dictionary containing processed trades and roundtrips with duplicates removed
        """
        logger.info("Processing hybrid data: CSV files + IBKR current year")

        try:
            # 1. Process CSV files first (historical data)
            logger.info("Processing historical CSV files...")
            csv_dataframes = []

            for file in files:
                try:
                    content = await file.read()
                    df = pd.read_csv(BytesIO(content))
                    csv_dataframes.append(df)
                    logger.info(f"Read {len(df)} rows from {file.filename}")
                except Exception as e:
                    logger.error(f"Error reading file {file.filename}: {e}")
                    raise ValueError(f"Error processing {file.filename}: {str(e)}")

            # Combine CSV files
            if csv_dataframes:
                csv_combined = pd.concat(csv_dataframes, ignore_index=True)
                logger.info(f"Combined CSV files: {len(csv_combined)} total trades")
            else:
                csv_combined = pd.DataFrame()

            # 2. Fetch current year data from IBKR
            logger.info("Fetching current year data from IBKR...")
            client = FlexQueryClient(token, query_id)
            ibkr_df = client.fetch_and_process_trades()
            logger.info(f"IBKR data: {len(ibkr_df)} trades")

            # 3. Combine both datasets
            if not csv_combined.empty and not ibkr_df.empty:
                # Map CSV columns to IBKR format for compatibility
                logger.info("Mapping CSV columns to IBKR format...")

                # Map CSV column names to IBKR format and convert dates
                csv_column_mapping = {
                    'TradeDate': 'Trade Date/Time',
                    'TradePrice': 'Trade Price'
                }

                # Apply column mapping to CSV data with proper date conversion
                for csv_col, ibkr_col in csv_column_mapping.items():
                    if csv_col in csv_combined.columns and ibkr_col not in csv_combined.columns:
                        if csv_col == 'TradeDate':
                            # Convert TradeDate from YYYYMMDD integer format to datetime string
                            csv_combined[ibkr_col] = pd.to_datetime(csv_combined[csv_col], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                            logger.info(f"Mapped and converted CSV '{csv_col}' to '{ibkr_col}' with date conversion")
                        else:
                            csv_combined[ibkr_col] = csv_combined[csv_col]
                            logger.info(f"Mapped CSV '{csv_col}' to '{ibkr_col}'")

                # Ensure both DataFrames have the same columns
                all_columns = set(csv_combined.columns) | set(ibkr_df.columns)

                # Add missing columns with NaN only for truly missing columns
                for col in all_columns:
                    if col not in csv_combined.columns:
                        csv_combined[col] = pd.NA
                    if col not in ibkr_df.columns:
                        ibkr_df[col] = pd.NA

                # Reorder columns to match
                common_columns = sorted(all_columns)
                csv_combined = csv_combined[common_columns]
                ibkr_df = ibkr_df[common_columns]

                # Combine both datasets
                combined_df = pd.concat([csv_combined, ibkr_df], ignore_index=True)
                logger.info(f"Before deduplication: {len(combined_df)} total trades")

                # 4. Remove duplicates based on key trade attributes
                combined_df = self._remove_duplicate_trades(combined_df)
                logger.info(f"After deduplication: {len(combined_df)} unique trades")

            elif not csv_combined.empty:
                combined_df = csv_combined
                logger.info("Using only CSV data (no IBKR data)")
            elif not ibkr_df.empty:
                combined_df = ibkr_df
                logger.info("Using only IBKR data (no CSV data)")
            else:
                raise ValueError("No valid data found in either CSV files or IBKR")

            # 5. Process combined data into roundtrips
            csv_content = combined_df.to_csv(index=False)
            roundtrips_df, open_long, open_short = process_ibkr_csv(csv_content)
            logger.info(f"Processed into {len(roundtrips_df)} roundtrips")

            # 5a. Apply stock split adjustments
            from core.stock_splits import apply_split_adjustments_to_roundtrips
            try:
                roundtrips_df = apply_split_adjustments_to_roundtrips(roundtrips_df)
                logger.info("Applied stock split adjustments to roundtrips")
            except Exception as e:
                logger.warning(f"Could not apply split adjustments: {e}")

            # 6. Calculate performance
            performance_data = calculate_performance(roundtrips_df)

            # 7. Process open positions
            from core.open_positions import process_open_positions
            try:
                open_positions_data = await process_open_positions(open_long, open_short, roundtrips_df)
                logger.info(f"Processed {len(open_positions_data['positions'])} open positions")
            except Exception as e:
                logger.warning(f"Could not process open positions: {e}")
                open_positions_data = {"positions": [], "summary": {}}

            return {
                "roundtrips": roundtrips_df,
                "performance": performance_data,
                "open_positions": open_positions_data
            }

        except Exception as e:
            logger.error(f"Error processing hybrid data: {e}")
            raise ValueError(f"Failed to process hybrid data: {str(e)}")

    def _remove_duplicate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate trades based on Symbol, Date, Quantity, Price, and Side

        Args:
            df: Combined DataFrame with potential duplicates

        Returns:
            DataFrame with duplicates removed
        """
        logger.info("Removing duplicate trades...")

        if df.empty:
            return df

        # Define key columns for identifying duplicates
        key_columns = []

        # Map different possible column names to standard names
        column_mapping = {
            'Symbol': ['Symbol', 'symbol'],
            'Trade Date/Time': ['Trade Date/Time', 'TradeDate', 'Date', 'trade_date'],
            'Quantity': ['Quantity', 'quantity', 'Qty'],
            'Trade Price': ['Trade Price', 'TradePrice', 'Price', 'price'],
            'Buy/Sell': ['Buy/Sell', 'Side', 'side', 'BuySell']
        }

        # Find which columns exist
        for standard_col, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    key_columns.append(possible_name)
                    break

        if not key_columns:
            logger.warning("No suitable columns found for duplicate removal, returning original data")
            return df

        logger.info(f"Using columns for duplicate detection: {key_columns}")

        # Remove duplicates based on key columns
        original_count = len(df)
        df_deduplicated = df.drop_duplicates(subset=key_columns, keep='first')
        removed_count = original_count - len(df_deduplicated)

        logger.info(f"Removed {removed_count} duplicate trades")

        return df_deduplicated