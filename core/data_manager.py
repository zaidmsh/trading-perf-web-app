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
        query_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get trades from specified source

        Args:
            source: Data source type ('csv' or 'ibkr')
            files: List of uploaded CSV files (for CSV source)
            token: IBKR FlexQuery token (for IBKR source)
            query_id: IBKR FlexQuery ID (for IBKR source)

        Returns:
            Dictionary containing processed trades and roundtrips
        """
        try:
            if source == "csv":
                return await self._process_csv_files(files)
            elif source == "ibkr":
                return await self._fetch_from_ibkr(token, query_id)
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
            roundtrips_df = process_ibkr_csv(csv_content)

            # Calculate performance
            performance_data = calculate_performance(roundtrips_df)

            return {
                "roundtrips": roundtrips_df,
                "performance": performance_data
            }
        else:
            raise ValueError("No valid data found in uploaded files")

    async def _fetch_from_ibkr(self, token: str, query_id: str) -> Dict[str, Any]:
        """
        Fetch trades from IBKR FlexQuery API

        Args:
            token: IBKR FlexQuery token
            query_id: IBKR FlexQuery ID

        Returns:
            Dictionary containing processed trades and roundtrips
        """
        if not token or not query_id:
            raise ValueError("IBKR token and query ID are required")

        try:
            # Create FlexQuery client
            client = FlexQueryClient(token, query_id)

            # Fetch and process trades
            logger.info("Fetching trades from IBKR FlexQuery...")
            trades_df = client.fetch_and_process_trades()

            if trades_df.empty:
                raise ValueError("No trades found in IBKR FlexQuery response")

            logger.info(f"Fetched {len(trades_df)} trades from IBKR")

            # Process trades using existing processor
            # Convert DataFrame to CSV string for processing
            csv_content = trades_df.to_csv(index=False)
            roundtrips_df = process_ibkr_csv(csv_content)

            # Calculate performance
            performance_data = calculate_performance(roundtrips_df)

            return {
                "roundtrips": roundtrips_df,
                "performance": performance_data
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