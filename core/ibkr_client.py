"""
IBKR FlexQuery Client for fetching trade data from Interactive Brokers
"""
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import xml.etree.ElementTree as ET
from io import StringIO

import requests
import pandas as pd
from ibflex import client, parser, Types

logger = logging.getLogger(__name__)


class FlexQueryClient:
    """Client for fetching and parsing IBKR FlexQuery data"""

    BASE_URL = "https://gdcdyn.interactivebrokers.com/Universal/servlet"

    def __init__(self, token: str, query_id: str, from_date: Optional[str] = None, to_date: Optional[str] = None):
        """
        Initialize FlexQuery client

        Args:
            token: FlexQuery service token
            query_id: FlexQuery report ID
            from_date: Optional start date in YYYYMMDD format (max 365 days range)
            to_date: Optional end date in YYYYMMDD format
        """
        self.token = token
        self.query_id = query_id
        self.from_date = from_date
        self.to_date = to_date

    def fetch_statement(self) -> Types.FlexQueryResponse:
        """
        Fetch activity statement from IBKR FlexQuery service

        Returns:
            Parsed FlexQueryResponse object
        """
        try:
            # Download the flex query response
            response = client.download(self.token, self.query_id)
            return response
        except Exception as e:
            logger.error(f"Error fetching FlexQuery statement: {e}")
            raise

    def fetch_trades_raw(self) -> str:
        """
        Fetch raw response from IBKR FlexQuery service

        Returns:
            Raw response string (XML or CSV)
        """
        try:
            # Step 1: Request the report
            request_url = f"{self.BASE_URL}/FlexStatementService.SendRequest"
            request_params = {
                "t": self.token,
                "q": self.query_id,
                "v": "3"
            }

            # Add date range parameters if specified
            if self.from_date:
                request_params["fd"] = self.from_date
                logger.info(f"Using custom from date: {self.from_date}")
            if self.to_date:
                request_params["td"] = self.to_date
                logger.info(f"Using custom to date: {self.to_date}")

            logger.info(f"Requesting FlexQuery report from IBKR...")
            request_response = requests.get(request_url, params=request_params)
            request_response.raise_for_status()

            # Log the raw response for debugging
            response_text = request_response.text
            logger.debug(f"Initial response (first 500 chars): {response_text[:500]}")

            # Check if the response starts with HTML (error page)
            if response_text.strip().startswith('<html') or response_text.strip().startswith('<!DOCTYPE'):
                logger.error("Received HTML instead of XML - likely an authentication error")
                raise Exception("Authentication failed - please check your token and query ID")

            # Parse the reference code from response
            try:
                root = ET.fromstring(response_text)
            except ET.ParseError as e:
                logger.error(f"Failed to parse initial response as XML: {e}")
                logger.error(f"Response content: {response_text[:1000]}")
                raise Exception(f"Invalid response from IBKR - please check your token is valid")

            if root.tag == "FlexStatementResponse":
                reference_code_elem = root.find("ReferenceCode")
                url_elem = root.find("Url")

                if reference_code_elem is not None and url_elem is not None:
                    reference_code = reference_code_elem.text
                    base_url = url_elem.text
                    logger.info(f"Got reference code: {reference_code}")
                else:
                    error_elem = root.find("ErrorMessage")
                    error_msg = error_elem.text if error_elem is not None else "Invalid response structure"
                    raise Exception(f"FlexQuery request failed: {error_msg}")
            else:
                error_elem = root.find("ErrorMessage")
                error_msg = error_elem.text if error_elem is not None else f"Unexpected root tag: {root.tag}"
                raise Exception(f"FlexQuery request failed: {error_msg}")

            # Step 2: Get the statement using reference code
            statement_url = f"{base_url}?q={reference_code}&t={self.token}&v=3"
            logger.info("Waiting for report generation...")

            # Wait a moment for report generation
            import time
            time.sleep(3)

            logger.info("Fetching generated report...")
            statement_response = requests.get(statement_url)
            statement_response.raise_for_status()

            final_response = statement_response.text
            logger.debug(f"Final response (first 500 chars): {final_response[:500]}")

            # Check if response is CSV format
            if final_response.strip().startswith('"') and ',' in final_response[:100]:
                logger.info("Detected CSV format response from FlexQuery")
            elif final_response.strip().startswith('<?xml') or final_response.strip().startswith('<FlexQuery'):
                logger.info("Detected XML format response from FlexQuery")

            return final_response

        except requests.RequestException as e:
            logger.error(f"HTTP error fetching FlexQuery data: {e}")
            raise Exception(f"Network error connecting to IBKR: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching raw FlexQuery data: {e}")
            raise

    def parse_trades_from_response(self, response_content: str) -> List[Dict[str, Any]]:
        """
        Parse trades from FlexQuery response (XML or CSV)

        Args:
            response_content: Raw response string from FlexQuery

        Returns:
            List of trade dictionaries
        """
        try:
            # Check if we got an HTML error page
            if response_content.strip().startswith('<html') or response_content.strip().startswith('<!DOCTYPE'):
                logger.error("Received HTML instead of data - authentication or query error")
                raise Exception("Invalid response from IBKR - received HTML error page instead of data")

            # Detect format and parse accordingly
            if response_content.strip().startswith('"') and ',' in response_content[:100]:
                # CSV format
                logger.info("Parsing CSV format response")
                return self._parse_trades_from_csv(response_content)
            elif response_content.strip().startswith('<?xml') or response_content.strip().startswith('<FlexQuery'):
                # XML format
                logger.info("Parsing XML format response")
                return self._parse_trades_from_xml(response_content)
            else:
                logger.error(f"Unrecognized response format. Content: {response_content[:200]}")
                raise Exception("Unrecognized response format from IBKR FlexQuery")

        except Exception as e:
            logger.error(f"Error parsing FlexQuery response: {e}")
            raise

    def _parse_trades_from_csv(self, csv_content: str) -> List[Dict[str, Any]]:
        """
        Parse trades from CSV format FlexQuery response

        Args:
            csv_content: Raw CSV string from FlexQuery

        Returns:
            List of trade dictionaries
        """
        try:
            # Parse CSV using pandas
            import pandas as pd
            from io import StringIO

            df = pd.read_csv(StringIO(csv_content))
            logger.info(f"Parsed CSV with {len(df)} rows and columns: {list(df.columns)}")

            trades = []
            for _, row in df.iterrows():
                # Map CSV columns to trade dictionary
                trade = {
                    "symbol": str(row.get("Symbol", "")),
                    "quantity": float(row.get("Quantity", 0)),
                    "price": float(row.get("TradePrice", row.get("Price", 0))),
                    "date": self._parse_date(str(row.get("TradeDate", row.get("Date", "")))),
                    "side": self._normalize_side(str(row.get("Buy/Sell", row.get("Side", "")))),
                    "commission": abs(float(row.get("IBCommission", row.get("Commission", 0)))),
                    "currency": str(row.get("Currency", "USD")),
                    "description": str(row.get("Description", "")),
                    "account": str(row.get("Account", "")),
                    "order_ref": str(row.get("OrderReference", ""))
                }

                # Skip if essential fields are missing
                if trade["symbol"] and trade["quantity"] != 0:
                    trades.append(trade)

            logger.info(f"Successfully parsed {len(trades)} trades from CSV")
            return trades

        except Exception as e:
            logger.error(f"Error parsing CSV response: {e}")
            raise

    def _parse_trades_from_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Parse trades from XML format FlexQuery response

        Args:
            xml_content: Raw XML string from FlexQuery

        Returns:
            List of trade dictionaries
        """
        try:
            # Parse XML
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError as e:
                logger.error(f"XML Parse Error: {e}")
                logger.error(f"Response content (first 500 chars): {xml_content[:500]}")
                raise Exception(f"Failed to parse IBKR response as XML: {str(e)}")

            trades = []

            # Check if this is an error response
            if root.tag == "FlexStatementResponse":
                error_elem = root.find("ErrorMessage")
                if error_elem is not None:
                    raise Exception(f"IBKR FlexQuery error: {error_elem.text}")

            # Navigate to trades - structure can vary
            for statement in root.findall(".//FlexStatement"):
                for trade_element in statement.findall(".//Trade"):
                    trade = self._parse_trade_element(trade_element)
                    if trade:
                        trades.append(trade)

            # Also check for trades directly under FlexQueryResponse
            for trade_element in root.findall(".//Trade"):
                trade = self._parse_trade_element(trade_element)
                if trade:
                    trades.append(trade)

            logger.info(f"Successfully parsed {len(trades)} trades from XML")
            return trades

        except Exception as e:
            logger.error(f"Error parsing XML response: {e}")
            raise

    def _parse_trade_element(self, trade_element: ET.Element) -> Optional[Dict[str, Any]]:
        """
        Parse individual trade element from XML

        Args:
            trade_element: XML Element representing a trade

        Returns:
            Trade dictionary or None if not a stock trade
        """
        try:
            # Get trade attributes
            attrs = trade_element.attrib

            # Filter for stock trades only (skip options, futures, etc.)
            asset_category = attrs.get("assetCategory", "")
            if asset_category not in ["STK", "STOCK"]:
                return None

            # Extract required fields
            trade = {
                "symbol": attrs.get("symbol", ""),
                "quantity": float(attrs.get("quantity", 0)),
                "price": float(attrs.get("tradePrice", attrs.get("price", 0))),
                "date": self._parse_date(attrs.get("tradeDate", attrs.get("dateTime", ""))),
                "side": self._normalize_side(attrs.get("buySell", "")),
                "commission": abs(float(attrs.get("ibCommission", attrs.get("commission", 0)))),
                "currency": attrs.get("currency", "USD"),
                "description": attrs.get("description", ""),
                "account": attrs.get("accountId", ""),
                "order_ref": attrs.get("orderReference", "")
            }

            # Skip if essential fields are missing
            if not trade["symbol"] or trade["quantity"] == 0:
                return None

            return trade

        except Exception as e:
            logger.warning(f"Error parsing trade element: {e}")
            return None

    def _parse_date(self, date_str: str) -> str:
        """
        Parse date string from various IBKR formats

        Args:
            date_str: Date string from IBKR

        Returns:
            Standardized date string
        """
        if not date_str or date_str == "nan":
            return ""

        # Try different date formats
        formats = [
            "%Y%m%d",
            "%Y-%m-%d",
            "%Y%m%d;%H%M%S",
            "%Y-%m-%d;%H:%M:%S",
            "%Y-%m-%d %H:%M:%S"
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.split(",")[0], fmt)
                return dt.strftime("%Y-%m-%d")  # Simplified format for CSV processing
            except:
                continue

        # Return as-is if no format matches
        logger.warning(f"Could not parse date: {date_str}")
        return date_str

    def _normalize_side(self, side: str) -> str:
        """
        Normalize buy/sell side to BUY/SELL

        Args:
            side: Buy/sell indicator from IBKR

        Returns:
            Normalized side (BUY or SELL)
        """
        side_upper = side.upper()
        if side_upper in ["BUY", "BOT", "B"]:
            return "BUY"
        elif side_upper in ["SELL", "SLD", "S"]:
            return "SELL"
        return side_upper

    def convert_to_dataframe(self, trades: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert trades list to DataFrame matching existing CSV format

        Args:
            trades: List of trade dictionaries

        Returns:
            DataFrame with trades
        """
        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)

        # Rename columns to match existing processor expectations
        column_mapping = {
            "symbol": "Symbol",
            "quantity": "Quantity",
            "price": "Trade Price",
            "date": "Trade Date/Time",
            "side": "Buy/Sell",
            "commission": "Commission",
            "currency": "Currency",
            "account": "Account",
            "order_ref": "Order Reference"
        }

        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_columns = ["Symbol", "Quantity", "Trade Price", "Trade Date/Time", "Buy/Sell"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")

        return df

    def fetch_and_process_trades(self) -> pd.DataFrame:
        """
        Complete workflow to fetch and process trades from IBKR

        Returns:
            DataFrame with processed trades
        """
        try:
            # Fetch raw XML
            logger.info("Fetching trades from IBKR FlexQuery...")
            xml_content = self.fetch_trades_raw()

            # Parse trades
            logger.info("Parsing trade data...")
            trades = self.parse_trades_from_response(xml_content)

            if not trades:
                logger.warning("No trades found in FlexQuery response")
                return pd.DataFrame()

            logger.info(f"Found {len(trades)} trades")

            # Convert to DataFrame
            df = self.convert_to_dataframe(trades)

            return df

        except Exception as e:
            logger.error(f"Error in fetch_and_process_trades: {e}")
            raise

    @classmethod
    def fetch_historical_data(cls, token: str, query_id: str, start_year: int, end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical data across multiple years by making separate requests per year

        Args:
            token: FlexQuery service token
            query_id: FlexQuery report ID
            start_year: Starting year (e.g., 2020)
            end_year: Ending year (optional, defaults to current year)

        Returns:
            Combined DataFrame with all historical trades (RAW TRADES, not processed)
        """
        if end_year is None:
            end_year = datetime.now().year

        logger.info(f"Fetching historical data from {start_year} to {end_year}")

        all_trades = []

        for year in range(start_year, end_year + 1):
            try:
                # Set date range for the entire year
                from_date = f"{year}0101"  # January 1st
                to_date = f"{year}1231"    # December 31st

                logger.info(f"Fetching data for year {year} ({from_date} to {to_date})")

                # Create client for this year
                client = cls(token, query_id, from_date, to_date)

                # Fetch RAW response and parse trades (not processed)
                response = client.fetch_trades_raw()
                trades = client.parse_trades_from_response(response)

                if trades:
                    logger.info(f"Retrieved {len(trades)} trades for year {year}")
                    all_trades.extend(trades)
                else:
                    logger.info(f"No trades found for year {year}")

                # Add a small delay between requests to be respectful to IBKR servers
                import time
                time.sleep(2)

            except Exception as e:
                logger.error(f"Error fetching data for year {year}: {e}")
                continue

        if all_trades:
            # Convert all trades to single DataFrame
            combined_df = client.convert_to_dataframe(all_trades)
            logger.info(f"Combined historical data: {len(combined_df)} total trades from {len(all_trades)} raw trades")
            return combined_df
        else:
            logger.warning("No historical data retrieved")
            return pd.DataFrame()


def test_connection(token: str, query_id: str) -> Tuple[bool, str]:
    """
    Test FlexQuery connection

    Args:
        token: FlexQuery service token
        query_id: FlexQuery report ID

    Returns:
        Tuple of (success, message)
    """
    try:
        client = FlexQueryClient(token, query_id)

        # Try to fetch data
        response_content = client.fetch_trades_raw()

        # Parse the response to check for trades
        trades = client.parse_trades_from_response(response_content)
        return True, f"Connection successful. Found {len(trades)} trades."

    except Exception as e:
        return False, f"Connection failed: {str(e)}"