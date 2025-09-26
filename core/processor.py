"""
IBKR Trades Processing Module
Handles parsing and converting IBKR trades to roundtrips
"""

import pandas as pd
from dateutil import parser as dtparser
import math
from typing import Dict, List, Any, Tuple


COLUMN_MAP = {
    "symbol": ["Symbol", "Underlying Symbol", "Contract Description"],
    "quantity": ["Quantity", "Qty"],
    "price": ["Trade Price", "Price", "T. Price", "TradePrice"],
    "datetime": ["Trade Date/Time", "Date/Time"],
    "date": ["Trade Date", "Date", "TradeDate"],
    "time": ["Trade Time", "Time"],
    "side": ["Buy/Sell", "Side"],
    "commission": ["Commission", "Comm/Fee", "IB Commission"],
    "portfolio": ["Account Alias", "Account Name", "Account Id", "Account ID"],
    "code": ["Order Reference", "Order Ref", "Client Order Id", "Client Order ID"],
}


def find_first_matching_col(cols: List[str], candidates: List[str]) -> str:
    """Find the first matching column from candidates"""
    for c in candidates:
        if c in cols:
            return c
    return None


def parse_date_only(s: str) -> pd.Timestamp:
    """Parse date-only tokens"""
    if pd.isna(s):
        return pd.NaT
    s = str(s).strip()
    if not s:
        return pd.NaT

    # Keep only digits for safety
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        digits = digits[-8:]  # use the last 8 digits
        try:
            return pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
        except Exception:
            pass

    # Fallback: try generic parse
    try:
        d = dtparser.parse(s, dayfirst=False)
        return pd.Timestamp(year=d.year, month=d.month, day=d.day)
    except Exception:
        return pd.NaT


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Normalize column names and parse dates"""
    cols = {c: c for c in df.columns}
    resolved = {}
    for key, candidates in COLUMN_MAP.items():
        resolved[key] = find_first_matching_col(list(cols.keys()), candidates)

    # Build a single parsed date column
    if resolved["datetime"] is not None:
        dt_series = df[resolved["datetime"]].apply(parse_date_only)
    elif resolved["date"] is not None:
        dt_series = df[resolved["date"]].apply(parse_date_only)
    else:
        raise ValueError("No date column found. Include 'Trade Date/Time' or 'Trade Date' in your IBKR CSV.")

    df = df.copy()
    df["_parsed_dt"] = dt_series

    # Require the basics
    for req in ("symbol", "quantity", "price", "side"):
        if resolved[req] is None:
            raise ValueError(f"Missing required column for '{req}'.")

    return df, resolved


def as_float(x: Any) -> float:
    """Convert to float safely"""
    try:
        return float(x)
    except Exception:
        return float("nan")


def as_int_abs(x: Any) -> int:
    """Convert to absolute integer safely"""
    try:
        return abs(int(round(float(x))))
    except Exception:
        try:
            return abs(int(x))
        except Exception:
            return 0


def build_round_trips(df: pd.DataFrame, col: Dict[str, str]) -> pd.DataFrame:
    """Build round trips from trades using FIFO matching"""
    # Sort executions by date only
    trades = df.sort_values(by="_parsed_dt", kind="mergesort").copy()

    commission_col = col["commission"] if col["commission"] else None
    portfolio_col = col["portfolio"] if col["portfolio"] else None
    code_col = col["code"] if col["code"] else None

    open_long = {}   # key -> list of BUY-open lots
    open_short = {}  # key -> list of SELL-open lots
    rows = []

    def push(qdict, key, lot):
        qdict.setdefault(key, []).append(lot)

    for _, r in trades.iterrows():
        symbol = str(r[col["symbol"]]).strip()
        qty = as_int_abs(r[col["quantity"]])
        if qty == 0:
            continue

        price = as_float(r[col["price"]])
        d = r["_parsed_dt"]  # date only
        side = str(r[col["side"]]).strip().upper()
        comm_total = as_float(r[commission_col]) if (commission_col and pd.notna(r[commission_col])) else 0.0
        portfolio = str(r[portfolio_col]).strip() if portfolio_col else ""
        code = str(r[code_col]).strip() if code_col else ""
        key = (portfolio, symbol, code)

        lot = {
            "symbol": symbol,
            "qty_left": qty,
            "price": price,
            "dt": d,
            "commission_total": comm_total,
            "portfolio": portfolio,
            "code": code,
        }

        if side in ("BUY", "B"):
            # Close shorts first, remainder opens long
            closing_qty = qty
            shorts = open_short.get(key, [])
            i = 0
            while closing_qty > 0 and i < len(shorts):
                open_lot = shorts[i]
                match_qty = min(open_lot["qty_left"], closing_qty)

                open_comm_used = (open_lot["commission_total"] * (match_qty / open_lot["qty_left"])) if open_lot["qty_left"] > 0 else 0.0
                close_comm_used = (comm_total * (match_qty / qty)) if qty > 0 else 0.0

                rows.append({
                    "Symbol": symbol,
                    "Shares": match_qty,
                    "Entry Price": open_lot["price"],
                    "Entry Date": open_lot["dt"],
                    "Exit Price": price,
                    "Exit Date": d,
                    "Type": "Short",
                    "Commission": open_comm_used + close_comm_used,
                    "Portfolio": portfolio,
                    "Code": code,
                })

                open_lot["qty_left"] -= match_qty
                open_lot["commission_total"] -= open_comm_used
                closing_qty -= match_qty
                comm_total -= close_comm_used

                if open_lot["qty_left"] == 0:
                    i += 1

            if i > 0:
                open_short[key] = shorts[i:]
            if closing_qty > 0:
                push(open_long, key, {
                    **lot,
                    "qty_left": closing_qty,
                    "commission_total": (comm_total * (closing_qty / qty)) if qty > 0 else 0.0
                })

        elif side in ("SELL", "S"):
            # Close longs first, remainder opens short
            closing_qty = qty
            longs = open_long.get(key, [])
            i = 0
            while closing_qty > 0 and i < len(longs):
                open_lot = longs[i]
                match_qty = min(open_lot["qty_left"], closing_qty)

                open_comm_used = (open_lot["commission_total"] * (match_qty / open_lot["qty_left"])) if open_lot["qty_left"] > 0 else 0.0
                close_comm_used = (comm_total * (match_qty / qty)) if qty > 0 else 0.0

                rows.append({
                    "Symbol": symbol,
                    "Shares": match_qty,
                    "Entry Price": open_lot["price"],
                    "Entry Date": open_lot["dt"],
                    "Exit Price": price,
                    "Exit Date": d,
                    "Type": "Long",
                    "Commission": open_comm_used + close_comm_used,
                    "Portfolio": portfolio,
                    "Code": code,
                })

                open_lot["qty_left"] -= match_qty
                open_lot["commission_total"] -= open_comm_used
                closing_qty -= match_qty
                comm_total -= close_comm_used

                if open_lot["qty_left"] == 0:
                    i += 1

            if i > 0:
                open_long[key] = longs[i:]
            if closing_qty > 0:
                push(open_short, key, {
                    **lot,
                    "qty_left": closing_qty,
                    "commission_total": (comm_total * (closing_qty / qty)) if qty > 0 else 0.0
                })

    if not rows:
        return pd.DataFrame(columns=["Symbol", "Shares", "Entry Price", "Entry Date", "Exit Price", "Exit Date", "Type", "Commission", "Portfolio", "Code"])

    out = pd.DataFrame(rows)
    out["Shares"] = out["Shares"].astype(int)

    # Ensure optional cols exist
    for c in ("Commission", "Portfolio", "Code"):
        if c not in out.columns:
            out[c] = ""

    # Sort round trips by Entry Date
    out = out.sort_values(by="Entry Date", kind="mergesort")

    return out


def process_ibkr_csv(csv_content: str) -> pd.DataFrame:
    """Main function to process IBKR CSV content and return roundtrips"""
    # Read CSV content into DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(csv_content))

    # Process to roundtrips
    df, col = normalize_columns(df)
    roundtrips_df = build_round_trips(df, col)

    return roundtrips_df