"""
Performance Calculation Module
Handles aggregating roundtrips into performance metrics
"""

import pandas as pd
import math
from typing import Dict, List, Any, Optional


def to_datetime_safe(s):
    """Parse dates safely"""
    return pd.to_datetime(s, errors="coerce")


def compute_trade_pct(row):
    """
    Percent return for a single round-trip:
      Long:  (Exit - Entry) / Entry
      Short: (Entry - Exit) / Entry
    Returns a float fraction (e.g., 0.1234 for +12.34%).
    """
    e = float(row["Entry Price"])
    x = float(row["Exit Price"])
    t = str(row["Type"]).strip().lower()
    if e == 0 or pd.isna(e) or pd.isna(x):
        return float("nan")
    if t == "long":
        return (x - e) / e
    elif t == "short":
        return (e - x) / e
    else:
        # default assume long
        return (x - e) / e


def safe_mean(series):
    if len(series) == 0:
        return 0.0
    return float(pd.Series(series).mean())


def safe_min(series):
    if len(series) == 0:
        return 0.0
    return float(pd.Series(series).min())


def safe_max(series):
    if len(series) == 0:
        return 0.0
    return float(pd.Series(series).max())


def agg_block(df_group):
    """
    Compute one row of metrics for a group of trades.
    """
    n_trades = len(df_group)
    if n_trades == 0:
        return None

    # Winners / Losers / Breakeven
    winners = df_group[df_group["_pct"] > 0]
    losers  = df_group[df_group["_pct"] < 0]
    breakeven = df_group[df_group["_pct"] == 0]

    wins = len(winners)
    losses = len(losers)
    be = len(breakeven)

    avg_gain = safe_mean(winners["_pct"]) * 100.0
    avg_loss = safe_mean(losers["_pct"]) * 100.0  # negative
    net = avg_gain + avg_loss

    ratio = 0.0
    if losses > 0 and avg_loss != 0:
        ratio = abs(avg_gain / avg_loss)

    win_pct = (wins / n_trades) * 100.0
    loss_pct = (losses / n_trades) * 100.0

    # Largest winner/loser (%)
    lg_gain = safe_max(df_group["_pct"]) * 100.0 if wins > 0 else 0.0
    lg_loss = abs(safe_min(df_group["_pct"])) * 100.0 if losses > 0 else 0.0  # Make loss positive
    lg_net = lg_gain - lg_loss  # Net should be gain minus loss
    lg_ratio = 0.0
    if losses > 0 and lg_loss != 0:
        lg_ratio = abs(lg_gain / lg_loss)

    # Average holding days (winners/losers)
    avg_days_gain = 0.0
    avg_days_losses = 0.0
    if wins > 0:
        avg_days_gain = safe_mean((winners["_exit_dt"] - winners["_entry_dt"]).dt.days)
    if losses > 0:
        avg_days_losses = safe_mean((losers["_exit_dt"] - losers["_entry_dt"]).dt.days)

    # Commission sum (keep numeric; sign as provided)
    comm_sum = float(df_group["Commission"].fillna(0.0).sum())

    return {
        "Date": "",  # Will be filled by caller
        "Avg Gain": round(avg_gain, 2),
        "Avg Loss": round(avg_loss, 2),
        "Net": round(net, 2),
        "Ratio": round(ratio, 2),
        "Win %": round(win_pct, 2),
        "Loss %": round(loss_pct, 2),
        "Wins": int(wins),
        "Losses": int(losses),
        "Break-even": int(be),
        "Total Trades": int(n_trades),
        "LG Gain": round(lg_gain, 2),
        "LG Loss": round(lg_loss, 2),
        "LG Net": round(lg_net, 2),
        "LG Ratio": round(lg_ratio, 2),
        "Avg Days Gain": int(round(avg_days_gain)) if not math.isnan(avg_days_gain) else 0,
        "Avg Days Losses": int(round(avg_days_losses)) if not math.isnan(avg_days_losses) else 0,
        "COMM": round(comm_sum, 2),
    }


def calculate_performance(roundtrips_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate performance metrics from roundtrips DataFrame
    Returns dict with monthly, quarterly, yearly, and since inception data
    """
    if roundtrips_df.empty:
        return {
            "monthly": [],
            "quarterly": [],
            "yearly": [],
            "since_inception": {},
            "summary": {"total_trades": 0, "win_rate": 0, "net_return": 0}
        }

    df = roundtrips_df.copy()

    # Parse dates (they're already Timestamps from roundtrip processing)
    df["_entry_dt"] = df["Entry Date"]
    df["_exit_dt"] = df["Exit Date"]

    # For string dates (if dates were formatted), parse them
    if df["_entry_dt"].dtype == 'object':
        df["_entry_dt"] = to_datetime_safe(df["Entry Date"])
        df["_exit_dt"] = to_datetime_safe(df["Exit Date"])

    # Compute percent return per trade
    df["_pct"] = df.apply(compute_trade_pct, axis=1)

    # Drop trades without valid dates or pct
    df = df.dropna(subset=["_entry_dt","_exit_dt","_pct"]).copy()

    if df.empty:
        return {
            "monthly": [],
            "quarterly": [],
            "yearly": [],
            "since_inception": {},
            "summary": {"total_trades": 0, "win_rate": 0, "net_return": 0}
        }

    # Build time keys BY EXIT DATE (realization date)
    exit_dt = df["_exit_dt"]
    df["_month_key"] = exit_dt.dt.to_period("M")     # e.g., 2025-09
    df["_quarter_key"] = exit_dt.dt.to_period("Q")   # e.g., 2025Q3
    df["_year_key"] = exit_dt.dt.year                # e.g., 2025

    # ----- Monthly -----
    monthly = []
    for key, g in df.groupby("_month_key"):
        m = agg_block(g)
        if m is None:
            continue
        # Pretty label like "Sep 2025"
        month_label = key.to_timestamp().strftime("%b %Y")
        m["Date"] = month_label
        m["_sort"] = key.to_timestamp()
        monthly.append(m)

    monthly = sorted(monthly, key=lambda x: x["_sort"])
    # Remove sorting keys before returning
    for item in monthly:
        item.pop("_sort", None)

    # ----- Quarterly -----
    quarterly = []
    for key, g in df.groupby("_quarter_key"):
        m = agg_block(g)
        if m is None:
            continue
        # Label like "Q3 2025"
        q_num = key.quarter
        y = key.year
        m["Date"] = f"Q{q_num} {y}"
        m["_sort"] = y * 10 + q_num
        quarterly.append(m)

    quarterly = sorted(quarterly, key=lambda x: x["_sort"])
    # Remove sorting keys before returning
    for item in quarterly:
        item.pop("_sort", None)

    # ----- Yearly -----
    yearly = []
    for key, g in df.groupby("_year_key"):
        m = agg_block(g)
        if m is None:
            continue
        m["Date"] = f"{int(key)}"
        m["_sort"] = int(key)
        yearly.append(m)

    yearly = sorted(yearly, key=lambda x: x["_sort"])
    # Remove sorting keys before returning
    for item in yearly:
        item.pop("_sort", None)

    # ----- Since inception (single row) -----
    since_inception = {}
    m_all = agg_block(df)
    if m_all is not None:
        m_all["Date"] = "Since Inception"
        since_inception = m_all

    # Enhanced summary for dashboard (5 key metrics)
    batting_average = since_inception.get("Win %", 0)
    avg_gain = since_inception.get("Avg Gain", 0)  # Already calculated for winners only
    avg_loss = abs(since_inception.get("Avg Loss", 0))  # Make positive for display

    # Calculate Win/Loss Ratio
    win_loss_ratio = avg_gain / avg_loss if avg_loss != 0 else 0

    # Calculate Adjusted Win/Loss Ratio (accounts for batting average)
    adjusted_win_loss_ratio = win_loss_ratio * (batting_average / 100) if batting_average > 0 else 0

    summary = {
        "total_trades": since_inception.get("Total Trades", 0),
        "batting_average": batting_average,  # Win rate percentage
        "average_gain": avg_gain,  # Average gain on winning trades only
        "average_loss": avg_loss,  # Average loss on losing trades only (positive)
        "win_loss_ratio": round(win_loss_ratio, 2),
        "adjusted_win_loss_ratio": round(adjusted_win_loss_ratio, 2),
        "net_return": since_inception.get("Net", 0)
    }

    return {
        "monthly": monthly,
        "quarterly": quarterly,
        "yearly": yearly,
        "since_inception": since_inception,
        "summary": summary
    }