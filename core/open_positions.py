"""
Open Positions Calculator
Handles calculation of unrealized P&L and position metrics
"""

import logging
from typing import Dict, List, Any

from core.price_service import get_stock_prices

logger = logging.getLogger(__name__)


def calculate_realized_pnl_for_symbol(symbol: str, portfolio: str, code: str, roundtrips_df) -> Dict[str, float]:
    """
    Calculate realized P&L for a symbol from roundtrips data
    
    Args:
        symbol: Stock symbol
        portfolio: Portfolio identifier
        code: Asset code
        roundtrips_df: DataFrame of completed roundtrips
        
    Returns:
        Dictionary with realized_pnl, realized_proceeds, total_cost_basis, total_shares_sold
    """
    if roundtrips_df is None or roundtrips_df.empty:
        return {"realized_pnl": 0.0, "realized_proceeds": 0.0, "total_cost_basis": 0.0, "total_shares_sold": 0}
    
    # Filter roundtrips for this specific symbol
    mask = roundtrips_df["Symbol"] == symbol
    if "Portfolio" in roundtrips_df.columns:
        mask &= roundtrips_df["Portfolio"] == portfolio
    if "Code" in roundtrips_df.columns:
        mask &= roundtrips_df["Code"] == code
    
    symbol_roundtrips = roundtrips_df[mask]
    
    if symbol_roundtrips.empty:
        return {"realized_pnl": 0.0, "realized_proceeds": 0.0, "total_cost_basis": 0.0, "total_shares_sold": 0}
    
    # Calculate realized P&L from completed roundtrips
    total_shares_sold = symbol_roundtrips["Shares"].sum()
    total_proceeds = (symbol_roundtrips["Shares"] * symbol_roundtrips["Exit Price"]).sum()
    total_cost_of_sold = (symbol_roundtrips["Shares"] * symbol_roundtrips["Entry Price"]).sum()
    total_commission = symbol_roundtrips["Commission"].sum()
    
    # Realized P&L = proceeds - cost - commission
    realized_pnl = total_proceeds - total_cost_of_sold - total_commission
    
    # Total cost basis of sold shares (for calculating original position size)
    total_cost_basis_sold = total_cost_of_sold + total_commission
    
    return {
        "realized_pnl": round(realized_pnl, 2),
        "realized_proceeds": round(total_proceeds, 2),
        "total_cost_basis": round(total_cost_basis_sold, 2),
        "total_shares_sold": int(total_shares_sold)
    }


def extract_open_positions(open_long: Dict, open_short: Dict, roundtrips_df=None) -> List[Dict[str, Any]]:
    """
    Extract open positions from the tracking dictionaries with detailed lot information
    
    Args:
        open_long: Dictionary of long positions by (portfolio, symbol, code)
        open_short: Dictionary of short positions by (portfolio, symbol, code)
        roundtrips_df: Optional DataFrame of completed roundtrips for realized P&L calculation
        
    Returns:
        List of open position dictionaries with enhanced details
    """
    positions = []
    
    # Process long positions
    for key, lots in open_long.items():
        if not lots:
            continue
            
        portfolio, symbol, code = key
        
        # Aggregate multiple lots for same symbol and prepare detailed lot info
        total_shares = 0
        total_cost = 0
        total_commission = 0
        orders_count = 0
        detailed_lots = []
        
        for lot in lots:
            shares = lot["qty_left"]
            price = lot["price"]
            commission = lot["commission_total"]
            entry_date = lot["dt"]
            
            total_shares += shares
            total_cost += (shares * price)
            total_commission += commission
            orders_count += 1
            
            # Create detailed lot information
            detailed_lots.append({
                "entry_date": entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date),
                "original_quantity": int(shares),  # For now, same as remaining since we don't track partial sales yet
                "quantity_remaining": int(shares),
                "entry_price": round(price, 4),
                "commission": round(commission, 2),
                "cost_basis": round((shares * price) + commission, 2),
                "partial_sales": []  # Will be populated when we add sales tracking
            })
        
        if total_shares > 0:
            avg_entry_price = total_cost / total_shares
            cost_basis = total_cost + total_commission
            
            # For open positions, we cannot reliably calculate realized P&L
            # from historical roundtrips as they may be from previous position cycles
            # TODO: Implement position lifecycle tracking to properly link trades
            realized_pnl = 0.0
            realized_proceeds = 0.0
            
            # High water mark equals current position size for now
            high_water_mark = int(total_shares)
            
            position = {
                "symbol": symbol,
                "position_type": "Long",
                "quantity": int(total_shares),
                "high_water_mark": high_water_mark,
                "avg_entry_price": round(avg_entry_price, 4),
                "cost_basis": round(cost_basis, 2),
                "total_commission": round(total_commission, 2),
                "orders_count": orders_count,
                
                # P&L tracking for open positions
                "realized_pnl": realized_pnl,
                "realized_proceeds": realized_proceeds,
                "total_cost_basis": round(cost_basis, 2),
                "remaining_cost_basis": round(cost_basis, 2),
                
                # Detailed lot information
                "lots": detailed_lots,
                
                # Will be filled by price service
                "current_price": None,
                "market_value": None,
                "unrealized_pnl": None,
                "unrealized_pnl_pct": None,
                "total_pnl": None,
                "total_pnl_pct": None
            }
            positions.append(position)
    
    # Process short positions
    for key, lots in open_short.items():
        if not lots:
            continue
            
        portfolio, symbol, code = key
        
        # Aggregate multiple lots for same symbol and prepare detailed lot info
        total_shares = 0
        total_cost = 0
        total_commission = 0
        orders_count = 0
        detailed_lots = []
        
        for lot in lots:
            shares = lot["qty_left"]
            price = lot["price"]
            commission = lot["commission_total"]
            entry_date = lot["dt"]
            
            total_shares += shares
            total_cost += (shares * price)
            total_commission += commission
            orders_count += 1
            
            # Create detailed lot information
            detailed_lots.append({
                "entry_date": entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date),
                "original_quantity": int(shares),  # For now, same as remaining since we don't track partial sales yet
                "quantity_remaining": int(shares),
                "entry_price": round(price, 4),
                "commission": round(commission, 2),
                "cost_basis": round((shares * price) + commission, 2),
                "partial_sales": []  # Will be populated when we add sales tracking
            })
        
        if total_shares > 0:
            avg_entry_price = total_cost / total_shares
            cost_basis = total_cost + total_commission
            
            # For open positions, we cannot reliably calculate realized P&L
            # from historical roundtrips as they may be from previous position cycles
            # TODO: Implement position lifecycle tracking to properly link trades
            realized_pnl = 0.0
            realized_proceeds = 0.0
            
            # High water mark equals current position size for now
            high_water_mark = int(total_shares)
            
            position = {
                "symbol": symbol,
                "position_type": "Short",
                "quantity": int(total_shares),
                "high_water_mark": high_water_mark,
                "avg_entry_price": round(avg_entry_price, 4),
                "cost_basis": round(cost_basis, 2),
                "total_commission": round(total_commission, 2),
                "orders_count": orders_count,
                
                # P&L tracking for open positions
                "realized_pnl": realized_pnl,
                "realized_proceeds": realized_proceeds,
                "total_cost_basis": round(cost_basis, 2),
                "remaining_cost_basis": round(cost_basis, 2),
                
                # Detailed lot information
                "lots": detailed_lots,
                
                # Will be filled by price service
                "current_price": None,
                "market_value": None,
                "unrealized_pnl": None,
                "unrealized_pnl_pct": None,
                "total_pnl": None,
                "total_pnl_pct": None
            }
            positions.append(position)
    
    logger.info("Extracted %d open positions", len(positions))
    return positions


async def calculate_position_pnl(positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Calculate P&L for positions using current market prices
    
    Args:
        positions: List of position dictionaries
        
    Returns:
        List of positions with P&L calculations added
    """
    if not positions:
        return positions
    
    # Get all unique symbols
    symbols = list(set(pos["symbol"] for pos in positions))
    
    # Fetch current prices
    try:
        current_prices = await get_stock_prices(symbols)
        logger.info("Fetched prices for %d symbols", len(symbols))
    except Exception as e:
        logger.error("Error fetching stock prices: %s", e)
        current_prices = {symbol: None for symbol in symbols}
    
    # Calculate P&L for each position
    updated_positions = []
    
    for position in positions:
        symbol = position["symbol"]
        position_type = position["position_type"]
        quantity = position["quantity"]
        avg_entry_price = position["avg_entry_price"]
        cost_basis = position["cost_basis"]
        
        current_price = current_prices.get(symbol)
        
        if current_price is not None:
            # Calculate market value
            market_value = current_price * quantity
            
            # Calculate unrealized P&L based on position type
            if position_type == "Long":
                # Long position: profit when current > entry
                unrealized_pnl = market_value - cost_basis
            else:  # Short
                # Short position: profit when current < entry
                # For shorts: we received cash (cost_basis) and owe shares (market_value)
                unrealized_pnl = cost_basis - market_value
            
            # Calculate percentage P&L
            unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            # Calculate total P&L (realized + unrealized)
            realized_pnl = position.get("realized_pnl", 0.0)
            total_pnl = realized_pnl + unrealized_pnl
            total_cost_basis = position.get("total_cost_basis", cost_basis)
            total_pnl_pct = (total_pnl / total_cost_basis) * 100 if total_cost_basis > 0 else 0
            
            # Update position
            position.update({
                "current_price": round(current_price, 4),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl_pct, 2)
            })
            
            logger.debug("%s (%s): Entry=$%.4f, Current=$%.4f, P&L=$%.2f (%.2f%%)", symbol, position_type, avg_entry_price, current_price, unrealized_pnl, unrealized_pnl_pct)
        else:
            # Price not available
            position.update({
                "current_price": None,
                "market_value": None,
                "unrealized_pnl": None,
                "unrealized_pnl_pct": None,
                "total_pnl": None,
                "total_pnl_pct": None
            })
            logger.warning("Could not fetch price for %s", symbol)
        
        updated_positions.append(position)
    
    return updated_positions


def calculate_portfolio_summary(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate portfolio-level summary metrics
    
    Args:
        positions: List of positions with P&L calculated
        
    Returns:
        Dictionary with portfolio summary metrics
    """
    if not positions:
        return {
            "total_positions": 0,
            "total_cost_basis": 0.0,
            "total_market_value": 0.0,
            "total_unrealized_pnl": 0.0,
            "overall_pnl_pct": 0.0,
            "positions_with_prices": 0,
            "long_positions": 0,
            "short_positions": 0
        }
    
    total_cost_basis = 0.0
    total_market_value = 0.0
    total_unrealized_pnl = 0.0
    positions_with_prices = 0
    long_positions = 0
    short_positions = 0
    
    for position in positions:
        # Count positions by type
        if position["position_type"] == "Long":
            long_positions += 1
        else:
            short_positions += 1
        
        # Aggregate values (only for positions with prices)
        if position["current_price"] is not None:
            positions_with_prices += 1
            total_cost_basis += position["cost_basis"]
            total_market_value += position["market_value"]
            total_unrealized_pnl += position["unrealized_pnl"]
    
    # Calculate overall percentage
    overall_pnl_pct = (total_unrealized_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
    
    return {
        "total_positions": len(positions),
        "total_cost_basis": round(total_cost_basis, 2),
        "total_market_value": round(total_market_value, 2),
        "total_unrealized_pnl": round(total_unrealized_pnl, 2),
        "overall_pnl_pct": round(overall_pnl_pct, 2),
        "positions_with_prices": positions_with_prices,
        "long_positions": long_positions,
        "short_positions": short_positions
    }


async def process_open_positions(open_long: Dict, open_short: Dict, roundtrips_df=None) -> Dict[str, Any]:
    """
    Complete workflow to extract and calculate open positions with P&L
    
    Args:
        open_long: Dictionary of long positions
        open_short: Dictionary of short positions
        roundtrips_df: Optional DataFrame of completed roundtrips for realized P&L calculation
        
    Returns:
        Dictionary containing positions and summary
    """
    # Extract positions from tracking dictionaries
    positions = extract_open_positions(open_long, open_short, roundtrips_df)
    
    if not positions:
        return {
            "positions": [],
            "summary": calculate_portfolio_summary([])
        }
    
    # Calculate P&L with current prices
    positions_with_pnl = await calculate_position_pnl(positions)
    
    # Calculate portfolio summary
    summary = calculate_portfolio_summary(positions_with_pnl)
    
    logger.info("Processed %d open positions with total unrealized P&L: $%.2f", len(positions_with_pnl), summary['total_unrealized_pnl'])
    
    return {
        "positions": positions_with_pnl,
        "summary": summary
    }