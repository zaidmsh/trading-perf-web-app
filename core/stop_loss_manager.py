"""
Stop Loss Management Module
Handles setting, storing, and managing stop losses for open positions
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class StopLoss:
    """Stop loss configuration for a position"""
    symbol: str
    position_type: str  # "Long" or "Short"
    stop_loss_type: str  # "amount" or "percentage"
    stop_loss_value: float  # Dollar amount or percentage
    stop_loss_price: float  # Calculated stop loss price
    risk_amount: float  # Total risk in dollars
    risk_percentage: float  # Risk as percentage of position
    created_at: datetime
    is_triggered: bool = False
    triggered_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "symbol": self.symbol,
            "position_type": self.position_type,
            "stop_loss_type": self.stop_loss_type,
            "stop_loss_value": self.stop_loss_value,
            "stop_loss_price": self.stop_loss_price,
            "risk_amount": self.risk_amount,
            "risk_percentage": self.risk_percentage,
            "created_at": self.created_at.isoformat(),
            "is_triggered": self.is_triggered,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StopLoss':
        """Create from dictionary"""
        return cls(
            symbol=data["symbol"],
            position_type=data["position_type"],
            stop_loss_type=data["stop_loss_type"],
            stop_loss_value=data["stop_loss_value"],
            stop_loss_price=data["stop_loss_price"],
            risk_amount=data["risk_amount"],
            risk_percentage=data["risk_percentage"],
            created_at=datetime.fromisoformat(data["created_at"]),
            is_triggered=data.get("is_triggered", False),
            triggered_at=datetime.fromisoformat(data["triggered_at"]) if data.get("triggered_at") else None
        )


class StopLossManager:
    """Manages stop losses for trading positions"""
    
    def __init__(self):
        # Session-based storage: {session_id: {symbol: StopLoss}}
        self.stop_losses: Dict[str, Dict[str, StopLoss]] = {}
    
    def calculate_stop_loss_price(self, entry_price: float, position_type: str, 
                                stop_type: str, stop_value: float) -> float:
        """
        Calculate stop loss price based on entry price and stop parameters
        
        Args:
            entry_price: Average entry price of the position
            position_type: "Long" or "Short"
            stop_type: "amount" or "percentage"
            stop_value: Dollar amount or percentage value
            
        Returns:
            Calculated stop loss price
        """
        if position_type.lower() == "long":
            if stop_type == "amount":
                # For longs: stop is entry price minus dollar amount
                return max(0, entry_price - stop_value)
            elif stop_type == "percentage":
                # For longs: stop is entry price minus percentage
                return max(0, entry_price * (1 - stop_value / 100))
        elif position_type.lower() == "short":
            if stop_type == "amount":
                # For shorts: stop is entry price plus dollar amount
                return entry_price + stop_value
            elif stop_type == "percentage":
                # For shorts: stop is entry price plus percentage
                return entry_price * (1 + stop_value / 100)
        
        raise ValueError(f"Invalid position_type: {position_type}")
    
    def calculate_risk_metrics(self, position: Dict[str, Any], stop_loss_price: float) -> Tuple[float, float]:
        """
        Calculate risk amount and risk percentage for a position
        
        Args:
            position: Position dictionary with entry price, quantity, etc.
            stop_loss_price: Calculated stop loss price
            
        Returns:
            Tuple of (risk_amount, risk_percentage)
        """
        entry_price = position["avg_entry_price"]
        quantity = position["quantity"]
        position_type = position["position_type"]
        
        if position_type.lower() == "long":
            risk_per_share = max(0, entry_price - stop_loss_price)
        else:  # short
            risk_per_share = max(0, stop_loss_price - entry_price)
        
        risk_amount = risk_per_share * quantity
        
        # Risk percentage relative to position cost
        cost_basis = position.get("cost_basis", entry_price * quantity)
        risk_percentage = (risk_amount / cost_basis * 100) if cost_basis > 0 else 0
        
        return risk_amount, risk_percentage
    
    def set_stop_loss(self, session_id: str, position: Dict[str, Any], 
                     stop_type: str, stop_value: float) -> StopLoss:
        """
        Set stop loss for a position
        
        Args:
            session_id: Trading session ID
            position: Position dictionary
            stop_type: "amount" or "percentage"
            stop_value: Dollar amount or percentage value
            
        Returns:
            Created StopLoss object
        """
        symbol = position["symbol"]
        position_type = position["position_type"]
        entry_price = position["avg_entry_price"]
        
        # Validate inputs
        if stop_type not in ["amount", "percentage"]:
            raise ValueError("stop_type must be 'amount' or 'percentage'")
        
        if stop_value <= 0:
            raise ValueError("stop_value must be positive")
        
        if stop_type == "percentage" and stop_value >= 100:
            raise ValueError("percentage stop_value must be less than 100")
        
        # Calculate stop loss price
        stop_loss_price = self.calculate_stop_loss_price(
            entry_price, position_type, stop_type, stop_value
        )
        
        # Calculate risk metrics
        risk_amount, risk_percentage = self.calculate_risk_metrics(position, stop_loss_price)
        
        # Create stop loss object
        stop_loss = StopLoss(
            symbol=symbol,
            position_type=position_type,
            stop_loss_type=stop_type,
            stop_loss_value=stop_value,
            stop_loss_price=stop_loss_price,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            created_at=datetime.now()
        )
        
        # Store in session
        if session_id not in self.stop_losses:
            self.stop_losses[session_id] = {}
        
        self.stop_losses[session_id][symbol] = stop_loss
        
        logger.info(f"Set stop loss for {symbol}: ${stop_loss_price:.4f} (Risk: ${risk_amount:.2f})")
        return stop_loss
    
    def get_stop_loss(self, session_id: str, symbol: str) -> Optional[StopLoss]:
        """Get stop loss for a specific position"""
        return self.stop_losses.get(session_id, {}).get(symbol)
    
    def get_all_stop_losses(self, session_id: str) -> Dict[str, StopLoss]:
        """Get all stop losses for a session"""
        return self.stop_losses.get(session_id, {})
    
    def remove_stop_loss(self, session_id: str, symbol: str) -> bool:
        """
        Remove stop loss for a position
        
        Returns:
            True if stop loss was removed, False if not found
        """
        if session_id in self.stop_losses and symbol in self.stop_losses[session_id]:
            del self.stop_losses[session_id][symbol]
            logger.info(f"Removed stop loss for {symbol}")
            return True
        return False
    
    def check_stop_triggers(self, session_id: str, positions: List[Dict[str, Any]]) -> List[str]:
        """
        Check if any stop losses are triggered by current prices
        
        Args:
            session_id: Trading session ID
            positions: List of positions with current prices
            
        Returns:
            List of symbols with triggered stops
        """
        triggered_symbols = []
        session_stops = self.stop_losses.get(session_id, {})
        
        for position in positions:
            symbol = position["symbol"]
            current_price = position.get("current_price")
            
            if symbol not in session_stops or current_price is None:
                continue
            
            stop_loss = session_stops[symbol]
            if stop_loss.is_triggered:
                continue  # Already triggered
            
            # Check if stop is triggered
            is_triggered = False
            position_type = position["position_type"]
            
            if position_type.lower() == "long":
                # Long position: triggered if current price <= stop price
                is_triggered = current_price <= stop_loss.stop_loss_price
            elif position_type.lower() == "short":
                # Short position: triggered if current price >= stop price
                is_triggered = current_price >= stop_loss.stop_loss_price
            
            if is_triggered:
                stop_loss.is_triggered = True
                stop_loss.triggered_at = datetime.now()
                triggered_symbols.append(symbol)
                logger.warning(f"Stop loss triggered for {symbol} at ${current_price:.4f} (stop: ${stop_loss.stop_loss_price:.4f})")
        
        return triggered_symbols
    
    def calculate_r_multiple(self, position: Dict[str, Any], stop_loss: StopLoss) -> Optional[float]:
        """
        Calculate R-Multiple for a position with stop loss
        
        R-Multiple = Current P&L / Risk Amount
        
        Args:
            position: Position dictionary with unrealized P&L
            stop_loss: StopLoss object
            
        Returns:
            R-Multiple value or None if cannot calculate
        """
        unrealized_pnl = position.get("unrealized_pnl")
        
        if unrealized_pnl is None or stop_loss.risk_amount <= 0:
            return None
        
        return unrealized_pnl / stop_loss.risk_amount
    
    def calculate_free_ride_shares(self, position: Dict[str, Any], r_multiple_levels: List[int] = None) -> List[Dict[str, Any]]:
        """
        Calculate SBE (Shares to Break Even): shares to sell at each R-Multiple to recover initial investment
        
        SBE concept: Calculate exact number of shares to sell at current price to recover the total cost basis,
        making remaining shares a "freeroll" (risk-free position)
        
        Args:
            position: Position dictionary with current price, cost basis, etc.
            r_multiple_levels: List of R-Multiple levels to calculate (default: 1-10)
            
        Returns:
            List of SBE recommendations for each R-Multiple level
        """
        if r_multiple_levels is None:
            r_multiple_levels = list(range(1, 11))  # 1R through 10R
        
        # Get position details
        symbol = position["symbol"]
        position_type = position["position_type"]
        quantity = position["quantity"]
        avg_entry_price = position["avg_entry_price"]
        cost_basis = position["cost_basis"]
        current_price = position.get("current_price")
        
        if current_price is None or current_price <= 0:
            return []
        
        # Get stop loss for risk calculation
        recommendations = []
        
        for r_level in r_multiple_levels:
            try:
                # Calculate target price for this R-Multiple level
                # For Long positions: target_price = entry + (r_level * risk_per_share)
                # For Short positions: target_price = entry - (r_level * risk_per_share)
                
                # We need to estimate risk per share based on current position
                # If no stop loss is set, we can't calculate precise R-Multiples
                # But we can still calculate free ride based on percentage gains
                
                if position_type.lower() == "long":
                    # For long positions, assume a reasonable risk (e.g., 5% of entry price)
                    estimated_risk_per_share = avg_entry_price * 0.05
                    target_price = avg_entry_price + (r_level * estimated_risk_per_share)
                else:  # short
                    # For short positions
                    estimated_risk_per_share = avg_entry_price * 0.05
                    target_price = avg_entry_price - (r_level * estimated_risk_per_share)
                
                # Calculate shares to sell to recover cost basis at target price
                if target_price > 0:
                    # First determine if target is reached
                    target_reached = False
                    if position_type.lower() == "long":
                        target_reached = current_price >= target_price
                    else:  # short
                        target_reached = current_price <= target_price
                    
                    # SBE calculation: How many shares to sell at current price to recover cost basis
                    # This is the key calculation for freerolling the position
                    if target_reached:
                        # Use current price for SBE calculation when target is reached
                        shares_to_sell_at_current = cost_basis / current_price
                    else:
                        # Show what SBE would be at target price
                        shares_to_sell_at_current = cost_basis / target_price
                    
                    shares_to_sell = min(shares_to_sell_at_current, quantity)  # Can't sell more than we have
                    shares_remaining = quantity - shares_to_sell
                    
                    # Calculate recovery amount at execution price
                    if target_reached:
                        recovery_amount = shares_to_sell * current_price
                    else:
                        recovery_amount = shares_to_sell * target_price
                    
                    remaining_value = shares_remaining * current_price if target_reached else None
                    
                    recommendation = {
                        "r_multiple": r_level,
                        "target_price": round(target_price, 4),
                        "shares_to_sell": round(shares_to_sell, 0),
                        "shares_remaining": round(shares_remaining, 0),
                        "recovery_amount": round(recovery_amount, 2),
                        "remaining_value": round(remaining_value, 2) if remaining_value else None,
                        "target_reached": target_reached,
                        "current_price": current_price,
                        "breakeven_protected": target_reached,
                        "sbe_percentage": round((shares_to_sell / quantity) * 100, 1) if quantity > 0 else 0
                    }
                    
                    recommendations.append(recommendation)
                    
            except (ValueError, ZeroDivisionError) as e:
                logger.warning(f"Error calculating free ride for {symbol} at {r_level}R: {e}")
                continue
        
        return recommendations
    
    def calculate_free_ride_with_stop_loss(self, position: Dict[str, Any], stop_loss: StopLoss) -> List[Dict[str, Any]]:
        """
        Calculate SBE (Shares to Break Even) using actual stop loss risk amount for precise R-Multiple calculations
        
        Args:
            position: Position dictionary
            stop_loss: StopLoss object with risk calculations
            
        Returns:
            List of SBE recommendations based on actual risk
        """
        # Get position details
        symbol = position["symbol"]
        position_type = position["position_type"]
        quantity = position["quantity"]
        avg_entry_price = position["avg_entry_price"]
        cost_basis = position["cost_basis"]
        current_price = position.get("current_price")
        
        if current_price is None or current_price <= 0 or stop_loss.risk_amount <= 0:
            return []
        
        # Calculate risk per share from stop loss
        risk_per_share = stop_loss.risk_amount / quantity
        
        recommendations = []
        
        for r_level in range(1, 11):  # 1R through 10R
            try:
                # Calculate target price for this R-Multiple level
                if position_type.lower() == "long":
                    target_price = avg_entry_price + (r_level * risk_per_share)
                else:  # short
                    target_price = avg_entry_price - (r_level * risk_per_share)
                
                if target_price <= 0:
                    continue
                
                # Calculate if current price has reached this target
                target_reached = False
                if position_type.lower() == "long":
                    target_reached = current_price >= target_price
                else:  # short
                    target_reached = current_price <= target_price
                
                # SBE calculation: shares to sell to recover cost basis
                if target_reached:
                    # When profitable, calculate SBE at current price
                    shares_to_sell = cost_basis / current_price
                else:
                    # Show what SBE would be at target price
                    shares_to_sell = cost_basis / target_price
                
                shares_to_sell = min(shares_to_sell, quantity)  # Can't sell more than we have
                shares_remaining = max(0, quantity - shares_to_sell)
                
                # Calculate recovery amount and remaining value
                recovery_amount = shares_to_sell * (current_price if target_reached else target_price)
                remaining_value = shares_remaining * current_price if target_reached else None
                
                # Calculate breakeven protection
                if target_reached and shares_to_sell > 0:
                    # If we sell the recommended shares now, what happens if stopped out?
                    remaining_cost = cost_basis - recovery_amount
                    stop_loss_value = shares_remaining * stop_loss.stop_loss_price
                    net_result = stop_loss_value - remaining_cost
                    breakeven_protected = net_result >= -1  # Allow $1 tolerance for commissions
                else:
                    breakeven_protected = False
                
                recommendation = {
                    "r_multiple": r_level,
                    "target_price": round(target_price, 4),
                    "shares_to_sell": round(shares_to_sell, 0),
                    "shares_remaining": round(shares_remaining, 0),
                    "recovery_amount": round(recovery_amount, 2),
                    "remaining_value": round(remaining_value, 2) if remaining_value else None,
                    "target_reached": target_reached,
                    "current_price": current_price,
                    "breakeven_protected": breakeven_protected,
                    "risk_per_share": round(risk_per_share, 4)
                }
                
                recommendations.append(recommendation)
                
            except (ValueError, ZeroDivisionError) as e:
                logger.warning(f"Error calculating free ride with stop loss for {symbol} at {r_level}R: {e}")
                continue
        
        return recommendations
    
    def clear_session(self, session_id: str):
        """Clear all stop losses for a session"""
        if session_id in self.stop_losses:
            del self.stop_losses[session_id]
            logger.info(f"Cleared stop losses for session {session_id}")


# Global instance
stop_loss_manager = StopLossManager()