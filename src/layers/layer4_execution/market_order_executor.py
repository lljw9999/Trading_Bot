"""
Market Order Executor

Basic execution engine that places market orders based on position sizing decisions.
Includes order management, fill tracking, and transaction cost analysis.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uuid

from ...utils.logger import get_logger


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class Order:
    """Order representation."""
    
    def __init__(self, 
                 symbol: str,
                 side: OrderSide,
                 quantity: Decimal,
                 order_type: str = "market",
                 price: Optional[Decimal] = None):
        """
        Initialize order.
        
        Args:
            symbol: Trading symbol
            side: Buy or sell
            quantity: Order quantity
            order_type: Order type (market, limit, etc.)
            price: Limit price (for limit orders)
        """
        self.order_id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side
        self.quantity = abs(quantity)  # Always positive
        self.order_type = order_type
        self.price = price
        
        self.status = OrderStatus.PENDING
        self.filled_quantity = Decimal('0')
        self.avg_fill_price = None
        self.commission = Decimal('0')
        
        self.created_at = datetime.utcnow()
        self.submitted_at = None
        self.filled_at = None
        
        self.fills = []  # List of fill records
        self.metadata = {}


class Fill:
    """Fill record."""
    
    def __init__(self, 
                 order_id: str,
                 quantity: Decimal,
                 price: Decimal,
                 commission: Decimal = Decimal('0')):
        """Initialize fill record."""
        self.fill_id = str(uuid.uuid4())
        self.order_id = order_id
        self.quantity = quantity
        self.price = price
        self.commission = commission
        self.timestamp = datetime.utcnow()


class MarketOrderExecutor:
    """
    Market order execution engine.
    
    Handles order placement, execution simulation, and trade tracking
    for the trading system.
    """
    
    def __init__(self, 
                 commission_rate: float = 0.001,  # 0.1% commission
                 slippage_bps: float = 2.0,       # 2bp average slippage
                 paper_trading: bool = True):
        """
        Initialize market order executor.
        
        Args:
            commission_rate: Commission as fraction of trade value
            slippage_bps: Average slippage in basis points
            paper_trading: Whether to simulate trades or use real API
        """
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.paper_trading = paper_trading
        
        self.logger = get_logger("execution.market_orders")
        
        # Order tracking
        self.orders = {}  # {order_id: Order}
        self.order_history = []
        self.fills = []
        
        # Position tracking
        self.positions = {}  # {symbol: Decimal}
        self.cash_balance = Decimal('100000')  # Starting cash
        
        # Performance tracking
        self.total_trades = 0
        self.total_commission = Decimal('0')
        self.total_slippage = Decimal('0')
        
        self.logger.info(
            f"Market executor initialized: commission={commission_rate:.1%}, "
            f"slippage={slippage_bps}bps, paper_trading={paper_trading}"
        )
    
    async def execute_order(self, 
                          symbol: str,
                          target_position: Decimal,
                          current_price: Decimal) -> Optional[Order]:
        """
        Execute order to reach target position.
        
        Args:
            symbol: Trading symbol
            target_position: Target position in dollars
            current_price: Current market price
            
        Returns:
            Order object if order was placed, None otherwise
        """
        try:
            # Calculate current position in dollars
            current_shares = self.positions.get(symbol, Decimal('0'))
            current_position_dollars = current_shares * current_price
            
            # Calculate required trade
            position_delta = target_position - current_position_dollars
            
            # Check if trade is needed
            min_trade_size = Decimal('10')  # Minimum $10 trade
            if abs(position_delta) < min_trade_size:
                self.logger.debug(f"No trade needed for {symbol}: delta=${position_delta:.2f}")
                return None
            
            # Determine order side and quantity
            if position_delta > 0:
                side = OrderSide.BUY
                shares_to_trade = position_delta / current_price
            else:
                side = OrderSide.SELL
                shares_to_trade = abs(position_delta) / current_price
            
            # Create order
            order = Order(
                symbol=symbol,
                side=side,
                quantity=shares_to_trade,
                order_type="market"
            )
            
            # Execute the order
            success = await self._execute_order_impl(order, current_price)
            
            if success:
                self.logger.info(
                    f"Executed {side.value} {shares_to_trade:.6f} shares of {symbol} "
                    f"at ~${current_price:.2f}"
                )
                return order
            else:
                self.logger.error(f"Failed to execute order for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing order: {e}")
            return None
    
    async def _execute_order_impl(self, order: Order, market_price: Decimal) -> bool:
        """
        Internal order execution implementation.
        
        In paper trading mode, simulates the execution.
        In live trading mode, would interface with broker API.
        """
        try:
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.utcnow()
            self.orders[order.order_id] = order
            
            if self.paper_trading:
                # Simulate execution
                return await self._simulate_execution(order, market_price)
            else:
                # TODO: Implement real broker API execution
                self.logger.warning("Live trading not implemented yet")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in order execution: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    async def _simulate_execution(self, order: Order, market_price: Decimal) -> bool:
        """Simulate order execution for paper trading."""
        try:
            # Simulate network delay
            await asyncio.sleep(0.1)
            
            # Calculate execution price with slippage
            slippage_factor = 1 + (self.slippage_bps / 10000)
            if order.side == OrderSide.BUY:
                execution_price = market_price * Decimal(str(slippage_factor))
            else:
                execution_price = market_price / Decimal(str(slippage_factor))
            
            # Calculate commission
            trade_value = order.quantity * execution_price
            commission = trade_value * Decimal(str(self.commission_rate))
            
            # Create fill
            fill = Fill(
                order_id=order.order_id,
                quantity=order.quantity,
                price=execution_price,
                commission=commission
            )
            
            # Update order
            order.filled_quantity = order.quantity
            order.avg_fill_price = execution_price
            order.commission = commission
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.utcnow()
            order.fills.append(fill)
            
            # Update positions
            shares_delta = order.quantity if order.side == OrderSide.BUY else -order.quantity
            self.positions[order.symbol] = self.positions.get(order.symbol, Decimal('0')) + shares_delta
            
            # Update cash balance
            cash_delta = -trade_value - commission if order.side == OrderSide.BUY else trade_value - commission
            self.cash_balance += cash_delta
            
            # Track performance
            self.total_trades += 1
            self.total_commission += commission
            self.total_slippage += abs(execution_price - market_price) * order.quantity
            
            # Store records
            self.fills.append(fill)
            self.order_history.append(order)
            
            self.logger.debug(
                f"Simulated fill: {order.symbol} {order.side.value} {order.quantity:.6f} "
                f"@ ${execution_price:.4f}, commission=${commission:.2f}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error simulating execution: {e}")
            return False
    
    def get_position(self, symbol: str) -> Decimal:
        """Get current position in shares for a symbol."""
        return self.positions.get(symbol, Decimal('0'))
    
    def get_position_value(self, symbol: str, current_price: Decimal) -> Decimal:
        """Get current position value in dollars."""
        shares = self.get_position(symbol)
        return shares * current_price
    
    def get_portfolio_value(self, current_prices: Dict[str, Decimal]) -> Decimal:
        """Calculate total portfolio value."""
        try:
            portfolio_value = self.cash_balance
            
            for symbol, shares in self.positions.items():
                if symbol in current_prices and shares != 0:
                    portfolio_value += shares * current_prices[symbol]
            
            return portfolio_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return self.cash_balance
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get list of open orders."""
        open_statuses = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}
        
        open_orders = [
            order for order in self.orders.values()
            if order.status in open_statuses
        ]
        
        if symbol:
            open_orders = [order for order in open_orders if order.symbol == symbol]
        
        return open_orders
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            if order_id in self.orders:
                order = self.orders[order_id]
                if order.status in {OrderStatus.PENDING, OrderStatus.SUBMITTED}:
                    order.status = OrderStatus.CANCELLED
                    self.logger.info(f"Cancelled order {order_id}")
                    return True
                else:
                    self.logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
                    return False
            else:
                self.logger.error(f"Order {order_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_trade_history(self, symbol: Optional[str] = None, 
                         days: int = 30) -> List[Order]:
        """Get trade history."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        trades = [
            order for order in self.order_history
            if order.filled_at and order.filled_at >= cutoff_date
        ]
        
        if symbol:
            trades = [order for order in trades if order.symbol == symbol]
        
        return trades
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get execution performance metrics."""
        try:
            recent_trades = self.get_trade_history(days=30)
            
            if recent_trades:
                avg_commission = sum(order.commission for order in recent_trades) / len(recent_trades)
                total_volume = sum(order.quantity * order.avg_fill_price for order in recent_trades)
                avg_slippage_bps = (self.total_slippage / total_volume) * 10000 if total_volume > 0 else 0
            else:
                avg_commission = 0
                total_volume = 0
                avg_slippage_bps = 0
            
            return {
                'total_trades': self.total_trades,
                'total_commission': float(self.total_commission),
                'total_slippage': float(self.total_slippage),
                'avg_commission_per_trade': float(avg_commission),
                'avg_slippage_bps': float(avg_slippage_bps),
                'recent_trades_30d': len(recent_trades),
                'cash_balance': float(self.cash_balance),
                'num_positions': len([p for p in self.positions.values() if p != 0])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            'executor_type': 'market_orders',
            'paper_trading': self.paper_trading,
            'parameters': {
                'commission_rate': self.commission_rate,
                'slippage_bps': self.slippage_bps
            },
            'performance': self.get_performance_metrics(),
            'positions': {symbol: float(shares) for symbol, shares in self.positions.items()},
            'open_orders': len(self.get_open_orders())
        } 