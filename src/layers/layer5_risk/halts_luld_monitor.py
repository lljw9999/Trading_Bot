"""
Trading Halts and LULD (Limit Up-Limit Down) Monitor

Monitors for trading halts and LULD band violations.
Blocks orders when halted and throttles when near LULD bands.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from ...utils.logger import get_logger


class HaltLULDMonitor:
    """Monitors trading halts and LULD conditions"""

    def __init__(self, luld_buffer_pct: float = 1.0, halt_check_interval: float = 5.0):
        """
        Initialize halt/LULD monitor.

        Args:
            luld_buffer_pct: Buffer percentage from LULD bands (default: 1%)
            halt_check_interval: How often to check halt status (default: 5s)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.luld_buffer_pct = luld_buffer_pct / 100.0
        self.halt_check_interval = halt_check_interval

        # Track halt states
        self.halted_symbols = set()
        self.luld_bands = {}  # symbol -> {"up": float, "down": float}

        self.logger.info(f"Halt/LULD Monitor initialized: buffer={luld_buffer_pct}%")

    def on_tick(
        self, symbol: str, price: float, bands: Optional[Dict[str, float]] = None
    ):
        """
        Process market tick and check for halt/LULD conditions.

        Args:
            symbol: Stock symbol
            price: Current price
            bands: LULD bands dict with 'up' and 'down' keys (if available)
        """
        try:
            # Update LULD bands if provided
            if bands:
                self.luld_bands[symbol] = bands

            # Check LULD proximity
            luld_status = self._check_luld_proximity(symbol, price)

            # Check halt status (would integrate with broker feed)
            halt_status = self._check_halt_status(symbol)

            return {
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "luld_status": luld_status,
                "halt_status": halt_status,
                "should_block_orders": halt_status["is_halted"],
                "should_throttle_orders": luld_status["near_bands"],
            }

        except Exception as e:
            self.logger.error(f"Error processing tick for {symbol}: {e}")
            return None

    def _check_luld_proximity(self, symbol: str, price: float) -> Dict[str, Any]:
        """Check proximity to LULD bands"""
        bands = self.luld_bands.get(symbol)

        if not bands:
            return {
                "has_bands": False,
                "near_bands": False,
                "distance_to_up_pct": None,
                "distance_to_down_pct": None,
            }

        up_band = bands.get("up")
        down_band = bands.get("down")

        if not up_band or not down_band:
            return {"has_bands": False, "near_bands": False}

        # Calculate distances
        distance_to_up_pct = ((up_band - price) / price) * 100.0
        distance_to_down_pct = ((price - down_band) / price) * 100.0

        # Check if near bands (within buffer)
        near_up = distance_to_up_pct <= self.luld_buffer_pct * 100.0
        near_down = distance_to_down_pct <= self.luld_buffer_pct * 100.0

        result = {
            "has_bands": True,
            "up_band": up_band,
            "down_band": down_band,
            "distance_to_up_pct": distance_to_up_pct,
            "distance_to_down_pct": distance_to_down_pct,
            "near_up_band": near_up,
            "near_down_band": near_down,
            "near_bands": near_up or near_down,
        }

        if near_up or near_down:
            self.logger.warning(
                f"⚠️ LULD WARNING for {symbol}: price {price} near bands [{down_band}-{up_band}]"
            )

        return result

    def _check_halt_status(self, symbol: str) -> Dict[str, Any]:
        """Check if symbol is halted"""
        # Mock implementation - would integrate with broker halt feeds
        is_halted = symbol in self.halted_symbols

        return {
            "is_halted": is_halted,
            "halt_reason": "MOCK_HALT" if is_halted else None,
            "halt_start_time": (
                datetime.now(timezone.utc).isoformat() if is_halted else None
            ),
            "estimated_resume_time": None,
        }

    def set_halt_status(
        self, symbol: str, is_halted: bool, reason: Optional[str] = None
    ):
        """Set halt status for a symbol (for testing/simulation)"""
        if is_halted:
            self.halted_symbols.add(symbol)
            self.logger.warning(f"🛑 HALT: {symbol} - {reason or 'Unknown reason'}")
        else:
            self.halted_symbols.discard(symbol)
            self.logger.info(f"✅ RESUME: {symbol}")

    def set_luld_bands(self, symbol: str, up_band: float, down_band: float):
        """Set LULD bands for a symbol"""
        self.luld_bands[symbol] = {"up": up_band, "down": down_band}
        self.logger.info(f"📊 LULD bands set for {symbol}: [{down_band}-{up_band}]")

    def get_halted_symbols(self) -> List[str]:
        """Get list of currently halted symbols"""
        return list(self.halted_symbols)

    def should_block_order(self, symbol: str) -> bool:
        """Check if orders should be blocked for a symbol"""
        return symbol in self.halted_symbols


def create_halt_luld_monitor(**kwargs) -> HaltLULDMonitor:
    """Factory function to create HaltLULDMonitor instance."""
    return HaltLULDMonitor(**kwargs)
