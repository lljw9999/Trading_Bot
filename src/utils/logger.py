"""
Logging utilities for the Trading System

Provides centralized logging configuration and utilities.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from .config_manager import config


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    Set up logging configuration for the entire system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        log_format: Log message format string
    """
    # Get configuration values
    log_level = log_level or config.get("logging.level", "INFO")
    log_file = log_file or config.get("logging.file_path", "logs/trading.log")
    log_format = log_format or config.get(
        "logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    max_size = config.get("logging.max_size_mb", 100) * 1024 * 1024  # Convert to bytes
    backup_count = config.get("logging.backup_count", 5)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_size, backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Create separate loggers for different components
    _setup_component_loggers()

    logging.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def _setup_component_loggers() -> None:
    """Set up component-specific loggers with appropriate levels."""

    # Trading-specific loggers
    trading_logger = logging.getLogger("trading")
    risk_logger = logging.getLogger("risk")
    execution_logger = logging.getLogger("execution")

    # Data loggers
    data_logger = logging.getLogger("data")
    market_data_logger = logging.getLogger("market_data")

    # Model loggers
    alpha_logger = logging.getLogger("alpha_models")
    ensemble_logger = logging.getLogger("ensemble")

    # External library loggers (reduce verbosity)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TradingLogger:
    """
    Specialized logger for trading operations with structured logging.
    """

    def __init__(self, name: str):
        """
        Initialize trading logger.

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)

    def log_trade(
        self, action: str, symbol: str, quantity: float, price: float, **kwargs
    ) -> None:
        """
        Log trade execution with structured data.

        Args:
            action: Trade action (BUY, SELL)
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            **kwargs: Additional trade metadata
        """
        trade_data = {
            "action": action,
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            **kwargs,
        }

        self.logger.info(f"TRADE_EXECUTED: {trade_data}")

    def log_signal(
        self, model: str, symbol: str, signal: float, confidence: float, **kwargs
    ) -> None:
        """
        Log alpha signal generation.

        Args:
            model: Model name
            symbol: Trading symbol
            signal: Signal value
            confidence: Signal confidence
            **kwargs: Additional signal metadata
        """
        signal_data = {
            "model": model,
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            **kwargs,
        }

        self.logger.info(f"SIGNAL_GENERATED: {signal_data}")

    def log_risk_event(
        self, event_type: str, severity: str, description: str, **kwargs
    ) -> None:
        """
        Log risk management events.

        Args:
            event_type: Type of risk event
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            description: Event description
            **kwargs: Additional event metadata
        """
        risk_data = {
            "event_type": event_type,
            "severity": severity,
            "description": description,
            **kwargs,
        }

        log_level = {
            "LOW": logging.INFO,
            "MEDIUM": logging.WARNING,
            "HIGH": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }.get(severity, logging.INFO)

        self.logger.log(log_level, f"RISK_EVENT: {risk_data}")

    def log_performance(
        self, metric: str, value: float, timeframe: str, **kwargs
    ) -> None:
        """
        Log performance metrics.

        Args:
            metric: Performance metric name
            value: Metric value
            timeframe: Timeframe for the metric
            **kwargs: Additional performance data
        """
        perf_data = {"metric": metric, "value": value, "timeframe": timeframe, **kwargs}

        self.logger.info(f"PERFORMANCE: {perf_data}")


# Initialize logging when module is imported
setup_logging()
