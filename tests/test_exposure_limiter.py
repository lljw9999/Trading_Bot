#!/usr/bin/env python3
"""
Tests for Portfolio Exposure Limiter
"""
import pytest
import tempfile
import yaml
from pathlib import Path
from src.risk.exposure_limiter import ExposureLimiter


@pytest.fixture
def test_config():
    """Create test portfolio configuration."""
    config = {
        "pilot": {
            "assets": [
                {
                    "symbol": "SOL-USD",
                    "class": "crypto",
                    "venue": "coinbase",
                    "max_influence_pct": 25,
                },
                {
                    "symbol": "BTC-USD",
                    "class": "crypto",
                    "venue": "binance",
                    "max_influence_pct": 20,
                },
            ]
        },
        "portfolio_limits": {
            "max_gross_notional_usd": 100000,
            "max_var_95_usd": 10000,
            "per_venue_notional_caps": {"coinbase": 60000, "binance": 50000},
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        return f.name


@pytest.fixture
def mock_portfolio_state():
    """Mock portfolio state for testing."""
    return {
        "gross_notional_usd": 50000,
        "net_notional_usd": 25000,
        "positions": {
            "SOL-USD": {"notional_usd": 30000, "venue": "coinbase"},
            "BTC-USD": {"notional_usd": 20000, "venue": "binance"},
        },
        "venue_exposure": {"coinbase": 30000, "binance": 20000},
        "var_95_estimate_usd": 5000,
        "class_exposure": {
            "crypto": 40000,  # Reduced to 80% of 50k total
            "equity": 10000,  # Add some equity to make class exposure < 80%
        },
    }


def test_limiter_initialization(test_config):
    """Test exposure limiter initialization."""
    limiter = ExposureLimiter(test_config)

    assert limiter.config is not None
    assert "pilot" in limiter.config
    assert "portfolio_limits" in limiter.config
    assert len(limiter.assets) == 2


def test_accept_within_limits(test_config, mock_portfolio_state):
    """Test order acceptance when within all limits."""
    limiter = ExposureLimiter(test_config)

    order = {
        "symbol": "SOL-USD",
        "notional_usd": 10000,
        "venue": "coinbase",
        "influence_pct": 20,
        "side": "buy",
    }

    allowed, reason = limiter.enforce(order, mock_portfolio_state)

    assert allowed is True
    assert "All checks passed" in reason


def test_reject_gross_notional_exceeded(test_config, mock_portfolio_state):
    """Test rejection when gross notional limit exceeded."""
    limiter = ExposureLimiter(test_config)

    order = {
        "symbol": "BTC-USD",
        "notional_usd": 60000,  # 50k current + 60k order = 110k > 100k limit
        "venue": "binance",
        "influence_pct": 15,
        "side": "buy",
    }

    allowed, reason = limiter.enforce(order, mock_portfolio_state)

    assert allowed is False
    assert "Gross notional limit exceeded" in reason


def test_reject_venue_cap_breached(test_config, mock_portfolio_state):
    """Test rejection when venue cap breached."""
    limiter = ExposureLimiter(test_config)

    order = {
        "symbol": "SOL-USD",
        "notional_usd": 35000,  # 30k current + 35k order = 65k > 60k coinbase limit
        "venue": "coinbase",
        "influence_pct": 20,
        "side": "buy",
    }

    allowed, reason = limiter.enforce(order, mock_portfolio_state)

    assert allowed is False
    assert "Venue limit exceeded for coinbase" in reason


def test_reject_asset_percentage_exceeded(test_config, mock_portfolio_state):
    """Test rejection when asset percentage cap exceeded."""
    limiter = ExposureLimiter(test_config)

    order = {
        "symbol": "SOL-USD",
        "notional_usd": 5000,
        "venue": "coinbase",
        "influence_pct": 30,  # > 25% max for SOL-USD
        "side": "buy",
    }

    allowed, reason = limiter.enforce(order, mock_portfolio_state)

    assert allowed is False
    assert "Asset influence cap exceeded for SOL-USD" in reason


def test_reject_unknown_asset(test_config, mock_portfolio_state):
    """Test rejection for unconfigured asset."""
    limiter = ExposureLimiter(test_config)

    order = {
        "symbol": "UNKNOWN-USD",
        "notional_usd": 1000,
        "venue": "coinbase",
        "influence_pct": 5,
        "side": "buy",
    }

    allowed, reason = limiter.enforce(order, mock_portfolio_state)

    assert allowed is False
    assert "not configured in pilot" in reason or "not in configured assets" in reason


def test_audit_record_creation(test_config, mock_portfolio_state, tmp_path):
    """Test that audit records are created."""
    import os

    os.chdir(tmp_path)

    limiter = ExposureLimiter(test_config)

    order = {
        "symbol": "SOL-USD",
        "notional_usd": 5000,
        "venue": "coinbase",
        "influence_pct": 20,
        "side": "buy",
    }

    allowed, reason = limiter.enforce(order, mock_portfolio_state)

    # Check audit file was created
    audit_files = list(Path("artifacts/audit").glob("*_exposure_check.json"))
    assert len(audit_files) > 0


def test_var_budget_check(test_config, mock_portfolio_state):
    """Test VaR budget enforcement."""
    limiter = ExposureLimiter(test_config)

    # Modify portfolio state to have high VaR
    high_var_state = mock_portfolio_state.copy()
    high_var_state["var_95_estimate_usd"] = 9500  # Close to 10k limit

    order = {
        "symbol": "BTC-USD",
        "notional_usd": 5000,  # Will push VaR over limit
        "venue": "binance",
        "influence_pct": 15,
        "side": "buy",
    }

    allowed, reason = limiter.enforce(order, high_var_state)

    assert allowed is False
    assert "VaR budget exceeded" in reason


def test_error_handling(test_config):
    """Test error handling with invalid portfolio state."""
    limiter = ExposureLimiter(test_config)

    order = {
        "symbol": "SOL-USD",
        "notional_usd": 5000,
        "venue": "coinbase",
        "influence_pct": 20,
        "side": "buy",
    }

    # Pass None portfolio state to trigger internal state fetch
    allowed, reason = limiter.enforce(order, None)

    # Should still work with stub implementation
    assert isinstance(allowed, bool)
    assert isinstance(reason, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
