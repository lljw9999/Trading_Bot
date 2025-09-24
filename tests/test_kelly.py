from src.risk.kelly_vol_sizer import kelly_size


def test_basic():
    assert abs(kelly_size(0.01, 0.2, 0.02) - 0.0049) < 1e-4


def test_edge_cases():
    """Test Kelly sizing edge cases."""
    # Zero edge should give zero size
    assert kelly_size(0.0, 0.2, 0.02) == 0.0

    # High volatility should reduce size
    high_vol_size = kelly_size(0.01, 0.5, 0.02)
    low_vol_size = kelly_size(0.01, 0.1, 0.02)
    assert high_vol_size < low_vol_size

    # Risk cap should be enforced
    large_edge_size = kelly_size(1.0, 0.1, 0.02)  # Very large edge
    assert abs(large_edge_size) <= 0.02  # Should be capped at risk_cap


def test_negative_edge():
    """Test Kelly sizing with negative edge (losing trade signal)."""
    negative_size = kelly_size(-0.01, 0.2, 0.02)
    positive_size = kelly_size(0.01, 0.2, 0.02)

    # Should be symmetric but opposite sign
    assert abs(negative_size + positive_size) < 1e-10
