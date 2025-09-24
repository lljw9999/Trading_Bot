def optimal_offset(spread, depth_bps):
    """
    spread = raw bid-ask spread in bp, depth_bps = book depth at px±n ticks.
    Returns limit-offset (±bp) that maximises (edge - slip).
    """
    penalty = depth_bps * 0.1  # 10 % of depth as cost
    return max(0.5, min(spread * 0.8 - penalty, spread - 0.1))
