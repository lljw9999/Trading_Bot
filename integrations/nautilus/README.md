# NautilusTrader Integration

This module provides integration with [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) for high-performance backtesting and optional live execution while maintaining compatibility with existing Redis-based monitoring and compliance systems.

## Overview

The integration consists of:

- **Bridge**: Bidirectional event bridge between Nautilus and Redis
- **Strategy Port**: Nautilus version of the basis carry strategy with feature parity
- **Backtest Runner**: Deterministic backtesting harness for acceptance testing
- **Shadow Executor**: Live execution with reduced size for performance measurement

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nautilus      │    │     Bridge      │    │     Redis       │
│   Engine        │◄──►│   (Events)      │◄──►│   Streams       │
│                 │    │                 │    │                 │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Orders        │    │ • Event Trans.  │    │ • exec:orders   │
│ • Fills         │    │ • Risk Commands │    │ • exec:fills    │
│ • Market Data   │    │ • Metrics Pub.  │    │ • market.raw.*  │
│ • Strategies    │    │ • FIFO Ledger   │    │ • metrics:*     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Event Mapping Tables

### Orders → Redis

| Nautilus Event | Redis Stream | Fields |
|----------------|--------------|---------|
| `OrderAccepted` | `exec:orders` | `{status:"accepted", venue, cl_ord_id, symbol, side, qty, tif, ts_ns}` |
| `OrderFilled` | `exec:fills` | `{fill_id, cl_ord_id, price, qty, fee, venue, ts_ns}` |
| `OrderCanceled` | `exec:orders` | `{status:"canceled", reason}` |

### Market Data → Redis

| Nautilus Event | Redis Stream | Fields |
|----------------|--------------|---------|
| `QuoteTick` | `market.raw.<asset>.<symbol>` | `{bid_px, bid_sz, ask_px, ask_sz, mid, ts_ns}` |
| `TradeTick` | `market.raw.trades.<symbol>` | `{price, size, side, ts_ns, trade_id}` |

### Metrics → Redis/Prometheus

| Metric | Description | Type |
|--------|-------------|------|
| `nautilus_bridge_events_processed` | Total events processed | Counter |
| `nautilus_bridge_orders_published` | Orders published to Redis | Counter |
| `nautilus_bridge_fills_published` | Fills published to Redis | Counter |
| `nautilus_bridge_redis_errors` | Redis publish errors | Counter |

## Usage

### 1. Backtesting

```bash
# Run acceptance backtest
python integrations/nautilus/run_backtest.py \
  --symbols BTC ETH SOL \
  --start 2025-07-01 --end 2025-07-07 \
  --dataset data/binance/spot.parquet \
  --output artifacts/acceptance/nautilus_results.json
```

### 2. Shadow Live Execution

```bash
# Start shadow execution (Deribit shadow only)
python integrations/nautilus/run_shadow_live.py \
  --venues BINANCE DERIBIT \
  --symbols BTC ETH SOL \
  --shadow
```

### 3. Bridge Integration

```python
from integrations.nautilus.bridge import NautilusRedisBridge

# Create and start bridge
bridge = NautilusRedisBridge()
await bridge.start()

# Register custom event handlers
bridge.register_fill_handler(my_fill_handler)
bridge.register_order_handler(my_order_handler)
```

## Feature Flags

Control integration behavior via Redis feature flags:

| Flag | Description | Default |
|------|-------------|---------|
| `features:nautilus_engine` | Enable Nautilus backtesting | `0` |
| `features:nautilus_shadow_exec` | Enable shadow execution | `0` |
| `features:nautilus_live_exec` | Enable live execution | `0` (stay off) |

## Acceptance Testing

The integration includes tolerance-based acceptance testing:

- **PnL difference**: ≤ 0.05%
- **Implementation Shortfall**: ≤ 0.5 bps
- **Slippage difference**: ≤ 1.0 bps

Run acceptance tests:

```bash
make accept-nt       # Run Nautilus backtest
make accept-compare  # Compare with original implementation
```

## Safety & Rollback

### Immediate Rollback
```bash
# Disable live execution, keep shadow/backtest
redis-cli set features:nautilus_live_exec 0
redis-cli set features:nautilus_shadow_exec 0

# Emergency halt
redis-cli set mode halt
```

### Monitoring Integration

The bridge ensures seamless integration with existing systems:

- **Grafana**: All existing dashboards continue to work
- **Ops Tab**: Status tiles show unified metrics
- **Compliance**: FIFO ledger and WORM archiver receive all fills
- **Risk Management**: Redis-based halts and limits apply to Nautilus

## Performance Benefits

Expected improvements with Nautilus:

- **Backtesting**: 10-100x faster execution
- **Latency**: Sub-millisecond order processing
- **Precision**: Nanosecond timestamp resolution
- **Memory**: Efficient Rust-based data structures

## Installation

Add to `requirements.txt`:
```
nautilus-trader>=1.194.0
pyarrow>=15.0.0
```

Ensure Rust toolchain is available for compilation.

## Development

### Running Tests

```bash
# Test bridge in mock mode
python integrations/nautilus/bridge.py --test

# Test strategy port
python integrations/nautilus/strategy_basis_carry_nt.py

# Test backtest runner
python integrations/nautilus/run_backtest.py --symbols BTC --start 2025-07-01 --end 2025-07-02
```

### Adding New Strategies

1. Extend `NautilusBasisCarryStrategy` or create new strategy class
2. Implement required Nautilus strategy methods
3. Add to backtest runner and bridge integration
4. Update acceptance tests with new tolerance checks

## Troubleshooting

### Common Issues

1. **Import Error**: `ModuleNotFoundError: No module named 'nautilus_trader'`
   - Install: `pip install nautilus-trader`
   - Or use fallback mode (automatically detected)

2. **Bridge Connection Failures**
   - Check Redis connectivity
   - Verify feature flags are set correctly
   - Review bridge logs for authentication errors

3. **Backtest Data Issues**
   - Ensure data files are in Parquet format
   - Check instrument definitions match data
   - Verify timestamp formats and timezones

### Logging

Enable detailed logging:
```python
import logging
logging.getLogger("nautilus_bridge").setLevel(logging.DEBUG)
logging.getLogger("nautilus_backtest").setLevel(logging.DEBUG)
```

## License Compliance

NautilusTrader is licensed under LGPL-3. This integration:
- Uses Nautilus as a dynamically-linked library
- Does not modify Nautilus core code
- Maintains separate codebase for custom logic

## Contributing

1. Follow existing code patterns
2. Add tests for new functionality
3. Update documentation and mapping tables
4. Ensure backward compatibility with Redis systems