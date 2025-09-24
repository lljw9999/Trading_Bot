#!/bin/bash
set -e

echo "📦 Enabling Equities Trading (Safe Mode)"
echo "======================================="

# Mode halt during reconfig
echo "🛑 Setting mode to halt during reconfiguration..."
redis-cli set mode halt

# Flags: disable crypto specifics, enable stocks
echo "🔧 Configuring feature flags..."
redis-cli hset features:flags FUNDING_BASIS 0 WHALE_ALERT 0 DERIBIT_HEDGE 0
redis-cli hset features:flags TRADE_CRYPTO 0 TRADE_STOCKS 1
redis-cli del symbols:crypto
redis-cli sadd symbols:stocks AAPL MSFT NVDA SPY

# Risk guards config
echo "🛡️ Configuring risk guards..."
redis-cli hset risk:market_hours_guard enabled 1 pre "09:25" post "16:05" tz "America/New_York" allow_afterhours 0
redis-cli hset risk:pdt_guard enabled 1 min_equity_usd 26000 max_daytrades_5d 3
redis-cli hset risk:ssr_guard enabled 1
redis-cli set risk:overnight_flat 1
redis-cli set risk:max_overnight_gross_pct 0

# Router weights for equities; RL in shadow
echo "⚖️ Setting router weights..."
redis-cli hset router:weights:equities OBP 0.35 MAM 0.25 TFT 0.25 NEWS 0.15 RL 0.00
redis-cli hset features:flags EXEC_RL_SHADOW 1 EXEC_RL_LIVE 0
redis-cli set risk:capital_stage 10

# Fee configuration for equities
echo "💰 Configuring equities fees..."
redis-cli hset fees:equities:alpaca commission_per_share 0.0 sec_fee_bps 0.00276 taf_fee_bps 0.0095 nscc_fee_usd 0.01

# Check if we have sudo access for systemd services
echo "🔧 Configuring services..."
if command -v systemctl >/dev/null 2>&1; then
    echo "📋 Installing systemd services (requires sudo)..."
    # Only copy services if we can write to /etc/systemd/system/
    if [ -w /etc/systemd/system/ ] 2>/dev/null || sudo -n true 2>/dev/null; then
        sudo cp systemd/market_hours_guard.service /etc/systemd/system/ 2>/dev/null || true
        sudo cp systemd/pdt_guard.service /etc/systemd/system/ 2>/dev/null || true
        sudo cp systemd/ssr_guard.service /etc/systemd/system/ 2>/dev/null || true
        sudo cp systemd/halts_luld.service /etc/systemd/system/ 2>/dev/null || true
        sudo cp systemd/corp_actions.service /etc/systemd/system/ 2>/dev/null || true
        
        sudo systemctl daemon-reload 2>/dev/null || true
        sudo systemctl enable market_hours_guard.service pdt_guard.service ssr_guard.service halts_luld.service corp_actions.service 2>/dev/null || true
        echo "✅ Systemd services configured"
    else
        echo "⚠️ No sudo access - services will run manually"
    fi
else
    echo "⚠️ systemctl not available - services will run manually"
fi

# Preflight check
echo "🚀 Running preflight check..."
python3 scripts/preflight_supercheck.py

# Test paper trading session (short run)
echo "📊 Testing paper trading session..."
if [ -f "run_stocks_session.py" ]; then
    python3 run_stocks_session.py --paper --symbols AAPL,MSFT,NVDA,SPY --minutes 5 || echo "⚠️ Paper session test failed"
else
    echo "⚠️ run_stocks_session.py not found - skipping paper test"
fi

# Check market status before enabling
echo "🕐 Checking market status..."
python3 -c "
import sys
sys.path.append('.')
from src.layers.layer5_risk.market_hours_guard import create_market_hours_guard
guard = create_market_hours_guard()
status = guard.get_market_status()
print(f'Market Status: {status}')
if status['should_block']:
    print('⚠️ Market is closed - system will remain in halt mode until market opens')
    exit(0)
else:
    print('✅ Market is open - proceeding to auto mode')
"

# Check result and conditionally enable auto mode
if [ $? -eq 0 ]; then
    echo "✅ Enabling auto trading mode..."
    redis-cli set mode auto
else
    echo "🛑 Keeping halt mode due to market hours"
fi

echo ""
echo "✅ Equities trading setup complete!"
echo "📊 System Status:"
echo "   Mode: $(redis-cli get mode)"
echo "   Capital Stage: $(redis-cli get risk:capital_stage)%"
echo "   RL Shadow: $(redis-cli hget features:flags EXEC_RL_SHADOW)"
echo "   RL Live: $(redis-cli hget features:flags EXEC_RL_LIVE)"
echo "   Symbols: $(redis-cli smembers symbols:stocks | tr '\n' ' ')"
echo ""
echo "🎯 Next Steps:"
echo "   1. Monitor A/B tests for RL promotion"
echo "   2. Review TCA reports for equities performance"
echo "   3. Check Grafana dashboards for risk metrics"