#!/usr/bin/env python3
"""
Aggressive Autopilot - Bypasses gates for more active trading
Uses your proven live trading script directly
"""
import os, subprocess, time, sys, shlex
import random


def run(cmd, env=None, bg=False, timeout=300):
    try:
        p = subprocess.Popen(
            shlex.split(cmd),
            env=env or os.environ,
            stdout=subprocess.PIPE if not bg else None,
            stderr=subprocess.PIPE if not bg else None,
            text=True,
        )
        if bg:
            return p

        stdout, stderr = p.communicate(timeout=timeout)
        if stdout:
            print(stdout)
        if stderr and p.returncode != 0:
            print(f"Error: {stderr}")
        return p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        print(f"Command timed out: {cmd}")
        return 1
    except Exception as e:
        print(f"Command failed: {cmd} - {e}")
        return 1


def check_balance(env):
    """Check USDT balance"""
    try:
        result = subprocess.run(
            [
                "python",
                "-c",
                "from scripts.live_pnl_tracker import LivePnLTracker; "
                "tracker = LivePnLTracker(); "
                "balances = tracker.get_current_balances(); "
                "print(float(balances.get('USDT', {}).get('total', 0)))",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            balance = float(result.stdout.strip())
            return balance
        else:
            print(f"Balance check failed: {result.stderr}")
            return 0
    except Exception as e:
        print(f"Balance check error: {e}")
        return 0


def execute_trade(env, balance):
    """Execute a trade based on current balance"""
    if balance < 5:
        print(f"💰 Insufficient balance for trading: ${balance:.2f}")
        return False

    # Determine allocation based on balance
    if balance > 50:
        alloc = "0.6,0.4"  # 60% BTC, 40% ETH for larger amounts
        timeout = 180
    elif balance > 20:
        alloc = "0.5,0.5"  # 50/50 split for medium amounts
        timeout = 120
    else:
        alloc = "1.0,0.0"  # All BTC for small amounts
        timeout = 90

    print(f"💰 Trading ${balance:.2f} USDT with allocation {alloc}")

    cmd = f"python binance_live_test_and_trade.py --symbols BTCUSDT,ETHUSDT --alloc {alloc} --canary 0 --timeout {timeout}"

    try:
        rc = run(cmd, env=env, timeout=timeout + 30)
        if rc == 0:
            print("✅ Trade executed successfully")
            return True
        else:
            print(f"❌ Trade failed with return code {rc}")
            return False
    except Exception as e:
        print(f"❌ Trade execution error: {e}")
        return False


def main():
    print("🚀 Starting Aggressive Autopilot Trading")
    print("💡 This bypasses gates and trades more actively")
    print("=" * 50)

    # Set up environment
    env = os.environ.copy()
    env["GO_LIVE"] = "1"

    # Verify API keys
    if not env.get("BINANCE_TRADING_API_KEY") or not env.get(
        "BINANCE_TRADING_SECRET_KEY"
    ):
        print("❌ Missing Binance API credentials")
        print("Please set BINANCE_TRADING_API_KEY and BINANCE_TRADING_SECRET_KEY")
        sys.exit(1)

    print("🔑 API credentials found")

    # Trading loop
    trade_count = 0
    last_balance = 0

    while True:
        try:
            current_time = time.strftime("%H:%M:%S")
            print(f"\n⏰ {current_time} - Trading cycle #{trade_count + 1}")

            # Check balance
            balance = check_balance(env)
            print(f"💰 Current USDT balance: ${balance:.2f}")

            # Only trade if balance changed significantly or it's been a while
            balance_change = abs(balance - last_balance)
            should_trade = balance > 10 and (  # Minimum viable balance
                balance_change > 5 or trade_count % 4 == 0
            )  # Balance changed or periodic trade

            if should_trade:
                print("📈 Conditions met - executing trade...")
                success = execute_trade(env, balance)

                if success:
                    trade_count += 1
                    last_balance = balance
                    print(f"✅ Trade #{trade_count} completed")
                else:
                    print("❌ Trade failed - will retry next cycle")
            else:
                print("⏸️  No trading needed this cycle")

            # Variable sleep time based on market conditions
            sleep_time = random.randint(180, 420)  # 3-7 minutes
            print(f"😴 Sleeping {sleep_time}s before next cycle...")
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n🛑 Aggressive autopilot stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in trading loop: {e}")
            print("😴 Sleeping 120s before retry...")
            time.sleep(120)

    print(f"\n📊 Final stats: {trade_count} trades executed")


if __name__ == "__main__":
    main()
