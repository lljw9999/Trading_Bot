#!/usr/bin/env bash
# 5-hour NVDA replay at 60× speed (~5 min wall-time)

export DATA_DIR=data/stocks/2025-07-02
export SYMBOLS=NVDA
export REPLAY_SPEED=60                  # 60× faster than real-time
export REPLAY_START=2025-07-02T10:00:00
export REPLAY_END=2025-07-02T15:00:00
export BANKROLL=100000                  # starting equity $

# 1  Start native services if not running
brew services start redis prometheus grafana

# 2  Start the historical tick publisher
python -m src.layers.layer0_data_ingestion.replay_publisher \
       --symbol $SYMBOLS \
       --start  $REPLAY_START \
       --end    $REPLAY_END \
       --speed  $REPLAY_SPEED &

# 3  Start the full 6-layer stock session (paper-trading mode)
python run_stocks_session.py --symbols $SYMBOLS --paper