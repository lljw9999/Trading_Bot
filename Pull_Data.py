import yfinance as yf, pandas as pd, os, pytz, sys
sym, date = 'NVDA', '2025-07-02'
start_et, end_et = f'{date} 10:00', f'{date} 15:01'
path = f'data/stocks/{date}/{sym}.csv'
os.makedirs(os.path.dirname(path), exist_ok=True)

print(f"⏬  downloading {sym} 1-min bars {start_et}→{end_et} …")
df = yf.download(
        sym,
        start=start_et, end=end_et,
        interval='1m', auto_adjust=False,
        progress=False, threads=False)

if df.empty:
    sys.exit("⚠️  Yahoo returned no rows – check network or symbol/date.")

# convert naive index → UTC ISO8601 (Grafana friendly)
df.index = (df.index
            .tz_localize('US/Eastern')
            .tz_convert('UTC'))
df.to_csv(path, index_label='timestamp')
print(f"✅  saved {len(df):,} rows → {path}")