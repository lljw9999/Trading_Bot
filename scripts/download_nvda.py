import yfinance as yf
import os
import pandas as pd
import datetime as dt

sym   = "NVDA"
date  = "2025-07-02"
start = dt.datetime.strptime(date+" 09:30", "%Y-%m-%d %H:%M")
end   = dt.datetime.strptime(date+" 16:00", "%Y-%m-%d %H:%M")

df = yf.download(sym, start=start, end=end, interval="1m", progress=False)
assert not df.empty, "Download failed – check network/VPN"

df.reset_index(inplace=True)
df.rename(columns={
    "Datetime": "timestamp",
    "Open":     "open",
    "High":     "high",
    "Low":      "low",
    "Close":    "close",
    "Volume":   "volume"}, inplace=True)

dest = f"data/stocks/{date}/"
os.makedirs(dest, exist_ok=True)
df.to_csv(dest + f"{sym}.csv", index=False)
print(f"✅  Saved → {dest}{sym}.csv  ({len(df)} rows)")