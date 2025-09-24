import pandas as pd, os, pytz, sys, glob

# locate NVDA file inside the bundle
path = glob.glob("tmp/minute_data/NVDA_*.csv")[0]
print(f"📂  reading {path}")
df = pd.read_csv(path, parse_dates=["datetime"])
# keep only 2025-07-02 10:00–15:00 ET
et = pytz.timezone("US/Eastern")
mask = (df["datetime"] >= "2025-07-02 10:00:00") & (
    df["datetime"] <= "2025-07-02 15:00:00"
)
cut = df.loc[mask].copy()
if cut.empty:
    sys.exit("❌  slice returned 0 rows – check data bundle.")
# rename / reorder columns to match your schema
cut = cut.rename(
    columns={
        "datetime": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
)
# convert to UTC ISO-8601
cut["timestamp"] = (
    pd.to_datetime(cut["timestamp"]).dt.tz_localize(et).dt.tz_convert("UTC")
)
out = "data/stocks/2025-07-02/NVDA.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
cut.to_csv(out, index=False)
print(f"✅  saved {len(cut):,} rows → {out}")
