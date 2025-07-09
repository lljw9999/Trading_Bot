#!/usr/bin/env python3
"""
Pull 1-minute A-share K-line data via iTick free API
  docs: https://itick-cn.readme.io/reference/get_stock-kline

Example:
  python scripts/get_itick_minute.py 600519.SH 2025-07-03 09:30 10:30
"""

import os, sys, time, requests, csv, pathlib, datetime as dt
from tenacity import retry, wait_fixed, stop_after_attempt
from tqdm import tqdm


TOKEN = os.getenv("ITICK_TOKEN")
if not TOKEN:
    sys.exit("❌  ITICK_TOKEN not set in environment")

BASE  = "https://api.itick.cn/api/v1/stock/kline"

SYMBOL, DATE, T0, T1 = sys.argv[1:5]               # 600519.SH  2025-07-03  09:30  10:30
start_dt  = dt.datetime.strptime(f"{DATE} {T0}", "%Y-%m-%d %H:%M")
end_dt    = dt.datetime.strptime(f"{DATE} {T1}", "%Y-%m-%d %H:%M")

# iTick returns max 1000 rows; we only need 60 for 1-hour slice
params_com = {
    "token" : TOKEN,
    "symbol": SYMBOL,
    "period": "1m",
    "adjust": "qfq"      # forward-adj close; change if you need
}

@retry(wait=wait_fixed(15), stop=stop_after_attempt(3))
def call_itick(**params):
    r = requests.get(BASE, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

rows = []
cur  = start_dt
pbar = tqdm(total=int((end_dt - start_dt).total_seconds() / 60), desc="pulling")
while cur < end_dt:
    # iTick needs timestamps in seconds
    params = dict(params_com,
                  start=int(cur.timestamp()),
                  end  =int((cur + dt.timedelta(minutes=1)).timestamp()))
    try:
        data = call_itick(**params)
        if data["code"] != 0:
            raise RuntimeError(data.get("msg", "api error"))
        if data["data"]:
            k = data["data"][0]   # one row
            rows.append([
                dt.datetime.utcfromtimestamp(k["time"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
                k["open"], k["high"], k["low"], k["close"], k["volume"]
            ])
    except Exception as e:
        print("⚠️ ", e, "— retry or skip")
    cur += dt.timedelta(minutes=1)
    pbar.update(1)
    # throttle: 5 req/min → pause after each request
    time.sleep(12.5)
pbar.close()

# ----- save -----
outdir = pathlib.Path(f"data/astocks/{DATE}")
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir / f"{SYMBOL}.csv"
with open(outfile, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamp","open","high","low","close","volume"])
    w.writerows(rows)
print(f"✅  wrote {len(rows)} rows → {outfile}") 