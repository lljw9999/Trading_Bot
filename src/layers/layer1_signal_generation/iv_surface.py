from deribit_api import RestClient, DeribitError
import redis, time, json, os

client = RestClient()
r = redis.Redis()


def pull_iv():
    for sym in ("BTC", "ETH"):
        atm = client.getsummary(f"{sym}-PERPETUAL")
        iv = atm["stats"]["volatility"]
        slope = iv - client.getsummary(f"{sym}-25DEC24").get("stats", {}).get(
            "volatility", iv
        )
        r.hset("iv_surface", mapping={f"{sym}:atm": iv, f"{sym}:slope": slope})


if __name__ == "__main__":
    while True:
        try:
            pull_iv()
        except DeribitError as e:
            print(e)
        time.sleep(60)
