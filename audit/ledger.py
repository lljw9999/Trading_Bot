import ipfshttpclient, hashlib, json, time, redis

cli = ipfshttpclient.connect()
R = redis.Redis()


def log_order(order_dict):
    block = {
        "ts": time.time(),
        "model_hash": R.get("model:hash").decode(),
        "order": order_dict,
    }
    j = json.dumps(block, sort_keys=True).encode()
    cid = cli.add_bytes(j)
    R.xadd("audit:orders", {"cid": cid}, maxlen=10000)
    return cid
