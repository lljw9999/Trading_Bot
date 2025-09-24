import json, datetime, os, mlflow

meta = {
    "checkpoint": os.environ["NEW_HASH"],
    "date": str(datetime.datetime.utcnow()),
    "sharpe_1h": os.environ["NEW_SHARPE"],
    "max_dd": os.environ["NEW_DD"],
    "entropy_mean": os.environ["ENT_MEAN"],
}
open(f"model_cards/{meta['checkpoint']}.json", "w").write(json.dumps(meta, indent=2))
mlflow.log_artifact(f"model_cards/{meta['checkpoint']}.json")
