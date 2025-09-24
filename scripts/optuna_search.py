import optuna, subprocess, json, os, mlflow

SEARCH_SPACE = {
    "actor_lr": (1e-5, 5e-4, "log"),
    "critic_lr": (1e-5, 5e-4, "log"),
    "α": (0.05, 0.5, "log"),
    "net_dim": (128, 512, "int"),
}


def objective(trial):
    cfg = {
        k: trial.suggest_float(k, *v) if v[2] == "log" else trial.suggest_int(k, *v[:2])
        for k, v in SEARCH_SPACE.items()
    }
    cfg_str = json.dumps(cfg)
    res = subprocess.run(
        ["python", "train_short.py", cfg_str],
        capture_output=True,
        text=True,
        timeout=900,
    )
    sharpe = float(res.stdout.strip().split()[-1])
    mlflow.log_params(cfg)
    mlflow.log_metric("sharpe", sharpe)
    return -sharpe  # minimise


optuna.create_study(
    study_name="sacdif_nightly", storage="sqlite:///optuna.db", direction="minimize"
).optimize(objective, n_trials=30)
