# Legacy Scripts and Configs

The following operational scripts/configs are no longer part of the active production
pipeline. Rather than modernizing every utility to the new datetime/async standards,
we are explicitly classifying them as legacy so that future cleanup can archive or
retire them without blocking current work.

| Path | Notes |
| --- | --- |
| scripts/morning_greenlight.py | One-off pre-market checklist; superseded by runbooks and dashboards |
| scripts/weekly_retro.py | Generates narrative status reports; replaced by automated JIRA/GitHub workflows |
| scripts/daily_pnl_close.py | Legacy accounting roll-up; superseded by `accounting/` services |
| scripts/exec_grid_sweep.py | Experimental grid-search harness retained for reference |
| scripts/capture_state.py | Snapshot utility used in earlier incident drills, now replaced by `ops_bot/` tooling |
| scripts/exp_decider.py | Old experiment scheduler replaced by Airflow DAGs |
| scripts/duty_cycler*.py | Prototype duty-cycle helpers now covered by SLO platform |
| scripts/green_window_ramp.py | Predecessor to capital ramp guard; left for documentation purposes |
| scripts/onnx_quantize.py | Historical model conversion tool; modern workflow uses `src/optimization/onnx_policy_server.py` |
| scripts/restore_from_s3.py | Manual DR helper replaced by infra-as-code runbooks |

Guidance:

- Leave these scripts untouched unless a future project revives them; the modern stack
  relies on the corresponding services/RUNBOOK entries instead.
- When executing repository-wide upgrades (e.g., datetime or FastAPI APIs), exclude
  these paths unless there is an explicit requirement to resurrect them.
- If long-term archival is desired, move the files into `archive/` with their
  documentation, but no runtime components currently import them.

This decision keeps modernization focused on the actively used RL/ensemble/dashboard
code paths while clearly documenting which legacy utilities are exempt.
