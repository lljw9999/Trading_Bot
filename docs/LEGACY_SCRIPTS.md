# Legacy Scripts and Configs

The following operational scripts/configs are no longer part of the active production
pipeline. Rather than modernizing every utility to the new datetime/async standards,
we are explicitly classifying them as legacy so that future cleanup can archive or
retire them without blocking current work.

## Legacy Scripts

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

## Archived Test Scripts (September 2025)

The following test files have been moved to `archive/test_scripts/` to clean up the root directory:

| Original Path | Purpose | Archive Location |
| --- | --- | --- |
| test_charts_visible.py | Chart visibility testing | archive/test_scripts/ |
| test_coinbase_*.py | Coinbase API integration tests | archive/test_scripts/ |
| test_complete_dashboard.py | Comprehensive dashboard testing | archive/test_scripts/ |
| test_dashboard_final.py | Final dashboard implementation test | archive/test_scripts/ |
| test_enhanced_*.py | Enhanced feature testing | archive/test_scripts/ |
| test_gcn.py | Graph neural network testing | archive/test_scripts/ |
| test_tft.py | Temporal fusion transformer testing | archive/test_scripts/ |
| test_copula.py | Copula modeling testing | archive/test_scripts/ |
| test_stat_arb.py | Statistical arbitrage testing | archive/test_scripts/ |
| test_ensemble_*.py | Ensemble system testing | archive/test_scripts/ |
| test_*_chart.py | Various chart component tests | archive/test_scripts/ |
| test_news_integration.py | News sentiment integration testing | archive/test_scripts/ |

## Guidance

- Leave these scripts untouched unless a future project revives them; the modern stack
  relies on the corresponding services/RUNBOOK entries instead.
- When executing repository-wide upgrades (e.g., datetime or FastAPI APIs), exclude
  these paths unless there is an explicit requirement to resurrect them.
- If long-term archival is desired, move the files into `archive/` with their
  documentation, but no runtime components currently import them.
- **For active testing**, use the proper `tests/` directory with pytest structure
- **Archived test scripts** may not work with the current codebase and are preserved for historical reference only

This decision keeps modernization focused on the actively used RL/ensemble/dashboard
code paths while clearly documenting which legacy utilities are exempt.
