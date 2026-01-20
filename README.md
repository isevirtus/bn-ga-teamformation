# Replication & Supplementary Package (BN + GA for Software Team Formation)

This repository contains the **core implementation** (Bayesian Network evaluator + Genetic Algorithm + feature extraction) and the **exact scripts/data artifacts** used to generate the **supplementary results** reported in the paper:

- **RQ1 — Expert Ranking vs. BN Ranking** (Table-1 agreement data + auxiliary ranking files)
- **RQ2 — GA Runtime using BN fitness** (per-run logs + summary)
- **BN Sanity Check** (extreme/consistency scenarios)

> **Important note on data**: running the experiments requires `Data/Dev_DB.json` and `Data/Graph_DB.json`.  

---

## Repository Structure

```
STFP_NSE_ICSE/
├── Algorithms/
│   ├── BN/
│   │   ├── bnetwork.py                 # BN engine and inference wrapper
│   │   ├── team_fit_bn.py              # BN structure + CPT generation/calibration
│   │   ├── utils.py                    # BN utilities (states, aggregation, helpers)
│   │   └── repository.json            # (if present) samples/centroids for ranked nodes
│   └── GA/
│       ├── engine.py                   # GA core loop (selection/crossover/mutation/stop)
│                               
│
├── Feature_Extraction/
│   ├── Dimension_Scoring/
│   │   ├── dimension_scoring.py        # Technical coverage scoring (Domain/Ecosystem/Language)
│   │   
│   └── PC_Transformer/
│       ├── pc_transformer.py           # Pair Compatibility (PC) → AC mapping
│       
│
├── Pipeline/
│   ├── evaluate_teams.py               # Main evaluator: AT + AC → BN inference → AE
│   
│
├── Data/
│   ├── Dev_DB.json                     # Developer database (required to run)
│   └── Graph_DB.json                   # Collaboration graph (required to run)
│
├── rq1_expert_ranking/
│   ├── eval_team_per_project_bn_full.py         # RQ1: scores candidate teams with BN
│   ├── compute_rank_metrics.py                  # RQ1: builds agreement and ranking metrics
│   ├── team_per_project.csv                     # candidate teams per project (input/record)
│   ├── team_per_project_bn_scored_full_005.csv  # BN-scored teams (generated / provided)
│   ├── ranking_expert.csv                       # expert ranking (input)
│   ├── ranking_bn.csv                           # BN ranking (generated / provided)
│   ├── rank_expert_rank_bn.csv                  # Table-1 agreement data (generated / provided)
│   └── metrics_rank.csv                         # extra ranking metrics (optional)
│
├── rq2_runtime/
│   ├── rt_ga_bn_only_engine.py          # RQ2: measures GA runtime using BN fitness
│   ├── ga_bn_only_runs.csv              # per-run runtime logs (generated / provided)
│   └── ga_bn_only_summary.csv           # aggregated runtime summary (generated / provided)
│
├── bn_sanity_check/
│   ├── run_sanity_bn.py                 # BN sanity scenarios runner
│   ├── scenarios_sanity_bn.csv          # sanity scenarios inputs
│   ├── sanity_bn_results.csv            # sanity outputs (generated / provided)
│   ├── sanity_bn_checks.txt             # sanity check report (generated / provided)
│   └── out/                             # optional extra outputs
│
├── GA_Runtime.xlsx                      # Human-readable spreadsheet (optional viewer)
├── Sanity_check.xlsx                    # Human-readable spreadsheet (optional viewer)
├── Experiment_Ranking_Expert_BN.xlsx    # Human-readable spreadsheet (optional viewer)
├── __init__.py

```

---

## Quick Start

### 1) Environment
Recommended: **Python 3.10+** (Windows/Linux/macOS).

Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

Install dependencies.

If you already have a `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

If you do **not** have `requirements.txt`, install the common packages used by this project:

```bash
pip install numpy pandas scipy networkx pgmpy
```

> If you run into a missing-import error, install the missing package reported by Python.

---

## Reproducing the Paper Results

### RQ1 — Expert Ranking vs BN Ranking (Table-1 agreement)

This reproduces the data used to compare the **expert top-1 / bottom-1** choices against the BN ranking.

Run:

```bash
python rq1_expert_ranking/eval_team_per_project_bn_full.py
python rq1_expert_ranking/compute_rank_metrics.py
```

Main files:

- `rq1_expert_ranking/team_per_project.csv`  
  Candidate teams per project (the same 5 teams per project used in the workshop/paper).
- `rq1_expert_ranking/ranking_expert.csv`  
  Expert’s ranking (input).
- `rq1_expert_ranking/team_per_project_bn_scored_full_005.csv`  
  BN-scored teams (output).
- `rq1_expert_ranking/ranking_bn.csv`  
  BN ranking derived from BN scores (output).
- `rq1_expert_ranking/rank_expert_rank_bn.csv`  
  **Agreement dataset** used to build Table-1 (output).
- `rq1_expert_ranking/metrics_rank.csv`  
  Optional extra ranking metrics (not required by the paper table, but useful for analysis).

**What to cite as supplementary for RQ1**
- `rank_expert_rank_bn.csv` (Table-1 agreement evidence)
- `ranking_expert.csv` + `ranking_bn.csv` (rankings)
- `team_per_project_bn_scored_full_005.csv` (scores behind BN ranking)

---

### BN Sanity Check (consistency/extremes)

This reproduces the sanity check mentioned in the paper (e.g., all-VL vs all-VH and related consistency scenarios).

Run:

```bash
python bn_sanity_check/run_sanity_bn.py
```

Inputs/outputs:

- Input: `bn_sanity_check/scenarios_sanity_bn.csv`
- Outputs:
  - `bn_sanity_check/sanity_bn_results.csv`
  - `bn_sanity_check/sanity_bn_checks.txt`
  - optional extra files under `bn_sanity_check/out/`

**What to cite as supplementary for sanity check**
- `sanity_bn_results.csv` + `sanity_bn_checks.txt`

---

### RQ2 — GA Runtime using BN fitness

This reproduces the runtime experiment for GA runs using the BN-based fitness evaluation.

Run:

```bash
python rq2_runtime/rt_ga_bn_only_engine.py
```

Outputs:

- `rq2_runtime/ga_bn_only_runs.csv`  
  Per-run logs (run, seed, team size, pop size, generations, duration, fitness calls, ms/fitness, best fitness, best team, stop reason).
- `rq2_runtime/ga_bn_only_summary.csv`  
  Aggregated summary (mean/median/min/max/percentiles, etc.).

**What to cite as supplementary for RQ2**
- `ga_bn_only_runs.csv` (per-run evidence)
- `ga_bn_only_summary.csv` (aggregated numbers)

---

## Where the “spreadsheets” fit

The `.xlsx` files in the repository root are **human-readable viewers** of the same results:

- `GA_Runtime.xlsx`
- `Sanity_check.xlsx`
- `Experiment_Ranking_Expert_BN.xlsx`

---

## License

Add your chosen license file (e.g., MIT) as `LICENSE`.
