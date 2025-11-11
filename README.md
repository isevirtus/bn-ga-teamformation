# BN-GA Team Formation

This repository contains the codebase for the **BN-GA Team Formation** project, which integrates a Bayesian Network (BN) calibrated with expert knowledge and a Genetic Algorithm (GA) to optimize software team formation, considering both technical coverage and collaborative compatibility.

## 📁 Project Structure

```
STFP/
├── Algorithms/
│   ├── BN/
│   │   ├── bnetwork.py                # BN evaluator and CPT generation
│   │   ├── team_fit_bn.py             # BN calibration and fitting
│   │   └── utils.py                   # BN utilities
│   ├── GA/
│   │   ├── engine.py                  # Genetic Algorithm main logic
│   │   ├── run_ga_10seeds.py          # Run GA for 10 seeds
│   │   ├── run_random_10seeds.py      # Random baseline runner
│   │   ├── random_search_baseline.py  # Random search baseline implementation
│   │   ├── benchmark_bn_runtime.py    # Benchmark BN runtime
│   │   └── scenario_consistency.py    # Analyzes scenario coverage
│   └── Reports/
│       └── ga_10seeds_results.csv     # Results from GA across seeds
│
├── Data/
│   ├── Dev_DB.json                    # Developer database
│   └── Graph_DB.json                  # Collaboration graph
│
├── Feature_Extraction/
│   ├── Dimension_Scoring/
│   │   ├── dimension_scoring.py       # Score AT and AC
│   │   ├── linear_regression_calibrator.py # Linear model for expert calibration
│   │   └── pesos_calibrados.json      # Calibrated weights for FS estimation
│   └── PC_Transformer/
│       ├── filter_devs_by_graph.py    # Filters developers by connectivity
│       └── pc_transformer.py          # PC calculation based on FS regression
│
├── Pipeline/
│   └── evaluate_teams.py              # Main entry to evaluate teams using BN
│
└── Validation/
    ├── run_ga.py                      # Run GA for final experiments
    ├── gradient_checking_rb.py        # Gradient behavior validation
    ├── grafico_gradient.py            # Plotting script for gradients
    ├── *.csv, *.png                   # Results and visualizations
```

## 🚀 How to Run

1. Prepare the datasets:
   - Place `Dev_DB.json` and `Graph_DB.json` in the `Data/` folder.

2. Run GA with calibrated BN:
```bash
python Validation/run_ga.py
```

3. Run baseline:
```bash
python Algorithms/GA/random_search_baseline.py
```

## 📊 Outputs

- Fitness evaluations (`AE`) per team.
- Scenario coverage report.
- Sensitivity plots and gradient checks.

## 📄 License

This project is licensed under the MIT License.

## 🙋‍♂️ Author

Felipe Oliveira Miranda Cunha – PPGCC/UFCG
