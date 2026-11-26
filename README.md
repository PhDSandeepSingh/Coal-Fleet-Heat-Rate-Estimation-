# Coal Fleet Heat-Rate Estimation & Retrofit Prioritisation

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat&logo=python&logoColor=white)](https://python.org)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)

An empirical validation and uncertainty-propagation benchmark for unit-level coal power plant Station Heat Rate (SHR) estimation and retrofit prioritisation across India's coal fleet (540 labelled CEEW units across 178 plants; 787 operating GEM fleet units).

## Key Experimental Findings

| Audit Dimension | Verified Finding |
| :--- | :--- |
| **Plant-Grouped vs. Random CV Leakage** | Random unit folds understate flexible-learner MAE by **43%–91%**; holding out whole plants places linear **Ridge (35.2 kcal/kWh)** ahead of GBM (39.3), RF (40.5), KNN (42.8), and Two-Model (39.9) |
| **Nonlinear Inverse-Load Check ($1/\text{PLF}$)** | Nonlinear Ridge ($1/\text{PLF}$ clipped at 0.15 + boiler interactions) achieves **35.6 ± 0.7 kcal/kWh** under identical $10 \times 5$ plant-grouped folds, closely matching linear Ridge |
| **Interval Calibration** | Random-fold calibration under-covers new plants (**78%–84%** vs. 90% nominal); plant-grouped calibration restores **90%–92%** coverage |
| **Coal-Price Unit Audit** | Source `coal_price` values (mean 2.36) are in **USD/MMBtu** despite an `INR/Mcal` header; converting at 83 INR/USD (`0.78 INR/Mcal` GEM mean vs. `0.79` combined / `0.77` subcritical CEEW) reduces stylised excess fuel cost from **INR 41,435** to **INR 13,638 crore/yr** |
| **Hybrid Observed/Predicted Matching** | 1-to-1 approximate CEEW–GEM match ($\pm 10$ MW, cleaned name tokens) matches **391 / 540 CEEW units (72.4%)**, covering **49.7%** of the 787-unit GEM fleet; placing matched units on a consistent $\text{PLF}=0.5$ basis gives **INR 15,531 crore/yr** (90% band **15,136–15,888**; +248 crore residual spread, +1,645 crore $s^{(10)}_b$ shift) |
| **Top-80 Ranking Stability** | Under 10-repeat unique-unit (`>= 30` units/cell) stratified residuals with $\varepsilon = 0$ on matched units, **62 / 80** point-ranked units achieve $\ge 90\%$ inclusion (upper bound; 52/80 point-ranked and 45/62 robust are matched), mean Jaccard **0.84** (5th pct **0.76**), and **141** distinct units ever enter the top 80 |
| **50% PLF Default Bias** | Relative to capacity-weighted utilisation (0.626 PLF), the 50% default raises fleet SHR by **+41 kcal/kWh** (linear Ridge) and **+24.3 kcal/kWh** (nonlinear $1/\text{PLF}$) |

## Repository Structure

```text
.
├── Paper2_CoalHeatRate.ipynb          # Core nested CV & interval-calibration notebook
├── Paper2_CoalHeatRate_EXECUTED.ipynb # Executed baseline notebook
├── make_notebook.py                   # Notebook builder
├── run_round2_p2.py                   # Nonlinear PLF & initial stratified uncertainty script
├── run_round3_p2.py                   # Final unique-unit fallback, ±10 MW match, PLF-0.5 adjustment & figures
├── gen_tables.py                      # Programmatic summary & table generator
├── data/                              # CEEW, GEM, coal price, and 5-zone mapping CSVs
└── results/
    ├── table_accuracy.csv             # Random vs. plant-grouped 10x5 nested CV metrics
    ├── table_coverage.csv             # Pooled & per-boiler Mondrian 90% interval coverage
    ├── table_plf_sensitivity.csv      # Capacity-weighted SHR across PLF 0.30–0.90
    ├── matched_ceew_gem_pairs.csv     # All 391 verified 1-to-1 CEEW–GEM unit matches
    ├── fleet_predictions_r2.csv       # Unit-level fleet predictions & top-80 inclusion probabilities
    ├── round2_p2_stats.json           # Complete verified summary metrics JSON
    ├── data_sha256.json               # Input file checksums
    ├── runs/                          # Cached 10x5 per-fold out-of-sample predictions
    └── fig_*.png / fig_*.pdf          # CV gap, interval coverage, and fleet sensitivity plots
```

## Quick Start & Reproducibility

```bash
# 1. Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib

# 2. Run final nonlinear PLF, unique-unit stratified uncertainty, and CEEW-GEM matching
python run_round3_p2.py

# 3. Regenerate summary tables and metrics from saved results
python gen_tables.py
```

## Data Attribution

Input tables derive from the open Zenodo v2 release (DOI: `10.5281/zenodo.13921645`, CC BY 4.0), integrating CEEW plant heat-rate estimates (Sep 2017–Feb 2020), Global Energy Monitor (GEM) Global Coal Plant Tracker attributes, and WRI Aqueduct water-stress scores. Under CC BY 4.0, note that the `coal_price` column is modified via unit conversion from USD/MMBtu to INR/Mcal at INR 83/USD.

## Topics / Tags

`coal-power` · `energy-analytics` · `grouped-cross-validation` · `uncertainty-quantification` · `conformal-prediction` · `heat-rate` · `decarbonization` · `india-power-grid`

## License

Code and analysis scripts are released under the **MIT License**. Underlying public energy datasets retain their original **CC BY 4.0** terms.
