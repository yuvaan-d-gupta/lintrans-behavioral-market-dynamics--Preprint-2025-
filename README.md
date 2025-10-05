
This repository contains data processing, feature engineering, model estimation, diagnostics, and visualization scripts used in the research analyzing how sentiment/anxiety indicators map to market returns. The workflow merges multiple weekly datasets (AAII sentiment, Google anxiety terms, UMich sentiment, and SPY returns), engineers transformations and lags, and estimates linear/multivariate relationships with cross‑validation. Figures and tables are written to `results/`.

## Repository Structure
- `raw_data/`: Input CSVs (already processed or fetched by scripts)
- `scripts/`: All Python scripts for data, modeling, and plots
- `results/`
  - `figures/`: Generated plots (PNG/PDF)
  - `tables/`: Generated tables (CSV/TXT/XLSX)

## Quickstart
1) Create and activate a virtual environment
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
```

2) Install dependencies
```bash
python -m pip install --upgrade pip setuptools
pip install -r requirements.txt
```

3) Run the end‑to‑end workflow (example)
- Process or confirm input datasets in `raw_data/`
- Run modeling and visualization scripts (see Script Catalog below)

## Data Inputs
Expected CSVs in `raw_data/`:
- `aaii_sentiment_processed.csv`: weekly AAII sentiment indicator(s)
- `google_anxiety_processed.csv`: weekly Google Trends composite anxiety index
- `umich_sentiment_weekly.csv`: weekly University of Michigan sentiment index
- `sca-table1-on-2025-Jun-16.csv`: source SCA table (if used in processing)
- `spy_weekly_returns.csv`: SPY weekly returns

Some scripts (e.g., `process_google_trends.py`) can help produce processed inputs.

## Script Catalog (selected)
Data processing
- `scripts/process_aaii.py`: cleans/prepares AAII data
- `scripts/process_google_trends.py`: fetches Google Trends terms, builds weekly composite Anxiety index
- `scripts/process_umich_sentiment.py`: prepares UMich sentiment to weekly frequency
- `scripts/01_load_and_merge.py`: loads all sources, merges into unified dataset
- `scripts/02_feature_engineering.py`: builds lags, transformations, and features

Modeling & analysis
- `scripts/estimate_aaii_mapping.py`: univariate/low‑dim mapping for AAII
- `scripts/enhanced_multivariate_mapping.py`: multivariate models (Ridge/Lasso/ElasticNet/RandomForest)
- `scripts/estimate_multivariate_mapping.py`: core multivariate mapping with CV
- `scripts/distributed_lag_analysis.py`: distributed‑lag regression diagnostics
- `scripts/var_irf_analysis.py`: VAR and impulse response analysis

Visualization & reporting
- `scripts/create_coefficient_heatmap.py`, `display_heatmap.py`, `simple_heatmap.py`, `show_heatmap.py`
- `scripts/create_scatter_plot.py`: multivariate fit scatter
- `scripts/create_cumulative_strategy_plot.py`, `show_cumulative_strategy.py`: strategy cumulatives
- `scripts/create_directional_hit_rate_plot.py`, `show_directional_hit_rate.py`
- `scripts/show_transformation_matrix.py`
- `scripts/diagram.py`: Sharpe simulation/visualization utilities
- `scripts/save_terminal_outputs_to_excel.py`: parse session logs to `results/tables/session_outputs.xlsx` (requires Excel writer)

Generated outputs
- `results/figures/`: heatmaps, scatter plots, strategy charts, etc.
- `results/tables/`: coefficients, CV summaries, in‑sample metrics, transformation matrices, session outputs

## Reproducibility
- Python version: 3.10+ recommended
- Install `requirements.txt` exactly
- Scripts use deterministic seeds where applicable; OS and library versions may affect exact figures

## Notable Dependencies
- pandas, numpy: data manipulation
- scikit‑learn: preprocessing, linear models, metrics, CV
- statsmodels: regression, VAR/TS analysis
- matplotlib, seaborn: visualization
- yfinance: SPY returns download (if needed)
- pytrends: Google Trends API wrapper (optional, only for fetching)
- openpyxl: Excel writer for `save_terminal_outputs_to_excel.py`

## Typical Commands (examples)
Generate heatmaps
```bash
python scripts/create_coefficient_heatmap.py
python scripts/display_heatmap.py
```

Estimate multivariate mapping
```bash
python scripts/estimate_multivariate_mapping.py --grid_search
```

Distributed‑lag analysis
```bash
python scripts/distributed_lag_analysis.py --indicator AAII
```

Sharpe simulation (diagram)
```bash
python scripts/diagram.py --target_sharpe 0.4266 --weeks 720 --fig results/figures/rolling_sharpe_sim.png
```

Save terminal outputs to Excel
```bash
python scripts/save_terminal_outputs_to_excel.py
```

## Data/Rate‑Limit Notes
- Google Trends (`pytrends`) is rate‑limited; the script includes delays and retries
- SPY data via `yfinance` may change due to vendor adjustments

## Citation
If you use this code or figures in your research, please cite this repository and reference the associated paper.

