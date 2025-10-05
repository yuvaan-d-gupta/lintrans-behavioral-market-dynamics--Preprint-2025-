import pandas as pd
from pathlib import Path
import re

# --- Paste your actual session logs and outputs here ---
raw_terminal_log = '''
n_lags=1, ridge=0.0, m_horizon=1 | R^2=0.0053, Corr=0.0738, MSE=4.947255e-04
n_lags=1, ridge=0.0, m_horizon=2 | R^2=0.0054, Corr=0.0739, MSE=4.947790e-04
n_lags=1, ridge=0.0, m_horizon=4 | R^2=0.0051, Corr=0.0716, MSE=4.937041e-04

=== Distributed‐Lag: AAII (K=12) ===
Lag |   γ_k    | p-value
  0 |  0.008715 |  0.2976
  1 | -0.009719 |  0.3340
  2 |  0.013730 |  0.1723
  3 | -0.017695 |  0.0794
  4 |  0.003748 |  0.7097
  5 | -0.011814 |  0.2413
  6 |  0.005667 |  0.5752
  7 |  0.011803 |  0.2439
  8 |  0.005862 |  0.5632
  9 | -0.004592 |  0.6498
 10 | -0.010155 |  0.3155
 11 |  0.000962 |  0.9094
Cumulative Σγ = -0.003488, MSE = 4.923527e-04, R² = 0.0167, Corr = 0.1291

=== Distributed‐Lag: Google (K=12) ===
Lag |   γ_k    | p-value
  0 |  0.000533 |  0.0829
  1 | -0.000165 |  0.6816
  2 |  0.000551 |  0.1773
  3 | -0.000650 |  0.1143
  4 | -0.000180 |  0.6675
  5 |  0.000511 |  0.2241
  6 | -0.000620 |  0.1397
  7 |  0.000387 |  0.3567
  8 |  0.000492 |  0.2400
  9 | -0.000034 |  0.9360
 10 |  0.000040 |  0.9240
 11 | -0.000496 |  0.1255
Cumulative Σγ = 0.000369, MSE = 4.865405e-04, R² = 0.0283, Corr = 0.1681

=== Distributed‐Lag: UMich (K=12) ===
Lag |   γ_k    | p-value
  0 | -0.000881 |  0.2361
  1 |  0.004774 |  0.0000
  2 | -0.003431 |  0.0011
  3 | -0.001655 |  0.1156
  4 |  0.001211 |  0.2496
  5 | -0.001081 |  0.3045
  6 |  0.002382 |  0.0238
  7 | -0.001845 |  0.0773
  8 |  0.000444 |  0.6673
  9 | -0.000626 |  0.5448
 10 |  0.001386 |  0.1801
 11 | -0.000695 |  0.3421
Cumulative Σγ = -0.000015, MSE = 4.733201e-04, R² = 0.0547, Corr = 0.2338

Indicator: AAII
 Lag  0: corr = 0.0142
 Lag  1: corr = -0.0168
 Lag  2: corr = -0.0050
 Lag  3: corr = -0.0484
 Lag  4: corr = -0.0289
 Lag  5: corr = -0.0349
 Lag  6: corr = 0.0050
 Lag  7: corr = 0.0270
 Lag  8: corr = 0.0125
 Lag  9: corr = -0.0259
 Lag 10: corr = -0.0425
 Lag 11: corr = -0.0294
 Lag 12: corr = -0.0233
 Lag 13: corr = 0.0009
 Lag 14: corr = -0.0079
 Lag 15: corr = -0.0140
 Lag 16: corr = -0.0185
 Lag 17: corr = -0.0232
 Lag 18: corr = -0.0072
 Lag 19: corr = 0.0020
 Lag 20: corr = 0.0204
 Lag 21: corr = 0.0371
 Lag 22: corr = 0.0404
 Lag 23: corr = 0.0125
 Lag 24: corr = -0.0197



 === Final Model (Fitted on Full Sample with Best Hyperparameters) ===
Selected params -> n_lags=1, m_horizon=1, ridge=1.0       
In-sample MSE: 4.986116e-04, R^2: -0.0025, Corr: 0.0872   
Hit rate: 0.5482
Strategy final return: 1.1378, Sharpe: 0.4266
Coefficients and p-values:
  const: coef=-0.009003, p=0.2049
  x0: coef=-0.000475, p=0.9286
  x1: coef=0.000396, p=0.0237
  x2: coef=0.000028, p=0.6637
python : Traceback (most recent call last):
At line:1 char:1
+ python -u scripts/estimate_multivariate_mapping.py      
--grid_search 2>&1 ...




ModuleNotFoundError: No module named 'statsmodels'
ValueError: name 'create_lagged_matrix' is not defined
'''

# --- Parsing functions ---
def parse_grid_search(log):
    rows = []
    for line in log.splitlines():
        m = re.match(r"n_lags=(\d+), ridge=([\deE.-]+), m_horizon=(\d+) \| R\^2=([\d.-]+), Corr=([\d.-]+), MSE=([\deE.-]+)", line)
        if m:
            rows.append({
                'n_lags': int(m.group(1)),
                'ridge': float(m.group(2)),
                'm_horizon': int(m.group(3)),
                'R2': float(m.group(4)),
                'Corr': float(m.group(5)),
                'MSE': float(m.group(6)),
            })
    return pd.DataFrame(rows)

def parse_distributed_lag(log, indicator):
    pattern = rf"=== Distributed‐Lag: {indicator} \(K=\d+\) ===(.+?)Cumulative Σγ =(.+?)\n"  # non-greedy
    m = re.search(pattern, log, re.DOTALL)
    if not m:
        return pd.DataFrame(), {}
    table = m.group(1)
    summary = m.group(2)
    rows = []
    for line in table.splitlines():
        m2 = re.match(r"\s*(\d+) \|\s*([\d.-eE]+) \|\s*([\d.]+)", line)
        if m2:
            rows.append({'Lag': int(m2.group(1)), 'gamma_k': float(m2.group(2)), 'p_value': float(m2.group(3))})
    # Parse summary
    summary_dict = {}
    m3 = re.search(r"Cumulative Σγ = ([\d.-eE]+), MSE = ([\d.-eE]+), R² = ([\d.-eE]+), Corr = ([\d.-eE]+)", summary)
    if m3:
        summary_dict = {'Cumulative_gamma': float(m3.group(1)), 'MSE': float(m3.group(2)), 'R2': float(m3.group(3)), 'Corr': float(m3.group(4))}
    return pd.DataFrame(rows), summary_dict

def parse_cross_corr(log, indicator):
    pattern = rf"Indicator: {indicator}(.+?)(\n\n|$)"
    m = re.search(pattern, log, re.DOTALL)
    if not m:
        return pd.DataFrame()
    table = m.group(1)
    rows = []
    for line in table.splitlines():
        m2 = re.match(r"\s*Lag\s*(\d+): corr = ([\d.-]+)", line)
        if m2:
            rows.append({'Lag': int(m2.group(1)), 'Corr': float(m2.group(2))})
    return pd.DataFrame(rows)

def parse_errors(log):
    return pd.DataFrame({'Error': [line for line in log.splitlines() if line.strip()]})

# --- Parse and save ---
output_path = Path('results/tables/session_outputs.xlsx')
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Raw log
    pd.DataFrame({'Log': raw_terminal_log.splitlines()}).to_excel(writer, sheet_name='RawLog', index=False)
    # Grid search
    parse_grid_search(raw_terminal_log).to_excel(writer, sheet_name='GridSearch', index=False)
    # Distributed lag (one sheet per indicator)
    for ind in ['AAII', 'Google', 'UMich']:
        df, summary = parse_distributed_lag(raw_terminal_log, ind)
        if not df.empty:
            df.to_excel(writer, sheet_name=f'DistLag_{ind}', index=False)
            if summary:
                pd.DataFrame([summary]).to_excel(writer, sheet_name=f'DistLag_{ind}_Summary', index=False)
    # Cross-correlation (one sheet per indicator)
    for ind in ['AAII', 'Google', 'UMich']:
        df = parse_cross_corr(raw_terminal_log, ind)
        if not df.empty:
            df.to_excel(writer, sheet_name=f'CrossCorr_{ind}', index=False)
    # Errors
    parse_errors(raw_terminal_log).to_excel(writer, sheet_name='Errors', index=False)

print(f"Session outputs saved to {output_path}") 