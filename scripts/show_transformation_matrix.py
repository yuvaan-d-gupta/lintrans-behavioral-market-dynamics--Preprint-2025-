

import pandas as pd
import numpy as np
from numpy.linalg import inv, pinv, LinAlgError
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_and_merge(aaii_path, google_path, umich_path, returns_path, tol_days=7):
    """Load and merge data as in the multivariate analysis."""
    df_aaii = pd.read_csv(aaii_path, parse_dates=['Date']).rename(columns={'Sentiment':'AAII'})
    df_google = pd.read_csv(google_path, parse_dates=['Date']).rename(columns={'Anxiety':'Google'})
    df_umich = pd.read_csv(umich_path, parse_dates=['Date']).rename(columns={'Sentiment':'UMich'})
    df_ret = pd.read_csv(returns_path, parse_dates=['Date']).rename(columns={'Return':'Return'})
    
    for df in (df_aaii, df_google, df_umich, df_ret):
        df.sort_values('Date', inplace=True)
    
    df = pd.merge_asof(df_aaii, df_google, on='Date', tolerance=pd.Timedelta(f'{tol_days}D'), direction='nearest')
    df = pd.merge_asof(df, df_umich, on='Date', tolerance=pd.Timedelta(f'{tol_days}D'), direction='nearest')
    df = pd.merge_asof(df, df_ret, on='Date', tolerance=pd.Timedelta(f'{tol_days}D'), direction='nearest')
    df.dropna(subset=['AAII','Google','UMich','Return'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def create_design_return(df, n_lags, m_horizon):
    """Create design matrix as in the multivariate analysis."""
    series_list = [df['AAII'].values, df['Google'].values, df['UMich'].values]
    returns = df['Return'].values
    T = len(df)
    T_prime = T - n_lags - m_horizon + 1
    if T_prime is None or T_prime <= 0:
        return np.empty((len(series_list) * n_lags, 0)), np.empty((m_horizon, 0)), 0

    # Build block design matrix with proper lag alignment
    B_blocks = []
    for series in series_list:
        lag_rows = []
        for i in range(n_lags):
            start_idx = (n_lags - 1 - i)
            lag_rows.append(series[start_idx : start_idx + T_prime])
        B_blocks.append(np.vstack(lag_rows))
    B_full = np.vstack(B_blocks)

    # Build targets for horizons 1..m_horizon
    R_rows = []
    for j in range(m_horizon):
        start_idx = n_lags + j
        R_rows.append(returns[start_idx : start_idx + T_prime])
    R = np.vstack(R_rows)
    return B_full, R, T_prime

def estimate_transformation(B, R, ridge_lambda):
    """Estimate transformation matrix as in the multivariate analysis."""
    BBt = B @ B.T
    if ridge_lambda > 0:
        BBt += ridge_lambda * np.eye(BBt.shape[0])
    try:
        inv_BBt = inv(BBt)
    except LinAlgError:
        inv_BBt = pinv(BBt)
    return R @ B.T @ inv_BBt

def main():
    """Load data and compute transformation matrix with exact parameters."""
    
    print("=== Transformation Matrix Calculator ===")
    print("Using parameters to show full transformation matrix:")
    print("  n_lags=4, ridge=1.0, m_horizon=12")
    print("  (This will show all horizons and more behavioral lags)")
    print()
    
    # Load data
    df = load_and_merge(
        "raw_data/aaii_sentiment_processed.csv",
        "raw_data/google_anxiety_processed.csv", 
        "raw_data/umich_sentiment_weekly.csv",
        "raw_data/spy_weekly_returns.csv"
    )
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Loaded data: {len(df)} observations from {df['Date'].min()} to {df['Date'].max()}")
    
    # Use parameters to show full transformation matrix (multiple horizons)
    n_lags_best = 4  # Use more lags to show more columns
    ridge_best = 1.0
    m_horizon_best = 12  # Use multiple horizons to show all rows
    
    # Create design and return matrices
    B_full, R_full, T_prime = create_design_return(df, n_lags_best, m_horizon_best)
    print(f"Design matrix B shape: {B_full.shape}")
    print(f"Return matrix R shape: {R_full.shape}")
    print(f"Effective sample size: {T_prime}")
    print()
    
    # Standardize features (as done in the multivariate analysis)
    scaler_full = StandardScaler(with_mean=True, with_std=True)
    B_full_s = scaler_full.fit_transform(B_full.T).T
    
    # Estimate transformation matrix
    M_final = estimate_transformation(B_full_s, R_full, ridge_best)
    
    # Create proper column and row names
    behavioral_series = ['AAII', 'Google', 'UMich']
    col_names = []
    for series_name in behavioral_series:
        for lag in range(n_lags_best):
            col_names.append(f'{series_name}_lag{lag}')
    
    row_names = [f'horizon_{h+1}' for h in range(m_horizon_best)]
    
    # Create DataFrame with transformation matrix
    M_df = pd.DataFrame(M_final, columns=col_names, index=row_names)
    
    # Display results with full precision
    print("=== Transformation Matrix M ===")
    print(f"Shape: {M_final.shape} (rows=horizons, columns=behavioral×lags)")
    print(f"Column names: {col_names}")
    print(f"Row names: {row_names}")
    print()
    
    # Show matrix with full precision
    print("Matrix values (full precision):")
    pd.set_option('display.precision', 15)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    print(M_df)
    print()
    
    # Show raw numpy array
    print("Raw numpy array (M_final):")
    print(repr(M_final))
    print()
    
    # Show individual coefficients with full precision
    print("=== Individual Coefficients (Full Precision) ===")
    for i, row_name in enumerate(row_names):
        print(f"{row_name}:")
        for j, col_name in enumerate(col_names):
            print(f"  {col_name}: {M_final[i, j]:.15f}")
        print()
    
    # Show scientific notation
    print("=== Individual Coefficients (Scientific Notation) ===")
    for i, row_name in enumerate(row_names):
        print(f"{row_name}:")
        for j, col_name in enumerate(col_names):
            print(f"  {col_name}: {M_final[i, j]:.15e}")
        print()
    
    # Save to CSV
    output_dir = Path('results/tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    M_df.to_csv(output_dir / 'transformation_matrix_standalone.csv', index=True)
    print(f"Matrix saved to: {output_dir / 'transformation_matrix_standalone.csv'}")
    
    # Show standardization parameters used
    print("\n=== Standardization Parameters ===")
    print("Feature means:")
    for i, col_name in enumerate(col_names):
        print(f"  {col_name}: {scaler_full.mean_[i]:.6f}")
    print("\nFeature standard deviations:")
    for i, col_name in enumerate(col_names):
        print(f"  {col_name}: {scaler_full.scale_[i]:.6f}")

if __name__ == "__main__":
    main()