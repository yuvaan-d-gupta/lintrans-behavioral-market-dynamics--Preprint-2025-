

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from numpy.linalg import inv, pinv, LinAlgError
from sklearn.preprocessing import StandardScaler

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

def create_heatmap():
    """Create coefficient heatmap for both OLS and Ridge regression."""
    
    # Load data
    df = load_and_merge(
        "raw_data/aaii_sentiment_processed.csv",
        "raw_data/google_anxiety_processed.csv", 
        "raw_data/umich_sentiment_weekly.csv",
        "raw_data/spy_weekly_returns.csv"
    )
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Parameters for full matrix
    n_lags = 4
    m_horizon = 12
    
    # Create design and return matrices
    B_full, R_full, T_prime = create_design_return(df, n_lags, m_horizon)
    
    # Standardize features
    scaler = StandardScaler(with_mean=True, with_std=True)
    B_full_s = scaler.fit_transform(B_full.T).T
    
    # Estimate transformation matrices for OLS and Ridge
    M_ols = estimate_transformation(B_full_s, R_full, 0.0)  # OLS
    M_ridge = estimate_transformation(B_full_s, R_full, 1.0)  # Ridge
    
    # Create proper column and row names
    behavioral_series = ['AAII', 'Google', 'UMich']
    col_names = []
    for series_name in behavioral_series:
        for lag in range(n_lags):
            col_names.append(f'{series_name}_lag{lag}')
    
    row_names = [f'Horizon {h+1}' for h in range(m_horizon)]
    
    # Create DataFrames
    M_ols_df = pd.DataFrame(M_ols, columns=col_names, index=row_names)
    M_ridge_df = pd.DataFrame(M_ridge, columns=col_names, index=row_names)
    
    # Create the heatmap
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
    
    # Color scale limits (symmetric around zero for better comparison)
    vmax = max(np.abs(M_ols).max(), np.abs(M_ridge).max())
    vmin = -vmax
    
    # OLS Heatmap
    sns.heatmap(M_ols_df, annot=True, fmt='.4f', cmap='RdBu_r', center=0,
                vmin=vmin, vmax=vmax, ax=ax1, cbar_kws={'label': 'Coefficient Value'})
    ax1.set_title('OLS Transformation Matrix M (Ridge λ = 0.0)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Behavioral Features (Indicator × Lag)', fontsize=12)
    ax1.set_ylabel('Return Horizon (weeks ahead)', fontsize=12)
    
    # Ridge Heatmap
    sns.heatmap(M_ridge_df, annot=True, fmt='.4f', cmap='RdBu_r', center=0,
                vmin=vmin, vmax=vmax, ax=ax2, cbar_kws={'label': 'Coefficient Value'})
    ax2.set_title('Ridge Transformation Matrix M (Ridge λ = 1.0)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Behavioral Features (Indicator × Lag)', fontsize=12)
    ax2.set_ylabel('Return Horizon (weeks ahead)', fontsize=12)
    
    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'coefficient_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'coefficient_heatmap.pdf', bbox_inches='tight')
    
    # Find and print top coefficients
    print("=== Top 10 Absolute Coefficients (OLS) ===")
    ols_flat = M_ols_df.stack().abs().sort_values(ascending=False)
    for i in range(min(10, len(ols_flat))):
        horizon, feature = ols_flat.index[i]
        value = M_ols_df.loc[horizon, feature]
        print(f"{i+1:2d}. {horizon} × {feature}: {value:8.4f}")
    
    print("\n=== Top 10 Absolute Coefficients (Ridge) ===")
    ridge_flat = M_ridge_df.stack().abs().sort_values(ascending=False)
    for i in range(min(10, len(ridge_flat))):
        horizon, feature = ridge_flat.index[i]
        value = M_ridge_df.loc[horizon, feature]
        print(f"{i+1:2d}. {horizon} × {feature}: {value:8.4f}")
    
    # Compare shrinkage effect
    print("\n=== Ridge Shrinkage Analysis ===")
    shrinkage = ((M_ols_df.abs() - M_ridge_df.abs()) / M_ols_df.abs()).fillna(0)
    print(f"Average shrinkage: {shrinkage.mean().mean():.2%}")
    print(f"Max shrinkage: {shrinkage.max().max():.2%}")
    print(f"Min shrinkage: {shrinkage.min().min():.2%}")
    
    # Save coefficient matrices to CSV
    M_ols_df.to_csv(output_dir.parent / 'tables' / 'transformation_matrix_ols.csv')
    M_ridge_df.to_csv(output_dir.parent / 'tables' / 'transformation_matrix_ridge.csv')
    
    print(f"\nHeatmap saved to: {output_dir / 'coefficient_heatmap.png'}")
    print(f"PDF version saved to: {output_dir / 'coefficient_heatmap.pdf'}")
    print(f"OLS matrix saved to: {output_dir.parent / 'tables' / 'transformation_matrix_ols.csv'}")
    print(f"Ridge matrix saved to: {output_dir.parent / 'tables' / 'transformation_matrix_ridge.csv'}")
    
    plt.close()

def create_single_heatmap():
    """Create a single detailed heatmap with the exact values from our previous run."""
    
    # Use the exact transformation matrix from our previous calculation
    # This is the 12x12 matrix we computed earlier
    transformation_matrix = np.array([
        [-1.26971351e-03,  2.79255933e-03, -3.01221023e-03,  3.04134629e-04,  6.09198382e-04,  3.47662323e-03, -2.33782085e-03,  8.62460692e-05,  3.91906879e-02, -3.35818004e-02, -1.62641574e-02,  1.07859863e-02],
        [ 1.75735967e-03, -2.90654274e-03,  7.38743581e-04, -3.27606167e-04,  3.94198609e-03, -2.72433518e-03, -7.10752382e-04,  1.62938949e-03, -6.58320599e-05, -1.19120808e-02,  1.09920149e-02,  9.37925000e-04],
        [-1.95593453e-03,  7.93812748e-04, -1.42656832e-03,  1.91260512e-03,  4.32963594e-04, -5.97047187e-04,  2.39500056e-03, -5.92644437e-04, -1.11649490e-02,  7.29645030e-03, -7.33485265e-03,  1.10999808e-02],
        [-7.46431452e-04, -1.71313882e-03,  2.11690461e-04,  2.08811711e-03, -1.82952222e-04,  2.16829717e-03, -3.18777952e-03,  3.11955698e-03, -2.77908001e-03, -8.20866924e-03,  1.84861948e-02, -7.61083196e-03],
        [-2.26874369e-03,  8.83443459e-05,  1.67955824e-03,  5.78567997e-04,  1.97395012e-03, -3.10480545e-03,  1.96332572e-03,  1.29132314e-03, -1.11143681e-02,  1.79433343e-02, -1.02462968e-02,  3.32170152e-03],
        [-1.12209450e-03,  1.79498624e-03,  1.46957931e-03, -2.09160072e-03, -1.45734471e-03,  1.90893807e-03,  3.01447254e-03, -2.15623078e-03,  1.10690572e-02, -1.42280679e-02,  1.10366340e-04,  2.94545399e-03],
        [ 1.23841365e-03,  1.65514322e-03, -9.20559936e-04, -1.78118376e-03,  7.36926220e-04,  3.10594107e-03, -4.76060498e-04, -2.14485234e-03, -4.26297428e-03,  6.66997236e-04, -4.33776408e-03,  7.81673676e-03],
        [ 2.54914457e-03, -6.54939691e-04, -1.69970058e-03, -2.83624261e-04,  3.52907184e-03, -4.00810481e-04,  2.42308928e-04, -2.69275221e-03, -1.84584772e-03, -5.04606754e-03,  1.23310452e-02, -5.60522245e-03],
        [ 1.03550213e-03, -1.70876046e-03, -5.45794973e-06,  7.48319582e-05,  2.27943229e-03,  4.99324099e-04, -3.09997843e-03,  6.83366943e-04, -6.21838074e-03,  1.02013593e-02, -1.28581286e-02,  8.70424907e-03],
        [-1.04574002e-03,  5.37548696e-05, -3.99693042e-04,  8.65872185e-04,  2.42274609e-03, -2.85259294e-03,  2.65007407e-03, -2.38991919e-03,  5.80505924e-03, -1.77347973e-02, -6.36548052e-03,  1.81816943e-02],
        [-7.38370485e-04, -6.89641630e-04,  7.74200617e-04,  4.00785392e-04, -1.32684139e-03,  2.82847488e-03, -4.70999527e-03,  3.25573521e-03, -1.29855277e-02, -8.14104750e-03,  4.80044705e-03,  1.63233825e-02],
        [-1.06724855e-03,  7.38137981e-04,  3.38067669e-04, -9.87108969e-07,  1.63342654e-03, -4.21948788e-03,  3.69438135e-03, -1.04146126e-03, -1.96558580e-02,  1.85022340e-03,  3.74005303e-03,  1.41521247e-02]
    ])
    
    # Create proper labels
    behavioral_series = ['AAII', 'Google', 'UMich']
    col_names = []
    for series_name in behavioral_series:
        for lag in range(4):
            col_names.append(f'{series_name}_lag{lag}')
    
    row_names = [f'Horizon {h+1}' for h in range(12)]
    
    # Create DataFrame
    M_df = pd.DataFrame(transformation_matrix, columns=col_names, index=row_names)
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with annotations for significant coefficients
    mask_small = np.abs(transformation_matrix) < 0.005  # Mask small coefficients
    
    sns.heatmap(M_df, annot=True, fmt='.4f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Coefficient Value', 'shrink': 0.8},
                linewidths=0.5, linecolor='gray')
    
    plt.title('Coefficient Heatmap of Transformation Matrix M\n(Ridge Regression, λ = 1.0)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Behavioral Features (Indicator × Lag)', fontsize=14)
    plt.ylabel('Return Horizon (weeks ahead)', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'coefficient_heatmap_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'coefficient_heatmap_detailed.pdf', bbox_inches='tight')
    
    # Find and print top coefficients
    print("=== Top 15 Absolute Coefficients ===")
    flat_coeffs = M_df.stack().abs().sort_values(ascending=False)
    for i in range(min(15, len(flat_coeffs))):
        horizon, feature = flat_coeffs.index[i]
        value = M_df.loc[horizon, feature]
        print(f"{i+1:2d}. {horizon} × {feature}: {value:8.4f}")
    
    # Analysis by feature type
    print("\n=== Analysis by Behavioral Indicator ===")
    for series in behavioral_series:
        series_cols = [col for col in col_names if col.startswith(series)]
        series_mean = M_df[series_cols].abs().mean().mean()
        series_max = M_df[series_cols].abs().max().max()
        print(f"{series:7s}: Mean |coeff| = {series_mean:.4f}, Max |coeff| = {series_max:.4f}")
    
    # Analysis by horizon
    print("\n=== Analysis by Horizon ===")
    for i in range(min(6, len(row_names))):  # Show first 6 horizons
        horizon = row_names[i]
        horizon_mean = M_df.loc[horizon].abs().mean()
        horizon_max = M_df.loc[horizon].abs().max()
        print(f"{horizon:10s}: Mean |coeff| = {horizon_mean:.4f}, Max |coeff| = {horizon_max:.4f}")
    
    print(f"\nDetailed heatmap saved to: {output_dir / 'coefficient_heatmap_detailed.png'}")
    print(f"PDF version saved to: {output_dir / 'coefficient_heatmap_detailed.pdf'}")
    
    # Close the plot to free memory
    plt.close()
    
    return M_df

if __name__ == "__main__":
    print("Creating coefficient heatmap using exact transformation matrix values...")
    M_df = create_single_heatmap()