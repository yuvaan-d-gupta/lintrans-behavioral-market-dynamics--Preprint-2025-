

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from numpy.linalg import inv, pinv, LinAlgError
import statsmodels.api as sm
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

def create_train_test_split(df, test_start_date="2020-01-01"):
    """Split data into train/test for visualization."""
    df['Date'] = pd.to_datetime(df['Date'])
    train_mask = df['Date'] < test_start_date
    test_mask = df['Date'] >= test_start_date
    return df[train_mask].copy(), df[test_mask].copy()

def create_scatter_plot():
    """Create the scatter plot using exact multivariate model results."""
    
    # Load data
    df = load_and_merge(
        "raw_data/aaii_sentiment_processed.csv",
        "raw_data/google_anxiety_processed.csv", 
        "raw_data/umich_sentiment_weekly.csv",
        "raw_data/spy_weekly_returns.csv"
    )
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Use the best parameters from the multivariate analysis results
    # From the attached results: n_lags=1, ridge=1.0, m_horizon=1
    n_lags_best = 1
    ridge_best = 1.0
    m_horizon_best = 1
    
    # Split into train/test
    df_train, df_test = create_train_test_split(df, "2020-01-01")
    
    # Create design matrices for train and test
    B_train, R_train, T_prime_train = create_design_return(df_train, n_lags_best, m_horizon_best)
    B_test, R_test, T_prime_test = create_design_return(df_test, n_lags_best, m_horizon_best)
    
    # Standardize features (fit on train, transform both)
    scaler = StandardScaler(with_mean=True, with_std=True)
    B_train_s = scaler.fit_transform(B_train.T).T
    B_test_s = scaler.transform(B_test.T).T
    
    # Fit model on training data
    M_final = estimate_transformation(B_train_s, R_train, ridge_best)
    
    # Make predictions
    R_hat_train = M_final @ B_train_s
    R_hat_test = M_final @ B_test_s
    
    # Extract horizon 1 predictions and actuals
    y_train_true = R_train[0, :]
    y_train_pred = R_hat_train[0, :]
    y_test_true = R_test[0, :]
    y_test_pred = R_hat_test[0, :]
    
    # Calculate metrics
    train_r2 = r2_score(y_train_true, y_train_pred)
    test_r2 = r2_score(y_test_true, y_test_pred)
    train_corr = np.corrcoef(y_train_true, y_train_pred)[0, 1]
    test_corr = np.corrcoef(y_test_true, y_test_pred)[0, 1]
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training data panel
    ax1.scatter(y_train_true, y_train_pred, alpha=0.6, s=20, color='blue', label='Training Data')
    
    # 45-degree identity line
    min_val = min(np.min(y_train_true), np.min(y_train_pred))
    max_val = max(np.max(y_train_true), np.max(y_train_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=1, label='Perfect Prediction (45°)')
    
    # OLS fit line
    z_train = np.polyfit(y_train_true, y_train_pred, 1)
    p_train = np.poly1d(z_train)
    ax1.plot(y_train_true, p_train(y_train_true), "r-", alpha=0.8, linewidth=2, label=f'OLS Fit (slope={z_train[0]:.3f})')
    
    ax1.set_xlabel('Realized Weekly Returns')
    ax1.set_ylabel('Predicted Returns')
    ax1.set_title(f'Training Data\nR² = {train_r2:.4f}, Corr = {train_corr:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test data panel
    ax2.scatter(y_test_true, y_test_pred, alpha=0.6, s=20, color='red', label='Test Data')
    
    # 45-degree identity line
    min_val = min(np.min(y_test_true), np.min(y_test_pred))
    max_val = max(np.max(y_test_true), np.max(y_test_pred))
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=1, label='Perfect Prediction (45°)')
    
    # OLS fit line
    z_test = np.polyfit(y_test_true, y_test_pred, 1)
    p_test = np.poly1d(z_test)
    ax2.plot(y_test_true, p_test(y_test_true), "r-", alpha=0.8, linewidth=2, label=f'OLS Fit (slope={z_test[0]:.3f})')
    
    ax2.set_xlabel('Realized Weekly Returns')
    ax2.set_ylabel('Predicted Returns')
    ax2.set_title(f'Test Data (2020+)\nR² = {test_r2:.4f}, Corr = {test_corr:.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('Predicted vs. Realized Returns: Multivariate Model\n(n_lags=1, ridge=1.0, m_horizon=1)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path('results/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'multivariate_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'multivariate_scatter_plot.pdf', bbox_inches='tight')
    
    print(f"Scatter plot saved to {output_dir / 'multivariate_scatter_plot.png'}")
    print(f"PDF version saved to {output_dir / 'multivariate_scatter_plot.pdf'}")
    
    # Print summary statistics
    print(f"\n=== Model Performance Summary ===")
    print(f"Training Data (pre-2020):")
    print(f"  R² = {train_r2:.4f}")
    print(f"  Correlation = {train_corr:.4f}")
    print(f"  Observations = {len(y_train_true)}")
    print(f"\nTest Data (2020+):")
    print(f"  R² = {test_r2:.4f}")
    print(f"  Correlation = {test_corr:.4f}")
    print(f"  Observations = {len(y_test_true)}")
    print(f"\nModel Parameters:")
    print(f"  n_lags = {n_lags_best}")
    print(f"  ridge_lambda = {ridge_best}")
    print(f"  m_horizon = {m_horizon_best}")
    
    plt.show()

if __name__ == "__main__":
    create_scatter_plot()