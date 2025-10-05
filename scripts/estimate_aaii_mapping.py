

import pandas as pd
import numpy as np
from numpy.linalg import inv, pinv, LinAlgError
from sklearn.metrics import mean_squared_error, r2_score

def load_and_merge(aaii_path, google_path, umich_path, returns_path):
    """
    Load processed behavioural CSVs and market returns CSV, merge on nearest Date (within tolerance).
    Expects:
      - aaii_path: ['Date','Sentiment'] -> renamed to 'AAII'
      - google_path: ['Date','Anxiety'] -> renamed to 'Google'
      - umich_path: ['Date','Sentiment'] -> renamed to 'UMich'
      - returns_path: ['Date','Return']
    """
    # Load and rename columns
    df_aaii = pd.read_csv(aaii_path, parse_dates=['Date']).rename(columns={'Sentiment':'AAII'})
    df_google = pd.read_csv(google_path, parse_dates=['Date']).rename(columns={'Anxiety':'Google'})
    df_umich = pd.read_csv(umich_path, parse_dates=['Date']).rename(columns={'Sentiment':'UMich'})
    df_ret = pd.read_csv(returns_path, parse_dates=['Date']).rename(columns={'Return':'Return'})

    # Sort for asof merge
    for df in (df_aaii, df_google, df_umich, df_ret):
        df.sort_values('Date', inplace=True)

    # Merge sequentially with 7-day tolerance
    df = pd.merge_asof(df_aaii, df_google, on='Date', tolerance=pd.Timedelta('7D'), direction='nearest')
    df = pd.merge_asof(df, df_umich, on='Date', tolerance=pd.Timedelta('7D'), direction='nearest')
    df = pd.merge_asof(df, df_ret, on='Date', tolerance=pd.Timedelta('7D'), direction='nearest')

    # Drop any rows with missing data
    df.dropna(subset=['AAII','Google','UMich','Return'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Merged DataFrame shape: {df.shape}")
    print(df.head(10))
    return df

def create_lagged_matrix(series, n_lags, T_prime, offset=0):
    """
    Build lagged matrix for one series: shape (n_lags, T_prime).
    series[offset + i : offset + i + T_prime] for i in 0..n_lags-1
    """
    B = np.zeros((n_lags, T_prime))
    for i in range(n_lags):
        B[i] = series[offset + i : offset + i + T_prime]
    return B

def create_return_matrix(series, m_horizon, T_prime, offset, n_lags):
    """
    Build return matrix: shape (m_horizon, T_prime),
    series[offset + n_lags + j : ...] for j in 0..m_horizon-1
    """
    R = np.zeros((m_horizon, T_prime))
    for j in range(m_horizon):
        R[j] = series[offset + n_lags + j : offset + n_lags + j + T_prime]
    return R

def estimate_transformation(B, R, ridge_lambda=0.0):
    """
    Estimate M in R = M B + E via OLS or Ridge.
    """
    BBt = B @ B.T
    if ridge_lambda > 0:
        BBt += ridge_lambda * np.eye(BBt.shape[0])
    try:
        inv_BBt = inv(BBt)
    except LinAlgError:
        inv_BBt = pinv(BBt)
    return R @ B.T @ inv_BBt

def predict_returns(M, B):
    """
    Predict returns R_hat = M B.
    """
    return M @ B

def evaluate_performance(R_true, R_pred):
    """
    Compute MSE, R^2, and correlation between flattened true and predicted returns.
    """
    y_true = R_true.flatten()
    y_pred = R_pred.flatten()
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0,1]
    return mse, r2, corr

def main(aaii_path, google_path, umich_path, returns_path, n_lags, m_horizon, ridge_lambda):
    df = load_and_merge(aaii_path, google_path, umich_path, returns_path)

    # Prepare series arrays
    aa = df['AAII'].values
    go = df['Google'].values
    um = df['UMich'].values
    ret = df['Return'].values

    # Determine effective sample size
    offset = 0
    T_prime = len(df) - max(n_lags, m_horizon)

    # Build lagged design blocks
    B1 = create_lagged_matrix(aa, n_lags, T_prime, offset)
    B2 = create_lagged_matrix(go, n_lags, T_prime, offset)
    B3 = create_lagged_matrix(um, n_lags, T_prime, offset)

    # Stack into full design matrix
    B = np.vstack([B1, B2, B3])

    # Build return matrix
    R_mat = create_return_matrix(ret, m_horizon, T_prime, offset, n_lags)

    # Estimate transformation
    M = estimate_transformation(B, R_mat, ridge_lambda)

    # Predict and evaluate
    R_hat = predict_returns(M, B)
    mse, r2, corr = evaluate_performance(R_mat, R_hat)

    print("=== Multivariate Behavioural → Market Returns Mapping ===")
    print(f"Lags (n): {n_lags}, Horizon (m): {m_horizon}, Ridge λ: {ridge_lambda}")
    print("Design matrix B shape:", B.shape)
    print("Estimated M shape:", M.shape)
    print("Transformation matrix M:\n", M)
    print("\nPerformance Metrics:")
    print(f"  MSE:  {mse:.6e}")
    print(f"  R^2:  {r2:.4f}")
    print(f"  Corr: {corr:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Estimate multivariate behavioural indicators → market returns mapping"
    )
    parser.add_argument("--aaii", default="raw_data/aaii_sentiment_processed.csv")
    parser.add_argument("--google", default="raw_data/google_anxiety_processed.csv")
    parser.add_argument("--umich", default="raw_data/umich_sentiment_weekly.csv")
    parser.add_argument("--returns", default="raw_data/spy_weekly_returns.csv")
    parser.add_argument("--n_lags", type=int, default=4,
                        help="Number of lags per behavioural series")
    parser.add_argument("--m_horizon", type=int, default=1,
                        help="Prediction horizon for returns")
    parser.add_argument("--ridge_lambda", type=float, default=0.0,
                        help="Ridge regularization parameter (0 for OLS)")
    args = parser.parse_args()
    main(args.aaii, args.google, args.umich, args.returns,
         args.n_lags, args.m_horizon, args.ridge_lambda)

