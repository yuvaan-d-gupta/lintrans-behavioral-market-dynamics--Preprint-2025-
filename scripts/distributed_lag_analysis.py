

import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse

def load_and_merge(aaii_path, google_path, umich_path, returns_path, tol_days=7):
    # Load behavioural series
    df_aaii   = pd.read_csv(aaii_path, parse_dates=['Date']).rename(columns={'Sentiment':'AAII'})
    df_google = pd.read_csv(google_path, parse_dates=['Date']).rename(columns={'Anxiety':'Google'})
    df_umich  = pd.read_csv(umich_path, parse_dates=['Date']).rename(columns={'Sentiment':'UMich'})
    df_ret    = pd.read_csv(returns_path, parse_dates=['Date']).rename(columns={'Return':'Return'})
    # Sort and merge asof
    for df in (df_aaii, df_google, df_umich, df_ret):
        df.sort_values('Date', inplace=True)
    df = pd.merge_asof(df_ret, df_aaii,   on='Date', tolerance=pd.Timedelta(f'{tol_days}D'), direction='nearest')
    df = pd.merge_asof(df,     df_google, on='Date', tolerance=pd.Timedelta(f'{tol_days}D'), direction='nearest')
    df = pd.merge_asof(df,     df_umich,  on='Date', tolerance=pd.Timedelta(f'{tol_days}D'), direction='nearest')
    df.dropna(subset=['AAII','Google','UMich','Return'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def verify_alignment(df, n_lags, check_n=5):
    print("\n--- Offset Verification ---")
    for t in range(min(check_n, len(df)-n_lags-1)):
        b_dates = df['Date'].iloc[t:t+n_lags].dt.strftime('%Y-%m-%d').tolist()
        r_date = df['Date'].iloc[t+n_lags]
        print(f"Column {t}: behaviour dates {b_dates} -> return date {r_date.strftime('%Y-%m-%d')}")

def explore_latent_effects(df, indicators, max_lag):
    print("\n--- Cross-Correlation (lag vs. return) ---")
    for ind in indicators:
        print(f"\nIndicator: {ind}")
        for k in range(max_lag+1):
            corr = df[ind].shift(k).corr(df['Return'])
            print(f" Lag {k:2d}: corr = {corr:.4f}")

def run_distributed_lag(df, indicator, K):
    """
    Fit distributed-lag regression: Return_t ~ sum_{k=0..K-1} γ_k * indicator_{t-k}
    """
    data = df[[indicator, 'Return']].copy()
    for k in range(K):
        data[f'{indicator}_lag{k}'] = data[indicator].shift(k)
    data.dropna(inplace=True)
    X = data[[f'{indicator}_lag{k}' for k in range(K)]]
    X = sm.add_constant(X)
    y = data['Return']
    model = sm.OLS(y, X).fit()
    gamma = model.params.values[1:]
    pvals  = model.pvalues.values[1:]
    cum_effect = gamma.sum()
    mse   = np.mean(model.resid**2)
    r2    = model.rsquared
    corr  = np.corrcoef(y, model.fittedvalues)[0,1]
    print(f"\n=== Distributed‐Lag: {indicator} (K={K}) ===")
    print("Lag |   γ_k    | p-value")
    for k, (g, p) in enumerate(zip(gamma, pvals)):
        print(f"{k:3d} | {g: .6f} | {p: .4f}")
    print(f"Cumulative Σγ = {cum_effect:.6f}, MSE = {mse:.6e}, R² = {r2:.4f}, Corr = {corr:.4f}")

def main(aaii_path, google_path, umich_path, returns_path, K, max_lag, tol_days):
    df = load_and_merge(aaii_path, google_path, umich_path, returns_path, tol_days)
    print(f"Merged DataFrame shape: {df.shape}")

    # 1. Verify offset alignment
    verify_alignment(df, K)

    # 2. Explore latent effects via cross-correlation
    explore_latent_effects(df, ['AAII','Google','UMich'], max_lag)

    # 3. Distributed-lag regression per indicator
    for indicator in ['AAII','Google','UMich']:
        run_distributed_lag(df, indicator, K)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed‐lag analysis with offset and latent-effect checks")
    parser.add_argument("--aaii",   default="raw_data/aaii_sentiment_processed.csv")
    parser.add_argument("--google", default="raw_data/google_anxiety_processed.csv")
    parser.add_argument("--umich",  default="raw_data/umich_sentiment_weekly.csv")
    parser.add_argument("--returns",default="raw_data/spy_weekly_returns.csv")
    parser.add_argument("--K",       type=int, default=12, help="Number of lags to include")
    parser.add_argument("--max_lag", type=int, default=24, help="Max lag for cross-correlation")
    parser.add_argument("--tol_days",type=int, default=7, help="Merge asof tolerance in days")
    args = parser.parse_args()
    main(args.aaii, args.google, args.umich, args.returns, args.K, args.max_lag, args.tol_days)

