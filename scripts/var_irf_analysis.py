

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import argparse

def load_and_merge(aaii_path, google_path, umich_path, returns_path, tol_days):
    # Load series
    df_ret    = pd.read_csv(returns_path, parse_dates=['Date']).rename(columns={'Return':'Return'})
    df_aaii   = pd.read_csv(aaii_path,   parse_dates=['Date']).rename(columns={'Sentiment':'AAII'})
    df_google = pd.read_csv(google_path, parse_dates=['Date']).rename(columns={'Anxiety':'Google'})
    df_umich  = pd.read_csv(umich_path,  parse_dates=['Date']).rename(columns={'Sentiment':'UMich'})
    # Sort for asof merge
    for df in (df_ret, df_aaii, df_google, df_umich):
        df.sort_values('Date', inplace=True)
    # Merge nearest-date within tolerance
    df = pd.merge_asof(df_ret,    df_aaii,   on='Date', tolerance=pd.Timedelta(f'{tol_days}D'), direction='nearest')
    df = pd.merge_asof(df,        df_google, on='Date', tolerance=pd.Timedelta(f'{tol_days}D'), direction='nearest')
    df = pd.merge_asof(df,        df_umich,  on='Date', tolerance=pd.Timedelta(f'{tol_days}D'), direction='nearest')
    df.dropna(subset=['Return','AAII','Google','UMich'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def extract_pc1(series_blocks, n_lags):
    # series_blocks: list of arrays [AAII, Google, UMich], each length T
    # Build full design matrix of lags
    T = len(series_blocks[0])
    T_prime = T - n_lags
    B = np.vstack([block[i:i+T_prime] 
                   for block in series_blocks 
                   for i in range(n_lags)])
    # PCA to 1 component
    X = B.T  # shape (T_prime, n_features)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X).flatten()
    return pc1, pca

def run_var_irf(df, pc1, returns, n_lags, maxlags, irf_horizon):
    # Align PC1 and returns for VAR
    T_prime = len(pc1)
    dates = df['Date'].iloc[n_lags:n_lags+T_prime]
    var_df = pd.DataFrame({
        'Return': returns[n_lags:n_lags+T_prime],
        'Sentiment_PC1': pc1
    }, index=dates)
    # Select VAR order by information criteria
    model = VAR(var_df)
    sel = model.select_order(maxlags=maxlags)
    best_aic = sel.aic
    best_bic = sel.bic
    p = best_aic or best_bic or sel.aic  # prefer AIC
    print(f"Selected VAR order (AIC): {best_aic}, (BIC): {best_bic}")
    # Fit VAR
    results = model.fit(p)
    print(results.summary())
    # Compute IRF
    irf = results.irf(irf_horizon)
    # Plot IRF for Sentiment_PC1 → Return
    fig = irf.plot(orth=False, impulse='Sentiment_PC1', response='Return')
    fig.suptitle('IRF: Shock in Sentiment_PC1 on Return')
    plt.tight_layout()
    plt.show()
    # Return IRF values
    return irf.irfs, sel

def main(args):
    # Load data
    df = load_and_merge(args.aaii, args.google, args.umich, args.returns, args.tol_days)
    print(f"Merged DataFrame shape: {df.shape}")
    # Extract PC1 from behavioural lags
    blocks = [df['AAII'].values, df['Google'].values, df['UMich'].values]
    pc1, pca = extract_pc1(blocks, args.n_lags)
    print(f"PC1 explained variance: {pca.explained_variance_ratio_[0]:.4f}")
    # Run VAR & IRF
    irf_values, sel = run_var_irf(df, pc1, df['Return'].values, args.n_lags, args.maxlags, args.irf_horizon)
    # Save IRF values for further analysis
    np.save(args.output_irf, irf_values)
    print(f"IRF values saved to {args.output_irf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAR+IRF analysis for behavioural PC1 and returns")
    parser.add_argument("--aaii",   default="raw_data/aaii_sentiment_processed.csv")
    parser.add_argument("--google", default="raw_data/google_anxiety_processed.csv")
    parser.add_argument("--umich",  default="raw_data/umich_sentiment_weekly.csv")
    parser.add_argument("--returns",default="raw_data/spy_weekly_returns.csv")
    parser.add_argument("--n_lags", type=int, default=12, help="lags per indicator for PC extraction")
    parser.add_argument("--maxlags", type=int, default=12, help="max VAR lags to consider")
    parser.add_argument("--irf_horizon", type=int, default=12, help="number of weeks for IRF")
    parser.add_argument("--tol_days", type=int, default=7, help="merge tolerance in days")
    parser.add_argument("--output_irf", default="results/irf_values.npy", help="path to save IRF array")
    args = parser.parse_args()
    main(args)