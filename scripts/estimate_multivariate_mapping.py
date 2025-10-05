

import pandas as pd
import numpy as np
from numpy.linalg import inv, pinv, LinAlgError, norm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_and_merge(aaii_path, google_path, umich_path, returns_path, tol_days=7):
    dfs = []
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
    """
    Build aligned design matrix B and target matrix R for multi-step horizons.
    Columns correspond to time index t = (n_lags - 1) .. (T - m_horizon - 1).
    For each column k (0-based), map to calendar t = (n_lags - 1 + k):
      - Predictors: for each series and each lag i in [0, n_lags-1], use value at (t - i)
      - Targets: returns at horizons t+1 .. t+m_horizon
    This yields consistent shapes:
      B shape: (num_series * n_lags, T_prime)
      R shape: (m_horizon, T_prime)
    where T_prime = T - n_lags - m_horizon + 1.
    """
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
    BBt = B @ B.T
    if ridge_lambda > 0:
        BBt += ridge_lambda * np.eye(BBt.shape[0])
    try:
        inv_BBt = inv(BBt)
    except LinAlgError:
        inv_BBt = pinv(BBt)
    return R @ B.T @ inv_BBt

def grid_search_cv(B, R, ridge_list, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    def cv_score(ridge_lambda):
        mses = []
        for train_idx, test_idx in tscv.split(B.T):
            B_tr, B_te = B[:, train_idx], B[:, test_idx]
            R_tr, R_te = R[:, train_idx], R[:, test_idx]
            M = estimate_transformation(B_tr, R_tr, ridge_lambda)
            R_hat = M @ B_te
            mses.append(mean_squared_error(R_te.flatten(), R_hat.flatten()))
        return np.mean(mses)
    scores = {lam: cv_score(lam) for lam in ridge_list}
    best = min(scores, key=scores.get)
    return best, scores

def evaluate(R_true, R_pred):
    y_true, y_pred = R_true.flatten(), R_pred.flatten()
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0,1]
    return mse, r2, corr

def directional_hit(R_true, R_pred):
    return np.mean(np.sign(R_true.flatten()) == np.sign(R_pred.flatten()))

def backtest(R_true, R_pred):
    strat = np.sign(R_pred.flatten())
    returns = R_true.flatten()
    strat_returns = strat * returns
    cum_ret = np.cumprod(1 + strat_returns) - 1
    sharpe = np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(52)
    return cum_ret[-1], sharpe

def coef_significance(B, R):
    # assumes m_horizon=1
    y = R.flatten()
    X = B.T
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    return model.params, model.pvalues

def stability_analysis(df, split_date, n_lags, m_horizon, ridge_lambda):
    df_pre = df[df.Date < split_date]
    df_post = df[df.Date >= split_date]
    B_pre, R_pre, _ = create_design_return(df_pre, n_lags, m_horizon)
    B_post, R_post, _ = create_design_return(df_post, n_lags, m_horizon)
    M_pre = estimate_transformation(B_pre, R_pre, ridge_lambda)
    M_post = estimate_transformation(B_post, R_post, ridge_lambda)
    delta = norm(M_pre - M_post, ord='fro')
    return delta

def main(aaii, google, umich, returns, n_lags, m_horizon, ridge_lambda):
    df = load_and_merge(aaii, google, umich, returns)
    df['Date'] = pd.to_datetime(df['Date'])
    # Prepare design and return matrices
    B, R, T_prime = create_design_return(df, n_lags, m_horizon)
    # Hyperparameter search
    best_lambda, cv_scores = grid_search_cv(B, R, [ridge_lambda])
    # Final estimate
    M = estimate_transformation(B, R, best_lambda)
    R_hat = M @ B
    mse, r2, corr = evaluate(R, R_hat)
    hit = directional_hit(R, R_hat)
    final_ret, sharpe = backtest(R, R_hat)
    params, pvals = coef_significance(B, R)
    delta = stability_analysis(df, "2015-01-01", n_lags, m_horizon, ridge_lambda)

    print(f"=== Multivariate Mapping (n_lags={n_lags}, m_horizon={m_horizon}) ===")
    print(f"Grid Ridge λ CV scores: {cv_scores}")
    print(f"Selected λ: {best_lambda}")
    print(f"MSE: {mse:.6e}, R^2: {r2:.4f}, Corr: {corr:.4f}")
    print(f"Hit rate: {hit:.4f}")
    print(f"Strategy final return: {final_ret:.4f}, Sharpe: {sharpe:.4f}")
    print("Coefficients and p-values:")
    for name, coef, pv in zip(['const'] + [f'x{i}' for i in range(B.shape[0])], params, pvals):
        print(f"  {name}: coef={coef:.6f}, p={pv:.4f}")
    print(f"Frobenius stability Delta pre/post 2015-01-01: {delta:.4f}")

def grid_search(aaii_path, google_path, umich_path, returns_path):
    """
    Cross-validated grid search over (n_lags, ridge_lambda, m_horizon) to maximize
    mean out-of-sample R^2 using TimeSeriesSplit. Prints per-combination CV R^2
    and reports the best configuration; then fits the final model on the full
    sample with the best hyperparameters and reports in-sample metrics as well.
    """
    import pandas as pd
    
    n_lags_list = [1, 2, 3, 4, 6, 8, 12]
    ridge_list = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    m_horizon_list = [1, 2, 4]

    df = load_and_merge(aaii_path, google_path, umich_path, returns_path)
    df['Date'] = pd.to_datetime(df['Date'])

    best_cv_r2 = -np.inf
    best_params = None
    cv_results = []

    for n_lags in n_lags_list:
        for ridge_lambda in ridge_list:
            for m_horizon in m_horizon_list:
                try:
                    B, R, T_prime = create_design_return(df, n_lags, m_horizon)
                    if T_prime <= 0 or B.shape[1] < 20:
                        # Not enough samples for CV
                        continue

                    # Choose number of splits based on sample size
                    n_obs = B.shape[1]
                    n_splits = 5 if n_obs >= 100 else (4 if n_obs >= 80 else (3 if n_obs >= 50 else 2))
                    tscv = TimeSeriesSplit(n_splits=n_splits)

                    fold_r2s = []
                    for train_idx, test_idx in tscv.split(B.T):
                        B_tr, B_te = B[:, train_idx], B[:, test_idx]
                        R_tr, R_te = R[:, train_idx], R[:, test_idx]
                        # Standardize features based on training data only
                        scaler = StandardScaler(with_mean=True, with_std=True)
                        B_tr_s = scaler.fit_transform(B_tr.T).T
                        B_te_s = scaler.transform(B_te.T).T
                        M_cv = estimate_transformation(B_tr_s, R_tr, ridge_lambda)
                        R_hat_te = M_cv @ B_te_s
                        r2_te = r2_score(R_te.flatten(), R_hat_te.flatten())
                        fold_r2s.append(r2_te)

                    mean_r2 = float(np.mean(fold_r2s)) if fold_r2s else -np.inf
                    cv_results.append((n_lags, ridge_lambda, m_horizon, mean_r2))
                    print(f"CV | n_lags={n_lags}, ridge={ridge_lambda:.1e}, m_horizon={m_horizon} -> mean R^2={mean_r2:.4f}")

                    if mean_r2 > best_cv_r2:
                        best_cv_r2 = mean_r2
                        best_params = (n_lags, ridge_lambda, m_horizon)
                except Exception as e:
                    print(f"CV | n_lags={n_lags}, ridge={ridge_lambda}, m_horizon={m_horizon} | ERROR: {e}")

    if best_params is None:
        print("No valid hyperparameter configuration found for CV grid search.")
        return

    print("\n=== Best CV R^2 Configuration ===")
    print(f"Best mean CV R^2: {best_cv_r2:.4f} at n_lags={best_params[0]}, ridge={best_params[1]}, m_horizon={best_params[2]}")

    # Fit final model on full sample with best hyperparameters
    n_lags_best, ridge_best, m_horizon_best = best_params
    B_full, R_full, _ = create_design_return(df, n_lags_best, m_horizon_best)
    # Standardize on full sample for reporting in-sample metrics
    scaler_full = StandardScaler(with_mean=True, with_std=True)
    B_full_s = scaler_full.fit_transform(B_full.T).T
    M_final = estimate_transformation(B_full_s, R_full, ridge_best)
    R_hat_full = M_final @ B_full_s
    mse, r2, corr = evaluate(R_full, R_hat_full)
    hit = directional_hit(R_full, R_hat_full)
    final_ret, sharpe = backtest(R_full, R_hat_full)
    params, pvals = coef_significance(B_full, R_full)
    delta = stability_analysis(df, "2015-01-01", n_lags_best, m_horizon_best, ridge_best)

    print("\n=== Final Model (Fitted on Full Sample with Best Hyperparameters) ===")
    # Avoid unicode arrows for Windows consoles
    print(f"Selected params -> n_lags={n_lags_best}, m_horizon={m_horizon_best}, ridge={ridge_best}")
    print(f"In-sample MSE: {mse:.6e}, R^2: {r2:.4f}, Corr: {corr:.4f}")
    print(f"Hit rate: {hit:.4f}")
    print(f"Strategy final return: {final_ret:.4f}, Sharpe: {sharpe:.4f}")
    print("Coefficients and p-values:")
    for name, coef, pv in zip(['const'] + [f'x{i}' for i in range(B_full.shape[0])], params, pvals):
        print(f"  {name}: coef={coef:.6f}, p={pv:.4f}")
    print(f"Frobenius stability Delta pre/post 2015-01-01: {delta:.4f}")

    # Save results to files
    out_dir = Path('results/tables')
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) CV results table
    try:
        import pandas as pd
        cv_df = pd.DataFrame(cv_results, columns=['n_lags', 'ridge_lambda', 'm_horizon', 'mean_cv_r2'])
        cv_df.to_csv(out_dir / 'multivariate_cv_results.csv', index=False)
    except Exception as e:
        print(f"Warning: failed to save CV results: {e}")

    # 2) Final model summary
    try:
        summary_path = out_dir / 'multivariate_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== Best CV R^2 Configuration ===\n")
            f.write(f"Best mean CV R^2: {best_cv_r2:.6f} at n_lags={n_lags_best}, ridge={ridge_best}, m_horizon={m_horizon_best}\n\n")
            f.write("=== Final Model (Fitted on Full Sample with Best Hyperparameters) ===\n")
            f.write(f"Selected params -> n_lags={n_lags_best}, m_horizon={m_horizon_best}, ridge={ridge_best}\n")
            f.write(f"In-sample MSE: {mse:.6e}, R^2: {r2:.6f}, Corr: {corr:.6f}\n")
            f.write(f"Hit rate: {hit:.6f}\n")
            f.write(f"Strategy final return: {final_ret:.6f}, Sharpe: {sharpe:.6f}\n")
            f.write(f"Frobenius stability Delta pre/post 2015-01-01: {delta:.6f}\n")
    except Exception as e:
        print(f"Warning: failed to save summary: {e}")

    # 3) Coefficients and p-values
    try:
        coef_names = ['const'] + [f'x{i}' for i in range(B_full.shape[0])]
        coef_df = pd.DataFrame({'name': coef_names, 'coef': params, 'p_value': pvals})
        coef_df.to_csv(out_dir / 'multivariate_coeffs.csv', index=False)
    except Exception as e:
        print(f"Warning: failed to save coefficients: {e}")

    # 4) In-sample predictions and strategy returns (horizon 1)
    try:
        horizon_idx = 0
        y_true = R_full[horizon_idx, :]
        y_pred = R_hat_full[horizon_idx, :]
        # Align with dates used for this design
        T_prime = y_true.shape[0]
        start_idx = n_lags_best
        end_idx = start_idx + T_prime
        dates_used = df['Date'].iloc[start_idx:end_idx].reset_index(drop=True)
        signal = np.sign(y_pred)
        strat_ret = signal * y_true
        pred_df = pd.DataFrame({
            'Date': dates_used,
            'Return_true': y_true,
            'Return_pred': y_pred,
            'Signal': signal,
            'StrategyReturn': strat_ret,
        })
        pred_df.to_csv(out_dir / 'multivariate_in_sample.csv', index=False)
    except Exception as e:
        print(f"Warning: failed to save predictions: {e}")

    # 5) Transformation matrix M with proper labels
    try:
        # Create column names: behavioral_series × lags
        behavioral_series = ['AAII', 'Google', 'UMich']
        col_names = []
        for series_name in behavioral_series:
            for lag in range(n_lags_best):
                col_names.append(f'{series_name}_lag{lag}')
        
        # Create row names: horizons
        row_names = [f'horizon_{h+1}' for h in range(m_horizon_best)]
        
        # Create DataFrame with transformation matrix
        M_df = pd.DataFrame(M_final, columns=col_names, index=row_names)
        M_df.to_csv(out_dir / 'multivariate_transformation_matrix.csv', index=True)
        
        print(f"\n=== Transformation Matrix M ===")
        print(f"Shape: {M_final.shape} (rows=horizons, columns=behavioral×lags)")
        print(f"Column names: {col_names}")
        print(f"Row names: {row_names}")
        print(f"\nMatrix values:")
        print(M_df)
    except Exception as e:
        print(f"Warning: failed to save transformation matrix: {e}")

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
    parser.add_argument("--grid_search", action="store_true",
                        help="Run grid search over n_lags, ridge_lambda, m_horizon to maximize R^2")
    args = parser.parse_args()
    if args.grid_search:
        grid_search(args.aaii, args.google, args.umich, args.returns)
    else:
        main(args.aaii, args.google, args.umich, args.returns,
             args.n_lags, args.m_horizon, args.ridge_lambda)
