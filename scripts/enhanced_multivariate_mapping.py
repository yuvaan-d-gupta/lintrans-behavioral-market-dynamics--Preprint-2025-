

import pandas as pd
import numpy as np
from numpy.linalg import inv, pinv, LinAlgError, norm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

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

def create_enhanced_features(df, n_lags):
    """Create enhanced features including rolling statistics and momentum"""
    features = []
    
    # Original series
    for col in ['AAII', 'Google', 'UMich']:
        series = df[col].values
        for lag in range(1, n_lags + 1):
            features.append(np.roll(series, lag))
    
    # Rolling statistics
    for col in ['AAII', 'Google', 'UMich']:
        series = df[col].values
        # Rolling mean (4 weeks)
        rolling_mean = pd.Series(series).rolling(window=4, min_periods=1).mean().values
        features.append(rolling_mean)
        
        # Rolling std (4 weeks)
        rolling_std = pd.Series(series).rolling(window=4, min_periods=1).std().values
        features.append(rolling_std)
        
        # Momentum (change over 4 weeks)
        momentum = series - np.roll(series, 4)
        features.append(momentum)
        
        # Volatility (rolling std of returns)
        returns = np.diff(series) / series[:-1]
        vol = pd.Series(returns).rolling(window=4, min_periods=1).std().values
        vol = np.concatenate([[np.nan], vol])  # Align with original series
        features.append(vol)
    
    # Cross-series interactions
    aaii = df['AAII'].values
    google = df['Google'].values
    umich = df['UMich'].values
    
    # Interaction terms
    features.append(aaii * google)
    features.append(aaii * umich)
    features.append(google * umich)
    
    # Sentiment divergence (AAII vs UMich)
    sentiment_divergence = aaii - umich
    features.append(sentiment_divergence)
    
    # Anxiety-sentiment ratio
    anxiety_sentiment_ratio = google / ((aaii + umich) / 2 + 1e-8)
    features.append(anxiety_sentiment_ratio)
    
    # Non-linear transformations
    features.append(np.log(np.abs(aaii) + 1e-8))
    features.append(np.log(np.abs(google) + 1e-8))
    features.append(np.log(np.abs(umich) + 1e-8))
    
    # Squared terms for non-linearity
    features.append(aaii ** 2)
    features.append(google ** 2)
    features.append(umich ** 2)
    
    return np.array(features).T

def create_design_return_enhanced(df, n_lags, m_horizon):
    """Create enhanced design matrix with engineered features"""
    features = create_enhanced_features(df, n_lags)
    returns = df['Return'].values
    
    T = len(df)
    T_prime = T - max(n_lags, m_horizon)
    
    # Use engineered features instead of simple lags
    B = features[n_lags:n_lags + T_prime]
    R = np.vstack([returns[n_lags + j : n_lags + j + T_prime] for j in range(m_horizon)])
    
    return B.T, R, T_prime

def estimate_transformation_enhanced(B, R, method='ridge', alpha=1.0, l1_ratio=0.5):
    """Enhanced estimation with multiple methods"""
    if method == 'ridge':
        model = Ridge(alpha=alpha)
    elif method == 'lasso':
        model = Lasso(alpha=alpha)
    elif method == 'elastic_net':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    elif method == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        # Original OLS method
        BBt = B @ B.T
        try:
            inv_BBt = inv(BBt)
        except LinAlgError:
            inv_BBt = pinv(BBt)
        return R @ B.T @ inv_BBt
    
    # For sklearn models, we need to fit on each horizon separately
    if method in ['ridge', 'lasso', 'elastic_net', 'random_forest']:
        M_list = []
        for i in range(R.shape[0]):
            y = R[i, :]
            model.fit(B.T, y)
            if hasattr(model, 'coef_'):
                M_list.append(model.coef_)
            else:
                # For random forest, we'll use feature importances as approximation
                M_list.append(model.feature_importances_)
        return np.array(M_list)
    else:
        return R @ B.T @ inv_BBt

def grid_search_enhanced(aaii_path, google_path, umich_path, returns_path):
    """Enhanced grid search with more sophisticated methods"""
    n_lags_list = [1, 2, 3, 4, 6, 8]
    m_horizon_list = [1, 2, 4]
    methods = ['ols', 'ridge', 'lasso', 'elastic_net', 'random_forest']
    alpha_list = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    
    best_r2 = -np.inf
    best_corr = -np.inf
    best_params = None
    best_metrics = None
    
    print("=== Enhanced Grid Search ===")
    print("Testing multiple methods and parameters...")
    
    for n_lags in n_lags_list:
        for m_horizon in m_horizon_list:
            for method in methods:
                if method == 'ols':
                    alpha_list_method = [0.0]
                else:
                    alpha_list_method = alpha_list
                
                for alpha in alpha_list_method:
                    try:
                        df = load_and_merge(aaii_path, google_path, umich_path, returns_path)
                        B, R, T_prime = create_design_return_enhanced(df, n_lags, m_horizon)
                        
                        if T_prime <= 10:  # Need sufficient data
                            continue
                        
                        M = estimate_transformation_enhanced(B, R, method, alpha)
                        R_hat = M @ B
                        
                        mse, r2, corr = evaluate(R, R_hat)
                        
                        print(f"n_lags={n_lags}, m_horizon={m_horizon}, method={method}, alpha={alpha} | R²={r2:.4f}, Corr={corr:.4f}, MSE={mse:.6e}")
                        
                        # Track best by R²
                        if r2 > best_r2:
                            best_r2 = r2
                            best_params = (n_lags, m_horizon, method, alpha)
                            best_metrics = (mse, r2, corr)
                        
                        # Track best by correlation
                        if corr > best_corr:
                            best_corr = corr
                    
                    except Exception as e:
                        print(f"Error with n_lags={n_lags}, m_horizon={m_horizon}, method={method}, alpha={alpha}: {e}")
    
    print("\n=== Best Results ===")
    print(f"Best R²: {best_r2:.4f} at params: {best_params}")
    print(f"Best Correlation: {best_corr:.4f}")
    print(f"Best MSE: {best_metrics[0]:.6e}")
    
    return best_params, best_metrics

def evaluate(R_true, R_pred):
    """Enhanced evaluation metrics"""
    y_true, y_pred = R_true.flatten(), R_pred.flatten()
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0,1]
    
    # Additional metrics
    directional_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
    
    return mse, r2, corr, directional_accuracy

def main_enhanced(aaii_path, google_path, umich_path, returns_path):
    """Run enhanced analysis with best parameters"""
    print("Running enhanced multivariate mapping analysis...")
    
    # Find best parameters
    best_params, best_metrics = grid_search_enhanced(aaii_path, google_path, umich_path, returns_path)
    
    # Run final analysis with best parameters
    n_lags, m_horizon, method, alpha = best_params
    
    df = load_and_merge(aaii_path, google_path, umich_path, returns_path)
    B, R, T_prime = create_design_return_enhanced(df, n_lags, m_horizon)
    M = estimate_transformation_enhanced(B, R, method, alpha)
    R_hat = M @ B
    
    mse, r2, corr, directional_accuracy = evaluate(R, R_hat)
    
    print(f"\n=== Final Results with Best Parameters ===")
    print(f"Parameters: n_lags={n_lags}, m_horizon={m_horizon}, method={method}, alpha={alpha}")
    print(f"R²: {r2:.4f}")
    print(f"Correlation: {corr:.4f}")
    print(f"MSE: {mse:.6e}")
    print(f"Directional Accuracy: {directional_accuracy:.4f}")
    
    # Feature importance analysis
    if method in ['ridge', 'lasso', 'elastic_net']:
        print(f"\nTop 10 Feature Coefficients:")
        feature_names = [f"Feature_{i}" for i in range(M.shape[1])]
        coef_importance = np.abs(M[0, :])  # Use first horizon
        top_indices = np.argsort(coef_importance)[-10:]
        for idx in reversed(top_indices):
            print(f"  {feature_names[idx]}: {M[0, idx]:.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Enhanced multivariate behavioural indicators → market returns mapping"
    )
    parser.add_argument("--aaii", default="raw_data/aaii_sentiment_processed.csv")
    parser.add_argument("--google", default="raw_data/google_anxiety_processed.csv")
    parser.add_argument("--umich", default="raw_data/umich_sentiment_weekly.csv")
    parser.add_argument("--returns", default="raw_data/spy_weekly_returns.csv")
    
    args = parser.parse_args()
    main_enhanced(args.aaii, args.google, args.umich, args.returns) 