

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

def load_and_merge(aaii_path, google_path, umich_path, returns_path, tol_days=7):
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
    """Create enhanced features with key improvements"""
    features = []
    
    # Original series with lags
    for col in ['AAII', 'Google', 'UMich']:
        series = df[col].values
        for lag in range(1, n_lags + 1):
            features.append(np.roll(series, lag))
    
    # Key engineered features
    aaii = df['AAII'].values
    google = df['Google'].values
    umich = df['UMich'].values
    
    # Rolling statistics (4-week window)
    for col in ['AAII', 'Google', 'UMich']:
        series = df[col].values
        rolling_mean = pd.Series(series).rolling(window=4, min_periods=1).mean().values
        rolling_std = pd.Series(series).rolling(window=4, min_periods=1).std().values
        features.extend([rolling_mean, rolling_std])
    
    # Momentum features
    for col in ['AAII', 'Google', 'UMich']:
        series = df[col].values
        momentum_4w = series - np.roll(series, 4)
        momentum_8w = series - np.roll(series, 8)
        features.extend([momentum_4w, momentum_8w])
    
    # Cross-series interactions
    features.append(aaii * google)  # AAII × Google
    features.append(aaii * umich)   # AAII × UMich
    features.append(google * umich) # Google × UMich
    
    # Sentiment divergence
    sentiment_divergence = aaii - umich
    features.append(sentiment_divergence)
    
    # Anxiety-sentiment ratio
    anxiety_sentiment_ratio = google / ((aaii + umich) / 2 + 1e-8)
    features.append(anxiety_sentiment_ratio)
    
    # Non-linear transformations
    features.append(np.log(np.abs(aaii) + 1e-8))
    features.append(np.log(np.abs(google) + 1e-8))
    features.append(np.log(np.abs(umich) + 1e-8))
    
    # Squared terms
    features.append(aaii ** 2)
    features.append(google ** 2)
    features.append(umich ** 2)
    
    return np.array(features).T

def evaluate_model(X, y, model):
    """Evaluate model performance"""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    corr = np.corrcoef(y, y_pred)[0,1]
    directional_accuracy = np.mean(np.sign(y) == np.sign(y_pred))
    return mse, r2, corr, directional_accuracy

def quick_grid_search(df, n_lags_list=[1,2,3,4], methods=['ridge', 'lasso', 'random_forest']):
    """Quick grid search with key parameters"""
    best_r2 = -np.inf
    best_corr = -np.inf
    best_params = None
    best_results = None
    
    print("=== Quick Enhanced Grid Search ===")
    
    for n_lags in n_lags_list:
        # Create features
        features = create_enhanced_features(df, n_lags)
        
        # Prepare target (next week's return)
        returns = df['Return'].values
        y = returns[n_lags:]  # Target: next week's return
        X = features[n_lags:-1]  # Features up to current week
        
        if len(X) < 20:  # Need sufficient data
            continue
        
        for method in methods:
            try:
                if method == 'ridge':
                    for alpha in [0.0, 0.1, 1.0, 10.0]:
                        model = Ridge(alpha=alpha)
                        model.fit(X, y)
                        mse, r2, corr, dir_acc = evaluate_model(X, y, model)
                        
                        print(f"n_lags={n_lags}, method={method}, alpha={alpha} | R²={r2:.4f}, Corr={corr:.4f}, DirAcc={dir_acc:.4f}")
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_params = (n_lags, method, alpha)
                            best_results = (mse, r2, corr, dir_acc)
                        
                        if corr > best_corr:
                            best_corr = corr
                
                elif method == 'lasso':
                    for alpha in [0.001, 0.01, 0.1, 1.0]:
                        model = Lasso(alpha=alpha)
                        model.fit(X, y)
                        mse, r2, corr, dir_acc = evaluate_model(X, y, model)
                        
                        print(f"n_lags={n_lags}, method={method}, alpha={alpha} | R²={r2:.4f}, Corr={corr:.4f}, DirAcc={dir_acc:.4f}")
                        
                        if r2 > best_r2:
                            best_r2 = r2
                            best_params = (n_lags, method, alpha)
                            best_results = (mse, r2, corr, dir_acc)
                
                elif method == 'random_forest':
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    mse, r2, corr, dir_acc = evaluate_model(X, y, model)
                    
                    print(f"n_lags={n_lags}, method={method} | R²={r2:.4f}, Corr={corr:.4f}, DirAcc={dir_acc:.4f}")
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = (n_lags, method, 0)
                        best_results = (mse, r2, corr, dir_acc)
            
            except Exception as e:
                print(f"Error with n_lags={n_lags}, method={method}: {e}")
    
    return best_params, best_results, best_corr

def main():
    """Run enhanced analysis"""
    print("Loading data...")
    df = load_and_merge(
        "raw_data/aaii_sentiment_processed.csv",
        "raw_data/google_anxiety_processed.csv", 
        "raw_data/umich_sentiment_weekly.csv",
        "raw_data/spy_weekly_returns.csv"
    )
    
    print(f"Data loaded: {len(df)} observations")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Run grid search
    best_params, best_results, best_corr = quick_grid_search(df)
    
    print(f"\n=== Best Results ===")
    print(f"Best R²: {best_results[1]:.4f}")
    print(f"Best Correlation: {best_corr:.4f}")
    print(f"Best Directional Accuracy: {best_results[3]:.4f}")
    print(f"Best Parameters: n_lags={best_params[0]}, method={best_params[1]}, alpha={best_params[2]}")
    
    # Show improvement over baseline
    print(f"\n=== Improvement Analysis ===")
    print(f"R² improvement: {best_results[1]:.4f} (baseline ~0.005)")
    print(f"Correlation improvement: {best_corr:.4f} (baseline ~0.07)")
    print(f"Directional accuracy: {best_results[3]:.4f}")
    
    # Feature importance for best model
    if best_params[1] in ['ridge', 'lasso']:
        print(f"\n=== Feature Importance ===")
        features = create_enhanced_features(df, best_params[0])
        returns = df['Return'].values
        y = returns[best_params[0]:]
        X = features[best_params[0]:-1]
        
        if best_params[1] == 'ridge':
            model = Ridge(alpha=best_params[2])
        else:
            model = Lasso(alpha=best_params[2])
        
        model.fit(X, y)
        coef_importance = np.abs(model.coef_)
        top_indices = np.argsort(coef_importance)[-10:]
        
        feature_names = [
            'AAII_lag1', 'AAII_lag2', 'AAII_lag3', 'AAII_lag4',
            'Google_lag1', 'Google_lag2', 'Google_lag3', 'Google_lag4', 
            'UMich_lag1', 'UMich_lag2', 'UMich_lag3', 'UMich_lag4',
            'AAII_rolling_mean', 'AAII_rolling_std', 'Google_rolling_mean', 'Google_rolling_std',
            'UMich_rolling_mean', 'UMich_rolling_std', 'AAII_momentum_4w', 'AAII_momentum_8w',
            'Google_momentum_4w', 'Google_momentum_8w', 'UMich_momentum_4w', 'UMich_momentum_8w',
            'AAII_Google_interaction', 'AAII_UMich_interaction', 'Google_UMich_interaction',
            'Sentiment_divergence', 'Anxiety_sentiment_ratio', 'log_AAII', 'log_Google', 'log_UMich',
            'AAII_squared', 'Google_squared', 'UMich_squared'
        ]
        
        print("Top 10 most important features:")
        for idx in reversed(top_indices):
            if idx < len(feature_names):
                print(f"  {feature_names[idx]}: {model.coef_[idx]:.6f}")

if __name__ == "__main__":
    main() 