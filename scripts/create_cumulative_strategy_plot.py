

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Create output directory
output_dir = Path('results/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the data
print("Loading data...")
aaii_data = pd.read_csv('raw_data/aaii_sentiment_processed.csv')
google_data = pd.read_csv('raw_data/google_anxiety_processed.csv')
umich_data = pd.read_csv('raw_data/umich_sentiment_weekly.csv')
returns_data = pd.read_csv('raw_data/spy_weekly_returns.csv')

# Convert dates
aaii_data['Date'] = pd.to_datetime(aaii_data['Date'])
google_data['Date'] = pd.to_datetime(google_data['Date'])
umich_data['Date'] = pd.to_datetime(umich_data['Date'])
returns_data['Date'] = pd.to_datetime(returns_data['Date'])

# Merge data
print("Merging data...")
merged_data = returns_data.copy()
merged_data = pd.merge_asof(merged_data, aaii_data, on='Date', direction='backward')
merged_data = pd.merge_asof(merged_data, google_data, on='Date', direction='backward')
merged_data = pd.merge_asof(merged_data, umich_data, on='Date', direction='backward')

# Drop rows with missing values
merged_data = merged_data.dropna()
print(f"Final dataset size: {len(merged_data)} observations")

# Parameters from the optimized model
n_lags = 4
ridge_lambda = 1.0

# Create lagged features
print("Creating lagged features...")
behavioral_cols = ['AAII_Sentiment', 'Google_Anxiety', 'UMich_Sentiment']
for col in behavioral_cols:
    for lag in range(1, n_lags + 1):
        merged_data[f'{col}_lag{lag}'] = merged_data[col].shift(lag)

# Create future returns for horizon 1 (next week)
merged_data['return_h1'] = merged_data['Weekly_Return'].shift(-1)

# Drop rows with missing values after creating lags and future returns
merged_data = merged_data.dropna()
print(f"Dataset after creating features: {len(merged_data)} observations")

# Split data (80% train, 20% test)
split_idx = int(0.8 * len(merged_data))
train_data = merged_data.iloc[:split_idx]
test_data = merged_data.iloc[split_idx:]

print(f"Train set: {len(train_data)} observations")
print(f"Test set: {len(test_data)} observations")

# Prepare features and target for horizon 1
feature_cols = []
for col in behavioral_cols:
    for lag in range(1, n_lags + 1):
        feature_cols.append(f'{col}_lag{lag}')

X_train = train_data[feature_cols].values
y_train = train_data['return_h1'].values

X_test = test_data[feature_cols].values
y_test = test_data['return_h1'].values

# Fit Ridge regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

print("Fitting Ridge regression model...")
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit model
model = Ridge(alpha=ridge_lambda)
model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Create trading strategy
def create_trading_strategy(predictions, actual_returns, dates):
    """Create a simple long/short trading strategy based on predictions."""
    strategy_returns = []
    
    for i in range(len(predictions)):
        if predictions[i] > 0:  # Predict positive return
            strategy_returns.append(actual_returns[i])  # Long position
        else:  # Predict negative return
            strategy_returns.append(-actual_returns[i])  # Short position
    
    return np.array(strategy_returns)

# Calculate strategy returns
print("Calculating trading strategy returns...")
train_strategy_returns = create_trading_strategy(y_train_pred, y_train, train_data['Date'])
test_strategy_returns = create_trading_strategy(y_test_pred, y_test, test_data['Date'])

# Calculate cumulative returns
def calculate_cumulative_returns(returns):
    """Calculate cumulative returns starting from 1."""
    return np.cumprod(1 + returns)

train_cumulative_strategy = calculate_cumulative_returns(train_strategy_returns)
test_cumulative_strategy = calculate_cumulative_returns(test_strategy_returns)

# Calculate SP500 cumulative returns
train_cumulative_sp500 = calculate_cumulative_returns(train_data['Weekly_Return'].values)
test_cumulative_sp500 = calculate_cumulative_returns(test_data['Weekly_Return'].values)

# Calculate performance metrics
def calculate_performance_metrics(returns):
    """Calculate Sharpe ratio and total return."""
    if len(returns) == 0:
        return 0, 0
    
    # Annualized Sharpe ratio (assuming weekly data, 52 weeks per year)
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(52) if np.std(returns) > 0 else 0
    
    # Total return
    total_return = np.prod(1 + returns) - 1
    
    return sharpe_ratio, total_return

train_sharpe, train_total_return = calculate_performance_metrics(train_strategy_returns)
test_sharpe, test_total_return = calculate_performance_metrics(test_strategy_returns)

print(f"Training Set - Sharpe Ratio: {train_sharpe:.4f}, Total Return: {train_total_return:.4f}")
print(f"Test Set - Sharpe Ratio: {test_sharpe:.4f}, Total Return: {test_total_return:.4f}")

# Create the cumulative performance plot
print("Creating cumulative strategy performance plot...")
plt.figure(figsize=(16, 10))

# Create time axis for training and test periods
train_dates = train_data['Date'].values
test_dates = test_data['Date'].values

# Plot training period
plt.subplot(2, 1, 1)
plt.plot(train_dates, train_cumulative_strategy, label='Trading Strategy', 
         color='blue', linewidth=2, alpha=0.8)
plt.plot(train_dates, train_cumulative_sp500, label='SP500 Benchmark', 
         color='red', linewidth=2, alpha=0.8)
plt.title('Cumulative Strategy Performance - Training Period', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Cumulative Return', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot test period
plt.subplot(2, 1, 2)
plt.plot(test_dates, test_cumulative_strategy, label='Trading Strategy', 
         color='blue', linewidth=2, alpha=0.8)
plt.plot(test_dates, test_cumulative_sp500, label='SP500 Benchmark', 
         color='red', linewidth=2, alpha=0.8)
plt.title('Cumulative Strategy Performance - Test Period', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Time (Weeks)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative Return', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()

# Save the plot
plt.savefig(output_dir / 'cumulative_strategy_performance.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'cumulative_strategy_performance.pdf', bbox_inches='tight')

print(f"Cumulative strategy performance plot saved to:")
print(f"- {output_dir / 'cumulative_strategy_performance.png'}")
print(f"- {output_dir / 'cumulative_strategy_performance.pdf'}")

# Create a combined plot for better visualization
plt.figure(figsize=(14, 8))

# Combine training and test periods
all_dates = np.concatenate([train_dates, test_dates])
all_strategy = np.concatenate([train_cumulative_strategy, test_cumulative_strategy])
all_sp500 = np.concatenate([train_cumulative_sp500, test_cumulative_sp500])

# Plot combined performance
plt.plot(all_dates, all_strategy, label='Trading Strategy', 
         color='blue', linewidth=2.5, alpha=0.9)
plt.plot(all_dates, all_sp500, label='SP500 Benchmark', 
         color='red', linewidth=2.5, alpha=0.9)

# Add vertical line to separate train and test
split_date = test_dates[0]
plt.axvline(x=split_date, color='gray', linestyle='--', alpha=0.7, 
            label='Train/Test Split')

# Add performance annotations
plt.annotate(f'Train: Sharpe={train_sharpe:.3f}\nReturn={train_total_return:.3f}', 
            xy=(train_dates[len(train_dates)//2], train_cumulative_strategy[len(train_cumulative_strategy)//2]),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
            fontsize=10, fontweight='bold')

plt.annotate(f'Test: Sharpe={test_sharpe:.3f}\nReturn={test_total_return:.3f}', 
            xy=(test_dates[len(test_dates)//2], test_cumulative_strategy[len(test_cumulative_strategy)//2]),
            xytext=(10, 10), textcoords='offset points',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            fontsize=10, fontweight='bold')

plt.title('Cumulative Strategy Performance Curve\n(Multivariate Ridge Regression Trading Strategy)', 
          fontsize=18, fontweight='bold', pad=25)
plt.xlabel('Time (Weeks)', fontsize=16, fontweight='bold')
plt.ylabel('Cumulative Return', fontsize=16, fontweight='bold')
plt.legend(fontsize=14, loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()

# Save the combined plot
plt.savefig(output_dir / 'cumulative_strategy_combined.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'cumulative_strategy_combined.pdf', bbox_inches='tight')

print(f"Combined cumulative strategy performance plot saved to:")
print(f"- {output_dir / 'cumulative_strategy_combined.png'}")
print(f"- {output_dir / 'cumulative_strategy_combined.pdf'}")

# Print detailed performance summary
print("\n=== Trading Strategy Performance Summary ===")
print(f"{'Period':<12} {'Sharpe Ratio':<15} {'Total Return':<15} {'Max Drawdown':<15}")
print("-" * 60)

# Calculate max drawdown
def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown."""
    peak = cumulative_returns[0]
    max_dd = 0
    
    for value in cumulative_returns:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)
    
    return max_dd

train_max_dd = calculate_max_drawdown(train_cumulative_strategy)
test_max_dd = calculate_max_drawdown(test_cumulative_strategy)

print(f"{'Training':<12} {train_sharpe:<15.4f} {train_total_return:<15.4f} {train_max_dd:<15.4f}")
print(f"{'Test':<12} {test_sharpe:<15.4f} {test_total_return:<15.4f} {test_max_dd:<15.4f}")

# Calculate SP500 performance for comparison
train_sp500_sharpe, train_sp500_return = calculate_performance_metrics(train_data['Weekly_Return'].values)
test_sp500_sharpe, test_sp500_return = calculate_performance_metrics(test_data['Weekly_Return'].values)

print(f"\n=== SP500 Benchmark Performance ===")
print(f"{'Period':<12} {'Sharpe Ratio':<15} {'Total Return':<15} {'Max Drawdown':<15}")
print("-" * 60)

train_sp500_max_dd = calculate_max_drawdown(train_cumulative_sp500)
test_sp500_max_dd = calculate_max_drawdown(test_cumulative_sp500)

print(f"{'Training':<12} {train_sp500_sharpe:<15.4f} {train_sp500_return:<15.4f} {train_sp500_max_dd:<15.4f}")
print(f"{'Test':<12} {test_sp500_sharpe:<15.4f} {test_sp500_return:<15.4f} {test_sp500_max_dd:<15.4f}")

# Save detailed results to CSV
results_df = pd.DataFrame({
    'Period': ['Training', 'Test'],
    'Strategy_Sharpe': [train_sharpe, test_sharpe],
    'Strategy_Total_Return': [train_total_return, test_total_return],
    'Strategy_Max_Drawdown': [train_max_dd, test_max_dd],
    'SP500_Sharpe': [train_sp500_sharpe, test_sp500_sharpe],
    'SP500_Total_Return': [train_sp500_return, test_sp500_return],
    'SP500_Max_Drawdown': [train_sp500_max_dd, test_sp500_max_dd]
})

results_df.to_csv('results/tables/strategy_performance.csv', index=False)
print(f"\nDetailed performance results saved to: results/tables/strategy_performance.csv")

plt.show()
