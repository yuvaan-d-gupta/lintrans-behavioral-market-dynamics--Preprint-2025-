

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
m_horizon = 12
ridge_lambda = 1.0

# Create lagged features
print("Creating lagged features...")
behavioral_cols = ['AAII_Sentiment', 'Google_Anxiety', 'UMich_Sentiment']
for col in behavioral_cols:
    for lag in range(1, n_lags + 1):
        merged_data[f'{col}_lag{lag}'] = merged_data[col].shift(lag)

# Create future returns for different horizons
print("Creating future returns...")
for horizon in range(1, m_horizon + 1):
    merged_data[f'return_h{horizon}'] = merged_data['Weekly_Return'].shift(-horizon)

# Drop rows with missing values after creating lags and future returns
merged_data = merged_data.dropna()
print(f"Dataset after creating features: {len(merged_data)} observations")

# Split data (80% train, 20% test)
split_idx = int(0.8 * len(merged_data))
train_data = merged_data.iloc[:split_idx]
test_data = merged_data.iloc[split_idx:]

print(f"Train set: {len(train_data)} observations")
print(f"Test set: {len(test_data)} observations")

# Function to calculate directional hit rate
def calculate_directional_hit_rate(y_true, y_pred):
    """Calculate percentage of correctly predicted signs."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    
    # Get signs
    true_signs = np.sign(y_true)
    pred_signs = np.sign(y_pred)
    
    # Calculate hit rate
    correct_predictions = np.sum(true_signs == pred_signs)
    total_predictions = len(y_true)
    
    return (correct_predictions / total_predictions) * 100

# Calculate directional hit rates for each horizon
print("Calculating directional hit rates...")
horizons = list(range(1, m_horizon + 1))
train_hit_rates = []
test_hit_rates = []

for horizon in horizons:
    print(f"Processing horizon {horizon}...")
    
    # Prepare features and target for this horizon
    feature_cols = []
    for col in behavioral_cols:
        for lag in range(1, n_lags + 1):
            feature_cols.append(f'{col}_lag{lag}')
    
    X_train = train_data[feature_cols].values
    y_train = train_data[f'return_h{horizon}'].values
    
    X_test = test_data[feature_cols].values
    y_test = test_data[f'return_h{horizon}'].values
    
    # Fit Ridge regression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
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
    
    # Calculate hit rates
    train_hit_rate = calculate_directional_hit_rate(y_train, y_train_pred)
    test_hit_rate = calculate_directional_hit_rate(y_test, y_test_pred)
    
    train_hit_rates.append(train_hit_rate)
    test_hit_rates.append(test_hit_rate)
    
    print(f"  Horizon {horizon}: Train = {train_hit_rate:.2f}%, Test = {test_hit_rate:.2f}%")

# Create the directional hit rate plot
print("Creating directional hit rate plot...")
plt.figure(figsize=(12, 8))

# Create bar chart
x_pos = np.arange(len(horizons))
width = 0.35

# Plot bars
bars1 = plt.bar(x_pos - width/2, train_hit_rates, width, label='Training Set', 
                color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
bars2 = plt.bar(x_pos + width/2, test_hit_rates, width, label='Test Set', 
                color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1)

# Add 50% baseline (random guessing)
plt.axhline(y=50, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
            label='Random Guessing (50%)')

# Add 55% significance line (commonly used threshold)
plt.axhline(y=55, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
            label='Significance Threshold (55%)')

# Customize the plot
plt.xlabel('Prediction Horizon (Weeks Ahead)', fontsize=14, fontweight='bold')
plt.ylabel('Directional Hit Rate (%)', fontsize=14, fontweight='bold')
plt.title('Directional Hit Rate by Prediction Horizon\n(Multivariate Ridge Regression Model)', 
          fontsize=16, fontweight='bold', pad=20)

plt.xticks(x_pos, [f'{h}' for h in horizons])
plt.ylim(40, 70)  # Focus on meaningful range
plt.grid(True, alpha=0.3, axis='y')
plt.legend(loc='upper right', fontsize=12)

# Add value labels on bars
def add_value_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

add_value_labels(bars1, train_hit_rates)
add_value_labels(bars2, test_hit_rates)

plt.tight_layout()

# Save the plot
plt.savefig(output_dir / 'directional_hit_rate_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'directional_hit_rate_plot.pdf', bbox_inches='tight')

print(f"Directional hit rate plot saved to:")
print(f"- {output_dir / 'directional_hit_rate_plot.png'}")
print(f"- {output_dir / 'directional_hit_rate_plot.pdf'}")

# Print summary statistics
print("\n=== Directional Hit Rate Summary ===")
print(f"{'Horizon':<8} {'Train (%)':<10} {'Test (%)':<10} {'Difference':<12}")
print("-" * 40)
for i, horizon in enumerate(horizons):
    diff = test_hit_rates[i] - train_hit_rates[i]
    print(f"{horizon:<8} {train_hit_rates[i]:<10.2f} {test_hit_rates[i]:<10.2f} {diff:<12.2f}")

# Calculate overall statistics
avg_train_hit_rate = np.mean(train_hit_rates)
avg_test_hit_rate = np.mean(test_hit_rates)
max_test_hit_rate = max(test_hit_rates)
max_horizon = horizons[np.argmax(test_hit_rates)]

print(f"\n=== Key Statistics ===")
print(f"Average Training Hit Rate: {avg_train_hit_rate:.2f}%")
print(f"Average Test Hit Rate: {avg_test_hit_rate:.2f}%")
print(f"Best Test Hit Rate: {max_test_hit_rate:.2f}% (Horizon {max_horizon})")
print(f"Horizons above 55%: {sum(1 for rate in test_hit_rates if rate > 55)}/{len(test_hit_rates)}")

# Save results to CSV
results_df = pd.DataFrame({
    'Horizon': horizons,
    'Train_Hit_Rate': train_hit_rates,
    'Test_Hit_Rate': test_hit_rates,
    'Difference': [test - train for train, test in zip(train_hit_rates, test_hit_rates)]
})

results_df.to_csv('results/tables/directional_hit_rates.csv', index=False)
print(f"\nDetailed results saved to: results/tables/directional_hit_rates.csv")

plt.show()
