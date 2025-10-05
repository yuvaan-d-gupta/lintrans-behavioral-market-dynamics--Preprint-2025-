

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def display_heatmap():
    """Display the coefficient heatmap using the exact transformation matrix values."""
    
    # Use the exact transformation matrix from our previous calculation
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
    
    # Set up matplotlib to use non-interactive backend
    plt.switch_backend('Agg')  # Use non-interactive backend
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap with annotations
    sns.heatmap(M_df, annot=True, fmt='.4f', cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Coefficient Value', 'shrink': 0.8},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_title('Coefficient Heatmap of Transformation Matrix M\n(Ridge Regression, λ = 1.0)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Behavioral Features (Indicator × Lag)', fontsize=14)
    ax.set_ylabel('Return Horizon (weeks ahead)', fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/figures/coefficient_heatmap_display.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/coefficient_heatmap_display.pdf', bbox_inches='tight')
    
    print("Coefficient Heatmap created successfully!")
    print("Files saved:")
    print("- results/figures/coefficient_heatmap_display.png")
    print("- results/figures/coefficient_heatmap_display.pdf")
    
    # Print analysis
    print("\n=== Top 10 Absolute Coefficients ===")
    flat_coeffs = M_df.stack().abs().sort_values(ascending=False)
    for i in range(min(10, len(flat_coeffs))):
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
    
    return M_df

if __name__ == "__main__":
    display_heatmap()
