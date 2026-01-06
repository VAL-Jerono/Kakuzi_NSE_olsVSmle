"""
Generate individual plot files for LaTeX document
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Generating individual plots for LaTeX document...")
print()

# ===== QUESTION 1 PLOTS =====
print("Question 1 Plots...")

# Example data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
n = len(X)
k = 2

# OLS estimation
X_mean = np.mean(X)
Y_mean = np.mean(Y)
beta_1_ols = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)
beta_0_ols = Y_mean - beta_1_ols * X_mean
Y_pred = beta_0_ols + beta_1_ols * X
residuals = Y - Y_pred
SSR = np.sum(residuals**2)
sigma2_ols = SSR / (n - k)
sigma2_mle = SSR / n
ratio = sigma2_ols / sigma2_mle

# Plot 1: OLS Regression Fit
fig = plt.figure(figsize=(9, 6))
plt.scatter(X, Y, s=120, alpha=0.7, edgecolors='black', linewidths=2, label='Data points', c='#3498db')
X_line = np.linspace(X.min()-0.5, X.max()+0.5, 100)
Y_line = beta_0_ols + beta_1_ols * X_line
plt.plot(X_line, Y_line, 'r-', linewidth=2.5, label=f'Y = {beta_0_ols:.2f} + {beta_1_ols:.2f}X')
for i in range(n):
    plt.plot([X[i], X[i]], [Y[i], Y_pred[i]], 'g--', alpha=0.6, linewidth=1.8)
plt.xlabel('X', fontsize=13, fontweight='bold')
plt.ylabel('Y', fontsize=13, fontweight='bold')
plt.title('OLS Regression Fit', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q1_regression_fit.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q1_regression_fit.png")

# Plot 2: Variance Comparison
fig = plt.figure(figsize=(8, 6))
methods = ['OLS', 'MLE']
variances = [sigma2_ols, sigma2_mle]
colors = ['#e74c3c', '#3498db']
bars = plt.bar(methods, variances, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
plt.ylabel('Variance Estimate (σ̂²)', fontsize=13, fontweight='bold')
plt.title('OLS vs MLE Variance Estimates', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for bar, var in zip(bars, variances):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
            f'{var:.6f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.text(0.5, (sigma2_ols + sigma2_mle)/2, 
        f'Ratio = {ratio:.3f}', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
plt.tight_layout()
plt.savefig('q1_variance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q1_variance_comparison.png")

# ===== QUESTION 2 PLOTS =====
print("\nQuestion 2 Plots...")

# Load data
df = pd.read_csv('kakuzi.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

# Compute log returns
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
df = df.dropna(subset=['Log_Return'])

# Compute volatility
window = 30
df['Volatility'] = df['Log_Return'].rolling(window=window).std()
df_analysis = df.dropna(subset=['Volatility']).copy()

# Plot 1: Price Series
fig = plt.figure(figsize=(11, 5))
plt.plot(df['Date'], df['Close'], linewidth=1.5, color='#2c3e50')
plt.title('Kakuzi Stock Price (Jan 2021 - Jan 2026)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price (KES)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('q2_price_series.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_price_series.png")

# Plot 2: Log Returns
fig = plt.figure(figsize=(11, 5))
plt.plot(df['Date'], df['Log_Return'], linewidth=0.9, color='#8e44ad', alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.6)
plt.title('Daily Log Returns', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Log Return', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('q2_log_returns.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_log_returns.png")

# Plot 3: Volatility
fig = plt.figure(figsize=(11, 5))
plt.plot(df['Date'], df['Volatility'], linewidth=1.5, color='#e67e22')
plt.title('Rolling Volatility (30-day window)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility (Std Dev)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('q2_volatility.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_volatility.png")

# Plot 4: Returns Histogram
fig = plt.figure(figsize=(10, 6))
plt.hist(df['Log_Return'].dropna(), bins=50, density=True, alpha=0.75, 
        color='#16a085', edgecolor='black', linewidth=1.2)
mu, sigma = df['Log_Return'].mean(), df['Log_Return'].std()
x = np.linspace(df['Log_Return'].min(), df['Log_Return'].max(), 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=3, 
        label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
plt.xlabel('Log Return', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Log Returns', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_returns_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_returns_histogram.png")

# Plot 5: Q-Q Plot
fig = plt.figure(figsize=(8, 6))
stats.probplot(df['Log_Return'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot for Normality Test', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_qq_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_qq_plot.png")

# OLS Estimation
y = df_analysis['Log_Return'].values
X = df_analysis['Volatility'].values
X_with_const = sm.add_constant(X)
model_ols = sm.OLS(y, X_with_const)
results_ols = model_ols.fit()
beta_0 = results_ols.params[0]
beta_1 = results_ols.params[1]

# Plot 6: OLS Scatter
fig = plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, s=35, edgecolors='black', linewidths=0.6, c='#3498db')
X_line = np.linspace(X.min(), X.max(), 100)
y_line = beta_0 + beta_1 * X_line
plt.plot(X_line, y_line, 'r-', linewidth=3, 
        label=f'Return = {beta_0:.4f} + {beta_1:.4f}×Volatility')
plt.xlabel('Volatility (Risk)', fontsize=12, fontweight='bold')
plt.ylabel('Log Return', fontsize=12, fontweight='bold')
plt.title('Risk-Return Relationship (OLS)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_ols_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_ols_scatter.png")

# Plot 7: OLS Residuals
fig = plt.figure(figsize=(10, 6))
residuals = results_ols.resid
plt.scatter(results_ols.fittedvalues, residuals, alpha=0.5, s=35, 
           edgecolors='black', linewidths=0.6, c='#e74c3c')
plt.axhline(y=0, color='blue', linestyle='--', linewidth=2.5)
plt.xlabel('Fitted Values', fontsize=12, fontweight='bold')
plt.ylabel('Residuals', fontsize=12, fontweight='bold')
plt.title('Residual Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_ols_residuals.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_ols_residuals.png")

# Diagnostic plots
residuals = results_ols.resid

# Plot 8: Diagnostic - Residuals vs Fitted
fig = plt.figure(figsize=(9, 6))
plt.scatter(results_ols.fittedvalues, residuals, alpha=0.5, s=35, c='#9b59b6')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Fitted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Diagnostic: Residuals vs Fitted Values', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_diagnostic_residuals_fitted.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_diagnostic_residuals_fitted.png")

# Plot 9: Diagnostic - Q-Q
fig = plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Diagnostic: Normal Q-Q Plot of Residuals', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_diagnostic_qq.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_diagnostic_qq.png")

# Plot 10: Diagnostic - Scale-Location
fig = plt.figure(figsize=(9, 6))
standardized_resid = (residuals - residuals.mean()) / residuals.std()
plt.scatter(results_ols.fittedvalues, np.sqrt(np.abs(standardized_resid)), 
           alpha=0.5, s=35, c='#f39c12')
plt.xlabel('Fitted Values', fontsize=12)
plt.ylabel('√|Standardized Residuals|', fontsize=12)
plt.title('Diagnostic: Scale-Location Plot', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_diagnostic_scale_location.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_diagnostic_scale_location.png")

# Plot 11: Diagnostic - Histogram
fig = plt.figure(figsize=(9, 6))
plt.hist(residuals, bins=40, density=True, alpha=0.75, 
        color='#1abc9c', edgecolor='black')
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=3)
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Diagnostic: Distribution of Residuals', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_diagnostic_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ q2_diagnostic_histogram.png")

print("\n✅ All plots generated successfully!")
print("\nGenerated files:")
print("  Question 1: q1_regression_fit.png, q1_variance_comparison.png")
print("  Question 2: q2_price_series.png, q2_log_returns.png, q2_volatility.png")
print("            q2_returns_histogram.png, q2_qq_plot.png")
print("            q2_ols_scatter.png, q2_ols_residuals.png")
print("            q2_diagnostic_*.png (4 diagnostic plots)")
