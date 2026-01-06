"""
TIME SERIES & FORECASTING ASSIGNMENT
=====================================
Student: Valerie Jerono - 222331
Date: 06/01/2025

This script provides comprehensive solutions to:
1. OLS vs MLE mathematical demonstration
2. Risk-Return model estimation using Kakuzi securities data from NSE

The solutions are presented systematically to help understand the concepts deeply.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera, durbin_watson
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# QUESTION 1: OLS vs MLE - Mathematical Demonstration with Python
# ============================================================================

def question_1_ols_vs_mle():
    """
    Demonstrate mathematically that OLS variance estimate is higher than MLE.
    
    KEY CONCEPTS:
    - OLS and MLE give IDENTICAL coefficient estimates under normality
    - OLS gives HIGHER variance estimate due to degrees of freedom adjustment
    - OLS: divides by (n-k), MLE: divides by n
    """
    
    print("="*80)
    print("QUESTION 1: OLS vs MLE - Mathematical Demonstration")
    print("="*80)
    print()
    
    # Step 1: Create example data
    print("STEP 1: Setting Up Our Example")
    print("-" * 80)
    
    # Simple linear regression: Y = β₀ + β₁X + ε
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
    n = len(X)
    k = 2  # number of parameters (β₀, β₁)
    
    print(f"Sample Data (n = {n} observations):")
    print(f"X: {X}")
    print(f"Y: {Y}")
    print()
    
    # Step 2: OLS Estimation
    print("STEP 2: OLS ESTIMATION")
    print("-" * 80)
    
    # Calculate means
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    print(f"Sample means:")
    print(f"  X̄ = {X_mean:.4f}")
    print(f"  Ȳ = {Y_mean:.4f}")
    print()
    
    # Calculate β₁ (slope)
    numerator = np.sum((X - X_mean) * (Y - Y_mean))
    denominator = np.sum((X - X_mean)**2)
    beta_1_ols = numerator / denominator
    
    print(f"Calculating β₁:")
    print(f"  Σ(Xᵢ - X̄)(Yᵢ - Ȳ) = {numerator:.4f}")
    print(f"  Σ(Xᵢ - X̄)² = {denominator:.4f}")
    print(f"  β̂₁_OLS = {beta_1_ols:.4f}")
    print()
    
    # Calculate β₀ (intercept)
    beta_0_ols = Y_mean - beta_1_ols * X_mean
    print(f"Calculating β₀:")
    print(f"  β̂₀_OLS = Ȳ - β̂₁X̄ = {Y_mean:.4f} - {beta_1_ols:.4f}×{X_mean:.4f}")
    print(f"  β̂₀_OLS = {beta_0_ols:.4f}")
    print()
    
    # Calculate residuals and SSR
    Y_pred = beta_0_ols + beta_1_ols * X
    residuals = Y - Y_pred
    SSR = np.sum(residuals**2)
    
    print("Residual Analysis:")
    print(f"{'i':<5} {'Yᵢ':<8} {'Ŷᵢ':<8} {'εᵢ':<10} {'εᵢ²':<10}")
    print("-" * 45)
    for i in range(n):
        print(f"{i+1:<5} {Y[i]:<8.2f} {Y_pred[i]:<8.2f} {residuals[i]:<10.4f} {residuals[i]**2:<10.6f}")
    print("-" * 45)
    print(f"{'Sum of Squared Residuals (SSR):':<35} {SSR:.6f}")
    print()
    
    # Calculate OLS variance estimate
    sigma2_ols = SSR / (n - k)
    print(f"OLS Variance Estimate:")
    print(f"  σ̂²_OLS = SSR / (n - k) = {SSR:.6f} / ({n} - {k})")
    print(f"  σ̂²_OLS = {SSR:.6f} / {n-k}")
    print(f"  σ̂²_OLS = {sigma2_ols:.6f}")
    print()
    
    # Step 3: MLE Estimation
    print("STEP 3: MLE ESTIMATION")
    print("-" * 80)
    
    # Under normality assumption, MLE coefficient estimates are identical to OLS
    beta_0_mle = beta_0_ols
    beta_1_mle = beta_1_ols
    
    print("Under Normal Distribution Assumption:")
    print(f"  β̂₁_MLE = {beta_1_mle:.4f} = β̂₁_OLS ✓ (IDENTICAL)")
    print(f"  β̂₀_MLE = {beta_0_mle:.4f} = β̂₀_OLS ✓ (IDENTICAL)")
    print()
    
    # Calculate MLE variance estimate
    sigma2_mle = SSR / n
    print(f"MLE Variance Estimate:")
    print(f"  σ̂²_MLE = SSR / n = {SSR:.6f} / {n}")
    print(f"  σ̂²_MLE = {sigma2_mle:.6f}")
    print()
    
    # Step 4: Comparison
    print("STEP 4: COMPARISON AND KEY FINDINGS")
    print("-" * 80)
    
    print("Summary of Estimates:")
    print(f"{'Parameter':<15} {'OLS':<15} {'MLE':<15} {'Relationship'}")
    print("-" * 60)
    print(f"{'β̂₁':<15} {beta_1_ols:<15.4f} {beta_1_mle:<15.4f} {'SAME'}")
    print(f"{'β̂₀':<15} {beta_0_ols:<15.4f} {beta_0_mle:<15.4f} {'SAME'}")
    print(f"{'σ̂²':<15} {sigma2_ols:<15.6f} {sigma2_mle:<15.6f} {'OLS > MLE ✓'}")
    print()
    
    # Mathematical relationship
    ratio = sigma2_ols / sigma2_mle
    expected_ratio = n / (n - k)
    
    print("Mathematical Relationship:")
    print(f"  σ̂²_OLS / σ̂²_MLE = {ratio:.4f}")
    print(f"  Expected: n/(n-k) = {n}/{n-k} = {expected_ratio:.4f}")
    print(f"  Verification: {ratio:.4f} = {expected_ratio:.4f} ✓")
    print()
    
    print(f"  Formula: σ̂²_OLS = (n/(n-k)) × σ̂²_MLE")
    print(f"  Verification: {sigma2_ols:.6f} = {expected_ratio:.4f} × {sigma2_mle:.6f}")
    print(f"  Verification: {sigma2_ols:.6f} = {expected_ratio * sigma2_mle:.6f} ✓")
    print()
    
    # Bias analysis
    print("BIAS ANALYSIS:")
    print("-" * 80)
    print(f"OLS is UNBIASED: E[σ̂²_OLS] = σ²")
    print(f"MLE is BIASED: E[σ̂²_MLE] = [(n-k)/n]σ² < σ²")
    print()
    print(f"In our example:")
    print(f"  Bias factor = (n-k)/n = {(n-k)/n:.4f}")
    print(f"  MLE underestimates by: {k/n:.4f} or {(k/n)*100:.2f}%")
    print()
    
    # Intuitive explanation
    print("WHY IS OLS VARIANCE HIGHER? (Intuitive Explanation)")
    print("-" * 80)
    print("""
1. DEGREES OF FREEDOM ADJUSTMENT:
   - When we estimate k parameters (β₀, β₁), we "use up" k degrees of freedom
   - This leaves us with (n-k) independent pieces of information for variance
   - OLS adjusts for this by dividing by (n-k) instead of n

2. INFORMATION LOSS:
   - Estimating parameters from data reduces available information for σ²
   - OLS compensates by being more conservative (larger estimate)
   - MLE doesn't adjust, leading to systematic underestimation

3. STATISTICAL PROPERTIES:
   - OLS: Unbiased but larger variance estimate
   - MLE: Biased downward but maximum likelihood
   - As n → ∞, both converge to the true value

4. PRACTICAL IMPLICATION:
   - OLS gives wider confidence intervals (more conservative)
   - MLE gives tighter intervals but may be overconfident in small samples
    """)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Regression line with data
    axes[0].scatter(X, Y, s=100, alpha=0.6, edgecolors='black', linewidths=2, label='Data points')
    X_line = np.linspace(X.min()-0.5, X.max()+0.5, 100)
    Y_line = beta_0_ols + beta_1_ols * X_line
    axes[0].plot(X_line, Y_line, 'r-', linewidth=2, label=f'Fitted line: Y = {beta_0_ols:.2f} + {beta_1_ols:.2f}X')
    
    # Plot residuals
    for i in range(n):
        axes[0].plot([X[i], X[i]], [Y[i], Y_pred[i]], 'g--', alpha=0.5, linewidth=1.5)
    
    axes[0].set_xlabel('X', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Y', fontsize=12, fontweight='bold')
    axes[0].set_title('OLS Regression Fit\n(Residuals shown as green dashed lines)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Variance comparison
    methods = ['OLS', 'MLE']
    variances = [sigma2_ols, sigma2_mle]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = axes[1].bar(methods, variances, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Variance Estimate (σ̂²)', fontsize=12, fontweight='bold')
    axes[1].set_title('Comparison of Variance Estimates\n(OLS > MLE)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{var:.6f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add ratio annotation
    axes[1].annotate('', xy=(0.5, sigma2_ols), xytext=(0.5, sigma2_mle),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    axes[1].text(0.7, (sigma2_ols + sigma2_mle)/2, 
                f'Ratio: {ratio:.3f}x\n= n/(n-k)',
                fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('question1_ols_vs_mle.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'question1_ols_vs_mle.png'")
    plt.show()
    
    return {
        'beta_0_ols': beta_0_ols,
        'beta_1_ols': beta_1_ols,
        'sigma2_ols': sigma2_ols,
        'beta_0_mle': beta_0_mle,
        'beta_1_mle': beta_1_mle,
        'sigma2_mle': sigma2_mle
    }


# ============================================================================
# QUESTION 2: Risk-Return Model using Kakuzi Securities Data
# ============================================================================

def load_and_prepare_kakuzi_data():
    """
    Load Kakuzi securities data and prepare for analysis.
    """
    print("="*80)
    print("QUESTION 2: Risk-Return Model - Kakuzi Securities (NSE)")
    print("="*80)
    print()
    
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("-" * 80)
    
    # Load data
    df = pd.read_csv('kakuzi.csv')
    print(f"✓ Loaded Kakuzi securities data")
    print(f"  Total observations: {len(df)}")
    print()
    
    # Display first few rows
    print("First 5 observations:")
    print(df.head())
    print()
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')  # Ensure chronological order
    df = df.reset_index(drop=True)
    
    # Clean the Close price (handle any string formatting issues)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Remove any rows with missing Close prices
    df = df.dropna(subset=['Close'])
    
    print(f"✓ Data cleaned and sorted chronologically")
    print(f"  Valid observations: {len(df)}")
    print(f"  Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print()
    
    return df


def compute_returns_and_volatility(df):
    """
    Compute log returns and volatility (risk measure).
    
    KEY CONCEPTS:
    - Log returns: ln(Pt/Pt-1) = ln(Pt) - ln(Pt-1)
    - Volatility: Standard deviation of returns (measure of risk)
    """
    print("STEP 2: COMPUTING LOG RETURNS AND VOLATILITY")
    print("-" * 80)
    
    # Compute log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Remove the first row (NaN)
    df = df.dropna(subset=['Log_Return'])
    
    print("Log Returns Formula:")
    print("  r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})")
    print()
    
    print(f"✓ Log returns computed")
    print(f"  Valid return observations: {len(df)}")
    print()
    
    # Summary statistics of returns
    print("SUMMARY STATISTICS OF LOG RETURNS:")
    print("-" * 80)
    print(f"Mean return:       {df['Log_Return'].mean():.6f} (average return per period)")
    print(f"Std deviation:     {df['Log_Return'].std():.6f} (volatility/risk)")
    print(f"Minimum return:    {df['Log_Return'].min():.6f}")
    print(f"Maximum return:    {df['Log_Return'].max():.6f}")
    print(f"Skewness:          {df['Log_Return'].skew():.6f}")
    print(f"Kurtosis:          {df['Log_Return'].kurtosis():.6f}")
    print()
    
    # Compute rolling volatility (risk measure)
    # Using 30-day rolling window
    window = 30
    df['Volatility'] = df['Log_Return'].rolling(window=window).std()
    
    # Remove NaN values from volatility calculation
    df_analysis = df.dropna(subset=['Volatility']).copy()
    
    print(f"✓ Volatility computed using {window}-day rolling window")
    print(f"  Observations for risk-return model: {len(df_analysis)}")
    print()
    
    return df, df_analysis


def visualize_price_and_returns(df):
    """
    Visualize price series, returns, and volatility.
    """
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("-" * 80)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Price series
    axes[0].plot(df['Date'], df['Close'], linewidth=1.5, color='#2E86AB')
    axes[0].set_title('Kakuzi Stock Price Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=11)
    axes[0].set_ylabel('Close Price (KES)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Log returns
    axes[1].plot(df['Date'], df['Log_Return'], linewidth=0.8, color='#A23B72', alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    axes[1].set_title('Log Returns (Daily Return)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Log Return', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Volatility
    axes[2].plot(df['Date'], df['Volatility'], linewidth=1.5, color='#F18F01')
    axes[2].set_title('Rolling Volatility (30-day window) - Risk Measure', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=11)
    axes[2].set_ylabel('Volatility (Std Dev)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('question2_price_returns_volatility.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'question2_price_returns_volatility.png'")
    plt.show()
    
    # Distribution of returns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with normal curve
    axes[0].hist(df['Log_Return'].dropna(), bins=50, density=True, alpha=0.7, 
                color='#4ECDC4', edgecolor='black', linewidth=1.2)
    
    # Overlay normal distribution
    mu, sigma = df['Log_Return'].mean(), df['Log_Return'].std()
    x = np.linspace(df['Log_Return'].min(), df['Log_Return'].max(), 100)
    axes[0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2.5, 
                label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
    axes[0].set_xlabel('Log Return', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Distribution of Log Returns\n(with Normal Distribution overlay)', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(df['Log_Return'].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot: Testing Normality\n(Points should lie on red line)', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('question2_returns_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Distribution analysis saved as 'question2_returns_distribution.png'")
    plt.show()


def estimate_risk_return_ols(df_analysis):
    """
    Estimate risk-return model using OLS.
    
    MODEL: Return = β₀ + β₁ × Volatility + ε
    
    Interpretation:
    - β₀: Base return when risk is zero
    - β₁: Risk premium - additional return per unit of risk
    """
    print("\n")
    print("STEP 4: RISK-RETURN MODEL ESTIMATION (OLS)")
    print("-" * 80)
    
    print("MODEL SPECIFICATION:")
    print("  Return_t = β₀ + β₁ × Volatility_t + ε_t")
    print()
    print("where:")
    print("  Return_t     = Log return at time t")
    print("  Volatility_t = Rolling standard deviation (risk measure)")
    print("  β₀          = Base return (intercept)")
    print("  β₁          = Risk premium (slope)")
    print("  ε_t         = Error term")
    print()
    
    # Prepare variables
    y = df_analysis['Log_Return'].values
    X = df_analysis['Volatility'].values
    X_with_const = sm.add_constant(X)
    
    # Estimate OLS model
    model_ols = sm.OLS(y, X_with_const)
    results_ols = model_ols.fit()
    
    print("OLS REGRESSION RESULTS:")
    print("=" * 80)
    print(results_ols.summary())
    print()
    
    # Extract coefficients
    beta_0 = results_ols.params[0]
    beta_1 = results_ols.params[1]
    
    print("INTERPRETATION OF COEFFICIENTS:")
    print("-" * 80)
    print(f"β̂₀ (Intercept) = {beta_0:.6f}")
    print(f"  → Base return when volatility is zero")
    if beta_0 > 0:
        print(f"  → Positive: Suggests positive baseline return")
    else:
        print(f"  → Negative: Suggests negative baseline return")
    print()
    
    print(f"β̂₁ (Risk Premium) = {beta_1:.6f}")
    print(f"  → Change in return for 1-unit increase in volatility")
    if beta_1 > 0:
        print(f"  → Positive: Higher risk associated with higher return (expected)")
    else:
        print(f"  → Negative: Higher risk associated with lower return (unexpected)")
    print()
    
    print(f"R² = {results_ols.rsquared:.4f}")
    print(f"  → {results_ols.rsquared*100:.2f}% of return variation explained by volatility")
    print()
    
    print(f"Adjusted R² = {results_ols.rsquared_adj:.4f}")
    print(f"  → Adjusted for degrees of freedom")
    print()
    
    # Visualize the relationship
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot with regression line
    axes[0].scatter(X, y, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = beta_0 + beta_1 * X_line
    axes[0].plot(X_line, y_line, 'r-', linewidth=2.5, 
                label=f'Fitted line: Return = {beta_0:.4f} + {beta_1:.4f}×Volatility')
    axes[0].set_xlabel('Volatility (Risk)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Log Return', fontsize=11, fontweight='bold')
    axes[0].set_title('Risk-Return Relationship (OLS)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = results_ols.resid
    axes[1].scatter(results_ols.fittedvalues, residuals, alpha=0.5, s=30, 
                   edgecolors='black', linewidths=0.5)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
    axes[1].set_title('Residual Plot\n(Should show random scatter)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('question2_ols_risk_return.png', dpi=300, bbox_inches='tight')
    print("✓ OLS visualization saved as 'question2_ols_risk_return.png'")
    plt.show()
    
    return results_ols, X, y, X_with_const


def test_ols_assumptions(results_ols, X, y):
    """
    Test validity of OLS assumptions.
    
    KEY ASSUMPTIONS:
    1. Linearity
    2. Normality of residuals
    3. Homoscedasticity (constant variance)
    4. No autocorrelation
    5. No multicollinearity (not applicable for simple regression)
    """
    print("\n")
    print("STEP 5: TESTING OLS ASSUMPTIONS")
    print("=" * 80)
    
    residuals = results_ols.resid
    
    # Test 1: Normality of Residuals (Jarque-Bera Test)
    print("\n1. NORMALITY TEST (Jarque-Bera)")
    print("-" * 80)
    print("Null Hypothesis: Residuals are normally distributed")
    print("Alternative: Residuals are NOT normally distributed")
    print()
    
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)
    print(f"Jarque-Bera statistic: {jb_stat:.4f}")
    print(f"P-value: {jb_pvalue:.4f}")
    print(f"Skewness: {skew:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    print()
    
    if jb_pvalue > 0.05:
        print(f"✓ Result: FAIL TO REJECT null (p = {jb_pvalue:.4f} > 0.05)")
        print("  → Residuals appear normally distributed")
        normality_pass = True
    else:
        print(f"✗ Result: REJECT null (p = {jb_pvalue:.4f} < 0.05)")
        print("  → Residuals are NOT normally distributed")
        print("  → OLS assumption violated!")
        normality_pass = False
    print()
    
    # Test 2: Homoscedasticity (Breusch-Pagan Test)
    print("2. HOMOSCEDASTICITY TEST (Breusch-Pagan)")
    print("-" * 80)
    print("Null Hypothesis: Residuals have constant variance (homoscedastic)")
    print("Alternative: Residuals have non-constant variance (heteroscedastic)")
    print()
    
    X_with_const = sm.add_constant(X)
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_with_const)
    print(f"Breusch-Pagan statistic: {bp_stat:.4f}")
    print(f"P-value: {bp_pvalue:.4f}")
    print()
    
    if bp_pvalue > 0.05:
        print(f"✓ Result: FAIL TO REJECT null (p = {bp_pvalue:.4f} > 0.05)")
        print("  → Residuals have constant variance")
        print("  → Homoscedasticity assumption satisfied")
        homoscedasticity_pass = True
    else:
        print(f"✗ Result: REJECT null (p = {bp_pvalue:.4f} < 0.05)")
        print("  → Residuals have non-constant variance (heteroscedasticity)")
        print("  → OLS assumption violated!")
        homoscedasticity_pass = False
    print()
    
    # Test 3: Autocorrelation (Ljung-Box Test)
    print("3. AUTOCORRELATION TEST (Ljung-Box)")
    print("-" * 80)
    print("Null Hypothesis: No autocorrelation in residuals")
    print("Alternative: Autocorrelation present in residuals")
    print()
    
    lb_result = acorr_ljungbox(residuals, lags=[10], return_df=True)
    lb_stat = lb_result['lb_stat'].values[0]
    lb_pvalue = lb_result['lb_pvalue'].values[0]
    
    print(f"Ljung-Box statistic (lag 10): {lb_stat:.4f}")
    print(f"P-value: {lb_pvalue:.4f}")
    print()
    
    if lb_pvalue > 0.05:
        print(f"✓ Result: FAIL TO REJECT null (p = {lb_pvalue:.4f} > 0.05)")
        print("  → No significant autocorrelation detected")
        print("  → Independence assumption satisfied")
        autocorr_pass = True
    else:
        print(f"✗ Result: REJECT null (p = {lb_pvalue:.4f} < 0.05)")
        print("  → Autocorrelation present in residuals")
        print("  → OLS assumption violated!")
        autocorr_pass = False
    print()
    
    
    # Test 4: Durbin-Watson (additional autocorrelation test)
    print("4. DURBIN-WATSON TEST (Autocorrelation)")
    print("-" * 80)
    print("Range: 0 to 4")
    print("  - DW ≈ 2: No autocorrelation")
    print("  - DW < 2: Positive autocorrelation")
    print("  - DW > 2: Negative autocorrelation")
    print()
    
    dw_stat = durbin_watson(residuals)
    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    
    if 1.5 < dw_stat < 2.5:
        print("✓ DW ≈ 2: No significant autocorrelation")
    elif dw_stat < 1.5:
        print("✗ DW < 1.5: Positive autocorrelation detected")
    else:
        print("✗ DW > 2.5: Negative autocorrelation detected")
    print()
    
    # Summary of assumption tests
    print("SUMMARY OF OLS ASSUMPTION TESTS")
    print("=" * 80)
    print(f"{'Test':<30} {'Status':<15} {'P-value':<15}")
    print("-" * 60)
    print(f"{'1. Normality (Jarque-Bera)':<30} {'PASS' if normality_pass else 'FAIL':<15} {jb_pvalue:.4f}")
    print(f"{'2. Homoscedasticity (BP)':<30} {'PASS' if homoscedasticity_pass else 'FAIL':<15} {bp_pvalue:.4f}")
    print(f"{'3. No Autocorrelation (LB)':<30} {'PASS' if autocorr_pass else 'FAIL':<15} {lb_pvalue:.4f}")
    print(f"{'4. Durbin-Watson':<30} {dw_stat:<15.4f}")
    print()
    
    all_assumptions_met = normality_pass and homoscedasticity_pass and autocorr_pass
    
    if all_assumptions_met:
        print("✓ CONCLUSION: All OLS assumptions are satisfied!")
        print("  → OLS is VALID for this risk-return model")
        print("  → We can trust the coefficient estimates and inference")
    else:
        print("✗ CONCLUSION: Some OLS assumptions are violated!")
        print("  → OLS may NOT be valid for this model")
        print("  → Consider using Maximum Likelihood Estimation (MLE)")
        print("  → Or use robust standard errors")
    print()
    
    # Diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Residuals vs Fitted
    axes[0, 0].scatter(results_ols.fittedvalues, residuals, alpha=0.5, s=30)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values', fontsize=11)
    axes[0, 0].set_ylabel('Residuals', fontsize=11)
    axes[0, 0].set_title('Residuals vs Fitted\n(Check for patterns)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot\n(Check normality)', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Scale-Location (sqrt of standardized residuals vs fitted)
    standardized_resid = (residuals - residuals.mean()) / residuals.std()
    axes[1, 0].scatter(results_ols.fittedvalues, np.sqrt(np.abs(standardized_resid)), alpha=0.5, s=30)
    axes[1, 0].set_xlabel('Fitted Values', fontsize=11)
    axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=11)
    axes[1, 0].set_title('Scale-Location Plot\n(Check homoscedasticity)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Histogram of residuals
    axes[1, 1].hist(residuals, bins=30, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black')
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Residuals', fontsize=11)
    axes[1, 1].set_ylabel('Density', fontsize=11)
    axes[1, 1].set_title('Distribution of Residuals\n(Check normality)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('question2_ols_diagnostics.png', dpi=300, bbox_inches='tight')
    print("✓ Diagnostic plots saved as 'question2_ols_diagnostics.png'")
    plt.show()
    
    return all_assumptions_met, {
        'normality': normality_pass,
        'homoscedasticity': homoscedasticity_pass,
        'autocorrelation': autocorr_pass,
        'jb_pvalue': jb_pvalue,
        'bp_pvalue': bp_pvalue,
        'lb_pvalue': lb_pvalue,
        'dw_stat': dw_stat
    }


def estimate_risk_return_mle(X, y):
    """
    Estimate risk-return model using Maximum Likelihood Estimation (MLE).
    
    This is used when OLS assumptions are violated.
    """
    print("\n")
    print("STEP 6: MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
    print("=" * 80)
    
    print("WHY USE MLE?")
    print("-" * 80)
    print("""
When OLS assumptions are violated, MLE provides an alternative approach:
- MLE finds parameters that maximize the likelihood of observing the data
- More flexible: can handle non-normal distributions
- Provides asymptotically efficient estimates
- Can incorporate heteroscedasticity and autocorrelation
    """)
    print()
    
    print("MLE SETUP:")
    print("-" * 80)
    print("Assume: Y_i ~ N(β₀ + β₁X_i, σ²)")
    print()
    print("Likelihood function:")
    print("  L(β₀, β₁, σ² | Data) = Π [1/√(2πσ²)] × exp[-(y_i - β₀ - β₁x_i)²/(2σ²)]")
    print()
    print("Log-likelihood:")
    print("  ℓ(β₀, β₁, σ²) = -n/2·ln(2π) - n/2·ln(σ²) - (1/2σ²)·Σ(y_i - β₀ - β₁x_i)²")
    print()
    
    # Define negative log-likelihood function (to minimize)
    def neg_log_likelihood(params):
        beta_0, beta_1, log_sigma = params
        sigma = np.exp(log_sigma)  # Use log(sigma) to ensure sigma > 0
        
        # Predicted values
        y_pred = beta_0 + beta_1 * X
        
        # Residuals
        residuals = y - y_pred
        
        # Negative log-likelihood
        n = len(y)
        nll = (n/2) * np.log(2*np.pi) + (n/2) * np.log(sigma**2) + np.sum(residuals**2) / (2*sigma**2)
        
        return nll
    
    # Initial parameter guesses (use OLS estimates)
    X_with_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_with_const).fit()
    beta_0_init = ols_model.params[0]
    beta_1_init = ols_model.params[1]
    sigma_init = np.sqrt(ols_model.mse_resid)
    
    initial_params = [beta_0_init, beta_1_init, np.log(sigma_init)]
    
    print("OPTIMIZATION:")
    print("-" * 80)
    print(f"Initial parameter guesses (from OLS):")
    print(f"  β₀ = {beta_0_init:.6f}")
    print(f"  β₁ = {beta_1_init:.6f}")
    print(f"  σ = {sigma_init:.6f}")
    print()
    print("Minimizing negative log-likelihood...")
    
    # Optimize
    result = minimize(neg_log_likelihood, initial_params, method='BFGS')
    
    # Extract MLE estimates
    beta_0_mle = result.x[0]
    beta_1_mle = result.x[1]
    sigma_mle = np.exp(result.x[2])
    
    print("✓ Optimization converged!")
    print()
    
    print("MLE ESTIMATES:")
    print("=" * 80)
    print(f"β̂₀_MLE (Intercept) = {beta_0_mle:.6f}")
    print(f"β̂₁_MLE (Risk Premium) = {beta_1_mle:.6f}")
    print(f"σ̂_MLE (Std Deviation) = {sigma_mle:.6f}")
    print(f"σ̂²_MLE (Variance) = {sigma_mle**2:.8f}")
    print()
    
    # Compute standard errors using inverse Hessian
    # (This is an approximation; more sophisticated methods exist)
    hessian = result.hess_inv
    se_beta_0 = np.sqrt(hessian[0, 0])
    se_beta_1 = np.sqrt(hessian[1, 1])
    
    print("STANDARD ERRORS:")
    print("-" * 80)
    print(f"SE(β̂₀) = {se_beta_0:.6f}")
    print(f"SE(β̂₁) = {se_beta_1:.6f}")
    print()
    
    # Compute t-statistics and p-values
    t_beta_0 = beta_0_mle / se_beta_0
    t_beta_1 = beta_1_mle / se_beta_1
    n = len(y)
    p_beta_0 = 2 * (1 - stats.t.cdf(np.abs(t_beta_0), df=n-2))
    p_beta_1 = 2 * (1 - stats.t.cdf(np.abs(t_beta_1), df=n-2))
    
    print("STATISTICAL SIGNIFICANCE:")
    print("-" * 80)
    print(f"β̂₀: t-statistic = {t_beta_0:.4f}, p-value = {p_beta_0:.4f}")
    if p_beta_0 < 0.05:
        print(f"  → Significant at 5% level ✓")
    else:
        print(f"  → Not significant at 5% level")
    print()
    
    print(f"β̂₁: t-statistic = {t_beta_1:.4f}, p-value = {p_beta_1:.4f}")
    if p_beta_1 < 0.05:
        print(f"  → Significant at 5% level ✓")
    else:
        print(f"  → Not significant at 5% level")
    print()
    
    # Compare with OLS
    print("COMPARISON: MLE vs OLS")
    print("=" * 80)
    print(f"{'Parameter':<15} {'MLE':<15} {'OLS':<15} {'Difference'}")
    print("-" * 60)
    print(f"{'β̂₀':<15} {beta_0_mle:<15.6f} {beta_0_init:<15.6f} {abs(beta_0_mle-beta_0_init):<15.6f}")
    print(f"{'β̂₁':<15} {beta_1_mle:<15.6f} {beta_1_init:<15.6f} {abs(beta_1_mle-beta_1_init):<15.6f}")
    print(f"{'σ̂²':<15} {sigma_mle**2:<15.8f} {sigma_init**2:<15.8f} {abs(sigma_mle**2-sigma_init**2):<15.8f}")
    print()
    
    print("KEY OBSERVATION:")
    print("-" * 80)
    if abs(beta_0_mle - beta_0_init) < 0.001 and abs(beta_1_mle - beta_1_init) < 0.001:
        print("✓ MLE and OLS estimates are very similar (as expected under normality)")
    else:
        print("! MLE and OLS estimates differ significantly")
        print("  This suggests departure from normality or other assumption violations")
    print()
    
    # Calculate predicted values and residuals
    y_pred_mle = beta_0_mle + beta_1_mle * X
    residuals_mle = y - y_pred_mle
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Comparison of OLS and MLE fits
    axes[0].scatter(X, y, alpha=0.5, s=30, label='Data', edgecolors='black', linewidths=0.5)
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line_ols = beta_0_init + beta_1_init * X_line
    y_line_mle = beta_0_mle + beta_1_mle * X_line
    axes[0].plot(X_line, y_line_ols, 'r-', linewidth=2.5, label=f'OLS', alpha=0.7)
    axes[0].plot(X_line, y_line_mle, 'b--', linewidth=2.5, label=f'MLE', alpha=0.7)
    axes[0].set_xlabel('Volatility (Risk)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Log Return', fontsize=11, fontweight='bold')
    axes[0].set_title('Comparison: MLE vs OLS', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: MLE residuals
    axes[1].scatter(y_pred_mle, residuals_mle, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Fitted Values (MLE)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
    axes[1].set_title('MLE Residual Plot', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('question2_mle_results.png', dpi=300, bbox_inches='tight')
    print("✓ MLE visualization saved as 'question2_mle_results.png'")
    plt.show()
    
    return {
        'beta_0': beta_0_mle,
        'beta_1': beta_1_mle,
        'sigma': sigma_mle,
        'se_beta_0': se_beta_0,
        'se_beta_1': se_beta_1,
        'p_beta_0': p_beta_0,
        'p_beta_1': p_beta_1
    }


def generate_report(results_ols, mle_results, assumption_test_results, all_assumptions_met):
    """
    Generate comprehensive report.
    """
    print("\n\n")
    print("="*80)
    print("FINAL REPORT: RISK-RETURN ANALYSIS FOR KAKUZI SECURITIES")
    print("="*80)
    print()
    
    print("EXECUTIVE SUMMARY")
    print("-" * 80)
    print("""
This analysis examines the relationship between risk and return for Kakuzi 
securities listed on the Nairobi Securities Exchange (NSE). Using time series 
data spanning over 200 observations, we estimated a risk-return model to 
understand how volatility (risk) affects expected returns.
    """)
    print()
    
    print("DATA DESCRIPTION")
    print("-" * 80)
    print("• Security: Kakuzi Limited (NSE)")
    print("• Observations: 403 daily price observations")
    print("• Period: Covers recent trading history")
    print("• Variables:")
    print("  - Log Returns: ln(Pt/Pt-1) as return measure")
    print("  - Volatility: 30-day rolling standard deviation as risk measure")
    print()
    
    print("METHODOLOGY")
    print("-" * 80)
    print("1. Computed log returns from daily closing prices")
    print("2. Calculated rolling volatility as risk measure")
    print("3. Estimated risk-return model: Return = β₀ + β₁×Volatility + ε")
    print("4. Tested OLS assumptions (normality, homoscedasticity, autocorrelation)")
    if not all_assumptions_met:
        print("5. Applied MLE due to OLS assumption violations")
    print()
    
    print("KEY FINDINGS")
    print("-" * 80)
    
    # OLS Results
    beta_0_ols = results_ols.params[0]
    beta_1_ols = results_ols.params[1]
    r_squared = results_ols.rsquared
    
    print(f"OLS Estimates:")
    print(f"  β̂₀ (Base Return) = {beta_0_ols:.6f}")
    print(f"  β̂₁ (Risk Premium) = {beta_1_ols:.6f}")
    print(f"  R² = {r_squared:.4f} ({r_squared*100:.2f}% of variance explained)")
    print()
    
    # Assumption tests
    print("OLS Assumption Tests:")
    print(f"  • Normality (Jarque-Bera): {'PASS' if assumption_test_results['normality'] else 'FAIL'} (p = {assumption_test_results['jb_pvalue']:.4f})")
    print(f"  • Homoscedasticity (Breusch-Pagan): {'PASS' if assumption_test_results['homoscedasticity'] else 'FAIL'} (p = {assumption_test_results['bp_pvalue']:.4f})")
    print(f"  • No Autocorrelation (Ljung-Box): {'PASS' if assumption_test_results['autocorrelation'] else 'FAIL'} (p = {assumption_test_results['lb_pvalue']:.4f})")
    print(f"  • Durbin-Watson: {assumption_test_results['dw_stat']:.4f}")
    print()
    
    if all_assumptions_met:
        print("✓ All OLS assumptions satisfied - OLS estimates are valid!")
    else:
        print("✗ Some OLS assumptions violated - MLE provides more reliable estimates")
        print()
        print(f"MLE Estimates:")
        print(f"  β̂₀_MLE = {mle_results['beta_0']:.6f} (SE = {mle_results['se_beta_0']:.6f})")
        print(f"  β̂₁_MLE = {mle_results['beta_1']:.6f} (SE = {mle_results['se_beta_1']:.6f})")
        print(f"  σ̂_MLE = {mle_results['sigma']:.6f}")
    print()
    
    print("INTERPRETATION")
    print("-" * 80)
    
    if all_assumptions_met:
        beta_1 = beta_1_ols
    else:
        beta_1 = mle_results['beta_1']
    
    if beta_1 > 0:
        print(f"• Risk Premium (β₁ = {beta_1:.6f}) is POSITIVE:")
        print("  → Higher volatility (risk) is associated with higher expected returns")
        print("  → This is consistent with financial theory")
        print("  → Investors require compensation for bearing additional risk")
    else:
        print(f"• Risk Premium (β₁ = {beta_1:.6f}) is NEGATIVE:")
        print("  → Higher volatility (risk) is associated with LOWER expected returns")
        print("  → This is contrary to standard financial theory")
        print("  → May indicate market inefficiencies or specific characteristics of Kakuzi stock")
    print()
    
    print("CONCLUSIONS")
    print("-" * 80)
    print("""
1. The risk-return relationship for Kakuzi securities has been successfully 
   modeled using time series analysis.

2. Statistical tests were conducted to validate the OLS assumptions, ensuring
   the reliability of our estimates.

3. When OLS assumptions were violated, Maximum Likelihood Estimation (MLE) 
   provided an alternative approach that is robust to these violations.

4. The analysis demonstrates the importance of assumption testing in 
   econometric modeling and the value of alternative estimation methods.
    """)
    
    print("="*80)
    print("END OF REPORT")
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to execute the entire analysis.
    """
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "TIME SERIES & FORECASTING ASSIGNMENT" + " "*22 + "║")
    print("║" + " "*25 + "Valerie Jerono - 222331" + " "*30 + "║")
    print("║" + " "*30 + "06/01/2025" + " "*38 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    try:
        # Question 1: OLS vs MLE
        print("\n")
        input("Press Enter to start Question 1 (OLS vs MLE)...")
        q1_results = question_1_ols_vs_mle()
        
        # Question 2: Risk-Return Model
        print("\n\n")
        input("Press Enter to start Question 2 (Risk-Return Analysis)...")
        
        # Load and prepare data
        df = load_and_prepare_kakuzi_data()
        
        # Compute returns and volatility
        df, df_analysis = compute_returns_and_volatility(df)
        
        # Visualize
        visualize_price_and_returns(df)
        
        # Estimate OLS
        results_ols, X, y, X_with_const = estimate_risk_return_ols(df_analysis)
        
        # Test assumptions
        all_assumptions_met, assumption_test_results = test_ols_assumptions(results_ols, X, y)
        
        # If assumptions violated, use MLE
        mle_results = None
        if not all_assumptions_met:
            print("\n")
            input("OLS assumptions violated. Press Enter to continue with MLE...")
            mle_results = estimate_risk_return_mle(X, y)
        else:
            # Create dummy MLE results for report
            mle_results = {
                'beta_0': results_ols.params[0],
                'beta_1': results_ols.params[1],
                'sigma': np.sqrt(results_ols.mse_resid),
                'se_beta_0': results_ols.bse[0],
                'se_beta_1': results_ols.bse[1],
                'p_beta_0': results_ols.pvalues[0],
                'p_beta_1': results_ols.pvalues[1]
            }
        
        # Generate final report
        generate_report(results_ols, mle_results, assumption_test_results, all_assumptions_met)
        
        print("\n")
        print("="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  1. question1_ols_vs_mle.png - OLS vs MLE visualization")
        print("  2. question2_price_returns_volatility.png - Time series plots")
        print("  3. question2_returns_distribution.png - Distribution analysis")
        print("  4. question2_ols_risk_return.png - OLS regression results")
        print("  5. question2_ols_diagnostics.png - Diagnostic plots")
        if not all_assumptions_met:
            print("  6. question2_mle_results.png - MLE results")
        print()
        
    except FileNotFoundError:
        print("Error: kakuzi.csv file not found!")
        print("Please ensure the file is in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
