# Kakuzi NSE: OLS vs MLE Analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/)

## 📊 Overview

This repository contains a comprehensive **Time Series & Forecasting** analysis comparing **Ordinary Least Squares (OLS)** and **Maximum Likelihood Estimation (MLE)** methods. The project analyzes **Kakuzi securities** from the **Nairobi Securities Exchange (NSE)** to build risk-return models and understand the theoretical and practical differences between OLS and MLE estimation approaches.

### Key Features

- 📈 **Mathematical demonstration** of OLS vs MLE variance estimation differences
- 📉 **Risk-return modeling** using real stock market data from NSE
- 🔍 **Comprehensive diagnostic testing** (normality, heteroscedasticity, autocorrelation)
- 📊 **Publication-ready visualizations** (12 high-quality plots)
- 📄 **LaTeX academic report** ready for Overleaf compilation
- 🐍 **Interactive Python scripts** for reproducible analysis

## 🎯 Project Objectives

### Question 1: OLS vs MLE Theoretical Comparison
Mathematically demonstrate that:
- OLS and MLE produce **identical coefficient estimates** under normality assumption
- OLS produces **higher variance estimates** due to degrees of freedom adjustment
- Understand the bias-variance tradeoff between the two methods

### Question 2: Kakuzi Risk-Return Analysis
- Estimate risk-return relationship using NSE stock data
- Perform comprehensive assumption testing
- Compare OLS and MLE estimation results
- Provide economic interpretation of findings

## 📁 Project Structure

```
Kakuzi_NSE_olsVSmle/
├── README.md                                    # This file
│
└── NSE_23and24/                                # Main analysis folder
    ├── time_series_assignment.py               # Interactive analysis script
    ├── generate_individual_plots.py            # Batch plot generation
    ├── time_series_assignment_individual_plots.py
    │
    ├── nse.ipynb                               # Jupyter notebook analysis
    │
    ├── kakuzi.csv                              # Kakuzi stock price data (405 obs)
    ├── kakuzi_solution.tex                     # Complete LaTeX report
    ├── kakuzi.md                               # Markdown version of report
    │
    ├── README_SOLUTION.md                      # Solution documentation
    │
    ├── Kenya Nairobi Securities Exchange (NSE) All Stocks/
    │   ├── NSE_data_all_stocks_2023.csv       # Full NSE data 2023
    │   ├── NSE_data_all_stocks_2024.csv       # Full NSE data 2024
    │   └── NSE_data_stock_market_sectors_2023_2024.csv
    │
    └── [Generated Images]
        ├── q1_regression_fit.png               # Question 1 plots
        ├── q1_variance_comparison.png
        ├── q2_price_series.png                 # Question 2 plots
        ├── q2_log_returns.png
        ├── q2_volatility.png
        ├── q2_returns_histogram.png
        ├── q2_qq_plot.png
        ├── q2_ols_scatter.png
        ├── q2_ols_residuals.png
        ├── q2_diagnostic_qq.png
        ├── q2_diagnostic_residuals_fitted.png
        └── q2_diagnostic_scale_location.png
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/VAL-Jerono/Kakuzi_NSE_olsVSmle.git
   cd Kakuzi_NSE_olsVSmle/NSE_23and24
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate   # On Windows
   ```

3. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib seaborn scipy statsmodels jupyter
   ```

### Running the Analysis

#### Option 1: Interactive Python Script
```bash
python time_series_assignment.py
```
This will:
- Run Question 1 analysis with step-by-step explanations
- Run Question 2 analysis with comprehensive diagnostics
- Generate all visualizations
- Display results in terminal

#### Option 2: Generate All Plots at Once
```bash
python generate_individual_plots.py
```
Generates all 12 publication-ready plots.

#### Option 3: Jupyter Notebook
```bash
jupyter notebook nse.ipynb
```
Interactive notebook with code cells and markdown explanations.

## 📊 Key Results

### Question 1: OLS vs MLE Comparison

Using a simple 5-observation example:

| Parameter | OLS | MLE | Relationship |
|-----------|-----|-----|--------------|
| β̂₁ (slope) | 1.9900 | 1.9900 | **IDENTICAL** |
| β̂₀ (intercept) | 0.0700 | 0.0700 | **IDENTICAL** |
| σ̂² (variance) | 0.0563 | 0.0338 | **OLS > MLE** ✓ |

**Key Finding**: OLS variance is exactly `(n/(n-k))` times the MLE variance, where:
- `n` = sample size
- `k` = number of parameters

**Mathematical Relationship**:
```
σ̂²_OLS = (n/(n-k)) × σ̂²_MLE = (5/3) × 0.0338 = 0.0563
```

### Question 2: Kakuzi Risk-Return Model

**Sample**: 403 daily observations (Jan 2024 - Jan 2026)

**OLS Estimates**:
- Risk-Return coefficient: β̂₁ = 0.000157
- Interpretation: 1% increase in volatility → 0.000157 increase in expected return
- R² = 0.0006 (low explanatory power)

**Diagnostic Test Results**:
| Test | Result | p-value | Interpretation |
|------|--------|---------|----------------|
| Jarque-Bera (Normality) | ✗ Failed | < 0.05 | Non-normal residuals |
| Breusch-Pagan (Homoscedasticity) | ✗ Failed | < 0.05 | Heteroscedastic errors |
| Ljung-Box (Autocorrelation) | ✗ Failed | < 0.05 | Serial correlation present |
| Durbin-Watson | ✓ Passed | ≈ 2.0 | No first-order autocorrelation |

**Conclusion**: MLE is more appropriate due to OLS assumption violations.

## 📈 Visualizations

The project generates 12 high-quality plots:

### Question 1 (2 plots)
- **Regression Fit**: OLS/MLE fit with residuals
- **Variance Comparison**: Visual comparison of variance estimates

### Question 2 (10 plots)
- **Price Series**: Kakuzi stock price over time
- **Log Returns**: Daily logarithmic returns
- **Volatility**: 30-day rolling volatility
- **Returns Distribution**: Histogram with normal overlay
- **Q-Q Plot**: Normality assessment
- **Risk-Return Scatter**: OLS regression line
- **Residual Plot**: Residuals vs fitted values
- **Diagnostic Plots**: Q-Q, residuals, scale-location

## 📄 Academic Report

The complete solution is available in LaTeX format:

### Compiling the Report

1. **Upload to Overleaf**:
   - Upload [kakuzi_solution.tex](NSE_23and24/kakuzi_solution.tex)
   - Upload all 12 `.png` files
   - Select pdfLaTeX compiler
   - Click "Recompile"

2. **Local Compilation** (requires LaTeX installation):
   ```bash
   cd NSE_23and24
   pdflatex kakuzi_solution.tex
   pdflatex kakuzi_solution.tex  # Run twice for TOC
   ```

### Report Structure

- **Title Page** with student information
- **Table of Contents**
- **Question 1** (4-5 pages): OLS vs MLE mathematical demonstration
- **Question 2** (8-10 pages): Risk-return analysis with diagnostics
- **Total**: ~15 pages with all visualizations and explanations

## 🔬 Methodology

### Data Collection
- **Source**: Nairobi Securities Exchange (NSE)
- **Stock**: Kakuzi Limited
- **Period**: January 2024 - January 2026
- **Observations**: 403 daily trading records

### Statistical Methods

1. **Log Returns Calculation**:
   ```
   r_t = ln(P_t / P_{t-1})
   ```

2. **Volatility Estimation** (30-day rolling standard deviation):
   ```
   σ_t = √(Var(r_{t-29}, ..., r_t))
   ```

3. **Risk-Return Model**:
   ```
   E[r_t] = β₀ + β₁ × σ_t + ε_t
   ```

4. **Estimation Approaches**:
   - **OLS**: Minimizes sum of squared residuals
   - **MLE**: Maximizes likelihood under normality assumption

### Diagnostic Tests

- **Jarque-Bera Test**: Tests for normality of residuals
- **Breusch-Pagan Test**: Tests for heteroscedasticity
- **Ljung-Box Test**: Tests for autocorrelation
- **Durbin-Watson Statistic**: Tests for first-order autocorrelation

## 🛠️ Dependencies

### Required Python Packages

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.12.0
jupyter>=1.0.0
```

### Optional for PDF Generation
- LaTeX distribution (TeXLive, MiKTeX, or MacTeX)
- pdflatex compiler

## 📚 Key Insights

### Theoretical Understanding

1. **OLS vs MLE Equivalence**: Under normal errors, coefficient estimates are identical
2. **Degrees of Freedom Matter**: OLS uses (n-k) for unbiased variance; MLE uses n for maximum likelihood
3. **Bias-Variance Tradeoff**: OLS variance is unbiased; MLE variance is biased but has lower MSE

### Practical Implications

1. **Model Assumptions**: Always test OLS assumptions before using results
2. **Robust Estimation**: When assumptions fail, consider MLE or robust standard errors
3. **Risk-Return Modeling**: Low R² suggests other factors affect returns beyond volatility
4. **Market Efficiency**: Weak relationship supports semi-strong form efficiency hypothesis

## 🎓 Educational Value

This project is ideal for:
- **Time Series Analysis** courses
- **Econometrics** students comparing estimation methods
- **Financial Econometrics** applications
- Understanding **statistical inference** concepts
- Learning **Python** for statistical analysis

## 👥 Author

**Valerie Jerono**  
Date: January 6, 2025

## 📝 License

This project is available under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Data source: Nairobi Securities Exchange (NSE)
- Statistical methods: Based on econometric theory from Greene (2018), Wooldridge (2020)
- Python libraries: NumPy, pandas, matplotlib, seaborn, scipy, statsmodels

## 📧 Contact

For questions or collaboration:
- GitHub: [@VAL-Jerono](https://github.com/VAL-Jerono)
- Repository: [Kakuzi_NSE_olsVSmle](https://github.com/VAL-Jerono/Kakuzi_NSE_olsVSmle)

## 🔗 Related Resources

- [Econometric Analysis (Greene)](https://www.pearson.com/en-us/subject-catalog/p/econometric-analysis/P200000005894)
- [Introductory Econometrics (Wooldridge)](https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge/)
- [Statsmodels Documentation](https://www.statsmodels.org/)
- [NSE Official Website](https://www.nse.co.ke/)

---

**⭐ If you find this project helpful, please consider giving it a star!**