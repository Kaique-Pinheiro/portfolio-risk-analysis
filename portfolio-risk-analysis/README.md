# Portfolio Risk Analysis — B3

> Quantitative risk analysis of a B3 equity portfolio using industry-standard methodologies: VaR, CVaR, Market Beta, Monte Carlo Simulation and Markowitz Efficient Frontier.

---

## Overview

This project implements a complete **quantitative risk management pipeline** focused on the Brazilian equity market. Built as a technical portfolio for investment banking positions, the codebase demonstrates command of the methodologies required in **Risk, Quant Finance and Structuring** roles.

**Portfolio:** ITUB4, BBDC4, PETR4, VALE3, BBAS3 (equally weighted, 20% each)  
**Benchmark:** Ibovespa (^BVSP)  
**Horizon:** 3 years of historical data via yfinance

---

## Methodologies

### Value at Risk (VaR)

A risk measure that answers: *"What is the maximum expected loss in a single day at X% confidence?"*

| Method | Assumption | Advantage | Limitation |
|---|---|---|---|
| **Parametric** | Returns follow a normal distribution | Analytical, fast | Underestimates fat tails |
| **Historical** | Empirical distribution of past returns | Captures real asymmetry | Depends on the historical window |
| **Monte Carlo** | Simulates N scenarios using historical µ and σ | Flexible, extensible to GBM | Computationally intensive |

### CVaR / Expected Shortfall (ES)

**CVaR** (Conditional Value at Risk) is the average loss in scenarios that *exceed* the VaR:

```
CVaR_α = E[L | L > VaR_α]
```

**Why does Basel III prefer CVaR over VaR?**  
VaR ignores the magnitude of losses in the tail — two portfolios with the same 2% VaR can have CVaRs of 3% and 8% respectively. CVaR is a coherent risk measure (satisfies subadditivity), which incentivizes diversification. The **FRTB (Basel IV)** framework replaced VaR with Expected Shortfall as the standard regulatory capital metric.

### Market Beta

Beta measures the sensitivity of an asset relative to the market (Ibovespa):

```
β = Cov(R_asset, R_market) / Var(R_market)
```

| Beta | Interpretation |
|---|---|
| β > 1 | More volatile than the market (aggressive) |
| β = 1 | Moves exactly with the market |
| 0 < β < 1 | Less volatile than the market (defensive) |
| β < 0 | Moves inversely to the market (natural hedge) |

### Monte Carlo Simulation

Generates **10,000 daily return scenarios** sampled from a normal distribution calibrated on historical parameters. Enables VaR and CVaR estimation without analytical assumptions about the shape of the distribution.

**Natural extension:** multivariate simulation with Cholesky decomposition to preserve the correlation structure across assets (multivariate GBM).

### Markowitz Efficient Frontier

Based on Modern Portfolio Theory (Markowitz, 1952), it identifies portfolios that **maximize return for a given level of risk**.

**Sharpe Ratio:**
```
Sharpe = (E[R_p] - Rf) / σ_p
```

Three portfolios are identified and compared:

| Portfolio | Criterion | Use Case |
|---|---|---|
| **Maximum Sharpe** | Maximizes risk-adjusted return | Standard allocation |
| **Minimum Variance** | Minimizes portfolio volatility | Conservative profile |
| **Equally Weighted** | 20% in each asset (baseline) | Naïve benchmark |

---

## Project Structure

```
portfolio-risk-analysis/
├── analysis.py          <- main pipeline (single file to run)
├── requirements.txt     <- pinned dependencies
├── README.md            <- this documentation
└── .gitignore           <- ignores outputs, __pycache__, .env
```

---

## Installation and Usage

```bash
# 1. Clone the repository
git clone https://github.com/Kaique-Pinheiro/portfolio-risk-analysis.git
cd portfolio-risk-analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python analysis.py
```

**Requirements:** Python 3.10+

---

## Dependencies

```
yfinance>=0.2.40    # Historical market data from B3 via Yahoo Finance
pandas>=2.0         # Time series manipulation
numpy>=1.26         # Linear algebra and vectorized operations
matplotlib>=3.8     # Visualizations and subplots
scipy>=1.12         # Statistical distributions and SLSQP optimization
```

---

## Output

### Terminal (sample output)

```
=================================================================
[1] DATA DOWNLOAD
=================================================================
  Tickers     : ['ITUB4.SA', 'BBDC4.SA', 'PETR4.SA', 'VALE3.SA', 'BBAS3.SA']
  Benchmark   : ^BVSP
  Shape       : (754, 5) (trading days x assets)
  Period      : 22/04/2023 -> 22/04/2026
  Data downloaded successfully!

=================================================================
[2] RETURNS AND DESCRIPTIVE STATISTICS
=================================================================
       Annual Return (%)  Annual Volatility (%)  Skewness  Kurtosis  Sharpe Ratio
ITUB4             18.42                  28.61   -0.2341    3.1204        0.2981
...
```

### Generated charts (`risk_analysis.png`)

| Subplot | Title | Description |
|---|---|---|
| 1 (top) | Cumulative Return | Evolution of the 5 assets on a 1.0 base |
| 2 | Return Distribution + VaR | Histogram with fitted normal curve and 3 VaR lines |
| 3 | Correlation Heatmap | Annotated correlation matrix |
| 4 | VaR by Asset | Comparative bar chart of historical VaR (95%) |
| 5 | Monte Carlo | Histogram of 10k simulated scenarios |
| 6 (bottom) | Efficient Frontier | Scatter of 3,000 portfolios colored by Sharpe Ratio |

---

## Assets Analyzed

| Ticker | Company | Sector |
|---|---|---|
| ITUB4.SA | Itaú Unibanco | Financial — Banks |
| BBDC4.SA | Banco Bradesco | Financial — Banks |
| PETR4.SA | Petrobras | Energy — Oil & Gas |
| VALE3.SA | Vale | Materials — Mining |
| BBAS3.SA | Banco do Brasil | Financial — Banks |
| ^BVSP | Ibovespa | Market benchmark |

**Risk-free rate:** Selic ≈ 10.75% p.a. (used as Rf in Sharpe Ratio and parametric VaR)

---

## Key Financial Concepts

### Why logarithmic returns?

```python
# Simple return (not time-additive):
simple_ret = (P_t / P_{t-1}) - 1

# Log return (time-additive — essential property for time series modeling):
log_ret = np.log(P_t / P_{t-1})
```

Log returns are preferred because: (1) they are time-additive, meaning weekly return equals the sum of daily returns; (2) they are unbounded below, avoiding the -100% floor constraint; (3) they are more compatible with the normality assumption required by parametric models.

### Why annualize with √252?

```python
# 252 = average number of trading days per year on B3
annual_volatility = daily_volatility * np.sqrt(252)
annual_return     = mean_daily_return * 252
```

This follows from the i.i.d. assumption: if daily returns are independent and identically distributed, annual variance is 252× the daily variance — so annual standard deviation scales by √252.

### Why compare Parametric vs Historical VaR?

Financial markets exhibit **fat tails** — extreme events occur more frequently than a normal distribution predicts (documented by Mandelbrot and Taleb). When Historical VaR > Parametric VaR, the data confirms that real tails are heavier than the normal model assumes, providing empirical validation for the regulatory preference for CVaR.

---

## References

- **Markowitz, H.** (1952). Portfolio Selection. *Journal of Finance*.
- **J.P. Morgan / Reuters** (1996). *RiskMetrics Technical Document*.
- **Hull, J.C.** (2018). *Risk Management and Financial Institutions*. Wiley.
- **Basel Committee on Banking Supervision** (2019). *Minimum capital requirements for market risk (FRTB)*.

---

## Author

**Kaique Pinheiro** — Computer Science student, FEI  
GitHub: [github.com/Kaique-Pinheiro](https://github.com/Kaique-Pinheiro)  
Email: kaique.pinheiro.dev@gmail.com
