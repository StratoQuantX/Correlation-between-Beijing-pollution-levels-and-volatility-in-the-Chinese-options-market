# Beijing Pollution & CSI 300 Volatility
### A Quantitative Research Project

> **Does Beijing Air Pollution (PM2.5) Structurally Drive the Volatility of Chinese Options Markets?**

---

## Overview

This project investigates whether Beijing air pollution levels (PM2.5) have a statistically significant and structurally meaningful impact on the realized volatility of the CSI 300 index, and quantifies the resulting mispricing in options markets.

The central contribution is a **pollution-modified Heston stochastic volatility model** in which PM2.5 enters as an exogenous drift modifier in the variance equation:

$$dS_t = rS_t \ dt + \sqrt{v_t} \ S_t \ dW_t^S$$

$$dv_t = \alpha(\theta - v_t) \ dt + \delta \cdot P_t \ dt + \beta\sqrt{v_t} \ dW_t^v \quad \langle dW^S dW^v \rangle = \rho \ dt$$

where $\delta$ captures the **systematic effect of pollution on the variance drift** — the key parameter of the model.

The full pipeline spans data engineering, econometric analysis, stochastic calibration via Extended Kalman Filter with Milstein discretization, Monte Carlo options pricing, and systematic backtesting of volatility trading strategies.

---

## Key Results

| Result | Value |
|---|---|
| **δ** — pollution effect on conditional variance | 0.0029 (p < 0.001) |
| Mean pollution contribution to conditional variance | **26.5%** |
| Kalman Filter — α (mean-reversion speed) | 2.661 |
| Kalman Filter — θ (long-run vol level) | 0.100 |
| Kalman Filter — half-life | 65.6 days |
| ρ (spot-vol correlation) | −0.097 |
| Heston-Pollution premium vs baseline — ATM, High pol | **+70.8%** |
| Heston-Pollution premium vs baseline — OTM +15%, High pol | **+143.2%** |
| Best strategy — Delta-neutral strangle | **Sharpe 1.67, WR 86%** |
| Optimal parameter zone | holding ∈ [40, 61d] × window ∈ [20, 35d] |
| VIX–realized vol spread at entry (mean) | −0.004 ≈ 0 |

---

## Project Structure

```
.
├── data/
│   ├── CSI.csv                     # CSI 300 index (Investing.com)
│   ├── beijing-air-quality.csv     # PM2.5 / PM10 daily — Beijing
│   ├── VIX.csv                     # CBOE VIX index
│   └── USDCNH.csv                  # USD/CNH exchange rate
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Phase 1 — Data processing & merging
│   ├── 02_eda.ipynb                # Phase 1 — Exploratory data analysis
│   ├── 03_econometrics.ipynb       # Phase 2 — Correlations, GARCH-X, Granger, VAR
│   ├── 04_stochastic_model.ipynb   # Phase 3 — SDE calibration & Heston pricing
│   └── 05_trading.ipynb            # Phase 4 — Signal construction & backtesting
│
├── src/
│   ├── features.py                 # Feature engineering pipeline
│   ├── models.py                   # GARCHPollution, PollutionSVModel,
│   │                               #   HestonPricer, HestonPollutionPricer
│   └── backtest.py                 # StraddleBacktest, DeltaNeutralStrangleBacktest,
│                                   #   RegimeSwitchingBacktest, StressTest
│
├── outputs/
│   └── figures/                    # All charts and visualizations
│
├── paper/                          # Research paper (LaTeX)
└── README.md
```

---

## Data Sources

| Variable | Source | Frequency | Period |
|---|---|---|---|
| CSI 300 Index | Investing.com | Daily | 2015–2025 |
| PM2.5 / PM10 — Beijing | US Embassy archive | Daily | 2015–2025 |
| VIX | Yahoo Finance | Daily | 2015–2025 |
| USD/CNH | Investing.com | Daily | 2015–2025 |

**Data notes:**
- Realized volatility: 20-day rolling standard deviation of log-returns, annualized (×√252)
- Missing pollution observations are **dropped** (no forward-fill) to preserve data integrity
- Final dataset: 2,524 trading days after alignment and cleaning

---

## Methodology

### Phase 1 — Data & Features
- CSI 300 log-returns and 20-day realized volatility
- PM2.5 lags (1–5 days), rolling means (5d, 20d), rolling std, standardized series
- Merge on CSI 300 trading days; VIX and USD/CNH as macro controls

### Phase 2 — Econometric Analysis
- Pearson / Spearman / Kendall correlations → all significant at p < 0.001
- OLS regressions with controls and monthly dummies → β significant across all specifications
- Two-step GARCH-X → **δ = 0.0029, p < 0.001**, 26.5% contribution to conditional variance
- Granger causality → non-significant PM2.5 → vol (linear); significant vol → PM2.5 (activity channel)
- VAR + IRF → positive persistent response; FEVD → 0.24% at horizon 20

### Phase 3 — Stochastic Volatility Model
- CIR-type SDE with pollution drift modifier δ·P_t
- Discretization: **Milstein scheme** (strong order 1.0)
- Calibration: **Extended Kalman Filter** via MLE
- Estimated parameters: α = 2.661, θ = 0.100, β = 0.537, δ = 0.00217
- Option pricing: **Heston-Pollution MC pricer** with correlated Brownians (ρ = −0.097)
- Benchmark: standard Heston baseline (δ = 0)

### Phase 4 — Trading Strategies

| Strategy | Sharpe | Win Rate | Max DD | N trades |
|---|---|---|---|---|
| Long straddle (baseline) | 1.53 | 81% | moderate | 37 |
| Regime-switching | 0.97 | 62% | high | 26 |
| **Delta-neutral strangle** | **1.67** | **86%** | low | 37 |

- **Signal**: PM2.5 > rolling mean + 2σ (25-day window)
- **Entry**: long OTM strangle (±5%), daily delta-hedge
- **Exit**: after 60-day holding period, so 3 months trading
- **Robustness**: Sharpe > 1.0 across all signal windows [20–45d] at holding ≥ 35d with the baseline strategy

---

## Key Hypotheses

| Hypothesis | Statement | Result |
|---|---|---|
| **H1** | Pollution ↑ → Realized Volatility ↑ | ✅ Confirmed |
| **H2** | Pollution effect is lagged (t+1 to t+5) | ✅ Confirmed (CCF, peak at lag 3) |
| **H3** | Pollution acts as structural drift modifier (δ) | ✅ Confirmed (δ > 0, p < 0.001) |

---

## Limitations

- Options P&L approximated via Black-Scholes using realized volatility — no historical IV data available for CSI 300 options. The near-zero VIX–realized vol spread at entry dates (mean = −0.004) suggests the approximation is reasonable, but results should be interpreted as an upper bound on achievable alpha.
- Granger causality non-significant in the linear VAR framework — the pollution–volatility relationship operates through the variance process, not the conditional mean, motivating the SDE approach.
- Walk-forward analysis identifies 2018 and 2020–2021 as drawdown periods, coinciding with structural vol regime shifts orthogonal to the pollution signal.
- The pollution signal is informative in one direction only, i.e. long vol performs, short vol don't (we tried an iron condor, huge disaster in terms of PnL), consistent with the asymmetric tail risk profile of Chinese equity markets.

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn statsmodels arch scipy
```

---

## Citation

```
Yassine Housseine, Gianni Marchetti (2026). Beijing Pollution and CSI 300 Volatility:
A Pollution-Modified Heston Framework for Options Pricing and Volatility Trading.
Research Project StratoQuant
```

---

*Yassine Housseine — M2 MMMEF, University of Paris 1 Panthéon-Sorbonne*

*Gianni Marchetti — M2 MOFI, Aix-Marseille University*

*Co-founders and Quant Researchers, StratoQuant*

*2025–2026*
