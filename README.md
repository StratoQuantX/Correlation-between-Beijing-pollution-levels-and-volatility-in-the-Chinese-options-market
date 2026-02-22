# Beijing Pollution & Chinese Options Market Volatility
### Quant Research Project — Work in Progress

> **Correlation Between Beijing Air Pollution (PM2.5) and the Volatility of the Chinese Options Market (CSI 300)**

---

## Overview

This project investigates whether Beijing air pollution levels (PM2.5, AQI) have a statistically significant and structurally meaningful impact on the implied and realized volatility of the CSI 300 options market.

The core hypothesis is that pollution acts as a **drift modifier** in a stochastic volatility framework, captured by a parameter δ in a modified Heston-type SDE:

$$d\sigma_t = \alpha(\theta - \sigma_t)dt + \delta \cdot \text{Pollution}_t \, dt + \beta \sqrt{\sigma_t} \, dW_t$$

Where:
- **α** — mean-reversion speed
- **θ** — long-run volatility level
- **β** — volatility of volatility
- **δ** — systematic effect of pollution on volatility drift

---

## Project Structure

```
.
├── data/
│   ├── raw/                        # Raw downloaded files
│   │   ├── CSI.csv                 # CSI 300 index (Macrotrends)
│   │   ├── beijing-air-quality.csv # PM2.5/PM10 daily (Beijing)
│   │   ├── VIX.csv                 # CBOE VIX index
│   │   └── USDCNH.csv              # USD/CNH exchange rate
│   └── processed/
│       └── csi300_pollution_df.csv # Clean merged dataset
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Phase 1 — Data processing & merging
│   ├── 02_eda.ipynb                # Phase 1 — Exploratory analysis
│   ├── 03_econometrics.ipynb       # Phase 2 — Correlations, regressions, GARCH-X
│   ├── 04_stochastic_model.ipynb   # Phase 3 — SDE calibration & Monte Carlo
│   └── 05_trading.ipynb            # Phase 4 — Signal & backtesting
├── src/
│   ├── features.py                 # Feature engineering pipeline
│   ├── models.py                   # Stochastic volatility model
│   └── backtest.py                 # Trading strategy & evaluation
├── outputs/
│   └── figures/                    # Charts and visualizations
├── paper/                          # Final research paper (LaTeX)
└── README.md
```

---

## Data Sources

| Variable | Source | Period |
|---|---|---|
| CSI 300 Index | Investing.com | 2015–2025 |
| PM2.5 / PM10 Beijing | US Embassy / WAQI archive | 2015–2025 |
| VIX | Yahoo Finance | 2015–2025 |
| USD/CNH | Investing.com | 2015–2025 |

**Note:** Realized volatility is computed as the 20-day rolling standard deviation of log-returns, annualized (×√252). Missing pollution days are dropped (no forward-fill) to preserve data integrity.

---

## Methodology (Roadmap)

| Phase | Content | Status |
|---|---|---|
| **Phase 1** | Data acquisition, cleaning, feature engineering | ✅ Done |
| **Phase 2** | Correlations, Granger causality, GARCH-X | 🔄 In progress |
| **Phase 3** | Stochastic volatility model (SDE + calibration) | ⏳ Pending |
| **Phase 4** | Trading signal & backtesting | ⏳ Pending |
| **Phase 5** | Research paper & presentation | ⏳ Pending |

---

## Key Hypotheses

- **H1** : Pollution ↑ → Realized Volatility ↑
- **H2** : Pollution impact is lagged (t+1 to t+5)
- **H3** : Pollution acts as a drift modifier in stochastic volatility dynamics (parameter δ)

---

## Requirements

```bash
pip install pandas numpy matplotlib statsmodels arch scipy yfinance
```

---

## Status

🚧 **Work in progress** — This README will be updated upon project completion.

---

*Research project — 2025*
