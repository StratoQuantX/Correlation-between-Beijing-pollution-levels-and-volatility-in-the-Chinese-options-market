import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from scipy.optimize import minimize
from scipy.stats import norm


class GARCHPollution:
    """
    Two-step GARCH-X model estimating the effect of pollution (δ)
    on conditional volatility.

    Step 1 : fit GARCH(p,q) on returns → extract conditional variance
    Step 2 : regress conditional variance on pollution → estimate δ
    Step 3 : simulate vol paths under different pollution scenarios
    
    Parameters
    ----------
    returns    : pd.Series  — asset log-returns
    pollution  : pd.Series  — pollution series (PM2.5, lagged or MA)
    p, q       : int        — GARCH lag orders
    asset_name : str        — label for plots
    """

    def __init__(self, returns, pollution, p=1, q=1,
                 asset_name='CSI 300', standardize=False):
        self.asset_name  = asset_name
        self.dates       = returns.index
        self.p           = p
        self.q           = q
        self.ret_raw     = returns
        self.pol_raw     = pollution

        if standardize:
            self.ret = ((returns - returns.mean()) / returns.std()) * 100
            self.pol = (pollution - pollution.mean()) / pollution.std()
        else:
            self.ret = returns * 100
            self.pol = pollution

        # Results
        self.garch_result_  = None
        self.ols_result_    = None
        self.cond_var_      = None
        self.delta_         = None
        self.omega_         = None
        self.alpha_         = None
        self.beta_          = None

    # ── Fit ───────────────────────────────────────────────────────
    def fit(self, disp=False):
        # Step 1 : GARCH(p,q)
        self.garch_result_ = arch_model(
            self.ret, vol='Garch', p=self.p, q=self.q, dist='normal'
        ).fit(disp='off' if not disp else 'on')

        self.cond_var_ = self.garch_result_.conditional_volatility ** 2
        self.omega_    = self.garch_result_.params['omega']
        self.alpha_    = self.garch_result_.params['alpha[1]']
        self.beta_     = self.garch_result_.params['beta[1]']

        # Step 2 : OLS conditional variance ~ pollution
        X = sm.add_constant(self.pol)
        self.ols_result_ = sm.OLS(self.cond_var_, X).fit(cov_type='HC3')
        self.delta_      = self.ols_result_.params.iloc[1]

        if disp:
            self.summary()

        return self

    # ── Summary ───────────────────────────────────────────────────
    def summary(self):
        self._check_fitted()

        from scipy import stats

        # GARCH params
        garch_params = self.garch_result_.params
        garch_pvals  = self.garch_result_.pvalues
        garch_bse    = self.garch_result_.std_err

        def stars(p):
            return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''

        rows = []

        # ── GARCH block ───────────────────────────────────────────────
        for name in ['omega', 'alpha[1]', 'beta[1]']:
            rows.append({
                'Block':     'GARCH',
                'Parameter': name,
                'Coef':      round(garch_params[name], 6),
                'Std Err':   round(garch_bse[name], 6),
                'P-value':   round(garch_pvals[name], 4),
                'Sig':       stars(garch_pvals[name])
            })

        # Derived GARCH stats
        persistence = self.alpha_ + self.beta_
        rows.append({
            'Block':     'GARCH',
            'Parameter': 'persistence (α+β)',
            'Coef':      round(persistence, 6),
            'Std Err':   '—',
            'P-value':   '—',
            'Sig':       '⚠️ unit root' if persistence >= 1 else '✓ stationary'
        })

        # ── Pollution block ───────────────────────────────────────────
        delta_p   = self.ols_result_.pvalues.iloc[1]
        delta_se  = self.ols_result_.bse.iloc[1]
        delta_ci  = self.ols_result_.conf_int().iloc[1]

        rows.append({
            'Block':     'Pollution (δ)',
            'Parameter': 'delta (pm25)',
            'Coef':      round(self.delta_, 6),
            'Std Err':   round(delta_se, 6),
            'P-value':   round(delta_p, 4),
            'Sig':       stars(delta_p)
        })
        rows.append({
            'Block':     'Pollution (δ)',
            'Parameter': '95% CI',
            'Coef':      f'[{delta_ci.iloc[0]:.6f}, {delta_ci.iloc[1]:.6f}]',
            'Std Err':   '—',
            'P-value':   '—',
            'Sig':       '—'
        })
        rows.append({
            'Block':     'Pollution (δ)',
            'Parameter': 'R² (step 2)',
            'Coef':      round(self.ols_result_.rsquared, 4),
            'Std Err':   '—',
            'P-value':   '—',
            'Sig':       '—'
        })

        # ── Model fit block ───────────────────────────────────────────
        rows.append({
            'Block':     'Model fit',
            'Parameter': 'Log-likelihood',
            'Coef':      round(self.garch_result_.loglikelihood, 2),
            'Std Err':   '—', 'P-value': '—', 'Sig': '—'
        })
        rows.append({
            'Block':     'Model fit',
            'Parameter': 'AIC',
            'Coef':      round(self.garch_result_.aic, 2),
            'Std Err':   '—', 'P-value': '—', 'Sig': '—'
        })
        rows.append({
            'Block':     'Model fit',
            'Parameter': 'BIC',
            'Coef':      round(self.garch_result_.bic, 2),
            'Std Err':   '—', 'P-value': '—', 'Sig': '—'
        })
        rows.append({
            'Block':     'Model fit',
            'Parameter': 'N observations',
            'Coef':      int(self.garch_result_.nobs),
            'Std Err':   '—', 'P-value': '—', 'Sig': '—'
        })

        df_summary = pd.DataFrame(rows).set_index(['Block', 'Parameter'])

        print(f"\n{'='*65}")
        print(f"  GARCHPollution Summary — {self.asset_name}")
        print(f"{'='*65}")
        print(df_summary.to_string())
        print(f"{'='*65}")
        print("  Significance: *** p<0.01  ** p<0.05  * p<0.10")
        print(f"{'='*65}\n")

        return df_summary

    # ── Plot 1 : conditional vol with/without pollution ───────────
    def plot_effect(self, threshold=150, figsize=(14, 10)):
        self._check_fitted()

        cond_vol         = np.sqrt(self.cond_var_)
        pol_contribution = self.delta_ * self.pol.values
        adj_var          = np.maximum(self.cond_var_ - pol_contribution, 1e-6)
        adj_vol          = np.sqrt(adj_var)
        contribution_pct = (pol_contribution / self.cond_var_) * 100

        fig, axes = plt.subplots(3, 1, figsize=figsize)

        # Panel 1 — vol with vs without pollution
        axes[0].plot(self.dates, cond_vol,
                     color='salmon', lw=1.2, label='Conditional vol (GARCH)')
        axes[0].plot(self.dates, adj_vol,
                     color='steelblue', lw=1.2, alpha=0.8,
                     label='Vol without pollution effect')
        axes[0].fill_between(self.dates, adj_vol, cond_vol,
                             where=(cond_vol > adj_vol),
                             color='salmon', alpha=0.25,
                             label='Pollution contribution')
        axes[0].set_title(f'Conditional Volatility — GARCH vs Pollution-adjusted ({self.asset_name})',
                          fontsize=12)
        axes[0].set_ylabel('Volatility')
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        # Panel 2 — pollution contribution % over time
        axes[1].fill_between(self.dates, contribution_pct,
                             color='coral', alpha=0.6)
        axes[1].axhline(contribution_pct.mean(), color='darkred',
                        linestyle='--', lw=1.5,
                        label=f'Mean: {contribution_pct.mean():.2f}%')
        axes[1].set_title('Pollution contribution to conditional variance (%)',
                          fontsize=12)
        axes[1].set_ylabel('%')
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)

        # Panel 3 — vol distribution by regime
        high = cond_vol[self.pol > threshold]
        low  = cond_vol[self.pol <= threshold]
        axes[2].hist(low,  bins=50, density=True, alpha=0.5,
                     color='steelblue',
                     label=f'Low pollution (≤{threshold})  n={len(low)}  mean={low.mean():.3f}')
        axes[2].hist(high, bins=50, density=True, alpha=0.5,
                     color='salmon',
                     label=f'High pollution (>{threshold})  n={len(high)}  mean={high.mean():.3f}')
        axes[2].axvline(low.mean(),  color='steelblue', linestyle='--', lw=1.5)
        axes[2].axvline(high.mean(), color='salmon',    linestyle='--', lw=1.5)
        axes[2].set_title('Conditional Vol distribution — High vs Low Pollution',
                          fontsize=12)
        axes[2].set_ylabel('Density')
        axes[2].legend(fontsize=9)
        axes[2].grid(alpha=0.3)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        return fig

    # ── Plot 2 : simulated vol paths under pollution scenarios ────
    def simulate_paths(self, pollution_scenarios: dict,
                       n_days=252, n_paths=1000, seed=42,
                       figsize=(14, 5)):
        """
        Simulate GARCH vol paths under different pollution levels.

        pollution_scenarios = {
            'Low (PM2.5=30)':     30,
            'Medium (PM2.5=100)': 100,
            'High (PM2.5=300)':   300,
        }
        """
        self._check_fitted()
        np.random.seed(seed)

        colors = ['steelblue', 'orange', 'salmon', 'purple', 'green']
        fig, ax = plt.subplots(figsize=figsize)

        summary_rows = []

        for (label, pm25_level), color in zip(
                pollution_scenarios.items(), colors):

            paths = np.zeros((n_paths, n_days))
            sigma2_init = self.omega_ / (1 - self.alpha_ - self.beta_)

            for i in range(n_paths):
                sigma2 = np.zeros(n_days)
                eps    = np.zeros(n_days)
                sigma2[0] = sigma2_init

                for t in range(1, n_days):
                    sigma2[t] = (self.omega_
                                 + self.alpha_ * eps[t-1] ** 2
                                 + self.beta_  * sigma2[t-1]
                                 + self.delta_ * pm25_level)
                    eps[t] = np.random.normal(0, np.sqrt(max(sigma2[t], 1e-8)))

                # annualize
                paths[i] = np.sqrt(sigma2) * np.sqrt(252) / 10  # /10 bc ret in %

            mean_path = paths.mean(axis=0)
            p5        = np.percentile(paths,  5, axis=0)
            p95       = np.percentile(paths, 95, axis=0)

            ax.plot(mean_path, color=color, lw=2,
                    label=f'{label}  |  mean={mean_path.mean():.3f}  '
                           f'p5={p5.mean():.3f}  p95={p95.mean():.3f}')
            ax.fill_between(range(n_days), p5, p95,
                            color=color, alpha=0.12)

            summary_rows.append({
                'Scenario':   label,
                'PM2.5':      pm25_level,
                'Mean vol':   round(mean_path.mean(), 4),
                'P5 vol':     round(p5.mean(), 4),
                'P95 vol':    round(p95.mean(), 4),
            })

        ax.set_title(f'Simulated Volatility Paths — Pollution Scenarios '
                     f'(δ={self.delta_:.5f})', fontsize=12)
        ax.set_xlabel('Days')
        ax.set_ylabel('Annualized Volatility')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(pd.DataFrame(summary_rows).to_string(index=False))
        return fig

    # ── Plot 3 : delta sensitivity ────────────────────────────────
    def plot_delta_sensitivity(self, pm25_range=(0, 500),
                               n_days=252, figsize=(10, 5)):
        """
        Show how mean annualized vol changes as a function of PM2.5 level.
        """
        self._check_fitted()
        pm25_values = np.linspace(*pm25_range, 100)
        sigma2_init = self.omega_ / (1 - self.alpha_ - self.beta_)
        mean_vols   = []

        for pm25 in pm25_values:
            sigma2 = np.zeros(n_days)
            sigma2[0] = sigma2_init
            eps = np.zeros(n_days)
            for t in range(1, n_days):
                sigma2[t] = (self.omega_
                             + self.alpha_ * eps[t-1] ** 2
                             + self.beta_  * sigma2[t-1]
                             + self.delta_ * pm25)
                eps[t] = 0  # deterministic path
            mean_vols.append(np.sqrt(sigma2).mean() * np.sqrt(252) / 10)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(pm25_values, mean_vols, color='steelblue', lw=2)
        ax.axvline(150, color='red', linestyle='--', lw=1.2,
                   label='PM2.5 = 150 (high pollution threshold)')
        ax.fill_betweenx([min(mean_vols), max(mean_vols)],
                         0, 150, color='green', alpha=0.05,
                         label='Low pollution zone')
        ax.fill_betweenx([min(mean_vols), max(mean_vols)],
                         150, 500, color='red', alpha=0.05,
                         label='High pollution zone')
        ax.set_title(f'Volatility sensitivity to PM2.5 level (δ={self.delta_:.5f})',
                     fontsize=12)
        ax.set_xlabel('PM2.5 level')
        ax.set_ylabel('Mean annualized volatility')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig

    # ── Internal ──────────────────────────────────────────────────
    def _check_fitted(self):
        assert self.garch_result_ is not None, \
            "Model not fitted. Call fit() first."