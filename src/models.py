import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from arch import arch_model
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from scipy import stats


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
        


class PollutionSVModel:
    """
    Stochastic Volatility Model with Pollution drift modifier.

    SDE (continuous):
        dσ_t = α(θ - σ_t)dt + δ·P_t·dt + β√σ_t·dW_t

    Discretization (Milstein scheme, strong order 1.0):
        σ_{t+1} = σ_t + α(θ - σ_t)Δt + δ·P_t·Δt
                + β√σ_t·ΔW_t + (β²/2)(ΔW_t² - Δt)

    Kalman Filter on log-variance for calibration.

    Parameters
    ----------
    sigma    : pd.Series — realized volatility (observed proxy for σ_t)
    pollution: pd.Series — PM2.5 series (aligned with sigma)
    dt       : float     — time step (1/252 for daily)
    """

    def __init__(self, sigma, pollution, dt=1/252):
        self.sigma     = np.asarray(sigma)
        self.pollution = np.asarray(pollution)
        self.dt        = dt
        self.T         = len(sigma)
        self.dates     = sigma.index if hasattr(sigma, 'index') else np.arange(self.T)

        # Estimated parameters
        self.alpha_ = None
        self.theta_ = None
        self.beta_  = None
        self.delta_ = None
        self.ll_    = None

        # Kalman outputs
        self.filtered_state_ = None
        self.filtered_cov_   = None
        self.sigma_hat_      = None

    # ── Milstein discretization ───────────────────────────────────
    def milstein_step(self, sigma_t, P_t, dW, alpha, theta, beta, delta):
        """
        Single Milstein step for the CIR-type SDE with pollution drift.
        σ must be clipped to avoid negative values (absorbing barrier).
        """
        sigma_t = max(sigma_t, 1e-8)
        drift   = alpha * (theta - sigma_t) * self.dt + delta * P_t * self.dt
        diffusion = beta * np.sqrt(sigma_t) * dW
        milstein  = 0.5 * beta**2 * (dW**2 - self.dt)  # Milstein correction
        return max(sigma_t + drift + diffusion + milstein, 1e-8)

    # ── Extended Kalman Filter ────────────────────────────────────
    def kalman_filter(self, params):
        """
        Extended Kalman Filter on the linearized SDE.
        State  : x_t = σ_t
        Obs    : y_t = σ_t^realized + ε_t,  ε ~ N(0, R)
        """
        alpha, theta, beta, delta = params

        # Noise variances
        Q = beta**2 * np.mean(self.sigma) * self.dt   # process noise
        R = np.var(np.diff(self.sigma)) * 0.1          # observation noise

        # Init
        x = self.sigma[0]
        P = Q
        ll = 0.0

        filtered_states = np.zeros(self.T)
        filtered_covs   = np.zeros(self.T)
        filtered_states[0] = x
        filtered_covs[0]   = P

        for t in range(1, self.T):
            # ── Predict ───────────────────────────────────────────
            F = 1 - alpha * self.dt   # Jacobian of f w.r.t. x
            x_pred = (x
                      + alpha * (theta - x) * self.dt
                      + delta * self.pollution[t-1] * self.dt)
            x_pred = max(x_pred, 1e-8)
            P_pred = F**2 * P + Q

            # ── Update ────────────────────────────────────────────
            y   = self.sigma[t]
            inn = y - x_pred          # innovation
            S   = P_pred + R          # innovation variance
            K   = P_pred / S          # Kalman gain
            x   = x_pred + K * inn
            P   = (1 - K) * P_pred

            filtered_states[t] = x
            filtered_covs[t]   = P

            # ── Log-likelihood ────────────────────────────────────
            ll += -0.5 * (np.log(2 * np.pi * S) + inn**2 / S)

        return -ll, filtered_states, filtered_covs  # return neg ll for minimization

    # ── Calibration ───────────────────────────────────────────────
    def fit(self, delta_init=None):
        """
        Calibrate (α, θ, β, δ) via MLE using the Kalman Filter likelihood.
        """
        # Initial parameter guesses
        alpha0 = 2.0
        theta0 = float(np.mean(self.sigma))
        beta0  = float(np.std(np.diff(self.sigma)) * np.sqrt(252))
        delta0 = delta_init if delta_init is not None else 0.0029

        x0     = [alpha0, theta0, beta0, delta0]
        bounds = [(0.01, 50),    # alpha  — positive mean reversion
                  (0.01, 2.0),   # theta  — long-run vol level
                  (0.01, 5.0),   # beta   — vol of vol
                  (-1.0, 1.0)]   # delta  — pollution effect

        def objective(params):
            try:
                nll, _, _ = self.kalman_filter(params)
                return nll
            except Exception:
                return 1e10

        result = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10}
        )

        self.alpha_, self.theta_, self.beta_, self.delta_ = result.x
        self.ll_ = -result.fun

        # Rerun filter with optimal params to store states
        _, self.filtered_state_, self.filtered_cov_ = self.kalman_filter(result.x)
        self.sigma_hat_ = self.filtered_state_

        return self

    # ── Summary ───────────────────────────────────────────────────
    def summary(self):
        self._check_fitted()

        def stars(val, threshold=0):
            return '(>0 ✓)' if val > threshold else '(<0 ✗)'

        rows = [
            {'Parameter': 'α (mean-reversion)',  'Value': round(self.alpha_, 6),
             'Interpretation': f'Half-life: {round(np.log(2)/self.alpha_ / (1/252), 1)} days'},
            {'Parameter': 'θ (long-run vol)',     'Value': round(self.theta_, 6),
             'Interpretation': f'Unconditional vol: {round(self.theta_, 4)}'},
            {'Parameter': 'β (vol of vol)',       'Value': round(self.beta_, 6),
             'Interpretation': f'Diffusion intensity'},
            {'Parameter': 'δ (pollution effect)', 'Value': round(self.delta_, 6),
             'Interpretation': stars(self.delta_)},
            {'Parameter': 'Log-likelihood',       'Value': round(self.ll_, 4),
             'Interpretation': ''},
        ]

        df_sum = pd.DataFrame(rows).set_index('Parameter')
        print(f"\n{'='*60}")
        print(f"  PollutionSVModel — Kalman Filter + Milstein")
        print(f"{'='*60}")
        print(df_sum.to_string())
        print(f"{'='*60}\n")
        return df_sum

    # ── Monte Carlo simulation ────────────────────────────────────
    def simulate_paths(self, pollution_scenarios: dict,
                       n_days=252, n_paths=1000,
                       seed=42, figsize=(14, 5)):
        self._check_fitted()
        np.random.seed(seed)

        colors = ['steelblue', 'orange', 'salmon', 'purple']
        fig, ax = plt.subplots(figsize=figsize)
        rows = []

        for (label, pm25_level), color in zip(pollution_scenarios.items(), colors):
            paths = np.zeros((n_paths, n_days))

            for i in range(n_paths):
                sigma_t = self.theta_
                dW = np.random.normal(0, np.sqrt(self.dt), n_days)
                for t in range(n_days):
                    sigma_t = self.milstein_step(
                        sigma_t, pm25_level, dW[t],
                        self.alpha_, self.theta_,
                        self.beta_,  self.delta_
                    )
                    paths[i, t] = sigma_t

            mean_path = paths.mean(axis=0)
            p5        = np.percentile(paths,  5, axis=0)
            p95       = np.percentile(paths, 95, axis=0)

            ax.plot(mean_path, color=color, lw=2,
                    label=f'{label}  mean={mean_path.mean():.4f}')
            ax.fill_between(range(n_days), p5, p95,
                            color=color, alpha=0.12)

            rows.append({
                'Scenario': label, 'PM2.5': pm25_level,
                'Mean σ': round(mean_path.mean(), 4),
                'P5 σ':  round(p5.mean(), 4),
                'P95 σ': round(p95.mean(), 4),
            })

        ax.axhline(self.theta_, color='black', linestyle='--',
                   lw=1, label=f'θ = {self.theta_:.4f}')
        ax.set_title(f'Monte Carlo — Milstein Paths (α={self.alpha_:.3f}, '
                     f'θ={self.theta_:.3f}, β={self.beta_:.3f}, δ={self.delta_:.5f})',
                     fontsize=11)
        ax.set_xlabel('Days')
        ax.set_ylabel('σ_t')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(pd.DataFrame(rows).to_string(index=False))

    # ── Plot filtered state vs observed ──────────────────────────
    def plot_filtered(self, figsize=(14, 5)):
        self._check_fitted()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.dates, self.sigma,
                color='salmon', lw=0.8, alpha=0.7, label='Observed σ (realized vol)')
        ax.plot(self.dates, self.sigma_hat_,
                color='steelblue', lw=1.5, label='Kalman filtered σ̂')

        ci = 1.96 * np.sqrt(self.filtered_cov_)
        ax.fill_between(self.dates,
                        self.sigma_hat_ - ci,
                        self.sigma_hat_ + ci,
                        color='steelblue', alpha=0.15, label='95% CI')

        ax.set_title('Kalman Filter — Observed vs Filtered Volatility', fontsize=12)
        ax.set_ylabel('σ_t')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    # ── Internal ──────────────────────────────────────────────────
    def _check_fitted(self):
        assert self.alpha_ is not None, "Model not fitted. Call fit() first."



class HestonPollutionPricer:
    """
    Option pricing under the Pollution-modified Heston model.

    Dynamics:
        dS_t = r·S_t·dt + √v_t·S_t·dW_t^S
        dv_t = α(θ - v_t)dt + δ·P_t·dt + β√v_t·dW_t^v
        dW_t^S·dW_t^v = ρ·dt

    Calibration of ρ from historical data.
    Pricing via Monte Carlo with Milstein discretization.

    Parameters
    ----------
    sv_model : PollutionSVModel — fitted model (α, θ, β, δ)
    S0       : float           — current spot price
    r        : float           — risk-free rate
    rho      : float           — spot-vol correlation (estimated or calibrated)
    """

    def __init__(self, sv_model, S0, returns_series, r=0.03, rho=None):
        self.sv  = sv_model
        self.S0  = S0
        self.r   = r
        self.rho = rho if rho is not None else self._estimate_rho(returns_series)

    def _estimate_rho(self, returns):
        innovations = self.sv.sigma[1:] - self.sv.sigma_hat_[:-1]
        ret = np.asarray(returns)[1:]
        n   = min(len(ret), len(innovations))
        rho = np.corrcoef(ret[:n], innovations[:n])[0, 1]
        print(f"Estimated ρ : {rho:.4f}")
        return rho

    # ── Cholesky decomposition for correlated Brownians ───────────
    def _correlated_brownians(self, n_steps, dt, seed=None):
        """
        Generate correlated (dW^S, dW^v) via Cholesky.
        """
        if seed is not None:
            np.random.seed(seed)
        Z1 = np.random.normal(0, 1, n_steps)
        Z2 = np.random.normal(0, 1, n_steps)
        dW_S = Z1 * np.sqrt(dt)
        dW_v = (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2) * np.sqrt(dt)
        return dW_S, dW_v

    # ── Single path simulation (Milstein) ─────────────────────────
    def _simulate_path(self, P_level, T, dt, seed=None):
        n_steps = int(T / dt)
        dW_S, dW_v = self._correlated_brownians(n_steps, dt, seed)

        S = self.S0
        v = self.sv.theta_  # start at long-run vol level

        for t in range(n_steps):
            v_pos = max(v, 1e-8)

            # Milstein for variance process
            v = (v
                 + self.sv.alpha_ * (self.sv.theta_ - v) * dt
                 + self.sv.delta_ * P_level * dt
                 + self.sv.beta_  * np.sqrt(v_pos) * dW_v[t]
                 + 0.5 * self.sv.beta_**2 * (dW_v[t]**2 - dt))
            v = max(v, 1e-8)

            # Milstein for asset price (GBM with stochastic vol)
            S = (S
                 + self.r * S * dt
                 + np.sqrt(v_pos) * S * dW_S[t]
                 + 0.5 * v_pos * S * (dW_S[t]**2 - dt))
            S = max(S, 1e-8)

        return S, np.sqrt(v)  # return final S and σ

    # ── Monte Carlo pricing ───────────────────────────────────────
    def mc_price(self, K, T, P_level, option_type='call',
                 n_paths=10000, dt=None, seed=42):
        """
        Price option via Monte Carlo under Heston-pollution SDE.
        """
        if dt is None:
            dt = self.sv.dt

        np.random.seed(seed)
        n_steps = int(T / dt)
        payoffs = np.zeros(n_paths)

        for i in range(n_paths):
            # Correlated Brownians
            Z1 = np.random.normal(0, 1, n_steps)
            Z2 = np.random.normal(0, 1, n_steps)
            dW_S = Z1 * np.sqrt(dt)
            dW_v = (self.rho * Z1 +
                    np.sqrt(1 - self.rho**2) * Z2) * np.sqrt(dt)

            S = self.S0
            v = self.sv.sigma[-1] ** 2

            for t in range(n_steps):
                v_pos = max(v, 1e-8)

                # Milstein vol
                v = (v
                     + self.sv.alpha_ * (self.sv.theta_ - v) * dt
                     + self.sv.delta_ * P_level * dt
                     + self.sv.beta_  * np.sqrt(v_pos) * dW_v[t]
                     + 0.5 * self.sv.beta_**2 * (dW_v[t]**2 - dt))
                v = max(v, 1e-8)

                # Milstein asset
                S = (S
                     + self.r * S * dt
                     + np.sqrt(v_pos) * S * dW_S[t]
                     + 0.5 * v_pos * S * (dW_S[t]**2 - dt))
                S = max(S, 1e-8)

            payoffs[i] = max(S - K, 0) if option_type == 'call' else max(K - S, 0)

        price = np.exp(-self.r * T) * np.mean(payoffs)
        se    = np.exp(-self.r * T) * np.std(payoffs) / np.sqrt(n_paths)
        return price, se


    # ── Plot sensitivity ──────────────────────────────────────────
    def plot_sensitivity(self, K, T,
                         pm25_range=(0, 500),
                         n_points=20,
                         option_type='call',
                         n_paths=2000,
                         figsize=(12, 5)):
        pm25_vals = np.linspace(*pm25_range, n_points)
        prices    = []
        ses       = []

        for P in pm25_vals:
            price, se = self.mc_price(K, T, P,
                                      option_type=option_type,
                                      n_paths=n_paths)
            prices.append(price)
            ses.append(se)

        prices = np.array(prices)
        ses    = np.array(ses)

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Price vs PM2.5
        axes[0].plot(pm25_vals, prices, color='salmon', lw=2)
        axes[0].fill_between(pm25_vals,
                             prices - 1.96*ses,
                             prices + 1.96*ses,
                             color='salmon', alpha=0.2,
                             label='95% MC confidence band')
        axes[0].axvline(150, color='red', linestyle='--',
                        lw=1, label='PM2.5=150')
        axes[0].set_title(f'Heston-Pollution MC price ({option_type})', fontsize=11)
        axes[0].set_xlabel('PM2.5')
        axes[0].set_ylabel('Price')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        # dPrice/dPM2.5
        dprice = np.gradient(prices, pm25_vals)
        axes[1].plot(pm25_vals, dprice, color='steelblue', lw=2)
        axes[1].axhline(0, color='black', lw=0.8, linestyle='--')
        axes[1].axvline(150, color='red', linestyle='--', lw=1)
        axes[1].set_title('dPrice / dPM2.5 (pollution delta)', fontsize=11)
        axes[1].set_xlabel('PM2.5')
        axes[1].set_ylabel('dP/dPM2.5')
        axes[1].grid(alpha=0.3)

        fig.suptitle(f'Heston-Pollution Pricing Sensitivity\n'
                     f'K={K:.0f}, T={T}y, ρ={self.rho:.3f}, '
                     f'α={self.sv.alpha_:.3f}, θ={self.sv.theta_:.3f}, '
                     f'β={self.sv.beta_:.3f}, δ={self.sv.delta_:.5f}',
                     fontsize=11)
        plt.tight_layout()
        plt.show()

    # ── Sensitivity table ─────────────────────────────────────────
    def scenario_table(self, strikes, T,
                       pm25_scenarios={'Low': 30, 'Medium': 100, 'High': 300},
                       option_type='call',
                       n_paths=5000):
        rows = []
        for K in strikes:
            for scenario, P in pm25_scenarios.items():
                price, se = self.mc_price(K, T, P,
                                          option_type=option_type,
                                          n_paths=n_paths)
                rows.append({
                    'Strike K':  round(K, 2),
                    'Scenario':  scenario,
                    'PM2.5':     P,
                    'Price':     round(price, 4),
                    'Std Err':   round(se, 4),
                })

        df = pd.DataFrame(rows)
        print(df)
        return pd.DataFrame(df)
    


class HestonPricer:
    """
    Standard Heston (1993) option pricer — no pollution term.

    Dynamics:
        dS_t = r·S_t·dt + √v_t·S_t·dW_t^S
        dv_t = α(θ - v_t)dt + β√v_t·dW_t^v
        dW_t^S·dW_t^v = ρ·dt

    Pricing via Monte Carlo with Milstein discretization.
    Used as benchmark against HestonPollutionPricer.

    Parameters
    ----------
    sv_model : PollutionSVModel — fitted model (α, θ, β) — δ ignored
    S0       : float           — current spot price
    r        : float           — risk-free rate
    rho      : float           — spot-vol correlation
    """

    def __init__(self, sv_model, S0, returns_series, r=0.03, rho=None):
        self.sv  = sv_model
        self.S0  = S0
        self.r   = r
        self.rho = rho if rho is not None else self._estimate_rho(returns_series)

    def _estimate_rho(self, returns):
        innovations = self.sv.sigma[1:] - self.sv.sigma_hat_[:-1]
        ret = np.asarray(returns)[1:]
        n   = min(len(ret), len(innovations))
        rho = np.corrcoef(ret[:n], innovations[:n])[0, 1]
        print(f"Estimated ρ : {rho:.4f}")
        return rho

    # ── Monte Carlo pricing ───────────────────────────────────────
    def mc_price(self, K, T, option_type='call',
                 n_paths=10000, dt=None, seed=42):
        if dt is None:
            dt = self.sv.dt

        np.random.seed(seed)
        n_steps = int(T / dt)
        payoffs = np.zeros(n_paths)

        for i in range(n_paths):
            Z1 = np.random.normal(0, 1, n_steps)
            Z2 = np.random.normal(0, 1, n_steps)
            dW_S = Z1 * np.sqrt(dt)
            dW_v = (self.rho * Z1 +
                    np.sqrt(1 - self.rho**2) * Z2) * np.sqrt(dt)

            S = self.S0
            v = self.sv.sigma[-1] ** 2  # current variance

            for t in range(n_steps):
                v_pos = max(v, 1e-8)

                # Milstein vol — no pollution term
                v = (v
                     + self.sv.alpha_ * (self.sv.theta_ - v) * dt
                     + self.sv.beta_  * np.sqrt(v_pos) * dW_v[t]
                     + 0.5 * self.sv.beta_**2 * (dW_v[t]**2 - dt))
                v = max(v, 1e-8)

                # Milstein asset
                S = (S
                     + self.r * S * dt
                     + np.sqrt(v_pos) * S * dW_S[t]
                     + 0.5 * v_pos * S * (dW_S[t]**2 - dt))
                S = max(S, 1e-8)

            payoffs[i] = max(S - K, 0) if option_type == 'call' else max(K - S, 0)

        price = np.exp(-self.r * T) * np.mean(payoffs)
        se    = np.exp(-self.r * T) * np.std(payoffs) / np.sqrt(n_paths)
        return price, se

    # ── Scenario table ────────────────────────────────────────────
    def scenario_table(self, strikes, T,
                       option_type='call',
                       n_paths=5000):
        rows = []
        for K in strikes:
            price, se = self.mc_price(K, T,
                                      option_type=option_type,
                                      n_paths=n_paths)
            rows.append({
                'Strike K': round(K, 2),
                'Price':    round(price, 4),
                'Std Err':  round(se, 4),
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        return df

    # ── Sensitivity plot ──────────────────────────────────────────
    def plot_sensitivity(self, K, T,
                         option_type='call',
                         n_paths=2000,
                         figsize=(10, 5)):
        price, se = self.mc_price(K, T,
                                   option_type=option_type,
                                   n_paths=n_paths)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(price, color='steelblue', lw=2,
                   label=f'Heston baseline: {price:.2f}')
        ax.fill_between([0, 1], price - 1.96*se, price + 1.96*se,
                        color='steelblue', alpha=0.2,
                        label=f'95% CI: [{price-1.96*se:.2f}, {price+1.96*se:.2f}]')
        ax.set_title(f'Heston Baseline Price\nK={K:.0f}, T={T}y, '
                     f'ρ={self.rho:.3f}, α={self.sv.alpha_:.3f}, '
                     f'θ={self.sv.theta_:.3f}, β={self.sv.beta_:.3f}',
                     fontsize=11)
        ax.set_ylabel('Price')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig

    # ── Full comparison vs HestonPollutionPricer ──────────────────
    def compare_with_pollution(self, pollution_pricer,
                                strikes, T,
                                pm25_scenarios={'Low': 30,
                                                'Medium': 100,
                                                'High': 300},
                                option_type='call',
                                n_paths=5000,
                                figsize=(14, 6)):
        """
        Side-by-side comparison : Heston baseline vs Heston-Pollution
        for each strike and pollution scenario.
        """
        rows = []
        for K in strikes:
            base_price, base_se = self.mc_price(K, T,
                                                 option_type=option_type,
                                                 n_paths=n_paths)
            for scenario, P in pm25_scenarios.items():
                pol_price, pol_se = pollution_pricer.mc_price(
                    K, T, P, option_type=option_type, n_paths=n_paths
                )
                rows.append({
                    'Strike K':      round(K, 2),
                    'Moneyness':     round(K / self.S0, 3),
                    'Scenario':      scenario,
                    'PM2.5':         P,
                    'Heston':        round(base_price, 4),
                    'Heston+Pol':    round(pol_price, 4),
                    'Premium (Pol)': round(pol_price - base_price, 4),
                    'Premium (%)':   round((pol_price / base_price - 1) * 100, 2),
                })

        df_comp = pd.DataFrame(rows)
        print(df_comp.to_string(index=False))

        # Plot pollution premium by moneyness
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for scenario, color in zip(['Low', 'Medium', 'High'],
                                    ['steelblue', 'orange', 'salmon']):
            sub = df_comp[df_comp['Scenario'] == scenario]
            axes[0].plot(sub['Moneyness'], sub['Heston+Pol'],
                         color=color, lw=2, label=f'Heston+Pol ({scenario})')
            axes[1].plot(sub['Moneyness'], sub['Premium (%)'],
                         color=color, lw=2, label=scenario)

        # Heston baseline
        base = df_comp[df_comp['Scenario'] == 'Low'][['Moneyness', 'Heston']].drop_duplicates()
        axes[0].plot(base['Moneyness'], base['Heston'],
                     color='black', lw=2, linestyle='--', label='Heston baseline')

        axes[0].set_title('Price by Strike — Heston vs Heston+Pollution', fontsize=11)
        axes[0].set_xlabel('Moneyness (K/S0)')
        axes[0].set_ylabel('Price')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].axhline(0, color='black', lw=0.8, linestyle='--')
        axes[1].set_title('Pollution Premium (%) over Heston baseline', fontsize=11)
        axes[1].set_xlabel('Moneyness (K/S0)')
        axes[1].set_ylabel('Premium (%)')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        fig.suptitle('Heston vs Heston-Pollution — Pricing Comparison', fontsize=13)
        plt.tight_layout()
        plt.show()
        return df_comp