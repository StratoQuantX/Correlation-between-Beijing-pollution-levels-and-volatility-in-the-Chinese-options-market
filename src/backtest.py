import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


class StraddleBacktest:
    """
    Long ATM Straddle strategy triggered by pollution signal.

    Signal  : PM2.5_t > rolling_mean(window) + rolling_std(window)
    Entry   : Buy ATM call + ATM put (straddle)
    Exit    : Hold for `holding_days`, then close
    P&L     : Approximated via BS vega × Δσ (no options market needed)

    Parameters
    ----------
    df           : pd.DataFrame — must contain realized_vol, pm25, Close
    sv_model     : PollutionSVModel — fitted model for σ_eff
    signal_window: int   — rolling window for dynamic threshold (default 60)
    holding_days : int   — days to hold each straddle (default 5)
    cost_bps     : float — transaction cost in bps (default 10)
    r            : float — risk-free rate (default 0.03)
    T_option     : float — option maturity in years (default 1/12 = 1 month)
    """

    def __init__(self, df, sv_model,
                 signal_window=60,
                 holding_days=5,
                 cost_bps=10,
                 r=0.03,
                 T_option=1/12):
        self.df            = df.copy()
        self.sv            = sv_model
        self.signal_window = signal_window
        self.holding_days  = holding_days
        self.cost_bps      = cost_bps / 10000
        self.r             = r
        self.T_option      = T_option

        self.trades_    = None
        self.equity_    = None
        self.metrics_   = None

    # ── Signal construction ───────────────────────────────────────
    def build_signal(self):
        df = self.df
        df['pm25_roll_mean'] = df['pm25'].rolling(self.signal_window).mean()
        df['pm25_roll_std']  = df['pm25'].rolling(self.signal_window).std()
        df['threshold']      = df['pm25_roll_mean'] + df['pm25_roll_std']
        df['signal']         = (df['pm25'] > df['threshold']).astype(int)

        # Avoid overlapping trades — signal only on first day of episode
        df['signal_entry'] = ((df['signal'] == 1) &
                               (df['signal'].shift(1) == 0)).astype(int)
        self.df = df
        return self

    # ── BS helpers ────────────────────────────────────────────────
    def _bs_price(self, S, K, T, sigma, option_type='call'):
        if sigma <= 0 or T <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-self.r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def _bs_vega(self, S, K, T, sigma):
        if sigma <= 0 or T <= 0:
            return 0
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)

    def _straddle_price(self, S, T, sigma):
        """ATM straddle = call + put, K = S"""
        call = self._bs_price(S, S, T, sigma, 'call')
        put  = self._bs_price(S, S, T, sigma, 'put')
        return call + put

    # ── Backtest ──────────────────────────────────────────────────
    def run(self):
        self.build_signal()
        df    = self.df
        trades = []

        i = self.signal_window  # skip warm-up
        while i < len(df) - self.holding_days:
            if df['signal_entry'].iloc[i] == 1:
                # ── Entry ─────────────────────────────────────────
                entry_date  = df.index[i]
                S_entry     = df['Close'].iloc[i]
                sigma_entry = df['realized_vol'].iloc[i]
                pm25_entry  = df['pm25'].iloc[i]
                T_rem       = self.T_option

                price_entry = self._straddle_price(S_entry, T_rem, sigma_entry)
                cost        = price_entry * self.cost_bps

                # ── Exit ──────────────────────────────────────────
                exit_idx    = min(i + self.holding_days, len(df) - 1)
                exit_date   = df.index[exit_idx]
                S_exit      = df['Close'].iloc[exit_idx]
                sigma_exit  = df['realized_vol'].iloc[exit_idx]
                T_rem_exit  = max(T_rem - self.holding_days/252, 1/252)
                pm25_exit   = df['pm25'].iloc[exit_idx]

                call_exit = self._bs_price(S_exit, S_entry, T_rem_exit, sigma_exit, 'call')
                put_exit  = self._bs_price(S_exit, S_entry, T_rem_exit, sigma_exit, 'put')
                price_exit = call_exit + put_exit
                pnl         = price_exit - price_entry - cost

                # Greeks at entry
                vega_entry  = 2 * self._bs_vega(S_entry, S_entry,
                                                  T_rem, sigma_entry)
                sigma_eff_entry = (self.sv.theta_
                                   + self.sv.delta_ * pm25_entry
                                   / self.sv.alpha_)
                vol_edge    = sigma_eff_entry - sigma_entry

                trades.append({
                    'entry_date':   entry_date,
                    'exit_date':    exit_date,
                    'S_entry':      round(S_entry, 2),
                    'sigma_entry':  round(sigma_entry, 4),
                    'sigma_eff':    round(sigma_eff_entry, 4),
                    'vol_edge':     round(vol_edge, 4),
                    'pm25_entry':   round(pm25_entry, 1),
                    'price_entry':  round(price_entry, 4),
                    'price_exit':   round(price_exit, 4),
                    'pnl':          round(pnl, 4),
                    'vega':         round(vega_entry, 4),
                    'cost':         round(cost, 4),
                })

                # Skip to after holding period to avoid overlap
                i += self.holding_days
            else:
                i += 1

        self.trades_ = pd.DataFrame(trades)
        self._compute_equity()
        self._compute_metrics()
        return self

    # ── Equity curve ──────────────────────────────────────────────
    def _compute_equity(self):
        trades = self.trades_.copy()
        trades = trades.set_index('entry_date').sort_index()
        trades['cum_pnl']  = trades['pnl'].cumsum()
        trades['drawdown'] = (trades['cum_pnl'].cummax()
                               - trades['cum_pnl'])
        self.equity_ = trades

    # ── Performance metrics ───────────────────────────────────────
    def _compute_metrics(self):
        pnl    = self.trades_['pnl']
        n      = len(pnl)
        wins   = (pnl > 0).sum()

        annual_factor = 252 / self.holding_days
        mean_pnl      = pnl.mean()
        std_pnl       = pnl.std()
        sharpe        = (mean_pnl / std_pnl * np.sqrt(annual_factor)
                         if std_pnl > 0 else np.nan)
        max_dd        = self.equity_['drawdown'].max()
        calmar        = (mean_pnl * annual_factor / max_dd
                         if max_dd > 0 else np.nan)

        self.metrics_ = {
            'N trades':       n,
            'Win rate':       round(wins / n, 4),
            'Mean P&L':       round(mean_pnl, 4),
            'Std P&L':        round(std_pnl, 4),
            'Sharpe':         round(sharpe, 4),
            'Max drawdown':   round(max_dd, 4),
            'Calmar':         round(calmar, 4),
            'Total P&L':      round(pnl.sum(), 4),
        }

    # ── Summary ───────────────────────────────────────────────────
    def summary(self):
        assert self.metrics_ is not None, "Run backtest first."
        print(f"\n{'='*50}")
        print(f"  Straddle Backtest — Pollution Signal")
        print(f"  Signal : PM2.5 > rolling mean + 1σ ({self.signal_window}d)")
        print(f"  Hold   : {self.holding_days} days | Cost: {self.cost_bps*10000}bps")
        print(f"{'='*50}")
        for k, v in self.metrics_.items():
            print(f"  {k:<20} {v}")
        print(f"{'='*50}\n")

    # ── Plots ─────────────────────────────────────────────────────
    def plot(self, figsize=(14, 12)):
        assert self.equity_ is not None, "Run backtest first."

        df     = self.df
        equity = self.equity_
        trades = self.trades_

        fig, axes = plt.subplots(4, 1, figsize=figsize)

        # Panel 1 — PM2.5 + signal + threshold
        axes[0].plot(df.index, df['pm25'],
                     color='salmon', lw=0.8, alpha=0.7, label='PM2.5')
        axes[0].plot(df.index, df['threshold'],
                     color='darkred', lw=1.2, linestyle='--',
                     label='Dynamic threshold (mean + 2σ)')
        entry_dates = trades['entry_date']
        axes[0].scatter(entry_dates,
                        df.loc[df.index.isin(entry_dates), 'pm25'],
                        color='red', s=20, zorder=5, label='Trade entry')
        axes[0].set_title('PM2.5 Signal — Dynamic Threshold', fontsize=11)
        axes[0].set_ylabel('PM2.5')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        # Panel 2 — Realized vol + σ_eff at entry
        axes[1].plot(df.index, df['realized_vol'],
                     color='steelblue', lw=0.8, label='Realized vol')
        axes[1].scatter(entry_dates,
                        trades.set_index('entry_date')['sigma_eff'],
                        color='orange', s=20, zorder=5,
                        label='σ_eff at entry (pollution-adjusted)')
        axes[1].set_title('Realized Vol vs σ_eff at Trade Entry', fontsize=11)
        axes[1].set_ylabel('Volatility')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        # Panel 3 — Cumulative P&L
        axes[2].plot(equity.index, equity['cum_pnl'],
                     color='steelblue', lw=1.5)
        axes[2].fill_between(equity.index, equity['cum_pnl'], 0,
                             where=(equity['cum_pnl'] > 0),
                             color='green', alpha=0.15)
        axes[2].fill_between(equity.index, equity['cum_pnl'], 0,
                             where=(equity['cum_pnl'] < 0),
                             color='red', alpha=0.15)
        axes[2].axhline(0, color='black', lw=0.8, linestyle='--')
        axes[2].set_title('Cumulative P&L', fontsize=11)
        axes[2].set_ylabel('P&L')
        axes[2].grid(alpha=0.3)

        # Panel 4 — Drawdown
        axes[3].fill_between(equity.index, -equity['drawdown'], 0,
                             color='red', alpha=0.4)
        axes[3].set_title('Drawdown', fontsize=11)
        axes[3].set_ylabel('Drawdown')
        axes[3].grid(alpha=0.3)

        fig.suptitle('Long Straddle Backtest — Pollution Signal\n'
                     f"Sharpe={self.metrics_['Sharpe']} | "
                     f"Win rate={self.metrics_['Win rate']} | "
                     f"N={self.metrics_['N trades']}",
                     fontsize=13)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
    


class StressTest:
    """
    Stress testing du StraddleBacktest sous différents régimes.

    Tests :
        1. Pollution spikes    — performance lors des épisodes PM2.5 extrêmes
        2. Vol regimes         — performance en low/mid/high vol environment
        3. Market crashes      — performance lors des drawdowns CSI 300 > 10%
        4. Subsample analysis  — rolling walk-forward par année
        5. Parameter sensitivity — robustesse à signal_window et holding_days
    """

    def __init__(self, df, sv_model):
        self.df  = df.copy()
        self.sv  = sv_model

    def _run_bt(self, df, **kwargs):
        bt = StraddleBacktest(df, self.sv, **kwargs)
        bt.run()
        return bt.metrics_

    # ── Test 1 : Pollution spikes ─────────────────────────────────
    def test_pollution_regimes(self, holding_days=20, figsize=(12, 5)):
        df = self.df
        p33 = df['pm25'].quantile(0.33)
        p66 = df['pm25'].quantile(0.66)

        regimes = {
            f'Low PM2.5 (< {p33:.0f})':          df[df['pm25'] <  p33],
            f'Mid PM2.5 ({p33:.0f}–{p66:.0f})':  df[(df['pm25'] >= p33) & (df['pm25'] < p66)],
            f'High PM2.5 (> {p66:.0f})':          df[df['pm25'] >= p66],
        }

        rows = []
        for label, subset in regimes.items():
            if len(subset) < 200:
                continue
            m = self._run_bt(subset, holding_days=holding_days)
            m['Regime'] = label
            rows.append(m)

        result = pd.DataFrame(rows).set_index('Regime')
        print("\n── Pollution Regime Analysis ──")
        print(result[['N trades', 'Win rate', 'Sharpe',
                       'Mean P&L', 'Max drawdown']].to_string())

        # Bar chart
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, col in zip(axes, ['Sharpe', 'Win rate', 'Mean P&L']):
            result[col].plot.bar(ax=ax, color=['steelblue','orange','salmon'],
                                  alpha=0.8)
            ax.set_title(col, fontsize=11)
            ax.axhline(0, color='black', lw=0.8, linestyle='--')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
            ax.grid(alpha=0.3)

        plt.suptitle('Strategy Performance by Pollution Regime', fontsize=13)
        plt.tight_layout()
        plt.show()
        return result

    # ── Test 2 : Vol regimes ──────────────────────────────────────
    def test_vol_regimes(self, holding_days=20, figsize=(12, 5)):
        df  = self.df
        v33 = df['realized_vol'].quantile(0.33)
        v66 = df['realized_vol'].quantile(0.66)

        regimes = {
            f'Low vol (< {v33:.2f})':          df[df['realized_vol'] <  v33],
            f'Mid vol ({v33:.2f}–{v66:.2f})':  df[(df['realized_vol'] >= v33) &
                                                    (df['realized_vol'] <  v66)],
            f'High vol (> {v66:.2f})':          df[df['realized_vol'] >= v66],
        }

        rows = []
        for label, subset in regimes.items():
            if len(subset) < 200:
                continue
            m = self._run_bt(subset, holding_days=holding_days)
            m['Regime'] = label
            rows.append(m)

        result = pd.DataFrame(rows).set_index('Regime')
        print("\n── Vol Regime Analysis ──")
        print(result[['N trades', 'Win rate', 'Sharpe',
                       'Mean P&L', 'Max drawdown']].to_string())

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for ax, col in zip(axes, ['Sharpe', 'Win rate', 'Mean P&L']):
            result[col].plot.bar(ax=ax, color=['steelblue','orange','salmon'],
                                  alpha=0.8)
            ax.set_title(col, fontsize=11)
            ax.axhline(0, color='black', lw=0.8, linestyle='--')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
            ax.grid(alpha=0.3)

        plt.suptitle('Strategy Performance by Volatility Regime', fontsize=13)
        plt.tight_layout()
        plt.show()
        return result

    # ── Test 3 : Market crashes ───────────────────────────────────
    def test_crash_periods(self, holding_days=20, crash_threshold=-0.10):
        df = self.df.copy()

        # Identify crash windows : CSI 300 drawdown > threshold
        roll_max   = df['Close'].rolling(60).max()
        df['dd']   = df['Close'] / roll_max - 1
        crash_mask = df['dd'] < crash_threshold

        crash_df  = df[crash_mask]
        normal_df = df[~crash_mask]

        rows = []
        for label, subset in [('Crash periods', crash_df),
                               ('Normal periods', normal_df)]:
            if len(subset) < 200:
                rows.append({'Regime': label, 'N trades': 0,
                             'Sharpe': np.nan, 'Win rate': np.nan,
                             'Mean P&L': np.nan, 'Max drawdown': np.nan})
                continue
            m = self._run_bt(subset, holding_days=holding_days)
            m['Regime'] = label
            rows.append(m)

        result = pd.DataFrame(rows).set_index('Regime')
        print("\n── Crash Period Analysis ──")
        print(result[['N trades', 'Win rate', 'Sharpe',
                       'Mean P&L', 'Max drawdown']].to_string())
        return result

    # ── Test 4 : Walk-forward by year ─────────────────────────────
    def test_walk_forward(self, holding_days=5, figsize=(12, 5)):
        df    = self.df
        years = sorted(df.index.year.unique())
        rows  = []

        for year in years:
            subset = df[df.index.year == year]
            if len(subset) < 100:
                continue
            try:
                m = self._run_bt(subset, holding_days=holding_days,
                                  signal_window=60)
                m['Year'] = year
                rows.append(m)
            except Exception:
                continue

        result = pd.DataFrame(rows).set_index('Year')
        print("\n── Walk-Forward Analysis (by year) ──")
        print(result[['N trades', 'Win rate', 'Sharpe', 'Mean P&L']].to_string())

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        result['Sharpe'].plot.bar(ax=axes[0],
                                   color=['green' if x > 0 else 'red'
                                          for x in result['Sharpe']],
                                   alpha=0.8)
        axes[0].axhline(0, color='black', lw=0.8, linestyle='--')
        axes[0].set_title('Sharpe by year', fontsize=11)
        axes[0].grid(alpha=0.3)

        result['Win rate'].plot.bar(ax=axes[1], color='steelblue', alpha=0.8)
        axes[1].axhline(0.5, color='red', lw=0.8, linestyle='--',
                         label='50% win rate')
        axes[1].set_title('Win rate by year', fontsize=11)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle('Walk-Forward Analysis — Annual Performance', fontsize=13)
        plt.tight_layout()
        plt.show()
        return result

    # ── Test 5 : Parameter sensitivity heatmap ────────────────────
    def test_parameter_sensitivity(self,
                                    windows=[30, 45, 60, 90, 120],
                                    holdings=[5, 10, 20, 30, 50],
                                    figsize=(10, 7)):
        sharpe_matrix = np.zeros((len(windows), len(holdings)))
        wr_matrix     = np.zeros((len(windows), len(holdings)))

        for i, w in enumerate(windows):
            for j, h in enumerate(holdings):
                try:
                    m = self._run_bt(self.df,
                                     signal_window=w,
                                     holding_days=h)
                    sharpe_matrix[i, j] = m['Sharpe']
                    wr_matrix[i, j]     = m['Win rate']
                except Exception:
                    sharpe_matrix[i, j] = np.nan
                    wr_matrix[i, j]     = np.nan

        fig, axes = plt.subplots(1, 2, figsize=(16, 9))

        for ax, matrix, title, fmt in zip(
            axes,
            [sharpe_matrix, wr_matrix],
            ['Sharpe ratio', 'Win rate'],
            ['.2f', '.2f']
        ):
            im = ax.imshow(matrix, cmap='coolwarm', aspect='auto',
                           vmin=0 if 'Sharpe' in title else 0.35,
                           vmax=1.5  if 'Sharpe' in title else 0.8)
            ax.set_xticks(range(len(holdings)))
            ax.set_xticklabels([f'{h}d' for h in holdings])
            ax.set_yticks(range(len(windows)))
            ax.set_yticklabels([f'{w}d' for w in windows])
            ax.set_xlabel('Holding period')
            ax.set_ylabel('Signal window')
            ax.set_title(title, fontsize=11)
            plt.colorbar(im, ax=ax)

            for ii in range(len(windows)):
                for jj in range(len(holdings)):
                    ax.text(jj, ii, f'{matrix[ii,jj]:{fmt}}',
                            ha='center', va='center', fontsize=8,
                            color='black')

        plt.suptitle('Parameter Sensitivity Heatmap', fontsize=13)
        plt.tight_layout()
        plt.show()

        return pd.DataFrame(sharpe_matrix,
                            index=[f'window={w}' for w in windows],
                            columns=[f'hold={h}' for h in holdings])
    


class RegimeSwitchingBacktest:
    """
    Regime-switching volatility strategy driven by pollution signal.

    Long vol regime  : PM2.5 > roll_mean + 2*roll_std  → long ATM straddle
    Short vol regime : PM2.5 < roll_mean - 2*roll_std  → short OTM strangle
    Neutral zone     : flat

    Short strangle = sell OTM call (K=S×(1+short_width))
                   + sell OTM put  (K=S×(1-short_width))
    Capped loss via stop-loss at `stop_loss_mult` × premium received.
    """

    def __init__(self, df, sv_model,
                 signal_window=60,
                 holding_days=5,
                 short_width=0.05,
                 stop_loss_mult=2.0,
                 cost_bps=10,
                 r=0.03,
                 T_option=1/12):
        self.df             = df.copy()
        self.sv             = sv_model
        self.signal_window  = signal_window
        self.holding_days   = holding_days
        self.short_width    = short_width
        self.stop_loss_mult = stop_loss_mult
        self.cost_bps       = cost_bps / 10000
        self.r              = r
        self.T_option       = T_option

        self.trades_  = None
        self.equity_  = None
        self.metrics_ = None

    # ── BS helpers ────────────────────────────────────────────────
    def _bs_price(self, S, K, T, sigma, option_type='call'):
        if sigma <= 0 or T <= 0:
            return max(S-K, 0) if option_type=='call' else max(K-S, 0)
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-self.r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def _bs_delta(self, S, K, T, sigma, option_type='call'):
        if sigma <= 0 or T <= 0:
            return 0
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return norm.cdf(d1) if option_type=='call' else norm.cdf(d1) - 1

    # ── Signal ────────────────────────────────────────────────────
    def build_signal(self):
        df = self.df
        df['roll_mean'] = df['pm25'].rolling(self.signal_window).mean()
        df['roll_std']  = df['pm25'].rolling(self.signal_window).std()
        df['upper']     = df['roll_mean'] + 2*df['roll_std']
        df['lower']     = df['roll_mean'] - 2*df['roll_std']

        df['regime'] = 'neutral'
        df.loc[df['pm25'] > df['upper'], 'regime'] = 'long_vol'
        df.loc[df['pm25'] < df['lower'], 'regime'] = 'short_vol'

        # Entry on regime change only
        df['regime_shift'] = df['regime'] != df['regime'].shift(1)
        df['entry_signal'] = (df['regime_shift'] &
                               (df['regime'] != 'neutral'))
        self.df = df
        return self

    # ── Run ───────────────────────────────────────────────────────
    def run(self):
        self.build_signal()
        df     = self.df
        trades = []

        i = self.signal_window
        while i < len(df) - self.holding_days:
            if not df['entry_signal'].iloc[i]:
                i += 1
                continue

            regime     = df['regime'].iloc[i]
            entry_date = df.index[i]
            S          = df['Close'].iloc[i]
            sigma      = df['realized_vol'].iloc[i]
            T          = self.T_option
            T_rem_exit = max(T - self.holding_days/252, 1/252)

            # ── Long straddle (long vol) ───────────────────────────
            if regime == 'long_vol':
                price_entry = (self._bs_price(S, S, T, sigma, 'call') +
                               self._bs_price(S, S, T, sigma, 'put'))
                cost        = price_entry * self.cost_bps

                exit_idx    = min(i + self.holding_days, len(df)-1)
                S_exit      = df['Close'].iloc[exit_idx]
                sigma_exit  = df['realized_vol'].iloc[exit_idx]
                price_exit  = (self._bs_price(S_exit, S, T_rem_exit,
                                               sigma_exit, 'call') +
                                self._bs_price(S_exit, S, T_rem_exit,
                                               sigma_exit, 'put'))
                pnl = price_exit - price_entry - cost

            # ── Short strangle (short vol) ────────────────────────
            else:
                K_call = S * (1 + self.short_width)
                K_put  = S * (1 - self.short_width)
                premium = (self._bs_price(S, K_call, T, sigma, 'call') +
                            self._bs_price(S, K_put,  T, sigma, 'put'))
                cost    = premium * self.cost_bps
                stop_loss = self.stop_loss_mult * premium

                exit_idx   = min(i + self.holding_days, len(df)-1)
                S_exit     = df['Close'].iloc[exit_idx]
                sigma_exit = df['realized_vol'].iloc[exit_idx]
                cost_to_close = (
                    self._bs_price(S_exit, K_call, T_rem_exit, sigma_exit, 'call') +
                    self._bs_price(S_exit, K_put,  T_rem_exit, sigma_exit, 'put')
                )
                raw_pnl = premium - cost_to_close - cost
                # Apply stop-loss
                pnl = max(raw_pnl, -stop_loss)

            trades.append({
                'entry_date': entry_date,
                'exit_date':  df.index[min(i + self.holding_days, len(df)-1)],
                'regime':     regime,
                'S_entry':    round(S, 2),
                'sigma':      round(sigma, 4),
                'pm25':       round(df['pm25'].iloc[i], 1),
                'pnl':        round(pnl, 4),
            })
            i += self.holding_days

        self.trades_ = pd.DataFrame(trades)
        self._compute_equity()
        self._compute_metrics()
        return self

    def _compute_equity(self):
        t = self.trades_.set_index('entry_date').sort_index()
        t['cum_pnl']  = t['pnl'].cumsum()
        t['drawdown'] = t['cum_pnl'].cummax() - t['cum_pnl']
        self.equity_  = t

    def _compute_metrics(self):
        pnl    = self.trades_['pnl']
        n      = len(pnl)
        wins   = (pnl > 0).sum()
        af     = 252 / self.holding_days
        sharpe = pnl.mean() / pnl.std() * np.sqrt(af) if pnl.std() > 0 else np.nan
        max_dd = self.equity_['drawdown'].max()

        long_trades  = self.trades_[self.trades_['regime'] == 'long_vol']
        short_trades = self.trades_[self.trades_['regime'] == 'short_vol']

        self.metrics_ = {
            'N trades':        n,
            'N long vol':      len(long_trades),
            'N short vol':     len(short_trades),
            'Win rate':        round(wins/n, 4),
            'WR long vol':     round((long_trades['pnl'] > 0).mean(), 4)
                               if len(long_trades) else np.nan,
            'WR short vol':    round((short_trades['pnl'] > 0).mean(), 4)
                               if len(short_trades) else np.nan,
            'Mean P&L':        round(pnl.mean(), 4),
            'Sharpe':          round(sharpe, 4),
            'Max drawdown':    round(max_dd, 4),
            'Total P&L':       round(pnl.sum(), 4),
        }

    def summary(self):
        print(f"\n{'='*55}")
        print(f"  Regime-Switching Backtest")
        print(f"  Long vol : PM2.5 > mean+2σ | Short vol : PM2.5 < mean-2σ")
        print(f"  Window={self.signal_window}d | Hold={self.holding_days}d")
        print(f"{'='*55}")
        for k, v in self.metrics_.items():
            print(f"  {k:<22} {v}")
        print(f"{'='*55}\n")

    def plot(self, figsize=(14, 12)):
        df     = self.df
        equity = self.equity_
        trades = self.trades_

        fig, axes = plt.subplots(4, 1, figsize=figsize)

        # Panel 1 — PM2.5 + regimes
        axes[0].plot(df.index, df['pm25'],
                     color='salmon', lw=0.8, alpha=0.6)
        axes[0].plot(df.index, df['upper'],
                     color='darkred', lw=1.2, linestyle='--',
                     label='Upper (long vol threshold)')
        axes[0].plot(df.index, df['lower'],
                     color='steelblue', lw=1.2, linestyle='--',
                     label='Lower (short vol threshold)')
        axes[0].fill_between(df.index, df['upper'],
                              df['pm25'].max() * 1.1,
                              where=(df['pm25'] > df['upper']),
                              color='red', alpha=0.1, label='Long vol zone')
        axes[0].fill_between(df.index, 0, df['lower'],
                              where=(df['pm25'] < df['lower']),
                              color='blue', alpha=0.1, label='Short vol zone')
        axes[0].set_title('Pollution Regimes — Long/Short Vol Zones', fontsize=11)
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        # Panel 2 — Trade P&L by regime
        long_t  = trades[trades['regime'] == 'long_vol']
        short_t = trades[trades['regime'] == 'short_vol']
        axes[1].bar(long_t['entry_date'],  long_t['pnl'],
                    color=['green' if p > 0 else 'red' for p in long_t['pnl']],
                    alpha=0.7, width=10, label='Long vol trades')
        axes[1].bar(short_t['entry_date'], short_t['pnl'],
                    color=['purple' if p > 0 else 'orange' for p in short_t['pnl']],
                    alpha=0.7, width=10, label='Short vol trades')
        axes[1].axhline(0, color='black', lw=0.8)
        axes[1].set_title('Individual Trade P&L by Regime', fontsize=11)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        # Panel 3 — Cumulative P&L
        axes[2].plot(equity.index, equity['cum_pnl'],
                     color='steelblue', lw=1.5)
        axes[2].fill_between(equity.index, equity['cum_pnl'], 0,
                              where=(equity['cum_pnl'] > 0),
                              color='green', alpha=0.15)
        axes[2].fill_between(equity.index, equity['cum_pnl'], 0,
                              where=(equity['cum_pnl'] < 0),
                              color='red', alpha=0.15)
        axes[2].axhline(0, color='black', lw=0.8, linestyle='--')
        axes[2].set_title('Cumulative P&L', fontsize=11)
        axes[2].grid(alpha=0.3)

        # Panel 4 — Drawdown
        axes[3].fill_between(equity.index, -equity['drawdown'], 0,
                              color='red', alpha=0.4)
        axes[3].set_title('Drawdown', fontsize=11)
        axes[3].grid(alpha=0.3)

        fig.suptitle(f'Regime-Switching Backtest\n'
                     f"Sharpe={self.metrics_['Sharpe']} | "
                     f"WR={self.metrics_['Win rate']} | "
                     f"N={self.metrics_['N trades']}",
                     fontsize=13)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()



class DeltaNeutralStrangleBacktest:
    """
    Delta-neutral long strangle triggered by pollution signal.

    Entry  : PM2.5 > dynamic threshold
             Buy OTM call (K = S×(1+width)) + OTM put (K = S×(1-width))
    Hedge  : Daily delta rebalancing via underlying
    Exit   : After holding_days

    The delta hedge converts the position to pure vega/gamma exposure —
    P&L driven entirely by vol realization vs vol at entry.
    """

    def __init__(self, df, sv_model,
                 signal_window=30,
                 holding_days=50,
                 width=0.05,
                 cost_bps=10,
                 hedge_cost_bps=2,
                 r=0.03,
                 T_option=1/12):
        self.df              = df.copy()
        self.sv              = sv_model
        self.signal_window   = signal_window
        self.holding_days    = holding_days
        self.width           = width
        self.cost_bps        = cost_bps / 10000
        self.hedge_cost_bps  = hedge_cost_bps / 10000
        self.r               = r
        self.T_option        = T_option

        self.trades_  = None
        self.equity_  = None
        self.metrics_ = None

    def _bs_price(self, S, K, T, sigma, option_type='call'):
        if sigma <= 0 or T <= 0:
            return max(S-K, 0) if option_type=='call' else max(K-S, 0)
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        if option_type == 'call':
            return S*norm.cdf(d1) - K*np.exp(-self.r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-self.r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    def _bs_delta(self, S, K, T, sigma, option_type='call'):
        if sigma <= 0 or T <= 0:
            return 0
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return norm.cdf(d1) if option_type=='call' else norm.cdf(d1) - 1

    def _bs_vega(self, S, K, T, sigma):
        if sigma <= 0 or T <= 0:
            return 0
        d1 = (np.log(S/K) + (self.r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)

    def build_signal(self):
        df = self.df
        df['roll_mean']  = df['pm25'].rolling(self.signal_window).mean()
        df['roll_std']   = df['pm25'].rolling(self.signal_window).std()
        df['threshold']  = df['roll_mean'] + df['roll_std']
        df['signal']     = (df['pm25'] > df['threshold']).astype(int)
        df['entry_signal'] = ((df['signal'] == 1) &
                               (df['signal'].shift(1) == 0)).astype(int)
        self.df = df
        return self

    def run(self):
        self.build_signal()
        df     = self.df
        trades = []

        i = self.signal_window
        while i < len(df) - self.holding_days:
            if df['entry_signal'].iloc[i] != 1:
                i += 1
                continue

            entry_date = df.index[i]
            S0         = df['Close'].iloc[i]
            sigma0     = df['realized_vol'].iloc[i]
            T          = self.T_option
            K_call     = S0 * (1 + self.width)
            K_put      = S0 * (1 - self.width)

            # Entry cost
            call_entry = self._bs_price(S0, K_call, T, sigma0, 'call')
            put_entry  = self._bs_price(S0, K_put,  T, sigma0, 'put')
            premium    = call_entry + put_entry
            entry_cost = premium * self.cost_bps

            # Initial delta
            delta_call = self._bs_delta(S0, K_call, T, sigma0, 'call')
            delta_put  = self._bs_delta(S0, K_put,  T, sigma0, 'put')
            net_delta  = delta_call + delta_put
            hedge_pnl  = 0.0
            hedge_cost = 0.0
            prev_delta = net_delta
            S_prev     = S0

            # Daily delta rebalancing
            exit_idx = min(i + self.holding_days, len(df)-1)
            for t in range(i+1, exit_idx+1):
                T_rem      = max(T - (t-i)/252, 1/252)
                S_t        = df['Close'].iloc[t]
                sigma_t    = df['realized_vol'].iloc[t]

                delta_call = self._bs_delta(S_t, K_call, T_rem, sigma_t, 'call')
                delta_put  = self._bs_delta(S_t, K_put,  T_rem, sigma_t, 'put')
                new_delta  = delta_call + delta_put

                # P&L from hedge rebalancing
                delta_change = new_delta - prev_delta
                hedge_pnl   += -prev_delta * (S_t - S_prev)
                hedge_cost  += abs(delta_change) * S_t * self.hedge_cost_bps

                prev_delta = new_delta
                S_prev     = S_t

            # Exit option value
            T_exit     = max(T - self.holding_days/252, 1/252)
            S_exit     = df['Close'].iloc[exit_idx]
            sigma_exit = df['realized_vol'].iloc[exit_idx]
            call_exit  = self._bs_price(S_exit, K_call, T_exit, sigma_exit, 'call')
            put_exit   = self._bs_price(S_exit, K_put,  T_exit, sigma_exit, 'put')
            exit_value = call_exit + put_exit

            # Total P&L
            option_pnl = exit_value - premium
            total_pnl  = option_pnl + hedge_pnl - entry_cost - hedge_cost

            # Vega at entry
            vega = (self._bs_vega(S0, K_call, T, sigma0) +
                    self._bs_vega(S0, K_put,  T, sigma0))
            vol_edge = (self.sv.theta_ +
                        self.sv.delta_ * df['pm25'].iloc[i] / self.sv.alpha_
                        - sigma0)

            trades.append({
                'entry_date':  entry_date,
                'exit_date':   df.index[exit_idx],
                'S_entry':     round(S0, 2),
                'K_call':      round(K_call, 2),
                'K_put':       round(K_put, 2),
                'sigma_entry': round(sigma0, 4),
                'vol_edge':    round(vol_edge, 4),
                'pm25':        round(df['pm25'].iloc[i], 1),
                'premium':     round(premium, 4),
                'option_pnl':  round(option_pnl, 4),
                'hedge_pnl':   round(hedge_pnl, 4),
                'hedge_cost':  round(hedge_cost, 4),
                'total_pnl':   round(total_pnl, 4),
                'vega':        round(vega, 4),
            })

            i += self.holding_days

        self.trades_ = pd.DataFrame(trades)
        self.trades_ = self.trades_.rename(columns={'total_pnl': 'pnl'})
        self._compute_equity()
        self._compute_metrics()
        return self

    def _compute_equity(self):
        t = self.trades_.set_index('entry_date').sort_index()
        t['cum_pnl']  = t['pnl'].cumsum()
        t['drawdown'] = t['cum_pnl'].cummax() - t['cum_pnl']
        self.equity_  = t

    def _compute_metrics(self):
        pnl    = self.trades_['pnl']
        n      = len(pnl)
        wins   = (pnl > 0).sum()
        af     = 252 / self.holding_days
        sharpe = pnl.mean() / pnl.std() * np.sqrt(af) if pnl.std() > 0 else np.nan
        max_dd = self.equity_['drawdown'].max()

        self.metrics_ = {
            'N trades':      n,
            'Win rate':      round(wins/n, 4),
            'Mean P&L':      round(pnl.mean(), 4),
            'Mean option PnL':round(self.trades_['option_pnl'].mean(), 4),
            'Mean hedge PnL': round(self.trades_['hedge_pnl'].mean(), 4),
            'Mean hedge cost':round(self.trades_['hedge_cost'].mean(), 4),
            'Sharpe':        round(sharpe, 4),
            'Max drawdown':  round(max_dd, 4),
            'Total P&L':     round(pnl.sum(), 4),
        }

    def summary(self):
        print(f"\n{'='*55}")
        print(f"  Delta-Neutral Strangle Backtest")
        print(f"  Width={self.width*100:.0f}% OTM | Window={self.signal_window}d"
              f" | Hold={self.holding_days}d")
        print(f"{'='*55}")
        for k, v in self.metrics_.items():
            print(f"  {k:<25} {v}")
        print(f"{'='*55}\n")

    def plot(self, figsize=(14, 12)):
        equity = self.equity_
        trades = self.trades_
        df     = self.df

        fig, axes = plt.subplots(4, 1, figsize=figsize)

        # Panel 1 — PM2.5 + signal
        axes[0].plot(df.index, df['pm25'],
                     color='salmon', lw=0.8, alpha=0.7)
        axes[0].plot(df.index, df['threshold'],
                     color='darkred', lw=1.2, linestyle='--',
                     label='Dynamic threshold')
        axes[0].scatter(trades['entry_date'],
                        df.loc[df.index.isin(trades['entry_date']), 'pm25'],
                        color='red', s=25, zorder=5, label='Entry')
        axes[0].set_title('PM2.5 Signal', fontsize=11)
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        # Panel 2 — P&L decomposition
        width_bar = pd.Timedelta(days=8)
        axes[1].bar(trades['entry_date'], trades['option_pnl'],
                    color='steelblue', alpha=0.7, width=15, label='Option P&L')
        axes[1].bar(trades['entry_date'], trades['hedge_pnl'],
                    color='orange', alpha=0.7, width=15,
                    bottom=trades['option_pnl'], label='Hedge P&L')
        axes[1].axhline(0, color='black', lw=0.8)
        axes[1].set_title('P&L Decomposition — Option vs Delta Hedge', fontsize=11)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        # Panel 3 — Cumulative P&L
        axes[2].plot(equity.index, equity['cum_pnl'],
                     color='steelblue', lw=1.5)
        axes[2].fill_between(equity.index, equity['cum_pnl'], 0,
                              where=(equity['cum_pnl'] > 0),
                              color='green', alpha=0.15)
        axes[2].fill_between(equity.index, equity['cum_pnl'], 0,
                              where=(equity['cum_pnl'] < 0),
                              color='red', alpha=0.15)
        axes[2].axhline(0, color='black', lw=0.8, linestyle='--')
        axes[2].set_title('Cumulative P&L', fontsize=11)
        axes[2].grid(alpha=0.3)

        # Panel 4 — Drawdown
        axes[3].fill_between(equity.index, -equity['drawdown'], 0,
                              color='red', alpha=0.4)
        axes[3].set_title('Drawdown', fontsize=11)
        axes[3].grid(alpha=0.3)

        fig.suptitle(f'Delta-Neutral Strangle Backtest\n'
                     f"Sharpe={self.metrics_['Sharpe']} | "
                     f"WR={self.metrics_['Win rate']} | "
                     f"N={self.metrics_['N trades']}",
                     fontsize=13)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()