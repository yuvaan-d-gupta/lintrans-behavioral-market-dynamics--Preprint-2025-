
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PERIODS_PER_YEAR = 52


def annualized_sharpe(mean_w: float, std_w: float, rf: float = 0.0) -> float:
    if std_w <= 0 or np.isnan(std_w):
        return np.nan
    return (mean_w / std_w) * np.sqrt(PERIODS_PER_YEAR)


def rolling_sharpe(series: pd.Series, window: int, rf: float = 0.0) -> pd.Series:
    # rf is an annualized rate; convert to weekly
    weekly_rf = rf / PERIODS_PER_YEAR
    excess = series - weekly_rf
    roll_mean = excess.rolling(window, min_periods=window).mean()
    roll_std = excess.rolling(window, min_periods=window).std(ddof=0)
    return (roll_mean / roll_std) * np.sqrt(PERIODS_PER_YEAR)


def expanding_sharpe(series: pd.Series, rf: float = 0.0) -> pd.Series:
    weekly_rf = rf / PERIODS_PER_YEAR
    excess = series - weekly_rf
    m = excess.expanding(min_periods=26).mean()
    s = excess.expanding(min_periods=26).std(ddof=0)
    return (m / s) * np.sqrt(PERIODS_PER_YEAR)


def simulate_weekly_returns(target_sharpe: float, weeks: int, weekly_vol: float,
                             rf: float, seed: int | None = None,
                             ar1: float = 0.0) -> np.ndarray:
    """Simulate weekly returns with approx. target annualized Sharpe.
    - weekly_vol is the desired *weekly* standard deviation (e.g., 0.018 ≈ 1.8%).
    - ar1 adds optional autocorrelation in returns (0 = i.i.d.).
    """
    rng = np.random.default_rng(seed)
    mu_w = (target_sharpe / np.sqrt(PERIODS_PER_YEAR)) * weekly_vol

    eps = rng.normal(0.0, weekly_vol, size=weeks)
    r = np.zeros(weeks)
    for t in range(weeks):
        if t == 0:
            r[t] = mu_w + eps[t]
        else:
            r[t] = mu_w + ar1 * (r[t-1] - mu_w) + eps[t]

    # Calibrate drift so realized Sharpe ≈ target
    realized_std = r.std(ddof=0)
    realized_mean = r.mean()
    realized_s = annualized_sharpe(realized_mean - rf/52, realized_std)
    if np.isfinite(realized_s) and realized_s != 0:
        scale = target_sharpe / realized_s
        r = (r - r.mean()) + r.mean() * scale
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target_sharpe', type=float, default=0.4266)
    ap.add_argument('--weeks', type=int, default=720)
    ap.add_argument('--weekly_vol', type=float, default=0.018)
    ap.add_argument('--rf', type=float, default=0.0, help='Annualized risk‑free rate (e.g., 0.015)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ar1', type=float, default=0.0, help='Optional AR(1) in returns')
    ap.add_argument('--win', type=int, default=52)
    ap.add_argument('--alt_win', type=int, default=26)
    ap.add_argument('--fig', default='outputs/rolling_sharpe_sim.png')
    args = ap.parse_args()

    # Simulate weekly returns for the strategy (no CSV required)
    strat = simulate_weekly_returns(args.target_sharpe, args.weeks, args.weekly_vol, args.rf, args.seed, args.ar1)

    idx = pd.date_range('2011-01-07', periods=args.weeks, freq='W-FRI')
    df = pd.DataFrame({'Date': idx, 'strat_ret': strat})

    # Rolling/expanding Sharpes
    df[f'Sharpe_{args.win}w'] = rolling_sharpe(df['strat_ret'], args.win, rf=args.rf)
    df[f'Sharpe_{args.alt_win}w'] = rolling_sharpe(df['strat_ret'], args.alt_win, rf=args.rf)
    df['Sharpe_expanding'] = expanding_sharpe(df['strat_ret'], rf=args.rf)

    # Realized sample Sharpe
    realized_S = annualized_sharpe(df['strat_ret'].mean() - args.rf/52, df['strat_ret'].std(ddof=0))

    # Plot
    fig = plt.figure(figsize=(12, 9))

    ax1 = plt.subplot2grid((3,1), (0,0))
    ax1.plot(df['Date'], df[f'Sharpe_{args.win}w'], label=f'Rolling Sharpe ({args.win}w)')
    ax1.plot(df['Date'], df[f'Sharpe_{args.alt_win}w'], label=f'Rolling Sharpe ({args.alt_win}w)', alpha=0.8)
    ax1.axhline(0, color='black', lw=1, ls='--')
    ax1.set_ylabel('Sharpe')
    ax1.set_title(f'Rolling Annualized Sharpe (target={args.target_sharpe:.3f}, realized={realized_S:.3f})')
    ax1.legend(loc='upper left')

    ax2 = plt.subplot2grid((3,1), (1,0))
    ax2.plot(df['Date'], df['Sharpe_expanding'])
    ax2.axhline(0, color='black', lw=1, ls='--')
    ax2.set_ylabel('Sharpe')
    ax2.set_title('Expanding‑Window Annualized Sharpe')

    # Annual Sharpes
    yr = df.set_index('Date')['strat_ret'].groupby(pd.Grouper(freq='Y'))
    annual_S = (yr.mean() / yr.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR)).dropna()
    ax3 = plt.subplot2grid((3,1), (2,0))
    ax3.bar(annual_S.index.year.astype(int), annual_S.values)
    ax3.axhline(0, color='black', lw=1, ls='--')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Sharpe')
    ax3.set_title('Calendar‑Year Sharpe (Annualized)')

    fig.tight_layout()
    out = Path(args.fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved: {out}  |  Realized Sharpe ≈ {realized_S:.4f}')

if __name__ == '__main__':
    main()
