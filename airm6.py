"""
airm6.py -- shared helpers for AIRM Lecture 6: Risk Management with AI.

Everything the four Lecture 6 notebooks need in one place: data loading with a
local cache, risk measures, conditional volatility models, backtesting
statistics, risk decomposition, and a Plotly theme that matches the lecture
slides.

Data
----
The default dataset is the Plotly S&P 500 five-year constituent panel
(Feb 2013 -- Feb 2018). It is used throughout the subject, it is small enough
for Binder, and every number quoted on the Lecture 6 slides was computed from
it -- so results here reconcile with the deck exactly.

Set ``USE_LIVE = True`` in a notebook and call ``get_prices(..., use_live=True)``
to pull a current window from Yahoo Finance instead. Live data will not match
the slide numbers, which is the point of the Stretch exercises.

Conventions
-----------
* ``r`` is always a series of **log returns**; negative values are losses.
* Risk measures return **positive loss magnitudes** (a 99% VaR of 2.3% is
  returned as ``0.023``), which is how they appear on the slides.
* ``alpha`` is always a **confidence level** (0.99 means 99%), never a tail
  probability. Riskfolio-Lib uses the opposite convention; ``rp_alpha()``
  converts.
"""

from __future__ import annotations

import os
import urllib.request
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

__all__ = [
    "NAVY", "BAND", "ORANGE", "RED", "GREY", "TEAL", "SERIES",
    "base_layout", "style",
    "load_panel", "load_live", "get_prices", "log_returns", "portfolio_returns",
    "rp_alpha",
    "var_hist", "es_hist", "var_normal", "es_normal", "var_t", "es_t",
    "risk_table",
    "ewma_var", "har_design", "realised_var_proxy", "qlike", "mse_loss",
    "loss_table",
    "breaches", "kupiec", "christoffersen_ind", "conditional_coverage",
    "basel_zone", "rolling_zone_summary", "kupiec_power",
    "risk_contributions", "inverse_vol_weights", "erc_weights",
    "annualise_vol",
    "pinball_loss", "fz0_loss", "diebold_mariano",
    "vol_target_leverage", "simulate_feedback",
]

# ---------------------------------------------------------------------------
# Palette and Plotly theme -- matches the Beamer deck (navy / pale band)
# ---------------------------------------------------------------------------

NAVY = "#123F69"
BAND = "#E2EBF4"
ORANGE = "#E38B29"
RED = "#B22222"
GREY = "#8A8A8A"
TEAL = "#2E7D7B"
SERIES = [NAVY, ORANGE, GREY, TEAL, RED]


def base_layout(title=None, xtitle=None, ytitle=None, height=430, legend_top=True):
    """Return a Plotly layout dict styled to match the lecture slides."""
    layout = dict(
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=30, t=60 if title else 30, b=50),
        font=dict(family="Helvetica, Arial, sans-serif", size=12, color="#222222"),
        colorway=SERIES,
        hovermode="x unified",
    )
    if title:
        layout["title"] = dict(text=f"<b>{title}</b>", font=dict(color=NAVY, size=15), x=0.01)
    if xtitle:
        layout["xaxis"] = dict(title=xtitle)
    if ytitle:
        layout["yaxis"] = dict(title=ytitle)
    if legend_top:
        layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.0,
                                xanchor="left", x=0, font=dict(size=11))
    return layout


def style(fig, **kwargs):
    """Apply the lecture theme to an existing figure. Returns the figure."""
    fig.update_layout(**base_layout(**kwargs))
    fig.update_xaxes(showgrid=True, gridcolor="#EDEDED", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EDEDED", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

PANEL_URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"
CACHE_DIR = "data"
CACHE_FILE = "all_stocks_5yr.csv"


def load_panel(cache_dir=CACHE_DIR, force=False, complete_only=True):
    """
    Load the Plotly S&P 500 five-year panel as a wide price frame.

    Downloads once and caches to ``cache_dir``; every later call reads from
    disk, so the notebooks work offline after the first run.

    Parameters
    ----------
    complete_only : bool
        Keep only tickers with no missing closes over the whole window. This
        removes survivorship-free listings and mid-sample additions, leaving
        470 names. It also means the panel is *survivorship biased* -- worth
        saying out loud to students.

    Returns
    -------
    pandas.DataFrame
        Index of dates, columns of tickers, values of closing prices.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, CACHE_FILE)
    if force or not os.path.exists(path):
        print(f"Downloading the S&P 500 panel to {path} (about 30 MB, once only) ...")
        urllib.request.urlretrieve(PANEL_URL, path)
    raw = pd.read_csv(path, parse_dates=["date"])
    prices = raw.pivot_table(index="date", columns="Name", values="close")
    if complete_only:
        prices = prices.loc[:, prices.notna().all()]
    prices.columns.name = None
    return prices


def load_live(tickers, start, end):
    """
    Pull adjusted closes from Yahoo Finance.

    Returns ``None`` (with a warning rather than an exception) if the download
    fails, so a notebook running behind a firewall degrades gracefully instead
    of stopping.
    """
    try:
        import yfinance as yf
    except ImportError:
        warnings.warn("yfinance is not installed; falling back to the cached panel.")
        return None
    try:
        raw = yf.download(list(tickers), start=start, end=end,
                          auto_adjust=True, progress=False)
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        close = close.dropna(how="all")
        if close.empty:
            raise ValueError("empty frame returned")
        return close
    except Exception as exc:                                  # noqa: BLE001
        warnings.warn(f"Live download failed ({exc}); falling back to the cached panel.")
        return None


def get_prices(tickers=None, use_live=False, start="2019-01-01", end=None,
               cache_dir=CACHE_DIR):
    """
    Single entry point for price data.

    With ``use_live=False`` (the default) this returns the cached Plotly panel,
    optionally restricted to ``tickers``. With ``use_live=True`` it tries Yahoo
    Finance first and silently falls back to the panel if that fails.
    """
    if use_live and tickers is not None:
        live = load_live(tickers, start, end)
        if live is not None:
            return live.dropna()
    prices = load_panel(cache_dir=cache_dir)
    if tickers is not None:
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise KeyError(f"Not in the panel: {missing}")
        prices = prices[list(tickers)]
    return prices


def log_returns(prices):
    """Log returns, with the first (all-NaN) row dropped."""
    return np.log(prices).diff().dropna(how="all").dropna()


def portfolio_returns(returns, weights=None):
    """
    Portfolio log returns under fixed weights, rebalanced every period.

    ``weights=None`` gives equal weights. Note that summing weighted *log*
    returns is an approximation; it is accurate at daily frequency and is the
    convention used on the slides.
    """
    if weights is None:
        return returns.mean(axis=1)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return returns.to_numpy() @ w if isinstance(returns, np.ndarray) else returns.dot(w)


def annualise_vol(daily_vol, periods=252):
    """Scale a daily volatility to annual by the square root of time."""
    return daily_vol * np.sqrt(periods)


def rp_alpha(confidence):
    """
    Convert a confidence level to the tail probability Riskfolio-Lib expects.

    ``rp_alpha(0.99) -> 0.01``. Getting this backwards is the single most
    common error when moving between the slides and Riskfolio-Lib. Rounded to
    twelve places so the result displays cleanly.
    """
    return round(1.0 - confidence, 12)


# ---------------------------------------------------------------------------
# Risk measures -- all return positive loss magnitudes
# ---------------------------------------------------------------------------

def var_hist(r, confidence=0.99, method="linear"):
    """
    Historical (empirical) Value at Risk.

    ``method`` is passed to :func:`numpy.quantile`. The default ``"linear"``
    interpolates between order statistics, which is the usual convention.
    Riskfolio-Lib instead takes the order statistic directly, equivalent to
    ``method="lower"``; on a five-year daily sample the two differ by about a
    basis point. Notebook 06a shows the comparison.
    """
    return float(-np.quantile(np.asarray(r, dtype=float), 1.0 - confidence,
                              method=method))


def es_hist(r, confidence=0.99, method="linear"):
    """Historical Expected Shortfall: mean loss in the worst tail."""
    r = np.asarray(r, dtype=float)
    cut = np.quantile(r, 1.0 - confidence, method=method)
    tail = r[r <= cut]
    return float(-tail.mean()) if tail.size else np.nan


def var_normal(r, confidence=0.99):
    """Gaussian VaR using the sample mean and standard deviation."""
    r = np.asarray(r, dtype=float)
    return float(-(r.mean() + r.std(ddof=1) * stats.norm.ppf(1.0 - confidence)))


def es_normal(r, confidence=0.99):
    """Closed-form Gaussian Expected Shortfall."""
    r = np.asarray(r, dtype=float)
    mu, sd, a = r.mean(), r.std(ddof=1), 1.0 - confidence
    return float(-(mu - sd * stats.norm.pdf(stats.norm.ppf(a)) / a))


def _fit_t(r):
    return stats.t.fit(np.asarray(r, dtype=float))


def var_t(r, confidence=0.99, params=None):
    """VaR under a fitted Student-t distribution."""
    nu, loc, scale = params if params is not None else _fit_t(r)
    return float(-stats.t.ppf(1.0 - confidence, nu, loc, scale))


def es_t(r, confidence=0.99, params=None, n_sim=2_000_000, seed=0):
    """
    Expected Shortfall under a fitted Student-t.

    Evaluated by simulation, which keeps the code readable and is accurate to
    well within display precision at the default sample size.
    """
    nu, loc, scale = params if params is not None else _fit_t(r)
    draws = stats.t.rvs(nu, loc, scale, size=n_sim, random_state=seed)
    cut = np.quantile(draws, 1.0 - confidence)
    return float(-draws[draws <= cut].mean())


def risk_table(r, confidences=(0.95, 0.975, 0.99), as_percent=True):
    """
    The measure-versus-assumption table from Section 2 of the lecture.

    Rows are distributional assumptions, columns are VaR at each confidence
    level plus ES at 97.5%.
    """
    r = pd.Series(r).dropna()
    tp = _fit_t(r)
    rows = {}
    for name, vf, ef, kw in [
        ("Normal", var_normal, es_normal, {}),
        (f"Student-t (nu={tp[0]:.1f})", var_t, es_t, {"params": tp}),
        ("Empirical", var_hist, es_hist, {}),
    ]:
        row = {f"VaR {int(c * 1000) / 10:g}%": vf(r, c, **kw) for c in confidences}
        row["ES 97.5%"] = ef(r, 0.975, **kw)
        rows[name] = row
    out = pd.DataFrame(rows).T
    return out * 100 if as_percent else out


# ---------------------------------------------------------------------------
# Conditional volatility
# ---------------------------------------------------------------------------

def ewma_var(r, lam=0.94, warmup=250):
    """
    RiskMetrics exponentially weighted variance.

    The value at date *t* uses information up to *t-1* only, so the series can
    be compared with the realised return on the same row without look-ahead.
    """
    r = pd.Series(r).astype(float)
    out = np.full(len(r), np.nan)
    var = r.iloc[:warmup].var()
    for i, x in enumerate(r.to_numpy()):
        if i >= warmup:
            out[i] = var
        var = lam * var + (1.0 - lam) * x ** 2
    return pd.Series(out, index=r.index, name=f"EWMA({lam})")


def realised_var_proxy(r):
    """Squared daily return -- a noisy but unbiased proxy for daily variance."""
    return pd.Series(r).astype(float) ** 2


def har_design(rv, lags=(1, 5, 22)):
    """
    HAR design matrix: daily, weekly and monthly averages of realised variance.

    All columns are lagged by one period, so a model fitted on them is a
    genuine one-step-ahead forecast.
    """
    rv = pd.Series(rv).astype(float)
    names = {1: "rv_d", 5: "rv_w", 22: "rv_m"}
    cols = {names.get(k, f"rv_{k}"): rv.rolling(k).mean().shift(1) for k in lags}
    return pd.DataFrame(cols).dropna()


def qlike(proxy, forecast):
    """
    QLIKE loss, robust to noise in the volatility proxy (Patton, 2011).

    Zero only when the forecast equals the proxy, and it penalises
    under-prediction of variance far more heavily than over-prediction.
    """
    p = np.asarray(proxy, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = (f > 0) & (p > 0) & np.isfinite(f) & np.isfinite(p)
    ratio = p[ok] / f[ok]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mse_loss(proxy, forecast):
    """Mean squared error on the variance scale -- also proxy-robust."""
    p = np.asarray(proxy, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = np.isfinite(f) & np.isfinite(p)
    return float(np.mean((p[ok] - f[ok]) ** 2))


def rmse_vol(proxy, forecast):
    """
    RMSE on the *volatility* scale -- deliberately included because it is
    NOT proxy-robust. Used in Notebook 06b to show a ranking reversal.
    """
    p = np.sqrt(np.asarray(proxy, dtype=float))
    f = np.sqrt(np.asarray(forecast, dtype=float))
    ok = np.isfinite(f) & np.isfinite(p)
    return float(np.sqrt(np.mean((p[ok] - f[ok]) ** 2)))


def loss_table(proxy, forecasts):
    """Score a dict of {name: forecast series} under QLIKE, MSE and RMSE-vol."""
    rows = {}
    for name, f in forecasts.items():
        idx = pd.Series(f).dropna().index.intersection(pd.Series(proxy).dropna().index)
        p, ff = pd.Series(proxy).loc[idx], pd.Series(f).loc[idx]
        rows[name] = {"QLIKE": qlike(p, ff), "MSE": mse_loss(p, ff),
                      "RMSE (vol)": rmse_vol(p, ff), "n": len(idx)}
    out = pd.DataFrame(rows).T
    for col in ("QLIKE", "MSE", "RMSE (vol)"):
        out[f"rank {col}"] = out[col].rank().astype(int)
    return out


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def breaches(r, var_series):
    """Boolean series: True where the realised loss exceeded the VaR forecast."""
    r = pd.Series(r).astype(float)
    v = pd.Series(var_series).astype(float).reindex(r.index)
    return (r < -v).where(v.notna())


def kupiec(n, x, p=0.01):
    """
    Kupiec unconditional coverage test.

    Returns ``(LR, p_value)``; LR is asymptotically chi-squared with 1 degree
    of freedom under a correctly specified model.
    """
    if x == 0:
        lr = -2.0 * n * np.log(1.0 - p)
    else:
        pi = x / n
        lr = -2.0 * ((n - x) * np.log(1 - p) + x * np.log(p)
                     - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return float(lr), float(1.0 - stats.chi2.cdf(lr, 1))


def christoffersen_ind(indicator):
    """
    Christoffersen independence test against a first-order Markov alternative.

    Detects *clustering*: breaches arriving together rather than scattered.
    Returns ``(LR, p_value)`` with one degree of freedom.
    """
    ind = pd.Series(indicator).dropna().astype(int).to_numpy()
    prev, cur = ind[:-1], ind[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return np.nan, np.nan
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    def _ll(p, a, b):
        if p <= 0 or p >= 1:
            return 0.0
        return a * np.log(1 - p) + b * np.log(p)
    lr = -2.0 * (_ll(pi, n00, n01) + _ll(pi, n10, n11)
                 - _ll(pi01, n00, n01) - _ll(pi11, n10, n11))
    return float(lr), float(1.0 - stats.chi2.cdf(lr, 1))


def conditional_coverage(indicator, p=0.01):
    """
    Christoffersen conditional coverage: coverage and independence combined.

    Returns a dict with the two component statistics and the joint test, which
    is chi-squared with 2 degrees of freedom.
    """
    ind = pd.Series(indicator).dropna().astype(int)
    n, x = len(ind), int(ind.sum())
    lr_uc, p_uc = kupiec(n, x, p)
    lr_ind, p_ind = christoffersen_ind(ind)
    lr_cc = lr_uc + lr_ind if np.isfinite(lr_ind) else np.nan
    p_cc = float(1.0 - stats.chi2.cdf(lr_cc, 2)) if np.isfinite(lr_cc) else np.nan
    return {"n": n, "breaches": x, "rate %": 100 * x / n, "expected": p * n,
            "LR_uc": lr_uc, "p_uc": p_uc, "LR_ind": lr_ind, "p_ind": p_ind,
            "LR_cc": lr_cc, "p_cc": p_cc}


def basel_zone(x):
    """Basel traffic-light zone for x breaches in a 250-day window at 99%."""
    return "green" if x <= 4 else ("yellow" if x <= 9 else "red")


def rolling_zone_summary(indicator, window=250):
    """Share of rolling windows landing in each Basel traffic-light zone."""
    ind = pd.Series(indicator).dropna().astype(int)
    counts = ind.rolling(window).sum().dropna()
    zones = counts.map(basel_zone)
    share = zones.value_counts(normalize=True).reindex(["green", "yellow", "red"]).fillna(0.0)
    return {"windows": int(len(counts)), "min": int(counts.min()), "max": int(counts.max()),
            "green %": 100 * share["green"], "yellow %": 100 * share["yellow"],
            "red %": 100 * share["red"]}


def kupiec_power(true_rate, n, p=0.01, size=0.05, sims=100_000, seed=0):
    """
    Simulated power of the Kupiec test.

    The probability of correctly rejecting a VaR model whose true breach rate
    is ``true_rate`` when it claims ``p``, given ``n`` observations.
    """
    rng = np.random.default_rng(seed)
    x = rng.binomial(n, true_rate, sims)
    pi = np.clip(x / n, 1e-12, 1 - 1e-12)
    lr = -2.0 * ((n - x) * np.log(1 - p) + x * np.log(p)
                 - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return float(np.mean(lr > stats.chi2.ppf(1 - size, 1)))


# ---------------------------------------------------------------------------
# Risk decomposition and allocation
# ---------------------------------------------------------------------------

def risk_contributions(w, Sigma, normalise=True):
    """
    Euler decomposition of portfolio volatility into per-asset contributions.

    Because volatility is homogeneous of degree one, the contributions sum to
    the portfolio volatility exactly -- this is an identity, not an
    approximation, which is what makes it usable as a budget.
    """
    w = np.asarray(w, dtype=float).ravel()
    S = np.asarray(Sigma, dtype=float)
    sigma_p = float(np.sqrt(w @ S @ w))
    trc = w * (S @ w) / sigma_p
    return trc / trc.sum() if normalise else trc


def inverse_vol_weights(Sigma):
    """Weights proportional to the reciprocal of each asset's volatility."""
    vol = np.sqrt(np.diag(np.asarray(Sigma, dtype=float)))
    w = 1.0 / vol
    return w / w.sum()


def erc_weights(Sigma, b=None, tol=1e-14, max_iter=2000):
    """
    Equal-risk-contribution (or risk-budgeting) weights.

    Solves Spinu's strictly convex reformulation

        min_{w > 0}  0.5 w' Sigma w  -  sum_i b_i log w_i

    and renormalises so the weights sum to one. ``b=None`` gives equal risk
    contributions; passing a budget vector gives risk budgeting.

    Cross-checked in the notebooks against Riskfolio-Lib's ``rp_optimization``;
    the two agree to within display precision.
    """
    S = np.asarray(Sigma, dtype=float)
    n = S.shape[0]
    b = np.repeat(1.0 / n, n) if b is None else np.asarray(b, dtype=float).ravel()
    b = b / b.sum()

    def obj(x):
        return 0.5 * x @ S @ x - b @ np.log(x)

    def grad(x):
        return S @ x - b / x

    x0 = inverse_vol_weights(S)
    res = minimize(obj, x0, jac=grad, method="L-BFGS-B",
                   bounds=[(1e-12, None)] * n,
                   options={"ftol": tol, "gtol": tol, "maxiter": max_iter})
    w = res.x / res.x.sum()
    return w


# ---------------------------------------------------------------------------
# Scoring functions for VaR and ES (the elicitability material, Section 2)
# ---------------------------------------------------------------------------

def pinball_loss(y, v, tail_prob=0.01):
    """
    Pinball (tick) loss -- the consistent scoring function for a quantile.

    ``v`` is the quantile of *returns* (so a negative number for a left tail).
    Minimised in expectation by the true ``tail_prob`` quantile, which is what
    makes VaR elicitable.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    hit = (y <= v).astype(float)
    return float(np.mean((hit - tail_prob) * (v - y)))


def fz0_loss(y, v, e, tail_prob=0.01):
    """
    Fissler--Ziegel loss, the ``FZ0`` member used by Patton, Ziegel and Chen (2019).

    Jointly minimised by the true ``(VaR, ES)`` pair, both expressed as
    quantities of *returns* and therefore negative in the left tail. Expected
    Shortfall has no consistent scoring function on its own; this one works
    because it scores the pair.

    Returns the mean loss, so lower is better.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    e = np.asarray(e, dtype=float)
    if np.any(e >= 0):
        raise ValueError("FZ0 requires the ES input to be a negative return level.")
    hit = (y <= v).astype(float)
    loss = (-1.0 / (tail_prob * e)) * hit * (v - y) + v / e + np.log(-e) - 1.0
    return float(np.mean(loss))


def diebold_mariano(loss_a, loss_b, lag=1):
    """
    Diebold--Mariano test of equal predictive accuracy between two models.

    Pass the *per-observation* loss series. A negative statistic favours model
    A. Uses a Newey--West variance with ``lag`` autocovariances, which matters
    because forecast losses are serially correlated.

    Returns ``(statistic, p_value)``; the statistic is asymptotically standard
    normal under the null of equal accuracy.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float(dc @ dc / n)
    var = gamma0
    for k in range(1, lag + 1):
        gk = float(dc[:-k] @ dc[k:] / n)
        var += 2.0 * (1.0 - k / (lag + 1)) * gk
    stat = dbar / np.sqrt(var / n)
    return float(stat), float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))


# ---------------------------------------------------------------------------
# Volatility targeting and the feedback loop (Section 6)
# ---------------------------------------------------------------------------

def vol_target_leverage(vol_forecast, target_annual=0.10, max_leverage=3.0,
                        periods=252):
    """
    Leverage implied by a volatility target.

    Leverage is an inverse function of estimated risk -- the mechanism that
    makes volatility targeting procyclical.
    """
    ann = np.asarray(vol_forecast, dtype=float) * np.sqrt(periods)
    with np.errstate(divide="ignore", invalid="ignore"):
        lev = target_annual / ann
    return np.clip(lev, 0.0, max_leverage)


def simulate_feedback(n_days=2500, impact=0.0, target_annual=0.10,
                      base_vol_annual=0.15, lam=0.94, max_leverage=3.0,
                      capital_share=1.0, seed=0):
    """
    Minimal model of a market containing volatility-targeting investors.

    Each day, in order:

    1. Forecast volatility by EWMA from returns up to yesterday.
    2. Set leverage from the volatility target.
    3. Trade the change in leverage.
    4. The realised return is a fundamental shock plus ``impact`` times that flow.

    With ``impact=0`` the targeting rule has no effect on prices and the market
    is the fundamental process. With ``impact>0`` a rise in volatility forces
    deleveraging, the selling pushes prices down, and volatility rises again.

    Returns a DataFrame with the shock, return, volatility forecast, leverage
    and flow for each day.
    """
    rng = np.random.default_rng(seed)
    sd = base_vol_annual / np.sqrt(252)
    shocks = rng.standard_normal(n_days) * sd

    ret = np.zeros(n_days)
    vol = np.zeros(n_days)
    lev = np.zeros(n_days)
    flow = np.zeros(n_days)

    var = sd ** 2
    prev_lev = target_annual / base_vol_annual
    for t in range(n_days):
        vol[t] = np.sqrt(var)
        lev[t] = float(vol_target_leverage(vol[t], target_annual, max_leverage))
        flow[t] = (lev[t] - prev_lev) * capital_share
        ret[t] = shocks[t] + impact * flow[t]
        var = lam * var + (1.0 - lam) * ret[t] ** 2
        prev_lev = lev[t]

    return pd.DataFrame({"shock": shocks, "return": ret,
                         "vol_forecast": vol, "leverage": lev, "flow": flow})
