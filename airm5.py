"""
airm5.py -- shared helpers for AI-powered Investment and Risk Management, Lecture 5.

Used by Lec05a-Lec05d. Nothing here is lecture-specific: it is the plumbing
(data, performance statistics, the deflated Sharpe machinery, the trial
counter, and the house plotting theme) that all four notebooks share.

Design notes
------------
* Offline-capable. Every loader takes a local path first and only reaches for
  the network if the file is missing. Cached files land in ./data/.
* Deck-matched by default. `load_shiller_monthly()` truncates to 2018-12 so the
  numbers reproduce the lecture slides; pass full=True for the live series.
* No plotting library beyond Plotly. Figures carry the house theme via
  `style(fig)`.

British spelling in prose; American spellings in code are library API keywords.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "C", "PALETTE", "style", "load_daily_panel", "load_shiller_monthly",
    "equal_weight_index", "sharpe", "ann_return", "ann_vol", "cagr",
    "max_drawdown", "summary", "ma_crossover", "expected_max_sharpe",
    "psr", "deflated_sharpe", "min_track_record_length", "TrialCounter",
    "purged_walk_forward", "triple_barrier", "load_ohlcv", "ta_features",
    "drop_correlated", "fracdiff", "fracdiff_weights", "break_even_accuracy",
]

# --------------------------------------------------------------------------
# House style. Anchored on the ColorSchemeBlue palette used in the slide deck
# so lab figures and lecture figures sit together without a colour clash.
# --------------------------------------------------------------------------

C = {
    "navy":   "#123F69",   # tabhead RGB(18,63,105)  -- primary
    "band":   "#E2EBF4",   # tabband RGB(226,235,244) -- fills, gridlines
    "blue":   "#2E7EBB",   # \bluebold -- "this is the point"
    "red":    "#C0392B",   # \redbold  -- "this is the trap"
    "amber":  "#E8A33D",   # secondary emphasis
    "green":  "#2E8B6B",   # benchmark / the honest answer
    "grey":   "#7A8994",   # de-emphasis
    "ink":    "#1C2833",   # text
    "white":  "#FFFFFF",
}
PALETTE = [C["navy"], C["red"], C["green"], C["amber"], C["blue"], C["grey"]]


def style(fig, height=430, legend_top=True, title=None, **kwargs):
    """Apply the house look to a Plotly figure. Returns the figure.

    `title` may be a plain string or a Plotly title dict; either way it is
    merged into the house title spec rather than replacing it.
    """
    title_spec = dict(font=dict(size=16, color=C["navy"]), x=0.0, xanchor="left")
    if isinstance(title, str):
        title_spec["text"] = title
    elif isinstance(title, dict):
        title_spec.update(title)

    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif",
                  size=13, color=C["ink"]),
        title=title_spec,
        margin=dict(l=60, r=30, t=60, b=50),
        colorway=PALETTE,
        hoverlabel=dict(font_size=12),
        **kwargs,
    )
    if legend_top:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      xanchor="left", x=0))
    fig.update_xaxes(showgrid=True, gridcolor=C["band"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=C["band"], zeroline=False)
    return fig


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

DAILY_URL = ("https://raw.githubusercontent.com/plotly/datasets/master/"
             "all_stocks_5yr.csv")
SHILLER_URL = ("https://raw.githubusercontent.com/datasets/s-and-p-500/"
               "main/data/data.csv")

# The lecture's figures were computed on the Shiller series as it stood ending
# December 2018. The hosted file keeps growing, so we truncate by default;
# otherwise the lab and the slides quietly disagree.
DECK_CUTOFF = "2018-12"


def _fetch(url: str, path: str) -> str:
    """Return a local path, downloading once if needed."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        pd.read_csv(url).to_csv(path, index=False)
    except Exception as exc:                                   # pragma: no cover
        raise RuntimeError(
            f"Could not reach {url} and no local copy at {path}. "
            "Download it once on a connected machine and place it there."
        ) from exc
    return path


def load_daily_panel(path="data/all_stocks_5yr.csv", full_history_only=True):
    """S&P 500 constituent daily closes, 2013-02-08 to 2018-02-07.

    Returns a (dates x tickers) frame of closing prices. With
    `full_history_only`, keeps only names quoted on every date -- which is a
    survivorship choice, and deliberately the same one the lecture makes.
    """
    df = pd.read_csv(_fetch(DAILY_URL, path), parse_dates=["date"])
    px = df.pivot(index="date", columns="Name", values="close").sort_index()
    if full_history_only:
        px = px[px.columns[px.notna().all()]]
    return px


def load_shiller_monthly(path="data/shiller.csv", full=False):
    """Shiller monthly S&P series from 1871.

    Returns a frame with a `total_return` column: the nominal monthly total
    return, treating the annualised `Dividend` figure as paid in twelfths.
    """
    s = pd.read_csv(_fetch(SHILLER_URL, path))
    s["Date"] = pd.to_datetime(s["Date"])
    s = s.set_index("Date").sort_index()
    if not full:
        s = s.loc[:DECK_CUTOFF]
    s["total_return"] = (s["SP500"] + s["Dividend"] / 12) / s["SP500"].shift(1) - 1
    s["cash_return"] = s["Long Interest Rate"] / 100 / 12
    return s


def equal_weight_index(prices: pd.DataFrame):
    """Daily-rebalanced equal-weighted index. Returns (level, returns)."""
    rets = prices.pct_change().mean(axis=1).dropna()
    level = (1 + rets).cumprod()
    return level, rets


def load_ohlcv(name="AAPL", path="data/all_stocks_5yr.csv"):
    """Single-ticker OHLCV frame from the panel, with the column names the
    technical-analysis libraries expect."""
    df = pd.read_csv(_fetch(DAILY_URL, path), parse_dates=["date"])
    d = df[df["Name"] == name].set_index("date").sort_index()
    if d.empty:
        raise KeyError(f"{name} is not in the panel.")
    return d.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})[
        ["Open", "High", "Low", "Close", "Volume"]]


def ta_features(ohlcv, drop_warmup=True, structural_threshold=0.2, report=False):
    """All 86 `ta` indicators, built the way you should build them.

    Deliberately calls `add_all_ta_features(..., fillna=False)`. The library's
    `fillna=True` is convenient and wrong: it removes the NaNs that tell you an
    indicator has not warmed up yet, replacing them with zeros, forward fills
    and -- for a couple of indicators -- values backfilled from the future.
    Lec05b measures exactly what that costs.

    Two different kinds of missingness are separated here, because conflating
    them is how the damage gets done:

    * **Structural.** `trend_psar_up` and `trend_psar_down` are missing by
      design whenever the parabolic SAR sits on the other side of price -- more
      than half the sample. Filling those with zero does not repair a gap; it
      fabricates an indicator reading that never existed.
    * **Warm-up.** The leading rows before a rolling window has enough history.
      Around 71 rows for the slowest indicator on a five-year daily series.

    Columns missing more than `structural_threshold` of the time are dropped;
    with `drop_warmup`, the remaining leading rows are then dropped too. You
    lose the rows. You do not invent them.
    """
    import ta as _ta
    ohlcv = pd.DataFrame(ohlcv)
    out = _ta.add_all_ta_features(ohlcv.copy(), "Open", "High", "Low",
                                  "Close", "Volume", fillna=False)
    feats = out.drop(columns=list(ohlcv.columns)).replace([np.inf, -np.inf], np.nan)

    missing = feats.isna().mean()
    structural = list(missing[missing > structural_threshold].index)
    feats = feats.drop(columns=structural)

    n_before = len(feats)
    if drop_warmup:
        feats = feats.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if report:
        print(f"{out.shape[1] - ohlcv.shape[1]} indicators built")
        print(f"  dropped as structurally missing : {structural}")
        print(f"  rows dropped as warm-up         : {n_before - len(feats)}")
        print(f"  usable feature matrix           : {feats.shape}")
    return feats


def drop_correlated(frame, threshold=0.9):
    """Drop any column correlated above `threshold` with an earlier column.

    Order-dependent by construction -- which column of a redundant pair
    survives is decided by the column ordering, not by the data. That is worth
    knowing before you report the survivors as though they were chosen.
    """
    cm = frame.corr().abs()
    upper = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    return frame.drop(columns=to_drop), to_drop


# --------------------------------------------------------------------------
# Fractional differentiation (Lopez de Prado): stationarity without amnesia
# --------------------------------------------------------------------------

def fracdiff_weights(d, threshold=1e-4, max_length=None):
    """Binomial expansion weights for the differencing operator (1-B)^d."""
    w = [1.0]
    while True:
        nxt = -w[-1] * (d - len(w) + 1) / len(w)
        if abs(nxt) < threshold or (max_length and len(w) >= max_length):
            break
        w.append(nxt)
    return np.array(w[::-1])


def fracdiff(series, d, threshold=1e-4, max_length=None):
    """Fixed-width fractional differencing.

    d = 0 returns the series untouched; d = 1 is the ordinary first difference.
    Values in between buy stationarity while keeping some of the memory that
    full differencing destroys.
    """
    s = pd.Series(series).dropna()
    w = fracdiff_weights(d, threshold, max_length or len(s) // 2)
    k = len(w)
    vals = [np.dot(w, s.values[i - k + 1: i + 1]) for i in range(k - 1, len(s))]
    return pd.Series(vals, index=s.index[k - 1:], name=f"fracdiff_d{d:g}")


def break_even_accuracy(cost_bps, mean_abs_return):
    """Directional accuracy needed to break even against a round-trip cost.

    Win `mean_abs_return` when right, lose it when wrong, pay `cost_bps`:
        p*E|r| - (1-p)*E|r| - c = 0   =>   p = 1/2 + c / (2 E|r|)
    """
    return 0.5 + (np.asarray(cost_bps) / 1e4) / (2 * mean_abs_return)


# --------------------------------------------------------------------------
# Performance statistics
# --------------------------------------------------------------------------

def sharpe(returns, periods=252):
    r = pd.Series(returns).dropna()
    sd = r.std()
    return np.nan if sd == 0 else r.mean() / sd * np.sqrt(periods)


def ann_return(returns, periods=252):
    return pd.Series(returns).dropna().mean() * periods


def ann_vol(returns, periods=252):
    return pd.Series(returns).dropna().std() * np.sqrt(periods)


def cagr(returns, periods=252):
    r = pd.Series(returns).dropna()
    return (1 + r).prod() ** (periods / len(r)) - 1


def max_drawdown(returns):
    c = (1 + pd.Series(returns).dropna()).cumprod()
    return (c / c.cummax() - 1).min()


def summary(returns, periods=252, name="strategy"):
    """One-row performance summary."""
    r = pd.Series(returns).dropna()
    return pd.Series({
        "CAGR": cagr(r, periods),
        "Volatility": ann_vol(r, periods),
        "Sharpe": sharpe(r, periods),
        "Max drawdown": max_drawdown(r),
        "Skew": r.skew(),
        "Excess kurtosis": r.kurtosis(),
        "Observations": len(r),
    }, name=name)


# --------------------------------------------------------------------------
# The strategy under study
# --------------------------------------------------------------------------

def ma_crossover(level, returns, fast, slow, cost_bps=0.0,
                 cash_return=None, allow_short=False):
    """Long when the fast moving average sits above the slow one.

    The signal is computed on data up to and including day t and acted on at
    t+1 (`shift(1)`), which is the whole of the point-in-time discipline this
    function is allowed to assume.

    `cash_return` -- if supplied, the return earned while out of the market.
    Leaving it None means flat cash, which is a modelling choice, not a
    neutral default. Lec05a measures what that choice is worth.
    """
    level, returns = pd.Series(level), pd.Series(returns)
    fa, sl = level.rolling(fast).mean(), level.rolling(slow).mean()
    signal = (fa > sl)
    pos = signal.astype(float)
    if allow_short:
        pos = pos * 2 - 1
    pos = pos.shift(1).fillna(0)

    gross = pos * returns
    if cash_return is not None:
        gross = gross + (1 - pos.clip(0, 1)) * pd.Series(cash_return).reindex(
            returns.index).fillna(0)
    turnover = pos.diff().abs().fillna(0)
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"position": pos, "gross": gross, "net": net,
                         "turnover": turnover})


# --------------------------------------------------------------------------
# Selection bias: the deflated Sharpe machinery (Bailey & Lopez de Prado)
# --------------------------------------------------------------------------

EULER = 0.5772156649015329


def expected_max_sharpe(n_trials, sd_trials):
    """The Sharpe you should expect from the *best* of N independent trials
    whose true edge is zero. This is the hurdle, not the result.

    `sd_trials` is the cross-sectional standard deviation of the Sharpe ratios
    across the trials, in whatever units you want the answer in.
    """
    n = max(int(n_trials), 2)
    a = norm.ppf(1 - 1 / n)
    b = norm.ppf(1 - 1 / (n * np.e))
    return sd_trials * ((1 - EULER) * a + EULER * b)


def psr(returns, benchmark_sr=0.0, periods=252):
    """Probabilistic Sharpe ratio: P(true Sharpe > benchmark), correcting for
    sample length and for the skew and fat tails of the return series.

    `benchmark_sr` is annualised, matching how everything else here is quoted.
    """
    r = pd.Series(returns).dropna()
    t = len(r)
    sr_hat = r.mean() / r.std()                     # per period
    sr_star = benchmark_sr / np.sqrt(periods)       # per period
    g3, g4 = r.skew(), r.kurtosis() + 3.0           # kurtosis(), not excess
    denom = np.sqrt(1 - g3 * sr_hat + (g4 - 1) / 4 * sr_hat ** 2)
    return float(norm.cdf((sr_hat - sr_star) * np.sqrt(t - 1) / denom))


def deflated_sharpe(returns, n_trials, sd_trials, periods=252):
    """Probability the winner's edge is real once the search is priced in.

    `sd_trials` annualised, like every other Sharpe here.
    """
    return psr(returns,
               benchmark_sr=expected_max_sharpe(n_trials, sd_trials),
               periods=periods)


def min_track_record_length(returns, benchmark_sr=0.0, confidence=0.95,
                            periods=252):
    """How many observations you would need before the result is significant."""
    r = pd.Series(returns).dropna()
    sr_hat = r.mean() / r.std()
    sr_star = benchmark_sr / np.sqrt(periods)
    if sr_hat <= sr_star:
        return np.inf
    g3, g4 = r.skew(), r.kurtosis() + 3.0
    num = 1 - g3 * sr_hat + (g4 - 1) / 4 * sr_hat ** 2
    return 1 + num * (norm.ppf(confidence) / (sr_hat - sr_star)) ** 2


# --------------------------------------------------------------------------
# The device: charge yourself for your own search
# --------------------------------------------------------------------------

@dataclass
class TrialCounter:
    """Counts every strategy you evaluate, so the notebook can price your search.

    A backtest result is not a number, it is a number *and* the count of
    alternatives that were tried before it was shown to you. Anything you
    evaluate through `.record()` is remembered; `.verdict()` reports the best
    result against the hurdle implied by the count.

    >>> tc = TrialCounter()
    >>> tc.record("MA 35/140", 1.02)
    >>> tc.verdict(periods=252)
    """
    label: str = "session"
    trials: list = field(default_factory=list)

    def record(self, name, sharpe_ratio, returns=None):
        """Log one evaluated strategy. Returns the Sharpe, so it can be wrapped
        around an existing call without changing the surrounding code."""
        if np.isfinite(sharpe_ratio):
            self.trials.append({"name": name, "sharpe": float(sharpe_ratio),
                                "returns": returns})
        return sharpe_ratio

    def record_many(self, names, sharpes):
        for n, s in zip(names, sharpes):
            self.record(n, s)

    @property
    def n(self):
        return len(self.trials)

    @property
    def sharpes(self):
        return np.array([t["sharpe"] for t in self.trials])

    @property
    def best(self):
        return max(self.trials, key=lambda t: t["sharpe"]) if self.trials else None

    def hurdle(self):
        """Expected best-of-N Sharpe under no edge, given this session's spread."""
        if self.n < 2:
            return np.nan
        return expected_max_sharpe(self.n, self.sharpes.std(ddof=1))

    def verdict(self, periods=252, quiet=False):
        """Print, and return, the honest reading of this session's search."""
        if self.n < 2:
            warnings.warn("Fewer than two trials recorded; nothing to deflate.")
            return {}
        b, h = self.best, self.hurdle()
        out = {"n_trials": self.n, "best_name": b["name"],
               "best_sharpe": b["sharpe"], "sd_trials": self.sharpes.std(ddof=1),
               "hurdle": h, "clears_hurdle": b["sharpe"] > h, "dsr": np.nan}
        if b["returns"] is not None:
            out["dsr"] = deflated_sharpe(b["returns"], self.n,
                                         out["sd_trials"], periods)
        if not quiet:
            print(f"Trials recorded this session : {out['n_trials']}")
            print(f"Best result                  : {out['best_name']} "
                  f"(Sharpe {out['best_sharpe']:.2f})")
            print(f"Spread across trials         : {out['sd_trials']:.3f}")
            print(f"Hurdle at N = {out['n_trials']:<5d}          : "
                  f"{out['hurdle']:.2f}")
            if np.isfinite(out["dsr"]):
                print(f"Deflated Sharpe ratio        : {out['dsr']:.3f}")
            print("Verdict                      : "
                  + ("clears the hurdle" if out["clears_hurdle"]
                     else "does NOT clear the hurdle -- this is what a search "
                          "of this size produces from noise"))
        return out

    def reset(self):
        self.trials.clear()


# --------------------------------------------------------------------------
# Validation helpers (used properly in Lec05c)
# --------------------------------------------------------------------------

def purged_walk_forward(n_obs, n_splits=5, test_size=None, purge=0, embargo=0,
                        mode="expanding"):
    """Yield (train_idx, test_idx) with purging, and optionally an embargo.

    Financial labels are built from *future* returns, so a label at time t
    overlaps the labels at t+1 ... t+h. Training on rows adjacent to the test
    block therefore leaks, even when nothing is shuffled. Purging removes
    training rows whose label horizon reaches into the test block.

    mode='expanding'
        Train on everything before the test block only. This is walk-forward:
        it never trains on data that postdates the test period, so an embargo
        is meaningless here and is ignored.

    mode='kfold'
        Train on everything outside the test block, before *and* after. More
        training data, but the post-test training rows sit immediately after
        the test period, and serial correlation makes them informative about
        it. That is what the embargo removes.
    """
    if mode not in {"expanding", "kfold"}:
        raise ValueError("mode must be 'expanding' or 'kfold'")
    idx = np.arange(n_obs)
    test_size = test_size or n_obs // (n_splits + 1)

    for k in range(n_splits):
        t0 = n_obs - (n_splits - k) * test_size
        t1 = t0 + test_size
        if t0 - purge <= 0:
            continue
        before = idx[: t0 - purge]
        if mode == "expanding":
            train = before
        else:
            after = idx[min(t1 + embargo, n_obs):]
            train = np.concatenate([before, after])
        if len(train) == 0:
            continue
        yield train, idx[t0:t1]


def triple_barrier(prices, events=None, horizon=10, upper=1.0, lower=1.0,
                   vol_span=50):
    """Label each event by whichever barrier is touched first.

    `upper`/`lower` are multiples of trailing daily volatility, estimated as a
    `vol_span`-day exponentially weighted standard deviation of daily returns; set either to
    0 to switch that barrier off. Returns a frame with the label (+1/-1/0),
    the bar at which it resolved, and the realised return to that bar.
    A label of 0 means the vertical barrier -- time -- ran out first, which is
    the outcome a fixed-horizon label is not able to express.
    """
    px = pd.Series(prices).dropna()
    events = px.index if events is None else pd.Index(events)
    vol = px.pct_change().ewm(span=vol_span).std()

    rows = []
    for t in events:
        i = px.index.get_loc(t)
        if i + horizon >= len(px) or not np.isfinite(vol.iloc[i]) or vol.iloc[i] == 0:
            continue
        path = px.iloc[i + 1: i + 1 + horizon] / px.iloc[i] - 1
        hi = upper * vol.iloc[i] if upper > 0 else np.inf
        lo = -lower * vol.iloc[i] if lower > 0 else -np.inf
        touch_hi = path[path >= hi]
        touch_lo = path[path <= lo]
        first_hi = px.index.get_loc(touch_hi.index[0]) if len(touch_hi) else np.inf
        first_lo = px.index.get_loc(touch_lo.index[0]) if len(touch_lo) else np.inf
        if first_hi < first_lo:
            lab, at = 1, first_hi
        elif first_lo < first_hi:
            lab, at = -1, first_lo
        else:
            lab, at = 0, i + horizon
        rows.append({"event": t, "label": lab, "bars_held": at - i,
                     "ret": px.iloc[int(at)] / px.iloc[i] - 1})
    return pd.DataFrame(rows).set_index("event")
