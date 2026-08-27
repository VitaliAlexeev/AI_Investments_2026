"""
airm7.py
========
Shared helpers for the Lecture 7 lab notebooks.

    25882 AI-powered Investment and Risk Management
    Lecture 7 -- Sentiment as an Investment Signal

    Lec07a  Three scorers, one corpus
    Lec07b  Attention is half the signal
    Lec07c  Is it reading, or remembering?

Design notes
------------
* Plotly only. No matplotlib anywhere in this module.
* Every data loader follows the same cascade:
      local cache  ->  GitHub raw  ->  bundled fixture
  so a firewalled JupyterHub still gets a working (if smaller) notebook and
  says so loudly rather than failing at cell 40.
* Core scorers (Loughran-McDonald, VADER) are pure `pip` and always available.
  FinBERT is optional: `score_finbert` returns None and prints why if the
  model or the `transformers` package is unavailable.
* British spelling in prose and in our own function names; American spelling
  only where a library API demands it (`normalize`, `color`, `analyzer`).

Vitali: after running prepare_phrasebank.py and committing the CSV, paste the
raw URL into PHRASEBANK_URL below. That is the only edit this module needs.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

__all__ = [
    "NAVY", "TINT", "RED", "BLUE", "GREY", "PALETTE", "LABELS",
    "load_phrasebank", "load_sp500_panel",
    "score_lm", "score_vader", "score_finbert",
    "majority_baseline", "accuracy", "confusion_frame",
    "plot_confusion", "plot_bars", "plot_lines", "plot_scatter",
    "FIRM_NAMES", "anonymise", "anonymise_series",
    "simulate_attention_panel", "omitted_variable_bias",
    "simulate_adoption_decay", "half_life", "sharpe", "add_abnormal_volume",
    "fig_layout",
]

# --------------------------------------------------------------------------
# 1. Look and feel -- matches the Lecture 7 Beamer deck
# --------------------------------------------------------------------------

NAVY = "#123F69"      # RGB(18, 63, 105)  -- deck table headers
TINT = "#E2EBF4"      # RGB(226,235,244)  -- deck table banding
RED = "#C11B17"
BLUE = "#0041C2"
GREY = "#7A8794"
PALETTE = [NAVY, RED, BLUE, "#2E8B57", "#B8860B", GREY]

LABELS = ["negative", "neutral", "positive"]

# The three notebooks share one figure style so the lab looks like the lecture.
_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Helvetica, Arial, sans-serif", size=13, color="#1a1a1a"),
    title=dict(font=dict(size=16, color=NAVY)),
    margin=dict(l=60, r=30, t=60, b=55),
    colorway=PALETTE,
    hoverlabel=dict(font_size=12),
)


def fig_layout(fig: go.Figure, title: str = "", **kwargs) -> go.Figure:
    """Apply the shared house style to any Plotly figure."""
    fig.update_layout(**{**_LAYOUT, **kwargs})
    if title:
        fig.update_layout(title_text=title)
    return fig


# --------------------------------------------------------------------------
# 2. Data loading
# --------------------------------------------------------------------------

# >>> EDIT ME after running prepare_phrasebank.py <<<
PHRASEBANK_URL = (
    "https://raw.githubusercontent.com/VitaliAlexeev/AI_Investments_2026/main/"
    "data/financial_phrasebank.csv"
)

SP500_URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"

CACHE = Path(os.environ.get("AIRM7_CACHE", "./data_cache"))


def _cached(url: str, name: str, note: str = "") -> Path | None:
    """Download `url` to CACHE/name once. Returns None if unreachable."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        import urllib.request
        print(f"  downloading {name} ...", end="", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "airm7"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f" done ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    except Exception as exc:                                   # noqa: BLE001
        print(f" FAILED\n    {type(exc).__name__}: {exc}")
        if note:
            print(f"    {note}")
        if dest.exists():
            dest.unlink()
        return None


def load_phrasebank(agreement: int = 50,
                    url: str | None = None,
                    local: str | Path | None = None) -> pd.DataFrame:
    """
    Load the Financial PhraseBank as a tidy frame.

    Parameters
    ----------
    agreement : int
        Keep sentences where at least this share of annotators agreed.
        100 -> AllAgree (hardest to disagree with, easiest to score)
         50 -> everything (the full corpus)
    url, local
        Override the default source. `local` wins if given.

    Returns
    -------
    DataFrame with columns: sentence, label, agreement

    Notes
    -----
    Malo et al. (2014), CC BY-NC-SA 3.0. See LICENSE_phrasebank.txt.
    """
    path = None
    if local is not None:
        path = Path(local)
        if not path.exists():
            raise FileNotFoundError(path)
    else:
        path = _cached(url or PHRASEBANK_URL, "financial_phrasebank.csv",
                       note="Check PHRASEBANK_URL in airm7.py, or pass local=...")

    if path is None:
        fx = Path("fixture/financial_phrasebank.csv")
        if fx.exists():
            print("    !! falling back to the small teaching FIXTURE.")
            print("    !! Numbers below will NOT match the real corpus.")
            path = fx
        else:
            raise RuntimeError(
                "Could not load the Financial PhraseBank. Set airm7.PHRASEBANK_URL "
                "to your mirrored CSV, or pass local='path/to/financial_phrasebank.csv'."
            )

    df = pd.read_csv(path)
    missing = {"sentence", "label", "agreement"} - set(df.columns)
    if missing:
        raise ValueError(f"PhraseBank CSV is missing columns: {sorted(missing)}")
    out = df[df["agreement"] >= agreement].reset_index(drop=True)
    print(f"  PhraseBank: {len(out):,} sentences at agreement >= {agreement}")
    return out


def load_sp500_panel(min_obs: int = 1000,
                     tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Daily S&P 500 panel: 505 names, Feb 2013 - Feb 2018, OHLCV.

    Adds the two columns the notebooks actually use:
        ret       simple close-to-close return
        logvol    log(1 + volume), the raw material for the attention proxy

    Source: plotly/datasets/all_stocks_5yr.csv (public, no auth).
    """
    path = _cached(SP500_URL, "all_stocks_5yr.csv")
    if path is None:
        raise RuntimeError("Could not download the S&P 500 panel from GitHub.")

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.rename(columns={"Name": "ticker"})
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)]

    df = df.sort_values(["ticker", "date"])
    df["ret"] = df.groupby("ticker", observed=True)["close"].pct_change()
    df["logvol"] = np.log1p(df["volume"])

    keep = df.groupby("ticker", observed=True)["ret"].transform("count") >= min_obs
    df = df[keep].dropna(subset=["ret"]).reset_index(drop=True)
    print(f"  S&P 500 panel: {df['ticker'].nunique()} tickers, "
          f"{len(df):,} firm-days, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
    return df


def add_abnormal_volume(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Attention proxy: log volume minus its own trailing mean, per firm.

    This is the standard 'abnormal volume' attention measure (Gervais, Kaniel
    and Mingelgrin 2001; Barber and Odean 2008). The trailing window is
    strictly backward-looking and excludes today, so the measure is usable at
    the close of day t.
    """
    g = df.groupby("ticker", observed=True)["logvol"]
    roll = g.transform(lambda s: s.shift(1).rolling(window, min_periods=window // 2).mean())
    sd = g.transform(lambda s: s.shift(1).rolling(window, min_periods=window // 2).std())
    out = df.copy()
    out["abn_vol"] = (out["logvol"] - roll) / sd
    return out


# --------------------------------------------------------------------------
# 3. Scorers
# --------------------------------------------------------------------------

def _to_label(polarity: float, band: float) -> str:
    if polarity > band:
        return "positive"
    if polarity < -band:
        return "negative"
    return "neutral"


def score_lm(texts: Iterable[str], band: float = 0.10) -> pd.DataFrame:
    """
    Loughran-McDonald dictionary tone.

    tone = (n_pos - n_neg) / (n_pos + n_neg), then banded into three classes.
    The dictionary ships inside `pysentiment2`, so there is no data to host.

    Returns a frame with columns: polarity, subjectivity, pred
    """
    import pysentiment2 as ps
    lm = ps.LM()
    rows = []
    for t in texts:
        sc = lm.get_score(lm.tokenize(str(t)))
        rows.append((float(sc["Polarity"]), float(sc["Subjectivity"])))
    out = pd.DataFrame(rows, columns=["polarity", "subjectivity"])
    out["pred"] = [_to_label(p, band) for p in out["polarity"]]
    return out


def score_vader(texts: Iterable[str], band: float = 0.05) -> pd.DataFrame:
    """
    VADER compound score -- a general-purpose social-media sentiment tool.

    Included deliberately as the WRONG instrument: it was tuned on tweets and
    product reviews, not on filings or wire copy. Its errors are the point.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    comp = [analyzer.polarity_scores(str(t))["compound"] for t in texts]
    out = pd.DataFrame({"polarity": comp})
    out["pred"] = [_to_label(c, band) for c in comp]
    return out


def score_finbert(texts: Sequence[str], batch_size: int = 32,
                  model_name: str = "ProsusAI/finbert") -> pd.DataFrame | None:
    """
    FinBERT (Araci 2019) -- optional third rung.

    Returns None, with an explanation, if `transformers`/`torch` are missing or
    the weights cannot be fetched. Every notebook that calls this must handle
    None so that a firewalled environment still completes.
    """
    try:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        import torch                                            # noqa: F401
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  FinBERT unavailable ({type(exc).__name__}). "
              f"Install with:  pip install transformers torch")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
            mdl.eval()
    except Exception as exc:                                    # noqa: BLE001
        print(f"  FinBERT weights unreachable ({type(exc).__name__}: {exc}).")
        print("  This is expected on a firewalled network. Core results below "
              "do not depend on it.")
        return None

    import torch
    id2label = {i: l.lower() for i, l in mdl.config.id2label.items()}
    texts = [str(t) for t in texts]
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt",
                      padding=True, truncation=True, max_length=256)
            probs.append(torch.softmax(mdl(**enc).logits, dim=-1).numpy())
    P = np.vstack(probs)
    cols = [id2label[i] for i in range(P.shape[1])]
    out = pd.DataFrame(P, columns=cols)
    out["pred"] = [cols[i] for i in P.argmax(axis=1)]
    if {"positive", "negative"} <= set(cols):
        out["polarity"] = out["positive"] - out["negative"]
    return out


# --------------------------------------------------------------------------
# 4. Evaluation
# --------------------------------------------------------------------------

def majority_baseline(y_true: Sequence[str]) -> tuple[str, float]:
    """The number every scorer must beat. Returns (class, accuracy)."""
    s = pd.Series(list(y_true))
    cls = s.value_counts().idxmax()
    return cls, float((s == cls).mean())


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    a = np.asarray(list(y_true))
    b = np.asarray(list(y_pred))
    return float((a == b).mean())


def confusion_frame(y_true: Sequence[str], y_pred: Sequence[str],
                    labels: Sequence[str] = LABELS,
                    normalize: str | None = None) -> pd.DataFrame:
    """
    Confusion matrix as a DataFrame: rows = human label, columns = prediction.

    normalize : None | 'row' | 'all'
    """
    cm = pd.crosstab(pd.Series(list(y_true), name="human"),
                     pd.Series(list(y_pred), name="predicted"))
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    if normalize == "row":
        cm = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    elif normalize == "all":
        cm = cm / cm.to_numpy().sum()
    return cm


def plot_confusion(cm: pd.DataFrame, title: str = "",
                   percent: bool = True) -> go.Figure:
    """Confusion matrix as a Plotly heatmap in the deck's navy/tint ramp."""
    z = cm.to_numpy(dtype=float)
    txt = [[f"{v:.0%}" if percent else f"{v:,.0f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=list(cm.columns), y=list(cm.index), text=txt,
        texttemplate="%{text}", textfont=dict(size=13),
        colorscale=[[0, "#FFFFFF"], [0.5, TINT], [1, NAVY]],
        showscale=False, hovertemplate="human=%{y}<br>predicted=%{x}<br>%{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title_text="human label")
    fig.update_xaxes(title_text="model prediction", side="bottom")
    return fig_layout(fig, title, height=340, width=460,
                      margin=dict(l=90, r=20, t=60, b=55))


def plot_bars(x: Sequence, y: Sequence, title: str = "",
              xlab: str = "", ylab: str = "", ref: float | None = None,
              ref_label: str = "baseline", text_fmt: str = "{:.3f}") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(x), y=list(y), marker_color=NAVY,
        text=[text_fmt.format(v) for v in y], textposition="outside",
    ))
    if ref is not None:
        fig.add_hline(y=ref, line_dash="dash", line_color=RED,
                      annotation_text=ref_label, annotation_position="top left",
                      annotation_font_color=RED)
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    return fig_layout(fig, title, height=380, showlegend=False)


def plot_lines(df: pd.DataFrame, x: str, ys: Sequence[str], title: str = "",
               xlab: str = "", ylab: str = "", hline: float | None = None) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(ys):
        fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines", name=col,
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2)))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color=GREY)
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    return fig_layout(fig, title, height=400)


def plot_scatter(x, y, title="", xlab="", ylab="", colour=None,
                 opacity=0.35, size=5) -> go.Figure:
    fig = go.Figure(go.Scattergl(
        x=list(x), y=list(y), mode="markers",
        marker=dict(size=size, opacity=opacity,
                    color=colour if colour is not None else NAVY),
    ))
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    return fig_layout(fig, title, height=420, showlegend=False)


# --------------------------------------------------------------------------
# 5. Anonymisation (Lec07c)
# --------------------------------------------------------------------------

#: Firms that recur in the Financial PhraseBank (Nordic business wire copy).
#: Deliberately a plain list, not an NER model -- students can read it, argue
#: with it, and extend it. The Stretch exercise swaps in spaCy NER.
FIRM_NAMES = [
    "Nokia", "Nokian Tyres", "Outokumpu", "Cargotec", "Ruukki", "Rautaruukki",
    "Kesko", "Stora Enso", "UPM-Kymmene", "UPM", "Elcoteq", "Neste Oil",
    "Neste", "Sanoma", "Metso", "Wartsila", "Konecranes", "Aspocomp",
    "Talvivaara", "Tieto", "TietoEnator", "Fortum", "YIT", "Amer Sports",
    "Huhtamaki", "Finnair", "Kemira", "Orion", "Poyry", "Vaisala", "Uponor",
    "Raisio", "Atria", "Componenta", "Okmetic", "Ahlstrom", "Alma Media",
    "Basware", "Biohit", "Cramo", "Digia", "Efore", "Ekokem", "Elisa",
    "Etteplan", "Exel", "Finnlines", "Glaston", "HKScan", "Honkarakenne",
    "Ilkka", "Incap", "Kemppi", "Kone", "Lassila", "Lemminkainen", "Marimekko",
    "Martela", "Munksjo", "Nordea", "Nurminen Logistics", "Olvi", "Oriola",
    "Outotec", "Panostaja", "PKC Group", "Pohjola", "Ramirent", "Rapala",
    "Sampo", "Scanfil", "Solteq", "Sponda", "SRV", "Stockmann", "Suominen",
    "Tallink", "Technopolis", "Tecnotree", "Teleste", "Tikkurila", "Trainers House",
    "Turkistuottajat", "Vacon", "Vaahto", "Valmet", "Viking Line", "Wulff",
]


def anonymise(text: str, names: Sequence[str] = FIRM_NAMES,
              placeholder: str = "the company") -> str:
    """
    Replace known firm names with a neutral placeholder.

    Longest names are replaced first so that 'Nokian Tyres' is not half-matched
    by 'Nokia'. Matching is case-sensitive and word-bounded, which is
    deliberately conservative: we would rather miss a name than corrupt an
    unrelated word.
    """
    import re
    out = str(text)
    for n in sorted(names, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(n)}\b", placeholder, out)
    return out


def anonymise_series(s: pd.Series, names: Sequence[str] = FIRM_NAMES,
                     placeholder: str = "the company") -> pd.Series:
    return s.map(lambda t: anonymise(t, names, placeholder))


# --------------------------------------------------------------------------
# 6. Simulation (Lec07b and Lec07c)
# --------------------------------------------------------------------------

def simulate_attention_panel(n_firms: int, dates: Sequence, *,
                             beta_true: float = 0.0,
                             gamma: float = 0.35,
                             rho_sa: float = 0.60,
                             sigma_ret: float | Sequence[float] = 0.018,
                             seed: int = 7) -> pd.DataFrame:
    """
    Firm-day panel in which sentiment has a KNOWN true effect on returns.

    Data-generating process
    -----------------------
        A[i,t] ~ N(0, 1)                                  attention
        S[i,t] = rho_sa * A[i,t] + sqrt(1-rho_sa^2) * e    sentiment
        r[i,t] = beta_true * S[i,t] + gamma * A[i,t] + sigma * u

    With `beta_true = 0`, sentiment does nothing at all. But because sentiment
    is correlated with attention and attention moves returns, a regression of
    r on S alone will report a large, significant, entirely spurious beta.
    That is the point of the exercise.

    Returns a tidy frame: ticker, date, sentiment, attention, ret
    """
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(pd.Index(dates))
    T, N = len(dates), n_firms

    A = rng.standard_normal((T, N))
    S = rho_sa * A + np.sqrt(max(0.0, 1 - rho_sa ** 2)) * rng.standard_normal((T, N))

    sig = np.asarray(sigma_ret, dtype=float)
    if sig.ndim == 0:
        sig = np.full(N, float(sig))
    r = beta_true * S + gamma * A + sig[None, :] * rng.standard_normal((T, N))

    tick = [f"SIM{i:03d}" for i in range(N)]
    return pd.DataFrame({
        "ticker": np.tile(tick, T),
        "date": np.repeat(dates.to_numpy(), N),
        "sentiment": S.ravel(),
        "attention": A.ravel(),
        "ret": r.ravel(),
    })


def omitted_variable_bias(df: pd.DataFrame, y: str = "ret",
                          x: str = "sentiment", z: str = "attention") -> float:
    """
    Analytical bias from leaving `z` out of a regression of `y` on `x`.

        plim(beta_hat) - beta_true = gamma * cov(x, z) / var(x)

    where gamma is the coefficient on z in the correctly specified model.
    Returned so students can check it against the number they estimate.
    """
    import statsmodels.api as sm
    full = sm.OLS(df[y], sm.add_constant(df[[x, z]])).fit()
    gamma = full.params[z]
    delta = df[[x, z]].cov().loc[x, z] / df[x].var()
    return float(gamma * delta)


def simulate_adoption_decay(n_years: int = 4, obs_per_year: int = 252, *,
                            edge0: float = 0.0042,
                            adopters0: float = 0.12,
                            growth: float = 1.2,
                            capacity: float = 0.92,
                            floor: float = 0.16,
                            sigma: float = 0.011,
                            seed: int = 11) -> pd.DataFrame:
    """
    A real signal being competed away as more traders act on it.

    Adoption follows a logistic curve. The edge available to a late arriver is
    the part the crowd has not yet priced, plus a residual floor -- real
    signals decay towards a small persistent edge, not to exactly zero:

        edge_t = edge0 * [ floor + (1 - floor) * (1 - adoption_t / capacity) ]

    The shipped defaults give an annualised Sharpe path of roughly
    5.6 / 4.2 / 2.6 / 1.3 over four years, with adoption rising from about
    20% to 83%. That is deliberately only *similar in shape* to the published
    path in Lopez-Lira and Tang (6.54 / 3.68 / 2.33 / 1.22) -- the parameters
    were not fitted to reproduce it, and the exercise is to compare the two
    half-lives, not to match the levels.

    Returns a daily frame: day, year, adoption, edge, ret
    """
    rng = np.random.default_rng(seed)
    n = n_years * obs_per_year
    t = np.arange(n) / obs_per_year

    k = np.log((capacity / adopters0) - 1)
    adoption = capacity / (1 + np.exp(k - growth * t))

    edge = edge0 * (floor + (1.0 - floor) * (1.0 - adoption / capacity))
    ret = edge + sigma * rng.standard_normal(n)

    return pd.DataFrame({
        "day": np.arange(n),
        "year": (t.astype(int) + 1),
        "adoption": adoption,
        "edge": edge,
        "ret": ret,
    })


def half_life(x: Sequence[float], dt: float = 1.0) -> float:
    """
    Half-life implied by fitting an exponential decay to a positive series.

    Fits log(x) = a + b*t by least squares and returns -log(2)/b. Returns NaN
    if the series does not decay.
    """
    x = np.asarray(list(x), dtype=float)
    ok = np.isfinite(x) & (x > 0)
    if ok.sum() < 3:
        return float("nan")
    t = np.arange(len(x), dtype=float)[ok] * dt
    b = np.polyfit(t, np.log(x[ok]), 1)[0]
    return float(-np.log(2) / b) if b < 0 else float("nan")


def sharpe(returns: Sequence[float], periods: int = 252) -> float:
    """Annualised Sharpe ratio of a return series (zero risk-free)."""
    r = np.asarray(list(returns), dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))
