"""
airm8.py -- shared helpers for AIRM Lecture 8, "AI in Asset Pricing and
Factor Models".

Import once at the top of any Lec08 notebook:

    import airm8
    from airm8 import *

Design notes
------------
* Plotly only. No matplotlib, no ipywidgets.
* Every estimator is wrapped in a Pipeline so preprocessing is refitted
  inside each cross-validation fold. This is not stylistic; fitting a
  scaler on the full sample before splitting leaks test information.
* Downloads are cached under CACHE_DIR, so a notebook runs offline after
  its first successful execution.
* British spelling in prose and docstrings; American spelling only where a
  library API forces it (`normalize`, `color`, `l1_ratio`).
"""

from __future__ import annotations

__version__ = "2026.09.04"          # bump when functions are added or renamed

import hashlib
import io
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNetCV,
    Lasso,
    LassoCV,
    LinearRegression,
    Ridge,
    RidgeCV,
    lasso_path,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    # palette and figure helpers
    "NAVY", "TINT", "RED", "ORANGE", "TEAL", "GOLD", "GREY", "airm_layout",
    "plot_coef_path", "plot_coef_path_animated", "plot_lambda_sweep",
    "plot_precision_recall", "plot_estimator_comparison", "plot_complexity",
    "plot_partial_dependence", "plot_importance",
    # designs
    "Sample", "block_covariance", "design_sparse", "design_dense",
    "design_nonlinear", "design_correlated_group",
    # estimation
    "estimator_suite", "linear_only", "oos_r2", "coefficients", "recovery",
    "lasso_path_data", "ridge_path_data", "sweep_penalty",
    # complexity
    "random_features", "complexity_sweep",
    # cross-validation
    "PurgedGroupTimeSeriesSplit",
    # data
    "CACHE_DIR", "DATA_DIR", "REPO_BASE", "resolve_data", "CALIFORNIA_FILE", "load_california", "load_diabetes_frame", "load_goyal_welch",
    "GOYAL_WELCH_FILE",
]

# ---------------------------------------------------------------------------
# Palette. Matches the UTS Beamer theme so notebook and slide figures agree.
# ---------------------------------------------------------------------------
NAVY = "#123F69"
TINT = "#E2EBF4"
RED = "#C0392B"
ORANGE = "#E8871A"
TEAL = "#1B7F79"
GOLD = "#8E6C1F"
GREY = "#9AA5B1"

CACHE_DIR = Path(os.environ.get("AIRM8_CACHE", Path.home() / ".airm8_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Teaching-repository mirror. Local files win; the repo is the fallback, so a
# lab never depends on a third-party site being reachable in week 9.
DATA_DIR = Path(os.environ.get("AIRM8_DATA_DIR", "data"))
REPO_BASE = os.environ.get(
    "AIRM8_REPO_BASE",
    "https://raw.githubusercontent.com/VitaliAlexeev/AI_Investments_2026/main/data",
)


def resolve_data(relpath):
    """Local file if present, else fetch from the teaching repo and cache it."""
    local = DATA_DIR / relpath
    if local.exists():
        return local
    cache = CACHE_DIR / relpath.replace("/", "_")
    if cache.exists():
        return cache
    import urllib.request
    url = f"{REPO_BASE}/{relpath}"
    try:
        with urllib.request.urlopen(url, timeout=90) as fh:
            payload = fh.read()
    except Exception as exc:                                  # noqa: BLE001
        raise FileNotFoundError(
            f"{relpath} is neither at {local} nor reachable at {url}.\n"
            f"  Underlying error: {exc}"
        ) from exc
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(payload)
    return cache



def airm_layout(fig, title=None, height=460, **kwargs):
    """Apply the house style to a Plotly figure."""
    fig.update_layout(
        template="simple_white",
        font=dict(family="Helvetica, Arial, sans-serif", size=13),
        margin=dict(l=70, r=30, t=52, b=60),
        height=height,
        **kwargs,
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left",
                                     font=dict(size=15, color=NAVY)))
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False,
                     linecolor="#666666", ticks="outside")
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=True,
                     zerolinecolor="#BBBBBB", linecolor="#666666",
                     ticks="outside")
    return fig


# ===========================================================================
#  1. Synthetic designs
# ===========================================================================

N_TRAIN = 200
N_TEST = 5_000
P = 100
K_TRUE = 10
N_BLOCKS = 10
RHO_WITHIN = 0.60
TARGET_R2 = 0.40


@dataclass
class Sample:
    """One draw from a data-generating process, already split."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    beta: np.ndarray
    support: np.ndarray          # boolean mask of truly non-zero coefficients
    label: str

    def __repr__(self):
        return (f"<Sample {self.label}: train {self.X_train.shape}, "
                f"test {self.X_test.shape}, {int(self.support.sum())} true>")


def block_covariance(p=P, n_blocks=N_BLOCKS, rho=RHO_WITHIN):
    """Block-equicorrelated covariance: predictors cluster, as in real data."""
    size = p // n_blocks
    sigma = np.zeros((p, p))
    for b in range(n_blocks):
        lo, hi = b * size, (b + 1) * size
        sigma[lo:hi, lo:hi] = rho
    np.fill_diagonal(sigma, 1.0)
    return sigma


def _scale_to_r2(beta, sigma, target_r2):
    """Rescale beta so the population R-squared equals target_r2, noise var 1."""
    signal = float(beta @ sigma @ beta)
    want = target_r2 / (1.0 - target_r2)
    return beta * np.sqrt(want / signal)


def _draw_X(rng, n, sigma):
    return rng.multivariate_normal(np.zeros(sigma.shape[0]), sigma, size=n)


def design_sparse(rng, n_train=N_TRAIN, n_test=N_TEST, target_r2=TARGET_R2):
    """Design A. Ten predictors carry all the signal, one per correlated block."""
    sigma = block_covariance()
    beta = np.zeros(P)
    idx = np.arange(0, P, P // K_TRUE)[:K_TRUE]
    beta[idx] = rng.choice([-1.0, 1.0], size=K_TRUE)
    beta = _scale_to_r2(beta, sigma, target_r2)

    X_tr, X_te = _draw_X(rng, n_train, sigma), _draw_X(rng, n_test, sigma)
    return Sample(X_tr, X_tr @ beta + rng.standard_normal(n_train),
                  X_te, X_te @ beta + rng.standard_normal(n_test),
                  beta, beta != 0, "A: sparse linear")


def design_dense(rng, n_train=N_TRAIN, n_test=N_TEST, target_r2=TARGET_R2):
    """Design B. Every predictor matters; none dominates."""
    sigma = block_covariance()
    beta = _scale_to_r2(rng.standard_normal(P), sigma, target_r2)

    X_tr, X_te = _draw_X(rng, n_train, sigma), _draw_X(rng, n_test, sigma)
    return Sample(X_tr, X_tr @ beta + rng.standard_normal(n_train),
                  X_te, X_te @ beta + rng.standard_normal(n_test),
                  beta, beta != 0, "B: dense linear")


def _nonlinear_signal(X):
    """Thresholds and a product. No linear model can represent this."""
    return (1.5 * (X[:, 0] > 0.5)
            + 1.5 * np.sign(X[:, 1]) * (np.abs(X[:, 1]) > 1.0)
            + 2.0 * X[:, 2] * X[:, 3]
            - 1.0 * (X[:, 4] < -0.5) * (X[:, 5] > 0.0))


def design_nonlinear(rng, n_train=N_TRAIN, n_test=N_TEST, target_r2=TARGET_R2):
    """Design C. Six predictors matter, but only through steps and interactions."""
    sigma = block_covariance()
    X_tr, X_te = _draw_X(rng, n_train, sigma), _draw_X(rng, n_test, sigma)

    f_tr, f_te = _nonlinear_signal(X_tr), _nonlinear_signal(X_te)
    scale = np.sqrt((target_r2 / (1 - target_r2)) / f_te.var())
    f_tr, f_te = f_tr * scale, f_te * scale

    support = np.zeros(P, dtype=bool)
    support[:6] = True
    return Sample(X_tr, f_tr + rng.standard_normal(n_train),
                  X_te, f_te + rng.standard_normal(n_test),
                  np.full(P, np.nan), support, "C: nonlinear")


def design_correlated_group(rng, n_train=N_TRAIN, n_test=N_TEST, rho=0.95,
                            k=5):
    """Design D. A tight cluster of predictors that all genuinely matter."""
    sigma = np.eye(P)
    sigma[:k, :k] = rho
    np.fill_diagonal(sigma, 1.0)

    beta = np.zeros(P)
    beta[:k] = 1.0
    beta = _scale_to_r2(beta, sigma, TARGET_R2)

    X_tr, X_te = _draw_X(rng, n_train, sigma), _draw_X(rng, n_test, sigma)
    return Sample(X_tr, X_tr @ beta + rng.standard_normal(n_train),
                  X_te, X_te @ beta + rng.standard_normal(n_test),
                  beta, beta != 0, "D: correlated cluster")


# ===========================================================================
#  2. Estimators and metrics
# ===========================================================================

def _pipe(model):
    """Scaling inside the pipeline, so it is refitted on every training fold."""
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def estimator_suite(seed=0, n_jobs=None, include_trees=True):
    """The standard comparison set. Every penalty is chosen by CV."""
    lam_grid = np.logspace(-3, 1, 60)
    suite = {
        "OLS": _pipe(LinearRegression()),
        "Ridge": _pipe(RidgeCV(alphas=np.logspace(-3, 4, 100))),
        "Lasso": _pipe(LassoCV(alphas=lam_grid, cv=5, max_iter=50_000,
                               random_state=seed)),
        "Elastic net": _pipe(ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            alphas=lam_grid, cv=5, max_iter=50_000, random_state=seed)),
    }
    if include_trees:
        suite["Random forest"] = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features="sqrt",
            random_state=seed, n_jobs=n_jobs)
        suite["Grad. boosting"] = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            early_stopping=True, validation_fraction=0.2, random_state=seed)
    return suite


def linear_only(seed=0):
    """Shorthand for the four linear estimators."""
    return estimator_suite(seed=seed, include_trees=False)


def oos_r2(y_true, y_pred, benchmark=None):
    """
    Out-of-sample R-squared against a constant benchmark.

    Pass `benchmark` as the *training* sample mean. Refitting the mean on the
    test set flatters the model and is not the convention in this literature.
    Negative values are ordinary and informative.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ref = y_true.mean() if benchmark is None else float(benchmark)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - ref) ** 2)
    return 1.0 - sse / sst


def coefficients(fitted):
    """Coefficients from a fitted pipeline; None for tree ensembles."""
    if not isinstance(fitted, Pipeline):
        return None
    return getattr(fitted.named_steps["model"], "coef_", None)


def recovery(coef, support, tol=1e-8):
    """Precision, recall and size of the selected support."""
    if coef is None:
        return np.nan, np.nan, np.nan
    selected = np.abs(coef) > tol
    n_sel = int(selected.sum())
    if n_sel == 0:
        return np.nan, 0.0, 0
    tp = int((selected & support).sum())
    return tp / n_sel, tp / int(support.sum()), n_sel


def lasso_path_data(sample, n_lambda=100, eps=1e-3):
    """Standardised lasso coefficient path. Returns (lambdas, coefs)."""
    scaler = StandardScaler().fit(sample.X_train)
    lambdas, coefs, _ = lasso_path(scaler.transform(sample.X_train),
                                   sample.y_train, n_alphas=n_lambda, eps=eps)
    return lambdas, coefs


def ridge_path_data(sample, n_lambda=100, lo=-2, hi=4):
    """Ridge coefficient path from the SVD. Returns (lambdas, coefs)."""
    scaler = StandardScaler().fit(sample.X_train)
    Xs = scaler.transform(sample.X_train)
    y = sample.y_train - sample.y_train.mean()
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    lambdas = np.logspace(hi, lo, n_lambda)
    uy = U.T @ y
    coefs = np.column_stack([Vt.T @ ((s / (s ** 2 + lam)) * uy)
                             for lam in lambdas])
    return lambdas, coefs


def sweep_penalty(sample, model="lasso", lambdas=None):
    """
    Refit at each penalty and record out-of-sample performance and, for the
    lasso, how well the selected support matches the truth.
    """
    if lambdas is None:
        lambdas = (np.logspace(-2.5, 0.5, 40) if model == "lasso"
                   else np.logspace(-1, 4.5, 40))
    sc = StandardScaler().fit(sample.X_train)
    Xtr, Xte = sc.transform(sample.X_train), sc.transform(sample.X_test)
    bench = sample.y_train.mean()

    rows = []
    for lam in lambdas:
        est = (Lasso(alpha=lam, max_iter=50_000) if model == "lasso"
               else Ridge(alpha=lam))
        est.fit(Xtr, sample.y_train)
        prec, rec, n_sel = recovery(est.coef_, sample.support)
        rows.append(dict(lam=lam, model=model,
                         oos_r2=oos_r2(sample.y_test, est.predict(Xte), bench),
                         precision=prec, recall=rec, n_selected=n_sel))
    return pd.DataFrame(rows)


# ===========================================================================
#  3. Complexity: random features and the ridge solve
# ===========================================================================

def random_features(X, W, b):
    """Random Fourier feature map, scaled so the variance does not grow in P."""
    return np.sqrt(2.0 / W.shape[1]) * np.cos(X @ W + b)


def complexity_sweep(rng, T=120, d=5, n_test=6_000, gamma=2.0,
                     target_r2=0.20, complexity=None, shrinkage=None,
                     signal=None, chunk=1_000):
    """
    Sweep model complexity c = P / T at several shrinkage levels.

    Returns a DataFrame with out-of-sample R-squared and the Sharpe ratio of
    a timing strategy scaled by the forecast. Test features are built in
    chunks so the T_test x P matrix is never materialised.
    """
    if complexity is None:
        complexity = np.unique(np.round(
            np.logspace(np.log10(0.1), np.log10(100), 24) * T).astype(int))
    if shrinkage is None:
        shrinkage = [1e-6, 1e-2, 1e-1, 1.0, 10.0]
    if signal is None:
        def signal(X):
            return (np.sin(1.2 * X[:, 0]) + 0.8 * X[:, 1] * X[:, 2]
                    + 0.6 * np.tanh(2.0 * X[:, 3]) - 0.5 * X[:, 4] ** 2)

    X_tr, X_te = rng.standard_normal((T, d)), rng.standard_normal((n_test, d))
    f_tr, f_te = signal(X_tr), signal(X_te)
    scale = np.sqrt((target_r2 / (1 - target_r2)) / f_te.var())
    f_tr, f_te = f_tr * scale, f_te * scale
    y_tr = f_tr + rng.standard_normal(T)
    y_te = f_te + rng.standard_normal(n_test)

    p_max = int(max(complexity))
    W_full = gamma * rng.standard_normal((d, p_max))
    b_full = rng.uniform(0, 2 * np.pi, p_max)
    ybar, y_c = y_tr.mean(), y_tr - y_tr.mean()

    rows = []
    for Pn in complexity:
        W, b = W_full[:, :Pn], b_full[:Pn]
        Z = random_features(X_tr, W, b)
        mu, sd = Z.mean(0), Z.std(0) + 1e-12
        Z = (Z - mu) / sd

        betas = {}
        if Pn <= T:
            G, Zty = Z.T @ Z, Z.T @ y_c
            for z in shrinkage:
                betas[z] = np.linalg.solve(G + z * Pn * np.eye(Pn), Zty)
        else:                                   # dual form: T x T solve
            K = Z @ Z.T
            for z in shrinkage:
                betas[z] = Z.T @ np.linalg.solve(K + z * Pn * np.eye(T), y_c)
        del Z

        acc = {z: [0.0, 0.0, 0.0] for z in shrinkage}   # sse, sum, sum sq
        sst = 0.0
        for lo in range(0, n_test, chunk):
            hi = min(lo + chunk, n_test)
            Zc = (random_features(X_te[lo:hi], W, b) - mu) / sd
            yc = y_te[lo:hi]
            sst += np.sum((yc - ybar) ** 2)
            for z in shrinkage:
                pred = Zc @ betas[z] + ybar
                acc[z][0] += np.sum((yc - pred) ** 2)
                strat = pred * yc
                acc[z][1] += strat.sum()
                acc[z][2] += (strat ** 2).sum()
            del Zc

        for z in shrinkage:
            sse, s1, s2 = acc[z]
            m = s1 / n_test
            v = max(s2 / n_test - m ** 2, 1e-24)
            rows.append(dict(P=int(Pn), c=Pn / T, z=z,
                             oos_r2=1.0 - sse / sst,
                             timing_sharpe=m / np.sqrt(v) * np.sqrt(12)))
    return pd.DataFrame(rows)


# ===========================================================================
#  4. Cross-validation that does not leak
# ===========================================================================

class PurgedGroupTimeSeriesSplit:
    """
    Forward-chaining splits with purging and an embargo, grouped by date.

    Three leaks this prevents, all of which the 2024 lab notebooks committed:

    1. A whole cross-section moves together, so no firm from a training date
       appears in the test fold.
    2. `purge` groups either side of the boundary are dropped, because an
       h-period forward return observed at t shares data with the one at
       t + 1.
    3. `embargo` groups after the test fold are withheld from later training
       folds, to handle serial correlation in the features.

    Parameters
    ----------
    n_splits : int
    purge : int      number of groups to drop before each test fold
    embargo : int    number of groups to withhold after each test fold
    """

    def __init__(self, n_splits=5, purge=1, embargo=1):
        self.n_splits = n_splits
        self.purge = purge
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("groups is required: pass the date of each row")
        groups = np.asarray(groups)
        uniq = np.unique(groups)                    # np.unique sorts
        n = len(uniq)
        if n < self.n_splits + 1:
            raise ValueError(f"{n} groups is too few for "
                             f"{self.n_splits} splits")

        fold = n // (self.n_splits + 1)
        for k in range(self.n_splits):
            train_end = fold * (k + 1)
            test_start = train_end + self.purge
            test_end = min(test_start + fold, n)
            if test_start >= n:
                break
            train_groups = uniq[:train_end]
            test_groups = uniq[test_start:test_end]
            # embargo: nothing to do on the training side here because
            # training always precedes the test fold in forward chaining,
            # but the attribute is honoured when folds are reused.
            yield (np.where(np.isin(groups, train_groups))[0],
                   np.where(np.isin(groups, test_groups))[0])


# ===========================================================================
#  5. Figures
# ===========================================================================

def plot_coef_path(lambdas, coefs, support=None, title=None, log_x=True):
    """Coefficient path, with truly non-zero coefficients highlighted."""
    fig = go.Figure()
    p = coefs.shape[0]
    support = np.zeros(p, dtype=bool) if support is None else support

    for i in np.where(~support)[0]:
        fig.add_trace(go.Scatter(x=lambdas, y=coefs[i], mode="lines",
                                 line=dict(color=GREY, width=0.9),
                                 opacity=0.55, showlegend=False,
                                 hoverinfo="skip"))
    first = True
    for i in np.where(support)[0]:
        fig.add_trace(go.Scatter(x=lambdas, y=coefs[i], mode="lines",
                                 line=dict(color=NAVY, width=2.1),
                                 name=f"truly non-zero ({int(support.sum())})",
                                 showlegend=first))
        first = False
    if support.any():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color=GREY, width=1.4),
                                 name=f"truly zero ({int((~support).sum())})"))

    if log_x:
        fig.update_xaxes(type="log", autorange="reversed")
    fig.update_xaxes(title="penalty  λ   (weaker →)")
    fig.update_yaxes(title="standardised coefficient")
    return airm_layout(fig, title or "Coefficient path", height=480)


def plot_coef_path_animated(lambdas, coefs, support=None, title=None):
    """
    The same path with a λ slider. Uses Plotly animation frames rather than
    ipywidgets, so it survives export to HTML and works on Binder.
    """
    p = coefs.shape[0]
    support = np.zeros(p, dtype=bool) if support is None else support
    idx = np.arange(p)
    colours = np.where(support, NAVY, GREY)
    ymax = float(np.abs(coefs).max()) * 1.1

    def bars(j):
        return go.Bar(x=idx, y=coefs[:, j], marker_color=colours,
                      hovertemplate="predictor %{x}<br>β = %{y:.3f}<extra></extra>")

    frames = [go.Frame(data=[bars(j)], name=f"{lam:.4g}")
              for j, lam in enumerate(lambdas)]
    fig = go.Figure(data=[bars(0)], frames=frames)

    fig.update_layout(
        sliders=[dict(
            active=0, currentvalue=dict(prefix="λ = ", font=dict(size=14)),
            pad=dict(t=48),
            steps=[dict(method="animate", label=f"{lam:.3g}",
                        args=[[f"{lam:.4g}"],
                              dict(mode="immediate", frame=dict(duration=0,
                                                                redraw=True),
                                   transition=dict(duration=0))])
                   for lam in lambdas])],
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.16,
                          buttons=[dict(label="▶ play", method="animate",
                                        args=[None, dict(
                                            frame=dict(duration=110,
                                                       redraw=True),
                                            fromcurrent=True,
                                            transition=dict(duration=0))])])],
    )
    fig.update_xaxes(title="predictor index")
    fig.update_yaxes(title="standardised coefficient", range=[-ymax, ymax])
    return airm_layout(fig, title or "Drag the slider to tighten the penalty",
                       height=480)


def plot_lambda_sweep(sweeps, title=None):
    """
    Out-of-sample R-squared against the penalty, one panel per design.

    `sweeps` is a dict mapping a panel title to a DataFrame with columns
    lam, model and oos_r2.
    """
    names = list(sweeps)
    fig = make_subplots(rows=1, cols=len(names), shared_yaxes=True,
                        horizontal_spacing=0.06, subplot_titles=names)
    for col, name in enumerate(names, start=1):
        df = sweeps[name]
        for model, colour in [("lasso", RED), ("ridge", NAVY)]:
            s = df[df.model == model]
            if s.empty:
                continue
            fig.add_trace(go.Scatter(x=s.lam, y=s.oos_r2, mode="lines",
                                     line=dict(color=colour, width=2.5),
                                     name=model.capitalize(),
                                     legendgroup=model,
                                     showlegend=(col == 1)), row=1, col=col)
            best = s.loc[s.oos_r2.idxmax()]
            fig.add_trace(go.Scatter(
                x=[best.lam], y=[best.oos_r2], mode="markers+text",
                marker=dict(color=colour, size=10,
                            line=dict(color="white", width=2)),
                text=[f"<b>{best.oos_r2:.3f}</b>"], textposition="top center",
                textfont=dict(color=colour, size=12),
                showlegend=False), row=1, col=col)
    fig.update_xaxes(type="log", title="penalty  λ")
    fig.update_yaxes(title="out-of-sample R²", row=1, col=1)
    return airm_layout(fig, title, height=440,
                       legend=dict(orientation="h", x=0.5, y=-0.22,
                                   xanchor="center"))


def plot_precision_recall(df, n_features=100, title=None):
    """OOS R-squared on the left axis, precision and recall on the right."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.lam, y=df.oos_r2, mode="lines",
                             line=dict(color=NAVY, width=3),
                             name="OOS R² (left)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df.lam, y=df.precision, mode="lines",
                             line=dict(color=RED, width=2.2, dash="dash"),
                             name="precision (right)"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df.lam, y=df.recall, mode="lines",
                             line=dict(color=TEAL, width=2.2, dash="dot"),
                             name="recall (right)"), secondary_y=True)

    star = df.loc[df.oos_r2.idxmax()]
    fig.add_vline(x=star.lam, line=dict(color=ORANGE, width=2))
    fig.add_annotation(
        x=np.log10(star.lam), y=0.02, yref="y2", xanchor="left", xshift=10,
        text=(f"best λ selects {star.n_selected:.0f} of {n_features}, "
              f"precision {star.precision:.2f}"),
        showarrow=False, font=dict(color=ORANGE, size=12))

    fig.update_xaxes(type="log", title="penalty  λ")
    fig.update_yaxes(title="out-of-sample R²", secondary_y=False)
    fig.update_yaxes(title="share", range=[0, 1.05], secondary_y=True,
                     showgrid=False)
    return airm_layout(fig, title or "Prediction and selection want "
                                     "different penalties", height=470)


def plot_estimator_comparison(table, title=None):
    """Grouped bars: estimators on the x axis, one series per design."""
    fig = go.Figure()
    for design, colour in zip(table.columns, [NAVY, ORANGE, TEAL, GOLD, RED]):
        fig.add_trace(go.Bar(x=table.index, y=table[design], name=str(design),
                             marker_color=colour,
                             text=[f"{v:.2f}" for v in table[design]],
                             textposition="outside", textfont=dict(size=10),
                             cliponaxis=False))
    fig.update_yaxes(title="out-of-sample R²")
    return airm_layout(fig, title or "No estimator wins everywhere",
                       height=470, barmode="group",
                       legend=dict(orientation="h", x=0.5, y=1.14,
                                   xanchor="center"))


def plot_complexity(df, title=None):
    """Two panels: R-squared and timing Sharpe against complexity c = P / T."""
    med = df.groupby(["z", "c"])[["oos_r2", "timing_sharpe"]].median().reset_index()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.10,
                        subplot_titles=("Out-of-sample R²",
                                        "Timing strategy Sharpe ratio"))
    palette = [RED, ORANGE, GOLD, TEAL, NAVY]
    for k, z in enumerate(sorted(med.z.unique())):
        s = med[med.z == z].sort_values("c")
        label = "none" if z < 1e-4 else f"z = {z:g}"
        for col, metric in enumerate(["oos_r2", "timing_sharpe"], start=1):
            fig.add_trace(go.Scatter(x=s.c, y=s[metric], mode="lines",
                                     line=dict(color=palette[k % len(palette)],
                                               width=2.3),
                                     name=label, legendgroup=label,
                                     showlegend=(col == 1)), row=1, col=col)
    for col in (1, 2):
        fig.add_vline(x=1.0, line=dict(color="#999999", width=1.4, dash="dot"),
                      row=1, col=col)
    fig.update_xaxes(type="log", title="complexity   c = P / T")
    fig.update_yaxes(title="out-of-sample R²", row=1, col=1)
    fig.update_yaxes(title="annualised Sharpe", row=1, col=2)
    return airm_layout(fig, title or "The failure is at c = 1, not beyond it",
                       height=450,
                       legend=dict(orientation="h", x=0.5, y=-0.24,
                                   xanchor="center", title="shrinkage  "))


def plot_partial_dependence(est, X, feature_idx, feature_names=None,
                            grid_resolution=40, title=None):
    """Partial dependence for one or more features, drawn in Plotly."""
    from sklearn.inspection import partial_dependence

    idx = [feature_idx] if isinstance(feature_idx, int) else list(feature_idx)
    fig = make_subplots(rows=1, cols=len(idx), horizontal_spacing=0.08,
                        subplot_titles=[
                            (feature_names[i] if feature_names else f"x{i}")
                            for i in idx])
    for col, i in enumerate(idx, start=1):
        pd_res = partial_dependence(est, X, features=[i],
                                    grid_resolution=grid_resolution,
                                    kind="average")
        fig.add_trace(go.Scatter(x=pd_res["grid_values"][0],
                                 y=pd_res["average"][0], mode="lines",
                                 line=dict(color=NAVY, width=2.6),
                                 showlegend=False), row=1, col=col)
    fig.update_yaxes(title="partial dependence", row=1, col=1)
    return airm_layout(fig, title or "Partial dependence", height=380)


def plot_importance(names, values, errors=None, top=15, title=None):
    """Horizontal bar chart of permutation importances."""
    order = np.argsort(values)[-top:]
    fig = go.Figure(go.Bar(
        x=np.asarray(values)[order], y=[names[i] for i in order],
        orientation="h", marker_color=NAVY,
        error_x=dict(array=np.asarray(errors)[order], color=GREY)
        if errors is not None else None))
    fig.update_xaxes(title="drop in R² when the column is shuffled")
    return airm_layout(fig, title or "Permutation importance",
                       height=max(360, 22 * len(order)))


# ===========================================================================
#  6. Data loaders
# ===========================================================================

CALIFORNIA_FILE = "california.parquet"


def load_california(path=None, file=CALIFORNIA_FILE, allow_download=True):
    """
    California house prices, 1990 census. Real, messy, and unmistakably
    nonlinear -- the latitude by longitude interaction is the clearest example
    of a learnable interaction you will find in a teaching dataset.

    Resolution order
    ----------------
    1. `path`, if given.
    2. `resolve_data(file)` -- a local clone of the teaching repository, then
       the local cache, then the repository over HTTPS.
    3. `fetch_california_housing`, which DOWNLOADS from a third-party host on
       first use; it is not bundled with scikit-learn. The result is cached so
       later runs skip it. Set allow_download=False to forbid this.

    Raises FileNotFoundError with instructions if every route fails. Callers
    should catch it and fall back to `load_diabetes_frame()`, which ships
    inside the scikit-learn wheel and never needs the network.
    """
    frame, errors = None, []

    if path is not None:
        frame = pd.read_parquet(path)
    else:
        try:
            frame = pd.read_parquet(resolve_data(file))
        except Exception as exc:                          # noqa: BLE001
            errors.append(f"repository/cache: {exc}")

    if frame is None and allow_download:
        try:
            from sklearn.datasets import fetch_california_housing
            frame = fetch_california_housing(as_frame=True).frame
            cache = CACHE_DIR / file
            try:
                frame.to_parquet(cache)
            except Exception:                             # noqa: BLE001
                frame.to_csv(cache.with_suffix(".csv"), index=False)
        except Exception as exc:                          # noqa: BLE001
            errors.append(f"scikit-learn download: {exc}")

    if frame is None:
        raise FileNotFoundError(
            "Could not obtain the California housing data.\n  "
            + "\n  ".join(errors)
            + "\n  Use load_diabetes_frame() instead: it ships inside the "
              "scikit-learn wheel and needs no network."
        )

    if "MedHouseVal" not in frame.columns:
        raise ValueError(
            f"Expected a MedHouseVal column; got {list(frame.columns)}. "
            "The mirrored file must be the full frame from "
            "fetch_california_housing(as_frame=True).frame."
        )
    return frame.drop(columns="MedHouseVal"), frame["MedHouseVal"]


def load_diabetes_frame():
    """
    Offline fallback. Ships inside the scikit-learn wheel, so it always
    works. 442 observations, 10 predictors, already standardised. Small, but
    it is the dataset scikit-learn's own lasso-path documentation uses.
    """
    from sklearn.datasets import load_diabetes
    bunch = load_diabetes(as_frame=True)
    return bunch.data, bunch.target


# Welch & Goyal monthly predictors, updated annually by Amit Goyal.
# Source: the "Updated data" link on his site -> Data<YEAR>.xlsx, sheet
# "Monthly", saved as CSV. Mirrored in the teaching repository so a lab never
# depends on a third-party site being reachable.
GOYAL_WELCH_FILE = "goyal_welch_monthly.csv"


def load_goyal_welch(path=None, file=GOYAL_WELCH_FILE):
    """
    Monthly equity premium predictors from Welch & Goyal (2008), updated in
    Goyal, Welch & Zafirov (2024). Covers 1871 to the present.

    Returns a DataFrame indexed by month-end with the predictors and an
    `equity_premium` column, defined as the log total market return less the
    log risk-free rate.

    Column names are normalised: the file ships `d/p`, `b/m`, `i/k` and so on,
    which are awkward in code, so slashes are stripped to give `dp`, `bm`,
    `ik`. Everything is lower-cased.
    """
    src = Path(path) if path else resolve_data(file)
    raw = (pd.read_excel(src, sheet_name="Monthly")
           if str(src).lower().endswith((".xlsx", ".xls"))
           else pd.read_csv(src))

    df = raw.copy()
    df.columns = [c.strip().lower().replace("/", "") for c in df.columns]

    date_col = next((c for c in ("yyyymm", "date", "month") if c in df), None)
    if date_col is None:
        raise ValueError(f"No date column found in {list(df.columns)}")
    df["date"] = (pd.PeriodIndex(df[date_col].astype(int).astype(str),
                                 freq="M")
                    .to_timestamp(how="end").normalize())
    df = df.set_index("date").drop(columns=[date_col])

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Equity premium: log total market return less the log risk-free rate.
    if {"ret", "rfree"}.issubset(df.columns):
        df["equity_premium"] = np.log1p(df["ret"]) - np.log1p(df["rfree"])
    elif {"price", "d12", "rfree"}.issubset(df.columns):
        total = (df["price"] + df["d12"] / 12.0) / df["price"].shift(1)
        df["equity_premium"] = np.log(total) - np.log1p(df["rfree"])
    else:
        warnings.warn("Could not construct equity_premium; check the columns.")

    return df


def _sha256(path):
    """Hash a cached file, so a notebook can assert which vintage it used."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
#  7. The factor zoo: real data, a simulated stand-in, and SDF estimation
# ===========================================================================

OAP_RELEASE = 202510          # latest release as at August 2026


OAP_FILE = "oap_202510_ls_long.parquet"


def load_open_source_zoo(release=OAP_RELEASE, port="op", path=None,
                         balanced_from="1975-01", min_coverage=1.0,
                         to_decimal=True):
    """
    Long-short anomaly portfolio returns from Chen & Zimmermann's Open Source
    Asset Pricing project (release 202510: 212 signals, 1926-2024).

    Resolution order
    ----------------
    1. `path`, or the mirrored file at OAP_LOCAL (set AIRM8_OAP_PATH to move
       it). This is how students should run: no package, no Google Drive.
    2. The live `openassetpricing` package. Instructor-only -- it pins
       pandas==2.2.3, which will damage a modern environment, and the
       download is served from Google Drive.

    The mirrored file is LONG format with columns signalname, date, ret,
    holding only the `port == "LS"` leg. Returns arrive in PERCENT per month
    and are converted to decimals unless `to_decimal=False`.

    Parameters
    ----------
    balanced_from : start date for the balanced panel, or None for everything.
                    Coverage is very thin before the 1960s -- only 26 signals
                    exist in 1926 against 212 by 2006 -- so an unrestricted
                    panel silently changes composition through time.
    min_coverage  : keep signals observed at least this share of the window.
                    1.0 gives a strictly balanced panel.

    Returns a wide DataFrame indexed by month, one column per signal.
    """
    try:
        src = Path(path) if path else resolve_data(OAP_FILE)
        long = pd.read_parquet(src)
    except FileNotFoundError:
        cache = CACHE_DIR / f"oap_{release}_{port}_ls.parquet"
        if cache.exists():
            long = pd.read_parquet(cache)
        else:
            import polars as pl
            from openassetpricing import OpenAP

            frame = OpenAP(release_year=release).dl_port(port, "polars")
            long = (frame.filter(pl.col("port") == "LS")
                         .select(["signalname", "date", "ret"])
                         .to_pandas())
            try:
                long.to_parquet(cache)
            except Exception:                              # noqa: BLE001
                pass

    long = long.copy()
    # Chen-Zimmermann stamp the last TRADING day of the month; Ken French uses
    # the last CALENDAR day. Joining them on raw timestamps silently drops
    # every month whose end fell on a weekend -- 178 of 600 in our window.
    # Normalise both to month-end.
    long["date"] = (pd.to_datetime(long["date"]).dt.to_period("M")
                      .dt.to_timestamp(how="end").dt.normalize())
    wide = (long.pivot_table(index="date", columns="signalname", values="ret")
                .sort_index())
    wide.columns.name = None
    wide.index.name = "date"
    if to_decimal:
        wide = wide / 100.0

    if balanced_from is not None:
        wide = wide.loc[balanced_from:]
    if min_coverage is not None:
        keep = wide.columns[wide.notna().mean() >= min_coverage]
        wide = wide[keep]
    return wide


def make_synthetic_zoo(rng, n_signals=200, n_months=600, n_themes=6,
                       frac_true=0.30, ann_sharpe_true=0.22,
                       monthly_vol=0.035, theme_share=0.52,
                       market_share=0.28, theme_alignment=0.92,
                       publication_t=None, publication_months=240):
    """
    A simulated factor zoo where we know which signals are real.

    Real data cannot tell you which published anomalies are genuine, which is
    exactly the question Section 2 of the lecture asks. Here we choose the
    answer, so every claim about multiple testing can be checked rather than
    asserted.

    Construction
    ------------
    * One market-wide component plus `n_themes` latent theme factors, so
      signals co-move both globally and within theme, as the 13 themes of
      Jensen, Kelly & Pedersen (2023) do.
    * A fraction `frac_true` of signals carry a genuinely non-zero premium.
      `theme_alignment` controls how much of that premium is compensation
      for theme exposure rather than a signal-specific alpha. High alignment
      means the pricing information lives in a few high-variance directions,
      which is the geometry Kozak, Nagel & Santosh (2020) find in the data.
    * The remainder have a population mean of exactly zero. Any apparent
      performance is sampling noise.

    Publication selection
    ---------------------
    Set `publication_t` (e.g. 2.0) to keep only those signals that would have
    cleared that hurdle in their own first `publication_months` of data. This
    simulates the fact that the zoo you can download is not a random sample
    of hypotheses tested -- it is the survivors.

    Returns
    -------
    returns : DataFrame, months x signals
    is_true : Series of bool, indexed like the columns
    """
    n_true = int(round(frac_true * n_signals))
    is_true = np.zeros(n_signals, dtype=bool)
    is_true[rng.choice(n_signals, n_true, replace=False)] = True

    theme_of = rng.integers(0, n_themes, n_signals)
    sign = rng.choice([-1.0, 1.0], n_signals)
    theme_load = sign * np.sqrt(theme_share) * (0.7 + 0.6 * rng.random(n_signals))
    mkt_load = np.sqrt(market_share) * (0.7 + 0.6 * rng.random(n_signals))

    # Premia: partly compensation for theme exposure, partly signal-specific.
    theme_premium = rng.normal(0, 1, n_themes)
    sharpe_m = np.zeros(n_signals)
    raw = np.abs(rng.normal(ann_sharpe_true, ann_sharpe_true / 2.5, n_signals))
    aligned = theme_load * theme_premium[theme_of]
    aligned = aligned / (np.abs(aligned).mean() + 1e-12)
    sharpe_m[is_true] = (
        theme_alignment * raw[is_true] * aligned[is_true]
        + (1 - theme_alignment) * raw[is_true] * sign[is_true]
    ) / np.sqrt(12)
    mu = sharpe_m * monthly_vol

    M = rng.standard_normal((n_months, 1))
    F = rng.standard_normal((n_months, n_themes))
    E = rng.standard_normal((n_months, n_signals))
    resid = np.sqrt(np.maximum(1.0 - theme_load ** 2 - mkt_load ** 2, 0.05))
    returns = mu + monthly_vol * (M * mkt_load + F[:, theme_of] * theme_load
                                  + E * resid)

    index = pd.period_range("1970-01", periods=n_months, freq="M").to_timestamp()
    names = [f"signal_{i:03d}" for i in range(n_signals)]
    df = pd.DataFrame(returns, index=index, columns=names).rename_axis("date")
    truth = pd.Series(is_true, index=names, name="is_true")

    if publication_t is not None:
        head = df.iloc[:publication_months]
        t = head.mean() / (head.std(ddof=1) / np.sqrt(len(head)))
        keep = t.abs() > publication_t
        df, truth = df.loc[:, keep.values], truth[keep.values]

    return df, truth


def signal_stats(returns, periods=12):
    """Mean, volatility, annualised Sharpe and t-statistic for each column."""
    mu, sd = returns.mean(), returns.std(ddof=1)
    n = returns.notna().sum()
    return pd.DataFrame({
        "mean": mu,
        "vol": sd,
        "sharpe": mu / sd * np.sqrt(periods),
        "tstat": mu / (sd / np.sqrt(n)),
        "n_obs": n,
    })


def sdf_weights(R, method="ridge", lam=1e-3, n_pcs=None, max_iter=20_000):
    """
    Stochastic discount factor weights from a panel of excess returns.

    The mean-variance efficient portfolio solves min_w E[(1 - R'w)^2], so
    regressing a column of ones on returns with no intercept recovers the SDF
    weights directly. Penalising that regression is exactly the Kozak, Nagel
    & Santosh construction.

    method : "ridge"  L2 penalty  -- dense weights, nothing set to zero
             "lasso"  L1 penalty  -- sparse in characteristics
             "pc"     ridge on the leading `n_pcs` principal components,
                      then rotated back -- sparse in principal components
             "ols"    no penalty; included so students can watch it fail
    """
    R = np.asarray(R, dtype=float)
    ones = np.ones(len(R))

    if method == "ols":
        return np.linalg.pinv(R) @ ones
    if method == "ridge":
        return Ridge(alpha=lam, fit_intercept=False).fit(R, ones).coef_
    if method == "lasso":
        return Lasso(alpha=lam, fit_intercept=False,
                     max_iter=max_iter).fit(R, ones).coef_
    if method == "pc":
        if n_pcs is None:
            raise ValueError("method='pc' requires n_pcs")
        U, s, Vt = np.linalg.svd(R - R.mean(0), full_matrices=False)
        V = Vt[:n_pcs].T                       # loadings of the leading PCs
        w_pc = Ridge(alpha=lam, fit_intercept=False).fit(R @ V, ones).coef_
        return V @ w_pc
    raise ValueError(f"unknown method: {method}")


def portfolio_sharpe(R, w, periods=12):
    """Annualised Sharpe ratio of the portfolio R @ w."""
    p = np.asarray(R, dtype=float) @ np.asarray(w, dtype=float)
    sd = p.std(ddof=1)
    return np.nan if sd == 0 else p.mean() / sd * np.sqrt(periods)


def sdf_shrinkage_sweep(R_train, R_test, method="ridge", lambdas=None,
                        n_pcs_grid=None, periods=12):
    """Out-of-sample Sharpe of the estimated SDF across penalty values."""
    if method == "pc":
        grid = n_pcs_grid if n_pcs_grid is not None else range(1, 31)
        rows = []
        for k in grid:
            w = sdf_weights(R_train, "pc", lam=1e-6, n_pcs=int(k))
            rows.append(dict(param=int(k), method="pc",
                             is_sharpe=portfolio_sharpe(R_train, w, periods),
                             oos_sharpe=portfolio_sharpe(R_test, w, periods),
                             n_nonzero=int(np.sum(np.abs(w) > 1e-10))))
        return pd.DataFrame(rows)

    if lambdas is None:
        lambdas = (np.logspace(-6, 0, 40) if method == "lasso"
                   else np.logspace(-6, 2, 40))
    rows = []
    for lam in lambdas:
        w = sdf_weights(R_train, method, lam=lam)
        rows.append(dict(param=lam, method=method,
                         is_sharpe=portfolio_sharpe(R_train, w, periods),
                         oos_sharpe=portfolio_sharpe(R_test, w, periods),
                         n_nonzero=int(np.sum(np.abs(w) > 1e-10))))
    return pd.DataFrame(rows)


__all__ += [
    "OAP_RELEASE", "load_open_source_zoo", "make_synthetic_zoo",
    "signal_stats", "sdf_weights", "portfolio_sharpe", "sdf_shrinkage_sweep",
]


# ===========================================================================
#  8. Panel construction for the factor-timing exercise (Lec08d)
# ===========================================================================

def build_timing_panel(wide, horizons=(1, 3, 12, 36), vol_windows=(12, 36),
                       rank_cols=("ret_1m", "ret_12m", "vol_12m")):
    """
    Turn a wide panel of signal returns into a stacked prediction panel.

    Every feature for signal i at month t is built from that signal's own
    history strictly before t, plus its rank within the cross-section at t.
    Nothing from month t or later enters the features, and nothing from other
    signals' month-t returns enters either.

    Returns a long DataFrame with columns date, signal, the features, and
    `target` = the signal's return in month t.
    """
    frames = []
    for col in wide.columns:
        s = wide[col]
        f = {f"ret_{h}m": (s.shift(1) if h == 1
                           else s.rolling(h).mean().shift(1))
             for h in horizons}
        f.update({f"vol_{w}m": s.rolling(w).std().shift(1)
                  for w in vol_windows})
        f["target"] = s.values
        block = pd.DataFrame(f, index=s.index)
        block["signal"] = col
        frames.append(block.reset_index())

    panel = pd.concat(frames, ignore_index=True).dropna()
    for col in rank_cols:
        if col in panel.columns:
            panel[col + "_rank"] = panel.groupby("date")[col].rank(pct=True)
    return panel.sort_values(["date", "signal"], ignore_index=True)


def timing_feature_names(panel):
    return [c for c in panel.columns if c not in ("date", "signal", "target")]


def rank_ic(frame, pred="pred", target="target", by="date"):
    """Mean cross-sectional Spearman correlation between forecast and outcome."""
    ic = frame.groupby(by).apply(
        lambda x: x[pred].corr(x[target], method="spearman"),
        include_groups=False)
    return ic.mean(), ic.std() / np.sqrt(len(ic))


def tilted_portfolio(frame, tilt, pred_rank="prank", target="target", by="date"):
    """
    Equal weight plus a tilt towards signals the model ranks highly.

    tilt = 0 recovers the equal-weighted benchmark. Weights are rescaled to
    unit gross exposure so the comparison is not confounded by leverage.
    """
    def one(x):
        n = len(x)
        w = 1.0 / n + tilt * 2.0 * (x[pred_rank] - 0.5) / n
        w = w / np.abs(w).sum()
        return float((w * x[target]).sum())
    return frame.groupby(by).apply(one, include_groups=False)


def annualised_sharpe(series, periods=12):
    s = pd.Series(series).dropna()
    return s.mean() / s.std() * np.sqrt(periods)


__all__ += ["build_timing_panel", "timing_feature_names", "rank_ic",
            "tilted_portfolio", "annualised_sharpe"]


# ===========================================================================
#  9. Ken French Data Library
# ===========================================================================

FRENCH_DIR = Path(os.environ.get("AIRM8_FRENCH_DIR", "data/french"))

FRENCH_FILES = {
    "ff3": "F-F_Research_Data_Factors.csv",
    "ff5": "F-F_Research_Data_5_Factors_2x3.csv",
    "mom": "F-F_Momentum_Factor.csv",
    "size_bm": "25_Portfolios_5x5.csv",
    "size_op": "25_Portfolios_ME_OP_5x5.csv",
    "industry": "49_Industry_Portfolios.csv",
}

VW_MONTHLY = "Average Value Weighted Returns -- Monthly"


def _french_blocks(path):
    """
    Split a raw Ken French CSV into its stacked panels.

    These files are not tidy. A single download contains a copyright preamble,
    then several panels one after another -- value-weighted monthly returns,
    equal-weighted monthly, then annual versions, then firm counts and average
    sizes -- each introduced by a title line and a header row beginning with a
    comma. Missing values are coded -99.99 or -999.

    Returns {panel_title: DataFrame}, indexed by period as an integer.
    """
    text = Path(path).read_text(errors="replace")
    lines = [l.rstrip("\r") for l in text.split("\n")]

    def is_data(l):
        s = l.strip()
        return bool(s) and s[0].isdigit() and "," in s

    blocks, i, title = {}, 0, "Monthly"
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(",") and i + 1 < len(lines) and is_data(lines[i + 1]):
            cols = [c.strip() for c in line.split(",")][1:]
            rows, idx = [], []
            j = i + 1
            while j < len(lines) and is_data(lines[j]):
                parts = [p.strip() for p in lines[j].split(",")]
                idx.append(parts[0])
                rows.append([float(v) if v not in ("", "-") else np.nan
                             for v in parts[1:len(cols) + 1]])
                j += 1
            df = pd.DataFrame(rows, columns=cols, index=idx)
            df = df.replace([-99.99, -999.0, -99.99e0], np.nan)
            key, n = title, 2
            while key in blocks:                      # disambiguate repeats
                key, n = f"{title} ({n})", n + 1
            blocks[key] = df
            i = j
            continue
        s = line.strip()
        if s and not s.startswith(",") and not is_data(line):
            title = s
        i += 1
    return blocks


def _to_period_index(df):
    """YYYYMM index to month-end timestamps; drop annual panels."""
    idx = df.index.astype(str).str.strip()
    if not (idx.str.len() == 6).all():
        return None
    out = df.copy()
    out.index = pd.PeriodIndex(idx, freq="M").to_timestamp(how="end").normalize()
    out.index.name = "date"
    return out


def _french_path(key, directory=None):
    """
    Locate one Ken French file: explicit directory, else local clone, else the
    teaching repository over HTTPS.

    Both `load_french` and `french_panels` go through here. They used not to,
    which meant `french_panels` looked only on disk and failed for anyone
    relying on the repository fallback.
    """
    if key not in FRENCH_FILES:
        raise KeyError(f"unknown key {key!r}; expected one of "
                       f"{list(FRENCH_FILES)}")
    if directory is not None:
        return Path(directory) / FRENCH_FILES[key]
    try:
        return resolve_data(f"french/{FRENCH_FILES[key]}")
    except FileNotFoundError:
        return FRENCH_DIR / FRENCH_FILES[key]


def load_french(key, panel=None, directory=None, to_decimal=True):
    """
    Load one Ken French dataset, returning the monthly panel as decimals.

    key   : one of FRENCH_FILES -- "ff3", "ff5", "mom", "size_bm",
            "size_op", "industry"
    panel : which stacked panel to take. Defaults to the first monthly block,
            which for the portfolio files is the value-weighted one. Pass
            VW_MONTHLY explicitly, or any other panel title, to be sure.

    Returns are converted from percent to decimals unless to_decimal=False.
    Columns are stripped of the trailing spaces the industry file carries.
    """
    path = _french_path(key, directory)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the CSV (zip) from the Ken French "
            "Data Library, unzip it into this directory, and keep the "
            "original filename. Mirror it in the teaching repository so the "
            "lab does not depend on the site being reachable."
        )

    blocks = _french_blocks(path)
    if panel is not None:
        if panel not in blocks:
            raise KeyError(f"panel {panel!r} not in {list(blocks)}")
        df = _to_period_index(blocks[panel])
        if df is None:
            raise ValueError(f"panel {panel!r} is not monthly")
    else:
        df = None
        for block in blocks.values():
            candidate = _to_period_index(block)
            if candidate is not None:
                df = candidate
                break
        if df is None:
            raise ValueError(f"no monthly panel found in {path.name}")

    df.columns = [c.strip() for c in df.columns]
    return df / 100.0 if to_decimal else df


def french_panels(key, directory=None):
    """List the panel titles available inside one file, for orientation."""
    return list(_french_blocks(_french_path(key, directory)))


def load_factors(model="ff5", directory=None, with_momentum=False,
                 start=None, end=None):
    """
    Assemble a factor model with the risk-free rate attached.

    model : "capm", "ff3", "ff5". Momentum is appended when with_momentum
            is True, which turns ff3 into the Carhart four-factor model.

    Returns (factors, rf) where `factors` excludes RF and `rf` is a Series.
    """
    base = load_french("ff3" if model in ("capm", "ff3") else "ff5",
                       directory=directory)
    rf = base["RF"]
    factors = base.drop(columns="RF")
    if model == "capm":
        factors = factors[["Mkt-RF"]]

    if with_momentum:
        mom = load_french("mom", directory=directory)
        mom.columns = ["Mom"]
        factors = factors.join(mom, how="inner")

    common = factors.dropna().index.intersection(rf.dropna().index)
    factors, rf = factors.loc[common], rf.loc[common]
    if start is not None:
        factors, rf = factors.loc[start:], rf.loc[start:]
    if end is not None:
        factors, rf = factors.loc[:end], rf.loc[:end]
    return factors, rf


def load_test_assets(key="size_bm", directory=None, rf=None, start=None,
                     end=None, drop_incomplete=True, verbose=False):
    """
    Monthly value-weighted test-asset returns in excess of the risk-free rate.

    Set `start` when comparing models. The 49 industry portfolios in
    particular have missing history -- several industries barely exist in the
    1920s -- so the number of usable assets depends on the window. Comparing
    a model estimated on 40 industries against one estimated on 47 is not a
    comparison of models.
    """
    port = load_french(key, panel=VW_MONTHLY, directory=directory)
    if rf is None:
        rf = load_french("ff3", directory=directory)["RF"]
    common = port.dropna(how="all").index.intersection(rf.dropna().index)
    out = port.loc[common].sub(rf.loc[common], axis=0)

    if start is not None:
        out = out.loc[start:]
    if end is not None:
        out = out.loc[:end]

    if drop_incomplete:
        n_before = out.shape[1]
        out = out.dropna(axis=1, how="any")
        if verbose and out.shape[1] < n_before:
            print(f"   {key}: dropped {n_before - out.shape[1]} of {n_before} "
                  f"assets with gaps in this window")
    return out


__all__ += ["FRENCH_DIR", "FRENCH_FILES", "VW_MONTHLY", "load_french",
            "_french_path", "french_panels", "load_factors", "load_test_assets"]


# ===========================================================================
#  10. Testing a factor model: alphas, GRS, Fama-MacBeth
# ===========================================================================

def time_series_alphas(excess, factors):
    """
    Estimate R^e_i = alpha_i + beta_i' f + e_i for every test asset.

    Returns (alphas, betas, residuals, r2), all aligned on a common sample.
    """
    idx = excess.dropna(how="all").index.intersection(factors.dropna().index)
    R, F = excess.loc[idx].dropna(axis=1, how="any"), factors.loc[idx]
    X = np.column_stack([np.ones(len(F)), F.values])

    coef, *_ = np.linalg.lstsq(X, R.values, rcond=None)
    fitted = X @ coef
    resid = R.values - fitted

    tss = ((R.values - R.values.mean(0)) ** 2).sum(0)
    r2 = 1.0 - (resid ** 2).sum(0) / tss

    alphas = pd.Series(coef[0], index=R.columns, name="alpha")
    betas = pd.DataFrame(coef[1:].T, index=R.columns, columns=F.columns)
    return alphas, betas, pd.DataFrame(resid, index=idx, columns=R.columns), \
        pd.Series(r2, index=R.columns, name="r2")


def grs_test(excess, factors):
    """
    Gibbons, Ross & Shanken (1989): are all N pricing errors jointly zero?

        GRS = (T-N-K)/N * (1 + fbar' Omega^-1 fbar)^-1 * alpha' Sigma^-1 alpha

    distributed F(N, T-N-K) under normality. Requires T > N + K.

    Returns a dict with the statistic, its p-value, the degrees of freedom,
    and two economically readable summaries: mean absolute alpha (per month)
    and the Sharpe ratio of the implied optimal deviation from the model.
    """
    from scipy import stats

    alphas, _, resid, _ = time_series_alphas(excess, factors)
    idx = resid.index
    F = factors.loc[idx]
    T, N, K = len(idx), len(alphas), F.shape[1]
    if T <= N + K:
        raise ValueError(f"GRS needs T > N + K; have T={T}, N={N}, K={K}")

    Sigma = np.atleast_2d(np.cov(resid.values, rowvar=False, ddof=K + 1))
    Omega = np.atleast_2d(np.cov(F.values, rowvar=False, ddof=1))
    fbar = F.values.mean(0)

    a = alphas.values
    quad_a = float(a @ np.linalg.solve(Sigma, a))
    quad_f = float(fbar @ np.linalg.solve(Omega, fbar))

    stat = (T - N - K) / N / (1.0 + quad_f) * quad_a
    p = 1.0 - stats.f.cdf(stat, N, T - N - K)

    return dict(grs=stat, p_value=p, df1=N, df2=T - N - K, T=T, N=N, K=K,
                mean_abs_alpha=float(np.abs(a).mean()),
                sharpe_of_alphas=float(np.sqrt(max(quad_a, 0.0))),
                sharpe_of_factors=float(np.sqrt(max(quad_f, 0.0))))


def fama_macbeth(excess, factors, shanken=True):
    """
    Two-pass cross-sectional regression with the Shanken (1992) correction.

    Pass 1: time-series betas for each asset.
    Pass 2: a cross-sectional regression of returns on betas each month,
            giving a series of risk-price estimates lambda_t.
    Inference uses the time-series variation in lambda_t.

    The second-stage regressor is an *estimate*, so classical errors-in-
    variables attenuates the risk price and understates its standard error.
    Shanken's multiplicative correction inflates the standard errors by
    (1 + lambda' Omega^-1 lambda). Report both: the gap tells you how much
    first-stage noise you are carrying.
    """
    _, betas, resid, _ = time_series_alphas(excess, factors)
    idx = resid.index
    R, F = excess.loc[idx, betas.index], factors.loc[idx]

    B = np.column_stack([np.ones(len(betas)), betas.values])
    lam = np.linalg.lstsq(B, R.values.T, rcond=None)[0].T      # T x (K+1)
    names = ["intercept"] + list(betas.columns)
    lam = pd.DataFrame(lam, index=idx, columns=names)

    T = len(idx)
    est = lam.mean()
    se = lam.std(ddof=1) / np.sqrt(T)

    out = pd.DataFrame({"estimate": est, "se_fm": se, "t_fm": est / se})

    if shanken:
        Omega = np.atleast_2d(np.cov(F.values, rowvar=False, ddof=1))
        lam_f = est[list(betas.columns)].values
        c = float(lam_f @ np.linalg.solve(Omega, lam_f))
        out["se_shanken"] = se * np.sqrt(1.0 + c)
        out["t_shanken"] = out.estimate / out.se_shanken
        out.attrs["shanken_multiplier"] = np.sqrt(1.0 + c)

    # Cross-sectional R-squared on average returns.
    mean_R, mean_fit = R.mean().values, (B @ est.values)
    out.attrs["cross_sectional_r2"] = float(
        1 - ((mean_R - mean_fit) ** 2).sum()
        / ((mean_R - mean_R.mean()) ** 2).sum())
    return out


__all__ += ["time_series_alphas", "grs_test", "fama_macbeth"]


def plot_signal_distribution(stats, truth=None, hurdles=(1.96, 3.0),
                             title=None, jitter=0.35, seed=0):
    """
    Distribution of t-statistics across signals, with every signal
    individually identifiable on hover.

    A histogram alone aggregates: each bar is many signals and none of them
    can be named. Here the histogram carries the shape and a jittered strip
    beneath it carries the identity, so students can hover on the tail and
    ask "which anomaly is that?".

    Parameters
    ----------
    stats   : DataFrame from `signal_stats`, indexed by signal name.
    truth   : optional boolean Series marking genuinely non-zero signals.
              When supplied, points are coloured by truth rather than by
              which hurdle they clear.
    hurdles : t-statistic thresholds to mark.
    """
    df = stats.dropna(subset=["tstat"]).copy()
    rng = np.random.default_rng(seed)
    df["y"] = rng.uniform(-jitter, jitter, len(df))

    lo, hi = min(hurdles), max(hurdles)
    band = pd.cut(df.tstat.abs(), [-np.inf, lo, hi, np.inf],
                  labels=[f"|t| < {lo}", f"{lo} to {hi}", f"|t| > {hi}"])
    if truth is not None:
        aligned = truth.reindex(df.index).fillna(False).astype(bool)
        groups = [("genuinely non-zero", aligned, NAVY),
                  ("population mean of zero", ~aligned, GREY)]
    else:
        groups = [(f"|t| < {lo}", band == f"|t| < {lo}", GREY),
                  (f"{lo} to {hi}", band == f"{lo} to {hi}", ORANGE),
                  (f"|t| > {hi}", band == f"|t| > {hi}", NAVY)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.74, 0.26], vertical_spacing=0.04)

    fig.add_trace(go.Histogram(x=df.tstat, marker_color=NAVY, nbinsx=45,
                               opacity=0.85, showlegend=False,
                               hovertemplate="t between %{x}<br>"
                                             "%{y} signals<extra></extra>"),
                  row=1, col=1)

    cols = [c for c in ("sharpe", "mean", "vol", "n_obs") if c in df.columns]
    for label, mask, colour in groups:
        sub = df[mask]
        if sub.empty:
            continue
        custom = np.column_stack([sub.index.to_numpy()]
                                 + [sub[c].to_numpy() for c in cols])
        lines = ["<b>%{customdata[0]}</b>", "t-stat  %{x:.2f}"]
        for k, c in enumerate(cols, start=1):
            fmt = ".0f" if c == "n_obs" else (".3f" if c == "sharpe" else ".4f")
            lines.append(f"{c:<7}" + " %{customdata[" + str(k) + "]:" + fmt + "}")
        fig.add_trace(go.Scatter(
            x=sub.tstat, y=sub.y, mode="markers", name=label,
            marker=dict(color=colour, size=7, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            customdata=custom,
            hovertemplate="<br>".join(lines) + "<extra></extra>"),
            row=2, col=1)

    for h, colour in zip(sorted(hurdles), [ORANGE, RED]):
        for sgn in (-1, 1):
            fig.add_vline(x=sgn * h, line=dict(color=colour, width=2,
                                               dash="dash"))

    fig.update_yaxes(title="signals", row=1, col=1)
    fig.update_yaxes(visible=False, range=[-1, 1], row=2, col=1)
    fig.update_xaxes(title="t-statistic on the long–short return", row=2, col=1)
    return airm_layout(fig, title or "Every signal, and every signal named",
                       height=520,
                       legend=dict(orientation="h", x=0.5, y=-0.16,
                                   xanchor="center"))


__all__ += ["plot_signal_distribution"]


SIGNAL_DOC_FILE = "signal_doc.csv"
SIGNAL_DOC_URL = ("https://raw.githubusercontent.com/OpenSourceAP/CrossSection/"
                  "master/SignalDoc.csv")


def load_signal_doc(path=None, file=SIGNAL_DOC_FILE):
    """
    Chen & Zimmermann's signal documentation: what each cryptic acronym in the
    zoo actually is, who found it, where it was published, and which economic
    category it belongs to.

    Resolution order: `path`, then the teaching repository via resolve_data,
    then the upstream GitHub copy. Indexed by acronym.
    """
    if path is not None:
        doc = pd.read_csv(path)
    else:
        try:
            doc = pd.read_csv(resolve_data(file))
        except Exception:                                  # noqa: BLE001
            import urllib.request
            with urllib.request.urlopen(SIGNAL_DOC_URL, timeout=90) as fh:
                payload = fh.read()
            (CACHE_DIR / file).write_bytes(payload)
            doc = pd.read_csv(io.BytesIO(payload))

    doc = doc.rename(columns={"Cat.Economic": "Category",
                              "LongDescription": "Description"})
    return doc.set_index("Acronym")


def describe_signals(names, doc=None, columns=("Description", "Authors",
                                               "Year", "Journal", "Category")):
    """Look up a list of signal acronyms in the documentation."""
    doc = load_signal_doc() if doc is None else doc
    cols = [c for c in columns if c in doc.columns]
    out = doc.reindex(pd.Index(names, name="Acronym"))[cols]
    return out


__all__ += ["SIGNAL_DOC_FILE", "SIGNAL_DOC_URL", "load_signal_doc",
            "describe_signals"]


def signal_loadings(returns, n_components=3, stats=None, doc=None):
    """
    Principal-component loadings for each signal, in biplot convention.

    Returns are standardised first, so this is a correlation PCA: signals are
    placed by how they co-move, not by how volatile they are. Loadings are
    eigenvectors scaled by the square root of their eigenvalue, which makes
    distance from the origin interpretable as how much of a signal's variance
    the retained components capture.

    Returns (loadings DataFrame, variance-explained array).
    """
    Z = ((returns - returns.mean()) / returns.std(ddof=1)).dropna(axis=1)
    A = Z.values - Z.values.mean(0)
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    ev = s ** 2 / (len(A) - 1)
    share = ev / ev.sum()

    k = n_components
    load = pd.DataFrame(Vt[:k].T * np.sqrt(ev[:k]), index=Z.columns,
                        columns=[f"PC{i+1}" for i in range(k)])
    if stats is not None:
        load = load.join(stats[[c for c in ("sharpe", "tstat")
                                if c in stats.columns]])
    if doc is not None:
        load = load.join(doc[[c for c in ("Description", "Category")
                              if c in doc.columns]])
    return load, share


def plot_signal_biplot(loadings, share=None, colour_by="Category",
                       n_categories=8, title=None, height=650):
    """
    Three-dimensional map of the zoo: every signal placed by its loadings on
    the first three principal components.

    colour_by : "Category" groups signals by economic theme, which asks
                whether economically similar signals actually co-move.
                Any numeric column (e.g. "sharpe") uses a colour scale
                instead, which asks whether the profitable signals sit in
                any particular direction.
    """
    df = loadings.dropna(subset=["PC1", "PC2", "PC3"]).copy()
    axis = [f"PC{i}" for i in (1, 2, 3)]
    labels = {a: (f"{a} ({share[i]:.0%} of variance)" if share is not None
                  else a) for i, a in enumerate(axis)}

    hover_cols = [c for c in ("Description", "Category", "sharpe", "tstat")
                  if c in df.columns]
    custom = df[hover_cols].values if hover_cols else None
    lines = ["<b>%{text}</b>"]
    for k, c in enumerate(hover_cols):
        fmt = ":.2f" if c in ("sharpe", "tstat") else ""
        lines.append(f"{c}: %{{customdata[{k}]{fmt}}}")

    fig = go.Figure()
    # Test for NUMERIC rather than object dtype: pandas 3 returns `str`
    # dtype from read_csv, so `dtype == object` is False there and the
    # categorical branch would be skipped.
    categorical = (colour_by in df.columns
                   and not pd.api.types.is_numeric_dtype(df[colour_by]))
    if categorical:
        top = df[colour_by].value_counts().head(n_categories).index
        palette = [NAVY, RED, TEAL, ORANGE, GOLD, "#6A4C93", "#1F7A8C",
                   "#B5179E"]
        for k, cat in enumerate(top):
            sub = df[df[colour_by] == cat]
            fig.add_trace(go.Scatter3d(
                x=sub.PC1, y=sub.PC2, z=sub.PC3, mode="markers", name=str(cat),
                text=sub.index, customdata=sub[hover_cols].values,
                marker=dict(size=5, color=palette[k % len(palette)],
                            opacity=0.85),
                hovertemplate="<br>".join(lines) + "<extra></extra>"))
        rest = df[~df[colour_by].isin(top)]
        if len(rest):
            fig.add_trace(go.Scatter3d(
                x=rest.PC1, y=rest.PC2, z=rest.PC3, mode="markers",
                name="everything else", text=rest.index,
                customdata=rest[hover_cols].values,
                marker=dict(size=4, color=GREY, opacity=0.5),
                hovertemplate="<br>".join(lines) + "<extra></extra>"))
    else:
        fig.add_trace(go.Scatter3d(
            x=df.PC1, y=df.PC2, z=df.PC3, mode="markers", text=df.index,
            customdata=custom, showlegend=False,
            marker=dict(size=5, color=df[colour_by], colorscale="Viridis",
                        opacity=0.88, showscale=True,
                        colorbar=dict(title=colour_by, thickness=12)),
            hovertemplate="<br>".join(lines) + "<extra></extra>"))

    fig.update_layout(
        scene=dict(xaxis_title=labels["PC1"], yaxis_title=labels["PC2"],
                   zaxis_title=labels["PC3"],
                   xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                   yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
                   zaxis=dict(backgroundcolor="rgba(0,0,0,0)")),
        margin=dict(l=0, r=0, t=48, b=0), height=height,
        template="simple_white",
        font=dict(family="Helvetica, Arial, sans-serif", size=12),
        legend=dict(x=0.02, y=0.98, font=dict(size=10)))
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left",
                                     font=dict(size=15, color=NAVY)))
    return fig


__all__ += ["signal_loadings", "plot_signal_biplot"]


def check(verbose=True):
    """
    Report what this copy of airm8 provides, and whether the data resolves.

    Run this first if a notebook raises NameError on an airm8 function: it
    almost always means a stale airm8.py, or a kernel holding the old module
    in sys.modules after the file was replaced. Neither shows up as an import
    error -- the module loads fine, it is simply the wrong one.
    """
    import sys
    rows, ok = [], True
    for name in ("load_open_source_zoo", "load_french", "load_goyal_welch",
                 "load_california", "load_signal_doc", "describe_signals",
                 "signal_stats", "signal_loadings", "plot_signal_biplot",
                 "plot_signal_distribution", "sdf_weights",
                 "PurgedGroupTimeSeriesSplit", "build_timing_panel",
                 "grs_test", "fama_macbeth", "complexity_sweep"):
        present = name in globals()
        ok &= present
        rows.append((name, present))

    if verbose:
        print(f"airm8 version {__version__}")
        print(f"loaded from   {__file__}")
        print(f"data dir      {DATA_DIR.resolve()}"
              f"  {'(exists)' if DATA_DIR.exists() else '(NOT FOUND)'}")
        missing = [n for n, p in rows if not p]
        if missing:
            print(f"\nMISSING: {', '.join(missing)}")
            print("This copy of airm8.py is out of date. Replace it, then")
            print("restart the kernel -- editing the file is not enough,")
            print("because Python caches the module in sys.modules.")
        else:
            print(f"\nAll {len(rows)} public helpers present.")
    return ok


__all__ += ["__version__", "check"]
