"""
airm10.py -- helper module for AIRM Lecture 10 (AI in Risk Forecasting and Stress Testing).

Plumbing for the Lecture 10 notebooks and the Stress Lab playground: data resolution
(local clone -> cache -> teaching repository over HTTPS), loaders for the mirrored French
and Goyal-Welch files, the hybrid model, tail-correlation and reverse-stress functions,
and the Plotly figures. The notebooks explain what each piece is for.

Model settings match anchor_lec10.py exactly, so every number agrees with the slides.
"""
from __future__ import annotations

import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

REPO_RAW = "https://raw.githubusercontent.com/VitaliAlexeev/AI_Investments_2026/main/"
C = dict(navy="#123F69", blue="#1F6FB2", orange="#E07B00", red="#B3261E",
         teal="#1B9AAA", grey="#8C8C8C", light="#E2EBF4")
FONT = dict(family="Arial, sans-serif", size=12)


# ============================================================================ data
def resolve_data(rel_path: str) -> str:
    """Local repository clone -> local cache -> teaching repository over HTTPS."""
    if os.path.exists(rel_path):
        return rel_path
    cache = os.path.join("_cache", rel_path)
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        url = REPO_RAW + rel_path.replace(" ", "%20")
        with urllib.request.urlopen(url, timeout=60) as r, open(cache, "wb") as f:
            f.write(r.read())
    return cache


def _french_block(path, start_marker, end_marker):
    lines = open(path, encoding="latin-1").read().splitlines()
    s = next(i for i, l in enumerate(lines) if start_marker in l)
    e = next(i for i, l in enumerate(lines) if i > s and end_marker in l)
    df = pd.read_csv(io.StringIO("\n".join(lines[s + 1:e])), index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.PeriodIndex(df.index.astype(str).str.strip(), freq="M")
    return df.replace([-99.99, -999], np.nan)


def load_industries() -> pd.DataFrame:
    """French 49 industry portfolios: value-weighted monthly returns, per cent."""
    return _french_block(resolve_data("data/french/49_Industry_Portfolios.csv"),
                         "Average Value Weighted Returns -- Monthly",
                         "Average Equal Weighted Returns -- Monthly")


def load_market() -> pd.Series:
    """Total US market return (Mkt-RF + RF), monthly, per cent."""
    p = resolve_data("data/french/F-F_Research_Data_Factors.csv")
    lines = open(p, encoding="latin-1").read().splitlines()
    s = next(i for i, l in enumerate(lines) if l.startswith(",Mkt-RF"))
    e = next(i for i, l in enumerate(lines) if i > s and l.strip() == "")
    ff = pd.read_csv(io.StringIO("\n".join(lines[s:e])), index_col=0)
    ff.columns = [c.strip() for c in ff.columns]
    ff.index = pd.PeriodIndex(ff.index.astype(str).str.strip(), freq="M")
    return (ff["Mkt-RF"] + ff["RF"]).rename("mkt")


def load_goyal_welch() -> pd.DataFrame:
    g = pd.read_csv(resolve_data("data/goyal_welch_monthly.csv"))
    g.index = pd.PeriodIndex(g["yyyymm"].astype(str), freq="M")
    return g


def market_banks() -> pd.DataFrame:
    """Monthly total market and Banks industry returns, per cent, 1926 onwards."""
    return pd.DataFrame({"mkt": load_market(), "banks": load_industries()["Banks"]}).dropna()


# ======================================================= experiment 1: extrapolation
TRAIN_WINDOWS = {"1990-2007": ("1990-01", "2007-12"), "1950-2007": ("1950-01", "2007-12"),
                 "1926-2007": ("1926-07", "2007-12"), "2010-2019": ("2010-01", "2019-12")}
REPLAYS = {"Oct 1929": "1929-10", "Oct 1987": "1987-10", "Oct 2008": "2008-10", "Mar 2020": "2020-03"}
MODEL_COLOURS = {"Linear regression": C["navy"], "Random forest": C["orange"],
                 "Gradient boosting": "#B86500", "Hybrid": C["teal"]}


class HybridRegressor:
    """Linear backbone plus gradient boosting on the residuals. The residual
    correction is switched off outside the range of the training inputs, so the
    model extrapolates with the backbone only."""

    def __init__(self, **hgb_kwargs):
        self.hgb_kwargs = {**dict(max_iter=300, learning_rate=0.05, min_samples_leaf=10,
                                  random_state=0), **hgb_kwargs}

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.linear_ = LinearRegression().fit(X, y)
        self.resid_ = HistGradientBoostingRegressor(**self.hgb_kwargs).fit(X, y - self.linear_.predict(X))
        self.lo_, self.hi_ = X.min(axis=0), X.max(axis=0)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        inside = np.all((X >= self.lo_) & (X <= self.hi_), axis=1)
        return self.linear_.predict(X) + np.where(inside, self.resid_.predict(X), 0.0)


def fit_trap_models(df: pd.DataFrame, start: str, end: str):
    """The four models of the extrapolation experiment, fitted on one training window."""
    train = df.loc[start:end]
    X, y = train[["mkt"]].values, train["banks"].values
    models = {
        "Linear regression": LinearRegression().fit(X, y),
        "Random forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                               random_state=0).fit(X, y),
        "Gradient boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                           min_samples_leaf=10, random_state=0).fit(X, y),
        "Hybrid": HybridRegressor().fit(X, y),
    }
    return models, train


def trap_table(df, models, shock):
    rows = [("Your shock", shock, np.nan)]
    rows += [(lab, float(df.loc[pd.Period(m, "M"), "mkt"]), float(df.loc[pd.Period(m, "M"), "banks"]))
             for lab, m in REPLAYS.items()]
    out = pd.DataFrame(rows, columns=["Scenario", "Market (%)", "Banks, actual (%)"]).set_index("Scenario")
    for name, mod in models.items():
        out[name] = mod.predict(out[["Market (%)"]].values)
    return out.round(1)


def trap_figure(df, train, models, shock, window_label, show=None):
    show = show or list(models)
    grid = np.linspace(-45, 20, 521).reshape(-1, 1)
    lo = float(train["mkt"].min())
    fig = go.Figure()
    fig.add_vrect(x0=-45, x1=lo, fillcolor="rgba(224,123,0,0.12)", line_width=0,
                  annotation_text="beyond the training data", annotation_position="top left",
                  annotation_font=dict(color=C["orange"], size=11))
    fig.add_trace(go.Scatter(x=train["mkt"], y=train["banks"], mode="markers",
                             marker=dict(size=4, color=C["grey"], opacity=0.5),
                             name=f"Training months, {window_label}"))
    for name in show:
        fig.add_trace(go.Scatter(x=grid.ravel(), y=models[name].predict(grid), mode="lines",
                                 line=dict(color=MODEL_COLOURS[name], width=2.2,
                                           dash="dash" if name == "Gradient boosting" else "solid"),
                                 name=name))
    rep = [(lab, df.loc[pd.Period(m, "M")]) for lab, m in REPLAYS.items()]
    fig.add_trace(go.Scatter(x=[r["mkt"] for _, r in rep], y=[r["banks"] for _, r in rep],
                             mode="markers+text", text=[lab for lab, _ in rep], textposition="bottom right",
                             marker=dict(size=10, color=C["red"]), textfont=dict(color=C["red"], size=11),
                             name="What banks actually did"))
    fig.add_vline(x=shock, line=dict(color=C["navy"], width=1, dash="dot"))
    for name in show:
        fig.add_trace(go.Scatter(x=[shock], y=[float(models[name].predict([[shock]])[0])], mode="markers",
                                 marker=dict(size=11, color=MODEL_COLOURS[name], line=dict(color="white", width=1)),
                                 showlegend=False, hovertemplate=f"{name}: %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(template="plotly_white", font=FONT, height=480, margin=dict(l=60, r=20, t=50, b=50),
                      title=f"Trained on {window_label}; the market falls {shock:.1f}% in a month",
                      xaxis=dict(title="US market return in the month (%)", range=[-45, 20], zeroline=True),
                      yaxis=dict(title="US banks return in the month (%)", range=[-45, 25], zeroline=True),
                      legend=dict(orientation="h", y=-0.22))
    return fig


# =========================================================== experiment 2: joint tails
def industries_since_1970():
    ind = load_industries()
    rets = ind.loc["1970-01":].dropna(axis=1)
    return rets, load_market().reindex(rets.index)


def tail_masks(mkt: pd.Series, tail: float):
    lo, q25, q75, hi = mkt.quantile([tail, 0.25, 0.75, 1 - tail])
    return {"calm": (mkt > q25) & (mkt < q75), "boom": mkt >= hi, "crash": mkt <= lo}


def avg_pairwise_corr(block: pd.DataFrame) -> float:
    c = block.corr().values
    n = c.shape[0]
    return float((c.sum() - n) / (n * (n - 1)))


def fr_adjusted_avg_corr(block, mkt, mask, calm_mask) -> float:
    """Forbes-Rigobon-style rescaling of every pairwise correlation for the larger
    variance of the market in the conditioning subset, relative to calm months."""
    c = block.corr().values
    delta = float(mkt[mask].var() / mkt[calm_mask].var()) - 1.0
    adj = c / np.sqrt(1.0 + delta * (1.0 - c ** 2))
    n = adj.shape[0]
    return float((adj.sum() - np.trace(adj)) / (n * (n - 1)))


def tail_correlations(rets, mkt, tail=0.10) -> dict:
    m = tail_masks(mkt, tail)
    out = {k: avg_pairwise_corr(rets[v]) for k, v in m.items()}
    out.update({f"{k}_adjusted": fr_adjusted_avg_corr(rets[m[k]], mkt, m[k], m["calm"]) for k in ("boom", "crash")})
    out.update({f"n_{k}": int(v.sum()) for k, v in m.items()})
    out.update({f"sd_{k}": float(mkt[v].std()) for k, v in m.items()})
    return out


def calm_order(rets, mkt):
    calm = rets[tail_masks(mkt, 0.10)["calm"]].corr().values
    return leaves_list(linkage(squareform(1 - calm, checks=False), "average"))


def tails_figure(rets, mkt, tail, res, order):
    m = tail_masks(mkt, tail)
    names = [rets.columns[i] for i in order]
    calm = rets[m["calm"]].corr().values[np.ix_(order, order)]
    crash = rets[m["crash"]].corr().values[np.ix_(order, order)]
    fig = make_subplots(rows=1, cols=3, column_widths=[0.38, 0.38, 0.24], horizontal_spacing=0.06,
                        subplot_titles=(f"Calm months (average {res['calm']:.2f})",
                                        f"Worst {tail:.0%} of months (average {res['crash']:.2f})",
                                        "Average correlation"))
    for col, mat in [(1, calm), (2, crash)]:
        fig.add_trace(go.Heatmap(z=mat, x=names, y=names, zmin=-0.2, zmax=1.0, colorscale="Blues",
                                 showscale=(col == 2), colorbar=dict(len=0.8, x=0.735)), row=1, col=col)
        fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_yaxes(showticklabels=False, autorange="reversed", row=1, col=col)
    labels = ["Calm", f"Best {tail:.0%}", f"Worst {tail:.0%}"]
    fig.add_trace(go.Bar(x=labels, y=[res["calm"], res["boom"], res["crash"]],
                         marker_color=[C["grey"], C["blue"], C["red"]], name="Raw",
                         text=[f"{v:.2f}" for v in (res["calm"], res["boom"], res["crash"])],
                         textposition="outside"), row=1, col=3)
    fig.add_trace(go.Bar(x=labels[1:], y=[res["boom_adjusted"], res["crash_adjusted"]],
                         marker=dict(color="rgba(0,0,0,0)", line=dict(color=C["navy"], width=2)),
                         name="Adjusted for the louder market"), row=1, col=3)
    fig.update_yaxes(range=[0, 0.55], row=1, col=3)
    fig.update_layout(template="plotly_white", font=FONT, height=430, barmode="overlay",
                      margin=dict(l=20, r=20, t=60, b=40), legend=dict(orientation="h", y=-0.12, x=0.62))
    return fig


# ====================================================== experiment 3: reverse stress
def quarterly_stock_bond() -> pd.DataFrame:
    """Quarterly compounded S&P 500 and Treasury returns (Goyal-Welch ret, ltr), per cent."""
    g = load_goyal_welch()[["ret", "ltr"]].dropna()
    q = (1 + g).groupby(g.index.asfreq("Q")).prod() - 1
    return (100 * q).rename(columns={"ret": "equities", "ltr": "treasuries"})


def regime(q, start_year, end_year):
    est = q.loc[f"{start_year}Q1":f"{end_year}Q4"]
    return est, est.mean().values, est.cov().values


def most_plausible(mu, S, w, L):
    return mu + (L - w @ mu) * (S @ w) / (w @ S @ w)


def distance(x, mu, S):
    d = np.asarray(x) - mu
    return float(np.sqrt(d @ np.linalg.inv(S) @ d))


def tail_probability(d):
    """P(a bivariate Gaussian draw is at least distance d from the centre) = exp(-d^2/2)."""
    return float(np.exp(-d ** 2 / 2))


def ellipse(mu, S, r, n=240):
    t = np.linspace(0, 2 * np.pi, n)
    L = np.linalg.cholesky(S)
    return (mu.reshape(2, 1) + r * L @ np.vstack([np.cos(t), np.sin(t)])).T


def rst_figure(q, start_year, end_year, equity_weight, loss):
    est, mu, S = regime(q, start_year, end_year)
    w = np.array([equity_weight, 1 - equity_weight])
    xs = most_plausible(mu, S, w, loss)
    d = distance(xs, mu, S)
    e = ellipse(mu, S, d)
    xr = np.array([-45, 35])
    fig = go.Figure()
    if abs(w[1]) > 1e-9:
        yline = (loss - w[0] * xr) / w[1]
        side = -60 if w[1] > 0 else 60
        fig.add_trace(go.Scatter(x=[xr[0], xr[1], xr[1], xr[0]], y=[yline[0], yline[1], side, side],
                                 fill="toself", fillcolor="rgba(179,38,30,0.10)", line=dict(width=0),
                                 hoverinfo="skip", name="Breach region"))
        fig.add_trace(go.Scatter(x=xr, y=yline, mode="lines", line=dict(color="black", dash="dash", width=1),
                                 name=f"Portfolio loses {abs(loss):.0f}%"))
    else:
        fig.add_vrect(x0=-45, x1=loss / w[0], fillcolor="rgba(179,38,30,0.10)", line_width=0)
    fig.add_trace(go.Scatter(x=est["equities"], y=est["treasuries"], mode="markers",
                             marker=dict(size=5, color=C["grey"], opacity=0.6),
                             name=f"Quarters used, {start_year}-{end_year}"))
    fig.add_trace(go.Scatter(x=e[:, 0], y=e[:, 1], mode="lines", line=dict(color=C["navy"], width=2),
                             name="Equally plausible quarters"))
    y22 = q.loc["2022Q1":"2022Q3"]
    fig.add_trace(go.Scatter(x=y22["equities"], y=y22["treasuries"], mode="markers+text",
                             text=[str(p)[-2:] for p in y22.index], textposition="bottom left",
                             marker=dict(size=10, color=C["red"], symbol="star"), name="2022"))
    fig.add_trace(go.Scatter(x=[xs[0]], y=[xs[1]], mode="markers", marker=dict(size=14, color=C["orange"],
                             line=dict(color="white", width=1.5)), name="Most plausible breach"))
    fig.update_layout(template="plotly_white", font=FONT, height=480, margin=dict(l=60, r=20, t=50, b=50),
                      title=(f"Most plausible way to lose {abs(loss):.0f}%: equities {xs[0]:.1f}%, "
                             f"Treasuries {xs[1]:+.1f}% (distance {d:.2f})"),
                      xaxis=dict(title="Equities, quarter (%)", range=[-45, 35]),
                      yaxis=dict(title="Treasuries, quarter (%)", range=[-30, 25]),
                      legend=dict(orientation="h", y=-0.2))
    return fig, xs, d


# ================================================================ playground data
def playground_data() -> dict:
    """Everything the Stress Lab HTML needs, computed with the settings above."""
    df = market_banks()
    grid = np.round(np.arange(-45, 20.0001, 0.25), 2)
    trap = {"grid": grid.tolist(), "windows": {}, "replays": []}
    for lab, m in REPLAYS.items():
        r = df.loc[pd.Period(m, "M")]
        trap["replays"].append({"label": lab, "mkt": round(float(r["mkt"]), 2), "banks": round(float(r["banks"]), 2)})
    rx = np.array([[r["mkt"]] for r in trap["replays"]])
    for name, (a, b) in TRAIN_WINDOWS.items():
        models, train = fit_trap_models(df, a, b)
        trap["windows"][name] = {
            "x": np.round(train["mkt"].values, 2).tolist(), "y": np.round(train["banks"].values, 2).tolist(),
            "min": round(float(train["mkt"].min()), 2), "minMonth": str(train["mkt"].idxmin()),
            "pred": {k: np.round(m.predict(grid.reshape(-1, 1)), 3).tolist() for k, m in models.items()},
            "replayPred": {k: np.round(m.predict(rx), 2).tolist() for k, m in models.items()}}
    rets, mkt = industries_since_1970()
    tails = {"names": list(rets.columns), "order": [int(i) for i in calm_order(rets, mkt)],
             "start": str(rets.index.min()), "end": str(rets.index.max()),
             "mkt": np.round(mkt.values, 3).tolist(), "R": np.round(rets.values, 2).tolist()}
    q = quarterly_stock_bond()
    rst = {"labels": [str(p) for p in q.index], "eq": np.round(q["equities"].values, 4).tolist(),
           "bd": np.round(q["treasuries"].values, 4).tolist()}
    return {"trap": trap, "tails": tails, "rst": rst}


if __name__ == "__main__":
    with open("stresslab_data.json", "w") as f:
        json.dump(playground_data(), f, separators=(",", ":"))
    print("wrote stresslab_data.json")


# ============================================================== labs 10a and 10b
LAB_INDUSTRIES = ["Banks", "Insur", "RlEst", "Oil", "Util", "Hlth", "Softw", "Rtail"]
LAB_ASSETS = LAB_INDUSTRIES + ["Treasuries", "Corp bonds"]
ALPHA = 0.025
GEN_COLOURS = {"Realised": C["red"], "Bootstrap": C["grey"], "Gaussian": C["navy"],
               "Student t": C["teal"], "Mixture": C["orange"]}


def asset_panel(start="1970-01", end="2025-12") -> pd.DataFrame:
    """Monthly returns (%) of eight industry portfolios, long Treasuries and corporate bonds."""
    ind = load_industries()[LAB_INDUSTRIES]
    g = load_goyal_welch()
    bonds = pd.DataFrame({"Treasuries": 100 * g["ltr"], "Corp bonds": 100 * g["corpr"]})
    return ind.join(bonds, how="inner").loc[start:end].dropna()[LAB_ASSETS]


def panel_summary(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"Mean (% a year)": 12 * X.mean(), "Volatility (% a year)": np.sqrt(12) * X.std(),
                         "Worst month (%)": X.min()}).round(1)


def benchmark_portfolios() -> pd.DataFrame:
    P = {"Equal-weight equities": {a: 1 / 8 for a in LAB_INDUSTRIES},
         "60/40": {**{a: 0.6 / 8 for a in LAB_INDUSTRIES}, "Treasuries": 0.4},
         "Financials": {"Banks": 0.5, "Insur": 0.5},
         "Banks minus utilities": {"Banks": 1.0, "Util": -1.0},
         "Credit carry": {"Corp bonds": 1.0, "Treasuries": -1.0},
         "Defensive": {"Util": 0.5, "Treasuries": 0.5}}
    return pd.DataFrame(P).T.reindex(columns=LAB_ASSETS).fillna(0.0)


def var_es(pnl, alpha=ALPHA):
    pnl = np.asarray(pnl, float)
    v = float(np.quantile(pnl, alpha))
    return v, float(pnl[pnl <= v].mean())


def fz0(y, v, e, alpha=ALPHA):
    """FZ0 joint score for (VaR, ES) at level alpha (Lecture 6); lower is better."""
    y = np.asarray(y, float)
    return float(np.mean(-((y <= v) * (v - y)) / (alpha * e) + v / e + np.log(-e) - 1.0))


def fit_student_t(X: pd.DataFrame, dfs=range(3, 41)):
    from scipy.stats import multivariate_t
    mu, S = X.mean().values, X.cov().values
    ll = {v: multivariate_t(mu, S * (v - 2) / v, df=v).logpdf(X.values).sum() for v in dfs}
    df = max(ll, key=ll.get)
    return mu, S * (df - 2) / df, df


def fit_mixture(X: pd.DataFrame, kmax=5, seed=0):
    from sklearn.mixture import GaussianMixture
    fits = {k: GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit(X.values)
            for k in range(1, kmax + 1)}
    bic = pd.Series({k: m.bic(X.values) for k, m in fits.items()}, name="BIC")
    bic.index.name = "Components"
    return fits[int(bic.idxmin())], bic


def scorecard(scenarios: dict, test: pd.DataFrame, portfolios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for port, w in portfolios.iterrows():
        w = w.values
        y = test.values @ w
        v, e = var_es(y)
        rows.append(dict(generator="Realised", portfolio=port, VaR=v, ES=e, FZ0=fz0(y, v, e), worst=float(y.min())))
        for name, Z in scenarios.items():
            p = Z @ w
            v, e = var_es(p)
            rows.append(dict(generator=name, portfolio=port, VaR=v, ES=e, FZ0=fz0(y, v, e), worst=float(p.min())))
    return pd.DataFrame(rows)


def crash_breadth(R, threshold=-5.0, k=6) -> float:
    """Share of months in which at least k of the eight industries lose more than |threshold|%."""
    R = np.asarray(R, float)[:, :len(LAB_INDUSTRIES)]
    return float(np.mean((R < threshold).sum(axis=1) >= k))


def _gcol(name):
    return GEN_COLOURS.get(name.split(" (")[0], C["blue"])


def cdf_figure(scenarios: dict, test: pd.DataFrame, w, title):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("The whole distribution", "The lower tail, log scale"))
    series = {"Realised": test.values @ w, **{k: Z @ w for k, Z in scenarios.items()}}
    for name, p in series.items():
        x = np.sort(p)
        y = np.arange(1, len(x) + 1) / len(x)
        step = max(1, len(x) // 4000)
        for col in (1, 2):
            fig.add_trace(go.Scatter(x=x[::step], y=y[::step], mode="lines", name=name,
                                     line=dict(color=_gcol(name), width=3 if name == "Realised" else 1.8),
                                     showlegend=(col == 1)), row=1, col=col)
    fig.update_yaxes(title_text="Share of months at or below", row=1, col=1)
    fig.update_yaxes(type="log", range=[np.log10(0.0005), np.log10(0.2)], row=1, col=2)
    fig.update_xaxes(title_text="Portfolio return in the month (%)", range=[-25, 15], row=1, col=1)
    fig.update_xaxes(title_text="Portfolio return in the month (%)", range=[-30, 0], row=1, col=2)
    fig.update_layout(template="plotly_white", font=FONT, height=420, title=title,
                      legend=dict(orientation="h", y=-0.25), margin=dict(l=60, r=20, t=80, b=40))
    return fig


def es_figure(card: pd.DataFrame, portfolios: pd.DataFrame, title):
    fig = go.Figure()
    for name in card["generator"].unique():
        sub = card[card["generator"] == name].set_index("portfolio").loc[portfolios.index]
        fig.add_trace(go.Bar(x=list(sub.index), y=sub["ES"], name=name, marker_color=_gcol(name)))
    fig.update_layout(barmode="group", template="plotly_white", font=FONT, height=430, title=title,
                      yaxis_title="Expected shortfall, 97.5% (% a month)",
                      legend=dict(orientation="h", y=-0.25), margin=dict(l=60, r=20, t=60, b=40))
    return fig


def fz0_table(card: pd.DataFrame) -> pd.DataFrame:
    T = card.pivot(index="generator", columns="portfolio", values="FZ0")
    T = T.loc[[g for g in T.index if g != "Realised"] + ["Realised"]]
    T["Average rank (1 = best)"] = T.drop(index="Realised").rank(axis=0).mean(axis=1)
    return T.round(3)


# ---------------------------------------------------------------- 10b helpers
def most_plausible_linear(mu, S, w, L):
    return mu + (L - w @ mu) * (S @ w) / (w @ S @ w)


def worst_within(mu, S, w, k):
    """Worst linear P&L among scenarios no further than k from the centre (Breuer et al.)."""
    return float(w @ mu - k * np.sqrt(w @ S @ w))


def mahalanobis_rows(Z, mu, S):
    D = np.atleast_2d(Z) - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", D, np.linalg.inv(S), D))


def bs_put_premium(moneyness=0.92, vol=0.16, months=1):
    """Black-Scholes price of a put struck at `moneyness` x spot, zero rates, in % of notional."""
    from scipy.stats import norm
    s = vol * np.sqrt(months / 12)
    d1 = (np.log(1 / moneyness) + 0.5 * s ** 2) / s
    return float(100 * (moneyness * norm.cdf(-(d1 - s)) - norm.cdf(-d1)))


def overlay_pnl(Z, w, strike=-8.0, notional=0.5, premium=None):
    """60/40-style P&L plus a short one-month put on the equal-weight equity basket."""
    Z = np.atleast_2d(Z)
    prem = bs_put_premium(1 + strike / 100) if premium is None else premium
    eq = Z[:, :len(LAB_INDUSTRIES)].mean(axis=1)
    return Z @ w + notional * (prem - np.maximum(strike - eq, 0.0))


def reverse_stress_opt(mu, S, pnl_fn, L, starts):
    """Most plausible scenario with pnl_fn(x) <= L, by constrained optimisation from several starts."""
    from scipy.optimize import minimize
    Si = np.linalg.inv(S)
    best = None
    for x0 in starts:
        r = minimize(lambda x: 0.5 * (x - mu) @ Si @ (x - mu), x0, jac=lambda x: Si @ (x - mu),
                     method="SLSQP", constraints=[{"type": "ineq", "fun": lambda x: L - pnl_fn(x)[0]}],
                     options={"maxiter": 500, "ftol": 1e-10})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    return best.x


def scenario_bars(scenarios: dict, title, colours=None):
    colours = colours or {}
    fig = go.Figure()
    palette = [C["navy"], C["teal"], C["orange"], C["red"], C["grey"], C["blue"]]
    for i, (name, x) in enumerate(scenarios.items()):
        fig.add_trace(go.Bar(x=LAB_ASSETS, y=np.asarray(x), name=name,
                             marker_color=colours.get(name, palette[i % len(palette)])))
    fig.update_layout(barmode="group", template="plotly_white", font=FONT, height=420, title=title,
                      yaxis_title="Return in the scenario month (%)",
                      legend=dict(orientation="h", y=-0.25), margin=dict(l=60, r=20, t=60, b=40))
    return fig


def payoff_figure(mu, w, pnl_fn, strike=-8.0):
    shocks = np.linspace(-30, 10, 161)
    Z = np.tile(mu, (len(shocks), 1))
    Z[:, :len(LAB_INDUSTRIES)] = shocks[:, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=shocks, y=Z @ w, name="60/40 alone", line=dict(color=C["navy"], width=2)))
    fig.add_trace(go.Scatter(x=shocks, y=pnl_fn(Z), name="60/40 plus the short put",
                             line=dict(color=C["orange"], width=2.5)))
    fig.add_vline(x=strike, line=dict(color=C["grey"], dash="dot"),
                  annotation_text="strike", annotation_position="top")
    fig.update_layout(template="plotly_white", font=FONT, height=380,
                      title="Portfolio return when every industry moves by the same amount",
                      xaxis_title="Equity return in the month (%)", yaxis_title="Portfolio return (%)",
                      legend=dict(orientation="h", y=-0.25), margin=dict(l=60, r=20, t=60, b=40))
    return fig


def breach_cloud(Z, pnl, L, best_points: dict, n_show=6000, seed=0):
    rng = np.random.default_rng(seed)
    eq, tsy = Z[:, :len(LAB_INDUSTRIES)].mean(axis=1), Z[:, LAB_ASSETS.index("Treasuries")]
    breach = pnl <= L
    calm_idx = rng.choice(np.where(~breach)[0], size=min(n_show, int((~breach).sum())), replace=False)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=eq[calm_idx], y=tsy[calm_idx], mode="markers", name="Generated months",
                               marker=dict(size=3, color=C["grey"], opacity=0.35)))
    fig.add_trace(go.Scattergl(x=eq[breach], y=tsy[breach], mode="markers", name=f"Breaches (loss of {abs(L):.0f}% or more)",
                               marker=dict(size=4, color=C["red"], opacity=0.6)))
    for name, x in best_points.items():
        fig.add_trace(go.Scatter(x=[np.mean(x[:len(LAB_INDUSTRIES)])], y=[x[LAB_ASSETS.index("Treasuries")]],
                                 mode="markers+text", text=[name], textposition="top left",
                                 marker=dict(size=13, color=C["orange"], line=dict(color="white", width=1.5)),
                                 showlegend=False))
    fig.update_layout(template="plotly_white", font=FONT, height=460,
                      xaxis=dict(title="Equal-weight equities in the month (%)", range=[-40, 25]),
                      yaxis=dict(title="Treasuries in the month (%)", range=[-20, 20]),
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=60, r=20, t=40, b=40))
    return fig


# ======================================================================= lab 10c
FED_FILES = {"history": "data/fed/2026_Final_Historic_Domestic.csv",
             "severely adverse": "data/fed/2026_Final_Supervisory_Severely_Adverse_Domestic.csv",
             "baseline": "data/fed/2026_Final_Supervisory_Baseline_Domestic.csv"}
FED_SHORT = {"Real GDP growth": "gdp", "Unemployment rate": "unemp", "3-month Treasury rate": "tb3",
             "10-year Treasury yield": "y10", "BBB corporate yield": "bbb",
             "Dow Jones Total Stock Market Index (Level)": "stock", "House Price Index (Level)": "hpi",
             "Commercial Real Estate Price Index (Level)": "cre", "Market Volatility Index (Level)": "vix"}
SATELLITE_FEATURES = ["stock_ret", "d_spread", "d_unemp", "hpi_g", "cre_g", "d_y10"]
FEATURE_LABELS = {"stock_ret": "Stock market return (%)", "d_spread": "Change in BBB spread (pp)",
                  "d_unemp": "Change in unemployment rate (pp)", "hpi_g": "House price growth (%)",
                  "cre_g": "Commercial property price growth (%)", "d_y10": "Change in 10-year yield (pp)"}
MONOTONE = [1, -1, -1, 1, 1, 0]
SAT_COLOURS = {"Actual": C["red"], "Linear": C["navy"], "Boosting": C["orange"],
               "Monotone boosting": "#A85C00", "Hybrid": C["teal"]}


def load_fed(name: str) -> pd.DataFrame:
    """Fed 2026 stress-test file ('history', 'severely adverse' or 'baseline'), quarterly levels."""
    d = pd.read_csv(resolve_data(FED_FILES[name]))
    d.index = pd.PeriodIndex(d["Date"].str.replace(" ", ""), freq="Q")
    return d.rename(columns=FED_SHORT)[list(FED_SHORT.values())].astype(float)


def fed_features(levels: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"stock_ret": 100 * levels["stock"].pct_change(),
                         "d_spread": (levels["bbb"] - levels["y10"]).diff(),
                         "d_unemp": levels["unemp"].diff(),
                         "hpi_g": 100 * levels["hpi"].pct_change(),
                         "cre_g": 100 * levels["cre"].pct_change(),
                         "d_y10": levels["y10"].diff()})


def fed_scenario_features(name: str) -> pd.DataFrame:
    """Scenario quarters as changes from the last historical quarter onwards."""
    levels = pd.concat([load_fed("history").iloc[[-1]], load_fed(name)])
    return fed_features(levels).iloc[1:]


def quarterly_returns(monthly: pd.Series) -> pd.Series:
    return 100 * ((1 + monthly / 100).groupby(monthly.index.asfreq("Q")).prod() - 1)


def satellite_data(target="Banks") -> pd.DataFrame:
    X = fed_features(load_fed("history"))[SATELLITE_FEATURES].dropna()
    y = quarterly_returns(load_industries()[target]).rename("target")
    return X.join(y, how="inner")


def cumulative(r):
    return 100 * np.cumprod(1 + np.asarray(r, float) / 100)


def beyond_range(X_hist: pd.DataFrame, X_scen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, x in X_scen.iterrows():
        for f in X_hist.columns:
            lo, hi = X_hist[f].min(), X_hist[f].max()
            if x[f] < lo or x[f] > hi:
                extreme = X_hist[f].idxmin() if x[f] < lo else X_hist[f].idxmax()
                rows.append({"Quarter": str(q), "Variable": FEATURE_LABELS[f], "Scenario": round(x[f], 1),
                             "Historical extreme": round(X_hist[f].loc[extreme], 1), "When": str(extreme)})
    return pd.DataFrame(rows).set_index(["Quarter", "Variable"])


def scenario_panel_figure(hist: pd.DataFrame, scenarios: dict, start="2000Q1"):
    h = hist.loc[start:].copy()
    h["spread"] = h["bbb"] - h["y10"]
    panels = [("stock", "Stock market index"), ("unemp", "Unemployment rate (%)"), ("hpi", "House prices (index)"),
              ("cre", "Commercial property prices (index)"), ("spread", "BBB spread over 10-year (pp)"),
              ("tb3", "3-month Treasury rate (%)")]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[t for _, t in panels], vertical_spacing=0.16)
    colours = {"Severely adverse": C["red"], "Baseline": C["grey"]}
    for i, (col, _) in enumerate(panels):
        r, c = divmod(i, 3)
        x = [str(p) for p in h.index]
        fig.add_trace(go.Scatter(x=x, y=h[col], line=dict(color=C["navy"], width=1.6), name="History",
                                 showlegend=(i == 0)), row=r + 1, col=c + 1)
        for name, s in scenarios.items():
            s = pd.concat([hist.iloc[[-1]], s]).copy()
            s["spread"] = s["bbb"] - s["y10"]
            fig.add_trace(go.Scatter(x=[str(p) for p in s.index], y=s[col], name=name, showlegend=(i == 0),
                                     line=dict(color=colours.get(name, C["teal"]), width=2.2)), row=r + 1, col=c + 1)
        fig.update_xaxes(nticks=5, row=r + 1, col=c + 1)
    fig.update_layout(template="plotly_white", font=FONT, height=560, margin=dict(l=50, r=20, t=60, b=40),
                      title="The Fed's 2026 scenarios against history", legend=dict(orientation="h", y=-0.12))
    return fig


def path_figure(preds: dict, index, actual=None, title=""):
    x = ["start"] + [str(p) for p in index]
    fig = go.Figure()
    series = ({"Actual": actual} if actual is not None else {}) | preds
    for name, r in series.items():
        fig.add_trace(go.Scatter(x=x, y=np.r_[100, cumulative(r)], name=name, mode="lines+markers",
                                 line=dict(color=SAT_COLOURS.get(name, C["blue"]), width=3.2 if name == "Actual" else 2,
                                           dash="dash" if name == "Monotone boosting" else "solid")))
    fig.update_layout(template="plotly_white", font=FONT, height=400, title=title,
                      yaxis_title="US bank stocks (start = 100)", legend=dict(orientation="h", y=-0.25),
                      margin=dict(l=60, r=20, t=60, b=40))
    return fig


# ======================================================================= lab 10d
LLM_CACHE = "data/lab10d_cached_llm_scenarios.json"
LLM_URL = "https://api.anthropic.com/v1/messages"


def call_claude(prompt: str, model: str, max_tokens=4000, api_key=None) -> str:
    """One Claude Messages API call over plain HTTPS (no SDK needed); returns the reply's text."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No API key: set ANTHROPIC_API_KEY in your environment, or keep USE_LIVE = False.")
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(LLM_URL, data=body, method="POST",
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        reply = json.loads(r.read())
    return "".join(b.get("text", "") for b in reply.get("content", []) if b.get("type") == "text")


def parse_json_reply(text):
    """Accept a JSON list, or text that contains one (replies sometimes add code fences or prose)."""
    if isinstance(text, (list, dict)):
        return text
    s = text.strip()
    start, end = s.find("["), s.rfind("]")
    if start < 0 or end < 0:
        raise ValueError("No JSON list found in the reply.")
    return json.loads(s[start:end + 1])


def scenario_frame(items, label):
    """Validate a list of scenarios against the schema; return (returns, narratives)."""
    problems, rows, meta = [], [], []
    for i, it in enumerate(items):
        name = str(it.get("name", f"Scenario {i + 1}"))
        r = it.get("returns", {})
        missing = [a for a in LAB_ASSETS if a not in r]
        if missing:
            problems.append(f"{name}: missing {missing}")
            continue
        try:
            vals = [float(r[a]) for a in LAB_ASSETS]
        except (TypeError, ValueError):
            problems.append(f"{name}: a return is not a number")
            continue
        problems += [f"{name}: {a} = {v} is outside -95..95" for a, v in zip(LAB_ASSETS, vals) if not -95 <= v <= 95]
        rows.append(vals)
        meta.append({"Prompt": label, "Scenario": name, "Narrative": it.get("narrative", "")})
    if problems:
        print("Schema problems:", *problems, sep="\n  ")
    idx = pd.MultiIndex.from_tuples([(m["Prompt"], m["Scenario"]) for m in meta], names=["Prompt", "Scenario"])
    return (pd.DataFrame(rows, index=idx, columns=LAB_ASSETS),
            pd.DataFrame(meta).set_index(["Prompt", "Scenario"]))


def load_cached_llm() -> dict:
    with open(resolve_data(LLM_CACHE), encoding="utf-8") as f:
        return json.load(f)


def conditional_z(R: pd.DataFrame, mu, S) -> pd.DataFrame:
    """z-score of each return given the other nine under N(mu, S): (Q d)_j / sqrt(Q_jj), Q = S^-1."""
    Q = np.linalg.inv(S)
    Z = ((R.values - mu) @ Q) / np.sqrt(np.diag(Q))
    return pd.DataFrame(Z, index=R.index, columns=R.columns)


def conditional_mean(R: pd.DataFrame, mu, S) -> pd.DataFrame:
    """What the other nine returns imply for each asset: x_j - (Q d)_j / Q_jj."""
    Q = np.linalg.inv(S)
    return pd.DataFrame(R.values - ((R.values - mu) @ Q) / np.diag(Q), index=R.index, columns=R.columns)


def distance_figure(dist: pd.Series, hist_max, hist_p99, title):
    fig = go.Figure()
    palette = [C["teal"], C["red"], C["orange"]]
    for i, (label, s) in enumerate(dist.groupby(level=0, sort=False)):
        fig.add_trace(go.Scatter(x=s.values, y=[label] * len(s), mode="markers", name=label,
                                 marker=dict(size=12, color=palette[i % 3], line=dict(color="white", width=1)),
                                 text=[n for _, n in s.index], hovertemplate="%{text}: %{x:.1f}<extra></extra>"))
    fig.add_vline(x=hist_p99, line=dict(color=C["grey"], dash="dot"), annotation_text="99% of months since 1970",
                  annotation_position="top left")
    fig.add_vline(x=hist_max, line=dict(color=C["navy"], dash="dash"), annotation_text="most extreme month",
                  annotation_position="top right")
    fig.update_layout(template="plotly_white", font=FONT, height=330, title=title, showlegend=False,
                      xaxis_title="Distance from the average month (Mahalanobis)", margin=dict(l=160, r=20, t=60, b=40))
    return fig


def z_heatmap(Z: pd.DataFrame, title):
    fig = go.Figure(go.Heatmap(z=Z.values, x=list(Z.columns), y=[f"{s}" for _, s in Z.index],
                               zmin=-6, zmax=6, zmid=0, colorscale="RdBu", text=np.round(Z.values, 1),
                               texttemplate="%{text}", colorbar=dict(title="z")))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", font=FONT, height=520, title=title, margin=dict(l=230, r=20, t=60, b=40))
    return fig
