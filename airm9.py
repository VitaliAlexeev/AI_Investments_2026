"""
airm9.py -- shared helpers for AIRM Lecture 9:
Ethical Considerations in AI-powered Finance.

Provides:
  * Data loaders with fallback chains (local copy -> subject repository -> library
    -> public mirror), so labs never depend on a single third party
  * California maps drawn from the data, with the state outline and city labels
  * Plotly renderers for the SHAP plots (waterfall, beeswarm, dependence, global bar),
    so the subject stays Plotly-only rather than mixing in matplotlib
  * Partial dependence and ICE in Plotly
  * Group fairness metrics and threshold search
  * Rashomon-set construction and the decision-flip fan

British spelling in prose; American spellings appear only where they are library
API keywords.

Vitali Alexeev, UTS Business School.
"""

from __future__ import annotations

import os
import warnings
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    "NAVY", "BLUE", "ORANGE", "RED", "GREY", "BAND", "PALETTE",
    "load_hmda", "load_california", "HMDA_FEATURES", "HMDA_LABELS",
    "plot_waterfall", "plot_beeswarm", "plot_global_bar", "plot_dependence",
    "plot_pdp_ice", "group_metrics", "threshold_for_fpr",
    "plot_group_metrics", "rashomon_set", "plot_rashomon_fan",
    "plot_explanation_spread", "layout",
    # California housing (warm-up)
    "CA_FEATURES", "CA_TARGET", "CA_LABELS", "CA_CITIES", "load_california_outline",
    "nearest_city", "california_axes", "add_california_context", "california_map", "california_contour",
    "california_tilemap", "plot_histogram_grid", "plot_correlation_heatmap",
    "regression_table", "plot_permutation_importance",
]

# --------------------------------------------------------------------------- #
# Palette -- matches ColorSchemeBlue in the slide deck
# --------------------------------------------------------------------------- #
NAVY = "#123F69"
BLUE = "#2E75B6"
ORANGE = "#E8820C"
RED = "#C00000"
GREY = "#8C8C8C"
BAND = "#E2EBF4"
PALETTE = [NAVY, ORANGE, BLUE, RED, GREY]

_DIVERGING = [[0.0, BLUE], [0.5, "#D9D9D9"], [1.0, ORANGE]]


def layout(fig: go.Figure, title: str = "", height: int = 460, **kwargs) -> go.Figure:
    """Apply the house style to a figure. Returns the figure for chaining."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=NAVY)) if title else None,
        template="plotly_white",
        height=height,
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=12),
        margin=dict(l=70, r=30, t=60 if title else 30, b=55),
        **kwargs,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE", zerolinecolor="#CCCCCC")
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", zerolinecolor="#CCCCCC")
    return fig


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
_HMDA_MIRROR = ("https://raw.githubusercontent.com/vincentarelbundock/"
                "Rdatasets/master/csv/AER/HMDA.csv")
_CAL_MIRROR = ("https://raw.githubusercontent.com/ageron/handson-ml2/"
               "master/datasets/housing/housing.csv")

HMDA_FEATURES = ["pirat", "hirat", "lvrat", "chist", "mhist", "phist",
                 "unemp", "selfemp", "insurance", "condomin", "single", "hschool"]

HMDA_LABELS = {
    "pirat": "Payment / income",
    "hirat": "Housing expense / income",
    "lvrat": "Loan / value",
    "chist": "Consumer credit history",
    "mhist": "Mortgage credit history",
    "phist": "Public bad credit record",
    "unemp": "Industry unemployment rate",
    "selfemp": "Self-employed",
    "insurance": "Mortgage insurance denied",
    "condomin": "Condominium",
    "single": "Single",
    "hschool": "High-school diploma",
    "afam": "Black applicant (protected)",
}


def _cache_path(cache_dir: str, name: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, name)


def load_hmda(use_live: bool = True, cache_dir: str = "data",
              verbose: bool = True) -> pd.DataFrame:
    """
    Boston HMDA mortgage applications (Munnell et al. 1996), as distributed in the
    R package AER. 2,380 rows.

    Returns a tidy DataFrame with integer-coded binaries: `deny` (outcome),
    `afam` (protected attribute) and the twelve features in HMDA_FEATURES.

    Fallback chain: local cache -> live mirror. Set use_live=False to force cache.
    """
    path = _cache_path(cache_dir, "hmda.csv")

    raw = None
    if os.path.exists(path):
        raw = pd.read_csv(path)
        if verbose:
            print(f"HMDA: loaded from cache ({path})")
    elif use_live:
        try:
            raw = pd.read_csv(_HMDA_MIRROR)
            raw.to_csv(path, index=False)
            if verbose:
                print(f"HMDA: downloaded and cached to {path}")
        except Exception as exc:                                    # noqa: BLE001
            raise RuntimeError(
                "Could not reach the HMDA mirror and no local cache exists. "
                f"Download {_HMDA_MIRROR} manually and save it to {path}.\n"
                f"Original error: {exc}") from exc
    else:
        raise FileNotFoundError(
            f"use_live=False but no cache at {path}. Run once with use_live=True.")

    raw = raw.drop(columns=[c for c in ("rownames", "Unnamed: 0") if c in raw.columns])
    yn = lambda s: (s.astype(str).str.strip().str.lower() == "yes").astype(int)

    df = pd.DataFrame({
        "deny": yn(raw["deny"]),
        "pirat": raw["pirat"].astype(float),
        "hirat": raw["hirat"].astype(float),
        "lvrat": raw["lvrat"].astype(float),
        "chist": raw["chist"].astype(float),
        "mhist": raw["mhist"].astype(float),
        "phist": yn(raw["phist"]),
        "unemp": raw["unemp"].astype(float),
        "selfemp": yn(raw["selfemp"]),
        "insurance": yn(raw["insurance"]),
        "condomin": yn(raw["condomin"]),
        "single": yn(raw["single"]),
        "hschool": yn(raw["hschool"]),
        "afam": yn(raw["afam"]),
    }).dropna().reset_index(drop=True)

    if verbose:
        print(f"HMDA: {len(df):,} applications | denial rate {df.deny.mean():.3f} "
              f"| Black applicants {int(df.afam.sum())}")
    return df


def _fetch_text(url: str, timeout: int = 20) -> str:
    """Download a small text resource (CSV or JSON) with a short timeout."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "airm9-teaching"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


# --------------------------------------------------------------------------- #
# California housing
# --------------------------------------------------------------------------- #
_REPO_RAW = "https://raw.githubusercontent.com/VitaliAlexeev/AI_Investments_2026/main"
_CAL_FILE = "california_housing.csv"
_OUTLINE_FILE = "california_outline.geojson"
_STATES_GEOJSON = ("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
                   "master/data/geojson/us-states.json")

CA_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
               "Population", "AveOccup", "Latitude", "Longitude"]
CA_TARGET = "MedHouseVal"

CA_LABELS = {
    "MedInc": "Median income ($10,000s)",
    "HouseAge": "Median house age (years)",
    "AveRooms": "Average rooms per household",
    "AveBedrms": "Average bedrooms per household",
    "Population": "Block-group population",
    "AveOccup": "Average occupants per household",
    "Latitude": "Latitude (\u00b0N)",
    "Longitude": "Longitude (\u00b0E; negative = west)",
    "MedHouseVal": "Median house value ($100,000s)",
}

# A dozen reference points, spread across the state, for labelling maps.
CA_CITIES = {
    "San Francisco": (37.77, -122.42), "San Jose": (37.34, -121.89),
    "Sacramento": (38.58, -121.49), "Stockton": (37.96, -121.29),
    "Fresno": (36.74, -119.79), "Bakersfield": (35.37, -119.02),
    "Santa Barbara": (34.42, -119.70), "Los Angeles": (34.05, -118.24),
    "San Diego": (32.72, -117.16), "Palm Springs": (33.83, -116.55),
    "Redding": (40.59, -122.39), "Eureka": (40.80, -124.16),
}

CALIFORNIA_SOURCE = None     # set by load_california(): which route supplied the data
_MEAN_LAT_FOR_ASPECT = 37.0  # 1 degree of longitude = cos(37 deg) x 1 degree of latitude


def load_california(n_points: int | None = None, use_live: bool = True,
                    cache_dir: str = "data", random_state: int = 0,
                    verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    California housing (1990 US Census, one row per census block group).
    Returns (X, y): the eight features and the median house value in $100,000s.

    Resolution chain, first success wins:
      1. data/california_housing.csv  -- present if you cloned the subject repository,
                                         or cached by an earlier run
      2. the subject repository on GitHub
      3. sklearn.datasets.fetch_california_housing
      4. shap.datasets.california
      5. the handson-ml2 mirror, with scikit-learn's eight features re-derived.
         This copy has 20,433 rows rather than 20,640 -- 207 block groups lack a
         bedroom count -- so printed figures will differ slightly from everyone
         else's. The notebook says so when it happens.
    """
    global CALIFORNIA_SOURCE
    import io
    path = _cache_path(cache_dir, _CAL_FILE)
    frame, source = None, None

    if os.path.exists(path):
        frame, source = pd.read_csv(path), "local copy (data/)"
    elif use_live:
        try:                                                        # 2. repository
            frame = pd.read_csv(io.StringIO(_fetch_text(f"{_REPO_RAW}/data/{_CAL_FILE}")))
            source = "subject repository"
        except Exception:                                           # noqa: BLE001
            pass
        if frame is None:                                           # 3. scikit-learn
            try:
                from sklearn.datasets import fetch_california_housing
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    frame = fetch_california_housing(as_frame=True).frame
                source = "scikit-learn"
            except Exception:                                       # noqa: BLE001
                pass
        if frame is None:                                           # 4. shap
            try:
                import shap
                Xs, ys = shap.datasets.california()
                frame = Xs.assign(MedHouseVal=ys)
                source = "shap.datasets"
            except Exception:                                       # noqa: BLE001
                pass
        if frame is None:                                           # 5. mirror
            try:
                r = pd.read_csv(io.StringIO(_fetch_text(_CAL_MIRROR))).dropna(
                    subset=["total_bedrooms"])
                frame = pd.DataFrame({
                    "MedInc": r.median_income,
                    "HouseAge": r.housing_median_age,
                    "AveRooms": r.total_rooms / r.households,
                    "AveBedrms": r.total_bedrooms / r.households,
                    "Population": r.population,
                    "AveOccup": r.population / r.households,
                    "Latitude": r.latitude,
                    "Longitude": r.longitude,
                    "MedHouseVal": r.median_house_value / 100_000.0,
                }).reset_index(drop=True)
                source = "handson-ml2 mirror (20,433 rows)"
            except Exception as exc:                                # noqa: BLE001
                raise RuntimeError(
                    "Every route to the California housing data failed. Copy "
                    f"{_CAL_FILE} from the subject repository into {cache_dir}/ "
                    f"and run again.\nLast error: {exc}") from exc
        frame.to_csv(path, index=False)
    else:
        raise FileNotFoundError(
            f"USE_LIVE is False and there is no {path}. Run once with USE_LIVE = True.")

    if "target" in frame.columns and CA_TARGET not in frame.columns:   # older caches
        frame = frame.rename(columns={"target": CA_TARGET})
    frame = frame[CA_FEATURES + [CA_TARGET]].astype(float)
    CALIFORNIA_SOURCE = source

    if n_points is not None and n_points < len(frame):
        frame = frame.sample(n_points, random_state=random_state).reset_index(drop=True)

    if verbose:
        print(f"California housing: {len(frame):,} block groups, 8 features  "
              f"[source: {source}]")
        if source and "mirror" in source:
            print("  note: this copy has 207 fewer rows than scikit-learn's, so your "
                  "numbers will differ\n  slightly from the ones quoted in the text. "
                  "The conclusions do not.")
    return frame[CA_FEATURES], frame[CA_TARGET].rename(CA_TARGET)


def load_california_outline(use_live: bool = True, cache_dir: str = "data",
                            verbose: bool = False):
    """
    The state boundary as a list of (longitudes, latitudes) arrays, one per ring.
    Returns None if no copy can be found -- maps still draw, just without the outline.
    """
    import json
    path = _cache_path(cache_dir, _OUTLINE_FILE)
    geo = None
    if os.path.exists(path):
        with open(path) as fh:
            geo = json.load(fh)
    elif use_live:
        for url in (f"{_REPO_RAW}/data/{_OUTLINE_FILE}", _STATES_GEOJSON):
            try:
                g = json.loads(_fetch_text(url))
                feats = [f for f in g["features"]
                         if f.get("properties", {}).get("name") in ("California", None)]
                if feats:
                    geo = {"type": "FeatureCollection", "features": feats[:1]}
                    with open(path, "w") as fh:
                        json.dump(geo, fh)
                    break
            except Exception:                                       # noqa: BLE001
                continue
    if geo is None:
        if verbose:
            print("State outline unavailable -- maps will be drawn without it.")
        return None

    rings = []
    geom = geo["features"][0]["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        for ring in poly:
            arr = np.asarray(ring, dtype=float)
            rings.append((arr[:, 0], arr[:, 1]))
    return rings


def nearest_city(lat: float, lon: float) -> tuple[str, float]:
    """Closest of CA_CITIES to a point, with the great-circle distance in km."""
    best, best_km = None, np.inf
    for name, (clat, clon) in CA_CITIES.items():
        p1, p2 = np.radians([lat, clat])
        dlat, dlon = np.radians(clat - lat), np.radians(clon - lon)
        h = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
        km = 2 * 6371.0 * np.arcsin(np.sqrt(h))
        if km < best_km:
            best, best_km = name, km
    return best, float(best_km)


def _geo_axes(fig: go.Figure, row=None, col=None) -> None:
    """Longitude/latitude axes with the aspect corrected for latitude."""
    ratio = 1.0 / np.cos(np.radians(_MEAN_LAT_FOR_ASPECT))
    kw = {} if row is None else dict(row=row, col=col)
    fig.update_xaxes(title_text="Longitude", range=[-124.6, -113.9],
                     showgrid=True, gridcolor="#EEEEEE", zeroline=False, **kw)
    if row is None:
        fig.update_yaxes(title_text="Latitude", range=[32.3, 42.2], scaleanchor="x",
                         scaleratio=ratio, showgrid=True, gridcolor="#EEEEEE",
                         zeroline=False)
    else:
        idx = (row - 1) * 2 + col
        anchor = "x" if idx == 1 else f"x{idx}"
        fig.update_yaxes(title_text="Latitude", range=[32.3, 42.2], scaleanchor=anchor,
                         scaleratio=ratio, showgrid=True, gridcolor="#EEEEEE",
                         zeroline=False, **kw)


def california_axes(fig: go.Figure, row=None, col=None) -> go.Figure:
    """Public wrapper: longitude/latitude axes with the latitude-corrected aspect."""
    _geo_axes(fig, row, col)
    return fig


def add_california_context(fig: go.Figure, outline=None, cities: bool = True,
                           row=None, col=None, city_color: str = "#1A1A1A") -> go.Figure:
    """Overlay the state outline and labelled cities on any longitude/latitude figure."""
    kw = {} if row is None else dict(row=row, col=col)
    if outline:
        for k, (lons, lats) in enumerate(outline):
            fig.add_trace(go.Scatter(x=lons, y=lats, mode="lines",
                                     line=dict(color="#3A3A3A", width=1.3),
                                     hoverinfo="skip", showlegend=False,
                                     name="state outline" if k == 0 else None), **kw)
    if cities:
        names = list(CA_CITIES)
        lats = [CA_CITIES[n][0] for n in names]
        lons = [CA_CITIES[n][1] for n in names]
        fig.add_trace(go.Scatter(
            x=lons, y=lats, mode="markers+text", text=names,
            textposition="middle right", textfont=dict(size=10, color=city_color),
            marker=dict(size=7, color="white", line=dict(color=city_color, width=1.6),
                        symbol="circle"),
            hovertemplate="%{text}<br>%{y:.2f}\u00b0N, %{x:.2f}\u00b0<extra></extra>",
            showlegend=False), **kw)
    return fig


def california_map(df: pd.DataFrame, color: str, size: str | None = None,
                   title: str = "", colorscale="Viridis", cmid: float | None = None,
                   color_label: str | None = None, outline=None, cities: bool = True,
                   height: int = 680, marker_size: float = 4.0,
                   opacity: float = 0.75, showscale: bool = True) -> go.Figure:
    """
    Block groups drawn at their coordinates. With ~20,000 points the data trace
    the shape of the state on their own; the outline and cities are context.
    """
    vals = df[color].to_numpy(dtype=float)
    if size is not None:
        s = df[size].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(s, [2, 98])
        sizes = 2.0 + 9.0 * np.clip((s - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    else:
        sizes = marker_size

    marker = dict(size=sizes, color=vals, colorscale=colorscale, opacity=opacity,
                  showscale=showscale, line=dict(width=0),
                  colorbar=dict(title=color_label or color, thickness=14, len=0.75))
    if cmid is not None:
        marker["cmid"] = cmid

    fig = go.Figure(go.Scattergl(
        x=df["Longitude"], y=df["Latitude"], mode="markers", marker=marker,
        customdata=vals, showlegend=False,
        hovertemplate=("%{y:.2f}\u00b0N, %{x:.2f}\u00b0<br>" + (color_label or color) +
                       " %{customdata:.3f}<extra></extra>")))
    add_california_context(fig, outline, cities)
    _geo_axes(fig)
    return layout(fig, title, height=height)


def california_contour(lons: np.ndarray, lats: np.ndarray, Z: np.ndarray,
                       title: str = "", colorscale="RdBu_r", zmid: float | None = None,
                       colorbar_title: str = "", outline=None, cities: bool = True,
                       districts: pd.DataFrame | None = None,
                       height: int = 680, ncontours: int = 18) -> go.Figure:
    """A surface over longitude x latitude (Z has shape [len(lats), len(lons)])."""
    kw = dict(zmid=zmid) if zmid is not None else {}
    fig = go.Figure(go.Contour(
        x=lons, y=lats, z=Z, colorscale=colorscale, ncontours=ncontours,
        contours=dict(showlines=True, coloring="fill"), line=dict(width=0.6),
        opacity=0.85, colorbar=dict(title=colorbar_title, thickness=14, len=0.75),
        hovertemplate="%{y:.2f}\u00b0N, %{x:.2f}\u00b0<br>%{z:.3f}<extra></extra>", **kw))
    if districts is not None:
        fig.add_trace(go.Scattergl(
            x=districts["Longitude"], y=districts["Latitude"], mode="markers",
            marker=dict(size=2, color="#222222", opacity=0.25),
            hoverinfo="skip", showlegend=False))
    add_california_context(fig, outline, cities)
    _geo_axes(fig)
    return layout(fig, title, height=height)


def california_tilemap(df: pd.DataFrame, color: str, n: int = 5000, seed: int = 0,
                       title: str = "") -> go.Figure:
    """
    Optional street-map version (OpenStreetMap tiles; needs an internet connection
    for the background, but no API key). Samples `n` block groups for speed.
    """
    import plotly.express as px
    d = df.sample(min(n, len(df)), random_state=seed)
    kw = dict(lat="Latitude", lon="Longitude", color=color, zoom=4.6, height=640,
              center=dict(lat=37.2, lon=-119.5), opacity=0.7,
              color_continuous_scale="Viridis")
    try:                                     # Plotly >= 5.24 (MapLibre, no token)
        fig = px.scatter_map(d, map_style="open-street-map", **kw)
    except (AttributeError, TypeError, ValueError):   # older Plotly
        fig = px.scatter_mapbox(d, mapbox_style="open-street-map", **kw)
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# --------------------------------------------------------------------------- #
# Exploratory helpers
# --------------------------------------------------------------------------- #
def plot_histogram_grid(df: pd.DataFrame, caps: dict | None = None, ncols: int = 3,
                        title: str = "", nbins: int = 60) -> go.Figure:
    """
    One histogram per column. `caps` maps a column to a value to mark with a red
    dashed line (a top-code or bottom-code), e.g. {"HouseAge": 52}.
    """
    cols = list(df.columns)
    nrows = int(np.ceil(len(cols) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=cols,
                        horizontal_spacing=0.07, vertical_spacing=0.11)
    for k, c in enumerate(cols):
        r, cc = k // ncols + 1, k % ncols + 1
        v = df[c].to_numpy(dtype=float)
        hi = np.nanpercentile(v, 99.5) if c in ("AveRooms", "AveBedrms", "AveOccup",
                                                "Population") else np.nanmax(v)
        fig.add_trace(go.Histogram(x=v[v <= hi], nbinsx=nbins, marker_color=NAVY,
                                   opacity=0.8, showlegend=False, name=c,
                                   hovertemplate="%{x}<br>count %{y}<extra></extra>"),
                      row=r, col=cc)
        if caps and c in caps:
            for val in np.atleast_1d(caps[c]):
                fig.add_vline(x=float(val), line=dict(color=RED, width=2, dash="dash"),
                              row=r, col=cc)
    fig.update_layout(bargap=0.02)
    return layout(fig, title, height=300 * nrows)


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Annotated Pearson correlation matrix."""
    C = df.corr().to_numpy()
    names = list(df.columns)
    fig = go.Figure(go.Heatmap(
        z=C, x=names, y=names, zmin=-1, zmax=1, colorscale="RdBu_r",
        text=np.round(C, 2), texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title="r", thickness=14),
        hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return layout(fig, title, height=560)


def regression_table(results) -> pd.DataFrame:
    """A tidy coefficient table from a fitted statsmodels OLS results object."""
    stars = lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 \
        else "." if p < 0.1 else ""
    tab = pd.DataFrame({
        "coefficient": results.params, "std error": results.bse,
        "t-stat": results.tvalues, "p-value": results.pvalues})
    tab["sig."] = [stars(p) for p in tab["p-value"]]
    return tab


def plot_permutation_importance(result, names, title: str = "",
                                x_label: str = "drop in R\u00b2 when shuffled") -> go.Figure:
    """Box plot of sklearn.inspection.permutation_importance output, largest on top."""
    order = np.argsort(result.importances_mean)
    fig = go.Figure()
    for j in order:
        fig.add_trace(go.Box(x=result.importances[j], name=names[j], orientation="h",
                             marker_color=NAVY, line=dict(color=NAVY), boxpoints="all",
                             jitter=0.3, pointpos=0, showlegend=False,
                             hovertemplate=f"{names[j]}<br>%{{x:.4f}}<extra></extra>"))
    return layout(fig, title, height=110 + 42 * len(order), xaxis_title=x_label)


# --------------------------------------------------------------------------- #
# SHAP renderers in Plotly
# --------------------------------------------------------------------------- #
def _as_matrix(shap_values) -> np.ndarray:
    """Accept a shap.Explanation, a list, or a plain array."""
    v = getattr(shap_values, "values", shap_values)
    v = np.asarray(v)
    if v.ndim == 3:            # (n, p, n_classes) -> take the positive class
        v = v[:, :, -1]
    return v


def plot_waterfall(shap_row, base_value: float, feature_names: Sequence[str],
                   feature_values: Sequence[float] | None = None,
                   max_display: int = 10, title: str = "",
                   value_label: str = "model output") -> go.Figure:
    """
    Local explanation for one observation, as a Plotly waterfall.

    shap_row : 1-D array of SHAP values for a single observation
    base_value : the expected model output over the background data
    """
    phi = np.asarray(_as_matrix(shap_row)).reshape(-1)
    names = list(feature_names)
    vals = list(feature_values) if feature_values is not None else [None] * len(names)

    order = np.argsort(-np.abs(phi))
    keep, rest = order[:max_display], order[max_display:]

    labels, deltas = [], []
    for i in keep[::-1]:
        lab = names[i] if vals[i] is None else f"{names[i]} = {vals[i]:.3g}"
        labels.append(lab)
        deltas.append(float(phi[i]))
    if len(rest):
        labels.insert(0, f"{len(rest)} other features")
        deltas.insert(0, float(phi[rest].sum()))

    labels = ["Base value"] + labels + ["Prediction"]
    measures = ["absolute"] + ["relative"] * (len(deltas)) + ["total"]
    values = [float(base_value)] + deltas + [0.0]

    fig = go.Figure(go.Waterfall(
        orientation="h", measure=measures, y=labels, x=values,
        connector=dict(line=dict(color="#BBBBBB", width=1)),
        increasing=dict(marker=dict(color=ORANGE)),
        decreasing=dict(marker=dict(color=BLUE)),
        totals=dict(marker=dict(color=NAVY)),
        text=[f"{v:+.3f}" if m == "relative" else f"{v:.3f}"
              for v, m in zip(values, measures)],
        textposition="outside", textfont=dict(size=10),
        hovertemplate="%{y}<br>contribution %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(waterfallgap=0.35)
    return layout(fig, title, height=90 + 34 * len(labels),
                  xaxis_title=value_label, showlegend=False)


def plot_beeswarm(shap_values, X: pd.DataFrame, max_display: int = 12,
                  title: str = "", jitter: float = 0.28,
                  labels: dict | None = None) -> go.Figure:
    """
    Global view: one point per observation per feature, positioned by SHAP value
    and coloured by the (normalised) feature value.
    """
    phi = _as_matrix(shap_values)
    names = list(X.columns)
    imp = np.abs(phi).mean(axis=0)
    order = np.argsort(imp)[-max_display:]          # ascending, plotted bottom-up

    rng = np.random.default_rng(0)
    fig = go.Figure()
    for row, j in enumerate(order):
        x = phi[:, j]
        raw = X.iloc[:, j].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(raw, [5, 95])
        col = np.clip((raw - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
        fig.add_trace(go.Scattergl(
            x=x, y=row + rng.uniform(-jitter, jitter, len(x)),
            mode="markers",
            marker=dict(size=4.5, color=col, colorscale=_DIVERGING, opacity=0.7,
                        showscale=bool(row == len(order) - 1),
                        colorbar=dict(title="Feature<br>value", tickvals=[0, 1],
                                      ticktext=["low", "high"], len=0.55,
                                      thickness=12)),
            name=names[j], showlegend=False,
            customdata=raw,
            hovertemplate=(f"{names[j]}<br>value %{{customdata:.3g}}"
                           "<br>SHAP %{x:+.4f}<extra></extra>"),
        ))
    fig.add_vline(x=0, line=dict(color="#999999", width=1))
    tick = [labels.get(names[j], names[j]) if labels else names[j] for j in order]
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=tick, showgrid=False)
    return layout(fig, title, height=110 + 30 * len(order),
                  xaxis_title="SHAP value (impact on model output)")


def plot_global_bar(shap_values, X: pd.DataFrame, max_display: int = 12,
                    title: str = "", labels: dict | None = None) -> go.Figure:
    """Mean absolute SHAP value per feature. The most over-interpreted plot in XAI."""
    phi = _as_matrix(shap_values)
    imp = np.abs(phi).mean(axis=0)
    names = list(X.columns)
    order = np.argsort(imp)[-max_display:]
    tick = [labels.get(names[j], names[j]) if labels else names[j] for j in order]

    fig = go.Figure(go.Bar(
        x=imp[order], y=tick, orientation="h", marker=dict(color=NAVY),
        text=[f"{v:.3f}" for v in imp[order]], textposition="outside",
        textfont=dict(size=10),
        hovertemplate="%{y}<br>mean |SHAP| %{x:.4f}<extra></extra>"))
    return layout(fig, title, height=110 + 30 * len(order),
                  xaxis_title="mean |SHAP value|", showlegend=False)


def plot_dependence(shap_values, X: pd.DataFrame, feature: str,
                    colour_by: str | None = None, title: str = "") -> go.Figure:
    """SHAP value for one feature against that feature's value."""
    phi = _as_matrix(shap_values)
    j = list(X.columns).index(feature)
    x = X[feature].to_numpy(dtype=float)

    marker = dict(size=5, color=NAVY, opacity=0.55)
    hover = f"{feature} %{{x:.3g}}<br>SHAP %{{y:+.4f}}<extra></extra>"
    custom = None
    if colour_by is not None:
        c = X[colour_by].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(c, [5, 95])
        marker = dict(size=5, opacity=0.7,
                      color=np.clip((c - lo) / (hi - lo if hi > lo else 1), 0, 1),
                      colorscale=_DIVERGING, showscale=True,
                      colorbar=dict(title=colour_by, tickvals=[0, 1],
                                    ticktext=["low", "high"], thickness=12, len=0.6))
        custom = c
        hover = (f"{feature} %{{x:.3g}}<br>SHAP %{{y:+.4f}}"
                 f"<br>{colour_by} %{{customdata:.3g}}<extra></extra>")

    fig = go.Figure(go.Scattergl(x=x, y=phi[:, j], mode="markers", marker=marker,
                                 customdata=custom, hovertemplate=hover))
    fig.add_hline(y=0, line=dict(color="#999999", width=1))
    return layout(fig, title or f"SHAP dependence: {feature}",
                  xaxis_title=feature, yaxis_title=f"SHAP value for {feature}")


# --------------------------------------------------------------------------- #
# Partial dependence and ICE
# --------------------------------------------------------------------------- #
def plot_pdp_ice(predict: Callable[[pd.DataFrame], np.ndarray], X: pd.DataFrame,
                 feature: str, ice: bool = True, n_ice: int = 60,
                 grid: int = 40, title: str = "",
                 y_label: str = "E[ f(x) | feature ]",
                 random_state: int = 0) -> go.Figure:
    """
    Partial dependence, optionally with Individual Conditional Expectation curves.

    `predict` takes a DataFrame and returns a 1-D array. For a classifier pass
    something like `lambda d: model.predict_proba(d)[:, 1]`.
    """
    vals = X[feature].to_numpy(dtype=float)
    lo, hi = np.nanpercentile(vals, [2, 98])
    gridpts = np.linspace(lo, hi, grid)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), min(n_ice, len(X)), replace=False)
    sub = X.iloc[idx].copy()

    curves = np.empty((len(sub), grid))
    for k, v in enumerate(gridpts):
        tmp = sub.copy()
        tmp[feature] = v
        curves[:, k] = predict(tmp)
    pdp = curves.mean(axis=0)

    fig = go.Figure()
    if ice:
        for i in range(len(sub)):
            fig.add_trace(go.Scattergl(
                x=gridpts, y=curves[i], mode="lines", showlegend=False,
                line=dict(color=BLUE, width=1), opacity=0.22, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=gridpts, y=pdp, mode="lines", name="Partial dependence",
                             line=dict(color=NAVY, width=3.5),
                             hovertemplate=f"{feature} %{{x:.3g}}<br>"
                                           "mean prediction %{y:.4f}<extra></extra>"))
    fig.add_vline(x=float(np.mean(vals)), line=dict(color=GREY, width=1, dash="dot"),
                  annotation_text="mean", annotation_position="top")
    fig.add_hline(y=float(np.mean(predict(X))), line=dict(color=GREY, width=1, dash="dot"))
    return layout(fig, title or f"Partial dependence{' with ICE' if ice else ''}: {feature}",
                  xaxis_title=feature, yaxis_title=y_label)


# --------------------------------------------------------------------------- #
# Fairness
# --------------------------------------------------------------------------- #
def group_metrics(y_true, scores, group, thresholds=0.5,
                  group_names: dict | None = None) -> pd.DataFrame:
    """
    Confusion-based rates per group.

    thresholds : a single float applied to everyone, or a {group_value: threshold}
                 mapping for group-specific thresholds.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    group = np.asarray(group)

    rows = []
    for gv in np.unique(group):
        t = thresholds[gv] if isinstance(thresholds, dict) else float(thresholds)
        m = group == gv
        yy, pp = y_true[m], scores[m]
        pred = (pp > t).astype(int)
        tp = int(((pred == 1) & (yy == 1)).sum())
        fp = int(((pred == 1) & (yy == 0)).sum())
        tn = int(((pred == 0) & (yy == 0)).sum())
        fn = int(((pred == 0) & (yy == 1)).sum())
        rows.append({
            "group": group_names.get(gv, gv) if group_names else gv,
            "n": int(m.sum()),
            "threshold": t,
            "base_rate": float(yy.mean()),
            "selection_rate": float(pred.mean()),
            "FPR": fp / max(fp + tn, 1),
            "FNR": fn / max(fn + tp, 1),
            "TPR": tp / max(tp + fn, 1),
            "PPV": tp / max(tp + fp, 1),
            "mean_score": float(pp.mean()),
        })
    return pd.DataFrame(rows).set_index("group")


def threshold_for_fpr(scores, y_true, target_fpr: float) -> float:
    """Smallest threshold whose false positive rate is closest to `target_fpr`."""
    from sklearn.metrics import roc_curve
    fpr, _, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(scores, float))
    return float(thr[int(np.argmin(np.abs(fpr - target_fpr)))])


def plot_group_metrics(tables: dict[str, pd.DataFrame],
                       metrics: Sequence[str] = ("FPR", "FNR", "PPV"),
                       title: str = "") -> go.Figure:
    """Side-by-side grouped bars for one or more threshold regimes."""
    panels = list(tables)
    fig = make_subplots(rows=1, cols=len(panels), shared_yaxes=True,
                        subplot_titles=panels, horizontal_spacing=0.06)
    for c, name in enumerate(panels, start=1):
        tab = tables[name]
        for k, grp in enumerate(tab.index):
            fig.add_trace(go.Bar(
                x=list(metrics), y=[tab.loc[grp, m] for m in metrics],
                name=str(grp), marker=dict(color=PALETTE[k % len(PALETTE)]),
                legendgroup=str(grp), showlegend=(c == 1),
                text=[f"{tab.loc[grp, m]:.3f}" for m in metrics],
                textposition="outside", textfont=dict(size=10),
                hovertemplate="%{x} %{y:.4f}<extra>" + str(grp) + "</extra>"),
                row=1, col=c)
    fig.update_yaxes(range=[0, 1.08], row=1, col=1, title_text="rate")
    return layout(fig, title, height=430, barmode="group",
                  legend=dict(orientation="h", y=1.14, x=0))


# --------------------------------------------------------------------------- #
# Predictive multiplicity
# --------------------------------------------------------------------------- #
def rashomon_set(fit_predict: Callable[[np.ndarray, int], np.ndarray],
                 n_train: int, n_models: int = 50, subsample: float = 0.9,
                 seed: int = 20260905):
    """
    Train `n_models` models that differ only in their random seed and in a
    `subsample` fraction of the training rows, and collect their test predictions.

    fit_predict(row_index, seed) must fit on those rows with that seed and return
    predicted probabilities for the (fixed) test set.

    Returns a matrix P of shape (n_test, n_models).
    """
    cols = []
    for s in range(n_models):
        # RandomState (not default_rng) so that notebooks reproduce the figures
        # in the lecture slides exactly. Both are fine; they must simply match.
        rng = np.random.RandomState(seed + s)
        idx = rng.choice(n_train, int(subsample * n_train), replace=False)
        cols.append(np.asarray(fit_predict(idx, seed + s), dtype=float).reshape(-1))
    return np.column_stack(cols)


def select_rashomon(P: np.ndarray, aucs: Iterable[float], epsilon: float = 0.01):
    """Keep the models whose AUC is within `epsilon` of the best. Returns (P_kept, mask)."""
    a = np.asarray(list(aucs), dtype=float)
    mask = a >= a.max() - epsilon
    return P[:, mask], mask


def plot_rashomon_fan(P: np.ndarray, threshold: float = 0.5, n_show: int = 70,
                      title: str = "", sort_desc: bool = True) -> go.Figure:
    """
    One vertical bar per applicant spanning the min-max predicted probability
    across the Rashomon set. Bars that straddle the threshold are highlighted.
    """
    lo, hi, mid = P.min(axis=1), P.max(axis=1), P.mean(axis=1)
    order = np.argsort(-mid if sort_desc else mid)[:n_show]
    flip = (hi[order] > threshold) & (lo[order] <= threshold)
    x = np.arange(len(order))

    fig = go.Figure()
    for mask, colour, nm in ((~flip, BLUE, "stable"), (flip, RED, "decision flips")):
        if not mask.any():
            continue
        xs, ys = [], []
        for xi, i in zip(x[mask], np.asarray(order)[mask]):
            xs += [xi, xi, None]
            ys += [lo[i], hi[i], None]
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=nm,
                                   line=dict(color=colour, width=3),
                                   opacity=0.75 if colour == BLUE else 1.0,
                                   hoverinfo="skip"))
    fig.add_trace(go.Scattergl(
        x=x, y=mid[order], mode="markers", name="mean",
        marker=dict(size=3.5, color=NAVY),
        hovertemplate="mean %{y:.3f}<extra></extra>"))
    fig.add_hline(y=threshold, line=dict(color="#333333", width=1.4, dash="dash"),
                  annotation_text=f"decision threshold {threshold:g}",
                  annotation_position="top left")

    n_flip = int((((P > threshold).any(1)) & ((P <= threshold).any(1))).sum())
    sub = (f"{n_flip} of {P.shape[0]} applicants ({100*n_flip/P.shape[0]:.1f}%) "
           f"flip across {P.shape[1]} models")
    return layout(fig, title or sub, height=460,
                  xaxis_title=f"applicants (highest {n_show} by mean prediction)",
                  yaxis_title="predicted probability")


def plot_explanation_spread(shap_rows: np.ndarray, feature_names: Sequence[str],
                            title: str = "", labels: dict | None = None) -> go.Figure:
    """
    For a single observation explained by several models, show the spread of SHAP
    values per feature -- one marker per model.
    """
    S = np.asarray(shap_rows, dtype=float)
    order = np.argsort(np.abs(S).mean(axis=0))
    tick = [labels.get(feature_names[j], feature_names[j]) if labels
            else feature_names[j] for j in order]

    fig = go.Figure()
    for row, j in enumerate(order):
        fig.add_trace(go.Scatter(
            x=S[:, j], y=[row] * S.shape[0], mode="markers",
            marker=dict(size=9, color=BLUE, opacity=0.55,
                        line=dict(width=0)), showlegend=False,
            hovertemplate=f"{feature_names[j]}<br>SHAP %{{x:+.4f}}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[S[:, j].mean()], y=[row], mode="markers", showlegend=False,
            marker=dict(size=14, color=NAVY, symbol="line-ns-open",
                        line=dict(width=2.5)), hoverinfo="skip"))
    fig.add_vline(x=0, line=dict(color="#999999", width=1))
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=tick, showgrid=False)
    return layout(fig, title, height=110 + 30 * len(order),
                  xaxis_title="SHAP value across models (one marker per model)")
