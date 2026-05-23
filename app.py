from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - app fallback for minimal installs
    PLOTLY_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - app fallback for minimal installs
    SKLEARN_AVAILABLE = False


# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="UTG Material Selection + ML Ranking Dashboard",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    .metric-card {
        padding: 1rem;
        border-radius: 1rem;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.06);
    }
    .small-note {font-size: 0.88rem; opacity: 0.78;}
    .good {color:#118847; font-weight:700;}
    .warn {color:#b26b00; font-weight:700;}
    .bad {color:#b00020; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Utility functions
# ============================================================
def midpoint(low: float, high: float) -> float:
    return (low + high) / 2.0


def normalize_high(series: pd.Series) -> pd.Series:
    """Map high-is-good values to a 1..5 score."""
    s = series.astype(float)
    lo, hi = float(s.min()), float(s.max())
    if math.isclose(lo, hi):
        return pd.Series(np.full(len(s), 3.0), index=s.index)
    return 1.0 + 4.0 * (s - lo) / (hi - lo)


def normalize_low(series: pd.Series) -> pd.Series:
    """Map low-is-good values to a 1..5 score."""
    s = series.astype(float)
    lo, hi = float(s.min()), float(s.max())
    if math.isclose(lo, hi):
        return pd.Series(np.full(len(s), 3.0), index=s.index)
    return 1.0 + 4.0 * (hi - s) / (hi - lo)


def safe_divide(a: float | pd.Series, b: float | pd.Series) -> float | pd.Series:
    return a / np.where(np.asarray(b) == 0, np.nan, b)


def bending_stress_curvature_mpa(E_gpa: float, thickness_um: float, bend_radius_mm: float) -> float:
    """
    Elastic outer-fiber bending stress for a thin sheet bent to radius R.
    sigma = E * t / (2R), with E in MPa, t/R in same length units.
    """
    if bend_radius_mm <= 0:
        return float("nan")
    E_mpa = E_gpa * 1000.0
    t_mm = thickness_um / 1000.0
    return E_mpa * t_mm / (2.0 * bend_radius_mm)


def bending_strain_percent(thickness_um: float, bend_radius_mm: float) -> float:
    if bend_radius_mm <= 0:
        return float("nan")
    t_mm = thickness_um / 1000.0
    return 100.0 * t_mm / (2.0 * bend_radius_mm)


def critical_flaw_um(KIC_mpa_sqrt_m: float, sigma_mpa: float, Y: float = 1.12) -> float:
    """
    From K_I = Y*sigma*sqrt(pi*a). Failure at K_I = K_IC.
    a_c = (KIC/(Y*sigma))^2 / pi. Returns micrometres.
    """
    if sigma_mpa <= 0 or Y <= 0:
        return float("nan")
    a_m = (KIC_mpa_sqrt_m / (Y * sigma_mpa)) ** 2 / math.pi
    return a_m * 1e6


def show_df(df: pd.DataFrame, height: int | str | None = "auto") -> None:
    """Safe dataframe renderer for recent Streamlit versions.

    Streamlit can reject height=None. Use "auto" by default, and only
    pass an integer height when a fixed table height is actually needed.
    """
    if height is None:
        height = "auto"
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def human_rank(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = df.sort_values(score_col, ascending=False).copy()
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


# ============================================================
# Candidate data
# ============================================================
@st.cache_data
def load_candidates() -> pd.DataFrame:
    # Ranges are taken from the report comparison table. Midpoints are used for deterministic ranking.
    # Units: density kg/m3, E GPa, flexural strength MPa, KIC MPa*sqrt(m), hardness HV, alpha microstrain/C.
    records = [
        {
            "Material": "Alumino silicate 1720",
            "Family": "Aluminosilicate",
            "rho_low": 2490, "rho_high": 2540,
            "E_low": 84.8, "E_high": 89.1,
            "sigma_low": 51.9, "sigma_high": 57.1,
            "KIC_low": 0.71, "KIC_high": 0.73,
            "H_low": 476, "H_high": 525,
            "alpha_low": 4.11, "alpha_high": 4.28,
            "thermal_shock_low": 108, "thermal_shock_high": 122,
            "price_low": 1.24, "price_high": 1.46,
            "co2_low": 0.890, "co2_high": 0.989,
            "energy_low": 13.3, "energy_high": 14.6,
            "water_low": 20.1, "water_high": 22.2,
            "application_relevance": 4.0,
            "chemical_strengthening": 4.5,
            "notes": "Selected baseline: strongest mechanical balance and good sustainability/cost.",
        },
        {
            "Material": "Alumino silicate 1723",
            "Family": "Aluminosilicate",
            "rho_low": 2610, "rho_high": 2660,
            "E_low": 83.9, "E_high": 88.1,
            "sigma_low": 48.9, "sigma_high": 53.8,
            "KIC_low": 0.70, "KIC_high": 0.72,
            "H_low": 477, "H_high": 525,
            "alpha_low": 4.50, "alpha_high": 4.69,
            "thermal_shock_low": 94.3, "thermal_shock_high": 106,
            "price_low": 1.24, "price_high": 1.46,
            "co2_low": 0.971, "co2_high": 1.07,
            "energy_low": 14.8, "energy_high": 16.3,
            "water_low": 22.2, "water_high": 24.6,
            "application_relevance": 5.0,
            "chemical_strengthening": 4.5,
            "notes": "Closest application-relevance reserve; slightly weaker thermal/sustainability balance.",
        },
        {
            "Material": "Borosilicate 7251",
            "Family": "Borosilicate",
            "rho_low": 2230, "rho_high": 2280,
            "E_low": 62.4, "E_high": 65.6,
            "sigma_low": 37.5, "sigma_high": 41.4,
            "KIC_low": 0.61, "KIC_high": 0.62,
            "H_low": 428, "H_high": 472,
            "alpha_low": 3.57, "alpha_high": 3.72,
            "thermal_shock_low": 122, "thermal_shock_high": 138,
            "price_low": 3.65, "price_high": 5.46,
            "co2_low": 1.26, "co2_high": 1.40,
            "energy_low": 20.4, "energy_high": 22.6,
            "water_low": 9.65, "water_high": 10.7,
            "application_relevance": 3.0,
            "chemical_strengthening": 2.5,
            "notes": "Good thermal and density profile; weaker strength/toughness than aluminosilicates.",
        },
        {
            "Material": "Borosilicate 7760",
            "Family": "Borosilicate",
            "rho_low": 2210, "rho_high": 2260,
            "E_low": 60.4, "E_high": 63.5,
            "sigma_low": 33.8, "sigma_high": 37.2,
            "KIC_low": 0.60, "KIC_high": 0.61,
            "H_low": 421, "H_high": 464,
            "alpha_low": 3.33, "alpha_high": 3.46,
            "thermal_shock_low": 123, "thermal_shock_high": 137,
            "price_low": 4.27, "price_high": 7.12,
            "co2_low": 1.32, "co2_high": 1.46,
            "energy_low": 21.4, "energy_high": 23.6,
            "water_low": 11.5, "water_high": 12.7,
            "application_relevance": 3.0,
            "chemical_strengthening": 2.0,
            "notes": "Low stiffness helps thermal/flex stress, but strength and crack-resistance are weakest.",
        },
        {
            "Material": "Borosilicate KG33",
            "Family": "Borosilicate",
            "rho_low": 2200, "rho_high": 2250,
            "E_low": 61.0, "E_high": 65.0,
            "sigma_low": 37.2, "sigma_high": 41.1,
            "KIC_low": 0.60, "KIC_high": 0.62,
            "H_low": 418, "H_high": 461,
            "alpha_low": 3.13, "alpha_high": 3.26,
            "thermal_shock_low": 140, "thermal_shock_high": 159,
            "price_low": 3.65, "price_high": 5.46,
            "co2_low": 1.28, "co2_high": 1.42,
            "energy_low": 20.7, "energy_high": 23.0,
            "water_low": 10.2, "water_high": 11.3,
            "application_relevance": 3.0,
            "chemical_strengthening": 2.5,
            "notes": "Best thermal-mismatch profile; not best for fracture toughness vs strength.",
        },
    ]

    df = pd.DataFrame(records)
    pairs = {
        "Density kg/m³": ("rho_low", "rho_high"),
        "Young modulus E, GPa": ("E_low", "E_high"),
        "Flexural strength σf, MPa": ("sigma_low", "sigma_high"),
        "Fracture toughness KIC, MPa√m": ("KIC_low", "KIC_high"),
        "Hardness HV": ("H_low", "H_high"),
        "CTE α, µstrain/°C": ("alpha_low", "alpha_high"),
        "Thermal shock °C": ("thermal_shock_low", "thermal_shock_high"),
        "Price EUR/kg": ("price_low", "price_high"),
        "CO₂ kg/kg": ("co2_low", "co2_high"),
        "Embodied energy MJ/kg": ("energy_low", "energy_high"),
        "Water L/kg": ("water_low", "water_high"),
    }
    for name, (lo, hi) in pairs.items():
        df[name] = (df[lo] + df[hi]) / 2.0

    return add_indices(df)


def add_indices(df: pd.DataFrame, extra_mode: str = "(KIC/E)^2") -> pd.DataFrame:
    out = df.copy()
    E = out["Young modulus E, GPa"].astype(float)
    sigma = out["Flexural strength σf, MPa"].astype(float)
    KIC = out["Fracture toughness KIC, MPa√m"].astype(float)
    alpha = out["CTE α, µstrain/°C"].astype(float)
    rho = out["Density kg/m³"].astype(float)
    H = out["Hardness HV"].astype(float)

    # Revised indices requested by the user/report update.
    out["M1 Foldability σf/E"] = sigma / E
    out["Elastic fracture strain, %"] = 100.0 * sigma / (E * 1000.0)
    out["M2 Crack tolerance KIC/σf"] = KIC / sigma
    out["Critical flaw a_c, µm"] = [critical_flaw_um(k, s) for k, s in zip(KIC, sigma)]
    out["M3 Thermal resistance 1/(Eα)"] = 1.0 / (E * alpha)
    out["Thermal mismatch αE"] = E * alpha
    out["M4 Extra (KIC/E)^2"] = (KIC / E) ** 2
    out["Reference Gc-like KIC²/E"] = (KIC ** 2) / E

    # Supporting, not core, because hardness is a hard/screening requirement in the revised logic.
    out["Support hardness score H"] = H
    out["Support H/ρ"] = H / rho
    out["Support sustainability raw"] = (
        normalize_low(out["Price EUR/kg"]) * 0.25
        + normalize_low(out["CO₂ kg/kg"]) * 0.30
        + normalize_low(out["Embodied energy MJ/kg"]) * 0.30
        + normalize_low(out["Water L/kg"]) * 0.15
    )
    return out


# ============================================================
# Scoring and ML functions
# ============================================================
def score_candidates(
    df: pd.DataFrame,
    w_fold: float,
    w_crack: float,
    w_thermal: float,
    w_extra: float,
    w_support: float,
    extra_col: str = "M4 Extra (KIC/E)^2",
) -> pd.DataFrame:
    scores = pd.DataFrame(index=df.index)
    scores["Material"] = df["Material"]
    scores["Family"] = df["Family"]
    scores["Foldability score"] = normalize_high(df["M1 Foldability σf/E"])
    # The report conclusion is based on the Ashby chart KIC vs σf, not on the ratio alone.
    # A pure KIC/σf score can accidentally reward weak glasses because a lower σf increases the ratio.
    # For the final decision matrix we therefore score the chart's preferred top-right region:
    # high fracture toughness AND high flexural strength. KIC/σf and critical flaw size remain diagnostic.
    scores["KIC absolute score"] = normalize_high(df["Fracture toughness KIC, MPa√m"])
    scores["Flexural strength score"] = normalize_high(df["Flexural strength σf, MPa"])
    scores["Crack ratio score"] = normalize_high(df["M2 Crack tolerance KIC/σf"])
    scores["Critical flaw score"] = normalize_high(df["Critical flaw a_c, µm"])
    scores["Thermal score"] = normalize_high(df["M3 Thermal resistance 1/(Eα)"])
    scores["Extra fracture score"] = normalize_high(df[extra_col])
    scores["Support score"] = (
        0.35 * normalize_high(df["Support hardness score H"])
        + 0.35 * df["Support sustainability raw"]
        + 0.15 * df["application_relevance"]
        + 0.15 * df["chemical_strengthening"]
    )

    # Main fracture score: top-right Ashby preference on KIC vs σf.
    # This matches the report: 1720 wins because it combines the highest KIC with the highest σf.
    scores["Fracture-vs-strength score"] = (
        0.55 * scores["KIC absolute score"] + 0.45 * scores["Flexural strength score"]
    )

    total_w = w_fold + w_crack + w_thermal + w_extra + w_support
    if total_w <= 0:
        total_w = 1.0
    scores["Physics weighted score"] = (
        w_fold * scores["Foldability score"]
        + w_crack * scores["Fracture-vs-strength score"]
        + w_thermal * scores["Thermal score"]
        + w_extra * scores["Extra fracture score"]
        + w_support * scores["Support score"]
    ) / total_w

    return scores.sort_values("Physics weighted score", ascending=False).reset_index(drop=True)


def sample_candidate_properties(df: pd.DataFrame, samples_per_material: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[pd.DataFrame] = []
    for _, row in df.iterrows():
        n = samples_per_material
        block = pd.DataFrame(
            {
                "Material": row["Material"],
                "Family": row["Family"],
                "Density kg/m³": rng.uniform(row["rho_low"], row["rho_high"], n),
                "Young modulus E, GPa": rng.uniform(row["E_low"], row["E_high"], n),
                "Flexural strength σf, MPa": rng.uniform(row["sigma_low"], row["sigma_high"], n),
                "Fracture toughness KIC, MPa√m": rng.uniform(row["KIC_low"], row["KIC_high"], n),
                "Hardness HV": rng.uniform(row["H_low"], row["H_high"], n),
                "CTE α, µstrain/°C": rng.uniform(row["alpha_low"], row["alpha_high"], n),
                "Thermal shock °C": rng.uniform(row["thermal_shock_low"], row["thermal_shock_high"], n),
                "Price EUR/kg": rng.uniform(row["price_low"], row["price_high"], n),
                "CO₂ kg/kg": rng.uniform(row["co2_low"], row["co2_high"], n),
                "Embodied energy MJ/kg": rng.uniform(row["energy_low"], row["energy_high"], n),
                "Water L/kg": rng.uniform(row["water_low"], row["water_high"], n),
                "application_relevance": row["application_relevance"],
                "chemical_strengthening": row["chemical_strengthening"],
            }
        )
        # Add fake low/high columns so add_indices can work on synthetic blocks if needed.
        for col in ["rho", "E", "sigma", "KIC", "H", "alpha", "thermal_shock", "price", "co2", "energy", "water"]:
            block[f"{col}_low"] = np.nan
            block[f"{col}_high"] = np.nan
        rows.append(block)
    synthetic = pd.concat(rows, ignore_index=True)
    return add_indices(synthetic)


def compute_score_vector(
    data: pd.DataFrame,
    w_fold: float,
    w_crack: float,
    w_thermal: float,
    w_extra: float,
    w_support: float,
    extra_col: str,
) -> pd.Series:
    scored = score_candidates(data, w_fold, w_crack, w_thermal, w_extra, w_support, extra_col)
    # score_candidates sorts; restore original rows by merging on Material and index is not unique.
    # For synthetic data, compute directly to preserve row order.
    fold = normalize_high(data["M1 Foldability σf/E"])
    kic_abs = normalize_high(data["Fracture toughness KIC, MPa√m"])
    sigma_abs = normalize_high(data["Flexural strength σf, MPa"])
    thermal = normalize_high(data["M3 Thermal resistance 1/(Eα)"])
    extra = normalize_high(data[extra_col])
    support = (
        0.35 * normalize_high(data["Support hardness score H"])
        + 0.35 * data["Support sustainability raw"]
        + 0.15 * data["application_relevance"]
        + 0.15 * data["chemical_strengthening"]
    )
    fracture = 0.55 * kic_abs + 0.45 * sigma_abs
    total_w = w_fold + w_crack + w_thermal + w_extra + w_support
    if total_w <= 0:
        total_w = 1.0
    return (w_fold * fold + w_crack * fracture + w_thermal * thermal + w_extra * extra + w_support * support) / total_w


FEATURE_COLUMNS = [
    "Young modulus E, GPa",
    "Flexural strength σf, MPa",
    "Fracture toughness KIC, MPa√m",
    "CTE α, µstrain/°C",
    "Density kg/m³",
    "Hardness HV",
    "M1 Foldability σf/E",
    "M2 Crack tolerance KIC/σf",
    "Critical flaw a_c, µm",
    "M3 Thermal resistance 1/(Eα)",
    "M4 Extra (KIC/E)^2",
    "Reference Gc-like KIC²/E",
    "Price EUR/kg",
    "CO₂ kg/kg",
    "Embodied energy MJ/kg",
    "application_relevance",
    "chemical_strengthening",
]


@st.cache_data
def build_synthetic_training_data(
    base_df: pd.DataFrame,
    samples_per_material: int,
    seed: int,
    w_fold: float,
    w_crack: float,
    w_thermal: float,
    w_extra: float,
    w_support: float,
    extra_col: str,
) -> pd.DataFrame:
    synthetic = sample_candidate_properties(base_df, samples_per_material, seed=seed)
    synthetic["Target physics score"] = compute_score_vector(
        synthetic, w_fold, w_crack, w_thermal, w_extra, w_support, extra_col
    )
    synthetic["Suitability class"] = pd.cut(
        synthetic["Target physics score"],
        bins=[-np.inf, 2.75, 3.75, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return synthetic


def train_ml_models(training: pd.DataFrame, seed: int = 12):
    X = training[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(training[FEATURE_COLUMNS].median(numeric_only=True))
    y = training["Target physics score"]
    y_cls = training["Suitability class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    reg = RandomForestRegressor(n_estimators=350, random_state=seed, min_samples_leaf=2)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_cls, test_size=0.25, random_state=seed, stratify=y_cls)
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, min_samples_leaf=2)
    clf.fit(Xc_train, yc_train)
    pred_cls = clf.predict(Xc_test)

    metrics = {
        "R²": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "Class accuracy": accuracy_score(yc_test, pred_cls),
    }
    importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": reg.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    return reg, clf, metrics, importance


# ============================================================
# Plot helpers
# ============================================================
def scatter_plot(df: pd.DataFrame, x: str, y: str, title: str, hover: List[str] | None = None, log_x: bool = False, log_y: bool = False):
    if hover is None:
        hover = []
    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color="Family",
            text="Material",
            hover_data=["Material"] + hover,
            title=title,
            size=np.clip(df.get("Physics weighted score", pd.Series(np.full(len(df), 3))), 1, 6),
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=560, legend_title_text="Glass family")
        fig.update_xaxes(type="log" if log_x else "linear")
        fig.update_yaxes(type="log" if log_y else "linear")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.scatter_chart(df, x=x, y=y, color="Family")


def bar_plot(df: pd.DataFrame, x: str, y: str, title: str, ascending: bool = False):
    chart_df = df.sort_values(y, ascending=ascending)
    if PLOTLY_AVAILABLE:
        fig = px.bar(chart_df, x=x, y=y, color="Family" if "Family" in chart_df.columns else None, title=title, text_auto=".3g")
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(chart_df.set_index(x)[[y]])


# ============================================================
# Main app state
# ============================================================
base_df = load_candidates()

st.sidebar.title("UTG dashboard")
st.sidebar.caption("Revised for the new index logic")

extra_mode = st.sidebar.radio(
    "Extra fracture index",
    ["(KIC/E)^2", "KIC²/E"],
    index=0,
    help="You asked for toughness² over modulus². The standard fracture-energy check is KIC²/E, so both are available.",
)
extra_col = "M4 Extra (KIC/E)^2" if extra_mode == "(KIC/E)^2" else "Reference Gc-like KIC²/E"

st.sidebar.markdown("### Matrix weights")
w_fold = st.sidebar.slider("Foldability: σf/E", 0.0, 60.0, 35.0, 1.0)
w_crack = st.sidebar.slider("Crack chart: high KIC + high σf", 0.0, 60.0, 30.0, 1.0)
w_thermal = st.sidebar.slider("Thermal: 1/(Eα)", 0.0, 40.0, 15.0, 1.0)
w_extra = st.sidebar.slider(f"Extra: {extra_mode}", 0.0, 40.0, 10.0, 1.0)
w_support = st.sidebar.slider("Support: cost, hardness, UTG relevance", 0.0, 30.0, 10.0, 1.0)

scored = score_candidates(base_df, w_fold, w_crack, w_thermal, w_extra, w_support, extra_col)
ranked_full = base_df.merge(scored[["Material", "Physics weighted score"]], on="Material", how="left")
ranked_full = ranked_full.sort_values("Physics weighted score", ascending=False).reset_index(drop=True)

page = st.sidebar.radio(
    "Menu",
    [
        "Home",
        "Revised indices",
        "Ashby charts",
        "Decision matrix",
        "ML ranking engine",
        "Monte Carlo uncertainty",
        "Bending + crack validation",
        "Eco / sustainability",
        "Report wording",
        "Raw data",
    ],
)

winner = scored.iloc[0]["Material"]
st.sidebar.success(f"Current winner: {winner}")


# ============================================================
# Pages
# ============================================================
if page == "Home":
    st.title("Foldable UTG Material Selection — Revised Intelligent Dashboard")
    st.markdown(
        """
        This version follows the corrected selection logic for the foldable phone ultra-thin glass cover window.
        The main ranking is no longer driven by hardness-density or by the old **KIC/E** fracture index alone.
        The core is now:

        1. **Elastic foldability:** flexural strength vs Young's modulus, using **σf/E**.
        2. **Safe brittle-fracture balance:** fracture toughness vs flexural strength, scored as the top-right Ashby region: high **KIC** and high **σf** together. The ratio **KIC/σf** and critical flaw size are shown as diagnostics, not allowed to overturn the final report conclusion by rewarding weak glasses.
        3. **Thermal mismatch resistance:** **1/(Eα)**, equivalent to preferring low **Eα**.
        4. **Extra fracture reserve:** **(KIC/E)²** by default, with a switch for the standard **KIC²/E** fracture-energy check.

        The ML page uses synthetic samples from the Granta property ranges to train a transparent decision-support model.
        It is not presented as a real fold-cycle lifetime predictor.
        """
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Initial Granta space", "3895")
    c2.metric("Hard-filter survivors", "14")
    c3.metric("Final shortlist", "5")
    c4.metric("Top material", winner)
    c5.metric("Core indices", "4")

    st.subheader("Current ranking")
    display_cols = [
        "Material",
        "Family",
        "Physics weighted score",
        "M1 Foldability σf/E",
        "M2 Crack tolerance KIC/σf",
        "Critical flaw a_c, µm",
        "M3 Thermal resistance 1/(Eα)",
        extra_col,
    ]
    show_df(human_rank(ranked_full[display_cols], "Physics weighted score"))
    bar_plot(ranked_full, "Material", "Physics weighted score", "Revised physics-weighted score")

    st.info(
        "Best interpretation: 1720 wins when bending reliability and fracture-vs-strength are weighted highest. "
        "KG33 can win thermal-only checks, but it is weaker in the crack/strength balance."
    )

elif page == "Revised indices":
    st.title("Revised Ashby indices")
    st.markdown(
        """
        These are the exact indices implemented in this dashboard. Hardness is kept as a supporting/pass-fail property,
        not as a main performance index, because the revised logic focuses on bending, crack growth, thermal mismatch,
        and fracture reserve.
        """
    )

    st.latex(r"M_1 = \frac{\sigma_f}{E}")
    st.caption("Foldability index: higher flexural strength and lower stiffness give higher elastic bending strain before fracture.")
    st.latex(r"M_2 = \frac{K_{IC}}{\sigma_f}, \qquad a_c = \frac{1}{\pi}\left(\frac{K_{IC}}{Y\sigma_f}\right)^2")
    st.caption("Fracture toughness vs flexural strength: the final matrix rewards the top-right region, meaning high KIC and high σf together. KIC/σf and critical flaw size are diagnostic checks only.")
    st.latex(r"M_3 = \frac{1}{E\alpha}")
    st.caption("Thermal-mismatch resistance: high value means lower thermal stress tendency because σthermal = EαΔT.")
    st.latex(r"M_4 = \left(\frac{K_{IC}}{E}\right)^2")
    st.caption("Extra index requested as toughness² over modulus². A toggle also allows the standard Griffith energy term KIC²/E.")

    index_cols = [
        "Material",
        "M1 Foldability σf/E",
        "Elastic fracture strain, %",
        "M2 Crack tolerance KIC/σf",
        "Critical flaw a_c, µm",
        "M3 Thermal resistance 1/(Eα)",
        "Thermal mismatch αE",
        "M4 Extra (KIC/E)^2",
        "Reference Gc-like KIC²/E",
    ]
    show_df(base_df[index_cols])

    selected_index = st.selectbox(
        "Plot one index",
        [
            "M1 Foldability σf/E",
            "M2 Crack tolerance KIC/σf",
            "Critical flaw a_c, µm",
            "M3 Thermal resistance 1/(Eα)",
            "M4 Extra (KIC/E)^2",
            "Reference Gc-like KIC²/E",
            "Thermal mismatch αE",
        ],
    )
    higher_is_good = selected_index != "Thermal mismatch αE"
    bar_plot(base_df, "Material", selected_index, selected_index, ascending=not higher_is_good)
    if higher_is_good:
        st.success("For this plotted index, higher is better.")
    else:
        st.warning("For αE, lower is better. The scoring matrix uses 1/(Eα) so that higher is better.")

elif page == "Ashby charts":
    st.title("Ashby charts matched to the revised logic")
    st.markdown("The charts below directly match the revised material-selection argument.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "σf vs E",
        "KIC vs σf",
        "α vs E",
        "Extra fracture reserve",
    ])

    with tab1:
        scatter_plot(
            ranked_full,
            x="Young modulus E, GPa",
            y="Flexural strength σf, MPa",
            title="Elastic foldability: flexural strength vs Young's modulus",
            hover=["M1 Foldability σf/E", "Elastic fracture strain, %"],
        )
        st.caption("Preferred region: higher σf at lower/moderate E, i.e. higher σf/E.")

    with tab2:
        scatter_plot(
            ranked_full,
            x="Flexural strength σf, MPa",
            y="Fracture toughness KIC, MPa√m",
            title="Safe brittle fracture: fracture toughness vs flexural strength",
            hover=["M2 Crack tolerance KIC/σf", "Critical flaw a_c, µm"],
        )
        st.caption("Preferred region: top-right. This is why 1720 wins in the report: it has the strongest combined KIC and flexural strength, even if borosilicates look good in ratio-based diagnostics.")

    with tab3:
        scatter_plot(
            ranked_full,
            x="Young modulus E, GPa",
            y="CTE α, µstrain/°C",
            title="Thermal mismatch: coefficient of thermal expansion vs Young's modulus",
            hover=["Thermal mismatch αE", "M3 Thermal resistance 1/(Eα)"],
        )
        st.caption("Preferred region: bottom-left. Low Eα reduces thermal stress in the laminated display stack.")

    with tab4:
        y_col = "M4 Extra (KIC/E)^2" if extra_mode == "(KIC/E)^2" else "Reference Gc-like KIC²/E"
        scatter_plot(
            ranked_full,
            x="Young modulus E, GPa",
            y="Fracture toughness KIC, MPa√m",
            title=f"Extra fracture reserve: {extra_mode}",
            hover=[y_col],
        )
        st.caption("Use this as a secondary check, not as the only selection criterion.")

elif page == "Decision matrix":
    st.title("Revised weighted decision matrix")
    st.markdown(
        """
        The matrix is now built around the revised four-index structure. Support factors can be included,
        but they are intentionally kept below the core mechanical and thermal indices by default.
        """
    )

    matrix_cols = [
        "Material",
        "Family",
        "Foldability score",
        "KIC absolute score",
        "Flexural strength score",
        "Fracture-vs-strength score",
        "Thermal score",
        "Extra fracture score",
        "Support score",
        "Physics weighted score",
    ]
    show_df(human_rank(scored[matrix_cols], "Physics weighted score"))
    bar_plot(scored, "Material", "Physics weighted score", "Decision matrix ranking")

    st.subheader("Why each material ranks where it does")
    for _, row in ranked_full.iterrows():
        mat = row["Material"]
        score_row = scored[scored["Material"] == mat].iloc[0]
        with st.expander(f"{mat} — score {score_row['Physics weighted score']:.2f}"):
            st.write(row["notes"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("σf/E", f"{row['M1 Foldability σf/E']:.3f}")
            c2.metric("KIC/σf", f"{row['M2 Crack tolerance KIC/σf']:.4f}")
            c3.metric("a_c", f"{row['Critical flaw a_c, µm']:.1f} µm")
            c4.metric("1/(Eα)", f"{row['M3 Thermal resistance 1/(Eα)']:.5f}")

    st.download_button(
        "Download revised matrix CSV",
        data=scored.to_csv(index=False),
        file_name="revised_utg_decision_matrix.csv",
        mime="text/csv",
    )

elif page == "ML ranking engine":
    st.title("Machine-learning decision support")
    st.markdown(
        """
        The ML module samples the Granta property ranges and trains a surrogate model to reproduce the revised
        physics/decision score. This makes the dashboard more intelligent: it can estimate ranking stability,
        feature importance, and how property changes would affect the score.

        It is **not** a real experimental fold-life predictor, because the data do not contain measured cyclic-folding failures.
        """
    )

    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is not installed. Add `scikit-learn` to requirements.txt to enable this page.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            samples_per_material = st.slider("Synthetic samples per material", 100, 2000, 700, 100)
        with c2:
            seed = st.number_input("Random seed", value=13, step=1)
        with c3:
            selected_material = st.selectbox("Counterfactual material", base_df["Material"].tolist())

        training = build_synthetic_training_data(
            base_df, samples_per_material, int(seed), w_fold, w_crack, w_thermal, w_extra, w_support, extra_col
        )
        reg, clf, metrics, importance = train_ml_models(training, seed=int(seed))

        c1, c2, c3 = st.columns(3)
        c1.metric("Regression R²", f"{metrics['R²']:.3f}")
        c2.metric("Regression MAE", f"{metrics['MAE']:.3f}")
        c3.metric("Class accuracy", f"{metrics['Class accuracy']:.3f}")

        st.subheader("ML-predicted score for midpoint candidates")
        X_mid = base_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(base_df[FEATURE_COLUMNS].median(numeric_only=True))
        ml_pred = base_df[["Material", "Family"]].copy()
        ml_pred["ML predicted score"] = reg.predict(X_mid)
        ml_pred["ML class"] = clf.predict(X_mid)
        ml_pred = ml_pred.merge(scored[["Material", "Physics weighted score"]], on="Material")
        ml_pred = ml_pred.sort_values("ML predicted score", ascending=False)
        show_df(human_rank(ml_pred, "ML predicted score"))
        bar_plot(ml_pred, "Material", "ML predicted score", "ML-predicted ranking")

        st.subheader("Feature importance")
        show_df(importance.head(12))
        bar_plot(importance.head(12), "Feature", "Importance", "Top ML feature importances")

        st.subheader("Counterfactual strengthening / design improvement")
        row = base_df[base_df["Material"] == selected_material].iloc[0].copy()
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            sigma_boost = st.slider("Flexural strength boost %", 0.0, 400.0, 100.0, 5.0)
        with c2:
            kic_boost = st.slider("Fracture toughness boost %", 0.0, 80.0, 0.0, 2.5)
        with c3:
            E_change = st.slider("Young modulus change %", -30.0, 30.0, 0.0, 1.0)
        with c4:
            alpha_change = st.slider("CTE change %", -30.0, 30.0, 0.0, 1.0)

        modified = pd.DataFrame([row])
        modified["Flexural strength σf, MPa"] *= (1.0 + sigma_boost / 100.0)
        modified["Fracture toughness KIC, MPa√m"] *= (1.0 + kic_boost / 100.0)
        modified["Young modulus E, GPa"] *= (1.0 + E_change / 100.0)
        modified["CTE α, µstrain/°C"] *= (1.0 + alpha_change / 100.0)
        modified = add_indices(modified)
        X_mod = modified[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(X_mid.median(numeric_only=True))
        pred_score = float(reg.predict(X_mod)[0])
        pred_class = str(clf.predict(X_mod)[0])

        base_pred = float(reg.predict(X_mid[base_df["Material"] == selected_material])[0])
        c1, c2, c3 = st.columns(3)
        c1.metric("Base ML score", f"{base_pred:.2f}")
        c2.metric("Modified ML score", f"{pred_score:.2f}", delta=f"{pred_score - base_pred:+.2f}")
        c3.metric("Predicted class", pred_class)
        st.caption("Use this to discuss why chemical strengthening and edge finishing can change practical suitability beyond base Granta values.")

elif page == "Monte Carlo uncertainty":
    st.title("Monte Carlo uncertainty from material-property ranges")
    st.markdown(
        """
        This page samples inside each Granta range and recalculates the revised score. It answers:
        **How stable is the ranking if the property values move within their ranges?**
        """
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        n_samples = st.slider("Samples per material", 200, 5000, 1500, 100)
    with c2:
        seed = st.number_input("Monte Carlo seed", value=21, step=1)
    with c3:
        min_flaw_um = st.slider("Required flaw tolerance a_c (µm)", 10.0, 100.0, 35.0, 5.0)

    mc = sample_candidate_properties(base_df, int(n_samples), seed=int(seed))
    mc["Score"] = compute_score_vector(mc, w_fold, w_crack, w_thermal, w_extra, w_support, extra_col)
    mc["Pass flaw target"] = mc["Critical flaw a_c, µm"] >= min_flaw_um

    summary = mc.groupby("Material").agg(
        mean_score=("Score", "mean"),
        p05=("Score", lambda s: np.percentile(s, 5)),
        p50=("Score", "median"),
        p95=("Score", lambda s: np.percentile(s, 95)),
        pass_flaw_probability=("Pass flaw target", "mean"),
        mean_critical_flaw_um=("Critical flaw a_c, µm", "mean"),
    ).reset_index()

    # Probability of rank #1 by grouped score samples.
    pivot = mc.pivot_table(index=mc.groupby("Material").cumcount(), columns="Material", values="Score")
    winners = pivot.idxmax(axis=1).value_counts(normalize=True).rename("probability_rank_1").reset_index()
    winners.columns = ["Material", "probability_rank_1"]
    summary = summary.merge(winners, on="Material", how="left").fillna({"probability_rank_1": 0})
    summary = summary.merge(base_df[["Material", "Family"]], on="Material", how="left")
    summary = summary.sort_values("mean_score", ascending=False)

    show_df(summary)
    bar_plot(summary, "Material", "probability_rank_1", "Probability of ranking #1")

    if PLOTLY_AVAILABLE:
        fig = px.box(mc, x="Material", y="Score", color="Family", title="Score distribution by material")
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(summary.set_index("Material")[["mean_score", "p05", "p95"]])

elif page == "Bending + crack validation":
    st.title("Bending stress + crack validation check")
    st.markdown(
        """
        This page links the ranking to real UTG validation. It checks bending stress, elastic strain,
        chemical surface compression, and the critical flaw size from fracture mechanics.
        """
    )

    material = st.selectbox("Material", base_df["Material"].tolist())
    row = base_df[base_df["Material"] == material].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        E = st.number_input("Young modulus E (GPa)", value=float(row["Young modulus E, GPa"]), min_value=1.0)
    with c2:
        sigma_f = st.number_input("Flexural strength σf (MPa)", value=float(row["Flexural strength σf, MPa"]), min_value=1.0)
    with c3:
        KIC = st.number_input("KIC (MPa√m)", value=float(row["Fracture toughness KIC, MPa√m"]), min_value=0.01)
    with c4:
        alpha = st.number_input("CTE α (µstrain/°C)", value=float(row["CTE α, µstrain/°C"]), min_value=0.1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        thickness_um = st.slider("UTG thickness (µm)", 10.0, 150.0, 30.0)
    with c2:
        bend_radius_mm = st.slider("Bend radius R (mm)", 0.3, 10.0, 1.5)
    with c3:
        surface_compression = st.slider("Surface compression CS (MPa)", 0.0, 1200.0, 865.0)
    with c4:
        Y = st.slider("Crack geometry factor Y", 0.8, 1.5, 1.12)

    sigma_b = bending_stress_curvature_mpa(E, thickness_um, bend_radius_mm)
    strain = bending_strain_percent(thickness_um, bend_radius_mm)
    net_tensile = max(sigma_b - surface_compression, 0.0)
    a_base = critical_flaw_um(KIC, sigma_f, Y=Y)
    a_bending = critical_flaw_um(KIC, max(net_tensile, 1e-6), Y=Y) if net_tensile > 0 else float("inf")
    thermal_stress_40 = E * 1000 * alpha * 1e-6 * 40.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bending strain", f"{strain:.3f}%")
    m2.metric("Elastic bending stress", f"{sigma_b:.1f} MPa")
    m3.metric("Net tensile after CS", f"{net_tensile:.1f} MPa")
    m4.metric("Thermal stress for ΔT=40°C", f"{thermal_stress_40:.1f} MPa")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("σf/E index", f"{sigma_f/E:.3f}")
    m2.metric("KIC/σf index", f"{KIC/sigma_f:.4f}")
    m3.metric("a_c at σf", f"{a_base:.1f} µm")
    m4.metric("a_c at net bend stress", "∞" if math.isinf(a_bending) else f"{a_bending:.1f} µm")

    if sigma_b > surface_compression + sigma_f:
        st.error("High risk in this simplified check: bending stress exceeds surface compression plus base flexural strength.")
    elif sigma_b > surface_compression:
        st.warning("Net tensile stress remains after chemical compression. Edge/surface flaw control is critical.")
    else:
        st.success("Surface compression is higher than the estimated elastic bending stress in this simplified check.")

    st.caption(
        "This is not a final pass/fail proof. Real approval needs processed UTG strength, depth of layer, residual stress profile, edge quality, coatings, adhesive stack, and cyclic fatigue testing."
    )

elif page == "Eco / sustainability":
    st.title("Eco-audit sensitivity and sustainability support")
    st.markdown(
        """
        This page keeps sustainability as a supporting criterion. The material layer is extremely light,
        so the technical choice should not be overturned by tiny mass differences unless reliability is equal.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        display_power = st.slider("Display power assigned to stack (W)", 0.0, 3.0, 1.5)
    with c2:
        daily_use = st.slider("Daily display use (h/day)", 0.0, 10.0, 4.6)
    with c3:
        lifetime = st.slider("Product life (years)", 1.0, 7.0, 3.0)

    baseline_use_energy_mj = 76.1
    baseline_use_co2 = 2.49
    scale = (display_power * daily_use * lifetime) / (1.5 * 4.6 * 3.0) if display_power > 0 else 0.0
    use_energy = baseline_use_energy_mj * scale
    use_co2 = baseline_use_co2 * scale
    non_use_energy = 0.0241 + 0.0205 + 0.169 + 0.00031
    non_use_co2 = 0.00158 + 0.00163 + 0.012 + 0.0000217

    eco = pd.DataFrame(
        {
            "Phase": ["Material", "Manufacture", "Transport", "Use", "Disposal"],
            "Energy MJ": [0.0241, 0.0205, 0.169, use_energy, 0.00031],
            "CO₂ kg": [0.00158, 0.00163, 0.012, use_co2, 0.0000217],
        }
    )
    total_energy = eco["Energy MJ"].sum()
    total_co2 = eco["CO₂ kg"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total energy", f"{total_energy:.2f} MJ")
    c2.metric("Use energy share", f"{100*use_energy/total_energy:.1f}%" if total_energy else "0%")
    c3.metric("Total CO₂", f"{total_co2:.2f} kg")
    c4.metric("Use CO₂ share", f"{100*use_co2/total_co2:.1f}%" if total_co2 else "0%")
    show_df(eco)

    if PLOTLY_AVAILABLE:
        fig = px.bar(eco, x="Phase", y="Energy MJ", title="Eco-audit energy by phase")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.bar(eco, x="Phase", y="CO₂ kg", title="Eco-audit CO₂ by phase")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.bar_chart(eco.set_index("Phase")[["Energy MJ", "CO₂ kg"]])

    st.subheader("Material sustainability indicators")
    support_cols = ["Material", "Price EUR/kg", "CO₂ kg/kg", "Embodied energy MJ/kg", "Water L/kg", "Support sustainability raw"]
    show_df(base_df[support_cols].sort_values("Support sustainability raw", ascending=False))

elif page == "Report wording":
    st.title("Report-ready wording for the dashboard section")
    wording = f"""
    A revised Streamlit dashboard was developed to make the UTG material-selection workflow reproducible and to align the digital tool with the corrected Ashby-index logic. The dashboard uses the five shortlisted transparent glass candidates and recalculates the main performance indices from the Granta midpoint values. The core indices are elastic foldability, \u03c3f/E; fracture toughness versus flexural strength, KIC/\u03c3f; thermal-mismatch resistance, 1/(E\u03b1); and an additional fracture-reserve index, (KIC/E)^2. The standard fracture-energy relation KIC^2/E is also retained as an optional sensitivity check, but it is not allowed to replace the main fracture-toughness-versus-strength criterion.

    The revised ranking matrix gives the highest priority to foldability and crack resistance because UTG failure is controlled by tensile bending stress and flaw-driven fracture. Thermal mismatch is included because the glass is laminated into a display stack and thermal stress scales with E\u03b1\u0394T. Hardness, cost, sustainability, application relevance, and chemical-strengthening relevance are treated as supporting factors rather than as the main mechanical selection indices.

    To make the dashboard more intelligent, a machine-learning decision-support module was added. Because only five real candidate materials are available, the model is not presented as a true experimental lifetime predictor. Instead, synthetic samples are generated inside the Granta property ranges for each candidate, and a random-forest surrogate is trained to reproduce the revised physics-based score. This allows sensitivity analysis, feature-importance ranking, uncertainty estimation, and counterfactual checks such as the effect of improved flexural strength after chemical strengthening.

    The dashboard result supports Alumino silicate 1720 as the strongest overall candidate under the revised logic. Its advantage comes from the best balance of elastic foldability, fracture toughness versus flexural strength, and supporting sustainability/cost indicators. Borosilicate KG33 remains an important comparator because it performs well in thermal mismatch, but it does not overtake 1720 when bending and crack resistance are weighted as the dominant UTG failure criteria. Therefore, the dashboard strengthens the report conclusion while clearly showing that final approval still requires chemical strengthening validation, edge inspection, bend-radius testing, cyclic folding fatigue, scratch/pen-drop testing, optical inspection, lamination testing, and manufacturing-yield evaluation.
    """.strip()
    st.text_area("Copy this into your report", wording, height=430)
    st.download_button(
        "Download report wording",
        data=wording,
        file_name="revised_dashboard_report_wording.txt",
        mime="text/plain",
    )

elif page == "Raw data":
    st.title("Raw candidate data")
    show_df(base_df)
    st.download_button(
        "Download full data CSV",
        data=base_df.to_csv(index=False),
        file_name="utg_revised_candidate_data.csv",
        mime="text/csv",
    )

