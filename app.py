from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

st.set_page_config(
    page_title="UTG Material Selection Dashboard",
    page_icon="📱",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Alumino silicate 1720": {
        "Density kg/m3": (2490, 2540),
        "E GPa": (84.8, 89.1),
        "Flexural strength MPa": (51.9, 57.1),
        "KIC MPa√m": (0.71, 0.73),
        "CTE µstrain/°C": (4.11, 4.28),
        "Price EUR/kg": (1.24, 1.46),
        "CO2 kg/kg": (0.890, 0.989),
        "Embodied energy MJ/kg": (13.3, 14.6),
        "Water L/kg": (20.1, 22.2),
        "Critical element risk": (1, 1),
        "Recycle fraction": (0.01, 0.01),
        "Application relevance": (4, 4),
    },
    "Alumino silicate 1723": {
        "Density kg/m3": (2610, 2660),
        "E GPa": (83.9, 88.1),
        "Flexural strength MPa": (48.9, 53.8),
        "KIC MPa√m": (0.70, 0.72),
        "CTE µstrain/°C": (4.50, 4.69),
        "Price EUR/kg": (1.24, 1.46),
        "CO2 kg/kg": (0.971, 1.07),
        "Embodied energy MJ/kg": (14.8, 16.3),
        "Water L/kg": (22.2, 24.6),
        "Critical element risk": (1, 1),
        "Recycle fraction": (0.01, 0.01),
        "Application relevance": (5, 5),
    },
    "Borosilicate 7251": {
        "Density kg/m3": (2230, 2280),
        "E GPa": (62.4, 65.6),
        "Flexural strength MPa": (37.5, 41.4),
        "KIC MPa√m": (0.61, 0.62),
        "CTE µstrain/°C": (3.57, 3.72),
        "Price EUR/kg": (3.65, 5.46),
        "CO2 kg/kg": (1.26, 1.40),
        "Embodied energy MJ/kg": (20.4, 22.6),
        "Water L/kg": (9.65, 10.7),
        "Critical element risk": (0, 0),
        "Recycle fraction": (0.24, 0.24),
        "Application relevance": (3, 3),
    },
    "Borosilicate 7760": {
        "Density kg/m3": (2210, 2260),
        "E GPa": (60.4, 63.5),
        "Flexural strength MPa": (33.8, 37.2),
        "KIC MPa√m": (0.60, 0.61),
        "CTE µstrain/°C": (3.33, 3.46),
        "Price EUR/kg": (4.27, 7.12),
        "CO2 kg/kg": (1.32, 1.46),
        "Embodied energy MJ/kg": (21.4, 23.6),
        "Water L/kg": (11.5, 12.7),
        "Critical element risk": (0, 0),
        "Recycle fraction": (0.24, 0.24),
        "Application relevance": (3, 3),
    },
    "Borosilicate KG33": {
        "Density kg/m3": (2200, 2250),
        "E GPa": (61.0, 65.0),
        "Flexural strength MPa": (37.2, 41.1),
        "KIC MPa√m": (0.60, 0.62),
        "CTE µstrain/°C": (3.13, 3.26),
        "Price EUR/kg": (3.65, 5.46),
        "CO2 kg/kg": (1.28, 1.42),
        "Embodied energy MJ/kg": (20.7, 23.0),
        "Water L/kg": (10.2, 11.3),
        "Critical element risk": (0, 0),
        "Recycle fraction": (0.24, 0.24),
        "Application relevance": (3, 3),
    },
}

DEFAULT_WEIGHTS = {
    "Foldability": 30.0,
    "Crack resistance": 20.0,
    "Flaw tolerance": 20.0,
    "Thermal compatibility": 20.0,
    "Cost and sustainability": 10.0,
}

# -----------------------------------------------------------------------------
# Calculation helpers
# -----------------------------------------------------------------------------

def midpoint_dataframe() -> pd.DataFrame:
    rows = []
    for material, props in RANGES.items():
        row = {"Material": material}
        for key, (lo, hi) in props.items():
            row[key] = (lo + hi) / 2
            row[f"{key} min"] = lo
            row[f"{key} max"] = hi
        rows.append(row)
    return pd.DataFrame(rows).set_index("Material")


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    E = out["E GPa"]
    sigma = out["Flexural strength MPa"]
    kic = out["KIC MPa√m"]
    alpha = out["CTE µstrain/°C"]

    out["M1 foldability σf/E"] = sigma / E
    out["M2 crack resistance KIC²/E"] = (kic ** 2) / E
    out["M3 flaw tolerance KIC/σf"] = kic / sigma
    out["M4 thermal resistance 1/(Eα)"] = 1 / (E * alpha)
    out["Thermal mismatch Eα"] = E * alpha
    out["Critical flaw size proxy (KIC/σf)²"] = (kic / sigma) ** 2
    return out


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    lo = s.min()
    hi = s.max()
    if math.isclose(float(lo), float(hi)):
        return pd.Series(3.0, index=s.index)
    raw = 1 + 4 * (s - lo) / (hi - lo)
    if not higher_is_better:
        raw = 1 + 4 * (hi - s) / (hi - lo)
    return raw.clip(1, 5)


def normalized_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(float(v), 0.0) for v in weights.values())
    if total <= 0:
        return {k: 1 / len(weights) for k in weights}
    return {k: max(float(v), 0.0) / total for k, v in weights.items()}


def compute_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    out = add_indices(df)

    # Block scores. The raw indices are shown, but decision scores apply the
    # physical interpretation used in the report.
    strength_score = minmax_score(out["Flexural strength MPa"], True)
    toughness_score = minmax_score(out["KIC MPa√m"], True)
    fold_index_score = minmax_score(out["M1 foldability σf/E"], True)
    crack_energy_score = minmax_score(out["M2 crack resistance KIC²/E"], True)
    flaw_ratio_score = minmax_score(out["M3 flaw tolerance KIC/σf"], True)
    thermal_score = minmax_score(out["M4 thermal resistance 1/(Eα)"], True)

    # Sustainability subscore: lower price, CO2, embodied energy, water and
    # critical-element risk are better; recycled content is positive but not
    # allowed to dominate functional reliability.
    cost_sustainability_score = (
        0.22 * minmax_score(out["Price EUR/kg"], False)
        + 0.26 * minmax_score(out["CO2 kg/kg"], False)
        + 0.26 * minmax_score(out["Embodied energy MJ/kg"], False)
        + 0.12 * minmax_score(out["Water L/kg"], False)
        + 0.09 * minmax_score(out["Critical element risk"], False)
        + 0.05 * minmax_score(out["Recycle fraction"], True)
    )

    out["Foldability score"] = 0.65 * fold_index_score + 0.35 * strength_score
    out["Crack resistance score"] = (
        0.45 * toughness_score + 0.35 * strength_score + 0.20 * crack_energy_score
    )
    out["Flaw tolerance score"] = (
        0.30 * flaw_ratio_score + 0.35 * toughness_score + 0.35 * strength_score
    )
    out["Thermal compatibility score"] = thermal_score
    out["Cost and sustainability score"] = cost_sustainability_score

    nw = normalized_weights(weights)
    out["Weighted score"] = (
        nw["Foldability"] * out["Foldability score"]
        + nw["Crack resistance"] * out["Crack resistance score"]
        + nw["Flaw tolerance"] * out["Flaw tolerance score"]
        + nw["Thermal compatibility"] * out["Thermal compatibility score"]
        + nw["Cost and sustainability"] * out["Cost and sustainability score"]
    )
    out["Rank"] = out["Weighted score"].rank(ascending=False, method="min").astype(int)
    out["Decision"] = np.select(
        [out["Rank"].eq(1), out["Rank"].le(3)],
        ["Selected for Eco Audit", "Reserve / comparator"],
        default="Not selected",
    )
    return out.sort_values("Weighted score", ascending=False)


def show_df(df: pd.DataFrame, height: int | str | None = "auto") -> None:
    if height is None:
        height = "auto"
    st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def format_table(df: pd.DataFrame, cols: list[str], decimals: int = 3) -> pd.DataFrame:
    out = df[cols].reset_index().rename(columns={"index": "Material"})
    for c in out.columns:
        if c != "Material" and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(decimals)
    return out


def sample_material_space(n_per_material: int, seed: int, weights: Dict[str, float]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for material, props in RANGES.items():
        for _ in range(n_per_material):
            row = {"Material": material}
            for key, (lo, hi) in props.items():
                if math.isclose(float(lo), float(hi)):
                    row[key] = lo
                else:
                    row[key] = rng.uniform(lo, hi)
            rows.append(row)
    sampled = pd.DataFrame(rows).set_index("Material")
    scored = compute_scores(sampled, weights).reset_index()
    # Decision class based on the same score, not a lifetime claim.
    scored["ML class target"] = pd.cut(
        scored["Weighted score"],
        bins=[-np.inf, 2.8, 3.7, np.inf],
        labels=["not selected", "reserve", "selected-range"],
    ).astype(str)
    return scored


def train_models(samples: pd.DataFrame):
    features = [
        "E GPa",
        "Flexural strength MPa",
        "KIC MPa√m",
        "CTE µstrain/°C",
        "Price EUR/kg",
        "CO2 kg/kg",
        "Embodied energy MJ/kg",
        "Water L/kg",
        "M1 foldability σf/E",
        "M2 crack resistance KIC²/E",
        "M3 flaw tolerance KIC/σf",
        "M4 thermal resistance 1/(Eα)",
    ]
    X = samples[features]
    y_reg = samples["Weighted score"]
    y_cls = samples["ML class target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.25, random_state=7
    )
    reg = RandomForestRegressor(n_estimators=300, random_state=7, min_samples_leaf=3)
    reg.fit(X_train, y_train)
    reg_r2 = r2_score(y_test, reg.predict(X_test))

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_cls, test_size=0.25, random_state=7, stratify=y_cls
    )
    clf = RandomForestClassifier(n_estimators=300, random_state=7, min_samples_leaf=3)
    clf.fit(Xc_train, yc_train)
    clf_acc = accuracy_score(yc_test, clf.predict(Xc_test))

    importances = pd.DataFrame(
        {
            "Feature": features,
            "Importance": reg.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    return reg, clf, reg_r2, clf_acc, importances

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("UTG Material Selection and Machine-Learning Ranking Dashboard")
st.caption(
    "Computational support for foldable-phone ultra-thin glass selection using "
    "Granta/Ashby indices, weighted decision scoring, and ML-based sensitivity analysis."
)

with st.sidebar:
    st.header("Decision weights")
    st.write("Weights are normalized automatically for the final weighted score.")
    weights = {
        "Foldability": st.slider("Elastic foldability", 0.0, 50.0, DEFAULT_WEIGHTS["Foldability"], 1.0),
        "Crack resistance": st.slider("Crack resistance", 0.0, 50.0, DEFAULT_WEIGHTS["Crack resistance"], 1.0),
        "Flaw tolerance": st.slider("Flaw tolerance", 0.0, 50.0, DEFAULT_WEIGHTS["Flaw tolerance"], 1.0),
        "Thermal compatibility": st.slider("Thermal compatibility", 0.0, 50.0, DEFAULT_WEIGHTS["Thermal compatibility"], 1.0),
        "Cost and sustainability": st.slider("Cost and sustainability", 0.0, 50.0, DEFAULT_WEIGHTS["Cost and sustainability"], 1.0),
    }
    normalized = normalized_weights(weights)
    st.write("Normalized weights used internally:")
    st.json({k: f"{100*v:.1f}%" for k, v in normalized.items()})

base_df = midpoint_dataframe()
ranked = compute_scores(base_df, weights)

winner = ranked.index[0]
st.success(f"Selected material with current weighting: {winner}")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Top candidate", winner)
kpi2.metric("Top weighted score", f"{ranked.iloc[0]['Weighted score']:.2f} / 5")
kpi3.metric("Shortlisted materials", len(ranked))
kpi4.metric("Core indices", "4")

page = st.sidebar.radio(
    "Dashboard page",
    [
        "Decision-making matrix",
        "Ashby charts",
        "Raw indices",
        "ML sensitivity",
        "Validation logic",
    ],
)

if page == "Decision-making matrix":
    st.subheader("Weighted decision matrix")
    st.write(
        "This page is the report decision-making matrix. It combines the new index logic with "
        "the report weights. The new weights sum to 100%, so the normalized values used internally match the report values. "
        "The raw flaw-tolerance ratio is shown, but the final flaw score is strength-gated so that "
        "a weak material is not overranked only because a lower flexural strength inflates KIC/σf."
    )

    weight_table = pd.DataFrame({
        "Selection block": list(DEFAULT_WEIGHTS.keys()),
        "Report weight": [f"{DEFAULT_WEIGHTS[k]:.0f}%" for k in DEFAULT_WEIGHTS],
        "Normalized weight used in app": [f"{100*normalized[k]:.1f}%" for k in DEFAULT_WEIGHTS],
        "Role": [
            "Primary resistance to repeated bending",
            "Energy-based crack resistance using KIC²/E and strength/toughness gating",
            "Defect tolerance using KIC/σf, but not allowed to reward weak glass alone",
            "Thermal-mismatch reduction using 1/(Eα)",
            "Supporting feasibility using price, CO₂, energy, water, critical elements and recycling",
        ],
    })
    show_df(weight_table)

    st.markdown(
        "**Weighted score formula:** "
        "`0.30(Foldability) + 0.20(Crack resistance) + 0.20(Flaw tolerance) + "
        "0.20(Thermal compatibility) + 0.10(Cost and sustainability)`"
    )
    cols = [
        "Rank",
        "Decision",
        "Weighted score",
        "Foldability score",
        "Crack resistance score",
        "Flaw tolerance score",
        "Thermal compatibility score",
        "Cost and sustainability score",
    ]
    show_df(format_table(ranked, cols, 3))

    fig = px.bar(
        ranked.reset_index(),
        x="Material",
        y="Weighted score",
        color="Decision",
        text=ranked["Weighted score"].round(2),
        title="Final material ranking",
    )
    fig.update_layout(yaxis_range=[0, 5.2], xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Score decomposition")
    decomp = ranked[
        [
            "Foldability score",
            "Crack resistance score",
            "Flaw tolerance score",
            "Thermal compatibility score",
            "Cost and sustainability score",
        ]
    ].reset_index()
    long = decomp.melt(id_vars="Material", var_name="Block", value_name="Score")
    fig2 = px.bar(long, x="Material", y="Score", color="Block", barmode="group")
    fig2.update_layout(yaxis_range=[0, 5.2], xaxis_tickangle=-20)
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Ashby charts":
    st.subheader("Ashby-style charts")
    chart_df = ranked.reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            chart_df,
            x="E GPa",
            y="Flexural strength MPa",
            color="Material",
            size="M1 foldability σf/E",
            title="Foldability: flexural strength vs Young's modulus",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            chart_df,
            x="Flexural strength MPa",
            y="KIC MPa√m",
            color="Material",
            size="Crack resistance score",
            title="Fracture toughness vs flexural strength",
        )
        st.plotly_chart(fig, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(
            chart_df,
            x="E GPa",
            y="KIC MPa√m",
            color="Material",
            size="M2 crack resistance KIC²/E",
            title="Crack resistance: fracture toughness vs Young's modulus",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.scatter(
            chart_df,
            x="E GPa",
            y="CTE µstrain/°C",
            color="Material",
            size="M4 thermal resistance 1/(Eα)",
            title="Thermal mismatch: CTE vs Young's modulus",
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Raw indices":
    st.subheader("Calculated indices from midpoint properties")
    index_cols = [
        "E GPa",
        "Flexural strength MPa",
        "KIC MPa√m",
        "CTE µstrain/°C",
        "M1 foldability σf/E",
        "M2 crack resistance KIC²/E",
        "M3 flaw tolerance KIC/σf",
        "M4 thermal resistance 1/(Eα)",
        "Thermal mismatch Eα",
        "Critical flaw size proxy (KIC/σf)²",
    ]
    show_df(format_table(add_indices(base_df).sort_index(), index_cols, 6))

    st.info(
        "M3 = KIC/σf is retained as the report flaw-tolerance diagnostic. In the final matrix, "
        "it is combined with toughness and flexural-strength scores to avoid ranking a weak glass "
        "as safer only because its strength is low."
    )

elif page == "ML sensitivity":
    st.subheader("Machine-learning sensitivity module")
    n = st.slider("Synthetic samples per material", 100, 3000, 800, 100)
    seed = st.number_input("Random seed", value=42, step=1)
    samples = sample_material_space(int(n), int(seed), weights)
    reg, clf, r2, acc, importances = train_models(samples)

    c1, c2 = st.columns(2)
    c1.metric("Random Forest regression R²", f"{r2:.3f}")
    c2.metric("Classification accuracy", f"{acc:.3f}")

    st.write(
        "The ML model is trained on synthetic samples generated inside the Granta property ranges. "
        "It is a surrogate for sensitivity analysis, not a real fold-cycle lifetime predictor."
    )

    fig = px.bar(importances.head(12), x="Importance", y="Feature", orientation="h", title="Feature importance for weighted score")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monte Carlo ranking stability")
    stability = (
        samples.groupby("Material")["Weighted score"]
        .agg(["mean", "std", "min", "max"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    show_df(stability.round(3))
    fig = px.box(samples, x="Material", y="Weighted score", color="Material", title="Score distribution within property ranges")
    fig.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

elif page == "Validation logic":
    st.subheader("Interpretation for validation planning")
    st.markdown(
        """
The dashboard result is a base-material selection. It supports Alumino silicate 1720 because the material gives the strongest overall balance of elastic foldability, flexural strength, fracture toughness, fracture-resistance behavior, cost, embodied energy and production CO₂. The computational result is exact for the implemented equations, normalization rules, property ranges and weights; it is not a substitute for physical product certification.

The remaining candidates are retained as validation comparators:

| Candidate | Role in validation |
|---|---|
| Alumino silicate 1723 | Reserve / close alternative |
| Borosilicate KG33 | Thermal comparator |
| Borosilicate 7251 | Secondary comparator |
| Borosilicate 7760 | Low-modulus comparator |

Final approval of Alumino silicate 1720 still requires chemical-strengthening validation, processed UTG strength testing, edge-quality control, optical inspection, coating and lamination assessment, and cyclic folding reliability testing.
        """
    )

st.divider()
st.caption(
    "Dashboard scope: early-stage material-selection support based on Granta property ranges, "
    "Ashby-style indices and decision-matrix weighting. Physical validation remains required."
)
