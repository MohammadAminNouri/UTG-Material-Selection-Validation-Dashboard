from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# UTG Material Selection & Validation Dashboard
# Foldable phone ultra-thin glass report companion
# ============================================================

st.set_page_config(
    page_title="UTG Material Selection & Validation Dashboard",
    page_icon="📱",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


# -----------------------------
# Default project data
# -----------------------------
def default_candidate_data() -> pd.DataFrame:
    """Midpoint values from the written report comparison table."""
    data = [
        {
            "Material": "Alumino silicate 1720",
            "rho_kg_m3": 2515,
            "E_GPa": 86.95,
            "sigma_f_MPa": 54.5,
            "KIC_MPa_sqrt_m": 0.72,
            "H_HV": 500.5,
            "alpha_microstrain_C": 4.195,
            "thermal_shock_C": 115,
            "price_EUR_kg": 1.35,
            "CO2_kg_kg": 0.94,
            "embodied_energy_MJ_kg": 13.95,
            "water_L_kg": 21.15,
            "application_relevance": 3.5,
            "utg_validation_score": 4.0,
            "report_score": 4.55,
        },
        {
            "Material": "Alumino silicate 1723",
            "rho_kg_m3": 2635,
            "E_GPa": 86.0,
            "sigma_f_MPa": 51.35,
            "KIC_MPa_sqrt_m": 0.71,
            "H_HV": 501.0,
            "alpha_microstrain_C": 4.595,
            "thermal_shock_C": 100.0,
            "price_EUR_kg": 1.35,
            "CO2_kg_kg": 1.02,
            "embodied_energy_MJ_kg": 15.55,
            "water_L_kg": 23.4,
            "application_relevance": 5.0,
            "utg_validation_score": 5.0,
            "report_score": 4.215,
        },
        {
            "Material": "Borosilicate 7251",
            "rho_kg_m3": 2255,
            "E_GPa": 64.0,
            "sigma_f_MPa": 39.45,
            "KIC_MPa_sqrt_m": 0.615,
            "H_HV": 450.0,
            "alpha_microstrain_C": 3.645,
            "thermal_shock_C": 130.0,
            "price_EUR_kg": 4.55,
            "CO2_kg_kg": 1.33,
            "embodied_energy_MJ_kg": 21.5,
            "water_L_kg": 10.18,
            "application_relevance": 2.5,
            "utg_validation_score": 2.0,
            "report_score": 3.815,
        },
        {
            "Material": "Borosilicate 7760",
            "rho_kg_m3": 2235,
            "E_GPa": 61.95,
            "sigma_f_MPa": 35.5,
            "KIC_MPa_sqrt_m": 0.605,
            "H_HV": 442.5,
            "alpha_microstrain_C": 3.395,
            "thermal_shock_C": 130.0,
            "price_EUR_kg": 5.70,
            "CO2_kg_kg": 1.39,
            "embodied_energy_MJ_kg": 22.5,
            "water_L_kg": 12.1,
            "application_relevance": 2.5,
            "utg_validation_score": 2.0,
            "report_score": 3.39,
        },
        {
            "Material": "Borosilicate KG33",
            "rho_kg_m3": 2225,
            "E_GPa": 63.0,
            "sigma_f_MPa": 39.15,
            "KIC_MPa_sqrt_m": 0.61,
            "H_HV": 439.5,
            "alpha_microstrain_C": 3.195,
            "thermal_shock_C": 149.5,
            "price_EUR_kg": 4.55,
            "CO2_kg_kg": 1.35,
            "embodied_energy_MJ_kg": 21.85,
            "water_L_kg": 10.75,
            "application_relevance": 2.0,
            "utg_validation_score": 2.5,
            "report_score": 3.965,
        },
    ]
    return pd.DataFrame(data)


def default_validation_plan() -> pd.DataFrame:
    rows = [
        {
            "Validation item": "Preliminary ANSYS folding simulation",
            "Purpose": "Identify critical stress location during folding.",
            "Status": "Done",
            "Report interpretation": "Maximum tensile stress occurs near the fold/hinge region; numerical support only, not final pass/fail proof.",
        },
        {
            "Validation item": "Chemical strengthening validation",
            "Purpose": "Confirm surface compressive stress and depth of layer after KNO3 ion exchange.",
            "Status": "Required",
            "Report interpretation": "Needed because bulk glass strength is not enough for foldable UTG reliability.",
        },
        {
            "Validation item": "Static bend-radius test",
            "Purpose": "Confirm minimum bend radius without fracture.",
            "Status": "Required",
            "Report interpretation": "Directly validates the foldability requirement used in the Ashby selection.",
        },
        {
            "Validation item": "Cyclic folding fatigue test",
            "Purpose": "Confirm repeated opening/closing durability.",
            "Status": "Required",
            "Report interpretation": "Needed because real failure may occur after repeated cycles, not only one static fold.",
        },
        {
            "Validation item": "Edge-strength inspection",
            "Purpose": "Check cracks and flaws from cutting, grinding, polishing, and handling.",
            "Status": "Required",
            "Report interpretation": "Edges are high-risk flaw-initiation sites in brittle UTG.",
        },
        {
            "Validation item": "Scratch / hardness / pen-drop test",
            "Purpose": "Assess daily touch, dust, fingernail, stylus, and local indentation damage.",
            "Status": "Required",
            "Report interpretation": "Surface durability is essential because the UTG is the user-facing cover layer.",
        },
        {
            "Validation item": "Optical inspection",
            "Purpose": "Measure haze, distortion, transmittance, and surface defects after processing.",
            "Status": "Required",
            "Report interpretation": "The selected material must remain display-quality after forming, strengthening, coating, and lamination.",
        },
        {
            "Validation item": "Lamination / stack test",
            "Purpose": "Check adhesive effects, neutral axis position, residual stress, and delamination risk.",
            "Status": "Required",
            "Report interpretation": "Real display-stack behaviour cannot be proven by bulk material data alone.",
        },
        {
            "Validation item": "Manufacturing yield trial",
            "Purpose": "Check repeatability, defect rate, and process feasibility.",
            "Status": "Future work",
            "Report interpretation": "Needed before industrial approval.",
        },
    ]
    return pd.DataFrame(rows)


def default_weight_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Material": "Alumino silicate 1720",
                "Foldability": 5.0,
                "Fracture": 5.0,
                "Surface": 5.0,
                "Thermal": 2.5,
                "Optical_application": 3.5,
                "Cost_sustainability": 5.0,
                "UTG_validation": 4.0,
            },
            {
                "Material": "Alumino silicate 1723",
                "Foldability": 4.0,
                "Fracture": 4.5,
                "Surface": 4.8,
                "Thermal": 2.0,
                "Optical_application": 5.0,
                "Cost_sustainability": 4.2,
                "UTG_validation": 5.0,
            },
            {
                "Material": "Borosilicate KG33",
                "Foldability": 4.8,
                "Fracture": 4.2,
                "Surface": 3.8,
                "Thermal": 5.0,
                "Optical_application": 2.0,
                "Cost_sustainability": 3.2,
                "UTG_validation": 2.5,
            },
            {
                "Material": "Borosilicate 7251",
                "Foldability": 4.2,
                "Fracture": 4.0,
                "Surface": 4.5,
                "Thermal": 4.1,
                "Optical_application": 2.5,
                "Cost_sustainability": 3.3,
                "UTG_validation": 2.0,
            },
            {
                "Material": "Borosilicate 7760",
                "Foldability": 3.0,
                "Fracture": 3.8,
                "Surface": 4.0,
                "Thermal": 4.6,
                "Optical_application": 2.5,
                "Cost_sustainability": 2.8,
                "UTG_validation": 2.0,
            },
        ]
    )


# -----------------------------
# Robust data loading
# -----------------------------
@st.cache_data
def load_csv_or_default(filename: str, default_df: pd.DataFrame) -> pd.DataFrame:
    path = DATA_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return default_df.copy()
    return default_df.copy()


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = [
        "rho_kg_m3",
        "E_GPa",
        "sigma_f_MPa",
        "KIC_MPa_sqrt_m",
        "H_HV",
        "alpha_microstrain_C",
    ]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["bending_index_sigmaf_over_E"] = df["sigma_f_MPa"] / df["E_GPa"]
    df["fracture_index_KIC_over_E"] = df["KIC_MPa_sqrt_m"] / df["E_GPa"]
    df["fracture_energy_index_KIC2_over_E"] = (df["KIC_MPa_sqrt_m"] ** 2) / df["E_GPa"]
    df["hardness_density_index_H_over_rho"] = df["H_HV"] / df["rho_kg_m3"]
    df["thermal_mismatch_index_alphaE"] = df["alpha_microstrain_C"] * df["E_GPa"]

    return df


def bending_stress_mpa(E_GPa: float, thickness_um: float, bend_interval_mm: float) -> float:
    """
    Two-point bending stress approximation used as a validation-support calculation.

    sigma = 1.198 * E * t / (D - t)

    E is converted to MPa.
    t and D are in mm.
    Result is MPa.
    """
    t_mm = thickness_um / 1000.0
    D_mm = bend_interval_mm
    if D_mm <= t_mm:
        return float("nan")
    E_MPa = E_GPa * 1000.0
    return 1.198 * E_MPa * t_mm / (D_mm - t_mm)


def weighted_score_matrix(scores_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    df = scores_df.copy()
    total_weight = sum(weights.values())
    if total_weight <= 0:
        total_weight = 1.0

    score = np.zeros(len(df))
    for col, weight in weights.items():
        score += df[col].astype(float).to_numpy() * (weight / total_weight)

    df["Weighted score"] = score
    return df.sort_values("Weighted score", ascending=False)


def safe_text_position(fig):
    """
    Avoid Plotly errors by using trace-specific label placement.
    Scatter traces accept 'top center'.
    Bar traces accept 'outside'.
    """
    for trace in fig.data:
        if trace.type == "scatter":
            trace.update(textposition="top center")
        elif trace.type == "bar":
            trace.update(textposition="outside")
    return fig


# -----------------------------
# Load data
# -----------------------------
candidates = add_indices(load_csv_or_default("candidate_materials.csv", default_candidate_data()))
validation_plan = load_csv_or_default("validation_plan.csv", default_validation_plan())
score_blocks = load_csv_or_default("weighted_matrix_report_scores.csv", default_weight_scores())

default_weights = {
    "Foldability": 25.0,
    "Fracture": 25.0,
    "Surface": 15.0,
    "Thermal": 10.0,
    "Optical_application": 10.0,
    "Cost_sustainability": 10.0,
    "UTG_validation": 5.0,
}

report_ranking = weighted_score_matrix(score_blocks, default_weights)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("UTG Dashboard")
st.sidebar.caption("Foldable phone glass material-selection companion")

page = st.sidebar.radio(
    "Navigation",
    [
        "0. Home",
        "1. Candidate data",
        "2. Ashby index explorer",
        "3. Weighted decision matrix",
        "4. Bending stress check",
        "5. Eco Audit sensitivity",
        "6. ML decision support",
        "7. Validation plan",
        "8. Report-ready wording",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Selected material in report:**")
st.sidebar.success("Alumino silicate 1720")


# ============================================================
# Page 0
# ============================================================
if page == "0. Home":
    st.title("UTG Material Selection & Validation Dashboard")
    st.markdown(
        """
        This dashboard is a reproducible digital companion for the written report on
        **foldable phone ultra-thin glass (UTG)** material selection.

        It supports the report through:
        - Granta/Ashby-style candidate comparison
        - performance-index calculation
        - weighted decision matrix reproduction
        - bending-stress validation support
        - Eco Audit sensitivity checks
        - ML-assisted ranking as a decision-support tool
        - validation-plan tracking
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Initial database", "3895 materials")
    c2.metric("After hard filters", "14 candidates")
    c3.metric("Final shortlist", "5 glasses")
    c4.metric("Selected", "Alumino silicate 1720")

    st.subheader("Report ranking reproduced by the dashboard")
    fig = px.bar(
        report_ranking,
        x="Material",
        y="Weighted score",
        text="Weighted score",
        title="Weighted decision matrix result",
    )
    fig.update_traces(texttemplate="%{text:.2f}")
    fig = safe_text_position(fig)
    fig.update_layout(xaxis_tickangle=-25, yaxis_range=[0, 5.2])
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The dashboard supports validation planning. It does not replace physical UTG validation, "
        "chemical strengthening tests, cyclic folding tests, optical inspection, or manufacturing trials."
    )


# ============================================================
# Page 1
# ============================================================
elif page == "1. Candidate data":
    st.title("Candidate material database")

    st.markdown(
        """
        The table uses midpoint values from the report's final comparison of the five shortlisted materials.
        These values are used for transparent recalculation of Ashby indices and ranking logic.
        """
    )

    display_cols = [
        "Material",
        "rho_kg_m3",
        "E_GPa",
        "sigma_f_MPa",
        "KIC_MPa_sqrt_m",
        "H_HV",
        "alpha_microstrain_C",
        "thermal_shock_C",
        "price_EUR_kg",
        "CO2_kg_kg",
        "embodied_energy_MJ_kg",
        "water_L_kg",
    ]
    st.dataframe(candidates[display_cols], use_container_width=True, hide_index=True)

    st.subheader("Calculated indices")
    index_cols = [
        "Material",
        "bending_index_sigmaf_over_E",
        "fracture_index_KIC_over_E",
        "fracture_energy_index_KIC2_over_E",
        "hardness_density_index_H_over_rho",
        "thermal_mismatch_index_alphaE",
    ]
    st.dataframe(candidates[index_cols], use_container_width=True, hide_index=True)


# ============================================================
# Page 2
# ============================================================
elif page == "2. Ashby index explorer":
    st.title("Ashby index explorer")

    chart = st.selectbox(
        "Choose chart",
        [
            "Flexural strength vs Young's modulus",
            "Fracture toughness vs Young's modulus",
            "Fracture toughness vs flexural strength",
            "Hardness vs density",
            "Thermal expansion vs Young's modulus",
        ],
    )

    if chart == "Flexural strength vs Young's modulus":
        st.markdown("Index used: **σf / E**. Higher is better for elastic foldability.")
        fig = px.scatter(
            candidates,
            x="E_GPa",
            y="sigma_f_MPa",
            text="Material",
            size="bending_index_sigmaf_over_E",
            hover_data=["bending_index_sigmaf_over_E"],
            title="Foldability chart: flexural strength vs Young's modulus",
        )
        fig.update_xaxes(title="Young's modulus, E (GPa)")
        fig.update_yaxes(title="Flexural strength, σf (MPa)")

    elif chart == "Fracture toughness vs Young's modulus":
        st.markdown("Indices used: **KIC / E** and **KIC² / E**. Higher is better for flaw tolerance.")
        fig = px.scatter(
            candidates,
            x="E_GPa",
            y="KIC_MPa_sqrt_m",
            text="Material",
            size="fracture_index_KIC_over_E",
            hover_data=["fracture_index_KIC_over_E", "fracture_energy_index_KIC2_over_E"],
            title="Flaw-tolerance chart: fracture toughness vs Young's modulus",
        )
        fig.update_xaxes(title="Young's modulus, E (GPa)")
        fig.update_yaxes(title="Fracture toughness, KIC (MPa√m)")

    elif chart == "Fracture toughness vs flexural strength":
        st.markdown("This chart checks whether a material combines strength with crack-growth resistance.")
        fig = px.scatter(
            candidates,
            x="sigma_f_MPa",
            y="KIC_MPa_sqrt_m",
            text="Material",
            size="fracture_energy_index_KIC2_over_E",
            hover_data=["fracture_index_KIC_over_E", "fracture_energy_index_KIC2_over_E"],
            title="Safe-fracture chart: KIC vs flexural strength",
        )
        fig.update_xaxes(title="Flexural strength, σf (MPa)")
        fig.update_yaxes(title="Fracture toughness, KIC (MPa√m)")

    elif chart == "Hardness vs density":
        st.markdown("Index used: **H / ρ**. Higher is better for surface durability per mass.")
        fig = px.scatter(
            candidates,
            x="rho_kg_m3",
            y="H_HV",
            text="Material",
            size="hardness_density_index_H_over_rho",
            hover_data=["hardness_density_index_H_over_rho"],
            title="Surface durability chart: hardness vs density",
        )
        fig.update_xaxes(title="Density, ρ (kg/m³)")
        fig.update_yaxes(title="Vickers hardness, H (HV)")

    else:
        st.markdown("Index used: **αE**. Lower is better for thermal-mismatch tendency.")
        fig = px.scatter(
            candidates,
            x="E_GPa",
            y="alpha_microstrain_C",
            text="Material",
            size=np.max(candidates["thermal_mismatch_index_alphaE"]) - candidates["thermal_mismatch_index_alphaE"] + 1,
            hover_data=["thermal_mismatch_index_alphaE"],
            title="Thermal mismatch chart: thermal expansion vs Young's modulus",
        )
        fig.update_xaxes(title="Young's modulus, E (GPa)")
        fig.update_yaxes(title="Thermal expansion, α (microstrain/°C)")

    fig = safe_text_position(fig)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page 3
# ============================================================
elif page == "3. Weighted decision matrix":
    st.title("Weighted decision matrix")

    st.markdown(
        """
        The default weights reproduce the written report. You can change the weights to test
        whether the final selection is sensitive to the decision logic.
        """
    )

    st.subheader("Weights")
    c1, c2, c3 = st.columns(3)
    with c1:
        w_fold = st.slider("Foldability", 0.0, 50.0, 25.0, 1.0)
        w_fracture = st.slider("Fracture / flaw tolerance", 0.0, 50.0, 25.0, 1.0)
        w_surface = st.slider("Surface durability", 0.0, 30.0, 15.0, 1.0)
    with c2:
        w_thermal = st.slider("Thermal compatibility", 0.0, 30.0, 10.0, 1.0)
        w_optical = st.slider("Optical / application relevance", 0.0, 30.0, 10.0, 1.0)
        w_cost = st.slider("Cost / sustainability", 0.0, 30.0, 10.0, 1.0)
    with c3:
        w_utg = st.slider("UTG validation risk", 0.0, 20.0, 5.0, 1.0)

    weights = {
        "Foldability": w_fold,
        "Fracture": w_fracture,
        "Surface": w_surface,
        "Thermal": w_thermal,
        "Optical_application": w_optical,
        "Cost_sustainability": w_cost,
        "UTG_validation": w_utg,
    }

    dynamic_ranking = weighted_score_matrix(score_blocks, weights)

    st.subheader("Ranking")
    st.dataframe(dynamic_ranking, use_container_width=True, hide_index=True)

    winner = dynamic_ranking.iloc[0]["Material"]
    st.success(f"Current winner: {winner}")

    fig = px.bar(
        dynamic_ranking,
        x="Material",
        y="Weighted score",
        text="Weighted score",
        title="Weighted material ranking",
    )
    fig.update_traces(texttemplate="%{text:.2f}")
    fig = safe_text_position(fig)
    fig.update_layout(xaxis_tickangle=-25, yaxis_range=[0, 5.2])
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Interpretation: this matrix supports the report conclusion only as a structured decision method. "
        "Final approval still requires physical UTG validation."
    )


# ============================================================
# Page 4
# ============================================================
elif page == "4. Bending stress check":
    st.title("Bending stress and validation-support check")

    material = st.selectbox("Choose material", candidates["Material"].tolist())
    row = candidates.loc[candidates["Material"] == material].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        E = st.number_input("Young's modulus, E (GPa)", value=float(row["E_GPa"]), min_value=1.0)
    with c2:
        thickness_um = st.slider("UTG thickness (μm)", 10.0, 150.0, 30.0, 1.0)
    with c3:
        bend_interval_mm = st.slider("Two-point bend interval, D (mm)", 1.0, 10.0, 3.0, 0.1)
    with c4:
        safety_factor = st.slider("Safety factor", 1.0, 5.0, 2.0, 0.1)

    sigma_b = bending_stress_mpa(E, thickness_um, bend_interval_mm)
    required_strength = sigma_b * safety_factor

    st.subheader("Stress result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated bending stress", f"{sigma_b:.1f} MPa")
    m2.metric("Required strength with SF", f"{required_strength:.1f} MPa")
    m3.metric("Base flexural strength", f"{row['sigma_f_MPa']:.1f} MPa")

    st.warning(
        "For UTG, this base-material strength check is intentionally conservative. "
        "The written report should state that the glass must be treated as chemically strengthened, "
        "edge-finished, inspected, and laminated UTG, not as untreated bulk glass."
    )

    st.subheader("Chemical strengthening support check")
    compressive_stress = st.slider(
        "Assumed surface compressive stress after ion exchange (MPa)",
        0.0,
        1200.0,
        865.0,
        5.0,
    )
    net_tensile_demand = max(sigma_b - compressive_stress, 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Surface compression assumption", f"{compressive_stress:.0f} MPa")
    c2.metric("Net tensile demand estimate", f"{net_tensile_demand:.1f} MPa")
    c3.metric("ANSYS tensile stress reference", "+658.76 MPa")

    if sigma_b > compressive_stress:
        st.error(
            "The assumed surface compression does not fully cover the calculated bending stress. "
            "This means chemical strengthening, edge quality, and cyclic testing are critical."
        )
    else:
        st.success(
            "The assumed surface compression is higher than the calculated bending stress. "
            "This supports the material route, but it is not final proof because flaws, fatigue, "
            "lamination, and real stress profiles still require validation."
        )


# ============================================================
# Page 5
# ============================================================
elif page == "5. Eco Audit sensitivity":
    st.title("Eco Audit sensitivity analysis")

    st.markdown(
        """
        Baseline from the report:
        - Material + manufacturing + transport + disposal are very small for the 30 μm UTG layer.
        - Use phase dominates when display electricity is assigned to the UTG/display function.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        power = st.slider("Display power assigned to use phase (W)", 0.1, 3.0, 1.5, 0.1)
    with c2:
        daily_use = st.slider("Daily use (h/day)", 0.5, 10.0, 4.6, 0.1)
    with c3:
        lifetime = st.slider("Product life (years)", 1.0, 6.0, 3.0, 0.5)

    baseline_power = 1.5
    baseline_daily = 4.6
    baseline_life = 3.0

    baseline_use_energy = 76.1
    baseline_use_co2 = 2.49
    non_use_energy = 0.0241 + 0.0205 + 0.169 + 0.00031
    non_use_co2 = 0.00158 + 0.00163 + 0.012 + 0.0000217

    scale = (power * daily_use * lifetime) / (baseline_power * baseline_daily * baseline_life)
    use_energy = baseline_use_energy * scale
    use_co2 = baseline_use_co2 * scale

    total_energy = use_energy + non_use_energy
    total_co2 = use_co2 + non_use_co2

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total energy", f"{total_energy:.2f} MJ")
    m2.metric("Use energy share", f"{100 * use_energy / total_energy:.1f}%")
    m3.metric("Total CO₂-eq", f"{total_co2:.2f} kg")
    m4.metric("Use CO₂ share", f"{100 * use_co2 / total_co2:.1f}%")

    phase_df = pd.DataFrame(
        {
            "Phase": ["Material", "Manufacture", "Transport", "Use", "Disposal"],
            "Energy_MJ": [0.0241, 0.0205, 0.169, use_energy, 0.00031],
            "CO2_kg": [0.00158, 0.00163, 0.012, use_co2, 0.0000217],
        }
    )

    fig_energy = px.bar(phase_df, x="Phase", y="Energy_MJ", text="Energy_MJ", title="Energy by life-cycle phase")
    fig_energy.update_traces(texttemplate="%{text:.3g}")
    fig_energy = safe_text_position(fig_energy)
    st.plotly_chart(fig_energy, use_container_width=True)

    fig_co2 = px.bar(phase_df, x="Phase", y="CO2_kg", text="CO2_kg", title="CO₂-eq by life-cycle phase")
    fig_co2.update_traces(texttemplate="%{text:.3g}")
    fig_co2 = safe_text_position(fig_co2)
    st.plotly_chart(fig_co2, use_container_width=True)


# ============================================================
# Page 6
# ============================================================
elif page == "6. ML decision support":
    st.title("ML decision-support check")

    st.info(
        "This ML module is a surrogate decision-support tool. It learns the report's weighted score pattern "
        "from the candidate-property table. It is not a real fold-cycle lifetime predictor."
    )

    feature_cols = [
        "E_GPa",
        "sigma_f_MPa",
        "KIC_MPa_sqrt_m",
        "H_HV",
        "rho_kg_m3",
        "alpha_microstrain_C",
        "thermal_shock_C",
        "price_EUR_kg",
        "CO2_kg_kg",
        "embodied_energy_MJ_kg",
        "water_L_kg",
        "application_relevance",
        "utg_validation_score",
        "bending_index_sigmaf_over_E",
        "fracture_index_KIC_over_E",
        "fracture_energy_index_KIC2_over_E",
        "hardness_density_index_H_over_rho",
        "thermal_mismatch_index_alphaE",
    ]

    ml_df = candidates.copy()
    X = ml_df[feature_cols].astype(float)
    y = ml_df["report_score"].astype(float)

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        min_samples_leaf=1,
                    ),
                ),
            ]
        )
        model.fit(X, y)
        ml_df["ML predicted score"] = model.predict(X)
        method = "RandomForestRegressor surrogate"
    except Exception:
        normalized = X.copy()
        for col in normalized.columns:
            span = normalized[col].max() - normalized[col].min()
            if span == 0:
                normalized[col] = 0.0
            else:
                normalized[col] = (normalized[col] - normalized[col].min()) / span

        lower_is_better = [
            "E_GPa",
            "rho_kg_m3",
            "alpha_microstrain_C",
            "price_EUR_kg",
            "CO2_kg_kg",
            "embodied_energy_MJ_kg",
            "water_L_kg",
            "thermal_mismatch_index_alphaE",
        ]
        for col in lower_is_better:
            normalized[col] = 1.0 - normalized[col]

        ml_df["ML predicted score"] = 1.0 + 4.0 * normalized.mean(axis=1)
        method = "Fallback normalized scoring model"

    ml_ranking = ml_df[["Material", "report_score", "ML predicted score"]].sort_values(
        "ML predicted score", ascending=False
    )

    st.subheader("ML check result")
    st.caption(f"Method used: {method}")
    st.dataframe(ml_ranking, use_container_width=True, hide_index=True)

    fig = px.bar(
        ml_ranking,
        x="Material",
        y="ML predicted score",
        text="ML predicted score",
        title="ML surrogate ranking",
    )
    fig.update_traces(texttemplate="%{text:.2f}")
    fig = safe_text_position(fig)
    fig.update_layout(xaxis_tickangle=-25, yaxis_range=[0, 5.2])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Correct report interpretation:**  
        The ML result is used only to check whether the numerical decision-support logic agrees with
        the weighted matrix. Since the dataset contains only five shortlisted materials, the model must
        not be presented as an experimental predictor of fatigue life, crack growth, or real product durability.
        """
    )


# ============================================================
# Page 7
# ============================================================
elif page == "7. Validation plan":
    st.title("Validation plan")

    st.markdown(
        """
        The validation plan separates what has already been done from what must still be proven.
        This is important because Granta, Ashby charts, FEA, and ML support material selection,
        but they do not replace physical UTG qualification.
        """
    )

    st.dataframe(validation_plan, use_container_width=True, hide_index=True)

    status_counts = validation_plan["Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    fig = px.bar(status_counts, x="Status", y="Count", text="Count", title="Validation status")
    fig = safe_text_position(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "Final approval requires physical validation: chemical strengthening response, processed strength, "
        "edge quality, scratch/pen-drop resistance, optical quality, lamination behaviour, and cyclic folding."
    )


# ============================================================
# Page 8
# ============================================================
elif page == "8. Report-ready wording":
    st.title("Report-ready wording")

    st.markdown(
        """
        Copy this paragraph into the validation-plan or digital-methodology part of the report.
        """
    )

    report_text = """
A Streamlit dashboard was developed as a digital validation-support tool for the foldable phone ultra-thin glass material-selection study. The dashboard reproduces the candidate-material database, Ashby performance indices, weighted decision matrix, bending-stress check, Eco Audit sensitivity analysis, and validation-plan checklist. The purpose of the tool is to make the Granta/Ashby selection logic more transparent and reproducible.

The dashboard also includes an ML-assisted decision-support module. This module is not used as an experimental predictor of fold-cycle lifetime or fracture behaviour. Instead, it is used as a surrogate ranking and sensitivity tool based on the calculated material indices and the weighted decision matrix. This avoids overclaiming, because the available dataset contains only five shortlisted candidate materials.

The digital validation result supports the selection of Alumino silicate 1720 as the strongest overall candidate because it maintains the best balance of elastic foldability, fracture/flaw tolerance, surface durability, cost, sustainability, and UTG relevance. However, the dashboard does not replace final physical validation. Final approval of the material still requires chemical strengthening validation, processed UTG strength measurement, edge-quality inspection, scratch and pen-drop testing, optical inspection, lamination assessment, cyclic folding fatigue testing, and manufacturing-yield evaluation.
""".strip()

    st.text_area("Report text", report_text, height=360)

    st.download_button(
        label="Download paragraph as TXT",
        data=report_text,
        file_name="report_ready_validation_text.txt",
        mime="text/plain",
    )


else:
    st.error("Unknown page selected.")
