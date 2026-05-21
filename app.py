from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# UTG Material Selection & Validation Dashboard
# Stable Streamlit-only version: no Plotly, no sklearn
# ============================================================

st.set_page_config(
    page_title="UTG Material Selection & Validation Dashboard",
    page_icon="📱",
    layout="wide",
)


# ============================================================
# Data
# ============================================================

@st.cache_data
def load_candidates() -> pd.DataFrame:
    data = [
        {
            "Material": "Alumino silicate 1720",
            "Density kg/m3": 2515,
            "Young modulus GPa": 86.95,
            "Flexural strength MPa": 54.5,
            "Fracture toughness MPa√m": 0.72,
            "Hardness HV": 500.5,
            "Thermal expansion microstrain/C": 4.195,
            "Thermal shock C": 115,
            "Price EUR/kg": 1.35,
            "CO2 kg/kg": 0.94,
            "Embodied energy MJ/kg": 13.95,
            "Water L/kg": 21.15,
            "Application relevance": 3.5,
            "UTG validation score": 4.0,
            "Report score": 4.55,
        },
        {
            "Material": "Alumino silicate 1723",
            "Density kg/m3": 2635,
            "Young modulus GPa": 86.0,
            "Flexural strength MPa": 51.35,
            "Fracture toughness MPa√m": 0.71,
            "Hardness HV": 501.0,
            "Thermal expansion microstrain/C": 4.595,
            "Thermal shock C": 100,
            "Price EUR/kg": 1.35,
            "CO2 kg/kg": 1.02,
            "Embodied energy MJ/kg": 15.55,
            "Water L/kg": 23.4,
            "Application relevance": 5.0,
            "UTG validation score": 5.0,
            "Report score": 4.215,
        },
        {
            "Material": "Borosilicate 7251",
            "Density kg/m3": 2255,
            "Young modulus GPa": 64.0,
            "Flexural strength MPa": 39.45,
            "Fracture toughness MPa√m": 0.615,
            "Hardness HV": 450.0,
            "Thermal expansion microstrain/C": 3.645,
            "Thermal shock C": 130,
            "Price EUR/kg": 4.55,
            "CO2 kg/kg": 1.33,
            "Embodied energy MJ/kg": 21.5,
            "Water L/kg": 10.18,
            "Application relevance": 2.5,
            "UTG validation score": 2.0,
            "Report score": 3.815,
        },
        {
            "Material": "Borosilicate 7760",
            "Density kg/m3": 2235,
            "Young modulus GPa": 61.95,
            "Flexural strength MPa": 35.5,
            "Fracture toughness MPa√m": 0.605,
            "Hardness HV": 442.5,
            "Thermal expansion microstrain/C": 3.395,
            "Thermal shock C": 130,
            "Price EUR/kg": 5.70,
            "CO2 kg/kg": 1.39,
            "Embodied energy MJ/kg": 22.5,
            "Water L/kg": 12.1,
            "Application relevance": 2.5,
            "UTG validation score": 2.0,
            "Report score": 3.39,
        },
        {
            "Material": "Borosilicate KG33",
            "Density kg/m3": 2225,
            "Young modulus GPa": 63.0,
            "Flexural strength MPa": 39.15,
            "Fracture toughness MPa√m": 0.61,
            "Hardness HV": 439.5,
            "Thermal expansion microstrain/C": 3.195,
            "Thermal shock C": 149.5,
            "Price EUR/kg": 4.55,
            "CO2 kg/kg": 1.35,
            "Embodied energy MJ/kg": 21.85,
            "Water L/kg": 10.75,
            "Application relevance": 2.0,
            "UTG validation score": 2.5,
            "Report score": 3.965,
        },
    ]

    df = pd.DataFrame(data)

    df["Bending index σf/E"] = df["Flexural strength MPa"] / df["Young modulus GPa"]
    df["Fracture index KIC/E"] = df["Fracture toughness MPa√m"] / df["Young modulus GPa"]
    df["Fracture energy index KIC²/E"] = (
        df["Fracture toughness MPa√m"] ** 2
    ) / df["Young modulus GPa"]
    df["Hardness-density index H/ρ"] = df["Hardness HV"] / df["Density kg/m3"]
    df["Thermal mismatch index αE"] = (
        df["Thermal expansion microstrain/C"] * df["Young modulus GPa"]
    )

    return df


@st.cache_data
def load_validation_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Validation item": "Preliminary ANSYS folding simulation",
                "Status": "Done",
                "Purpose": "Identify the critical stress region during folding.",
                "Report meaning": "Supports the fold-line risk argument but is not final proof.",
            },
            {
                "Validation item": "Chemical strengthening validation",
                "Status": "Required",
                "Purpose": "Confirm surface compressive stress and depth of layer.",
                "Report meaning": "Essential because untreated bulk glass strength is not enough.",
            },
            {
                "Validation item": "Static bend-radius test",
                "Status": "Required",
                "Purpose": "Confirm minimum bend radius without fracture.",
                "Report meaning": "Directly validates the foldability requirement.",
            },
            {
                "Validation item": "Cyclic folding fatigue test",
                "Status": "Required",
                "Purpose": "Confirm repeated folding durability.",
                "Report meaning": "Needed because real UTG can fail after repeated cycles.",
            },
            {
                "Validation item": "Edge-strength inspection",
                "Status": "Required",
                "Purpose": "Check flaws from cutting, grinding, polishing, and handling.",
                "Report meaning": "Edges are high-risk crack-initiation regions.",
            },
            {
                "Validation item": "Scratch / hardness / pen-drop test",
                "Status": "Required",
                "Purpose": "Assess touch-surface damage resistance.",
                "Report meaning": "Surface durability is critical for the user-facing cover glass.",
            },
            {
                "Validation item": "Optical inspection",
                "Status": "Required",
                "Purpose": "Measure haze, distortion, transmittance, and defects.",
                "Report meaning": "UTG must remain display-quality after processing.",
            },
            {
                "Validation item": "Lamination / display-stack test",
                "Status": "Required",
                "Purpose": "Check adhesive stress, neutral axis, and delamination risk.",
                "Report meaning": "Bulk material data cannot fully represent the laminated stack.",
            },
            {
                "Validation item": "Manufacturing yield trial",
                "Status": "Future work",
                "Purpose": "Check repeatability, defect rate, and production feasibility.",
                "Report meaning": "Needed before industrial approval.",
            },
        ]
    )


def weighted_ranking(
    df: pd.DataFrame,
    w_fold: float,
    w_fracture: float,
    w_surface: float,
    w_thermal: float,
    w_optical: float,
    w_cost: float,
    w_validation: float,
) -> pd.DataFrame:
    scores = pd.DataFrame()
    scores["Material"] = df["Material"]

    # 1 to 5 normalized scores.
    scores["Foldability"] = normalize_high(df["Bending index σf/E"])
    scores["Fracture"] = 0.5 * normalize_high(df["Fracture index KIC/E"]) + 0.5 * normalize_high(
        df["Fracture energy index KIC²/E"]
    )
    scores["Surface"] = 0.5 * normalize_high(df["Hardness HV"]) + 0.5 * normalize_high(
        df["Hardness-density index H/ρ"]
    )
    scores["Thermal"] = 0.5 * normalize_low(df["Thermal mismatch index αE"]) + 0.5 * normalize_high(
        df["Thermal shock C"]
    )
    scores["Optical/application"] = df["Application relevance"]
    scores["Cost/sustainability"] = (
        0.35 * normalize_low(df["Price EUR/kg"])
        + 0.35 * normalize_low(df["CO2 kg/kg"])
        + 0.30 * normalize_low(df["Embodied energy MJ/kg"])
    )
    scores["UTG validation"] = df["UTG validation score"]

    total_weight = w_fold + w_fracture + w_surface + w_thermal + w_optical + w_cost + w_validation
    if total_weight == 0:
        total_weight = 1

    scores["Dashboard weighted score"] = (
        scores["Foldability"] * w_fold
        + scores["Fracture"] * w_fracture
        + scores["Surface"] * w_surface
        + scores["Thermal"] * w_thermal
        + scores["Optical/application"] * w_optical
        + scores["Cost/sustainability"] * w_cost
        + scores["UTG validation"] * w_validation
    ) / total_weight

    scores = scores.sort_values("Dashboard weighted score", ascending=False)
    return scores


def normalize_high(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v:
        return pd.Series([3.0] * len(series), index=series.index)
    return 1 + 4 * (series - min_v) / (max_v - min_v)


def normalize_low(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v:
        return pd.Series([3.0] * len(series), index=series.index)
    return 1 + 4 * (max_v - series) / (max_v - min_v)


def bending_stress_mpa(E_GPa: float, thickness_um: float, bend_interval_mm: float) -> float:
    t_mm = thickness_um / 1000
    if bend_interval_mm <= t_mm:
        return float("nan")
    return 1.198 * (E_GPa * 1000) * t_mm / (bend_interval_mm - t_mm)


def show_table(df: pd.DataFrame):
    st.dataframe(df, width="stretch", hide_index=True)


df = load_candidates()
validation_plan = load_validation_plan()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("UTG Dashboard")
page = st.sidebar.radio(
    "Menu",
    [
        "Home",
        "Candidate data",
        "Ashby indices",
        "Weighted decision matrix",
        "Bending stress check",
        "Eco Audit sensitivity",
        "ML-style decision support",
        "Validation plan",
        "Report wording",
    ],
)

st.sidebar.success("Selected material: Alumino silicate 1720")


# ============================================================
# Pages
# ============================================================

if page == "Home":
    st.title("UTG Material Selection & Validation Dashboard")

    st.markdown(
        """
        This dashboard supports the written report on **foldable phone ultra-thin glass (UTG)**.
        It reproduces the material-selection logic using Ashby indices, weighted ranking,
        bending-stress checking, Eco Audit sensitivity, and validation planning.

        The dashboard is a **report-support and reproducibility tool**. It is not a replacement
        for experimental validation.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Initial Granta space", "3895")
    c2.metric("After hard filters", "14")
    c3.metric("Final shortlist", "5")
    c4.metric("Selected", "1720")

    st.subheader("Report weighted scores")
    report_scores = df[["Material", "Report score"]].sort_values("Report score", ascending=False)
    show_table(report_scores)
    st.bar_chart(report_scores.set_index("Material"))

    st.info(
        "Main conclusion: Alumino silicate 1720 is selected because it gives the best overall balance, "
        "not because it wins every individual property."
    )


elif page == "Candidate data":
    st.title("Candidate material data")

    st.markdown("Midpoint values from the five shortlisted candidates are used for the dashboard.")
    show_table(df)

    st.download_button(
        "Download candidate table as CSV",
        data=df.to_csv(index=False),
        file_name="utg_candidate_materials.csv",
        mime="text/csv",
    )


elif page == "Ashby indices":
    st.title("Ashby performance indices")

    index_cols = [
        "Material",
        "Bending index σf/E",
        "Fracture index KIC/E",
        "Fracture energy index KIC²/E",
        "Hardness-density index H/ρ",
        "Thermal mismatch index αE",
    ]

    show_table(df[index_cols])

    st.subheader("Index comparison charts")

    selected_index = st.selectbox(
        "Choose index",
        [
            "Bending index σf/E",
            "Fracture index KIC/E",
            "Fracture energy index KIC²/E",
            "Hardness-density index H/ρ",
            "Thermal mismatch index αE",
        ],
    )

    chart_df = df[["Material", selected_index]].set_index("Material")
    st.bar_chart(chart_df)

    if selected_index == "Thermal mismatch index αE":
        st.warning("For αE, lower is better because it indicates lower thermal-mismatch tendency.")
    else:
        st.success("For this index, higher is better.")


elif page == "Weighted decision matrix":
    st.title("Weighted decision matrix")

    st.markdown(
        """
        The default weights follow the report logic: foldability and fracture/flaw tolerance
        are the most important criteria.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        w_fold = st.slider("Foldability", 0.0, 50.0, 25.0)
        w_fracture = st.slider("Fracture/flaw tolerance", 0.0, 50.0, 25.0)
        w_surface = st.slider("Surface durability", 0.0, 30.0, 15.0)

    with c2:
        w_thermal = st.slider("Thermal compatibility", 0.0, 30.0, 10.0)
        w_optical = st.slider("Optical/application relevance", 0.0, 30.0, 10.0)
        w_cost = st.slider("Cost/sustainability", 0.0, 30.0, 10.0)

    with c3:
        w_validation = st.slider("UTG validation risk", 0.0, 20.0, 5.0)

    ranking = weighted_ranking(
        df,
        w_fold,
        w_fracture,
        w_surface,
        w_thermal,
        w_optical,
        w_cost,
        w_validation,
    )

    st.subheader("Dynamic ranking")
    show_table(ranking)

    winner = ranking.iloc[0]["Material"]
    st.success(f"Current dashboard winner: {winner}")

    st.bar_chart(ranking.set_index("Material")[["Dashboard weighted score"]])

    st.caption(
        "This is a decision-support ranking. Final material approval still requires physical UTG validation."
    )


elif page == "Bending stress check":
    st.title("Bending stress check")

    material = st.selectbox("Material", df["Material"].tolist())
    row = df[df["Material"] == material].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        E = st.number_input("Young modulus E (GPa)", value=float(row["Young modulus GPa"]))
    with c2:
        thickness_um = st.slider("Thickness (μm)", 10.0, 150.0, 30.0)
    with c3:
        bend_interval_mm = st.slider("Bend interval D (mm)", 1.0, 10.0, 3.0)
    with c4:
        safety_factor = st.slider("Safety factor", 1.0, 5.0, 2.0)

    sigma_b = bending_stress_mpa(E, thickness_um, bend_interval_mm)
    required = sigma_b * safety_factor

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated bending stress", f"{sigma_b:.1f} MPa")
    m2.metric("Required with safety factor", f"{required:.1f} MPa")
    m3.metric("Base flexural strength", f"{row['Flexural strength MPa']:.1f} MPa")

    st.subheader("Chemical strengthening check")
    surface_compression = st.slider(
        "Assumed surface compressive stress after ion exchange (MPa)",
        0.0,
        1200.0,
        865.0,
    )

    net_demand = max(sigma_b - surface_compression, 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Surface compression", f"{surface_compression:.0f} MPa")
    c2.metric("Net tensile demand estimate", f"{net_demand:.1f} MPa")
    c3.metric("ANSYS tensile reference", "+658.76 MPa")

    st.warning(
        "This calculation supports the validation plan only. Real UTG approval must include flaws, "
        "edge condition, ion-exchange stress profile, coating, adhesive layers, and cyclic fatigue."
    )


elif page == "Eco Audit sensitivity":
    st.title("Eco Audit sensitivity")

    st.markdown(
        """
        Baseline report result: use phase dominates when display electricity is assigned to
        the UTG/display function.
        """
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        power = st.slider("Display power (W)", 0.1, 3.0, 1.5)
    with c2:
        daily_use = st.slider("Daily use (h/day)", 0.5, 10.0, 4.6)
    with c3:
        lifetime = st.slider("Lifetime (years)", 1.0, 6.0, 3.0)

    baseline_power = 1.5
    baseline_daily_use = 4.6
    baseline_lifetime = 3.0

    baseline_use_energy = 76.1
    baseline_use_co2 = 2.49

    non_use_energy = 0.0241 + 0.0205 + 0.169 + 0.00031
    non_use_co2 = 0.00158 + 0.00163 + 0.012 + 0.0000217

    scale = (power * daily_use * lifetime) / (
        baseline_power * baseline_daily_use * baseline_lifetime
    )

    use_energy = baseline_use_energy * scale
    use_co2 = baseline_use_co2 * scale

    total_energy = use_energy + non_use_energy
    total_co2 = use_co2 + non_use_co2

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total energy", f"{total_energy:.2f} MJ")
    c2.metric("Use energy share", f"{100 * use_energy / total_energy:.1f}%")
    c3.metric("Total CO₂-eq", f"{total_co2:.2f} kg")
    c4.metric("Use CO₂ share", f"{100 * use_co2 / total_co2:.1f}%")

    phase_df = pd.DataFrame(
        {
            "Phase": ["Material", "Manufacture", "Transport", "Use", "Disposal"],
            "Energy MJ": [0.0241, 0.0205, 0.169, use_energy, 0.00031],
            "CO2 kg": [0.00158, 0.00163, 0.012, use_co2, 0.0000217],
        }
    )

    show_table(phase_df)

    st.subheader("Energy by phase")
    st.bar_chart(phase_df.set_index("Phase")[["Energy MJ"]])

    st.subheader("CO₂-eq by phase")
    st.bar_chart(phase_df.set_index("Phase")[["CO2 kg"]])


elif page == "ML-style decision support":
    st.title("ML-style decision support")

    st.info(
        "This page uses a transparent surrogate scoring model. It is not a real fatigue-life or fracture predictor."
    )

    features = pd.DataFrame()
    features["Material"] = df["Material"]
    features["Foldability score"] = normalize_high(df["Bending index σf/E"])
    features["Fracture score"] = normalize_high(df["Fracture index KIC/E"])
    features["Surface score"] = normalize_high(df["Hardness HV"])
    features["Thermal score"] = normalize_low(df["Thermal mismatch index αE"])
    features["Sustainability score"] = (
        normalize_low(df["CO2 kg/kg"]) + normalize_low(df["Embodied energy MJ/kg"])
    ) / 2
    features["Application score"] = df["Application relevance"]
    features["Validation score"] = df["UTG validation score"]

    features["ML-style suitability score"] = (
        0.25 * features["Foldability score"]
        + 0.25 * features["Fracture score"]
        + 0.15 * features["Surface score"]
        + 0.10 * features["Thermal score"]
        + 0.10 * features["Sustainability score"]
        + 0.10 * features["Application score"]
        + 0.05 * features["Validation score"]
    )

    features = features.sort_values("ML-style suitability score", ascending=False)

    show_table(features)
    st.bar_chart(features.set_index("Material")[["ML-style suitability score"]])

    st.markdown(
        """
        **Report interpretation:**  
        The ML-style result supports the same direction as the written report: Alumino silicate 1720
        remains the strongest balanced candidate. However, this is only a decision-support check.
        It must not be presented as a trained lifetime-prediction model because the dataset is too small
        and does not contain experimental cyclic-folding failure data.
        """
    )


elif page == "Validation plan":
    st.title("Validation plan")

    show_table(validation_plan)

    st.subheader("Validation status count")
    status_count = validation_plan["Status"].value_counts().reset_index()
    status_count.columns = ["Status", "Count"]
    show_table(status_count)
    st.bar_chart(status_count.set_index("Status"))

    st.warning(
        "Only the preliminary FEA simulation is treated as done. All physical qualification tests remain required."
    )


elif page == "Report wording":
    st.title("Report-ready wording")

    text = """
A Streamlit dashboard was developed as a digital validation-support tool for the foldable phone ultra-thin glass material-selection study. The dashboard reproduces the candidate-material database, Ashby performance indices, weighted decision matrix, bending-stress check, Eco Audit sensitivity analysis, and validation-plan checklist. The purpose of the tool is to make the Granta/Ashby selection logic more transparent and reproducible.

The dashboard also includes an ML-style decision-support module. This module is not used as an experimental predictor of fold-cycle lifetime or fracture behaviour. Instead, it is used as a transparent surrogate ranking and sensitivity tool based on the calculated material indices and the weighted decision matrix. This avoids overclaiming, because the available dataset contains only five shortlisted candidate materials and does not include experimental cyclic-folding failure data.

The digital validation result supports the selection of Alumino silicate 1720 as the strongest overall candidate because it maintains the best balance of elastic foldability, fracture/flaw tolerance, surface durability, cost, sustainability, and UTG relevance. However, the dashboard does not replace final physical validation. Final approval of the material still requires chemical strengthening validation, processed UTG strength measurement, edge-quality inspection, scratch and pen-drop testing, optical inspection, lamination assessment, cyclic folding fatigue testing, and manufacturing-yield evaluation.
""".strip()

    st.text_area("Copy this into the report", text, height=360)

    st.download_button(
        "Download report wording",
        data=text,
        file_name="report_ready_validation_text.txt",
        mime="text/plain",
    )
