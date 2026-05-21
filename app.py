import streamlit as st
import pandas as pd
import plotly.express as px
from src.calculations import add_midpoints_and_indices, calculate_data_driven_scores, weighted_score_from_report_blocks, REPORT_WEIGHTS
from src.ml_model import fit_ml_decision_support

st.set_page_config(page_title="UTG Validation Dashboard", page_icon="📱", layout="wide")

@st.cache_data
def load_data():
    materials = pd.read_csv("data/candidate_materials.csv")
    weighted = pd.read_csv("data/weighted_matrix_report_scores.csv")
    eco = pd.read_csv("data/eco_audit_baseline.csv")
    validation = pd.read_csv("data/validation_plan.csv")
    return add_midpoints_and_indices(materials), weighted, eco, validation

materials, weighted, eco, validation = load_data()

st.title("Foldable Phone Glass: Validation and Decision-Support Dashboard")
st.caption("Digital appendix for the written material-selection report. The tool supports the report; it does not replace experimental UTG validation.")

page = st.sidebar.radio(
    "Pages",
    [
        "1. Summary",
        "2. Candidate data and Ashby indices",
        "3. Weighted decision matrix",
        "4. ML decision-support check",
        "5. Bending / strengthening check",
        "6. Eco Audit sensitivity",
        "7. Validation plan",
    ],
)

if page == "1. Summary":
    st.header("Report result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected material", "Alumino silicate 1720")
    c2.metric("Report weighted score", "4.55 / 5")
    c3.metric("Closest reserve", "Alumino silicate 1723")

    st.markdown("""
    **Interpretation:** Alumino silicate 1720 is selected because it gives the strongest overall balance
    of elastic foldability, fracture/flaw tolerance, surface hardness, cost, CO₂, embodied energy, and
    acceptable UTG relevance. The result is not a final product approval: processed ultra-thin glass still
    needs chemical strengthening, edge-quality control, lamination checks, optical validation, and cyclic
    folding tests.
    """)

    st.subheader("Screening workflow")
    st.code("3895 materials → optical/hardness/fracture/thermal filters → 14 candidates → Ashby ranking → 5 candidates → weighted matrix → Alumino silicate 1720", language="text")

elif page == "2. Candidate data and Ashby indices":
    st.header("Candidate data and Ashby indices")
    st.dataframe(materials, use_container_width=True)

    chart = st.selectbox("Chart", [
        "Flexural strength vs Young's modulus",
        "Fracture toughness vs Young's modulus",
        "Hardness vs density",
        "Thermal expansion vs Young's modulus",
        "Index comparison",
    ])

    if chart == "Flexural strength vs Young's modulus":
        fig = px.scatter(materials, x="E_GPa", y="flex_MPa", text="material", size="bending_index_flex_over_E",
                         title="Foldability chart: flexural strength vs Young's modulus")
    elif chart == "Fracture toughness vs Young's modulus":
        fig = px.scatter(materials, x="E_GPa", y="KIC_MPa_sqrtm", text="material", size="fracture_index_KIC_over_E",
                         title="Flaw tolerance chart: KIC vs Young's modulus")
    elif chart == "Hardness vs density":
        fig = px.scatter(materials, x="density_kg_m3", y="HV", text="material", size="hardness_density_index_HV_over_rho",
                         title="Surface durability chart: hardness vs density")
    elif chart == "Thermal expansion vs Young's modulus":
        fig = px.scatter(materials, x="E_GPa", y="alpha_microstrain_C", text="material", size="thermal_mismatch_alphaE",
                         title="Thermal mismatch chart: alpha vs Young's modulus")
    else:
        melt = materials.melt(id_vars="material", value_vars=[
            "bending_index_flex_over_E", "fracture_index_KIC_over_E",
            "fracture_energy_index_KIC2_over_E", "hardness_density_index_HV_over_rho",
            "thermal_mismatch_alphaE"
        ])
        fig = px.bar(melt, x="material", y="value", color="variable", barmode="group", title="Raw index comparison")
   # Plotly trace-specific text placement
# Scatter traces accept values such as "top center".
# Bar traces accept values such as "outside", "inside", "auto", or "none".
for trace in fig.data:
    if trace.type == "scatter":
        trace.update(textposition="top center")
    elif trace.type == "bar":
        trace.update(textposition="outside")

st.plotly_chart(fig, use_container_width=True)

elif page == "3. Weighted decision matrix":
    st.header("Weighted decision matrix")
    st.write("Main report matrix:")
    st.dataframe(weighted, use_container_width=True)

    st.subheader("Adjust weights for sensitivity")
    cols = st.columns(4)
    new_w = {}
    for i, (name, val) in enumerate(REPORT_WEIGHTS.items()):
        new_w[name] = cols[i % 4].slider(name, 0.0, 0.60, float(val), 0.01)
    total = sum(new_w.values())
    if total == 0:
        st.error("At least one weight must be positive.")
    else:
        new_w = {k: v / total for k, v in new_w.items()}
        recalculated = weighted_score_from_report_blocks(weighted.drop(columns=["weighted_score"]), new_w)
        st.dataframe(recalculated, use_container_width=True)
        winner = recalculated.iloc[0]["material"]
        st.success(f"Winner under these weights: {winner}")

elif page == "4. ML decision-support check":
    st.header("ML decision-support check")
    ml_table, diagnostics = fit_ml_decision_support(materials, weighted)
    st.warning(diagnostics["warning"])
    st.metric("Leave-one-out MAE, Ridge surrogate", f"{diagnostics['ridge_leave_one_out_mae']:.3f} score points")
    st.dataframe(ml_table[["material", "weighted_score", "ml_score_ridge", "loo_prediction_ridge", "ml_score_random_forest"]], use_container_width=True)

    fig = px.scatter(ml_table, x="weighted_score", y="ml_score_ridge", text="material",
                     title="ML surrogate score vs report weighted score")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Correct interpretation for the report:** this ML module is a surrogate model that checks the consistency
    of the weighted decision matrix. It is not trained on real cyclic-folding lifetime data and must not be
    described as a final lifetime predictor.
    """)

elif page == "5. Bending / strengthening check":
    st.header("Bending and strengthening check")
    selected = st.selectbox("Material", materials["material"].tolist(), index=0)
    row = materials[materials["material"] == selected].iloc[0]
    c1, c2, c3 = st.columns(3)
    E = c1.number_input("Young's modulus E (GPa)", value=float(row["E_GPa"]), min_value=1.0)
    factor = c2.number_input("Geometry factor used in report (MPa/GPa)", value=12.1, min_value=0.1)
    cs = c3.number_input("Surface compressive stress after strengthening (MPa)", value=864.6, min_value=0.0)

    tensile_stress = factor * E
    fea_tensile = 658.76
    net_after_cs = tensile_stress - cs
    st.metric("Estimated bending stress from report relation", f"{tensile_stress:.1f} MPa")
    st.metric("ANSYS tensile stress reported", f"{fea_tensile:.2f} MPa")
    st.metric("Stress after idealized surface compression offset", f"{net_after_cs:.1f} MPa")

    if net_after_cs <= 0:
        st.success("Idealized compression offset exceeds the simplified bending stress. Experimental validation is still required.")
    else:
        st.warning("Residual tensile stress remains in the simplified check. Strengthening, edge finishing, and cyclic testing are critical.")

elif page == "6. Eco Audit sensitivity":
    st.header("Eco Audit sensitivity")
    baseline_total_energy = eco["energy_MJ"].sum()
    baseline_total_co2 = eco["co2_kg"].sum()
    power = st.slider("Display power, W", 0.5, 3.0, 1.5, 0.05)
    use_hours = st.slider("Daily display use, h/day", 1.0, 8.0, 4.6, 0.1)
    life = st.slider("Product life, years", 1.0, 6.0, 3.0, 0.5)

    use_factor = (power * use_hours * life) / (1.5 * 4.6 * 3.0)
    adjusted = eco.copy()
    adjusted.loc[adjusted.phase == "Use", "energy_MJ"] = 76.1 * use_factor
    adjusted.loc[adjusted.phase == "Use", "co2_kg"] = 2.49 * use_factor

    c1, c2 = st.columns(2)
    c1.metric("Adjusted total energy", f"{adjusted.energy_MJ.sum():.2f} MJ", f"{adjusted.energy_MJ.sum()-baseline_total_energy:.2f} MJ")
    c2.metric("Adjusted total CO₂-eq", f"{adjusted.co2_kg.sum():.3f} kg", f"{adjusted.co2_kg.sum()-baseline_total_co2:.3f} kg")

    fig = px.bar(adjusted, x="phase", y="energy_MJ", title="Adjusted energy by phase")
    st.plotly_chart(fig, use_container_width=True)

elif page == "7. Validation plan":
    st.header("Validation plan for written report")
    st.dataframe(validation, use_container_width=True)
    st.markdown("""
    **Main message:** The completed FEA simulation supports the selection by locating the critical stress at the hinge/fold region.
    However, final approval requires real UTG validation because the model does not fully include surface flaws, residual compressive
    stress from ion exchange, coatings, adhesive layers, or fatigue damage.
    """)
