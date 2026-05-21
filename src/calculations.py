import pandas as pd
import numpy as np

REPORT_WEIGHTS = {
    "foldability": 0.25,
    "fracture_flaw_tolerance": 0.25,
    "surface_durability": 0.15,
    "thermal_compatibility": 0.10,
    "optical_application_relevance": 0.10,
    "cost_sustainability": 0.10,
    "utg_validation_risk": 0.05,
}

def add_midpoints_and_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add midpoints and Ashby-style indices used in the UTG report."""
    df = df.copy()
    midmap = {
        "density_kg_m3": ("density_min_kg_m3", "density_max_kg_m3"),
        "E_GPa": ("E_min_GPa", "E_max_GPa"),
        "flex_MPa": ("flex_min_MPa", "flex_max_MPa"),
        "KIC_MPa_sqrtm": ("KIC_min_MPa_sqrtm", "KIC_max_MPa_sqrtm"),
        "HV": ("HV_min", "HV_max"),
        "alpha_microstrain_C": ("alpha_min_microstrain_C", "alpha_max_microstrain_C"),
        "price_EUR_kg": ("price_min_EUR_kg", "price_max_EUR_kg"),
        "co2_kgkg": ("co2_min_kgkg", "co2_max_kgkg"),
        "embodied_MJkg": ("embodied_min_MJkg", "embodied_max_MJkg"),
        "water_Lkg": ("water_min_Lkg", "water_max_Lkg"),
        "thermal_shock_C": ("thermal_shock_min_C", "thermal_shock_max_C"),
    }
    for out, (lo, hi) in midmap.items():
        if out not in df.columns:
            df[out] = (df[lo] + df[hi]) / 2

    df["bending_index_flex_over_E"] = df["flex_MPa"] / df["E_GPa"]
    df["fracture_index_KIC_over_E"] = df["KIC_MPa_sqrtm"] / df["E_GPa"]
    df["fracture_energy_index_KIC2_over_E"] = (df["KIC_MPa_sqrtm"] ** 2) / df["E_GPa"]
    df["hardness_density_index_HV_over_rho"] = df["HV"] / df["density_kg_m3"]
    df["thermal_mismatch_alphaE"] = df["alpha_microstrain_C"] * df["E_GPa"]

    # Report's simplified geometry relation: sigma_bending = 12.1 * E MPa
    df["estimated_bending_stress_MPa"] = 12.1 * df["E_GPa"]
    df["base_strength_margin"] = df["flex_MPa"] / df["estimated_bending_stress_MPa"]
    return df

def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale numeric data to 1-5. Used for transparent sensitivity checks, not as experimental truth."""
    s = series.astype(float)
    if np.isclose(s.max(), s.min()):
        return pd.Series([3.0] * len(s), index=s.index)
    scaled = (s - s.min()) / (s.max() - s.min())
    if not higher_is_better:
        scaled = 1 - scaled
    return 1 + 4 * scaled

def weighted_score_from_report_blocks(block_scores: pd.DataFrame, weights: dict = REPORT_WEIGHTS) -> pd.DataFrame:
    out = block_scores.copy()
    out["weighted_score"] = sum(out[col] * weights[col] for col in weights)
    return out.sort_values("weighted_score", ascending=False)

def calculate_data_driven_scores(df: pd.DataFrame, weights: dict = REPORT_WEIGHTS) -> pd.DataFrame:
    """Alternative reproducible scoring from raw property columns.

    This is useful for sensitivity, but the report matrix remains the main decision basis.
    """
    d = add_midpoints_and_indices(df)
    out = d[["material"]].copy()

    out["foldability"] = (
        minmax_score(d["bending_index_flex_over_E"], True)
        + minmax_score(d["estimated_bending_stress_MPa"], False)
        + minmax_score(d["base_strength_margin"], True)
    ) / 3

    out["fracture_flaw_tolerance"] = (
        minmax_score(d["KIC_MPa_sqrtm"], True)
        + minmax_score(d["fracture_index_KIC_over_E"], True)
        + minmax_score(d["fracture_energy_index_KIC2_over_E"], True)
    ) / 3

    out["surface_durability"] = (
        minmax_score(d["HV"], True)
        + minmax_score(d["hardness_density_index_HV_over_rho"], True)
    ) / 2

    out["thermal_compatibility"] = (
        minmax_score(d["thermal_mismatch_alphaE"], False)
        + minmax_score(d["thermal_shock_C"], True)
    ) / 2

    out["optical_application_relevance"] = d["application_score_report"]
    out["utg_validation_risk"] = d["utg_validation_score_report"]

    out["cost_sustainability"] = (
        minmax_score(d["price_EUR_kg"], False)
        + minmax_score(d["co2_kgkg"], False)
        + minmax_score(d["embodied_MJkg"], False)
        + minmax_score(d["water_Lkg"], False)
    ) / 4

    return weighted_score_from_report_blocks(out, weights)
