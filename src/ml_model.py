import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error

FEATURES = [
    "E_GPa",
    "flex_MPa",
    "KIC_MPa_sqrtm",
    "HV",
    "density_kg_m3",
    "alpha_microstrain_C",
    "price_EUR_kg",
    "co2_kgkg",
    "embodied_MJkg",
    "water_Lkg",
    "thermal_shock_C",
    "bending_index_flex_over_E",
    "fracture_index_KIC_over_E",
    "fracture_energy_index_KIC2_over_E",
    "hardness_density_index_HV_over_rho",
    "thermal_mismatch_alphaE",
]

def fit_ml_decision_support(materials_with_indices: pd.DataFrame, weighted_scores: pd.DataFrame):
    """Train a small decision-support model to reproduce the report score.

    Important: this is not a lifetime predictor. It is a transparent surrogate for the weighted
    decision matrix, useful for sensitivity checks and explaining which features drive the decision.
    """
    data = materials_with_indices.merge(weighted_scores[["material", "weighted_score"]], on="material")
    X = data[FEATURES]
    y = data["weighted_score"]

    # Ridge is safer for only five rows; RandomForest is provided as an optional non-linear comparison.
    ridge = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    ridge.fit(X, y)
    data["ml_score_ridge"] = ridge.predict(X)

    # Leave-one-out diagnostic: with only five materials, this is an instability check, not proof.
    loo_pred = cross_val_predict(ridge, X, y, cv=LeaveOneOut())
    data["loo_prediction_ridge"] = loo_pred
    loo_mae = mean_absolute_error(y, loo_pred)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=1)
    rf.fit(X, y)
    data["ml_score_random_forest"] = rf.predict(X)

    return data.sort_values("weighted_score", ascending=False), {
        "ridge_leave_one_out_mae": float(loo_mae),
        "warning": "Only five candidates are available. ML is a decision-support surrogate, not an experimental fold-life predictor.",
    }
