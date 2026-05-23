# UTG Material Selection and Machine-Learning Ranking Dashboard

This repository contains the Python/Streamlit dashboard developed to support the foldable-phone ultra-thin glass material-selection report. The dashboard follows the Granta/Ashby methodology used in the report and provides a transparent computational tool for calculating material indices, generating the weighted decision matrix, visualizing the candidate ranking, and performing machine-learning-based sensitivity analysis.

## Project Links

- GitHub repository: [insert GitHub link here]
- Streamlit dashboard: [insert Streamlit app link here]

## Dashboard Pages

The app contains the following pages:

1. **Decision-making matrix** — report-aligned weighted matrix, normalized weights, final ranking, and score decomposition.
2. **Ashby charts** — foldability, fracture toughness versus flexural strength, fracture toughness versus Young's modulus, and thermal mismatch charts.
3. **Raw indices** — calculated midpoint material properties and index values.
4. **ML sensitivity** — Random Forest feature importance and Monte Carlo ranking stability.
5. **Validation logic** — final interpretation and role of reserve materials.

## Methodology Implemented in the Dashboard

The dashboard evaluates the five shortlisted Granta candidates:

- Alumino silicate 1720
- Alumino silicate 1723
- Borosilicate 7251
- Borosilicate 7760
- Borosilicate KG33

The material ranking is based on the following indices:

| Index | Formula | Purpose |
|---|---:|---|
| Elastic foldability | `M1 = sigma_f / E` | Evaluates the balance between flexural strength and Young's modulus for repeated bending |
| Crack resistance | `M2 = KIC^2 / E` | Evaluates brittle fracture-energy resistance |
| Flaw tolerance | `M3 = KIC / sigma_f` | Used as a defect-tolerance diagnostic, with strength gating in the final score |
| Thermal-mismatch resistance | `M4 = 1 / (E * alpha)` | Favors materials with lower thermal-stress tendency in the display stack |

Hardness is retained as a supporting/pass-fail property rather than a main ranking index, because the final methodology is controlled mainly by bending, crack resistance, flaw tolerance, and thermal mismatch.

## Weighted Decision Matrix

The dashboard applies the report weighting logic:

| Selection block | Report weight | Role in material selection |
|---|---:|---|
| Elastic foldability | 30% | Main requirement for repeated folding without brittle fracture |
| Crack resistance | 20% | Fracture-energy resistance using `KIC^2 / E` and high `KIC` with sufficient strength |
| Flaw tolerance | 20% | Ability to remain functional in the presence of defects |
| Thermal compatibility | 20% | Reduces thermal-mismatch risk in the laminated display stack |
| Cost and sustainability | 10% | Includes price, CO2 footprint, embodied energy, water use, critical elements, and recycling indicators |

The report weights sum to 100%, and the app also normalizes them internally as a safeguard. With the default values, the normalized values are identical to the report weights.

The weighted score is calculated as:

`0.30(Foldability) + 0.20(Crack resistance) + 0.20(Flaw tolerance) + 0.20(Thermal compatibility) + 0.10(Cost and sustainability)`

With these weights, the dashboard selects **Alumino silicate 1720** as the strongest final material candidate.

## Machine-Learning Support

The machine-learning module uses property-range sampling around the five shortlisted Granta candidates. Synthetic samples are generated within the Granta min-max ranges, and the material indices are recalculated for each generated sample.

A Random Forest regression model is trained to reproduce the physics-weighted score, while a classification model supports the selected, reserve, and not-selected decision categories.

The model is not used as a real fold-life predictor, because experimental cyclic-folding failure data are not available. Its role is to support the decision matrix and identify which properties should be prioritized during physical validation.

## Final Material Decision

The dashboard supports the selection of **Alumino silicate 1720** because it provides the strongest overall balance of elastic foldability, flexural strength, fracture toughness, fracture-energy resistance, cost, production CO2, embodied energy, and application relevance.

The remaining candidates are retained as validation comparators:

| Candidate | Role |
|---|---|
| Alumino silicate 1723 | Reserve / close alternative |
| Borosilicate KG33 | Thermal comparator |
| Borosilicate 7251 | Secondary comparator |
| Borosilicate 7760 | Low-modulus comparator |

The result is a robust base-material selection, not final product certification. Final approval still requires chemical strengthening validation, processed UTG strength testing, edge-quality control, optical inspection, coating and lamination assessment, and cyclic folding reliability testing.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Settings

- Main file path: `app.py`
- Python version: `3.11`

## Files to Replace in GitHub

Replace the existing files with:

- `app.py`
- `requirements.txt`
- `runtime.txt`
- `README.md`

Optionally add:

- `.streamlit/config.toml`
