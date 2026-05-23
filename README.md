# UTG Material Selection and Machine-Learning Ranking Dashboard

This repository contains the Python/Streamlit dashboard developed to support the foldable-phone ultra-thin glass material-selection report. The dashboard follows the Granta/Ashby methodology used in the report and provides a transparent computational tool for calculating material indices, generating the weighted decision matrix, visualizing the candidate ranking, and performing machine-learning-based sensitivity analysis.

The dashboard supports the report methodology by linking hard Granta screening, Ashby performance indices, weighted decision-making, and validation planning in one reproducible workflow.

## Project Links

- GitHub repository: [insert GitHub link here]
- Streamlit dashboard: [insert Streamlit app link here]

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
| Elastic foldability | `M1 = sigma_f / E` | Evaluates the balance between flexural strength and Young’s modulus for repeated bending |
| Fracture toughness versus flexural strength | `KIC` and `sigma_f` scored together | Rewards materials with both high fracture toughness and high flexural strength |
| Diagnostic crack-tolerance ratio | `M2 = KIC / sigma_f` | Used as a supporting diagnostic value, not as the only fracture score |
| Thermal-mismatch resistance | `M3 = 1 / (E * alpha)` | Favors materials with lower thermal-stress tendency in the display stack |
| Additional fracture reserve | `M4 = (KIC / E)^2` | Provides a toughness-to-stiffness reserve check |
| Fracture-energy reference | `KIC^2 / E` | Used as a sensitivity check for brittle fracture-energy resistance |

Hardness is retained as a supporting/pass-fail property rather than a main ranking index, because the final methodology is controlled mainly by bending, crack resistance, thermal mismatch, and fracture-reserve behavior.

## Weighted Decision Matrix

The dashboard applies the following weighting logic:

| Selection block | Weight | Role in material selection |
|---|---:|---|
| Elastic foldability | 35% | Main requirement for repeated folding without brittle fracture |
| Fracture/crack resistance | 30% | Accounts for flaw-controlled failure from surface scratches, edge defects, and indentation marks |
| Thermal compatibility | 15% | Reduces thermal-mismatch risk in the laminated display stack |
| Optical/application relevance | 10% | Supports suitability for transparent display-cover use |
| Cost and sustainability | 10% | Includes price, CO₂ footprint, embodied energy, water use, and recycling indicators |

With these weights, the dashboard selects **Alumino silicate 1720** as the strongest final material candidate.

## Machine-Learning Support

The machine-learning module uses property-range sampling around the five shortlisted Granta candidates. Synthetic samples are generated within the Granta min–max ranges, and the material indices are recalculated for each generated sample.

A Random Forest regression model is trained to reproduce the physics-weighted score, while a classification model supports the selected, reserve, and not-selected decision categories.

The machine-learning module is used for:

- sensitivity analysis;
- feature-importance ranking;
- counterfactual property testing;
- checking the stability of the material ranking under property variation;
- supporting the validation plan.

The model is not used as a real fold-life predictor, because experimental cyclic-folding failure data are not available. Its role is to support the decision matrix and identify which properties should be prioritized during physical validation.

## Final Material Decision

The dashboard supports the selection of **Alumino silicate 1720** because it provides the strongest overall balance of elastic foldability, flexural strength, fracture toughness, fracture toughness versus flexural strength performance, cost, production CO₂, embodied energy, and application relevance.

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
