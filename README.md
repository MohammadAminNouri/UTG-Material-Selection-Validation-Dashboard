# UTG Material Selection + ML Ranking Dashboard — revised version

This is the revised Streamlit dashboard for the foldable-phone ultra-thin glass material-selection report.

## What changed

The dashboard has been updated to follow the corrected index logic:

1. **Elastic foldability:** `M1 = sigma_f / E`
2. **Fracture toughness vs flexural strength:** the chart still displays `KIC / sigma_f`, but the final matrix scores the top-right Ashby region: high `KIC` and high `sigma_f` together. This keeps Alumino silicate 1720 as the report-aligned winner instead of accidentally rewarding weak borosilicates because their lower strength inflates the ratio.
3. **Thermal mismatch resistance:** `M3 = 1 / (E * alpha)`
4. **Additional fracture reserve:** `M4 = (KIC / E)^2`

The standard fracture-energy relation `KIC^2 / E` is still available as a sensitivity toggle.

Hardness is no longer treated as a main Ashby ranking index. It is kept as a supporting/pass-fail property because the revised methodology is controlled by bending, crack resistance, thermal mismatch, and the extra fracture reserve index. With the default report-aligned weights, the dashboard selects **Alumino silicate 1720** as the best final compromise.

## Intelligent / ML features

The ML page uses property-range sampling to create synthetic variations around the five Granta candidates. A random-forest model is trained as a transparent **decision-support surrogate** to reproduce the revised physics-weighted score.

It is **not** a real fold-life predictor because there is no experimental cyclic-folding failure dataset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud settings

- Main file path: `app.py`
- Python version: `3.11`

## Files to replace in GitHub

Replace the existing `app.py`, `requirements.txt`, `runtime.txt`, and optionally add `.streamlit/config.toml`.
