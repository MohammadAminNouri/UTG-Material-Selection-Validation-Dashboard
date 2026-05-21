# Foldable Phone Glass UTG Validation Dashboard

This repository is a digital companion to a written material-selection report for a foldable phone ultra-thin glass cover window.

## Selected material

**Alumino silicate 1720** is selected as the strongest compromise before final experimental validation.

## What the app does

- Reproduces the five-material candidate shortlist.
- Calculates Ashby-style indices:
  - flexural strength / Young's modulus
  - fracture toughness / Young's modulus
  - fracture toughness² / Young's modulus
  - hardness / density
  - thermal expansion × Young's modulus
- Recreates the weighted decision matrix used in the report.
- Adds sensitivity testing for weights.
- Adds a safe ML decision-support module.
- Adds a bending and chemical-strengthening check.
- Adds Eco Audit sensitivity.
- Provides a validation checklist.

## Important limitation

The ML module is **not** a real fold-life predictor. It is a decision-support surrogate trained to reproduce the report's weighted material-selection score. Final approval still requires chemical strengthening validation, edge inspection, bend-radius testing, cyclic folding, scratch/pen-drop testing, optical inspection, lamination testing, and manufacturing trials.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Use:
- Main file path: `app.py`
- Python version: 3.11 or later
