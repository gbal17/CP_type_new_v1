# Crop-Type Modeling Pipeline — Simple Walkthrough (Markdown)

This guide explains, in plain words, what each code file/notebook does. It covers **data preparation**, **model training**, and **simulation/robustness** checks. No coding background is needed.

---

## Big Picture

You start with raw weekly satellite signals (Sentinel‑2 vegetation indices and Sentinel‑1 radar), clean and relabel them, keep only meaningful growing‑season weeks, convert every field’s time series into the **same 25‑week format**, train a classifier, and then simulate how well it works with **early‑season** or **missing‑weeks** data.

```
Raw yearly CSVs
   → (1) process1 / process2: clean + relabel + quality filters + class caps
   → (2) filter_dates: keep only valid seasonal weeks + light features
   → (3) reduce_25week: align each sample to a fixed 25-week window
   → (4) fit_classifier: select features + train final model (XGBoost)
   → (5) generate_out_for_simulation: prepare fast-to-load packs
   → (6) simulations: early-season / missing-weeks + bootstrapped uncertainty
   → Final: metrics, plots, trained model + selected features
```

---

## Data Preparation

### `1_process1_10week.ipynb`
**Goal:** Build a clean, balanced dataset from early seasons.  
**What it does:**
- Merges per‑year CSVs, drops missing rows.
- Fixes mixed labels like “Wheat + Maize” by **week of year** (wheat season → Wheat; otherwise the other crop).
- Keeps only rows with **NDVI ≥ 0.2** (clear vegetation signal).
- Ensures **dual‑sensor coverage**: at least **10 valid vegetation weeks** **and** **10 valid radar weeks** per *(FIELDID, Year, Crop)*.
- Rebalances classes (caps large crops per year).  
**Output:** `InputModel/SB10r_n0.2_process1.csv`

### `1_process2_10week.ipynb`
**Goal:** Do the same as above for later seasons.  
**What it does:** Same logic (merge, relabel, NDVI filter, ≥10 veg + ≥10 SAR weeks, rebalance).  
**Output:** `InputModel/SB10r_n0.2_process2.csv`

### `2_filter_dates.ipynb`
**Goal:** Keep only weeks that make agronomic sense + add a few simple features.  
**What it does:**
- Stacks the two outputs above and renames radar cols `sar_* → s1_*`.
- **Seasonal filtering** by crop (e.g., Wheat ≈ weeks 17–47; summer crops drop weeks 22–40; Lucern depends on irrigation).
- Adds light features: week sin/cos, simple growth rates (max–min), mean vegetation index, NDVI/EVI ratio.  
**Output:** `InputModel/SB10r_n0.2_process_filt.csv`

### `3_reduce_25week.ipynb`
**Goal:** Make every sample the **same length**.  
**What it does:**
- For each *(FIELDID, Year, Crop)*, selects a consistent **25‑week window** representing the growing season.
- Converts time series into slots: `week_01 … week_25` for all `veg_*` and `s1_*` features.
- Drops samples that still lack coverage.  
**Output:** A fixed‑width table where each row is one field‑year with **exactly 25 weeks** of features.

---

## Model Training & Feature Selection

### `1_SB25rAll_Fit_ClassifierX_noMet_v1.ipynb`
**Goal:** Train a **crop‑type classifier** (XGBoost) using the 25‑week dataset — **without meteorological variables**.  
**What it does:**
- Loads the fixed‑length dataset(s), picks common columns, stacks them.
- Chooses predictors and the target (`Crop_num`); drops invalid rows.
- Trains a **first XGBoost** with **class weights** to **rank features**; keeps **top‑N**.
- Runs **Forward Feature Selection** (FFS) with a **true held‑out year (2019)** and grouping **by field‑year** to avoid leakage.
- Retrains the **final model** on the selected features.
- Evaluates on 2019, saving **classification report**, **confusion matrix**, and **feature importance** plots.
- Saves the **model `.joblib`** and the **selected features `.json`**.

**Outputs (typical):**
- `Models/<...>.joblib` — final trained model  
- `Models/<...>.json` — selected features  
- Plots: confusion matrix, feature importance  
- Text/JSON metrics for each class

---

## Simulation & Robustness

### `4_SB25rAll_generate_out_for_simulation.py`
**Goal:** Prepare **ready‑to‑load packs** so simulations run fast and are reproducible.  
**What it does:**
- Loads the 25‑week table, the **trained model**, and **selected features**.
- Builds **scenario packs**:
  - **First‑N weeks** (e.g., 4/6/8/…/25).
  - **Random K weeks** (e.g., 6/10/12), with stored **week masks** for reproducibility.
- (Optionally) **Precomputes predictions** and per‑class probabilities for each pack.
- Saves everything (CSV/Parquet) and a **manifest JSON** so other scripts can find the files quickly.

### `2_SB25rAll_Simulation.ipynb`
**Goal:** Test real‑world usage scenarios **without uncertainty bands**.  
**What it does:**
- Loads the model + selected features and the prepared packs.
- For each scenario, predicts per field (aggregating week info per field) and computes:
  - **Overall accuracy** and **per‑crop precision/recall/F1**.
- Plots **accuracy vs. number of weeks**, **per‑crop curves**, and confusion matrices.
- Saves CSV metrics and PNG plots.

### `3_SB25rAll_Simulation.py`
**Goal:** Produce a **week‑by‑week** progress table (fast script).  
**What it does:**
- Loops through weeks 1..25, evaluates metrics for each cutoff.
- Records **MAE**, **MSE**, **R²**, and **Accuracy** by week.
- Saves a tidy CSV: `..._week_metrics.csv` for quick plotting/tracking.

### `3_SB25rAll_SimulationBoots.py`
**Goal:** Add **bootstrapped confidence intervals** to your simulations.  
**What it does:**
- Repeats each scenario many times on **resampled data** (bootstrapping).
- Summarizes **mean accuracy/F1** with **low–high bounds** (e.g., 95% CI).
- Plots **accuracy vs. weeks with ribbons** and saves CSVs and PNGs.

---

## What You Get at the End

- A **clean, balanced, 25‑week dataset** for each field‑year.
- A trained **XGBoost crop‑type model** + the list of **selected features**.
- **Simulation results** showing:  
  - How early in the season predictions become reliable.  
  - How robust the model is to **missing weeks**.  
  - Which crops are **easier or harder** to identify over time.  
- **Plots and CSVs** ready for reports.

---

## Quick Glossary

- **NDVI / EVI / LAI** — Vegetation indices (how green/dense plants are).  
- **SAR / Sentinel‑1** — Radar data (works through clouds).  
- **Valid weeks** — Weeks inside a crop’s growing season with the expected signals.  
- **Rebalancing** — Limiting very large classes to reduce model bias.  
- **Forward Feature Selection** — Adding features one‑by‑one and keeping them only if they help.  
- **Bootstrapping** — Repeating evaluations on resampled data to estimate **uncertainty**.
