# Crop-Type Pipeline — Simple Walkthrough

This short guide explains, in plain words, what each notebook does and how they fit together to prepare data for crop‑type modeling.

---

## Big Picture

You start with raw weekly signals (vegetation indices from Sentinel‑2 and radar/SAR), clean and relabel them, keep only reasonable growing‑season weeks, and finally reshape every field’s time series into the **same 25‑week format** ready for training.

```
Raw yearly CSVs
   → (1) process1 / process2: clean + relabel + quality filters + class caps
   → (2) filter_dates: keep only valid seasonal weeks + light features
   → (3) reduce_25week: align each sample to a fixed 25-week window
   → Final fixed-width table ready for modeling
```

---

## (1) `1_process1_10week.ipynb` — Clean, relabel, minimum coverage, rebalance

**Inputs:** Cleaned per‑year CSVs (multiple seasons).  
**Key steps (human terms):**
- **Merge & clean:** Stack years, drop missing rows.
- **Fix “Wheat + X” labels:** If it’s wheat season (weeks ~17–44), call it **Wheat**; otherwise, call it the **other crop** (Maize, Sunflower, Soy).
- **Keep decent vegetation signal:** Only rows with `veg_median_ndvi ≥ 0.2`.
- **Tidy columns & numeric labels:** e.g., rename `SoyaBeans → Soy`, add `Crop_num`.
- **Minimum dual‑sensor coverage by parcel–year–crop:** keep groups with **≥10 valid vegetation weeks** **and** **≥10 valid SAR weeks**.
- **Rebalance (cap big classes):** limit per‑crop counts per year to reduce dominance.
  
**Output:** `InputModel/SB10r_n0.2_process1.csv`

---

## (1b) `1_process2_10week.ipynb` — Same idea for other seasons

Same logic as above but applied to another set of seasons (e.g., 2019–20, 2020–21):
- Merge & clean; fix Wheat‑blend labels by week;
- NDVI threshold (`≥ 0.2`);
- Define vegetation (`veg_*`) and radar (`sar_*`) features;
- Enforce **≥10 valid weeks** in **both** sensor families per parcel–year–crop;
- Rebalance classes per year;
  
**Output:** `InputModel/SB10r_n0.2_process2.csv`

---

## (2) `2_filter_dates.ipynb` — Keep only weeks that make agronomic sense

**Inputs:** The two outputs above (stacked together). Radar columns are renamed `sar_* → s1_*` for consistency.

**What it does:**
- **Seasonal filtering by crop:** Remove weeks that don’t fit each crop’s growing season  
  - **Wheat:** keep ~weeks **17–47**  
  - **Summer crops** (Maize, Soy, Sunflower, Pasture, Tree, Fallow, Groundnuts, Sorghum): drop weeks **22–40** (or similar rule)  
  - **Lucern:** irrigated behaves like Wheat; rainfed behaves like summer crops
- **Quick checks & plots:** class counts, average NDVI by week, parcels per crop/year.
- **Light features:** `week_sin/cos`, simple growth rates (max‑min), mean of key vegetation indices, a simple NDVI/EVI ratio.

**Output:** `InputModel/SB10r_n0.2_process_filt.csv`

---

## (3) `3_reduce_25week.ipynb` — Make every sample the same length

**Goal:** Some fields have 18 valid weeks, others 32. Models often prefer **fixed‑length** input.  
**What it does:**
- For each **(FIELDID, Year, Crop)**, select a consistent **25‑week window** representing the growing season.
- Convert the time series into **slots**: `week_01 .. week_25` for **all** `veg_*` and `s1_*` features.
- Drop samples that still don’t meet minimum coverage inside that window.
  
**Result:** A **uniform, 25‑week feature matrix** where every sample has the same columns — ready for training.

---

## Key Outputs (where to find them)

- `InputModel/SB10r_n0.2_process1.csv` — Cleaned, rebalanced set (part 1).  
- `InputModel/SB10r_n0.2_process2.csv` — Cleaned, rebalanced set (part 2).  
- `InputModel/SB10r_n0.2_process_filt.csv` — Season‑filtered + light features.  
- Final **25‑week** fixed‑width table — produced by `3_reduce_25week.ipynb`.

---

## Quick Glossary

- **NDVI / EVI / LAI:** Vegetation indicators from optical satellites (plant greenness/leaf area).  
- **SAR / S1:** Radar measurements (Sentinel‑1) — useful even with clouds.  
- **Valid week:** A week with the expected features present and inside the crop’s season.  
- **Rebalancing:** Capping large classes so the model doesn’t learn a bias toward them.

---

**Bottom line:** These notebooks turn messy, multi‑year weekly data into a clean, balanced, season‑aware and **fixed‑length** dataset that most ML models can digest easily.
