# 🌱 Dataset Preparation for Crop Type Classification

This section of the project includes a series of Jupyter notebooks for **cleaning, filtering, and structuring crop data** before training the classification model using XGBoost. It ensures the dataset is high-quality, temporally consistent, and suitable for supervised machine learning.

---

## 📁 File Descriptions

### [`1_process1_10week.ipynb`](./1_process1_10week.ipynb)

Initial cleaning and aggregation of raw crop data.

- Load raw field-level data with weekly crop, vegetation, and climate indicators.
- Filter records based on coverage thresholds and NDVI quality.
- Compute summary statistics (mean, max, min, std) across vegetation and environmental variables.
- Output: Cleaned DataFrame with 10-week observations per field-year.

---

### [`1_process2_10week.ipynb`](./1_process2_10week.ipynb)

Restructuring the cleaned data into a **long-format weekly table**.

- Load the cleaned output from `1_process1_10week.ipynb`.
- Flatten weekly data: one row per `FIELDID`–`Year`–`Week`.
- Add metadata (e.g., `FIELDID`, `Year`, `Week`) to support later merges and filtering.

---

### [`2_filter_dates.ipynb`](./2_filter_dates.ipynb)

Filtering the dataset by **growing season windows**.

- Define crop-specific valid time ranges (e.g., sowing to harvest).
- Remove weeks outside the defined growing season.
- Ensure consistency in weekly coverage across field-year samples.

---

### [`3_reduce_25week.ipynb`](./3_reduce_25week.ipynb)

Preparing the **final feature set** for model calibration.

- Select key features: NDVI, EVI, soil moisture, precipitation, etc.
- Remove rows with missing or sparse values.
- Add crop type labels (`Crop_num`) for supervised training.
- Output: Final `.csv` ready to be used in `1_SB25r_Fit_Classifier.ipynb`.

---

## ⚙️ Environment Setup

Install required Python packages:

```bash
pip install pandas numpy matplotlib seaborn
