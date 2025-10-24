<<<<<<< HEAD
# 🌱 Crop Type Prediction using XGBoost

This project contains a series of Python notebooks and scripts designed to facilitate data preparation, model fitting, and evaluation for predicting crop types such as **Maize, Soy, Sunflower, Wheat, Lucern, Pasture, Tree, Fallow, Groundnuts**, and **Sorghum**. The workflow enables effective data processing, feature engineering, machine learning model training, and simulation using a pre-trained classifier.

---

## 📁 File Descriptions

### [`1_SB25r_Fit_Classifier.ipynb`](./1_SB25r_Fit_Classifier.ipynb)

Train and evaluate an XGBoost-based crop type classifier with a focus on feature selection and model performance.

- **Model Fitting and Feature Selection**: Applies XGBoost and Sequential Forward Selection (SFS).
- **Model Evaluation**: Accuracy, F1 score, confusion matrix.
- **Saving Outputs**: Saves model (`.joblib`) and selected features (`.json`).

---

### [`2_SB25r_Simulation.ipynb`](./2_SB25r_Simulation.ipynb)

Simulate predictions and visualize results using the pre-trained model and selected features.

- **Load Model and Features** from disk.
- **Feature Engineering**: Cyclical encoding for `week`, vegetation growth rates.
- **Model Evaluation**: Accuracy, confusion matrix.
- **Plotting**: Feature importance and time series by crop.

---

### [`3_SB25r_Simulation.py`](./3_SB25r_Simulation.py)

Script-based version for batch prediction and evaluation of model performance by week.

- **Load Model and Features**.
- **Feature Engineering**: Week transformations and vegetation growth.
- **Prediction**: Apply model to full dataset.
- **Weekly Evaluation**: MAE, MSE, R², accuracy.
- **Output**: CSV with observed and predicted crop types.

---

### [`3_SB25r_SimulationBoots.py`](./3_SB25r_SimulationBoots.py)

Robust evaluation using bootstrapped accuracy per crop and week with smoothed visualization.

- **Bootstrapping**: Calculates weekly accuracy and std dev via resampling.
- **Gaussian Smoothing**: Applies smoothing to accuracy trends.
- **Plotting**: Crop-wise accuracy plots with confidence bands.
- **Outputs**: CSV + PNG plots.

---

### [`4_SB25r_generate_out_for_simulation.py`](./4_SB25r_generate_out_for_simulation.py)


Generates a feature-engineered dataset ready for simulation, ensuring consistency with the training phase.

- **Load Dataset**.
- **Feature Engineering**:
  - Sine/cosine transformation of `week`
  - Growth rates of NDVI, EVI, LAI
  - Interaction terms (e.g., vegetation × soil moisture)
- **Select Required Features**.
- **Save Cleaned Output** as `.csv`.

---

## ⚙️ General Instructions

### 📦 Environment Setup

Ensure the following packages are installed:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
=======
# CP_type_new
Machine learning pipeline for crop type classification using XGBoost – includes dataset preparation, feature engineering, model training, simulation, and evaluation with bootstrapped accuracy metrics.
>>>>>>> 7645d6e3301f2cbe5ca6137e44b6089d85b21ca4
