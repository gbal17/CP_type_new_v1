# # Setting up the environment
# Import necessary libraries
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import joblib  # For loading the saved model
import json
from sklearn.utils import resample

# Start tracking time for performance evaluation
start_time = time.time()

# # Load Pre-Trained Model and Selected Features
# The model and selected features are loaded from files in the `Models` folder.
model_file = 'Models/SB25rAll_n0.2_process_filt_xgb200_nf25_noMet_v1.joblib'
features_file = 'Models/SB25rAll_n0.2_process_filt_xgb200_nf25_noMet_v1.json'

# Load pre-trained model
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file {model_file} not found.")
model = joblib.load(model_file)

# Load selected features
if not os.path.exists(features_file):
    raise FileNotFoundError(f"Features file {features_file} not found.")
with open(features_file, 'r') as f:
    selected_features = json.load(f)['features']

# # Dataset Configuration
# Set paths to input files.
dataset_name = 'SB25rAll'
reduction_method = '_n0.2_process_filt'
input_file = os.path.join('Data_Preparation/InputModel', f'{dataset_name}{reduction_method}.csv')
output_folder = 'Output'
os.makedirs(output_folder, exist_ok=True)

# Load data from CSV, raising an error if file is missing
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file {input_file} not found.")
df = pd.read_csv(input_file)


############################################################################################################
# # Feature Engineering
############################################################################################################
# # Handling Missing Data
# Ensure there are no missing values in the selected features
X = df[selected_features]
X = X.dropna()  # Drop rows with missing values in selected features
y = df["Crop_num"].loc[X.index]  # Ensure y matches the cleaned X dataset

# # Using Pre-Trained Model for Prediction
# Skip fitting and directly predict using the pre-trained model
y_pred = model.predict(X)

# Ensure that y and y_pred are aligned before indexing
y = y.reset_index(drop=True)
y_pred = pd.Series(y_pred).reset_index(drop=True)

# Now iterate over weeks safely
week_metrics = []
for week in sorted(df['week'].unique()):
    week_indices = df[df['week'] == week].index

    # Ensure indices are within the valid range
    valid_indices = [idx for idx in week_indices if idx < len(y_pred)]
    
    if not valid_indices:
        continue  # Skip if no valid indices found

    y_true_week = y.loc[valid_indices]
    y_pred_week = y_pred.loc[valid_indices]

    # Calculate metrics
    mae_week = mean_absolute_error(y_true_week, y_pred_week)
    mse_week = mean_squared_error(y_true_week, y_pred_week)
    r2_week = r2_score(y_true_week, y_pred_week)
    accuracy_week = accuracy_score(y_true_week, y_pred_week)

    week_metrics.append({
        'Week': week,
        'MAE': mae_week,
        'MSE': mse_week,
        'R²': r2_week,
        'Accuracy': accuracy_week
    })

# Convert metrics list to DataFrame for easier analysis
week_metrics_df = pd.DataFrame(week_metrics)

# Print and save metrics
print(week_metrics_df)
week_metrics_df.to_csv(os.path.join(output_folder, f'{dataset_name}_week_metrics.csv'), index=False)
