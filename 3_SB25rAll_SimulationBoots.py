# # Setting up the environment
# Import necessary libraries
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import joblib  # For loading the saved model
import json

# Start tracking time for performance evaluation
start_time = time.time()

# # Load Pre-Trained Model and Selected Features
dataset_name = 'SB25rAll'
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
input_file = os.path.join('Data_Preparation/InputModel', f'{dataset_name}_n0.2_process_filt.csv')
output_folder = 'Output'
os.makedirs(output_folder, exist_ok=True)

# Load data from CSV
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file {input_file} not found.")
df = pd.read_csv(input_file)

############################################################################################################
# # Model Prediction
############################################################################################################

X = df[selected_features].dropna()
y = df["Crop_num"].loc[X.index]
y_pred = model.predict(X)

############################################################################################################
# # Bootstrapping Accuracy Calculation per Crop per Week
############################################################################################################

def bootstrap_accuracy(y_true, y_pred, n_bootstraps=100):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    accuracies = [accuracy_score(y_true[np.random.choice(len(y_true), len(y_true), replace=True)], y_pred) for _ in range(n_bootstraps)]
    return np.mean(accuracies), np.std(accuracies)

# Ensure alignment before selecting with week_indices
y = y.reset_index(drop=True)
y_pred = pd.Series(y_pred).reset_index(drop=True)
df = df.reset_index(drop=True)  # Reset index of df as well

accuracy_results = []
for crop in df['Crop_type'].unique():
    for week in sorted(df['week'].unique()):
        week_mask = (df['week'] == week) & (df['Crop_type'] == crop)
        week_indices = df[week_mask].index.to_list()  # Ensure indices exist

        if len(week_indices) > 0:
            # Use .loc instead of iloc to avoid index misalignment
            y_true_week = y.loc[week_indices].values
            y_pred_week = y_pred.loc[week_indices].values
            
            # Check if y_true_week and y_pred_week are of equal length
            if len(y_true_week) != len(y_pred_week):
                print(f"Warning: Mismatched lengths for {crop}, Week {week}. Skipping.")
                continue

            mean_acc, std_acc = bootstrap_accuracy(y_true_week, y_pred_week, n_bootstraps=100)
            accuracy_results.append({'Crop': crop, 'Week': week, 'Mean_Accuracy': mean_acc, 'Accuracy_Std': std_acc})

accuracy_df = pd.DataFrame(accuracy_results)

############################################################################################################
# # Plot Smoothed Accuracy per Crop per Week
############################################################################################################

import numpy as np
from scipy.ndimage import gaussian_filter1d  

def smooth_data(y, sigma=2):
    return gaussian_filter1d(y, sigma=sigma)
crop_mapping = {0: 'Maize', 1: 'Soy', 2: 'Sunflower', 3: 'Wheat', 4: 'Lucern', 5: 'Pasture', 6: 'Tree', 7: 'Fallow', 8: 'Groundnuts', 9: 'Sorghum'}
crop_colors = {'Maize': 'red', 'Soy': 'blue', 'Sunflower': 'green', 'Wheat': 'orange', 'Lucern': 'purple', 'Pasture': 'yellow', 'Tree': 'gray', 'Fallow': 'brown', 'Groundnuts': 'pink', 'Sorghum': 'cyan'}

plt.figure(figsize=(10,  8))
for crop_num, crop_name in crop_mapping.items():
    crop_data = accuracy_df[accuracy_df['Crop'] == crop_name]
    weeks, mean_acc, std_acc = crop_data['Week'].values, crop_data['Mean_Accuracy'].values, crop_data['Accuracy_Std'].values
    smoothed_mean_acc = smooth_data(mean_acc, sigma=2)

    plt.plot(weeks, smoothed_mean_acc, label=crop_name, color=crop_colors[crop_name], linewidth=2)
    plt.fill_between(weeks, smoothed_mean_acc - std_acc, smoothed_mean_acc + std_acc, color=crop_colors[crop_name], alpha=0.3)

plt.ylim(0, 1)
plt.title('Smoothed Bootstrapped Accuracy per Crop per Week')
plt.xlabel('Week of the Year')
plt.ylabel('Accuracy')
plt.legend(title="Crop Type")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, f'{dataset_name}_boots_accuracy.png'))

accuracy_df.to_csv(os.path.join(output_folder, f'{dataset_name}_boots_accuracy.csv'), index=False)

print(f"Script completed in {time.time() - start_time:.2f} seconds")
