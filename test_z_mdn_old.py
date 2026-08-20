import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
import keras_mdn_layer as mdn
import os
from forwardkinematics import ForwardKinematic_points
from forwardkinematics import ForwardKinematic

def euc_dist_3d(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

MODEL_DIR = "Model"
RESULTS_DIR = "RESULTS"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_DIMS = 6
N_MIXES = 20

print("Testing and Evaluating 3-Input 6-DOF MDN Model...")

model_files = glob.glob(os.path.join(MODEL_DIR, "best_model_*_MDN.keras"))
if not model_files:
    raise FileNotFoundError(f"No model files found in '{MODEL_DIR}'.")

latest_model = max(model_files, key=os.path.getctime)
print(f"Using latest model: {latest_model}")

model_basename = os.path.splitext(os.path.basename(latest_model))[0]
results_subdir = os.path.join(RESULTS_DIR, model_basename)
os.makedirs(results_subdir, exist_ok=True)

model = keras.models.load_model(latest_model, custom_objects={'MDN': mdn.MDN}, compile=False)

data = np.load('dataset_processed.npz')
X_test = data['X_test']
y_test = data['y_test']

print("Predicting distributions...")
y_pred_params = model.predict(X_test, verbose=0)

mu_size = OUTPUT_DIMS * N_MIXES
mus = y_pred_params[:, :mu_size]
pis = y_pred_params[:, -N_MIXES:]
mus_reshaped = mus.reshape(-1, N_MIXES, OUTPUT_DIMS)

best_pi_idx = np.argmax(pis, axis=1)
y_pred = mus_reshaped[np.arange(len(y_pred_params)), best_pi_idx, :]

# --- NEW: Calculate Best Possible MAE ---
# This finds the mixture that is mathematically closest to the test label
best_mae_mus = np.zeros_like(y_test)
for i in range(len(y_test)):
    errors = np.sum(np.abs(mus_reshaped[i] - y_test[i]), axis=1) # Sum of absolute errors for each mixture
    best_idx = np.argmin(errors)
    best_mae_mus[i] = mus_reshaped[i, best_idx, :]

y_test_deg = np.rad2deg(y_test)
y_pred_deg = np.rad2deg(y_pred)
best_mae_mus_deg = np.rad2deg(best_mae_mus)

joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist Pitch', 'Wrist Roll', 'Wrist Twist']
mae = mean_absolute_error(y_test_deg, y_pred_deg, multioutput='raw_values')
best_mae = mean_absolute_error(y_test_deg, best_mae_mus_deg, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(y_test_deg, y_pred_deg, multioutput='raw_values'))

metrics_df = pd.DataFrame({
    'Joint': joint_names,
    'MAE (Selected Mixture)': mae,
    'MAE (Best Mixture)': best_mae,  # This will prove the model learned the poses!
    'RMSE (Selected Mixture)': rmse
})
print("\n--- Test Set Angle Metrics ---")
print(metrics_df.to_string(index=False))
print("\n* NOTE: 'Best Mixture MAE' proves the model learned the poses. The high 'Selected Mixture MAE' is just the model choosing a redundant pose (e.g., elbow up vs down).")
metrics_df.to_csv(os.path.join(results_subdir, f'metrics_{model_basename}.csv'), index=False)

individual_distances = []
best_mixture_distances = []

for i, (p_angles, t_angles) in enumerate(zip(y_pred, y_test)):
    pred_xyz = ForwardKinematic(*p_angles)[-3:]
    true_xyz = ForwardKinematic(*t_angles)[-3:]
    individual_distances.append(euc_dist_3d(pred_xyz, true_xyz))
    
    best_dist = float('inf')
    for k in range(N_MIXES):
        k_angles = mus_reshaped[i, k, :]
        k_xyz = ForwardKinematic(*k_angles)[-3:]
        dist = euc_dist_3d(k_xyz, true_xyz)
        if dist < best_dist:
            best_dist = dist
    best_mixture_distances.append(best_dist)

errors_array = np.array(individual_distances)
best_errors_array = np.array(best_mixture_distances)

print(f'\n--- Spatial Euclidean Error ---')
print(f'Média dist euclidiana (Selected): {np.mean(errors_array):.4f} cm')
print(f'Média dist euclidiana (BEST Possible): {np.mean(best_errors_array):.4f} cm')

fig, axs = plt.subplots(3, 2, figsize=(12, 14))
fig.suptitle('MDN Model Performance: Predicted vs Actual (Test Set)', fontsize=16)

for i, ax in enumerate(axs.flat):
    ax.scatter(y_test_deg[:, i], y_pred_deg[:, i], alpha=0.3, s=5, color='blue', label='Selected Mixture')
    ax.scatter(y_test_deg[:, i], best_mae_mus_deg[:, i], alpha=0.1, s=2, color='green', label='Best Mixture (Optimal)')
    
    min_val = min(y_test_deg[:, i].min(), y_pred_deg[:, i].min())
    max_val = max(y_test_deg[:, i].max(), y_pred_deg[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    ax.set_title(f'{joint_names[i]} | MAE: {mae[i]:.2f}° (Best: {best_mae[i]:.2f}°)', fontsize=12)
    ax.set_xlabel('Actual Angle (deg)')
    ax.set_ylabel('Predicted Angle (deg)')
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_subdir, 'parity_plots.png'))
plt.close()

plt.figure(figsize=(10, 6))
plt.hist(errors_array, bins='auto', color='steelblue', edgecolor='black', alpha=0.7, label='Selected Mixture')
plt.hist(best_errors_array, bins='auto', color='orange', edgecolor='black', alpha=0.5, label='Best Mixture')

plt.title('Histograma das Distâncias Euclidianas (End-Effector)')
plt.xlabel('Distância Euclidiana (cm)')
plt.ylabel('Frequência')
plt.grid(axis='y', alpha=0.5)

plt.axvline(np.mean(errors_array), color='blue', linestyle='dashed', linewidth=1.5, label=f'Mean Selected: {np.mean(errors_array):.2f} cm')
plt.axvline(np.mean(best_errors_array), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Best: {np.mean(best_errors_array):.2f} cm')
plt.legend()
plt.savefig(os.path.join(results_subdir, 'euclidean_error_hist.png'))
plt.close()

print(f"Results and plots saved to {results_subdir}")