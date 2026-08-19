import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
import os
from forwardkinematics import ForwardKinematic_points
from forwardkinematics import ForwardKinematic
def euc_dist_3d(p1, p2):
    """Calculates the Euclidean distance between two 3D points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

# Setup plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

#DATA_DIR = "Data"
MODEL_DIR = "Model"
RESULTS_DIR = "RESULTS"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Testing and Evaluating 6-DOF Model...")

# 1. Locate the latest model
model_files = glob.glob(os.path.join(MODEL_DIR, "best_model_*AdamW.keras"))
#model_files = glob.glob(os.path.join(MODEL_DIR, "best_model_*MuOn.keras"))
if not model_files:
    raise FileNotFoundError(f"No model files found in '{MODEL_DIR}'. Train a model first.")

#latest_model = max(model_files, key=os.path.getctime)
latest_model = os.path.join(MODEL_DIR,'best_model_20260810_174659_MuOn.keras')
print(f"Using latest model: {latest_model}")

model_basename = os.path.splitext(os.path.basename(latest_model))[0]
results_subdir = os.path.join(RESULTS_DIR, model_basename)
os.makedirs(results_subdir, exist_ok=True)

# 2. Load Model and Test Data
model = keras.models.load_model(latest_model)
#data = np.load(os.path.join(DATA_DIR, 'dataset_processed.npz'))
data = np.load('dataset_processed.npz')

X_test = data['X_test']
y_test = data['y_test']

# Load Scaler to inverse transform features if needed, though predictions here are pure angles
scaler_X = joblib.load('scaler_X.pkl')

# 3. Predict & Convert
y_pred = model.predict(X_test, verbose=1)

y_test_deg = np.rad2deg(y_test)
y_pred_deg = np.rad2deg(y_pred)

# 4. Calculate Angle Metrics
joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist Pitch', 'Wrist Roll', 'Wrist Twist']
mae = mean_absolute_error(y_test_deg, y_pred_deg, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(y_test_deg, y_pred_deg, multioutput='raw_values'))

metrics_df = pd.DataFrame({
    'Joint': joint_names,
    'MAE (deg)': mae,
    'RMSE (deg)': rmse
})
print("\n--- Test Set Metrics ---")
print(metrics_df.to_string(index=False))
metrics_df.to_csv(os.path.join(results_subdir, f'metrics_{model_basename}.csv'), index=False)

# 5. Calculate Physical Spatial Error (Euclidean Distance of the End-Effector)
individual_distances = []
for p_angles, t_angles in zip(y_pred, y_test):
    # ForwardKinematic_points returns 21 values. The last 3 are the end-effector (X, Y, Z)
    '''
    pred_xyz = ForwardKinematic_points(*p_angles)[-3:]
    true_xyz = ForwardKinematic_points(*t_angles)[-3:]
    individual_distances.append(euc_dist_3d(pred_xyz, true_xyz))
    '''
    pred_xyz = ForwardKinematic(*p_angles)
    true_xyz = ForwardKinematic(*t_angles)
    individual_distances.append(euc_dist_3d(pred_xyz, true_xyz))
    

errors_array = np.array(individual_distances)
print(f'\nMédia dist euclidiana: {np.mean(errors_array):.4f} cm')
print(f'Desvio padrao: {np.std(errors_array):.4f} cm')

# 6. Visualization: Parity Plots (3x2 grid for 6 joints)
fig, axs = plt.subplots(3, 2, figsize=(12, 14))
fig.suptitle('Model Performance: Predicted vs Actual (Test Set)', fontsize=16)

for i, ax in enumerate(axs.flat):
    ax.scatter(y_test_deg[:, i], y_pred_deg[:, i], alpha=0.3, s=5, color='blue')
    
    # Diagonal Line
    min_val = min(y_test_deg[:, i].min(), y_pred_deg[:, i].min())
    max_val = max(y_test_deg[:, i].max(), y_pred_deg[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    ax.set_title(f'{joint_names[i]} | MAE: {mae[i]:.2f}°', fontsize=12)
    ax.set_xlabel('Actual Angle (deg)')
    ax.set_ylabel('Predicted Angle (deg)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(results_subdir, 'parity_plots.png'))
plt.close()

# 7. Visualization: Spatial Error Histogram
plt.figure(figsize=(10, 6))
plt.hist(errors_array, bins='auto', color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Histograma das Distâncias Euclidianas (End-Effector)')
plt.xlabel('Distância Euclidiana (cm)')
plt.ylabel('Frequência')
plt.grid(axis='y', alpha=0.5)

plt.axvline(np.mean(errors_array), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {np.mean(errors_array):.2f} cm')
plt.legend()
plt.savefig(os.path.join(results_subdir, 'euclidean_error_hist.png'))
plt.close()

print(f"Results and plots saved to {results_subdir}")