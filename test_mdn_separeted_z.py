import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
import keras_mdn_layer as mdn
import os
from forwardkinematics import ForwardKinematic

def euc_dist_3d(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

# Config
MODEL_DIR = "Model"
RESULTS_DIR = "RESULTS_PAPER"
os.makedirs(RESULTS_DIR, exist_ok=True)

# IMPORTANT: Change this to match the MDN model you want to test!
# If your model was trained with 10 mixtures, change this to 10.
OUTPUT_DIMS = 6
N_MIXES = 5  # or 10, depending on what you actually trained

joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist Pitch', 'Wrist Roll', 'Wrist Twist']

# Load Data
data = np.load('dataset_processed.npz')
X_test, y_test = data['X_test'], data['y_test']
y_test_deg = np.rad2deg(y_test)

# ==========================================
# 1. LOAD MODELS (Explicitly declared to prevent glob mixups)
# ==========================================
print("Loading models...")

# --- EXPLICIT PATHS ---
# 1. Put the EXACT name of your MDN model here
MDN_MODEL_FILE = "best_model_20260730_091004_MDN.keras" # <--- CHANGE THIS TO YOUR ACTUAL MDN FILE
# 2. Put the EXACT name of your Standard DNN model here
DNN_MODEL_FILE = "best_model_20260726_204920_MuOn.keras" # <--- CHANGE THIS TO YOUR ACTUAL DNN FILE

# Load MDN
mdn_path = os.path.join(MODEL_DIR, MDN_MODEL_FILE)
if not os.path.exists(mdn_path):
    # Fallback to glob if exact file not found, but filter for _MDN
    mdn_files = glob.glob(os.path.join(MODEL_DIR, "*_MDN.keras"))
    if not mdn_files: raise FileNotFoundError(f"Could not find MDN model in {MODEL_DIR}")
    mdn_path = max(mdn_files, key=os.path.getctime)
    
print(f"Loading MDN from: {mdn_path}")
mdn_model = keras.models.load_model(mdn_path, custom_objects={'MDN': mdn.MDN}, compile=False)

# Load DNN
dnn_path = os.path.join(MODEL_DIR, DNN_MODEL_FILE)
if not os.path.exists(dnn_path):
    # Fallback to glob, but strictly ignore any file with "MDN" in the name
    dnn_files = [f for f in glob.glob(os.path.join(MODEL_DIR, "best_model_*.keras")) if "MDN" not in os.path.basename(f)]
    if not dnn_files: 
        print("Warning: No Standard DNN model found. Skipping DNN evaluation.")
        dnn_model = None
    else:
        dnn_path = max(dnn_files, key=os.path.getctime)

if dnn_path:
    print(f"Loading Standard DNN from: {dnn_path}")
    dnn_model = keras.models.load_model(dnn_path, compile=False)
else:
    dnn_model = None


# ==========================================
# 2. MDN INDIVIDUAL MIXTURE ANALYSIS
# ==========================================
print("\n--- Analyzing MDN Mixtures ---")
y_pred_params = mdn_model.predict(X_test, verbose=0)
mu_size = OUTPUT_DIMS * N_MIXES
mus = y_pred_params[:, :mu_size]
pis = y_pred_params[:, -N_MIXES:]
mus_reshaped = mus.reshape(-1, N_MIXES, OUTPUT_DIMS)

# Selected Mixture (Highest pi)
best_pi_idx = np.argmax(pis, axis=1)
y_pred_selected = mus_reshaped[np.arange(len(y_pred_params)), best_pi_idx, :]

# Best Mixture (Oracle)
best_mae_mus = np.zeros_like(y_test)
for i in range(len(y_test)):
    errors = np.sum(np.abs(mus_reshaped[i] - y_test[i]), axis=1)
    best_mae_mus[i] = mus_reshaped[i, np.argmin(errors), :]

# Evaluate each mixture individually
mixture_metrics = []
for k in range(N_MIXES):
    y_pred_k = mus_reshaped[:, k, :]
    mae_k = mean_absolute_error(y_test_deg, np.rad2deg(y_pred_k), multioutput='raw_values')
    dists_k = [euc_dist_3d(ForwardKinematic(*p)[-3:], ForwardKinematic(*t)[-3:]) for p, t in zip(y_pred_k, y_test)]
    
    mixture_metrics.append({
        'Mixture': f'Mixture {k}',
        'Mean MAE (deg)': np.mean(mae_k),
        'Mean Euc Dist (cm)': np.mean(dists_k)
    })

# Add Selected and Best to the table
mixture_metrics.append({
    'Mixture': 'Selected (Highest pi)',
    'Mean MAE (deg)': np.mean(mean_absolute_error(y_test_deg, np.rad2deg(y_pred_selected), multioutput='raw_values')),
    'Mean Euc Dist (cm)': np.mean([euc_dist_3d(ForwardKinematic(*p)[-3:], ForwardKinematic(*t)[-3:]) for p, t in zip(y_pred_selected, y_test)])
})
mixture_metrics.append({
    'Mixture': 'Best (Oracle)',
    'Mean MAE (deg)': np.mean(mean_absolute_error(y_test_deg, np.rad2deg(best_mae_mus), multioutput='raw_values')),
    'Mean Euc Dist (cm)': np.mean([euc_dist_3d(ForwardKinematic(*p)[-3:], ForwardKinematic(*t)[-3:]) for p, t in zip(best_mae_mus, y_test)])
})

mdn_df = pd.DataFrame(mixture_metrics)
print(mdn_df.to_string(index=False))
mdn_df.to_csv(os.path.join(RESULTS_DIR, 'mdn_mixture_analysis.csv'), index=False)

# ==========================================
# 3. DNN COMPARISON
# ==========================================
if dnn_model:
    print("\n--- Analyzing Standard DNN ---")
    y_pred_dnn = dnn_model.predict(X_test, verbose=0)
    mae_dnn = mean_absolute_error(y_test_deg, np.rad2deg(y_pred_dnn), multioutput='raw_values')
    dists_dnn = [euc_dist_3d(ForwardKinematic(*p)[-3:], ForwardKinematic(*t)[-3:]) for p, t in zip(y_pred_dnn, y_test)]
    
    dnn_metrics = pd.DataFrame([{
        'Model': 'Standard DNN (MSE)',
        'Mean MAE (deg)': np.mean(mae_dnn),
        'Mean Euc Dist (cm)': np.mean(dists_dnn)
    }])
    print(dnn_metrics.to_string(index=False))
    dnn_metrics.to_csv(os.path.join(RESULTS_DIR, 'dnn_analysis.csv'), index=False)

# ==========================================
# 4. INFERENCE TIME BENCHMARKING
# ==========================================
print("\n--- Benchmarking Inference Time (Simulating Raspberry Pi) ---")
# Take a random sample of 1000 points to average the time
sample_X = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]

# MDN Time
start_time = time.perf_counter()
mdn_model.predict(sample_X, verbose=0, batch_size=1)
mdn_time_ms = (time.perf_counter() - start_time) / 10  # divided by 1000 samples, * 1000 for ms -> /10

# DNN Time
if dnn_model:
    start_time = time.perf_counter()
    dnn_model.predict(sample_X, verbose=0, batch_size=1)
    dnn_time_ms = (time.perf_counter() - start_time) / 10

    time_df = pd.DataFrame([
        {'Model': 'MDN', 'Inference Time per Sample (ms)': mdn_time_ms},
        {'Model': 'DNN', 'Inference Time per Sample (ms)': dnn_time_ms}
    ])
    print(time_df.to_string(index=False))
    time_df.to_csv(os.path.join(RESULTS_DIR, 'inference_time.csv'), index=False)

# ==========================================
# 5. ACADEMIC PLOT: MULTIMODALITY VISUALIZATION
# ==========================================
print("\nGenerating multimodality plot for paper...")
# We plot the Elbow (Index 2) because it usually shows the most redundancy (Elbow Up/Down)
joint_idx = 2 
joint_name = joint_names[joint_idx]

fig, ax = plt.subplots(figsize=(8, 8))

# Plot each mixture as a different color
colors = ['red', 'blue', 'green', 'purple', 'orange']
for k in range(N_MIXES):
    ax.scatter(y_test_deg[:, joint_idx], np.rad2deg(mus_reshaped[:, k, joint_idx]), 
               alpha=0.15, s=10, color=colors[k], label=f'Mixture {k}')

# Plot DNN if available
if dnn_model:
    ax.scatter(y_test_deg[:, joint_idx], np.rad2deg(y_pred_dnn[:, joint_idx]), 
               alpha=0.1, s=5, color='black', marker='x', label='Standard DNN (Averaged)')

# Ideal line
min_val, max_val = y_test_deg[:, joint_idx].min(), y_test_deg[:, joint_idx].max()
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal (y=x)')

ax.set_title(f'Multimodality in IK: {joint_name} Joint', fontsize=14)
ax.set_xlabel('Actual Angle (deg)', fontsize=12)
ax.set_ylabel('Predicted Angle (deg)', fontsize=12)
ax.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'multimodality_proof.png'), dpi=300)
plt.close()

print(f"All paper results saved to {RESULTS_DIR}")