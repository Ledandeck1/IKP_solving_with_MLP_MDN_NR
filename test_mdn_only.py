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
RESULTS_DIR = "RESULTS_MDN_ONLY"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_DIMS = 6
N_MIXES = 5
joint_names = ['Base', 'Shoulder', 'Elbow', 'Wrist Pitch', 'Wrist Roll', 'Wrist Twist']

# Load Data
data = np.load('dataset_processed.npz')
X_test, y_test = data['X_test'], data['y_test']
y_test_deg = np.rad2deg(y_test)

# ==========================================
# 1. LOAD MDN MODEL
# ==========================================
print("Loading MDN model...")
MDN_MODEL_FILE = "best_model_20260730_091004_MDN.keras" 
mdn_path = os.path.join(MODEL_DIR, MDN_MODEL_FILE)

if not os.path.exists(mdn_path):
    mdn_files = glob.glob(os.path.join(MODEL_DIR, "*_MDN.keras"))
    if not mdn_files: raise FileNotFoundError(f"Could not find MDN model in {MODEL_DIR}")
    mdn_path = max(mdn_files, key=os.path.getctime)

mdn_model = keras.models.load_model(mdn_path, custom_objects={'MDN': mdn.MDN}, compile=False)

# ==========================================
# 2. PREDICT AND EXTRACT MIXTURES
# ==========================================
print("\nPredicting distributions...")
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

# ==========================================
# 3. INDIVIDUAL MIXTURE ANALYSIS
# ==========================================
print("\n--- Analyzing Each Mixture Individually ---")
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

# Calculate distances for Selected and Oracle explicitly for plots
selected_dists = [euc_dist_3d(ForwardKinematic(*p)[-3:], ForwardKinematic(*t)[-3:]) for p, t in zip(y_pred_selected, y_test)]
oracle_dists = [euc_dist_3d(ForwardKinematic(*p)[-3:], ForwardKinematic(*t)[-3:]) for p, t in zip(best_mae_mus, y_test)]

# Add Selected and Best to the table
mixture_metrics.append({
    'Mixture': 'Selected (Highest pi)',
    'Mean MAE (deg)': np.mean(mean_absolute_error(y_test_deg, np.rad2deg(y_pred_selected), multioutput='raw_values')),
    'Mean Euc Dist (cm)': np.mean(selected_dists)
})
mixture_metrics.append({
    'Mixture': 'Best (Oracle)',
    'Mean MAE (deg)': np.mean(mean_absolute_error(y_test_deg, np.rad2deg(best_mae_mus), multioutput='raw_values')),
    'Mean Euc Dist (cm)': np.mean(oracle_dists)
})

mdn_df = pd.DataFrame(mixture_metrics)
print(mdn_df.to_string(index=False))
mdn_df.to_csv(os.path.join(RESULTS_DIR, 'mdn_mixture_analysis.csv'), index=False)


# ==========================================
# 4. INFERENCE TIME BENCHMARKING (PER MIXTURE)
# ==========================================
print("\n--- Benchmarking Inference Time per Mixture ---")
sample_X = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]

# Warmup (GPU/CPU initialization)
mdn_model.predict(sample_X[:10], verbose=0, batch_size=1)

# 1. Time the full network prediction (Calculates ALL mixtures)
start_time = time.perf_counter()
y_pred_sample = mdn_model.predict(sample_X, verbose=0, batch_size=1)
full_pred_time_ms = (time.perf_counter() - start_time) / 10  # 1000 samples -> /10 to get ms/sample

mus_sample = y_pred_sample[:, :mu_size].reshape(-1, N_MIXES, OUTPUT_DIMS)
pis_sample = y_pred_sample[:, -N_MIXES:]

time_metrics = []

# 2. Time extraction for each individual mixture
for k in range(N_MIXES):
    start_time = time.perf_counter()
    _ = mus_sample[:, k, :] # Simulate extracting mixture k
    extract_time = (time.perf_counter() - start_time) / 10
    total_time = full_pred_time_ms + extract_time
    time_metrics.append({
        'Mixture': f'Mixture {k}',
        'NN Predict Time (ms)': full_pred_time_ms,
        'Extraction Time (ms)': extract_time,
        'Total Time (ms)': total_time
    })

# 3. Time extraction for Selected Mixture (Requires argmax)
start_time = time.perf_counter()
best_pi_sample = np.argmax(pis_sample, axis=1)
_ = mus_sample[np.arange(len(sample_X)), best_pi_sample, :]
extract_time = (time.perf_counter() - start_time) / 10
time_metrics.append({
    'Mixture': 'Selected (Deploy)',
    'NN Predict Time (ms)': full_pred_time_ms,
    'Extraction Time (ms)': extract_time,
    'Total Time (ms)': full_pred_time_ms + extract_time
})

time_df = pd.DataFrame(time_metrics)
print(time_df.to_string(index=False))
time_df.to_csv(os.path.join(RESULTS_DIR, 'inference_time_per_mixture.csv'), index=False)

# ==========================================
# 5. GENERATE ACADEMIC DASHBOARD
# ==========================================
print("\nGenerating Academic Dashboard...")
fig = plt.figure(figsize=(16, 12))
fig.suptitle('MDN Mixture Analysis Dashboard for 6-DOF Inverse Kinematics', fontsize=18, fontweight='bold')

# Panel A: Bar Chart - Euclidean Distance
ax1 = plt.subplot(2, 3, 1)
colors = ['#1f77b4'] * N_MIXES + ['#2ca02c', '#d62728']
ax1.bar(mdn_df['Mixture'], mdn_df['Mean Euc Dist (cm)'], color=colors)
ax1.set_title('A. Mean Spatial Error (Euclidean)', fontsize=12)
ax1.set_ylabel('Distance (cm)')
ax1.tick_params(axis='x', rotation=90)

# Panel B: Bar Chart - Joint MAE
ax2 = plt.subplot(2, 3, 2)
ax2.bar(mdn_df['Mixture'], mdn_df['Mean MAE (deg)'], color=colors)
ax2.set_title('B. Mean Joint Angle Error (MAE)', fontsize=12)
ax2.set_ylabel('MAE (degrees)')
ax2.tick_params(axis='x', rotation=90)

# Panel C: Box Plot - Distribution of Selected vs Oracle
ax3 = plt.subplot(2, 3, 3)
box_data = [selected_dists, oracle_dists]
box = ax3.boxplot(box_data, patch_artist=True, tick_labels=['Selected', 'Oracle (Best)'])
for patch, color in zip(box['boxes'], ['#2ca02c', '#d62728']):
    patch.set_facecolor(color)
ax3.set_title('C. Spatial Error Distribution', fontsize=12)
ax3.set_ylabel('Euclidean Distance (cm)')
ax3.set_ylim(0, np.percentile(oracle_dists, 95)) # Zoom in to ignore massive outliers

# Panel D: Histogram - Error overlap
ax4 = plt.subplot(2, 3, 4)
ax4.hist(selected_dists, bins=50, color='#2ca02c', alpha=0.6, label='Selected (Deploy)')
ax4.hist(oracle_dists, bins=50, color='#d62728', alpha=0.5, label='Oracle (Best)')
ax4.set_title('D. Histogram of Spatial Errors', fontsize=12)
ax4.set_xlabel('Euclidean Distance (cm)')
ax4.set_ylabel('Frequency')
ax4.legend()

# Panel E: Multimodality Proof
ax5 = plt.subplot(2, 3, 5)
joint_idx = 2 # Elbow
y_pred_selected_deg = np.rad2deg(y_pred_selected)

sc = ax5.scatter(y_test_deg[:, joint_idx], y_pred_selected_deg[:, joint_idx], 
                 c=best_pi_idx, cmap='viridis', vmin=-0.5, vmax=N_MIXES-0.5, alpha=0.6, s=10)

min_val = min(y_test_deg[:, joint_idx].min(), y_pred_selected_deg[:, joint_idx].min())
max_val = max(y_test_deg[:, joint_idx].max(), y_pred_selected_deg[:, joint_idx].max())
ax5.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

cbar = plt.colorbar(sc, ax=ax5, ticks=range(N_MIXES))
cbar.set_label('Selected Mixture Index')
ax5.set_title(f'E. Multimodality Proof: {joint_names[joint_idx]} Joint', fontsize=12)
ax5.set_xlabel('Actual Angle (deg)')
ax5.set_ylabel('Predicted Angle (deg)')

# Panel F: Inference Time Table (Text plot)
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

sel_time = time_df[time_df['Mixture'] == 'Selected (Deploy)']['Total Time (ms)'].values[0]

table_text = f"DEPLOYMENT METRICS\n\n"
table_text += f"Model Type: Mixture Density Network\n"
table_text += f"Mixtures (K): {N_MIXES}\n"
table_text += f"Parameters: 6-DOF (3D -> 6D)\n\n"
table_text += f"Mean Inference Time:\n{sel_time:.3f} ms / sample\n\n"
table_text += f"Deployed Strategy:\nSelected Mixture (Highest Pi)\n\n"
table_text += f"Deployed Mean Error:\n{np.mean(selected_dists):.2f} cm (Euclidean)"

ax6.text(0.1, 0.5, table_text, fontsize=14, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
dashboard_path = os.path.join(RESULTS_DIR, 'mdn_academic_dashboard.png')
plt.savefig(dashboard_path, dpi=300)
plt.close()

print(f"\nDashboard saved to: {dashboard_path}")
print("Evaluation complete.")