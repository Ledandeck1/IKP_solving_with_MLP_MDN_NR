# evaluate_all.py
"""
Comprehensive scientific evaluation for IK neural network comparison.
FIXED: skips old models with incompatible input shapes.
"""
import os, glob, json, time, re
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
import keras_mdn_layer as mdn
from forwardkinematics import ForwardKinematic

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR      = "Model"
OUTPUT_DIR     = "RESULTS_FINAL"
MAX_ERROR_CM   = 5.0
TAUS           = [1.0, 2.0, 5.0]
N_TIMING       = 200
NR_MAX_ITER    = 3
NR_LAMBDA     = 1e-3
NR_TOL         = 1e-3
JOINT_NAMES    = ['Base', 'Shoulder', 'Elbow', 'Wrist Pitch',
                  'Wrist Roll', 'Wrist Twist']
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.dpi': 200, 'font.size': 11, 'font.family': 'serif',
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
})
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def euc_dist(pred_angles, true_angles):
    p = ForwardKinematic(*pred_angles)
    t = ForwardKinematic(*true_angles)
    return np.sqrt((p[0]-t[0])**2 + (p[1]-t[1])**2 + (p[2]-t[2])**2)

def compute_jacobian(thetas):
    eps = 1e-4
    J = np.zeros((3, 6))
    cur = ForwardKinematic(*thetas)
    for i in range(6):
        t = thetas.copy(); t[i] += eps
        J[:, i] = (ForwardKinematic(*t) - cur) / eps
    return J

def newton_raphson(theta_0, target_xyz, max_iter=NR_MAX_ITER, tol=NR_TOL):
    theta = np.array(theta_0, dtype=np.float64)
    for i in range(max_iter):
        cur = ForwardKinematic(*theta)
        err = target_xyz - cur
        if np.linalg.norm(err) < tol:
            return theta, i
        J = compute_jacobian(theta)
        JJT = J @ J.T + NR_LAMBDA * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(JJT)
        theta = theta + J_pinv @ err
    return theta, max_iter

def parse_model_name(filename):
    base = os.path.basename(filename)
    arch = 'mdn' if ('mdn' in base.lower() or 'MDN' in base) else 'mlp'
    opt = 'unknown'
    for o in ['adamw', 'muon', 'shampoo']:
        if o in base.lower(): opt = o; break
    seed_match = re.search(r'seed(\d+)', base)
    seed = int(seed_match.group(1)) if seed_match else -1
    return arch, opt, seed

def get_input_dim(model):
    """Safely get the expected input dimension from a Keras model."""
    try:
        ishape = model.input_shape
        if isinstance(ishape, list):
            ishape = ishape[0]
        dim = ishape[-1] if ishape and ishape[-1] is not None else None
        return dim
    except:
        return None

def measure_inference_time(model, X, n=N_TIMING):
    indices = np.random.choice(len(X), min(n, len(X)), replace=False)
    _ = model.predict(X[:1], verbose=0)
    times = []
    for idx in indices:
        t0 = time.perf_counter()
        _ = model.predict(X[idx:idx+1], verbose=0)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.mean(times), np.std(times)

# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("COMPREHENSIVE IK EVALUATION")
print("=" * 70)

data = np.load('dataset_processed.npz')
X_test = data['X_test']
y_test = data['y_test']
scaler_X = joblib.load('scaler_X.pkl')
X_test_unscaled = scaler_X.inverse_transform(X_test)
EXPECTED_INPUT_DIM = X_test.shape[1]
print(f"Test samples: {len(X_test)}")
print(f"Expected input dim: {EXPECTED_INPUT_DIM}")

# ============================================================
# DISCOVER & FILTER MODELS
# ============================================================
all_model_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "best*run*.keras")))
print(f"Found {len(all_model_paths)} .keras files in {MODEL_DIR}/")

# Pre-filter: load each model briefly to check input shape
compatible_models = []
skipped_count = 0
for mp in all_model_paths:
    arch, opt, seed = parse_model_name(mp)
    custom = {'MDN': mdn.MDN} if arch == 'mdn' else {}
    try:
        m = keras.models.load_model(mp, custom_objects=custom, compile=False)
        input_dim = get_input_dim(m)
        if input_dim == EXPECTED_INPUT_DIM:
            compatible_models.append(mp)
        else:
            print(f"  [SKIP] {os.path.basename(mp)}: expects {input_dim} inputs, need {EXPECTED_INPUT_DIM}")
            skipped_count += 1
        del m  # free memory
    except Exception as e:
        print(f"  [SKIP] {os.path.basename(mp)}: load error: {e}")
        skipped_count += 1

print(f"\nCompatible models: {len(compatible_models)}")
print(f"Skipped (old/incompatible): {skipped_count}")
all_models = compatible_models

if not all_models:
    print("\nERROR: No compatible models found. Check your Model/ directory.")
    exit(1)

# ============================================================
# EVALUATE EACH MODEL
# ============================================================
all_results = []

for mi, model_path in enumerate(all_models):
    arch, opt, seed = parse_model_name(model_path)
    name = os.path.basename(model_path).replace('.keras', '')
    print(f"\n[{mi+1}/{len(all_models)}] {name}")
    print(f"  Arch={arch}  Opt={opt}  Seed={seed}")

    custom = {'MDN': mdn.MDN} if arch == 'mdn' else {}
    model = keras.models.load_model(model_path, custom_objects=custom, compile=False)

    # Inference time
    inf_mean, inf_std = measure_inference_time(model, X_test)
    print(f"  Inference: {inf_mean:.3f} ± {inf_std:.3f} ms/sample")

    # Predict
    if arch == 'mdn':
        params = model.predict(X_test, verbose=0)
        N_MIXES = 20
        mu_size = 6 * N_MIXES
        mus = params[:, :mu_size].reshape(-1, N_MIXES, 6)
        pis = params[:, -N_MIXES:]
        pis = np.exp(pis - pis.max(axis=1, keepdims=True))
        pis = pis / pis.sum(axis=1, keepdims=True)
        sel_idx = np.argmax(pis, axis=1)
        y_pred = mus[np.arange(len(mus)), sel_idx, :]
        l1 = np.abs(mus - y_test[:, None, :]).sum(axis=2)
        ora_idx = np.argmin(l1, axis=1)
        y_ora = mus[np.arange(len(mus)), ora_idx, :]
    else:
        y_pred = model.predict(X_test, verbose=0)
        y_ora = None

    # Euclidean errors
    errors_sel = np.array([euc_dist(y_pred[i], y_test[i])
                           for i in range(len(y_test))])
    if y_ora is not None:
        errors_ora = np.array([euc_dist(y_ora[i], y_test[i])
                               for i in range(len(y_test))])
    else:
        errors_ora = None

    # Filter outliers > MAX_ERROR_CM
    mask = errors_sel <= MAX_ERROR_CM
    n_filtered = len(errors_sel) - mask.sum()
    errors_sel_f = errors_sel[mask]
    errors_ora_f = errors_ora[mask] if errors_ora is not None else None
    print(f"  Filtered {n_filtered} outliers (>{MAX_ERROR_CM} cm)")

    # Joint metrics
    ytd = np.rad2deg(y_test[mask])
    ypd = np.rad2deg(y_pred[mask])
    mae = mean_absolute_error(ytd, ypd, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(ytd, ypd, multioutput='raw_values'))
    succ = {f'{tau:.0f}cm': float(np.mean(errors_sel_f <= tau) * 100)
            for tau in TAUS}

    res = {
        'method': arch, 'optimizer': opt, 'seed': seed,
        'model_path': model_path,
        'euclid_mean': float(errors_sel_f.mean()),
        'euclid_std': float(errors_sel_f.std()),
        'euclid_median': float(np.median(errors_sel_f)),
        'euclid_p95': float(np.percentile(errors_sel_f, 95)),
        'n_filtered': int(n_filtered),
        'mae_mean_deg': float(mae.mean()),
        'rmse_mean_deg': float(rmse.mean()),
        'succ_1cm': succ['1cm'], 'succ_2cm': succ['2cm'],
        'succ_5cm': succ['5cm'],
        'inf_time_ms': float(inf_mean),
        'inf_time_std_ms': float(inf_std),
        'errors_filtered': errors_sel_f,
    }
    if errors_ora_f is not None:
        res['oracle_mean'] = float(errors_ora_f.mean())
        res['oracle_p95'] = float(np.percentile(errors_ora_f, 95))
        res['errors_oracle_filtered'] = errors_ora_f
    all_results.append(res)

    # ---- If MLP: run hybrid (MLP + Newton-Raphson) ----
    # ---- If MLP: run hybrid (MLP + Newton-Raphson) ----
    if arch == 'mlp':
        N_HYBRID = min(10000, len(X_test))
        print(f"  Running hybrid (MLP+NR, {N_HYBRID} samples, max {NR_MAX_ITER} iter)...")

        # BATCH predict all at once — ONE call, no segfault
        theta_0_all = model.predict(X_test[:N_HYBRID], verbose=0)

        hybrid_errors = []
        hybrid_iters = []
        nr_times = []
        t_nr0 = time.perf_counter()
        for i in range(N_HYBRID):
            if i % 2000 == 0 and i > 0:
                print(f"    progress: {i}/{N_HYBRID}")
            
            theta_ref, iters = newton_raphson(theta_0_all[i], X_test_unscaled[i])
            t_nr1 = time.perf_counter()
            hybrid_errors.append(euc_dist(theta_ref, y_test[i]))
            hybrid_iters.append(iters)
        t_nr1 = time.perf_counter()
        nr_times.append((t_nr1 - t_nr0) * 1000)
        hybrid_errors = np.array(hybrid_errors)
        mask_h = hybrid_errors <= MAX_ERROR_CM
        hybrid_errors_f = hybrid_errors[mask_h]
        hybrid_iters_arr = np.array(hybrid_iters)
        #hybrid_times_arr = np.array(hybrid_times)
        nr_times_arr = np.array(nr_times)
        # Total hybrid time = MLP inference + NR computation
        hybrid_times_arr = np.full(len(nr_times_arr), inf_mean + np.mean(nr_times_arr))
        print(f"  Hybrid: mean={hybrid_errors_f.mean():.4f} cm, "
              f"iters={hybrid_iters_arr.mean():.2f}, "
              f"time={hybrid_times_arr.mean():.3f} ms")
        res_h = {
            'method': 'hybrid', 'optimizer': opt, 'seed': seed,
            'model_path': model_path,
            'euclid_mean': float(hybrid_errors_f.mean()),
            'euclid_std': float(hybrid_errors_f.std()),
            'euclid_median': float(np.median(hybrid_errors_f)),
            'euclid_p95': float(np.percentile(hybrid_errors_f, 95)),
            'n_filtered': int(len(hybrid_errors) - mask_h.sum()),
            'mae_mean_deg': float('nan'), 'rmse_mean_deg': float('nan'),
            'succ_1cm': float(np.mean(hybrid_errors_f <= 1.0) * 100),
            'succ_2cm': float(np.mean(hybrid_errors_f <= 2.0) * 100),
            'succ_5cm': float(np.mean(hybrid_errors_f <= 5.0) * 100),
            'inf_time_ms': float(hybrid_times_arr.mean()),
            'inf_time_std_ms': float(hybrid_times_arr.std()),
            'nr_iters_mean': float(hybrid_iters_arr.mean()),
            'errors_filtered': hybrid_errors_f,
        }
        all_results.append(res_h)

# ============================================================
# NR-ONLY BASELINE (random initial guess)
# ============================================================
print("\n[NR-ONLY baseline: random init, max 10 iter]")
nr_errors, nr_iters, nr_times = [], [], []
np.random.seed(42)
for i in range(min(5000, len(X_test))):
    theta_rand = np.random.uniform(-1.5, 1.5, 6)
    t0 = time.perf_counter()
    theta_ref, iters = newton_raphson(theta_rand, X_test_unscaled[i],
                                      max_iter=10, tol=NR_TOL)
    t1 = time.perf_counter()
    nr_errors.append(euc_dist(theta_ref, y_test[i]))
    nr_iters.append(iters)
    nr_times.append((t1 - t0) * 1000)
nr_errors = np.array(nr_errors)
nr_iters = np.array(nr_iters)
nr_times = np.array(nr_times)
mask_nr = nr_errors <= MAX_ERROR_CM
nr_errors_f = nr_errors[mask_nr]
print(f"  NR-only: mean={nr_errors_f.mean():.4f} cm, "
      f"iters={nr_iters.mean():.2f}, time={nr_times.mean():.3f} ms")
all_results.append({
    'method': 'nr_only', 'optimizer': 'baseline', 'seed': 0,
    'model_path': 'N/A',
    'euclid_mean': float(nr_errors_f.mean()),
    'euclid_std': float(nr_errors_f.std()),
    'euclid_median': float(np.median(nr_errors_f)),
    'euclid_p95': float(np.percentile(nr_errors_f, 95)),
    'n_filtered': int(len(nr_errors) - mask_nr.sum()),
    'mae_mean_deg': float('nan'), 'rmse_mean_deg': float('nan'),
    'succ_1cm': float(np.mean(nr_errors_f <= 1.0) * 100),
    'succ_2cm': float(np.mean(nr_errors_f <= 2.0) * 100),
    'succ_5cm': float(np.mean(nr_errors_f <= 5.0) * 100),
    'inf_time_ms': float(nr_times.mean()),
    'inf_time_std_ms': float(nr_times.std()),
    'nr_iters_mean': float(nr_iters.mean()),
    'errors_filtered': nr_errors_f,
})

# ============================================================
# BUILD DATAFRAME
# ============================================================
df = pd.DataFrame([{k: v for k, v in r.items()
                    if k != 'errors_filtered' and k != 'errors_oracle_filtered'}
                   for r in all_results])
df.to_csv(os.path.join(OUTPUT_DIR, 'all_results.csv'), index=False)

# ============================================================
# PUBLICATION-QUALITY PLOTS
# ============================================================
print("\nGenerating publication-quality plots...")
opt_order = ['adamw', 'muon', 'shampoo']

# --- PLOT 1: Boxplot MLP by optimizer ---
fig, ax = plt.subplots(figsize=(8, 5))
plot_data = []
for opt in opt_order:
    for r in all_results:
        if r['method'] == 'mlp' and r['optimizer'] == opt:
            for e in r['errors_filtered']:
                plot_data.append({'Optimizer': opt.upper(),
                                  'Euclidean Error (cm)': e})
if plot_data:
    pdf = pd.DataFrame(plot_data)
    sns.boxplot(data=pdf, x='Optimizer', y='Euclidean Error (cm)',
                order=[o.upper() for o in opt_order], palette='Set2', ax=ax)
    sns.stripplot(data=pdf, x='Optimizer', y='Euclidean Error (cm)',
                  order=[o.upper() for o in opt_order],
                  color='black', alpha=0.4, size=3, ax=ax, dodge=True)
    ax.set_title(f'MLP: Euclidean Error by Optimizer (filtered ≤{MAX_ERROR_CM} cm)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_mlp_optimizers.png'), dpi=200)
plt.close()

# --- PLOT 2: Boxplot MDN by optimizer ---
fig, ax = plt.subplots(figsize=(8, 5))
plot_data = []
for opt in opt_order:
    for r in all_results:
        if r['method'] == 'mdn' and r['optimizer'] == opt:
            for e in r['errors_filtered']:
                plot_data.append({'Optimizer': opt.upper(),
                                  'Euclidean Error (cm)': e})
if plot_data:
    pdf = pd.DataFrame(plot_data)
    sns.boxplot(data=pdf, x='Optimizer', y='Euclidean Error (cm)',
                order=[o.upper() for o in opt_order], palette='Set3', ax=ax)
    sns.stripplot(data=pdf, x='Optimizer', y='Euclidean Error (cm)',
                  order=[o.upper() for o in opt_order],
                  color='black', alpha=0.4, size=3, ax=ax, dodge=True)
    ax.set_title(f'MDN: Euclidean Error by Optimizer (filtered ≤{MAX_ERROR_CM} cm)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_mdn_optimizers.png'), dpi=200)
plt.close()

# --- PLOT 3: Boxplot Hybrid by optimizer ---
fig, ax = plt.subplots(figsize=(8, 5))
plot_data = []
for opt in opt_order:
    for r in all_results:
        if r['method'] == 'hybrid' and r['optimizer'] == opt:
            for e in r['errors_filtered']:
                plot_data.append({'Optimizer': opt.upper(),
                                  'Euclidean Error (cm)': e})
if plot_data:
    pdf = pd.DataFrame(plot_data)
    sns.boxplot(data=pdf, x='Optimizer', y='Euclidean Error (cm)',
                order=[o.upper() for o in opt_order], palette='Set1', ax=ax)
    sns.stripplot(data=pdf, x='Optimizer', y='Euclidean Error (cm)',
                  order=[o.upper() for o in opt_order],
                  color='black', alpha=0.4, size=3, ax=ax, dodge=True)
    ax.set_title(f'Hybrid (MLP+NR): Euclidean Error by Optimizer (filtered ≤{MAX_ERROR_CM} cm)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_hybrid_optimizers.png'), dpi=200)
plt.close()

# --- PLOT 4: Method comparison (best of each) ---
fig, ax = plt.subplots(figsize=(9, 5))
methods_to_compare = []
for method_name, method_label in [('mlp', 'MLP'), ('mdn', 'MDN'),
                                   ('hybrid', 'MLP+NR'), ('nr_only', 'NR only')]:
    method_results = [r for r in all_results if r['method'] == method_name]
    if method_results:
        best = min(method_results, key=lambda x: x['euclid_mean'])
        for e in best['errors_filtered']:
            methods_to_compare.append({'Method': method_label,
                                        'Euclidean Error (cm)': e})
if methods_to_compare:
    pdf = pd.DataFrame(methods_to_compare)
    sns.boxplot(data=pdf, x='Method', y='Euclidean Error (cm)',
                palette='Set2', ax=ax)
    sns.stripplot(data=pdf, x='Method', y='Euclidean Error (cm)',
                  color='black', alpha=0.3, size=2, ax=ax)
    ax.set_title(f'Method Comparison: Best Model per Method (filtered ≤{MAX_ERROR_CM} cm)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_method_comparison.png'), dpi=200)
plt.close()

# --- PLOT 5: Clean side-by-side histograms (subplots, NOT overlaid) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_specs = [
    ('mlp', 'MLP (best)', 'selected'),
    ('hybrid', 'Hybrid MLP+NR (best)', 'selected'),
    ('mdn', 'MDN Selected (best)', 'selected'),
    ('mdn', 'MDN Oracle (best)', 'oracle'),
]
for ax_idx, (method_name, title, error_type) in enumerate(plot_specs):
    ax = axes[ax_idx // 2][ax_idx % 2]
    method_results = [r for r in all_results if r['method'] == method_name]
    if method_results:
        best = min(method_results, key=lambda x: x['euclid_mean'])
        if error_type == 'oracle' and 'errors_oracle_filtered' in best:
            errors = best['errors_oracle_filtered']
        else:
            errors = best['errors_filtered']
        ax.hist(errors, bins=80, color='steelblue', edgecolor='black', alpha=0.8)
        ax.axvline(np.mean(errors), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {np.mean(errors):.4f} cm')
        ax.axvline(np.median(errors), color='green', linestyle=':', linewidth=1.5,
                   label=f'Median: {np.median(errors):.4f} cm')
        ax.set_title(title)
        ax.set_xlabel('Euclidean Error (cm)')
        ax.set_ylabel('Frequency')
        ax.legend()
    else:
        ax.set_title(f'{title} (no data)')
        ax.set_visible(False)
fig.suptitle(f'Error Distribution by Method (filtered ≤{MAX_ERROR_CM} cm)', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(OUTPUT_DIR, 'hist_comparison_subplots.png'), dpi=200)
plt.close()

# --- PLOT 6: Parity plots (best MLP) ---
mlp_results = [r for r in all_results if r['method'] == 'mlp']
if mlp_results:
    best_mlp = min(mlp_results, key=lambda x: x['euclid_mean'])
    model = keras.models.load_model(best_mlp['model_path'], compile=False)
    y_pred = model.predict(X_test, verbose=0)
    ytd = np.rad2deg(y_test)
    ypd = np.rad2deg(y_pred)
    fig, axs = plt.subplots(3, 2, figsize=(11, 14))
    fig.suptitle('Parity Plots: Best MLP Model', fontsize=14)
    for i, ax in enumerate(axs.flat):
        ax.scatter(ytd[:, i], ypd[:, i], alpha=0.2, s=3, color='blue')
        mn = min(ytd[:, i].min(), ypd[:, i].min())
        mx = max(ytd[:, i].max(), ypd[:, i].max())
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5)
        j_mae = mean_absolute_error(ytd[:, i], ypd[:, i])
        ax.set_title(f'{JOINT_NAMES[i]} | MAE: {j_mae:.2f}°')
        ax.set_xlabel('Actual (°)')
        ax.set_ylabel('Predicted (°)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'parity_plots_best_mlp.png'), dpi=200)
    plt.close()

# --- PLOT 7: Success rate bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
methods_for_bars = []
for method_name, method_label in [('mlp', 'MLP'), ('mdn', 'MDN'),
                                  ('hybrid', 'MLP+NR'), ('nr_only', 'NR only')]:
    method_results = [r for r in all_results if r['method'] == method_name]
    if method_results:
        best = min(method_results, key=lambda x: x['euclid_mean'])
        methods_for_bars.append({
            'Method': method_label,
            'τ≤1cm': best['succ_1cm'],
            'τ≤2cm': best['succ_2cm'],
            'τ≤5cm': best['succ_5cm'],
        })
if methods_for_bars:
    pdf = pd.DataFrame(methods_for_bars)
    pdf_melted = pdf.melt(id_vars='Method', var_name='Threshold',
                          value_name='Success Rate (%)')
    sns.barplot(data=pdf_melted, x='Threshold', y='Success Rate (%)',
                hue='Method', errorbar=None, palette='Set2')
    ax.set_ylim(0, 105)
    ax.set_title('Task-Space Success Rate (best model per method)')
    ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'success_rate_bars.png'), dpi=200)
plt.close()

# --- PLOT 8: Inference time bar chart ---
fig, ax = plt.subplots(figsize=(10, 5))
time_data = []
colors_list = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
for mi, (method_name, method_label) in enumerate([
    ('mlp', 'MLP'), ('mdn', 'MDN'), ('hybrid', 'MLP+NR'), ('nr_only', 'NR only')]):
    method_results = [r for r in all_results if r['method'] == method_name]
    if method_results:
        best = min(method_results, key=lambda x: x['euclid_mean'])
        time_data.append({
            'Method': method_label,
            'Inference Time (ms)': best['inf_time_ms'],
            'Error (cm)': best['euclid_mean'],
            'Color': colors_list[mi],
        })
if time_data:
    pdf = pd.DataFrame(time_data)
    bars = ax.bar(pdf['Method'], pdf['Inference Time (ms)'],
                  color=pdf['Color'], edgecolor='black', alpha=0.85)
    for bar, err in zip(bars, pdf['Error (cm)']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{err:.3f} cm', ha='center', va='bottom', fontsize=10,
                fontweight='bold')
    ax.set_ylabel('Inference Time (ms/sample)')
    ax.set_title('Inference Time vs Precision (best model per method)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'inference_time_bars.png'), dpi=200)
plt.close()

# --- PLOT 9: MDN oracle vs selected ---
mdn_results = [r for r in all_results if r['method'] == 'mdn']
if mdn_results:
    best_mdn = min(mdn_results, key=lambda x: x['euclid_mean'])
    if 'errors_oracle_filtered' in best_mdn:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, max(MAX_ERROR_CM,
                        best_mdn['errors_filtered'].max()), 80)
        ax.hist(best_mdn['errors_filtered'], bins=bins, alpha=0.6,
                color='steelblue', edgecolor='black',
                label=f"Selected (μ={best_mdn['euclid_mean']:.3f} cm)")
        ax.hist(best_mdn['errors_oracle_filtered'], bins=bins, alpha=0.5,
                color='orange', edgecolor='black',
                label=f"Oracle (μ={best_mdn['oracle_mean']:.3f} cm)")
        ax.axvline(best_mdn['euclid_mean'], color='blue', ls='--', lw=1)
        ax.axvline(best_mdn['oracle_mean'], color='red', ls='--', lw=1)
        ax.set_xlabel('Euclidean Error (cm)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'MDN: Selected vs Oracle (filtered ≤{MAX_ERROR_CM} cm)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'hist_mdn_oracle_vs_selected.png'), dpi=200)
        plt.close()

# ============================================================
# PRINT SUMMARY TABLE
# ============================================================
print("\n" + "=" * 100)
print("SUMMARY TABLE — BEST MODEL PER METHOD")
print("=" * 100)
print(f"{'Method':<12} {'Opt':<10} {'Mean(cm)':<10} {'Std(cm)':<10} "
      f"{'P95(cm)':<10} {'1cm(%)':<8} {'5cm(%)':<8} {'Time(ms)':<10} {'Iters':<6}")
print("-" * 92)
for method_name, method_label in [('mlp', 'MLP'), ('mdn', 'MDN'),
                                  ('hybrid', 'MLP+NR'), ('nr_only', 'NR only')]:
    method_results = [r for r in all_results if r['method'] == method_name]
    if method_results:
        best = min(method_results, key=lambda x: x['euclid_mean'])
        iters = best.get('nr_iters_mean', '—')
        iters_str = f"{iters:.2f}" if isinstance(iters, float) else str(iters)
        print(f"{method_label:<12} {best['optimizer']:<10} "
              f"{best['euclid_mean']:<10.4f} {best['euclid_std']:<10.4f} "
              f"{best['euclid_p95']:<10.4f} {best['succ_1cm']:<8.1f} "
              f"{best['succ_5cm']:<8.1f} {best['inf_time_ms']:<10.3f} "
              f"{iters_str:<6}")

print("\n" + "=" * 100)
print("PER-OPTIMIZER BREAKDOWN (mean ± std across seeds)")
print("=" * 100)
for method_name, method_label in [('mlp', 'MLP'), ('mdn', 'MDN'),
                                  ('hybrid', 'MLP+NR')]:
    print(f"\n--- {method_label} ---")
    for opt in opt_order:
        opt_results = [r for r in all_results
                       if r['method'] == method_name and r['optimizer'] == opt]
        if opt_results:
            means = [r['euclid_mean'] for r in opt_results]
            print(f"  {opt.upper():<10} mean={np.mean(means):.4f} ± "
                  f"{np.std(means):.4f} cm  (n={len(means)} seeds)")

# ============================================================
# SAVE SUMMARY JSON
# ============================================================
summary = {}
for method_name, method_label in [('mlp', 'MLP'), ('mdn', 'MDN'),
                                  ('hybrid', 'MLP+NR'), ('nr_only', 'NR only')]:
    method_results = [r for r in all_results if r['method'] == method_name]
    if method_results:
        best = min(method_results, key=lambda x: x['euclid_mean'])
        summary[method_label] = {
            'optimizer': best['optimizer'], 'seed': best['seed'],
            'euclid_mean_cm': best['euclid_mean'],
            'euclid_std_cm': best['euclid_std'],
            'euclid_p95_cm': best['euclid_p95'],
            'succ_1cm': best['succ_1cm'], 'succ_2cm': best['succ_2cm'],
            'succ_5cm': best['succ_5cm'],
            'inf_time_ms': best['inf_time_ms'],
            'nr_iters_mean': best.get('nr_iters_mean', None),
            'n_filtered': best['n_filtered'],
        }
        if 'oracle_mean' in best:
            summary[method_label]['oracle_mean_cm'] = best['oracle_mean']

with open(os.path.join(OUTPUT_DIR, 'summary_best_per_method.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nAll results saved to {OUTPUT_DIR}/")
print("Done.")