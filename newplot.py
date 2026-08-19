# newplot.py

import os, json, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
warnings.filterwarnings('ignore')

OUT = "RESULTS_FINAL"; RUNS_DIR = "Runs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    'figure.dpi': 300, 'font.size': 11, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linestyle': '--',
    'axes.spines.top': False, 'axes.spines.right': False,
})
OPT_ORDER = ['adamw', 'muon', 'shampoo']
OPT_LABELS = {'adamw': 'AdamW', 'muon': 'MuOn', 'shampoo': 'Shampoo'}
METHOD_MAP = {'mlp': 'MLP', 'mdn': 'MDN', 'hybrid': 'MLP+NR', 'nr_only': 'NR-only'}
METHOD_ORDER = ['MLP', 'MDN', 'MLP+NR', 'NR-only']
PAL = {'MLP': '#0173B2', 'MDN': '#DE8F05', 'MLP+NR': '#029E73', 'NR-only': '#CC78BC'}
OPT_PAL = {'AdamW': '#0173B2', 'MuOn': '#029E73', 'Shampoo': '#DE8F05'}
JOINTS = ['Base', 'Shoulder', 'Elbow', 'Wrist Pitch', 'Wrist Roll', 'Wrist Twist']

def mboxplot_pure(ax, data_dict, order, colors, title='', ylabel=''):
    pos = list(range(1, len(order) + 1))
    dlist = [data_dict.get(g, np.array([])) for g in order]
    clist = [colors.get(g, '#888') for g in order]
    valid = [(p, d, c, g) for p, d, c, g in zip(pos, dlist, clist, order) if len(d) > 0]
    if not valid:
        ax.set_visible(False); return
    vp = [v[0] for v in valid]; vd = [v[1] for v in valid]
    vc = [v[2] for v in valid]; vg = [v[3] for v in valid]
    bp = ax.boxplot(vd, positions=vp, patch_artist=True, widths=0.5,
        showmeans=True, meanline=True, medianprops=dict(color='black', lw=1.5),
        meanprops=dict(color='red', lw=1, ls='--'), boxprops=dict(lw=1.2),
        whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
        flierprops=dict(marker='o', mfc='none', ms=4, alpha=0.5, mec='gray'))
    for patch, c in zip(bp['boxes'], vc):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    np.random.seed(42)
    for i, d in enumerate(vd):
        x = np.random.normal(vp[i], 0.06, size=len(d))
        ax.scatter(x, d, alpha=0.8, color='black', s=25, zorder=3)
    ax.set_xticks(vp); ax.set_xticklabels(vg)
    if title: ax.set_title(title, fontweight='bold', pad=10)
    if ylabel: ax.set_ylabel(ylabel)

def mkde_pure(ax, data, color, title='', xlabel='Euclidean Error (cm)', xlim=None):
    if len(data) < 2:
        ax.set_visible(False); return
    kde = gaussian_kde(data, bw_method=0.08)
    x_max = xlim[1] if xlim else max(data) * 1.1
    x = np.linspace(0, x_max, 300)
    y = kde(x)
    ax.plot(x, y, color=color, lw=2)
    ax.fill_between(x, 0, y, alpha=0.3, color=color)
    ax.axvline(np.mean(data), color='red', ls='--', lw=1.5, label=f'$\\mu$={np.mean(data):.4f}')
    ax.axvline(np.median(data), color='green', ls=':', lw=1.5, label=f'med={np.median(data):.4f}')
    if title: ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel('Density'); ax.legend(fontsize=8)
    if xlim: ax.set_xlim(xlim)


df = pd.read_csv(f'{OUT}/all_results.csv')
df['method_label'] = df['method'].map(METHOD_MAP)
df['optimizer_label'] = df['optimizer'].str.upper().map(OPT_LABELS).fillna(df['optimizer'].str.upper())
print(f"Loaded {len(df)} rows")

# TABLE 1: Erro Euclidiano (mean ± std, best in bold, NO P95)
print("[Table 1] Euclidean error comparison...")
main = df[df['method'] != 'nr_only']
best_err = main.groupby(['method', 'optimizer'])['euclid_mean'].mean().min()
tex = [r"\begin{table}[H]", r"\centering",
    r"\caption{Euclidean error comparison (mean $\pm$ std, $n=10$ seeds, filtered $\leq$10 cm). Best in \textbf{bold}.}",
    r"\label{tab:euclid}", r"\begin{tabular}{llc}", r"\toprule",
    r"Method & Optimizer & Error (cm) \\", r"\midrule"]
for method in ['mlp', 'mdn', 'hybrid', 'nr_only']:
    ml = METHOD_MAP[method]
    for opt in OPT_ORDER + ['baseline']:
        s = df[(df['method'] == method) & (df['optimizer'] == opt)]
        if s.empty: continue
        m, sd = s['euclid_mean'].mean(), s['euclid_mean'].std()
        e = f"${m:.4f} \\pm {sd:.4f}$"
        if abs(m - best_err) < 1e-4: e = f"\\textbf{{{e}}}"
        tex.append(f"{ml} & {opt.upper()} & {e} \\\\")
        if method == 'mdn' and opt != 'baseline' and not pd.isna(s['oracle_mean'].mean()):
            om = s['oracle_mean'].mean()
            tex.append(r"\quad (oracle) & & " + f"${om:.4f}$ \\\\")
    tex.append(r"\midrule")
tex[-1] = r"\bottomrule"
tex += [r"\end{tabular}", r"\end{table}"]
with open(f'{OUT}/table1_euclidean.tex', 'w') as f: f.write('\n'.join(tex))

# TABLE 2: Pairwise t-test p-values between methods
print("[Table 2] t-test between methods...")
method_means = {}
for method in ['mlp', 'mdn', 'hybrid', 'nr_only']:
    s = df[df['method'] == method]
    if s.empty: continue
    bo = s.groupby('optimizer')['euclid_mean'].mean().idxmin()
    vals = s[s['optimizer'] == bo].sort_values('seed')['euclid_mean'].values
    method_means[METHOD_MAP[method]] = vals
tex2 = [r"\begin{table}[H]", r"\centering",
    r"\caption{Pairwise Student's $t$-test between methods (best optimizer, $n=10$ seeds).}",
    r"\label{tab:ttest}", r"\begin{tabular}{lcccc}", r"\toprule",
    r" & MLP & MDN & MLP+NR & NR-only \\", r"\midrule"]
methods_present = [m for m in METHOD_ORDER if m in method_means]
for m1 in methods_present:
    row = [m1]
    for m2 in methods_present:
        if m1 == m2: row.append("—")
        elif m1 in method_means and m2 in method_means:
            a, b = method_means[m1], method_means[m2]
            n = min(len(a), len(b))
            if n >= 5:
                t_s, t_p = stats.ttest_rel(a[:n], b[:n])
                sig = '***' if t_p < 0.001 else ('**' if t_p < 0.01 else ('*' if t_p < 0.05 else 'ns'))
                row.append(f"${t_p:.4f}$ ({sig})")
            else: row.append("—")
        else: row.append("—")
    tex2.append(" & ".join(row) + r" \\")
tex2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(f'{OUT}/table2_ttest.tex', 'w') as f: f.write('\n'.join(tex2))
print("\n" + "=" * 70); print("PAIRWISE t-TEST RESULTS"); print("=" * 70)
for i, m1 in enumerate(methods_present):
    for m2 in methods_present[i+1:]:
        a, b = method_means[m1], method_means[m2]
        n = min(len(a), len(b))
        if n >= 5:
            t_s, t_p = stats.ttest_rel(a[:n], b[:n])
            sig = '***' if t_p < 0.001 else ('**' if t_p < 0.01 else ('*' if t_p < 0.05 else 'ns'))
            print(f"  {m1:<10} vs {m2:<10}: t={t_s:+.3f}, p={t_p:.6f}  {sig}")

# TABLE 3: Inference Time 
print("[Table 3] Inference time...")
tex3 = [r"\begin{table}[H]", r"\centering",
    r"\caption{Inference time (mean $\pm$ std, best optimizer).}",
    r"\label{tab:time}", r"\begin{tabular}{lcc}", r"\toprule",
    r"Method & Time (ms) & NR Iters \\", r"\midrule"]
for method in ['mlp', 'mdn', 'hybrid', 'nr_only']:
    s = df[df['method'] == method]
    if s.empty: continue
    bo = s.groupby('optimizer')['euclid_mean'].mean().idxmin()
    bs = s[s['optimizer'] == bo]
    t, ts = bs['inf_time_ms'].mean(), bs['inf_time_ms'].std()
    iters = bs['nr_iters_mean'].mean() if bs['nr_iters_mean'].notna().any() else float('nan')
    its = f"${iters:.2f}$" if not np.isnan(iters) else "—"
    tex3.append(f"{METHOD_MAP[method]} & ${t:.2f} \\pm {ts:.2f}$ & {its} \\\\")
tex3 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
with open(f'{OUT}/table3_inference_time.tex', 'w') as f: f.write('\n'.join(tex3))

# # TABLE 4 + FIG 1 + FIG 2: Load models, compute joint metrics + histograms + parity
# print("[Table 4 + Fig 1 + Fig 2] Loading models...")
# try:
#     import joblib, gc
#     from tensorflow import keras
#     import keras_mdn_layer as mdn
#     from forwardkinematics import ForwardKinematic

#     d = np.load('dataset_processed.npz')
#     X_test, y_test = d['X_test'], d['y_test']
#     scaler_X = joblib.load('scaler_X.pkl')
#     X_unscaled = scaler_X.inverse_transform(X_test)

#     def euc(a, b):
#         p = ForwardKinematic(*a); t = ForwardKinematic(*b)
#         return np.sqrt((p[0]-t[0])**2 + (p[1]-t[1])**2 + (p[2]-t[2])**2)

#     def jac(th):
#         eps = 1e-4; J = np.zeros((3, 6)); c = ForwardKinematic(*th)
#         for i in range(6):
#             t = th.copy(); t[i] += eps; J[:, i] = (ForwardKinematic(*t) - c) / eps
#         return J

#     def nr(th0, tgt, mi=3, tol=1e-3):
#         th = np.array(th0, dtype=np.float64)
#         for i in range(mi):
#             c = ForwardKinematic(*th); e = tgt - c
#             if np.linalg.norm(e) < tol: return th
#             J = jac(th); JJT = J @ J.T + 1e-3 * np.eye(3)
#             th = th + J.T @ np.linalg.inv(JJT) @ e
#         return th

#     def calc_metrics(yp, yt):
#         ytd = np.rad2deg(yt); ypd = np.rad2deg(yp)
#         maes, rmses, r2s = [], [], []
#         for j in range(6):
#             maes.append(np.mean(np.abs(ytd[:, j] - ypd[:, j])))
#             rmses.append(np.sqrt(np.mean((ytd[:, j] - ypd[:, j]) ** 2)))
#             ss_r = np.sum((ytd[:, j] - ypd[:, j]) ** 2)
#             ss_t = np.sum((ytd[:, j] - np.mean(ytd[:, j])) ** 2)
#             r2s.append(1 - ss_r / ss_t if ss_t > 0 else 0)
#         return maes, rmses, r2s

#     # Best MLP
#     mr = df[df['method'] == 'mlp']; br = mr.loc[mr['euclid_mean'].idxmin()]
#     mmodel = keras.models.load_model(br['model_path'], compile=False)
#     y_mlp = mmodel.predict(X_test, verbose=0)
#     mlp_mae, mlp_rmse, mlp_r2 = calc_metrics(y_mlp, y_test)

#     # Hybrid (MLP + NR, 10k)
#     n_h = min(10000, len(X_test))
#     y_hyb = np.array([nr(y_mlp[i], X_unscaled[i]) for i in range(n_h)])
#     hyb_mae, hyb_rmse, hyb_r2 = calc_metrics(y_hyb, y_test[:n_h])

#     # Best MDN
#     mr2 = df[df['method'] == 'mdn']; br2 = mr2.loc[mr2['euclid_mean'].idxmin()]
#     mdn_model = keras.models.load_model(br2['model_path'],
#         custom_objects={'MDN': mdn.MDN}, compile=False)
#     params = mdn_model.predict(X_test, verbose=0)
#     mus = params[:, :120].reshape(-1, 20, 6)
#     pis = params[:, -20:]; pis = np.exp(pis - pis.max(1, keepdims=True)); pis /= pis.sum(1, keepdims=True)
#     y_mdn = mus[np.arange(len(mus)), np.argmax(pis, 1), :]
#     mdn_mae, mdn_rmse, mdn_r2 = calc_metrics(y_mdn, y_test)
#     l1 = np.abs(mus - y_test[:, None, :]).sum(2)
#     ora = mus[np.arange(len(mus)), np.argmin(l1, 1), :]

#     # --- Table 4 ---
#     tex4 = [r"\begin{table}[H]", r"\centering",
#         r"\caption{Joint angle prediction metrics (best model per method).}",
#         r"\label{tab:joints}", r"\resizebox{\textwidth}{!}{%",
#         r"\begin{tabular}{lccccccccc}", r"\toprule",
#         r" & \multicolumn{3}{c}{MLP} & \multicolumn{3}{c}{MDN} & \multicolumn{3}{c}{MLP+NR} \\",
#         r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}",
#         r"Joint & MAE ($^\circ$) & RMSE ($^\circ$) & R$^2$ & MAE ($^\circ$) & RMSE ($^\circ$) & R$^2$ & MAE ($^\circ$) & RMSE ($^\circ$) & R$^2$ \\",
#         r"\midrule"]
#     for j in range(6):
#         tex4.append(f"{JOINTS[j]} & "
#             f"${mlp_mae[j]:.2f}$ & ${mlp_rmse[j]:.2f}$ & ${mlp_r2[j]:.3f}$ & "
#             f"${mdn_mae[j]:.2f}$ & ${mdn_rmse[j]:.2f}$ & ${mdn_r2[j]:.3f}$ & "
#             f"${hyb_mae[j]:.2f}$ & ${hyb_rmse[j]:.2f}$ & ${hyb_r2[j]:.3f}$ \\\\")
#     tex4.append(r"\midrule")
#     tex4.append(f"Mean & "
#         f"${np.mean(mlp_mae):.2f}$ & ${np.mean(mlp_rmse):.2f}$ & ${np.mean(mlp_r2):.3f}$ & "
#         f"${np.mean(mdn_mae):.2f}$ & ${np.mean(mdn_rmse):.2f}$ & ${np.mean(mdn_r2):.3f}$ & "
#         f"${np.mean(hyb_mae):.2f}$ & ${np.mean(hyb_rmse):.2f}$ & ${np.mean(hyb_r2):.3f}$ \\\\")
#     tex4 += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
#     with open(f'{OUT}/table4_joint_metrics.tex', 'w') as f: f.write('\n'.join(tex4))
#     print("  Table 4 generated.")

#     # --- Fig 1: Histograms with KDE (hybrid ZOOMED) ---
#     print("[Fig 1] Histograms with KDE...")
#     MAX_E = 10.0
#     em = np.array([euc(y_mlp[i], y_test[i]) for i in range(len(y_test))]); em = em[em <= MAX_E]
#     eh = np.array([euc(y_hyb[i], y_test[:n_h][i]) for i in range(n_h)]); eh = eh[eh <= MAX_E]
#     es = np.array([euc(y_mdn[i], y_test[i]) for i in range(len(y_test))]); es = es[es <= MAX_E]
#     eo = np.array([euc(ora[i], y_test[i]) for i in range(len(y_test))]); eo = eo[eo <= MAX_E]
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#     mkde_pure(axes[0][0], em, PAL['MLP'], 'MLP', xlim=(0, 10))
#     mkde_pure(axes[0][1], eh, PAL['MLP+NR'], 'MLP+NR (Hybrid)', xlim=(0, 0.05))
#     mkde_pure(axes[1][0], es, PAL['MDN'], 'MDN (Selected)', xlim=(0, 5))
#     mkde_pure(axes[1][1], eo, PAL['MDN'], 'MDN (Oracle)', xlim=(0, 5))
#     fig.suptitle('Error Distribution (KDE, filtered ≤10 cm)', fontsize=14, y=1.01)
#     plt.tight_layout(); plt.savefig(f'{OUT}/fig1_histograms_kde.png', bbox_inches='tight'); plt.close()
#     print("  Fig 1 generated.")

#     # --- Fig 2: Parity plots ---
#     print("[Fig 2] Parity plots...")
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     for ri, (yp, ml, yt) in enumerate([(y_mlp, 'MLP', y_test), (y_hyb, 'MLP+NR', y_test[:n_h])]):
#         ytd = np.rad2deg(yt); ypd = np.rad2deg(yp)
#         for j in range(6):
#             ax = axes[ri][j]
#             ax.scatter(ytd[:, j], ypd[:, j], alpha=0.12, s=2, c=PAL[ml])
#             mn = min(ytd[:, j].min(), ypd[:, j].min()); mx = max(ytd[:, j].max(), ypd[:, j].max())
#             ax.plot([mn, mx], [mn, mx], 'r--', lw=1)
#             ss_r = np.sum((ytd[:, j] - ypd[:, j])**2)
#             ss_t = np.sum((ytd[:, j] - np.mean(ytd[:, j]))**2)
#             r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
#             mae_j = np.mean(np.abs(ytd[:, j] - ypd[:, j]))
#             ax.set_title(f'{ml} — {JOINTS[j]}\nR²={r2:.3f}, MAE={mae_j:.2f}°', fontsize=9)
#             ax.set_xlabel('Actual (°)' if ri == 1 else '')
#             ax.set_ylabel('Predicted (°)' if j == 0 else '')
#             ax.set_aspect('equal'); ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
#     fig.suptitle('Parity Plots: Predicted vs Actual Joint Angles', fontsize=13, y=1.01)
#     plt.tight_layout(); plt.savefig(f'{OUT}/fig2_parity_plots.png', bbox_inches='tight'); plt.close()
#     print("  Fig 2 generated.")

#     del mmodel, mdn_model; gc.collect()
# except Exception as e:
#     print(f"  [SKIP] Model loading: {e}")



# FIG 4: Success rate bars

print("[Fig 4] Success rate bars...")
fig, ax = plt.subplots(figsize=(9, 5))
succ = []
for method in ['mlp', 'mdn', 'hybrid', 'nr_only']:
    s = df[df['method'] == method]
    if s.empty: continue
    bo = s.groupby('optimizer')['euclid_mean'].mean().idxmin()
    bs = s[s['optimizer'] == bo]
    ml = METHOD_MAP[method]
    for col, lab in [('succ_1cm', 'τ≤1cm'), ('succ_2cm', 'τ≤2cm'), ('succ_5cm', 'τ≤5cm')]:
        v = bs[col].values
        succ.append({'Method': ml, 'Thr': lab, 'Rate': np.mean(v), 'Std': np.std(v) if len(v) > 1 else 0})
sdf = pd.DataFrame(succ)
methods = sdf['Method'].unique(); n = len(methods); width = 0.2
for i, m in enumerate(methods):
    md = sdf[sdf['Method'] == m]
    xp = np.arange(len(md)) + (i - n/2 + 0.5) * width
    ax.bar(xp, md['Rate'], width, yerr=md['Std'], label=m, color=PAL.get(m, '#888'),
           edgecolor='black', alpha=0.85, capsize=3, error_kw={'lw': 1})
ax.set_xticks(np.arange(3)); ax.set_xticklabels(['τ≤1cm', 'τ≤2cm', 'τ≤5cm'])
ax.set_ylim(0, 110); ax.set_ylabel('Success Rate (%)')
ax.set_title('Task-Space Success Rate (best optimizer, mean ± std)', fontweight='bold')
ax.legend(loc='upper left', framealpha=0.9)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_success_rates.png', bbox_inches='tight'); plt.close()


# SUMMARY
print("\n" + "=" * 100)
print("SUMMARY — BEST MODEL PER METHOD")
print("=" * 100)
print(f"{'Method':<10} {'Best Opt':<10} {'Error (cm)':<20} {'1cm(%)':<10} {'5cm(%)':<10} {'Time(ms)':<12}")
print("-" * 72)
for method in ['mlp', 'mdn', 'hybrid', 'nr_only']:
    s = df[df['method'] == method]
    if s.empty: continue
    bo = s.groupby('optimizer')['euclid_mean'].mean().idxmin()
    bs = s[s['optimizer'] == bo]
    m, sd = bs['euclid_mean'].mean(), bs['euclid_mean'].std()
    s1, s5 = bs['succ_1cm'].mean(), bs['succ_5cm'].mean()
    t = bs['inf_time_ms'].mean()
    print(f"{METHOD_MAP[method]:<10} {bo.upper():<10} {m:.4f} ± {sd:.4f}   {s1:<10.1f} {s5:<10.1f} {t:<12.3f}")
print(f"\nOUTPUTS in {OUT}/: 4 tables (.tex) + 5 figures (.png)")