# test_z_mdn.py
"""
Unified evaluation script. Auto-detects MLP vs MDN.
Computes: joint MAE/RMSE, task-space Euclidean error,
success-rate at τ∈{1,2,5} cm, MDN oracle/best-mixture metrics,
entropy of mixture weights, and dumps per-sample errors for later stats.
"""
import os, json, glob, argparse
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
import keras_mdn_layer as mdn
from forwardkinematics import ForwardKinematic

sns.set_style("whitegrid"); plt.rcParams["figure.dpi"] = 120
plt.rcParams.update({"font.family": "serif", "font.size": 11})

MODEL_DIR, RESULTS_DIR = "Model", "RESULTS"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_DIMS = 6
N_MIXES     = 20
TAUS        = [1.0, 2.0, 5.0]   # cm thresholds for success rate
JOINT_NAMES = ["Base","Shoulder","Elbow","Wrist Pitch","Wrist Roll","Wrist Twist"]

def euc3(a,b): return np.linalg.norm(a-b)

# ---------------------------------------------------------------------------
def evaluate_one(checkpoint_path: str, out_subdir: str):
    is_mdn = "_mdn_" in os.path.basename(checkpoint_path).lower() or \
             "MDN" in os.path.basename(checkpoint_path)

    custom = {"MDN": mdn.MDN} if is_mdn else {}
    model = keras.models.load_model(checkpoint_path,
                                    custom_objects=custom, compile=False)

    data = np.load("dataset_processed.npz")
    X_test, y_test = data["X_test"], data["y_test"]

    # ---------- predictions ----------
    if is_mdn:
        params = model.predict(X_test, verbose=0)
        mu_size = OUTPUT_DIMS * N_MIXES
        mus = params[:, :mu_size].reshape(-1, N_MIXES, OUTPUT_DIMS)
        pis = params[:, -N_MIXES:]
        # softmax just in case layer didn't
        pis = np.exp(pis - pis.max(axis=1, keepdims=True))
        pis = pis / pis.sum(axis=1, keepdims=True)

        sel_idx = np.argmax(pis, axis=1)
        y_pred  = mus[np.arange(len(mus)), sel_idx, :]

        # oracle mixture (closest in joint L1 to ground truth)
        l1 = np.abs(mus - y_test[:,None,:]).sum(axis=2)   # (N,K)
        ora_idx = np.argmin(l1, axis=1)
        y_ora   = mus[np.arange(len(mus)), ora_idx, :]
        # entropy of mixture weights (diversity)
        ent = -np.sum(pis * np.log(pis + 1e-12), axis=1)
    else:
        y_pred = model.predict(X_test, verbose=0)
        y_ora  = None; ent = None

    # ---------- joint-space metrics ----------
    ytd = np.rad2deg(y_test); ypd = np.rad2deg(y_pred)
    mae  = mean_absolute_error(ytd, ypd, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(ytd, ypd, multioutput="raw_values"))

    # ---------- task-space (Euclidean) ----------
    sel_d, ora_d = [], []
    for i,(pa,ta) in enumerate(zip(y_pred, y_test)):
        sel_d.append(euc3(ForwardKinematic(*pa), ForwardKinematic(*ta)))
        if is_mdn:
            od = min(euc3(ForwardKinematic(*mus[i,k]), ForwardKinematic(*ta))
                     for k in range(N_MIXES))
            ora_d.append(od)
    sel_d = np.array(sel_d)
    ora_d = np.array(ora_d) if is_mdn else None

    # ---------- success rates ----------
    succ = {f"τ≤{τ:.0f}cm": float(np.mean(sel_d <= τ)*100) for τ in TAUS}
    succ_ora = {f"τ≤{τ:.0f}cm (oracle)": float(np.mean(ora_d <= τ)*100)
                for τ in TAUS} if is_mdn else {}

    # ---------- assemble metrics ----------
    metrics = {
        "Joint": JOINT_NAMES,
        "MAE (deg)":  mae,
        "RMSE (deg)": rmse,
    }
    df = pd.DataFrame(metrics)
    if is_mdn:
        ora_mae = mean_absolute_error(ytd, np.rad2deg(y_ora), multioutput="raw_values")
        df["MAE Oracle (deg)"] = ora_mae
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_subdir, "joint_metrics.csv"), index=False)

    summary = {
        "model_path": checkpoint_path,
        "is_mdn": is_mdn,
        "n_test": len(y_test),
        "joint_MAE_mean_deg":   float(mae.mean()),
        "joint_RMSE_mean_deg":  float(rmse.mean()),
        "task_euclid_mean_cm":  float(sel_d.mean()),
        "task_euclid_std_cm":   float(sel_d.std()),
        "task_euclid_median_cm":float(np.median(sel_d)),
        "task_euclid_p95_cm":   float(np.percentile(sel_d,95)),
        "success_rates":        succ,
    }
    if is_mdn:
        summary.update({
            "task_euclid_oracle_mean_cm": float(ora_d.mean()),
            "task_euclid_oracle_p95_cm":  float(np.percentile(ora_d,95)),
            "success_rates_oracle":       succ_ora,
            "mixture_entropy_mean":       float(ent.mean()),
            "mixture_entropy_max":        float(ent.max()),
        })
    with open(os.path.join(out_subdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # save per-sample errors for cross-run statistics
    np.save(os.path.join(out_subdir, "euclid_errors.npy"), sel_d)
    if is_mdn:
        np.save(os.path.join(out_subdir, "oracle_errors.npy"), ora_d)
        np.save(os.path.join(out_subdir, "pis.npy"), pis)

    # ---------- plots ----------
    # Parity
    fig, axs = plt.subplots(3,2, figsize=(11,13))
    fig.suptitle(f"Predicted vs Actual — {os.path.basename(checkpoint_path)}",
                 fontsize=14)
    for i,ax in enumerate(axs.flat):
        ax.scatter(ytd[:,i], ypd[:,i], s=4, alpha=0.3, label="Selected")
        if is_mdn:
            ax.scatter(ytd[:,i], np.rad2deg(y_ora)[:,i],
                       s=2, alpha=0.15, color="green", label="Oracle")
        mn = min(ytd[:,i].min(), ypd[:,i].min())
        mx = max(ytd[:,i].max(), ypd[:,i].max())
        ax.plot([mn,mx],[mn,mx],"r--",lw=1.5)
        ax.set_title(f"{JOINT_NAMES[i]} | MAE={mae[i]:.2f}°")
        ax.set_xlabel("Actual (°)"); ax.set_ylabel("Predicted (°)")
        ax.legend(fontsize=8)
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.savefig(os.path.join(out_subdir,"parity_plots.png")); plt.close()

    # Euclidean histogram
    plt.figure(figsize=(9,5))
    plt.hist(sel_d, bins=80, alpha=0.7,
             label=f"Selected (μ={sel_d.mean():.2f} cm)")
    if is_mdn:
        plt.hist(ora_d, bins=80, alpha=0.5,
                 label=f"Oracle (μ={ora_d.mean():.2f} cm)")
    plt.xlabel("Euclidean Error (cm)"); plt.ylabel("Frequency")
    plt.title("End-Effector Position Error Distribution")
    plt.axvline(sel_d.mean(), color="blue",  ls="--", lw=1)
    if is_mdn:
        plt.axvline(ora_d.mean(), color="orange", ls="--", lw=1)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_subdir,"euclid_hist.png")); plt.close()

    # Success-rate bar
    plt.figure(figsize=(7,4))
    keys = list(succ.keys())
    vals = list(succ.values())
    plt.bar(keys, vals, color="steelblue", alpha=0.8, label="Selected")
    if is_mdn:
        plt.bar(keys, list(succ_ora.values()), color="orange",
                alpha=0.5, width=0.4, label="Oracle")
    plt.ylabel("Success Rate (%)"); plt.ylim(0,100)
    plt.title("Task-Space Success Rate"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_subdir,"success_rate.png")); plt.close()

    return summary

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="Path to .keras checkpoint to evaluate")
    args = ap.parse_args()
    sub = os.path.join(RESULTS_DIR,
                       os.path.splitext(os.path.basename(args.model_path))[0])
    os.makedirs(sub, exist_ok=True)
    evaluate_one(args.model_path, sub)
    print(f"Saved to: {sub}")