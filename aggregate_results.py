# aggregate_results.py
import os, json, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import wilcoxon

RUNS_DIR     = "Runs"
RESULTS_DIR  = "RESULTS"
AGG_DIR      = "Aggregated"
os.makedirs(AGG_DIR, exist_ok=True)

records = []
per_sample = {}  # key=(model,opt) -> list of np.array errors

for cfg in glob.glob(os.path.join(RUNS_DIR, "config_*.json")):
    with open(cfg) as f: c = json.load(f)
    base = os.path.basename(cfg).replace("config_","").replace(".json","")
    hist_path = os.path.join(RUNS_DIR, f"history_{base}.json")
    if not os.path.exists(hist_path): continue
    with open(hist_path) as f: h = json.load(f)

    # Find matching RESULTS subdir
    matches = glob.glob(os.path.join(RESULTS_DIR, f"*{base}*"))
    if not matches: continue
    sub = matches[0]
    sum_path = os.path.join(sub, "summary.json")
    if not os.path.exists(sum_path): continue
    with open(sum_path) as f: s = json.load(f)

    rec = {
        "model": c["model"], "optimizer": c["optimizer"],
        "seed": c["seed"], "run_id": c["run_id"],
        "epochs": h.get("stopped_epoch", len(h.get("loss",[]))),
        "train_time_sec": h.get("train_time_sec", np.nan),
        "joint_MAE_deg": s["joint_MAE_mean_deg"],
        "joint_RMSE_deg": s["joint_RMSE_mean_deg"],
        "task_mean_cm": s["task_euclid_mean_cm"],
        "task_std_cm": s["task_euclid_std_cm"],
        "task_p95_cm": s["task_euclid_p95_cm"],
        "succ_1cm": s["success_rates"]["τ≤1cm"],
        "succ_2cm": s["success_rates"]["τ≤2cm"],
        "succ_5cm": s["success_rates"]["τ≤5cm"],
    }
    if s["is_mdn"]:
        rec["oracle_mean_cm"] = s["task_euclid_oracle_mean_cm"]
        rec["oracle_p95_cm"]  = s["task_euclid_oracle_p95_cm"]
        rec["mixture_entropy"] = s["mixture_entropy_mean"]
    records.append(rec)

    key = (c["model"], c["optimizer"])
    errs = np.load(os.path.join(sub, "euclid_errors.npy"))
    per_sample.setdefault(key, []).append(errs)

df = pd.DataFrame(records)
df.to_csv(os.path.join(AGG_DIR, "all_runs.csv"), index=False)

# ----- Aggregate table: mean ± std across 10 seeds -----
agg = df.groupby(["model","optimizer"]).agg(
    joint_MAE_deg=("joint_MAE_deg","mean"),
    joint_MAE_std=("joint_MAE_deg","std"),
    task_mean_cm =("task_mean_cm","mean"),
    task_std_cm  =("task_mean_cm","std"),
    task_p95_cm  =("task_p95_cm","mean"),
    succ_1cm     =("succ_1cm","mean"),
    succ_2cm     =("succ_2cm","mean"),
    succ_5cm     =("succ_5cm","mean"),
    train_time_s =("train_time_sec","mean"),
    epochs       =("epochs","mean"),
).reset_index()

# Format for LaTeX-ready string
def fmt(mean, std): return f"{mean:.3f} ± {std:.3f}"
agg["task_euclid_str"] = agg.apply(lambda r: fmt(r["task_mean_cm"], r["task_std_cm"]), axis=1)
agg["joint_MAE_str"]   = agg.apply(lambda r: fmt(r["joint_MAE_deg"], r["joint_MAE_std"]), axis=1)
agg.to_csv(os.path.join(AGG_DIR, "aggregated_mean_std.csv"), index=False)

# ----- Statistical tests: paired Wilcoxon between optimizers per model -----
stat_rows = []
for model in df["model"].unique():
    sub_m = df[df["model"]==model]
    opts = sub_m["optimizer"].unique()
    for i,a in enumerate(opts):
        for b in opts[i+1:]:
            ra = sub_m[sub_m["optimizer"]==a].sort_values("seed")["task_mean_cm"].values
            rb = sub_m[sub_m["optimizer"]==b].sort_values("seed")["task_mean_cm"].values
            if len(ra)==len(rb) and len(ra)>=5:
                try:
                    w,p = wilcoxon(ra, rb)
                except Exception:
                    w,p = np.nan, np.nan
                stat_rows.append({"model":model,"opt_A":a,"opt_B":b,
                                  "W":w,"p_value":p,
                                  "mean_A":ra.mean(),"mean_B":rb.mean()})
pd.DataFrame(stat_rows).to_csv(
    os.path.join(AGG_DIR,"wilcoxon_pairwise.csv"), index=False)

# ----- Boxplots -----
plt.figure(figsize=(9,5))
sns.boxplot(data=df, x="optimizer", y="task_mean_cm", hue="model")
sns.stripplot(data=df, x="optimizer", y="task_mean_cm", hue="model",
              dodge=True, color="black", alpha=0.5, size=3)
plt.ylabel("End-Effector Euclidean Error (cm)")
plt.title("Optimizer × Architecture — 10 runs each")
plt.tight_layout()
plt.savefig(os.path.join(AGG_DIR,"boxplot_task_error.png"), dpi=200)
plt.close()

plt.figure(figsize=(9,5))
sns.boxplot(data=df, x="optimizer", y="joint_MAE_deg", hue="model")
plt.ylabel("Joint-space MAE (°)")
plt.title("Joint Error — 10 runs each")
plt.tight_layout()
plt.savefig(os.path.join(AGG_DIR,"boxplot_joint_mae.png"), dpi=200)
plt.close()

# ----- Success rate bar (with error bars) -----
succ_df = df.melt(id_vars=["model","optimizer"],
                  value_vars=["succ_1cm","succ_2cm","succ_5cm"],
                  var_name="threshold", value_name="rate")
plt.figure(figsize=(9,5))
sns.barplot(data=succ_df, x="threshold", y="rate", hue="model",
            errorbar="sd", palette="Set2")
plt.ylabel("Success Rate (%)"); plt.ylim(0,100)
plt.title("Task-Space Success Rate (mean ± std, n=10)")
plt.tight_layout()
plt.savefig(os.path.join(AGG_DIR,"success_rate_bars.png"), dpi=200)
plt.close()

print("Aggregation done. Outputs in:", AGG_DIR)
print(agg.to_string(index=False))