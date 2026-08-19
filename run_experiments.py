# run_experiments.py
"""
Runs the full 10×2×3 = 60-experiment grid.

"""
import os, json, glob, subprocess, sys

SEEDS = list(range(10))            # 10 independent runs
MODELS = ["mlp", "mdn"]
OPTIMIZERS = ["adamw", "muon", "shampoo"]

RUNS_DIR = "Runs"
os.makedirs(RUNS_DIR, exist_ok=True)

def already_done(model, opt, seed, run_id):
    pat = os.path.join(RUNS_DIR, f"history_*_{model}_{opt}_seed{seed}_run{run_id}.json")
    return len(glob.glob(pat)) > 0

for run_id, seed in enumerate(SEEDS):
    for model in MODELS:
        for opt in OPTIMIZERS:
            if already_done(model, opt, seed, run_id):
                print(f"[skip] {model}/{opt}/seed{seed}/run{run_id}")
                continue
            cmd = [sys.executable, "train_z_mdn.py",
                   "--model", model,
                   "--optimizer", opt,
                   "--seed", str(seed),
                   "--run_id", str(run_id)]
            print("[run]", " ".join(cmd))
            subprocess.run(cmd, check=False)

print("All experiments submitted.")