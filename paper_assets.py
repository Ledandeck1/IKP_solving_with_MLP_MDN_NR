#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
paper_assets.py  --  final analyses, tables and figures for the IEEE Access
                     manuscript on neural inverse kinematics
--------------------------------------------------------------------------------
Runs after extract_paper_data.py. It adds the analyses that the earlier pipeline
did not cover and emits the complete set of LaTeX tables, so that the manuscript
can be written against a single frozen artefact directory.

    python paper_assets.py --stage all

--------------------------------------------------------------------------------
WHAT IS NEW HERE, AND WHY
--------------------------------------------------------------------------------

1. THE GRIPPER IS NOT AN IK VARIABLE.
   Joint 6 of this manipulator has a = 0 and alpha = 0, so its rotation cannot
   translate the end-effector, and physically it actuates the gripper rather
   than positioning it. It is therefore excluded from the inverse kinematics
   mapping, which is stated here as

        f^-1 : R^3  ->  R^5 ,      x_d = (x, y, z)  |->  theta_1..5

   The networks were trained with six outputs; the sixth is ignored at
   evaluation time. This needs no retraining: forward kinematics does not depend
   on it and the Newton-Raphson update never moves it, because the sixth column
   of the Jacobian is identically zero. Every joint-space metric in this file is
   computed over joints 1-5.

2. MULTIMODALITY, MEASURED RATHER THAN ASSERTED.
   Filho and Santos (IEEE Access, 2026) report a parity plot with heavy
   dispersion and read it as the network failing to learn. For a position-only
   target on a redundant arm that reading is incomplete: a point far from the
   diagonal may be a perfectly valid alternative solution that the dataset
   simply did not record. This script separates the two cases by colouring the
   parity plot with the task-space error, and quantifies the separation with

        alternative-solution rate
            = P( joint error > tau_theta  AND  task error <= tau_x )

   which is, to our knowledge, not reported anywhere in the neural IK
   literature. It is the direct measurement of the phenomenon that the prior
   work describes qualitatively and declares out of scope.

3. DISTRIBUTIONS ACROSS THE FULL 60-MODEL GRID.
   The earlier figures showed means. A boxplot over the ten seeds of every
   architecture-optimizer cell shows the spread that the mean hides, which is
   the honest way to present a factor that explains little of the variance.

4. THE GLOB BUG IS FIXED.
   `err_mlp_nr_*.npz` also matched `err_mlp_nr_iter1_*.npz`, so figure 1 was
   plotting the one-iteration variant under the full-refinement label. Method
   names are now matched with an anchored regular expression.

--------------------------------------------------------------------------------
OUTPUTS  (all under PAPER_DATA/)
--------------------------------------------------------------------------------
    multimodality.json          alternative-solution rates and joint statistics
    per_sample/parity_*.npz     true/predicted joints + task error, for replots
    tables/*.tex                every table in the manuscript
    figures/fig*.pdf|.png       the new figures
    MANUSCRIPT_NUMBERS.md       every number to be quoted, in one place

Author: Gustavo Henrique Germano Ledandeck (UFABC)
License: MIT
================================================================================
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

PATHS: Dict[str, str] = {
    "out_dir": "PAPER_DATA",
    "fig_dir": os.path.join("PAPER_DATA", "figures"),
    "tab_dir": os.path.join("PAPER_DATA", "tables"),
    "raw_dir": os.path.join("PAPER_DATA", "per_sample"),
    "model_dir": "Model",
    "data_npz": "dataset_processed.npz",
    "scaler": "scaler_X.pkl",
}

# Joint 6 actuates the gripper and is excluded from the IK mapping.
GRIPPER_JOINT = 5
IK_JOINTS = [0, 1, 2, 3, 4]
JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist pitch", "Wrist roll", "Gripper"]
IK_JOINT_LABELS = [r"$\theta_1$", r"$\theta_2$", r"$\theta_3$",
                   r"$\theta_4$", r"$\theta_5$"]

# The third training arm requested Shampoo but every one of its checkpoints
# recorded the class Adam: tensorflow_addons is archived and does not build
# against TF >= 2.14, so the fallback in train_z_mdn.py fired silently. That arm
# is therefore excluded from the analysis rather than relabelled, and the
# exclusion is stated in the manuscript. Add it back with --optimizers if the
# runs are ever repeated with a genuine implementation.
OPTIMIZERS = ["adamw", "muon"]
OPT_DISPLAY = {"adamw": "AdamW", "muon": "Muon", "shampoo": "Shampoo"}
EXCLUDED_OPTIMIZERS = ["shampoo"]

# thresholds for the alternative-solution rate
TAU_THETA_DEG = 30.0     # "a genuinely different posture"
TAU_X_MM = 1.0           # "the tool is where it was asked to be"
# The alternative-solution rate is conditional on solving the task, so a method
# that rarely solves it at 1 mm yields a vacuous statistic. Reporting the sweep
# lets each method be read at a threshold it actually reaches.
TAU_X_SWEEP_MM = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

# The multiple-comparison family is the set of comparisons the manuscript
# reports, declared here rather than inferred from whatever happens to be in the
# results file. Correcting over every derived variant as well (mlp_nr_iter1,
# mlp_clamped, the oracle rows) would correct over quantities that are neither
# independent nor reported, and would inflate the p-values without protecting
# against anything.
PRIMARY_METHODS = ["mlp", "mdn_argmax_pi", "mdn_fk_best", "mlp_nr", "mdn_nr"]

COL1, COL2 = 3.5, 7.16
RNG_SEED = 20260907

OKABE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermil": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000", "grey": "#999999",
}

DH_TABLE = np.array([
    [6.5, 0.0, np.pi / 2],
    [0.0, 11.0, 0.0],
    [0.0, 15.0, 0.0],
    [0.0, 0.0, np.pi / 2],
    [0.0, 0.0, -np.pi / 2],
    [18.0, 0.0, 0.0],
], dtype=np.float64)

JOINT_LIMITS_DEG = np.array([[15., 165.], [15., 115.], [-90., 0.],
                             [-90., 0.], [-90., 90.], [0., 90.]])
JOINT_LIMITS = np.deg2rad(JOINT_LIMITS_DEG)


def log(msg: str = "", level: int = 0) -> None:
    print("  " * level + msg, flush=True)


def rule(title: str = "") -> None:
    print("\n" + "=" * 76, flush=True)
    if title:
        print(title, flush=True)
        print("=" * 76, flush=True)


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            v = float(o)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=NpEncoder, ensure_ascii=False)
    log(f"wrote {path}", 1)


def load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        if default is not None:
            return default
        raise


def fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "--"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if np.isnan(v) else f"{v:.{nd}f}"


# ============================================================================
# KINEMATICS  (self-contained; validated against forwardkinematics.py)
# ============================================================================

def fk_frames(theta: np.ndarray) -> np.ndarray:
    theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
    n = theta.shape[0]
    d, a, alpha = DH_TABLE[:, 0], DH_TABLE[:, 1], DH_TABLE[:, 2]
    ct, st = np.cos(theta), np.sin(theta)
    ca = np.broadcast_to(np.cos(alpha), theta.shape)
    sa = np.broadcast_to(np.sin(alpha), theta.shape)
    aa = np.broadcast_to(a, theta.shape)
    dd = np.broadcast_to(d, theta.shape)
    A = np.zeros((n, 6, 4, 4))
    A[:, :, 0, 0] = ct
    A[:, :, 0, 1] = -st * ca
    A[:, :, 0, 2] = st * sa
    A[:, :, 0, 3] = aa * ct
    A[:, :, 1, 0] = st
    A[:, :, 1, 1] = ct * ca
    A[:, :, 1, 2] = -ct * sa
    A[:, :, 1, 3] = aa * st
    A[:, :, 2, 1] = sa
    A[:, :, 2, 2] = ca
    A[:, :, 2, 3] = dd
    A[:, :, 3, 3] = 1.0
    T = np.zeros((n, 7, 4, 4))
    T[:, 0] = np.eye(4)
    cur = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for i in range(6):
        cur = cur @ A[:, i]
        T[:, i + 1] = cur
    return T


def fk_batch(theta: np.ndarray) -> np.ndarray:
    return fk_frames(theta)[:, 6, :3, 3]


def jacobian_batch(theta: np.ndarray) -> np.ndarray:
    T = fk_frames(theta)
    p_e = T[:, 6, :3, 3]
    z = T[:, :6, :3, 2]
    p = T[:, :6, :3, 3]
    return np.transpose(np.cross(z, p_e[:, None, :] - p), (0, 2, 1))


def newton_raphson_batch(theta0, targets, max_iter=10, lam=1e-3, tol_cm=1e-3,
                         step_clip=0.5):
    theta = np.array(theta0, dtype=np.float64, copy=True)
    targets = np.asarray(targets, dtype=np.float64)
    n = theta.shape[0]
    iters = np.zeros(n, dtype=np.int32)
    conv = np.zeros(n, dtype=bool)
    pos = fk_batch(theta)
    err = targets - pos
    dist = np.linalg.norm(err, axis=1)
    conv |= dist < tol_cm
    eye3 = np.eye(3)
    for _ in range(max_iter):
        idx = np.where(~conv)[0]
        if idx.size == 0:
            break
        J = jacobian_batch(theta[idx])
        JT = np.transpose(J, (0, 2, 1))
        y = np.linalg.solve(J @ JT + lam * eye3, err[idx][:, :, None])
        dth = (JT @ y)[:, :, 0]
        nrm = np.linalg.norm(dth, axis=1, keepdims=True)
        dth *= np.minimum(1.0, step_clip / np.maximum(nrm, 1e-12))
        theta[idx] += dth
        iters[idx] += 1
        pos = fk_batch(theta)
        err = targets - pos
        dist = np.linalg.norm(err, axis=1)
        conv |= dist < tol_cm
    return {"theta": theta, "iters": iters, "converged": conv,
            "final_dist_cm": dist}


# ============================================================================
# ANGLE HANDLING
# ============================================================================

def wrap_deg(d: np.ndarray) -> np.ndarray:
    """Signed angular difference in (-180, 180].

    Joint errors must be compared on the circle. Reporting a raw subtraction
    turns a 1-degree disagreement across the +-180 boundary into 359 degrees and
    inflates every joint-space statistic.
    """
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def joint_error_deg(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample, per-joint absolute angular error over the IK joints only."""
    return np.abs(wrap_deg(np.rad2deg(pred[:, IK_JOINTS] - true[:, IK_JOINTS])))


# ============================================================================
# ARTEFACT ACCESS  (with the glob bug fixed)
# ============================================================================

_TAG = r"(?:mlp|mdn|baseline)_(?:adamw|muon|shampoo|none)_seed\d+_run\d+"


class Artefacts:
    """Reader for the directory produced by extract_paper_data.py."""

    def __init__(self, out_dir: Optional[str] = None) -> None:
        d = out_dir or PATHS["out_dir"]
        self.dir = d
        self.models: List[Dict] = load_json(os.path.join(d, "per_model_full.json"), [])
        self.baselines: List[Dict] = load_json(os.path.join(d, "baselines.json"), [])
        self.stats: Dict = load_json(os.path.join(d, "statistics.json"), {})
        self.audit: Dict = load_json(os.path.join(d, "audit.json"), {})
        self.trajectory: Dict = load_json(os.path.join(d, "trajectory.json"), {})
        self.history: Dict = load_json(os.path.join(d, "training_history.json"), {})
        self.master: Dict = load_json(os.path.join(d, "results_master.json"), {})
        self.all = self.models + self.baselines

    def rows(self, method: str) -> List[Dict]:
        return [r for r in self.all
                if r.get("method") == method
                and r.get("optimizer", "none") in OPTIMIZERS + ["none"]]

    def best(self, method: str) -> Optional[Dict]:
        rs = self.rows(method)
        return min(rs, key=lambda r: r["task_space"]["position_error_mm"]["mean"]) if rs else None

    def seed_means(self, method: str, opt: Optional[str] = None) -> np.ndarray:
        rs = [r for r in self.rows(method)
              if opt is None or r.get("optimizer") == opt]
        return np.array([r["task_space"]["position_error_mm"]["mean"] for r in rs])

    def errors_mm(self, method: str) -> Optional[np.ndarray]:
        """Per-sample error vector, matched with an anchored pattern.

        A plain glob on `err_{method}_*` also matches longer method names that
        share the prefix, which silently swapped series in the earlier figures.
        """
        pat = re.compile(rf"^err_{re.escape(method)}_(?:{_TAG})\.npz$")
        files = [p for p in sorted(glob.glob(os.path.join(PATHS["raw_dir"], "*.npz")))
                 if pat.match(os.path.basename(p))]
        if not files:
            solo = os.path.join(PATHS["raw_dir"], f"err_{method}.npz")
            files = [solo] if os.path.exists(solo) else []
        return np.load(files[0])["error_mm"] if files else None


MODEL_RE = re.compile(
    r"best_model_(?P<arch>mlp|mdn)_(?P<opt>adamw|muon|shampoo)"
    r"_seed(?P<seed>\d+)_run(?P<run>\d+)", re.IGNORECASE)


def discover_models() -> List[Dict[str, Any]]:
    out = []
    for p in sorted(glob.glob(os.path.join(PATHS["model_dir"], "*.keras"))):
        m = MODEL_RE.search(os.path.basename(p))
        if m and m.group("opt").lower() in OPTIMIZERS:
            out.append({"path": p, "arch": m.group("arch").lower(),
                        "optimizer": m.group("opt").lower(),
                        "seed": int(m.group("seed")), "run": int(m.group("run"))})
    return out


def load_dataset():
    import joblib

    d = np.load(PATHS["data_npz"])
    scaler = joblib.load(PATHS["scaler"])
    return {
        "X_test_s": d["X_test"], "y_test": d["y_test"],
        "X_test": scaler.inverse_transform(d["X_test"]),
        "y_train": d["y_train"], "scaler": scaler,
    }


def load_keras(path: str, arch: str):
    from tensorflow import keras

    custom = {}
    if arch == "mdn":
        import keras_mdn_layer as mdn

        custom = {"MDN": mdn.MDN}
    return keras.models.load_model(path, custom_objects=custom, compile=False)


def decode_mdn(params: np.ndarray, n_out: int, n_mix: int) -> Dict[str, np.ndarray]:
    k = n_out * n_mix
    mu = params[:, :k].reshape(-1, n_mix, n_out)
    sigma = params[:, k:2 * k].reshape(-1, n_mix, n_out)
    logits = params[:, -n_mix:]
    if np.any(sigma <= 0):
        sigma = np.where(sigma > 0, sigma, np.expm1(np.minimum(sigma, 0.))) + 1. + 1e-8
    z = logits - logits.max(axis=1, keepdims=True)
    pi = np.exp(z)
    pi /= pi.sum(axis=1, keepdims=True)
    return {"mu": mu, "sigma": sigma, "pi": pi}


# ============================================================================
# STAGE 1 -- MULTIMODALITY
# ============================================================================

def stage_multimodality(args) -> Dict[str, Any]:
    """Separate "the network failed" from "the network found another solution".

    For every method we record, per test sample, the predicted joint vector, the
    ground-truth joint vector and the resulting task-space error. Three
    quantities follow, and together they answer the interpretation that prior
    work places on a dispersed parity plot:

      alternative_solution_rate
          share of samples whose posture differs from the recorded one by more
          than tau_theta while placing the tool within tau_x of the target.
          These are correct answers that a joint-space metric scores as failures.

      failure_rate
          large joint error AND large task error. These are genuine failures.

      joint_error_given_success
          the distribution of joint error restricted to samples that solved the
          task. If the mapping were single-valued this would concentrate near
          zero; the width of this distribution is a direct measure of how
          multi-valued the problem is at the accuracy the solver achieves.
    """
    rule("STAGE 1/4  MULTIMODALITY AND PARITY ANALYSIS")
    ds = load_dataset()
    y_true = ds["y_test"]
    x_target = ds["X_test"]
    n = len(y_true)
    log(f"test samples: {n}", 1)

    art = Artefacts()
    refs = discover_models()
    if not refs:
        raise FileNotFoundError(f"no models under {PATHS['model_dir']}/")

    def pick(arch: str, method: str) -> Optional[Dict]:
        b = art.best(method)
        if b is None:
            cand = [r for r in refs if r["arch"] == arch]
            return cand[0] if cand else None
        for r in refs:
            if (r["arch"] == arch and r["optimizer"] == b["optimizer"]
                    and r["seed"] == b["seed"]):
                return r
        cand = [r for r in refs if r["arch"] == arch]
        return cand[0] if cand else None

    ref_mlp = pick("mlp", "mlp")
    ref_mdn = pick("mdn", "mdn_fk_best")

    predictions: Dict[str, np.ndarray] = {}

    if ref_mlp is not None:
        log(f"MLP  : {os.path.basename(ref_mlp['path'])}", 1)
        m = load_keras(ref_mlp["path"], "mlp")
        th_mlp = m.predict(ds["X_test_s"], verbose=0, batch_size=4096)
        predictions["mlp"] = th_mlp
        nr = newton_raphson_batch(th_mlp, x_target, max_iter=args.nr_iters)
        predictions["mlp_nr"] = nr["theta"]
        del m

    if ref_mdn is not None:
        log(f"MDN  : {os.path.basename(ref_mdn['path'])}", 1)
        m = load_keras(ref_mdn["path"], "mdn")
        raw = m.predict(ds["X_test_s"], verbose=0, batch_size=4096)
        dec = decode_mdn(raw, y_true.shape[1], args.n_mixes)
        mu, pi = dec["mu"], dec["pi"]
        K = mu.shape[1]
        pos_all = fk_batch(mu.reshape(-1, mu.shape[2])).reshape(n, K, 3)
        d_all = np.linalg.norm(pos_all - x_target[:, None, :], axis=2)
        rows = np.arange(n)
        predictions["mdn_argmax_pi"] = mu[rows, np.argmax(pi, axis=1), :]
        predictions["mdn_fk_best"] = mu[rows, np.argmin(d_all, axis=1), :]
        nr = newton_raphson_batch(predictions["mdn_fk_best"], x_target,
                                  max_iter=args.nr_iters)
        predictions["mdn_nr"] = nr["theta"]

        # Postures kept for the 3D figure: for a handful of targets, every
        # mixture component together with the position it reaches. This is the
        # raw material for showing multimodality as arms in space rather than as
        # a statistic.
        rng_ex = np.random.default_rng(RNG_SEED)
        d_mm_all = d_all * 10.0
        good = np.where((d_mm_all <= 5.0).sum(axis=1) >= 2)[0]
        pick = (rng_ex.choice(good, min(args.posture_examples, good.size),
                              replace=False) if good.size else np.arange(0))
        np.savez_compressed(
            os.path.join(PATHS["raw_dir"], "postures_mdn.npz"),
            target_index=pick.astype(np.int64),
            targets_cm=x_target[pick].astype(np.float32),
            mu_rad=mu[pick].astype(np.float32),
            pi=pi[pick].astype(np.float32),
            comp_err_mm=d_mm_all[pick].astype(np.float32),
            y_true_rad=y_true[pick].astype(np.float32))
        log(f"kept {pick.size} targets with >=2 components within 5 mm "
            f"for the posture figure", 1)

        # How many DISTINCT postures does the mixture offer for one target?
        # Two components count as distinct if their IK joints differ by more
        # than tau_theta; only components that actually solve the task count.
        # A mode count at a single tolerance says as much about whether the
        # mixture reaches that tolerance as about how many postures it holds, so
        # the count is reported across the whole sweep.
        sub = slice(0, min(4000, n))
        mu_s = mu[sub][:, :, IK_JOINTS]
        pair = np.abs(wrap_deg(np.rad2deg(
            mu_s[:, :, None, :] - mu_s[:, None, :, :]))).max(axis=-1)
        d_sub_mm = d_all[sub] * 10.0

        def count_modes(tau_mm: float) -> np.ndarray:
            ok = d_sub_mm <= tau_mm
            res = np.zeros(mu_s.shape[0], dtype=int)
            for i in range(mu_s.shape[0]):
                valid = np.where(ok[i])[0]
                if valid.size == 0:
                    continue
                keep = [valid[0]]
                for c in valid[1:]:
                    if all(pair[i, c, k] > TAU_THETA_DEG for k in keep):
                        keep.append(c)
                res[i] = len(keep)
            return res

        mdn_modes = {
            "tau_theta_deg": TAU_THETA_DEG,
            "n_evaluated": int(mu_s.shape[0]),
            "by_tolerance": {},
            "note": "components count as distinct postures only when they both "
                    "solve the task and differ by more than tau_theta on at "
                    "least one IK joint; the count at a tight tolerance is "
                    "limited by whether the mixture reaches it at all",
        }
        for t in TAU_X_SWEEP_MM:
            dc = count_modes(t)
            mdn_modes["by_tolerance"][f"tau_x_{t}mm"] = {
                "mean_distinct_valid_postures": float(dc.mean()),
                "median_distinct_valid_postures": float(np.median(dc)),
                "share_with_2_or_more_pct": float(np.mean(dc >= 2) * 100),
                "share_with_none_pct": float(np.mean(dc == 0) * 100),
                "max_observed": int(dc.max()),
            }
            log(f"tau_x={t:>5g} mm: {dc.mean():5.2f} distinct postures on "
                f"average, {np.mean(dc >= 2) * 100:5.1f}% of targets have >=2", 2)
        del m
    else:
        mdn_modes = {}

    try:
        from tensorflow import keras

        keras.backend.clear_session()
    except Exception:
        pass

    # ---- per-method statistics ---------------------------------------------
    results: Dict[str, Any] = {
        "definitions": {
            "tau_theta_deg": TAU_THETA_DEG,
            "tau_x_mm": TAU_X_MM,
            "ik_joints": [JOINT_NAMES[i] for i in IK_JOINTS],
            "excluded_joint": JOINT_NAMES[GRIPPER_JOINT],
            "exclusion_reason": "actuates the gripper; a = 0 and alpha = 0, so it "
                                "cannot translate the end-effector and does not "
                                "belong to the inverse kinematics mapping",
        },
        "reference_models": {
            "mlp": os.path.basename(ref_mlp["path"]) if ref_mlp else None,
            "mdn": os.path.basename(ref_mdn["path"]) if ref_mdn else None,
        },
        "mdn_mode_count": mdn_modes,
        "methods": {},
    }

    for name, th in predictions.items():
        e_task_mm = np.linalg.norm(fk_batch(th) - x_target, axis=1) * 10.0
        e_joint = joint_error_deg(th, y_true)          # (n, 5)
        e_joint_max = e_joint.max(axis=1)
        e_joint_mean = e_joint.mean(axis=1)

        different = e_joint_max > TAU_THETA_DEG
        sweep = {}
        for t in TAU_X_SWEEP_MM:
            sv = e_task_mm <= t
            sweep[f"tau_x_{t}mm"] = {
                "task_success_pct": float(sv.mean() * 100),
                "alternative_solution_rate_pct": float(np.mean(sv & different) * 100),
                "matched_ground_truth_pct": float(np.mean(sv & ~different) * 100),
                "failure_rate_pct": float(np.mean(~sv & different) * 100),
                "share_of_successes_that_are_alternative_pct": (
                    float(np.mean(different[sv]) * 100) if sv.any() else float("nan")),
                "median_joint_error_given_success_deg": (
                    float(np.median(e_joint_mean[sv])) if sv.any() else float("nan")),
                "p95_joint_error_given_success_deg": (
                    float(np.percentile(e_joint_mean[sv], 95)) if sv.any() else float("nan")),
            }
        solved = e_task_mm <= TAU_X_MM

        entry = {
            "n": int(n),
            "task_error_mm": {
                "mean": float(e_task_mm.mean()),
                "median": float(np.median(e_task_mm)),
                "p95": float(np.percentile(e_task_mm, 95)),
            },
            "joint_error_deg_ik_only": {
                "mae_per_joint": e_joint.mean(axis=0).tolist(),
                "mae_mean": float(e_joint.mean()),
                "rmse_mean": float(np.sqrt((e_joint ** 2).mean())),
                "median_max_over_joints": float(np.median(e_joint_max)),
            },
            "task_success_pct": float(solved.mean() * 100),
            "alternative_solution_rate_pct": float(np.mean(solved & different) * 100),
            "failure_rate_pct": float(np.mean(~solved & different) * 100),
            "matched_ground_truth_pct": float(np.mean(solved & ~different) * 100),
            "share_of_successes_that_are_alternative_pct": (
                float(np.mean(different[solved]) * 100) if solved.any() else float("nan")),
            "threshold_sweep": sweep,
            "joint_error_given_success_deg": {
                "mean": float(e_joint_mean[solved].mean()) if solved.any() else float("nan"),
                "median": float(np.median(e_joint_mean[solved])) if solved.any() else float("nan"),
                "p95": float(np.percentile(e_joint_mean[solved], 95)) if solved.any() else float("nan"),
                "max": float(e_joint_mean[solved].max()) if solved.any() else float("nan"),
            },
        }
        results["methods"][name] = entry

        # R^2 per IK joint, on the circle
        yt = np.rad2deg(y_true[:, IK_JOINTS])
        resid = wrap_deg(np.rad2deg(th[:, IK_JOINTS]) - yt)
        ss_res = (resid ** 2).sum(axis=0)
        ss_tot = ((yt - yt.mean(axis=0)) ** 2).sum(axis=0)
        entry["r2_per_ik_joint"] = (1.0 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)).tolist()

        keep = min(n, args.parity_samples)
        sel = np.random.default_rng(RNG_SEED).choice(n, keep, replace=False)
        np.savez_compressed(
            os.path.join(PATHS["raw_dir"], f"parity_{name}.npz"),
            y_true_deg=np.rad2deg(y_true[sel][:, IK_JOINTS]).astype(np.float32),
            y_pred_deg=np.rad2deg(th[sel][:, IK_JOINTS]).astype(np.float32),
            task_error_mm=e_task_mm[sel].astype(np.float32),
            joint_error_deg=e_joint[sel].astype(np.float32))

        log(f"{name:<16} task {e_task_mm.mean():8.4f} mm | "
            f"solved {entry['task_success_pct']:6.2f}% | "
            f"alternative {entry['alternative_solution_rate_pct']:6.2f}% | "
            f"failure {entry['failure_rate_pct']:6.2f}%", 1)

    save_json(results, os.path.join(PATHS["out_dir"], "multimodality.json"))
    return results


# ============================================================================
# STAGE 1b -- OPTIMIZER PROVENANCE
# ============================================================================

def stage_provenance(args) -> Dict[str, Any]:
    """Read the optimizer that each checkpoint actually recorded.

    The environment audit in extract_paper_data.py resolves the optimizers on
    the machine where the audit runs, which is evidence about that machine and
    not about the machine that produced the weights. The `.keras` container is a
    zip archive whose `config.json` carries the `compile_config` written at save
    time, so the class name used for a given run is recorded in the run's own
    artefact. That is the only source that settles the question, and it is read
    here per model rather than inferred.

    Three outcomes are possible for a checkpoint:
      recorded    the class name is present and is reported verbatim;
      absent      the file was saved without a compile configuration, which
                  happens when only weights were checkpointed; nothing can be
                  concluded from that file either way;
      unreadable  the archive could not be opened.
    """
    rule("STAGE 1b  OPTIMIZER PROVENANCE")
    import zipfile

    # provenance audits every checkpoint on disk, including arms excluded from
    # the analysis: the reason for excluding one is exactly what it reports
    refs = []
    for p in sorted(glob.glob(os.path.join(PATHS["model_dir"], "*.keras"))):
        m = MODEL_RE.search(os.path.basename(p))
        if m:
            refs.append({"path": p, "arch": m.group("arch").lower(),
                         "optimizer": m.group("opt").lower(),
                         "seed": int(m.group("seed")), "run": int(m.group("run"))})
    if not refs:
        log("no checkpoints found", 1)
        return {}

    out: Dict[str, Any] = {
        "method": "class_name read from compile_config inside each .keras archive",
        "per_model": [], "summary": {}, "conflicts": [],
    }

    for r in refs:
        rec = {"file": os.path.basename(r["path"]), "arch": r["arch"],
               "requested_optimizer": r["optimizer"], "seed": r["seed"],
               "recorded_class": None, "recorded_config": None,
               "status": "absent"}
        try:
            with zipfile.ZipFile(r["path"]) as z:
                names = [n for n in z.namelist() if n.endswith("config.json")]
                blob = None
                for n in names:
                    cfg = json.loads(z.read(n).decode("utf-8"))
                    cc = cfg.get("compile_config") or {}
                    if cc.get("optimizer"):
                        blob = cc["optimizer"]
                        break
                if blob is not None:
                    if isinstance(blob, dict):
                        rec["recorded_class"] = (blob.get("class_name")
                                                 or blob.get("module", "") )
                        inner = blob.get("config", {})
                        rec["recorded_config"] = {
                            k: inner.get(k) for k in
                            ("learning_rate", "weight_decay", "momentum",
                             "beta_1", "beta_2") if k in inner}
                    else:
                        rec["recorded_class"] = str(blob)
                    rec["status"] = "recorded"
        except Exception as exc:
            rec["status"] = "unreadable"
            rec["error"] = str(exc)
        out["per_model"].append(rec)

    # aggregate per requested optimizer
    for opt in ("adamw", "muon", "shampoo"):
        sub = [r for r in out["per_model"] if r["requested_optimizer"] == opt]
        if not sub:
            continue
        classes = {}
        for r in sub:
            if r["status"] == "recorded":
                classes[r["recorded_class"]] = classes.get(r["recorded_class"], 0) + 1
        out["summary"][opt] = {
            "n_checkpoints": len(sub),
            "n_recorded": sum(1 for r in sub if r["status"] == "recorded"),
            "classes": classes,
        }
        low = {(k or "").lower() for k in classes}
        if low and not any(opt.replace("adamw", "adamw") in c for c in low):
            expect = {"adamw": "adamw", "muon": "muon", "shampoo": "shampoo"}[opt]
            if not any(expect in c for c in low):
                out["conflicts"].append(
                    f"runs requested '{opt}' recorded {sorted(classes)}")
        log(f"{opt:<9} recorded classes: {classes or 'none stored'}", 1)

    if out["conflicts"]:
        log("", 1)
        for c in out["conflicts"]:
            log(f"CONFLICT: {c}", 1)
        out["verdict"] = ("At least one arm recorded a class other than the one "
                          "requested. Report the recorded class.")
    elif all(v["n_recorded"] == 0 for v in out["summary"].values()):
        out["verdict"] = ("No checkpoint stored a compile configuration, so the "
                          "provenance cannot be settled from the weights. Report "
                          "the requested names and state in the manuscript which "
                          "library version supplied each optimizer, verified by "
                          "re-running one configuration in the training "
                          "environment.")
        log(out["verdict"], 1)
    else:
        out["verdict"] = "Every checkpoint recorded the optimizer that was requested."
        log(out["verdict"], 1)

    save_json(out, os.path.join(PATHS["out_dir"], "optimizer_provenance.json"))
    return out


# ============================================================================
# STAGE 1c -- OPTIMIZER STATISTICS, RECOMPUTED ON THE KEPT ARMS
# ============================================================================

def stage_optstats(args) -> Dict[str, Any]:
    """Recompute every optimizer statistic on the arms that survived the audit.

    The values in statistics.json were computed over three arms. Dropping one
    invalidates three of them and they cannot simply be filtered:

      Friedman         an omnibus test over three related samples is undefined
                       for two, and is unnecessary: with a single pair the
                       Wilcoxon test is the whole analysis.
      Holm correction  the adjustment depends on the size of the comparison
                       family. Three comparisons multiplied the smallest raw
                       p-value by three; one comparison multiplies it by one, so
                       the corrected values must be recomputed, not filtered.
      variance share   the share of variance attributed to the optimizer was a
                       ratio of between-group to total sum of squares over three
                       groups. Removing a group changes both terms.

    Everything below is therefore derived from per_model.csv directly.
    """
    rule("STAGE 1c  OPTIMIZER STATISTICS ON THE KEPT ARMS")
    import pandas as pd
    from scipy import stats as sstats

    df = pd.read_csv(os.path.join(PATHS["out_dir"], "per_model.csv"))
    df = df[df["optimizer"].isin(OPTIMIZERS)]
    rng = np.random.default_rng(RNG_SEED)

    out: Dict[str, Any] = {
        "kept_optimizers": list(OPTIMIZERS),
        "excluded_optimizers": list(EXCLUDED_OPTIMIZERS),
        "exclusion_reason": (
            "every checkpoint of the arm requesting Shampoo recorded the class "
            "Adam; see optimizer_provenance.json"),
        "n_models_analysed": int(len(df["method"].unique()) and
                                 len(df[df["method"] == df["method"].iloc[0]])),
        "per_method": {}, "pairwise": [], "variance_share": {},
    }

    tost_margin_mm = 0.5
    raw_p, buf = [], []

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method]
        cell = {}
        for opt in OPTIMIZERS:
            v = sub[sub["optimizer"] == opt]["err_mean_mm"].to_numpy()
            if v.size == 0:
                continue
            idx = rng.integers(0, v.size, size=(10000, v.size))
            boot = v[idx].mean(axis=1)
            cell[opt] = {
                "n_seeds": int(v.size),
                "mean_mm": float(v.mean()),
                "std_mm": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                "median_mm": float(np.median(v)),
                "ci95_mm": [float(np.percentile(boot, 2.5)),
                            float(np.percentile(boot, 97.5))],
            }
        out["per_method"][method] = cell

        piv = sub.pivot_table(index="seed", columns="optimizer",
                              values="err_mean_mm").dropna()
        if piv.shape[0] < 3 or piv.shape[1] < 2:
            continue
        for i, a in enumerate(OPTIMIZERS):
            for b in OPTIMIZERS[i + 1:]:
                if a not in piv.columns or b not in piv.columns:
                    continue
                va, vb = piv[a].to_numpy(), piv[b].to_numpy()
                d = va - vb
                try:
                    w_stat, w_p = sstats.wilcoxon(va, vb)
                except ValueError:
                    w_stat, w_p = float("nan"), 1.0
                idx = rng.integers(0, d.size, size=(10000, d.size))
                boot = d[idx].mean(axis=1)
                nz = d[d != 0]
                if nz.size:
                    r = sstats.rankdata(np.abs(nz))
                    rb = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())
                else:
                    rb = 0.0
                se = d.std(ddof=1) / np.sqrt(d.size)
                if se > 0:
                    p_lo = 1 - sstats.t.cdf((d.mean() + tost_margin_mm) / se, d.size - 1)
                    p_hi = sstats.t.cdf((d.mean() - tost_margin_mm) / se, d.size - 1)
                    p_tost = float(max(p_lo, p_hi))
                else:
                    p_tost = 0.0
                rec = {
                    "method": method, "comparison": f"{a} vs {b}",
                    "n_seeds": int(d.size),
                    "mean_a_mm": float(va.mean()), "mean_b_mm": float(vb.mean()),
                    "delta_mm": float(d.mean()),
                    "delta_ci95_mm": [float(np.percentile(boot, 2.5)),
                                      float(np.percentile(boot, 97.5))],
                    "delta_pct_of_a": float(100 * d.mean() / va.mean()) if va.mean() else np.nan,
                    "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else None,
                    "p_raw": float(w_p),
                    "rank_biserial": rb,
                    "tost": {"margin_mm": tost_margin_mm, "p_tost": p_tost,
                             "equivalent": bool(p_tost < 0.05)},
                }
                rec["in_holm_family"] = method in PRIMARY_METHODS
                if rec["in_holm_family"]:
                    raw_p.append(w_p)
                buf.append(rec)

        # variance attributable to the optimizer, on the kept arms only
        grand = sub["err_mean_mm"].mean()
        ss_tot = float(((sub["err_mean_mm"] - grand) ** 2).sum())
        ss_opt = float(sum(len(g) * (g["err_mean_mm"].mean() - grand) ** 2
                           for _, g in sub.groupby("optimizer")))
        out["variance_share"][method] = {
            "optimizer_pct": float(100 * ss_opt / ss_tot) if ss_tot > 0 else 0.0,
            "seed_and_noise_pct": float(100 * (1 - ss_opt / ss_tot)) if ss_tot > 0 else 100.0,
        }

    # Holm-Bonferroni over the declared family only
    if raw_p:
        order = np.argsort(raw_p)
        m = len(raw_p)
        running = 0.0
        adj = [0.0] * m
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * raw_p[i])
            adj[i] = min(1.0, running)
        fam = [r for r in buf if r["in_holm_family"]]
        for rec, pa in zip(fam, adj):
            rec["p_holm"] = float(pa)
            rec["significant_holm"] = bool(pa < 0.05)
    for rec in buf:
        if not rec.get("in_holm_family"):
            rec["p_holm"] = None
            rec["significant_holm"] = None
            rec["note"] = ("derived variant, reported without correction because "
                           "it is outside the declared comparison family")
    out["pairwise"] = buf
    out["holm_family"] = {
        "methods": [m for m in PRIMARY_METHODS if m in out["per_method"]],
        "size": len(raw_p),
        "definition": "one comparison per primary method; the manuscript must "
                      "state this family when reporting corrected p-values",
    }

    log(f"Holm family: {len(raw_p)} comparison(s) over "
        f"{[m for m in PRIMARY_METHODS if m in out['per_method']]}", 1)
    for rec in buf:
        if rec["method"] in PRIMARY_METHODS:
            log(f"{rec['method']:<14} {rec['comparison']:<16} "
                f"d={rec['delta_mm']:+8.4f} mm  p_raw={rec['p_raw']:.4f}  "
                f"p_holm={rec['p_holm']:.4f}  "
                f"equiv={rec['tost']['equivalent']}", 1)
    for k in ("mlp", "mdn_fk_best", "mlp_nr"):
        if k in out["variance_share"]:
            log(f"{k:<14} optimizer explains "
                f"{out['variance_share'][k]['optimizer_pct']:5.1f}% of the variance", 1)

    save_json(out, os.path.join(PATHS["out_dir"], "optimizer_stats.json"))
    return out


# ============================================================================
# STAGE 2 -- FIGURES
# ============================================================================

def setup_mpl(fontsize: float = 8.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": fontsize, "axes.titlesize": fontsize + 0.5,
        "axes.labelsize": fontsize, "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1, "legend.fontsize": fontsize - 1,
        "legend.frameon": False, "axes.grid": True,
        "grid.color": "#BBBBBB", "grid.alpha": 0.45, "grid.linewidth": 0.35,
        "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6,
        "lines.linewidth": 1.3, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return plt


def _save(fig, name: str) -> None:
    os.makedirs(PATHS["fig_dir"], exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PATHS["fig_dir"], f"{name}.{ext}"))
    import matplotlib.pyplot as plt

    plt.close(fig)
    log(f"wrote {os.path.join(PATHS['fig_dir'], name)}.pdf|.png", 1)


def _panel(ax, text, dx=-0.18, dy=1.10):
    ax.text(dx, dy, text, transform=ax.transAxes, fontweight="bold",
            va="top", ha="left", fontsize=9)


def fig_parity(art: Artefacts, plt) -> None:
    """Parity plot coloured by task-space error.

    This is the figure that reinterprets the prior literature. A point far from
    the diagonal is not evidence of failure by itself; if it is dark, the tool
    reached the requested position through a different posture, which is a valid
    solution the recorded label happened not to contain. Only the light points
    off the diagonal are failures.
    """
    from matplotlib.colors import LogNorm

    method = getattr(fig_parity, "method", "mlp_nr")
    path = os.path.join(PATHS["raw_dir"], f"parity_{method}.npz")
    if not os.path.exists(path):
        alt = sorted(glob.glob(os.path.join(PATHS["raw_dir"], "parity_*.npz")))
        if not alt:
            log("[skip] parity: run --stage multimodality first", 1)
            return
        path = alt[0]
        method = os.path.basename(path)[7:-4]
        log(f"parity: falling back to {method}", 1)
    z = np.load(path)
    yt, yp, et = z["y_true_deg"], z["y_pred_deg"], z["task_error_mm"]
    mm = load_json(os.path.join(PATHS["out_dir"], "multimodality.json"), {})
    stats = mm.get("methods", {}).get(method, {})

    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.1),
                             gridspec_kw=dict(wspace=0.34, hspace=0.46))
    norm = LogNorm(vmin=max(np.percentile(et, 1), 1e-2),
                   vmax=np.percentile(et, 99))
    order = np.argsort(-et)          # draw accurate points last, on top
    sc = None
    for j, ax in enumerate(axes.ravel()[:5]):
        lo = min(yt[:, j].min(), yp[:, j].min())
        hi = max(yt[:, j].max(), yp[:, j].max())
        ax.plot([lo, hi], [lo, hi], color="0.25", lw=0.7, ls=(0, (4, 2)), zorder=1)
        # viridis, not its reverse: an accurate sample is dark, so "dark point
        # off the diagonal" reads directly as "different posture, tool on target"
        sc = ax.scatter(yt[order, j], yp[order, j], c=et[order], norm=norm,
                        cmap="viridis", s=1.1, linewidths=0, rasterized=True,
                        zorder=2)
        ax.set_xlabel(f"true {IK_JOINT_LABELS[j]} (deg)")
        ax.set_ylabel(f"predicted {IK_JOINT_LABELS[j]} (deg)")
        ax.set_title(JOINT_NAMES[IK_JOINTS[j]], fontsize=7, pad=3)
        ax.grid(alpha=0.25)

    ax = axes.ravel()[5]
    ax.axis("off")
    if stats:
        sw = stats.get("threshold_sweep", {}).get(f"tau_x_{TAU_X_MM}mm", stats)
        txt = (
            f"Method: {method.replace('_', ' ')}\n"
            "Colour: task-space error, dark is accurate.\n\n"
            "A dark point away from the diagonal is a\n"
            "valid alternative posture, not an error.\n\n"
            f"tool within {TAU_X_MM:.0f} mm: "
            f"{sw.get('task_success_pct', float('nan')):.2f}%\n"
            f"of those, posture differs by\n"
            f"  more than {TAU_THETA_DEG:.0f}$^\\circ$: "
            f"{sw.get('share_of_successes_that_are_alternative_pct', float('nan')):.2f}%\n"
            f"tool off target: "
            f"{sw.get('failure_rate_pct', float('nan')):.2f}%\n\n"
            "Joint error among solved samples:\n"
            f"  median {stats.get('joint_error_given_success_deg', {}).get('median', float('nan')):.1f}"
            f"$^\\circ$, P95 "
            f"{stats.get('joint_error_given_success_deg', {}).get('p95', float('nan')):.1f}$^\\circ$"
        )
        ax.text(0.0, 0.98, txt, va="top", ha="left", fontsize=6.4,
                transform=ax.transAxes)
    if sc is not None:
        # a dedicated axes under the text block, so the bar never lands on it
        cax = fig.add_axes([0.695, 0.075, 0.20, 0.020])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label("Task-space error (mm)", fontsize=6.4, labelpad=1)
        cb.ax.tick_params(labelsize=5.8, pad=1)
    for k, a in enumerate(axes.ravel()[:5]):
        _panel(a, f"({chr(97 + k)})", dx=-0.30, dy=1.20)
    _save(fig, f"fig_parity_{method}")


def fig_boxplot_grid(art: Artefacts, plt) -> None:
    """Seed-level distribution for every architecture-optimizer cell.

    Two layouts are available through --boxplot-style.

    "single" puts all cells on one logarithmic axis. This is the conventional
    layout and it is what a reader expects, because the ordering of the methods
    is then immediate. Its cost is that a four-decade axis compresses each box
    to near a line, so the median value is printed above each group to recover
    the resolution the axis loses.

    "facet" gives every method its own linear axis. Nothing is compressed and
    the within-cell spread is fully visible, at the price of five different
    vertical scales the reader has to track.

    Neither is more correct; the single-panel version is the default because it
    answers the comparison question, and the faceted one is the supplement when
    the spread within a cell is the point.
    """
    import pandas as pd

    style = getattr(fig_boxplot_grid, "style", "single")
    csv = os.path.join(PATHS["out_dir"], "per_model.csv")
    if not os.path.exists(csv):
        log("[skip] boxplot: per_model.csv missing", 1)
        return
    df = pd.read_csv(csv)
    df = df[df["optimizer"].isin(OPTIMIZERS)]

    groups = [
        ("mlp", "MLP"),
        ("mdn_argmax_pi", r"MDN, $\arg\max\pi_k$"),
        ("mdn_fk_best", "MDN, FK selection"),
        ("mlp_nr", "MLP + NR"),
        ("mdn_nr", "MDN + NR"),
    ]
    groups = [(m, l) for m, l in groups if m in set(df["method"])]
    if not groups:
        log("[skip] boxplot: no matching methods", 1)
        return

    colors = {"adamw": OKABE["blue"], "muon": OKABE["vermil"],
              "shampoo": OKABE["green"]}
    hatches = {"adamw": "", "muon": "//", "shampoo": ".."}
    rng = np.random.default_rng(RNG_SEED)

    def draw(ax, data, cols, hats, pos, width=0.68):
        bp = ax.boxplot(data, positions=pos, widths=width, patch_artist=True,
                        medianprops=dict(color="black", lw=1.1),
                        whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                        flierprops=dict(marker="", ms=0))
        for patch, c, h in zip(bp["boxes"], cols, hats):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)
            patch.set_hatch(h)
        for pp, v in zip(pos, data):
            ax.scatter(pp + rng.uniform(-0.15, 0.15, v.size), v, s=4.5,
                       color="black", alpha=0.7, zorder=4, linewidths=0)

    if style == "facet":
        fig, axes = plt.subplots(1, len(groups), figsize=(COL2, 2.5),
                                 gridspec_kw=dict(wspace=0.55))
        axes = np.atleast_1d(axes)
        for ax, (method, label) in zip(axes, groups):
            data, cols, hats, ticks = [], [], [], []
            for opt in OPTIMIZERS:
                v = df[(df["method"] == method) &
                       (df["optimizer"] == opt)]["err_mean_mm"].to_numpy()
                if v.size:
                    data.append(v); cols.append(colors[opt])
                    hats.append(hatches[opt]); ticks.append(OPT_DISPLAY[opt])
            if not data:
                continue
            pos = np.arange(len(data), dtype=float)
            draw(ax, data, cols, hats, pos, width=0.62)
            ax.set_xticks(pos)
            ax.set_xticklabels(ticks, fontsize=6.0, rotation=32, ha="right")
            ax.set_title(label, fontsize=6.6, pad=4)
            ax.set_xlim(-0.6, len(data) - 0.4)
            ax.tick_params(axis="y", labelsize=6.0)
            ax.grid(axis="x", visible=False)
            lo = min(v.min() for v in data); hi = max(v.max() for v in data)
            pad = max((hi - lo) * 0.25, hi * 0.002)
            ax.set_ylim(lo - pad, hi + pad)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3),
                                useMathText=True)
            ax.yaxis.get_offset_text().set_fontsize(5.6)
        axes[0].set_ylabel("Mean position error (mm)")
        note = ("Each box summarises ten independent seeds; points are the "
                "individual runs. Vertical scales differ between panels.")
        fig.text(0.5, -0.14, note, ha="center", fontsize=6.0, color="0.35")
        _save(fig, "fig_boxplot_facet")
        return

    # ---- single panel, logarithmic --------------------------------------
    fig, ax = plt.subplots(figsize=(COL2, 2.7))
    pos, data, cols, hats, centres, labels, medians, tops = [], [], [], [], [], [], [], []
    p = 0.0
    for method, label in groups:
        start = p
        grp = []
        gmax = 0.0
        for opt in OPTIMIZERS:
            v = df[(df["method"] == method) &
                   (df["optimizer"] == opt)]["err_mean_mm"].to_numpy()
            if v.size == 0:
                continue
            pos.append(p); data.append(v); cols.append(colors[opt])
            hats.append(hatches[opt]); grp.append(np.median(v))
            gmax = max(gmax, float(v.max()))
            p += 1.0
        centres.append((start + p - 1.0) / 2.0)
        labels.append(label)
        medians.append(float(np.median(grp)) if grp else np.nan)
        tops.append(gmax)
        p += 1.2
    draw(ax, data, cols, hats, np.asarray(pos))

    ax.set_yscale("log")
    ax.set_ylim(min(v.min() for v in data) * 0.35,
                max(v.max() for v in data) * 3.2)
    for c, med, gmax in zip(centres, medians, tops):
        # the log axis compresses each box, so the number it hides is printed
        # directly above the group it belongs to
        ax.annotate(f"{med:.4g}", xy=(c, gmax * 1.7), ha="center",
                    fontsize=6.2, fontweight="bold")
    ax.set_xticks(centres)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean position error (mm, log scale)")
    ax.set_xlim(-0.9, p - 0.7)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[o], alpha=0.55,
                             edgecolor="black", linewidth=0.5, hatch=hatches[o],
                             label=OPT_DISPLAY[o]) for o in OPTIMIZERS]
    ax.legend(handles=handles, ncol=3, fontsize=6.6, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    fig.text(0.5, -0.28,
             "Boxes are the ten seeds of each architecture-optimizer cell; "
             "points are the individual runs; the value above each group is its "
             "median in mm.", ha="center", fontsize=6.0, color="0.35")
    _save(fig, "fig_boxplot_grid")


def fig_multimodality_summary(art: Artefacts, plt) -> None:
    """How much of a method's success comes from a posture other than the label.

    The earlier version of this figure stacked three shares per method and was
    unreadable, mostly because the shares are dominated by whether the method
    reaches the tolerance at all, which Table~I already answers. The question
    that belongs here is narrower and is asked conditionally: *given* that the
    tool arrived, did it arrive through the recorded posture?

    Panel (a) answers that across tolerances, so the reader can see the effect is
    not an artefact of one threshold. Panel (b) gives the size of the
    disagreement in degrees. Under a single-valued mapping both panels would sit
    at zero.

    Newton-Raphson variants are excluded by default: refinement moves the error,
    not the ambiguity, and including them invites the reader to attribute the
    multi-valuedness to the solver rather than to the problem.
    """
    mm = load_json(os.path.join(PATHS["out_dir"], "multimodality.json"), {})
    methods = mm.get("methods", {})
    if not methods:
        log("[skip] multimodality summary: run --stage multimodality first", 1)
        return

    include_nr = getattr(fig_multimodality_summary, "include_nr", False)
    catalogue = [
        ("mlp", "MLP", OKABE["vermil"], "-", "o"),
        ("mdn_argmax_pi", r"MDN, $\arg\max\pi_k$", OKABE["orange"], "--", "s"),
        ("mdn_fk_best", "MDN, FK selection", OKABE["green"], "-.", "D"),
    ]
    if include_nr:
        catalogue += [("mlp_nr", "MLP + NR", OKABE["blue"], "-", "P"),
                      ("mdn_nr", "MDN + NR", OKABE["sky"], ":", "^")]
    series = [c for c in catalogue if c[0] in methods]
    if not series:
        log("[skip] multimodality summary: no matching methods", 1)
        return

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6),
                             gridspec_kw=dict(width_ratios=[1.25, 1.0],
                                              wspace=0.40))
    ax, axb = axes

    # ---- (a) conditional share, swept over the tolerance -------------------
    for key, lbl, c, ls, m in series:
        sweep = methods[key].get("threshold_sweep", {})
        xs, ys, ns = [], [], []
        for t in TAU_X_SWEEP_MM:
            e = sweep.get(f"tau_x_{t}mm")
            if not e:
                continue
            v = e.get("share_of_successes_that_are_alternative_pct")
            if v is None or np.isnan(v):
                continue
            # a share computed on a handful of samples is noise, not a result
            if e.get("task_success_pct", 0.0) < 1.0:
                continue
            xs.append(t)
            ys.append(v)
            ns.append(e["task_success_pct"])
        if xs:
            ax.plot(xs, ys, color=c, ls=ls, marker=m, markersize=3.4, label=lbl)
    ax.set_xscale("log")
    ax.set_xlabel(r"Tolerance $\tau_x$ (mm)")
    ax.set_ylabel("Successes reached through a posture\n"
                  rf"differing by $>{TAU_THETA_DEG:.0f}^\circ$ (%)")
    ax.set_ylim(-3, 103)
    ax.axhline(0, color="0.4", lw=0.6, ls=(0, (3, 3)))
    ax.annotate("a single-valued mapping would lie here",
                xy=(TAU_X_SWEEP_MM[0], 3.5), fontsize=5.8, color="0.4")
    ax.legend(fontsize=6.2, loc="lower right")
    ax.annotate("curves start where the method solves at least 1% of targets",
                xy=(0.5, -0.32), xycoords="axes fraction", ha="center",
                fontsize=5.7, color="0.4")
    _panel(ax, "(a)", dx=-0.26)

    # ---- (b) how large the disagreement is ---------------------------------
    # A single tolerance would leave the weaker methods out of this panel
    # entirely, so each method is read at the tightest threshold in the sweep
    # where it still solves at least 5% of the targets, and that threshold is
    # printed next to the bar.
    labs, med, p95, cols, taus = [], [], [], [], []
    for key, lbl, c, ls, m in series:
        sweep = methods[key].get("threshold_sweep", {})
        chosen = None
        for t in TAU_X_SWEEP_MM:
            e = sweep.get(f"tau_x_{t}mm")
            if e and e.get("task_success_pct", 0.0) >= 5.0:
                chosen = (t, e)
                break
        if chosen is None:
            g = methods[key].get("joint_error_given_success_deg", {})
            if g and not np.isnan(g.get("median", float("nan"))):
                labs.append(lbl); med.append(g["median"]); p95.append(g["p95"])
                cols.append(c); taus.append(TAU_X_MM)
            continue
        t, e = chosen
        mv = e.get("median_joint_error_given_success_deg")
        pv = e.get("p95_joint_error_given_success_deg", mv)
        if mv is None or np.isnan(mv):
            continue
        labs.append(lbl); med.append(mv)
        p95.append(pv if pv is not None and not np.isnan(pv) else mv)
        cols.append(c); taus.append(t)
    if labs:
        y = np.arange(len(labs))
        axb.barh(y, p95, color="0.88", edgecolor="black", linewidth=0.5,
                 height=0.6, label="P95")
        axb.barh(y, med, color=cols, edgecolor="black", linewidth=0.5,
                 height=0.6)
        for yi, (a_, b_, t_) in enumerate(zip(med, p95, taus)):
            axb.text(b_ + max(p95) * 0.02, yi,
                     f"{a_:.0f} / {b_:.0f}  ($\\tau_x$={t_:g} mm)",
                     va="center", fontsize=5.8)
        axb.set_yticks(y)
        axb.set_yticklabels(labs, fontsize=6.4)
        axb.invert_yaxis()
        axb.set_xlim(0, max(p95) * 1.62)
        axb.set_xlabel("Joint error among solved samples (deg)", fontsize=7)
        axb.set_title("median / P95, each read at the tightest\n"
                      "tolerance the method reaches", fontsize=6.4, pad=4)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor="0.88",
                                 edgecolor="black", linewidth=0.5, label="P95"),
                   plt.Rectangle((0, 0), 1, 1, facecolor="0.35",
                                 edgecolor="black", linewidth=0.5,
                                 label="median (bar colour is the method)")]
        axb.legend(handles=handles, fontsize=5.8, loc="lower right")
        axb.grid(axis="y", visible=False)
    _panel(axb, "(b)", dx=-0.40)

    _save(fig, "fig_multimodality_summary")


def stage_figures(args) -> None:
    rule("STAGE 2/4  NEW FIGURES")
    plt = setup_mpl(args.font_size)
    art = Artefacts()
    fig_parity.method = getattr(args, "parity_method", "mlp_nr")
    fig_boxplot_grid.style = getattr(args, "boxplot_style", "single")
    fig_multimodality_summary.include_nr = getattr(args, "multimodality_with_nr", False)
    for fn in (fig_parity, fig_boxplot_grid, fig_multimodality_summary):
        try:
            fn(art, plt)
        except Exception as exc:
            import traceback

            log(f"FAILED {fn.__name__}: {exc}", 1)
            traceback.print_exc()


# ============================================================================
# STAGE 3 -- TABLES
# ============================================================================

def _tex(name: str, body: str) -> None:
    os.makedirs(PATHS["tab_dir"], exist_ok=True)
    p = os.path.join(PATHS["tab_dir"], name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(body + "\n")
    log(f"wrote {p}", 1)


def stage_tables(args) -> None:
    """Every table of the manuscript, generated from the artefacts."""
    rule("STAGE 3/4  LATEX TABLES")
    import pandas as pd

    art = Artefacts()
    df = pd.read_csv(os.path.join(PATHS["out_dir"], "per_model.csv"))
    df = df[df["optimizer"].isin(OPTIMIZERS)]
    mm = load_json(os.path.join(PATHS["out_dir"], "multimodality.json"), {})
    methods_mm = mm.get("methods", {})

    display = [
        ("mlp", "MLP"),
        ("mdn_argmax_pi", r"MDN, $\arg\max_k \pi_k$"),
        ("mdn_oracle_joint", "MDN, joint-space oracle"),
        ("mdn_fk_best", r"\textbf{MDN, FK selection}"),
        ("mlp_nr_iter1", "MLP + NR (1 it.)"),
        ("mlp_nr_iter3", "MLP + NR (3 it.)"),
        ("mlp_nr", "MLP + NR (converged)"),
        ("mdn_nr", "MDN + NR (converged)"),
    ]

    # ---------------- Table I : main comparison -----------------------------
    L = [
        r"% generated by paper_assets.py",
        r"\begin{table*}[!t]",
        r"\caption{Task-space accuracy on the complete test split "
        r"($n = 50{,}000$ targets, mean $\pm$ standard deviation over ten seeds "
        r"per configuration). No samples are discarded. Errors are Euclidean "
        r"distances between the requested position and the position reached by "
        r"the predicted configuration.}",
        r"\label{tab:main}",
        r"\centering",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Optimizer & Mean (mm) & Median (mm) & P95 (mm) & "
        r"$d\!\leq\!1$\,mm (\%) & $d\!\leq\!5$\,mm (\%) & Latency (ms) \\",
        r"\midrule",
    ]
    for method, label in display:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        first = True
        for opt in OPTIMIZERS:
            s = sub[sub["optimizer"] == opt]
            if s.empty:
                continue
            lat = s["t_numpy_single_ms"].mean()
            if not np.isfinite(lat):
                lat = s["t_call_single_ms"].mean()
            L.append(
                f"{label if first else ''} & {OPT_DISPLAY[opt]} & "
                f"${fmt(s['err_mean_mm'].mean(), 4)} \\pm "
                f"{fmt(s['err_mean_mm'].std(ddof=1), 4)}$ & "
                f"{fmt(s['err_median_mm'].mean(), 4)} & "
                f"{fmt(s['err_p95_mm'].mean(), 4)} & "
                f"{fmt(s['succ_1.0mm_pct'].mean(), 2)} & "
                f"{fmt(s['succ_5.0mm_pct'].mean(), 2)} & "
                f"{fmt(lat, 3)} \\\\")
            first = False
        L.append(r"\addlinespace")
    L.append(r"\midrule")
    L.append(r"\multicolumn{8}{l}{\textit{Solvers without a learned "
             r"initialisation, identical damping, tolerance and budget}} \\")
    for b in art.baselines:
        pe = b["task_space"]["position_error_mm"]
        sr = b["task_space"]["success_rate_pct"]
        nm = b["method"].replace("_", " ")
        L.append(f"{nm} & -- & {fmt(pe['mean'], 4)} & {fmt(pe['median'], 4)} & "
                 f"{fmt(pe['p95'], 4)} & {fmt(sr['tau_1.0mm'], 2)} & "
                 f"{fmt(sr['tau_5.0mm'], 2)} & "
                 f"{fmt(b.get('nr', {}).get('timing', {}).get('single_ms', {}).get('mean'), 3)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    _tex("tab1_main.tex", "\n".join(L))

    # ---------------- Table II : multimodality ------------------------------
    L = [
        r"\begin{table}[!t]",
        rf"\caption{{Decomposition of every prediction into three outcomes, with "
        rf"$\tau_x = {TAU_X_MM:.0f}$\,mm and $\tau_\theta = {TAU_THETA_DEG:.0f}^\circ$. "
        rf"A prediction is an \emph{{alternative solution}} when it places the tool "
        rf"within $\tau_x$ of the target through a posture differing from the "
        rf"recorded label by more than $\tau_\theta$ on at least one joint. "
        rf"Joint-space statistics use joints 1--5; joint 6 actuates the gripper "
        rf"and is not part of the inverse kinematics mapping.}}",
        r"\label{tab:multimodality}",
        r"\centering",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Recorded & Alternative & Off target & Joint err.\ if \\",
        r" & posture (\%) & solution (\%) & (\%) & solved (deg) \\",
        r"\midrule",
    ]
    disp = {"mlp": "MLP", "mdn_argmax_pi": r"MDN, $\arg\max_k \pi_k$",
            "mdn_fk_best": "MDN, FK selection", "mlp_nr": "MLP + NR",
            "mdn_nr": "MDN + NR"}
    for m, lbl in disp.items():
        e = methods_mm.get(m)
        if not e:
            continue
        g = e.get("joint_error_given_success_deg", {})
        # the four shares partition the test set, so they must sum to 100
        near_off = (100.0 - e["matched_ground_truth_pct"]
                    - e["alternative_solution_rate_pct"] - e["failure_rate_pct"])
        L.append(f"{lbl} & {fmt(e['matched_ground_truth_pct'], 2)} & "
                 f"{fmt(e['alternative_solution_rate_pct'], 2)} & "
                 f"{fmt(near_off, 2)} & "
                 f"{fmt(e['failure_rate_pct'], 2)} & "
                 f"{fmt(g.get('median'), 1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex("tab2_multimodality.tex", "\n".join(L))

    # ---------------- Table III : mixture selection ablation ----------------
    L = [
        r"\begin{table}[!t]",
        r"\caption{Ablation of the rule used to turn the predictive mixture into "
        r"a single configuration. All rows use the same trained networks; only "
        r"the selection rule changes. The joint-space oracle is the rule adopted "
        r"in earlier work and is included to show that it does not upper-bound "
        r"task-space performance.}",
        r"\label{tab:selection}",
        r"\centering",
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"Selection rule & Mean (mm) & P95 (mm) & $d\!\leq\!1$\,mm (\%) & Requires \\",
        r"\midrule",
    ]
    sel_rows = [
        ("mdn_argmax_pi", r"$\arg\max_k \pi_k$", "nothing"),
        ("mdn_oracle_joint", "nearest mean in joint space", "ground truth"),
        ("mdn_fk_sampled", r"best of $S$ samples by FK", "forward kinematics"),
        ("mdn_fk_best", r"best of $K$ means by FK", "forward kinematics"),
    ]
    for key, lbl, req in sel_rows:
        s = df[df["method"] == key]
        if s.empty:
            continue
        L.append(f"{lbl} & {fmt(s['err_mean_mm'].mean(), 3)} & "
                 f"{fmt(s['err_p95_mm'].mean(), 3)} & "
                 f"{fmt(s['succ_1.0mm_pct'].mean(), 2)} & {req} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex("tab3_selection.tex", "\n".join(L))

    # ---------------- Table IV : optimizers ---------------------------------
    L = [
        r"\begin{table}[!t]",
        r"\caption{AdamW versus Muon, paired over ten seeds. $p$ is the "
        r"Wilcoxon signed-rank test and $p_{\mathrm{Holm}}$ applies the "
        r"Holm--Bonferroni correction over the family of primary comparisons "
        r"reported here. Equivalence is a two one-sided test against a "
        r"$\pm0.5$\,mm margin, so \emph{yes} states that the difference is "
        r"confidently smaller than any practically relevant size. A third "
        r"training arm was excluded; see Section~\ref{sec:audit}.}",
        r"\label{tab:optimizers}",
        r"\centering",
        r"\begin{tabular}{lrlrrc}",
        r"\toprule",
        r"Model family & $\Delta$ (mm) & 95\% CI & $p$ & "
        r"$p_{\mathrm{Holm}}$ & Equivalent \\",
        r"\midrule",
    ]
    ostats = load_json(os.path.join(PATHS["out_dir"], "optimizer_stats.json"),
                       {"pairwise": [], "variance_share": {}})
    for rec in ostats.get("pairwise", art.stats.get("pairwise", [])):
        if not rec.get("in_holm_family", rec["method"] in PRIMARY_METHODS):
            continue
        lo, hi = rec["delta_ci95_mm"]
        L.append(f"{rec['method'].replace('_', ' ')} & "
                 f"{fmt(rec['delta_mm'], 4)} & "
                 f"[{fmt(lo, 4)}, {fmt(hi, 4)}] & "
                 f"{fmt(rec['p_raw'], 4)} & "
                 f"{fmt(rec.get('p_holm'), 4)} & "
                 f"{'yes' if rec.get('tost', {}).get('equivalent') else 'no'} \\\\")
    L += [r"\midrule"]
    sv = ostats.get("variance_share", {})
    for k in ("mlp", "mdn_fk_best", "mlp_nr"):
        if k in sv:
            L.append(f"\\multicolumn{{6}}{{l}}{{\\footnotesize "
                     f"{k.replace('_', ' ')}: the optimizer explains "
                     f"{sv[k]['optimizer_pct']:.0f}\\% of "
                     f"the variance across runs.}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _tex("tab4_optimizers.tex", "\n".join(L))

    # ---------------- Table V : training cost -------------------------------
    summ = art.history.get("summary", {})
    if summ:
        L = [
            r"\begin{table}[!t]",
            r"\caption{Training cost and convergence, ten runs per cell, on a "
            r"quad-core CPU without any accelerator. A third arm was trained but "
            r"is excluded from the analysis; see Section~\ref{sec:audit}.}",
            r"\label{tab:training}",
            r"\centering",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Model & Optimizer & Best epoch & Best val.\ loss & Time (min) \\",
            r"\midrule",
        ]
        for key in sorted(k for k in summ if not k.startswith("_")):
            v = summ[key]
            arch, opt = key.split("_", 1)
            if opt not in OPTIMIZERS:
                continue
            L.append(f"{arch.upper()} & {OPT_DISPLAY.get(opt, opt)} & "
                     f"${fmt(v['best_epoch']['mean'], 1)} \\pm "
                     f"{fmt(v['best_epoch']['std'], 1)}$ & "
                     f"{fmt(v['best_val_loss']['mean'], 4)} & "
                     f"{fmt(v['train_time_sec']['mean'] / 60, 1)} \\\\")
        kr, kh, ar, ah = _kept_compute(art)
        L.append(r"\midrule")
        L.append(f"\\multicolumn{{5}}{{l}}{{Total for the analysed arms: "
                 f"{kr} runs, {fmt(kh, 1)} h wall clock}} \\\\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        _tex("tab5_training.tex", "\n".join(L))

    # ---------------- Table VI : related work -------------------------------
    L = [
        r"\begin{table*}[!t]",
        r"\caption{Positioning against recent learning-based inverse kinematics "
        r"work. Task column distinguishes position-only targets from full "
        r"$SE(3)$ pose. Errors are as reported by the original authors and are "
        r"not directly comparable across different manipulators; the last column "
        r"normalises by the maximum reach where it can be established.}",
        r"\label{tab:related}",
        r"\centering",
        r"\begin{tabular}{lllclll}",
        r"\toprule",
        r"Work & Model & Manipulator & DOF & Task & Position error & Multi-valued "
        r"solutions \\",
        r"\midrule",
        r"Duka (2014) & shallow MLP & planar & 3 & position & n/r & not addressed \\",
        r"D'Souza \textit{et al.} (2001) & LWPR & SARCOS & 7 & velocity & n/r & "
        r"local averaging \\",
        r"Bensadoun \textit{et al.} (2022) & IKNet & --- & 4 & pose & "
        r"$\sim$123\,mm & hypernetwork \\",
        r"Ames \textit{et al.} (2022) & IKFlow & Panda & 7 & $SE(3)$ & "
        r"$<$10\,mm & normalising flow \\",
        r"Lu \textit{et al.} (2022) & MLP + partitioning & six-axis & 6 & pose & "
        r"$<$1\,mm & workspace split \\",
        r"Wagaa \textit{et al.} (2023) & CNN/LSTM/GRU & several & 3--6 & position & "
        r"$<$1\,mm & not addressed \\",
        r"Calzada-Garcia \textit{et al.} (2025) & MLP/CNN/RNN & TIAGo & 7 & "
        r"$SE(3)$ & 139\,mm & duplicates removed \\",
        r"Cruz-Caos \textit{et al.} (2025) & LSTM & 3D-printed & 4 & position & "
        r"1.07\,mm & not addressed \\",
        r"Filho and Santos (2026) & Hybrid-DNN + TRF & ABB IRB120 & 6 & $SE(3)$ & "
        r"1.406\,mm & $q_6$ constrained \\",
        r"\midrule",
        r"This work & MDN, FK selection & 5-DOF + gripper & 5 & position & "
        r"\textbf{FILL} & mixture, measured \\",
        r"This work & MDN + NR & 5-DOF + gripper & 5 & position & \textbf{FILL} & "
        r"mixture seeds solver \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    _tex("tab6_related.tex", "\n".join(L))
    log("tab6_related.tex has two FILL cells; complete them from Table I", 1)


# ============================================================================
# STAGE 4 -- THE NUMBERS SHEET
# ============================================================================

def _kept_compute(art: "Artefacts") -> Tuple[int, float, int, float]:
    """Runs and wall-clock hours, for the kept arms and for everything.

    The 60-run total in training_history.json includes the arm that was
    excluded, so quoting it beside a two-optimizer analysis would overstate the
    compute the reported results rest on. Both figures are given.
    """
    summ = art.history.get("summary", {})
    kept_runs = kept_sec = all_runs = all_sec = 0.0
    for key, v in summ.items():
        if key.startswith("_"):
            continue
        opt = key.split("_", 1)[1]
        n = v.get("n_runs", 0)
        sec = v.get("train_time_sec", {}).get("mean", 0.0) * n
        all_runs += n
        all_sec += sec
        if opt in OPTIMIZERS:
            kept_runs += n
            kept_sec += sec
    return int(kept_runs), kept_sec / 3600.0, int(all_runs), all_sec / 3600.0


def stage_numbers(args) -> None:
    """One markdown file holding every number that will appear in the text.

    Writing a manuscript by reading numbers off figures is how transcription
    errors enter a paper. This file is the single source for the abstract, the
    results section and the conclusion.
    """
    rule("STAGE 4/4  MANUSCRIPT NUMBERS")
    import pandas as pd

    art = Artefacts()
    df = pd.read_csv(os.path.join(PATHS["out_dir"], "per_model.csv"))
    df = df[df["optimizer"].isin(OPTIMIZERS)]
    mm = load_json(os.path.join(PATHS["out_dir"], "multimodality.json"), {})
    ds = art.master.get("dataset", {})
    aud = art.audit

    def best_of(method: str) -> Dict[str, Any]:
        s = df[df["method"] == method]
        if s.empty:
            return {}
        return {"mean": s["err_mean_mm"].mean(), "std": s["err_mean_mm"].std(ddof=1),
                "median": s["err_median_mm"].mean(), "p95": s["err_p95_mm"].mean(),
                "s1": s["succ_1.0mm_pct"].mean(),
                "iters": s["nr_mean_iters"].mean(),
                "conv": s["nr_convergence_pct"].mean()}

    lines = ["# Numbers for the manuscript", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Quote from here, not "
             f"from the figures.", "",
             "## Setup", ""]
    ws = ds.get("workspace", {})
    lines += [
        f"- Test targets: {ds.get('n_test', 'n/a')}; train {ds.get('n_train', 'n/a')}; "
        f"val {ds.get('n_val', 'n/a')}",
        f"- Maximum reach: {fmt(ws.get('radial_distance_cm', {}).get('max'), 2)} cm",
        f"- Workspace convex-hull volume: "
        f"{fmt(ws.get('volume_convex_hull_cm3'), 0)} cm^3",
        f"- IK mapping: R^3 -> R^5 (joint 6 actuates the gripper and is excluded)",
        f"- Training compute over the analysed arms: "
        f"{fmt(_kept_compute(art)[1], 1)} h over {_kept_compute(art)[0]} runs "
        f"on a quad-core CPU "
        f"(all {_kept_compute(art)[2]} runs including the excluded arm: "
        f"{fmt(_kept_compute(art)[3], 1)} h)",
        f"- Optimizers analysed: {', '.join(OPT_DISPLAY[o] for o in OPTIMIZERS)}; "
        f"excluded: {', '.join(EXCLUDED_OPTIMIZERS) or 'none'}",
        "",
        "## Integrity audit",
        "",
        f"- {aud.get('optimizer_audit', {}).get('verdict', 'not run')}",
        f"- Muon resolved to: "
        f"{aud.get('optimizer_audit', {}).get('requested', {}).get('muon', {}).get('class')}",
        f"- Shampoo resolved to (current environment): "
        f"{aud.get('optimizer_audit', {}).get('requested', {}).get('shampoo', {}).get('class')}",
        f"- Recorded inside the checkpoints: see optimizer_provenance.json",
        "",
        "## Headline accuracy", "",
        "| Method | mean (mm) | median | P95 | <=1 mm (%) | NR iters | conv (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in ("mlp", "mdn_argmax_pi", "mdn_oracle_joint", "mdn_fk_best",
              "mlp_nr_iter1", "mlp_nr_iter3", "mlp_nr", "mdn_nr"):
        b = best_of(m)
        if not b:
            continue
        lines.append(f"| {m} | {fmt(b['mean'], 4)} | {fmt(b['median'], 4)} | "
                     f"{fmt(b['p95'], 4)} | {fmt(b['s1'], 2)} | "
                     f"{fmt(b['iters'], 2)} | {fmt(b['conv'], 2)} |")

    lines += ["", "## Baselines without a learned initialisation", "",
              "| Method | mean (mm) | iters | conv (%) |", "|---|---|---|---|"]
    for b in art.baselines:
        nr = b.get("nr", {})
        lines.append(f"| {b['method']} | "
                     f"{fmt(b['task_space']['position_error_mm']['mean'], 4)} | "
                     f"{fmt(nr.get('mean_iterations'), 2)} | "
                     f"{fmt(nr.get('convergence_rate_pct'), 2)} |")

    lines += ["", "## Multimodality", ""]
    for m, e in mm.get("methods", {}).items():
        g = e.get("joint_error_given_success_deg", {})
        lines.append(f"- **{m}**: tool on target {e['task_success_pct']:.2f}%, of "
                     f"which {e['share_of_successes_that_are_alternative_pct']:.2f}% "
                     f"reach it through a posture differing by more than "
                     f"{TAU_THETA_DEG:.0f} deg; median joint error among solved "
                     f"samples {fmt(g.get('median'), 1)} deg")
    mc = mm.get("mdn_mode_count", {})
    if mc:
        lines.append(f"- The mixture offers on average "
                     f"{fmt(mc.get('mean_distinct_valid_postures'), 2)} distinct "
                     f"valid postures per target; "
                     f"{fmt(mc.get('share_with_2_or_more'), 1)}% of targets have "
                     f"two or more")

    lines += ["", "## Trajectory continuity", ""]
    for pname, entry in art.trajectory.get("paths", {}).items():
        for k, v in entry.items():
            if isinstance(v, dict) and "tracking_error_mm" in v:
                jc = v["joint_continuity"]
                lines.append(f"- {pname} / {k}: "
                             f"{v['tracking_error_mm']['mean']:.4f} mm, largest "
                             f"joint step {jc['max_step_deg']:.1f} deg, "
                             f"{jc['n_discontinuities_gt_10deg']} steps above 10 deg")
        break

    lines += ["", "## Comparison anchors from the literature", "",
              "- Filho and Santos (IEEE Access, 2026), ABB IRB120, SE(3): "
              "Hybrid-DNN 1.406 mm and 0.900 deg, 94.2% success, 14.592 function "
              "evaluations, 117.651 ms; Conventional-Multistart-10 1.019 mm, "
              "98.2%, 294.970 evaluations, 671.884 ms; standalone DNN in the full "
              "joint space 126.050 mm and 0.0% success.",
              "- Calzada-Garcia et al. (Appl. Sci., 2025), TIAGo, SE(3): best model "
              "SubnetMlp, 0.1395 m position error; classical KDL solvers failed to "
              "converge on about 13% of targets.",
              "- Cruz-Caos et al. (ENC, 2025), 4-DOF, position only: LSTM 1.069 mm "
              "mean, kinematic control 0.009 mm.",
              "",
              "## Claims that must NOT be made", "",
              "- Do not claim the hybrid learned-seed formulation as novel; it is "
              "published in the target venue (Filho and Santos, 2026). Frame it as "
              "a reproduction and as the baseline the MDN is measured against.",
              "- Do not report joint-space MAE as a quality metric without the "
              "alternative-solution decomposition beside it.",
              "- Report the optimizer factor as AdamW versus Muon only. The "
              "third arm requested Shampoo and every checkpoint recorded Adam, "
              "so it is excluded; say so in the methodology and cite the reason "
              "(tensorflow_addons is archived and incompatible with TF >= 2.14).",
              "- Quote the training compute of the analysed arms, not the 60-run "
              "total, when describing the cost of the reported results.",
              ]
    p = os.path.join(PATHS["out_dir"], "MANUSCRIPT_NUMBERS.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote {p}", 1)


# ============================================================================
# ENTRY POINT
# ============================================================================

STAGES = {
    "provenance": stage_provenance,
    "optstats": stage_optstats,
    "multimodality": stage_multimodality,
    "figures": stage_figures,
    "tables": stage_tables,
    "numbers": stage_numbers,
}
ORDER = ["provenance", "optstats", "multimodality", "figures",
         "tables", "numbers"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Final analyses, tables and figures for the IEEE Access "
                    "manuscript. Run after extract_paper_data.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--stage", nargs="+", default=["all"],
                    choices=["all"] + ORDER)
    ap.add_argument("--out-dir", default="PAPER_DATA")
    ap.add_argument("--model-dir", default="Model")
    ap.add_argument("--data-npz", default="dataset_processed.npz")
    ap.add_argument("--scaler", default="scaler_X.pkl")
    ap.add_argument("--optimizers", nargs="+", default=["adamw", "muon"],
                    choices=["adamw", "muon", "shampoo"],
                    help="training arms kept in the analysis")
    ap.add_argument("--n-mixes", type=int, default=20)
    ap.add_argument("--nr-iters", type=int, default=10)
    ap.add_argument("--posture-examples", type=int, default=60,
                    help="targets kept for the 3D posture figure")
    ap.add_argument("--parity-samples", type=int, default=20000,
                    help="points kept per method for the parity plots")
    ap.add_argument("--boxplot-style", choices=["single", "facet"],
                    default="single",
                    help="one logarithmic axis, or one linear axis per method")
    ap.add_argument("--multimodality-with-nr", action="store_true",
                    help="include the Newton-Raphson variants in the "
                         "multimodality figure; off by default because "
                         "refinement changes the error, not the ambiguity")
    ap.add_argument("--parity-method", default="mlp_nr",
                    help="method shown in the parity figure; a solver that "
                         "actually reaches the tolerance makes the "
                         "alternative-solution reading meaningful")
    ap.add_argument("--font-size", type=float, default=8.0)
    args = ap.parse_args(argv)

    global OPTIMIZERS, EXCLUDED_OPTIMIZERS
    OPTIMIZERS = list(args.optimizers)
    EXCLUDED_OPTIMIZERS = [o for o in ("adamw", "muon", "shampoo")
                           if o not in OPTIMIZERS]

    PATHS["out_dir"] = os.path.expanduser(args.out_dir)
    PATHS["fig_dir"] = os.path.join(PATHS["out_dir"], "figures")
    PATHS["tab_dir"] = os.path.join(PATHS["out_dir"], "tables")
    PATHS["raw_dir"] = os.path.join(PATHS["out_dir"], "per_sample")
    PATHS["model_dir"] = os.path.expanduser(args.model_dir)
    PATHS["data_npz"] = os.path.expanduser(args.data_npz)
    PATHS["scaler"] = os.path.expanduser(args.scaler)

    if not os.path.isdir(PATHS["out_dir"]):
        print(f"error: {PATHS['out_dir']} not found. Run extract_paper_data.py "
              f"first, or pass --out-dir.")
        return 2
    for d in ("fig_dir", "tab_dir", "raw_dir"):
        os.makedirs(PATHS[d], exist_ok=True)

    todo = ORDER if "all" in args.stage else args.stage
    rule("PAPER ASSETS")
    log(f"stages: {', '.join(todo)}")
    log(f"artefacts: {os.path.abspath(PATHS['out_dir'])}")

    failed = []
    t0 = time.perf_counter()
    for s in todo:
        try:
            STAGES[s](args)
        except Exception as exc:
            import traceback

            log(f"STAGE '{s}' FAILED: {exc}", 1)
            traceback.print_exc()
            failed.append(s)

    rule("DONE")
    log(f"elapsed {(time.perf_counter() - t0) / 60:.1f} min")
    if failed:
        log(f"failed: {failed}")
    else:
        log(f"read {os.path.join(PATHS['out_dir'], 'MANUSCRIPT_NUMBERS.md')} next")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
