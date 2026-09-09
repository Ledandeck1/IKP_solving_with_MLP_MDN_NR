#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
make_figures.py  --  publication figures for the IEEE Access manuscript
--------------------------------------------------------------------------------
Standalone replacement for `stage_figures` in extract_paper_data.py. It reads
only the artefacts already written to PAPER_DATA/, so figures can be iterated on
in seconds without re-running any model.

    python make_figures.py                 # all figures
    python make_figures.py --only 2 5 6    # selected figures
    python make_figures.py --grayscale-check

--------------------------------------------------------------------------------
WHY THIS SET OF FIGURES
--------------------------------------------------------------------------------
The previous set had one figure per computed quantity. A journal figure has to
carry an argument, not a quantity. Each figure below states one claim, and every
claim is one the manuscript needs:

 F1  Success rate versus tolerance.
     Replaces the error CDF. The CDF made every method look adequate because
     the eye reads "the curve reaches 100%". Plotted as the fraction of targets
     solved within a user-specified tolerance, the separation between methods
     becomes the subject of the figure rather than a detail of the axis range.

 F2  Mixture selection is the bottleneck, not mixture learning.
     The paper's strongest new result. The FK-based selector reaches 1.89 mm
     against 4.83 mm for argmax-pi, while the joint-space oracle used in the
     conference version is worse than both at 5.62 mm despite having the lowest
     joint-space error. Panel (b) makes the decoupling explicit: joint proximity
     and task proximity are different objectives on a redundant arm.

 F3  What the network contributes is iterations, not accuracy.
     Final accuracy is a property of Newton-Raphson; every seeding strategy
     reaches the tolerance eventually. The seed determines how fast. Panel (b)
     turns that into the number a practitioner budgets for.

 F4  The optimizer comparison, reported honestly.
     Paired per-seed lines plus a forest plot with the equivalence margin drawn
     in. AdamW versus Muon is statistically significant and practically
     negligible at the same time, and the figure shows both facts at once.

 F5  Where in the workspace the error lives.
     Rebuilt with scipy.stats.binned_statistic_2d, one shared logarithmic
     colour scale, and the kinematic structure annotated. The previous version
     had three colourbars overlapping the axis labels.

 F6  Accuracy is not enough: joint-space continuity along a Cartesian path.
     The MDN attains 1.13 mm per waypoint yet jumps up to 121 degrees between
     adjacent waypoints. No reviewer will accept the solver as usable without
     this figure, and no competing paper in this space shows it.

 F7  The MDN discovers the unobservable joint on its own.
     Predicted sigma is 23.8 degrees on the wrist twist against 4-7 degrees
     elsewhere. That is the density model correctly reporting that the joint is
     unidentifiable from position, and it is the cleanest possible evidence
     that the network learned the structure of the problem.

 F8  Accuracy, latency and iteration cost on one Pareto plane.

--------------------------------------------------------------------------------
FORMATTING
--------------------------------------------------------------------------------
IEEE Access is online-only and colour is free, but the house guidance still asks
that figures survive a greyscale printout and that meaning never rests on hue
alone. Every series therefore carries a distinct line style and marker as well
as a colour, and the palette is Okabe-Ito, which is safe for the common forms of
colour vision deficiency. Output is vector PDF, with a 600 dpi PNG alongside for
drafts. Widths are the IEEE single column (3.5 in) and double column (7.16 in).

Author: Gustavo Henrique Germano Ledandeck (UFABC)
================================================================================
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------ paths ----
# Every path the module touches lives here, and every one of them is settable
# from the command line. Nothing is resolved relative to the location of this
# file, so it can be copied anywhere or run from any working directory.
PATHS: Dict[str, str] = {
    "out_dir":   "PAPER_DATA",
    "fig_dir":   os.path.join("PAPER_DATA", "figures"),
    "raw_dir":   os.path.join("PAPER_DATA", "per_sample"),
    "data_npz":  "dataset_processed.npz",
    "scaler":    "scaler_X.pkl",
}

OUT_DIR = PATHS["out_dir"]
FIG_DIR = PATHS["fig_dir"]
RAW_DIR = PATHS["raw_dir"]

COL1, COL2 = 3.5, 7.16          # IEEE column widths, inches
TAUS_MM = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist pitch", "Wrist roll", "Wrist twist"]
# The third arm requested Shampoo and every checkpoint recorded Adam, so it is
# excluded from the analysis. See optimizer_provenance.json.
OPTIMIZERS = ["adamw", "muon"]

# Okabe-Ito, safe under deuteranopia and protanopia
OKABE = {
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "green":   "#009E73",
    "vermil":  "#D55E00",
    "purple":  "#CC79A7",
    "sky":     "#56B4E9",
    "yellow":  "#F0E442",
    "black":   "#000000",
    "grey":    "#999999",
}

# One visual identity per method, reused across every figure so the reader
# learns it once. Colour, line style and marker all carry the distinction.
STYLE: Dict[str, Dict[str, Any]] = {
    "mlp":              dict(c=OKABE["vermil"], ls="-",   m="o", label="MLP"),
    "mdn_argmax_pi":    dict(c=OKABE["orange"], ls="--",  m="s", label=r"MDN, $\arg\max\pi_k$"),
    "mdn_oracle_joint": dict(c=OKABE["purple"], ls=":",   m="v", label="MDN, joint-space oracle"),
    "mdn_fk_best":      dict(c=OKABE["green"],  ls="-.",  m="D", label="MDN, FK selection"),
    "mdn_nr":           dict(c=OKABE["sky"],    ls="-",   m="^", label="MDN + NR"),
    "mlp_nr":           dict(c=OKABE["blue"],   ls="-",   m="P", label="MLP + NR"),
    "mlp_nr_iter1":     dict(c=OKABE["blue"],   ls=":",   m="P", label="MLP + NR (1 it.)"),
    "mlp_nr_iter2":     dict(c=OKABE["blue"],   ls="--",  m="P", label="MLP + NR (2 it.)"),
    "mlp_nr_iter3":     dict(c=OKABE["blue"],   ls="-.",  m="P", label="MLP + NR (3 it.)"),
    "nr_midrange_nr":   dict(c=OKABE["grey"],   ls="--",  m="X", label="NR, mid-range seed"),
    "nr_random_in_limits_nr": dict(c=OKABE["grey"], ls=":", m="*", label="NR, random seed"),
    "nr_random_legacy_nr":    dict(c=OKABE["black"], ls=":", m="x", label="NR, seed outside joint limits"),
    "nn_lookup_nr":     dict(c=OKABE["purple"], ls="--",  m="h", label="NR, $k$-d tree seed"),
}

OPT_STYLE = {
    "adamw":   dict(c=OKABE["blue"],   m="o", ls="-",  label="AdamW"),
    "muon":    dict(c=OKABE["vermil"], m="s", ls="--", label="Muon"),
    "shampoo": dict(c=OKABE["green"],  m="D", ls=":",  label="Shampoo"),
}

# The provenance of the third arm is settled by paper_assets.py, which reads the
# optimizer class recorded inside each checkpoint. Nothing is asserted here.
SHAMPOO_NOTE = ""


# ------------------------------------------------------------- kinematics ----
# Duplicated from extract_paper_data.py on purpose. This module is meant to be
# runnable on its own, from any directory, with nothing beside it but the
# PAPER_DATA folder, so it must not import the evaluation pipeline.

DH_TABLE = np.array([
    [6.5, 0.0, np.pi / 2],   # 1 Base
    [0.0, 11.0, 0.0],        # 2 Shoulder
    [0.0, 15.0, 0.0],        # 3 Elbow
    [0.0, 0.0, np.pi / 2],   # 4 Wrist pitch
    [0.0, 0.0, -np.pi / 2],  # 5 Wrist roll
    [18.0, 0.0, 0.0],        # 6 Wrist twist / end effector
], dtype=np.float64)


def fk_frames(theta: np.ndarray) -> np.ndarray:
    """Cumulative transforms T_0^0 .. T_0^6, shape (N, 7, 4, 4)."""
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
    """End-effector position in cm, shape (N, 3)."""
    return fk_frames(theta)[:, 6, :3, 3]


def jacobian_batch(theta: np.ndarray) -> np.ndarray:
    """Analytic positional Jacobian, shape (N, 3, 6)."""
    T = fk_frames(theta)
    p_e = T[:, 6, :3, 3]
    z = T[:, :6, :3, 2]
    p = T[:, :6, :3, 3]
    return np.transpose(np.cross(z, p_e[:, None, :] - p), (0, 2, 1))


def manipulability(theta: np.ndarray) -> np.ndarray:
    """Yoshikawa positional manipulability, shape (N,)."""
    J = jacobian_batch(theta)
    det = np.linalg.det(J @ np.transpose(J, (0, 2, 1)))
    return np.sqrt(np.clip(det, 0.0, None))


# --------------------------------------------------------------- plotting ----
def setup(fontsize: float = 8.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,          # embed TrueType, required by IEEE
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": fontsize,
        "axes.titlesize": fontsize + 0.5,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.borderaxespad": 0.3,
        "axes.grid": True,
        "grid.color": "#BBBBBB",
        "grid.alpha": 0.45,
        "grid.linewidth": 0.35,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.3,
        "lines.markersize": 3.4,
        "errorbar.capsize": 2,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
    })
    return plt


def save(fig, name: str, formats: Sequence[str] = ("pdf", "png")) -> None:
    out = PATHS["fig_dir"]
    os.makedirs(out, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  wrote {os.path.join(out, name)}." + "|".join(formats))


def panel_label(ax, text: str, dx: float = -0.16, dy: float = 1.06) -> None:
    """IEEE wants multi-part figures labelled (a), (b), (c)."""
    ax.text(dx, dy, text, transform=ax.transAxes, fontweight="bold",
            va="top", ha="left", fontsize=9)


def style_for(method: str) -> Dict[str, Any]:
    return STYLE.get(method, dict(c=OKABE["grey"], ls="-", m=".", label=method))


# ------------------------------------------------------------------ data -----
def _load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Results:
    """Thin accessor over the PAPER_DATA artefacts."""

    def __init__(self, out_dir: Optional[str] = None) -> None:
        out_dir = out_dir or PATHS["out_dir"]
        self.dir = out_dir
        self.models: List[Dict] = self._maybe(os.path.join(out_dir, "per_model_full.json"), [])
        self.baselines: List[Dict] = self._maybe(os.path.join(out_dir, "baselines.json"), [])
        self.stats: Dict = self._maybe(os.path.join(out_dir, "statistics.json"), {})
        self.trajectory: Dict = self._maybe(os.path.join(out_dir, "trajectory.json"), {})
        self.audit: Dict = self._maybe(os.path.join(out_dir, "audit.json"), {})
        self.history: Dict = self._maybe(os.path.join(out_dir, "training_history.json"), {})
        self.all = self.models + self.baselines

    @staticmethod
    def _maybe(path: str, default: Any) -> Any:
        try:
            return _load(path)
        except Exception:
            print(f"  [warn] missing {path}")
            return default

    def rows(self, method: str) -> List[Dict]:
        return [r for r in self.all if r.get("method") == method]

    def best(self, method: str) -> Optional[Dict]:
        rs = self.rows(method)
        if not rs:
            return None
        return min(rs, key=lambda r: r["task_space"]["position_error_mm"]["mean"])

    def mean_err(self, method: str, opt: Optional[str] = None) -> np.ndarray:
        """Per-seed mean position error in mm."""
        rs = [r for r in self.rows(method)
              if opt is None or r.get("optimizer") == opt]
        return np.array([r["task_space"]["position_error_mm"]["mean"] for r in rs])

    def errors_mm(self, method: str, prefer_seed: Optional[int] = None) -> Optional[np.ndarray]:
        """Per-sample error vector from PAPER_DATA/per_sample, if it was kept."""
        # An anchored pattern, not a glob. `err_mlp_nr_*` also matches
        # `err_mlp_nr_iter1_*`, which silently plotted the one-iteration variant
        # under the full-refinement label in earlier versions of this figure.
        raw = PATHS["raw_dir"]
        tag = r"(?:mlp|mdn|baseline)_(?:adamw|muon|shampoo|none)_seed\d+_run\d+"
        rx = re.compile(rf"^err_{re.escape(method)}_(?:{tag})\.npz$")
        files: List[str] = [
            f for f in sorted(glob.glob(os.path.join(raw, "*.npz")))
            if rx.match(os.path.basename(f))]
        if not files:
            solo = os.path.join(raw, f"err_{method}.npz")
            files = [solo] if os.path.exists(solo) else []
        if not files:
            return None
        if prefer_seed is not None:
            hit = [f for f in files if f"seed{prefer_seed}_" in f]
            if hit:
                files = hit
        return np.load(files[0])["error_mm"]

    def success_curve(self, method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Tolerance sweep. Uses per-sample errors when available, otherwise the
        eight stored thresholds, so the figure still builds from JSON alone."""
        e = self.errors_mm(method)
        if e is not None and e.size:
            tol = np.logspace(-2.2, 2.4, 400)
            return tol, np.array([(e <= t).mean() * 100 for t in tol])
        r = self.best(method)
        if r is None:
            return np.array([]), np.array([])
        sr = r["task_space"]["success_rate_pct"]
        tol = np.array(TAUS_MM)
        return tol, np.array([sr[f"tau_{t}mm"] for t in tol])


# =============================================================================
# FIGURE 1  --  success rate versus tolerance
# =============================================================================

def fig1_success_vs_tolerance(R: Results, plt) -> None:
    """Claim: at any tolerance a user might specify, the methods are separated
    by orders of magnitude, and the pure MLP never becomes usable.

    The conference figure plotted a CDF of the error, which invites the reader
    to look at where each curve saturates rather than where it rises. Flipping
    the roles of the axes -- tolerance on x, solved fraction on y -- turns the
    plot into the question an engineer actually asks: if I need 1 mm, what
    fraction of targets do I get?
    """
    methods = ["mlp", "mdn_argmax_pi", "mdn_fk_best", "mlp_nr_iter1",
               "mlp_nr_iter3", "mlp_nr", "nr_midrange_nr"]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.45),
                             gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.52))
    ax, axb = axes

    for m in methods:
        tol, pct = R.success_curve(m)
        if tol.size == 0:
            continue
        s = style_for(m)
        marker_every = max(1, len(tol) // 7) if len(tol) > 20 else 1
        ax.plot(tol, pct, color=s["c"], ls=s["ls"], marker=s["m"],
                markevery=marker_every, markersize=3.0, label=s["label"])

    for t, txt in ((1.0, "1 mm"), (5.0, "5 mm")):
        ax.axvline(t, color="0.35", lw=0.6, ls=(0, (1, 2)), zorder=0)
        ax.annotate(txt, xy=(t, 46), xytext=(t * 1.18, 46), fontsize=6.3,
                    color="0.35", rotation=90, va="center")

    ax.set_xscale("log")
    ax.set_xlim(1e-2, 2e2)
    ax.set_ylim(-2, 103)
    ax.set_xlabel("Position tolerance $\\tau$ (mm)")
    ax.set_ylabel("Targets solved with $d \\leq \\tau$ (%)")
    ax.legend(loc="upper left", ncol=1, fontsize=6.3)
    panel_label(ax, "(a)", dx=-0.13)

    # (b) the single number a reader will quote: success at 1 mm, per method
    labels, vals, cols, hatches = [], [], [], []
    hatch_cycle = ["", "//", "..", "", "//", "", ".."]
    for m, h in zip(methods, hatch_cycle):
        r = R.best(m)
        if r is None:
            continue
        s = style_for(m)
        labels.append(s["label"])
        vals.append(r["task_space"]["success_rate_pct"]["tau_1.0mm"])
        cols.append(s["c"])
        hatches.append(h)
    y = np.arange(len(labels))
    bars = axb.barh(y, vals, color=cols, edgecolor="black", linewidth=0.5, height=0.68)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    for yi, v in zip(y, vals):
        axb.text(min(v + 2, 96), yi, f"{v:.2f}", va="center", fontsize=6.3)
    axb.set_yticks(y)
    axb.set_yticklabels(labels, fontsize=6.3)
    axb.invert_yaxis()
    axb.set_xlim(0, 108)
    axb.set_xlabel("Targets solved with $d \\leq 1$ mm (%)")
    axb.grid(axis="y", visible=False)
    panel_label(axb, "(b)", dx=-0.48, dy=1.10)

    save(fig, "fig1_success_vs_tolerance")


# =============================================================================
# FIGURE 2  --  mixture selection
# =============================================================================

def fig2_mdn_selection(R: Results, plt) -> None:
    """Claim: the MDN already contains a good solution for almost every target.
    The rule used to pick one of its components is what throws the accuracy
    away, and replacing that rule costs K forward-kinematics evaluations.

    Panel (a) is a schematic, because the result is impossible to argue about
    without first fixing what a "selection rule" is. For one target the network
    emits K = 20 candidate joint vectors. Forward kinematics maps each of them
    to a point in the workspace, and those points sit at different distances
    from the requested target. A selection rule is nothing more than a choice of
    which candidate to return, and the three rules compared here disagree.

    Panel (b) shows the consequence on all ten seeds, panel (c) shows why the
    joint-space rule fails: on a redundant arm, the candidate whose joint angles
    are closest to the recorded ground truth is not the candidate whose tool
    position is closest to the target. Those are different objectives, and the
    conference version optimised the wrong one.
    """
    rules = ["mdn_argmax_pi", "mdn_oracle_joint", "mdn_fk_best"]
    present = [m for m in rules if R.rows(m)]
    if not present:
        print("  [skip] fig2: no MDN rows")
        return

    fig = plt.figure(figsize=(COL2, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 1.05], wspace=0.40)
    axs = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[0, 2])

    # ---- (a) what a selection rule is, drawn ---------------------------------
    rng = np.random.default_rng(7)
    K = 20
    ang = rng.uniform(0, 2 * np.pi, K)
    rad = np.concatenate([rng.uniform(0.26, 0.48, 4), rng.uniform(0.55, 1.0, K - 4)])
    cx, cy = rad * np.cos(ang), rad * np.sin(ang)
    weight = rng.dirichlet(np.full(K, 3.0))

    for r in (0.33, 0.66, 1.0):
        axs.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.4,
                                 color="0.75", ls=(0, (2, 2)), zorder=0))
    axs.scatter(cx, cy, s=12 + 900 * weight, facecolor="0.80",
                edgecolor="0.35", linewidth=0.4, zorder=2)
    axs.scatter([0], [0], marker="*", s=95, color=OKABE["black"], zorder=5)
    axs.annotate("requested\nposition", xy=(0, 0), xytext=(-14, 14),
                 textcoords="offset points", fontsize=5.8, ha="right",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45"))

    i_fk = int(np.argmin(rad))                       # closest to the target
    i_pi = int(np.argmax(weight))                    # heaviest component
    far = np.argsort(-rad)
    i_oj = int([k for k in far if k not in (i_fk, i_pi)][0])

    for idx, col, lbl, off in ((i_pi, OKABE["orange"], "heaviest weight", (11, -3)),
                               (i_oj, OKABE["purple"], "joint-space oracle", (11, -10)),
                               (i_fk, OKABE["green"], "FK selection", (11, 6))):
        axs.scatter([cx[idx]], [cy[idx]], s=12 + 900 * weight[idx],
                    facecolor=col, edgecolor="black", linewidth=0.6, zorder=4)
        axs.plot([0, cx[idx]], [0, cy[idx]], color=col, lw=0.8, zorder=1)
        axs.annotate(lbl, xy=(cx[idx], cy[idx]), xytext=off,
                     textcoords="offset points", fontsize=5.8, color=col,
                     fontweight="bold")
    axs.set_xlim(-1.30, 2.30)
    axs.set_ylim(-1.60, 1.30)
    axs.set_aspect("equal")
    axs.axis("off")
    axs.set_title("schematic: the $K=20$ candidates\nreached by forward kinematics",
                  fontsize=6.3, pad=6)
    axs.annotate("marker area $\\propto$ mixture weight $\\pi_k$",
                 xy=(0.5, 0.015), xycoords="axes fraction", ha="center",
                 fontsize=5.6, color="0.35")
    panel_label(axs, "(a)", dx=-0.05, dy=1.14)

    # ---- (b) consequence, paired over the ten seeds --------------------------
    x = np.arange(len(present), dtype=float)
    per_seed: Dict[Any, List[float]] = {}
    for m in present:
        for r in R.rows(m):
            per_seed.setdefault((r["optimizer"], r["seed"]), []).append(
                r["task_space"]["position_error_mm"]["mean"])
    for vals in per_seed.values():
        if len(vals) == len(present):
            ax.plot(x, vals, color="0.75", lw=0.45, zorder=1)
    for i, m in enumerate(present):
        v = R.mean_err(m)
        st = style_for(m)
        ax.scatter(np.full(v.size, x[i]), v, s=10, color=st["c"],
                   edgecolor="black", linewidth=0.3, zorder=3)
        ax.plot([x[i] - 0.22, x[i] + 0.22], [v.mean()] * 2, color="black",
                lw=1.5, zorder=4)
        ax.annotate(f"{v.mean():.2f}", xy=(x[i] + 0.24, v.mean()), xytext=(2, 0),
                    textcoords="offset points", fontsize=6.4, ha="left",
                    va="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\arg\max\,\pi_k$", "joint\noracle",
                        "FK\nselect."], fontsize=6.4)
    ax.set_xlim(-0.45, len(present) - 0.10)
    ax.set_ylim(0, max(R.mean_err(m).max() for m in present) * 1.22)
    ax.set_ylabel("Mean position error (mm)")
    ax.set_title("same network, three rules", fontsize=6.6, pad=4)
    panel_label(ax, "(b)", dx=-0.34, dy=1.14)

    # ---- (c) the two objectives disagree ------------------------------------
    for m in present:
        st = style_for(m)
        xs = [r["joint_space"]["mean_mae_deg_learnable_only"] for r in R.rows(m)]
        ys = [r["task_space"]["position_error_mm"]["mean"] for r in R.rows(m)]
        axb.scatter(xs, ys, s=16, color=st["c"], marker=st["m"],
                    edgecolor="black", linewidth=0.35, label=st["label"])
    axb.set_xlabel("Joint-space MAE, joints 1-5 (deg)", fontsize=6.8)
    axb.set_ylabel("Mean position error (mm)")
    axb.legend(loc="lower left", fontsize=5.8)
    oj = R.rows("mdn_oracle_joint")
    if oj:
        axb.annotate("closest in joint space,\nfarthest in task space",
                     xy=(np.mean([r["joint_space"]["mean_mae_deg_learnable_only"] for r in oj]),
                         np.mean([r["task_space"]["position_error_mm"]["mean"] for r in oj])),
                     xytext=(16, 22), textcoords="offset points", fontsize=5.8,
                     color="0.25",
                     arrowprops=dict(arrowstyle="->", lw=0.6, color="0.45"))
    axb.set_ylim(0, None)
    panel_label(axb, "(c)", dx=-0.30, dy=1.14)

    save(fig, "fig2_mdn_selection")


# =============================================================================
# FIGURE 3  --  what the seed buys
# =============================================================================

def fig3_seed_quality(R: Results, plt) -> None:
    """Claim: every seeding strategy converges to the tolerance, so the network
    is not buying accuracy. It is buying iterations, and that is the number to
    put in the abstract.

    The earlier version of this figure carried two identically labelled curves
    because it picked the first two rows with a stored history, and its y-axis
    was dominated by the tolerance floor. Here each curve is labelled by its
    seed, the tolerance is drawn as a horizontal rule so the floor is explained
    rather than mysterious, and panel (b) converts the curves into the budget a
    practitioner sets.
    """
    wanted = [
        ("mdn_nr", "MDN seed"),
        ("mlp_nr", "MLP seed"),
        ("nn_lookup_nr", "$k$-d tree seed"),
        ("nr_midrange_nr", "mid-range seed"),
        ("nr_random_in_limits_nr", "random seed, in limits"),
        ("nr_random_legacy_nr", "random seed, outside limits"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5),
                             gridspec_kw=dict(width_ratios=[1.30, 1.0], wspace=0.60))
    ax, axb = axes

    tol_mm = 0.01
    bar_rows = []
    for method, lbl in wanted:
        r = R.best(method)
        if r is None or "nr" not in r:
            continue
        curve = r["nr"].get("error_vs_iteration_mm")
        if not curve:
            continue
        s = style_for(method)
        med = np.asarray(curve["median"], dtype=float)
        it = np.arange(len(med))
        ax.plot(it, np.maximum(med, 1e-5), color=s["c"], ls=s["ls"],
                marker=s["m"], markersize=3.2, label=lbl)
        if "p95" in curve and method in ("mdn_nr", "mlp_nr"):
            p95 = np.maximum(np.asarray(curve["p95"], dtype=float), 1e-5)
            ax.fill_between(it, np.maximum(med, 1e-5), p95,
                            color=s["c"], alpha=0.13, linewidth=0)
        # panel (b) reports the operating cost, so the deliberately broken
        # seeding used in the course draft is shown only in panel (a), where it
        # illustrates the failure mode, and is excluded from the budget bars
        if method != "nr_random_legacy_nr":
            bar_rows.append((lbl, r["nr"].get("mean_iterations"),
                             r["nr"].get("convergence_rate_pct"), s["c"], s["m"]))

    ax.axhline(tol_mm, color="0.3", lw=0.7, ls=(0, (4, 2)))
    ax.annotate("solver tolerance", xy=(4.4, tol_mm * 1.35),
                fontsize=6.2, color="0.3")
    ax.axhline(1.0, color="0.55", lw=0.6, ls=(0, (1, 2)))
    ax.annotate("1 mm", xy=(7.3, 1.3), fontsize=6.2, color="0.45")
    ax.set_yscale("log")
    ax.set_ylim(tol_mm / 4, 1.2e3)
    ax.set_xlabel("Newton\u2013Raphson iteration")
    ax.set_ylabel("Median position error (mm)")
    ax.set_xlim(-0.2, 8.2)
    ax.legend(loc="lower left", fontsize=6.0, ncol=1)
    panel_label(ax, "(a)", dx=-0.15, dy=1.10)

    # ---- (b) mean iterations, annotated with convergence rate --------------
    bar_rows = [b for b in bar_rows if b[1] is not None]
    y = np.arange(len(bar_rows))
    vals = [b[1] for b in bar_rows]
    cols = [b[3] for b in bar_rows]
    bars = axb.barh(y, vals, color=cols, edgecolor="black", linewidth=0.5, height=0.66)
    for h, b in zip(["", "//", "..", "", "//", ".."], bars):
        b.set_hatch(h)
    for yi, (lbl, it, conv, _, _) in zip(y, bar_rows):
        axb.text(it + 0.12, yi, f"{it:.2f}" + (f"  ({conv:.1f}%)" if conv else ""),
                 va="center", fontsize=6.2)
    axb.set_yticks(y)
    axb.set_yticklabels([b[0] for b in bar_rows], fontsize=6.3)
    axb.invert_yaxis()
    axb.set_xlabel("Mean iterations to tolerance")
    axb.set_title("cost of one query, and the share that converges",
                  fontsize=6.3, pad=4)
    axb.set_xlim(0, max(vals) * 1.55)
    axb.grid(axis="y", visible=False)
    panel_label(axb, "(b)", dx=-0.55, dy=1.10)

    fig.text(0.5, -0.16,
             "Seeds: MDN and MLP, the network output; $k$-d tree, the joint "
             "vector of the nearest training sample in task space; mid-range, "
             "the centre of every joint interval.",
             ha="center", fontsize=5.9, color="0.30")
    save(fig, "fig3_seed_quality")


# =============================================================================
# FIGURE 4  --  optimizer comparison
# =============================================================================

def fig4_optimizers(R: Results, plt) -> None:
    """Claim: Muon is reliably but negligibly better than AdamW, the Shampoo arm
    is not Shampoo, and the random seed matters more than the optimizer.

    Three facts have to be visible at once for this to be an honest figure: the
    per-seed consistency that produces the small p-value, the size of the effect
    against a margin of practical relevance, and how little of the spread the
    optimizer explains. Panel (a) carries the first and the third; panel (b)
    carries the second, with the equivalence margin drawn as a shaded band so
    that "significant" and "negligible" can be read off the same axis.
    """
    from matplotlib.patches import Rectangle

    # statistics recomputed on the kept arms, if paper_assets.py has run
    ost = R._maybe(os.path.join(PATHS["out_dir"], "optimizer_stats.json"), {})

    fig = plt.figure(figsize=(COL2, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.55], wspace=0.30)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # ---- (a) the three optimizers on an axis that starts at zero ------------
    # A paired plot zoomed to the 22.8-23.8 mm window makes a 0.34 mm gap look
    # decisive. Drawn against the full magnitude of the error, the same three
    # numbers are visually indistinguishable, which is the honest impression and
    # the one the equivalence test in panel (b) confirms.
    x = np.arange(len(OPTIMIZERS), dtype=float)
    for i, o in enumerate(OPTIMIZERS):
        v = R.mean_err("mlp", o)
        if v.size == 0:
            continue
        st = OPT_STYLE[o]
        axa.bar(x[i], v.mean(), width=0.62, color=st["c"], edgecolor="black",
                linewidth=0.5, yerr=v.std(ddof=1), capsize=3, zorder=2)
        axa.scatter(np.full(v.size, x[i]), v, s=7, color="black", alpha=0.55,
                    zorder=4)
        axa.annotate(f"{v.mean():.2f}", xy=(x[i], v.mean() * 0.90),
                     ha="center", va="top", fontsize=6.4, color="white",
                     fontweight="bold")
    axa.set_xticks(x)
    axa.set_xticklabels([OPT_STYLE[o]["label"] for o in OPTIMIZERS], fontsize=7)
    axa.set_xlim(-0.6, len(OPTIMIZERS) - 0.4)
    axa.set_ylim(0, None)
    axa.set_ylabel("MLP mean position error (mm)")
    axa.set_title("full scale, error bars over 10 seeds", fontsize=6.6, pad=4)

    # the variance attribution, stated where it is relevant rather than as a
    # panel of its own: the figure should not give the optimizer more visual
    # weight than the analysis gives it
    _sv = ost.get("variance_share") or R.stats.get("seed_variability", {})
    share = _sv.get("mlp", {}).get(
        "optimizer_pct", _sv.get("mlp", {}).get(
            "variance_explained_by_optimizer_pct"))
    if share is not None:
        axa.annotate(f"the optimizer explains {share:.0f}% of the spread;"
                     f"\nthe random seed explains the rest"
                     if share is not None else "",
                     xy=(0.5, -0.30), xycoords="axes fraction", ha="center",
                     fontsize=6.0, color="0.30", va="top")

    # ---- (b) forest plot with the equivalence margin ------------------------
    pool = ost.get("pairwise") or R.stats.get("pairwise", [])
    pw = [p for p in pool
          if p["method"] in ("mlp", "mdn_fk_best", "mlp_nr")
          and all(o in OPTIMIZERS for o in p["comparison"].split(" vs "))]
    order = {"mlp": 0, "mdn_fk_best": 1, "mlp_nr": 2}
    pw.sort(key=lambda p: (order.get(p["method"], 9), p["comparison"]))
    if pw:
        margin = 0.5   # mm, the TOST margin used in the statistics stage
        lo_all = min(p["delta_ci95_mm"][0] for p in pw)
        hi_all = max(p["delta_ci95_mm"][1] for p in pw)
        span = max(abs(lo_all), abs(hi_all), margin) * 1.15
        left = -span * 2.35          # room for the labels inside the axes

        axb.add_patch(Rectangle((-margin, -1), 2 * margin, len(pw) + 2,
                                facecolor="0.90", edgecolor="none", zorder=0))
        axb.axvline(0, color="black", lw=0.7, zorder=1)
        axb.axvline(-margin, color="0.55", lw=0.5, ls=(0, (2, 2)), zorder=1)
        axb.axvline(margin, color="0.55", lw=0.5, ls=(0, (2, 2)), zorder=1)

        for i, p in enumerate(pw):
            lo, hi = p["delta_ci95_mm"]
            sig = p.get("significant_holm", False)
            eq = p.get("tost", {}).get("equivalent", False)
            axb.plot([lo, hi], [i, i], color="black", lw=1.0, zorder=3)
            axb.scatter([p["delta_mm"]], [i], s=24, zorder=4,
                        marker="D" if sig else "o",
                        color=OKABE["vermil"] if sig else "white",
                        edgecolor="black", linewidth=0.6)
            lbl = (f"{p['method'].replace('_', ' ')}: "
                   f"{p['comparison']}")
            axb.text(left * 0.98, i, lbl, va="center", ha="left", fontsize=6.0)
            verdict = ("$p_{\\mathrm{Holm}}$ < 0.01" if p["p_holm"] < 0.01
                       else f"$p$ = {p['p_holm']:.2f}")
            if eq:
                verdict += ", equivalent"
            axb.text(span * 1.02, i, verdict, va="center", fontsize=5.9,
                     color="0.25")

        axb.set_yticks([])
        axb.set_ylim(len(pw) - 0.4, -0.9)
        axb.set_xlim(left, span * 2.30)
        axb.spines["left"].set_visible(False)
        axb.set_xticks(np.round(np.linspace(-span, span, 5), 2))
        axb.set_xlabel("Difference in mean position error (mm)")
        axb.set_title("effect size with 95% CI; shaded band is the "
                      "equivalence margin", fontsize=6.6, pad=4)
        axb.grid(axis="y", visible=False)
    panel_label(axb, "(b)", dx=-0.02, dy=1.11)

    if SHAMPOO_NOTE:
        fig.text(0.5, -0.07, SHAMPOO_NOTE, ha="center", fontsize=6, color="0.35")
    save(fig, "fig4_optimizers")


# =============================================================================
# FIGURE 5  --  spatial structure of the error
# =============================================================================

def fig5_spatial(R: Results, plt) -> None:
    """Claim: the MLP error is not uniform noise. It is organised by the
    kinematics and grows where the arm is least dexterous.

    Rebuilt with scipy.stats.binned_statistic_2d instead of hexbin. The previous
    version placed one colourbar per axes, so three colourbars and their labels
    collided with the axis labels of their neighbours. Here the two projections
    share a single horizontal colourbar on a logarithmic scale, which also makes
    them directly comparable, and cells with too few samples are left blank
    rather than shown as a noisy median of two points.
    """
    from scipy import stats as sstats
    from matplotlib.colors import LogNorm

    try:
        import joblib

        d = np.load(PATHS["data_npz"])
        scaler = joblib.load(PATHS["scaler"])
        P = scaler.inverse_transform(d["X_test"])
        y_test = d["y_test"]
    except Exception as exc:
        print(f"  [skip] fig5: could not load {PATHS['data_npz']} / "
              f"{PATHS['scaler']} ({exc})")
        print("         pass --data-npz and --scaler if they live elsewhere")
        return

    e = R.errors_mm("mlp")
    if e is None:
        print("  [skip] fig5: per-sample errors for the MLP were not kept")
        return
    n = min(len(e), len(P))
    P, e, y_test = P[:n], e[:n], y_test[:n]

    # bin count and occupancy floor scale with the sample size, so the figure
    # is neither blocky on 500k points nor speckled on a subsample
    nbins = int(np.clip(np.sqrt(n) / 4.5, 22, 60))
    min_count = max(4, int(n / (nbins * nbins * 12)))

    fig = plt.figure(figsize=(COL2, 2.85))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.30],
                          height_ratios=[1, 0.06],
                          wspace=0.40, hspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, 0:2])
    ax2 = fig.add_subplot(gs[:, 2])

    vlo = max(np.percentile(e, 2), 1e-2)
    vhi = np.percentile(e, 99)
    norm = LogNorm(vmin=vlo, vmax=vhi)

    pc = None
    for ax, (i, j), lab, tag in ((ax0, (0, 1), ("$x$", "$y$"), "top view ($xy$)"),
                                 (ax1, (0, 2), ("$x$", "$z$"), "side view ($xz$)")):
        H, xe, ye, _ = sstats.binned_statistic_2d(
            P[:, i], P[:, j], e, statistic="median", bins=nbins)
        C, _, _, _ = sstats.binned_statistic_2d(
            P[:, i], P[:, j], e, statistic="count", bins=[xe, ye])
        H = np.where(C >= min_count, H, np.nan)
        pc = ax.pcolormesh(xe, ye, H.T, norm=norm, cmap="viridis",
                           shading="auto", rasterized=True)
        ax.plot(0, 0, marker="o", markersize=4, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=0.8, zorder=5)
        ax.annotate("base", xy=(0, 0), xytext=(4, -9),
                    textcoords="offset points", fontsize=5.8, color="black")
        ax.set_xlabel(f"{lab[0]} (cm)")
        ax.set_ylabel(f"{lab[1]} (cm)")
        ax.set_title(tag, fontsize=7, pad=3)
        ax.set_aspect("equal")
        ax.grid(False)

    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label("Median MLP position error (mm)", fontsize=6.6, labelpad=1)
    cb.ax.tick_params(labelsize=6, pad=1)
    panel_label(ax0, "(a)", dx=-0.30, dy=1.16)
    panel_label(ax1, "(b)", dx=-0.30, dy=1.16)

    # ---- (c) error against manipulability -----------------------------------
    try:
        w = manipulability(y_test)
        edges = np.quantile(w, np.linspace(0, 1, 13))
        med, _, _ = sstats.binned_statistic(w, e, "median", bins=edges)
        p25, _, _ = sstats.binned_statistic(
            w, e, lambda v: np.percentile(v, 25), bins=edges)
        p75, _, _ = sstats.binned_statistic(
            w, e, lambda v: np.percentile(v, 75), bins=edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax2.fill_between(centres, p25, p75, color=OKABE["vermil"], alpha=0.18,
                         linewidth=0, label="interquartile range")
        ax2.plot(centres, med, color=OKABE["vermil"], marker="o", label="median")
        ax2.set_xlabel(r"Manipulability $w=\sqrt{\det(\mathbf{J}\mathbf{J}^{\top})}$",
                       fontsize=7)
        ax2.set_ylabel("MLP position error (mm)")
        ax2.set_xscale("log")
        ax2.set_ylim(0, float(np.nanmax(p75)) * 1.12)
        ax2.legend(fontsize=6.0, loc="upper right")
        ax2.set_title("least dexterous $\\rightarrow$ most dexterous",
                      fontsize=6.6, pad=3)
        panel_label(ax2, "(c)", dx=-0.24, dy=1.16)
    except Exception as exc:
        print(f"  [warn] fig5 panel (c) skipped: {exc}")

    save(fig, "fig5_spatial_error")


# =============================================================================
# FIGURE 6  --  path tracking: accuracy is not enough
# =============================================================================

def fig6_path_tracking(R: Results, plt) -> None:
    """Claim: per-waypoint accuracy is not sufficiency. A solver can place the
    tool within a millimetre of every point of a smooth circle and still be
    unusable, because it reaches consecutive points through configurations that
    are far apart in joint space.

    The setting is the one a manipulator is actually used for. A smooth
    Cartesian path is discretised into waypoints, the solver is queried at each
    of them independently, and the resulting joint vectors are what the
    controller would have to execute in sequence. Because a redundant arm has
    many configurations for the same point, an independent per-point solver is
    free to return one posture at waypoint i and a completely different posture
    at waypoint i+1. The tool position is correct at both, and the motion
    between them is not.

    Panel (a) shows this directly, as the commanded angle of the joint most
    affected: the MDN swings the base through about 100 degrees between two
    adjacent points a few millimetres apart, which a real actuator cannot do and
    a planner would reject. Panel (b) confirms that this is not an accuracy
    problem, and panel (c) places every method on the accuracy-continuity plane.
    """
    paths = R.trajectory.get("paths", {})
    if not paths:
        print("  [skip] fig6: no trajectory data")
        return
    pname = "circle_xy" if "circle_xy" in paths else next(iter(paths))
    entry = paths[pname]

    series = [
        ("mlp", "MLP", OKABE["vermil"], "-", "o"),
        ("mdn_fk_best", "MDN, FK sel.", OKABE["green"], "-.", "D"),
        ("mlp_nr", "MLP + NR", OKABE["blue"], "-", "P"),
        ("nr_warmstart_sequential", "NR, warm start", OKABE["grey"], "--", "X"),
    ]
    series = [s for s in series if s[0] in entry]
    if not series:
        print("  [skip] fig6: no comparable methods")
        return

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.45),
                             gridspec_kw=dict(width_ratios=[1.25, 1.10, 1.0],
                                              wspace=0.36))
    ax0, ax1, ax2 = axes

    # pick the joint that discriminates the methods most strongly, rather than
    # fixing one a priori
    def series_of(key: str) -> np.ndarray:
        return np.asarray(entry[key]["joint_series_deg"], dtype=float)

    jumps = np.zeros(6)
    for key, *_ in series:
        jumps = np.maximum(jumps, np.abs(np.diff(series_of(key), axis=0)).max(axis=0))
    j_idx = int(np.argmax(jumps))

    # ---- (a) the commanded angle of that joint along the path ---------------
    for key, lbl, c, ls, m in series:
        ax0.plot(series_of(key)[:, j_idx], color=c, ls=ls, lw=1.1, label=lbl)
    ax0.set_xlabel("Waypoint index")
    ax0.set_ylabel(f"Commanded {JOINT_NAMES[j_idx].lower()} angle (deg)", fontsize=7)
    ax0.set_title(f"{pname.replace('_', ' ')} path", fontsize=6.4, pad=4)

    worst = max(series, key=lambda t: np.abs(
        np.diff(series_of(t[0])[:, j_idx])).max())
    dif = np.abs(np.diff(series_of(worst[0])[:, j_idx]))
    k = int(np.argmax(dif))
    ax0.annotate(f"{dif[k]:.0f}$^\\circ$ between two\nadjacent waypoints",
                 xy=(k + 0.5, series_of(worst[0])[k:k + 2, j_idx].mean()),
                 xytext=(0.30, 0.06), textcoords="axes fraction",
                 fontsize=6.0, color=worst[2], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=worst[2]))
    panel_label(ax0, "(a)", dx=-0.24, dy=1.13)

    # ---- (b) accuracy is not the problem ------------------------------------
    floor = 1e-3
    for key, lbl, c, ls, m in series:
        y = np.asarray(entry[key]["error_series_mm"], dtype=float)
        ax1.plot(np.maximum(y, floor), color=c, ls=ls, lw=1.0)
    ax1.axhspan(floor / 2, 1.0, color="0.85", alpha=0.5, zorder=0)
    ax1.annotate("within 1 mm", xy=(0.03, 0.05), xycoords="axes fraction",
                 fontsize=6.0, color="0.35")
    ax1.set_yscale("log")
    ax1.set_ylim(floor / 2, None)
    ax1.set_xlabel("Waypoint index")
    ax1.set_ylabel("Tool position error (mm)")
    ax1.set_title("accuracy", fontsize=6.4, pad=4)
    panel_label(ax1, "(b)", dx=-0.28, dy=1.13)

    # ---- (c) the plane that matters -----------------------------------------
    offsets = {"mlp": (-8, 12), "mdn_fk_best": (-14, -18),
               "mlp_nr": (9, 5), "nr_warmstart_sequential": (9, -13)}
    for key, lbl, c, ls, m in series:
        v = entry[key]
        xx = max(v["tracking_error_mm"]["mean"], 1e-4)
        yy = max(v["joint_continuity"]["max_step_deg"], 1e-2)
        ax2.scatter(xx, yy, s=40, color=c, marker=m, edgecolor="black",
                    linewidth=0.45, zorder=3)
        ax2.annotate(lbl, xy=(xx, yy), xytext=offsets.get(key, (7, 6)),
                     textcoords="offset points", fontsize=6.0, color=c,
                     fontweight="bold")
    ax2.axhspan(10, 1e4, color=OKABE["vermil"], alpha=0.09, zorder=0)
    ax2.axvspan(1.0, 1e4, color=OKABE["vermil"], alpha=0.09, zorder=0)
    ax2.annotate("not executable", xy=(0.04, 0.96), xycoords="axes fraction",
                 fontsize=5.9, color=OKABE["vermil"], va="top")
    ax2.annotate("usable", xy=(0.05, 0.30), xycoords="axes fraction",
                 fontsize=6.0, color="0.35")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(0.08, 5e3)
    ax2.set_xlabel("Mean tracking error (mm)", fontsize=6.8)
    ax2.set_ylabel(r"Largest joint step $\|\Delta\boldsymbol{\theta}\|$ (deg)")
    ax2.set_title("the plane that matters", fontsize=6.4, pad=4)
    panel_label(ax2, "(c)", dx=-0.30, dy=1.13)

    handles = [plt.Line2D([], [], color=c, ls=ls, marker=m, markersize=4, lw=1.2,
                          label=lbl) for _, lbl, c, ls, m in series]
    fig.legend(handles=handles, loc="lower center", ncol=len(series),
               fontsize=6.4, bbox_to_anchor=(0.5, -0.11))
    save(fig, "fig6_path_tracking")


# =============================================================================
# FIGURE 7  --  the MDN recovers the structure of the problem
# =============================================================================

def fig7_mdn_structure(R: Results, plt) -> None:
    """Claim: the density model recovers the structure of the manipulator on its
    own, and it does not collapse onto a single mode.

    Panel (a) normalises the predicted standard deviation by the mean over the
    five joints that do move the tool, so the wrist twist is read as a ratio
    rather than as a bar that flattens the others. A value near one means the
    network is as confident about that joint as it is about the rest; the wrist
    twist sits at several times that, which is the network stating that the
    joint is not determined by the input. That is the correct answer, and it was
    never supplied as supervision.

    Panel (b) shows the same asymmetry from a different angle: the K components
    disagree with each other in the joints that move the tool and agree on the
    one that does not, which is what a model of the solution set should do.

    Panel (c) answers the question the oracle number was meant to answer and
    could not: is the mixture actually being used? Of the 20 components declared,
    about 11 carry meaningful weight, and about 7 land within 5 mm of the
    requested position. A collapsed mixture would show one and one.
    """
    r = R.best("mdn_fk_best") or R.best("mdn_argmax_pi")
    if r is None or "mdn" not in r:
        print("  [skip] fig7: no MDN diagnostics")
        return
    d = r["mdn"]

    fig = plt.figure(figsize=(COL2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.42)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))

    unobs = 5
    try:
        u = R.audit["kinematics_selftest"]["structurally_unobservable_joints"]
        unobs = u[0] if u else 5
    except Exception:
        pass
    obs = [i for i in range(6) if i != unobs]
    x = np.arange(6)
    cols = [OKABE["grey"]] * 6
    cols[unobs] = OKABE["vermil"]

    # ---- (a) predicted uncertainty, relative to the observable joints -------
    sig = np.asarray(d["predicted_sigma_deg_per_joint"], dtype=float)
    base = sig[obs].mean()
    rel = sig / base
    bars = ax0.bar(x, rel, color=cols, edgecolor="black", linewidth=0.5)
    bars[unobs].set_hatch("///")
    ax0.axhline(1.0, color="0.35", lw=0.7, ls=(0, (4, 2)))
    ax0.annotate("mean of the joints\nthat move the tool", xy=(0.02, 1.06),
                 xycoords=("axes fraction", "data"), fontsize=5.8, color="0.35")
    ax0.annotate(f"{rel[unobs]:.1f}$\\times$", xy=(unobs, rel[unobs]),
                 xytext=(0, 4), textcoords="offset points", ha="center",
                 fontsize=7, color=OKABE["vermil"], fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(JOINT_NAMES, rotation=38, ha="right", fontsize=6.2)
    ax0.set_ylabel(r"Predicted $\sigma_k$, relative to" "\n" "the observable joints")
    ax0.set_ylim(0, rel.max() * 1.22)
    ax0.set_title("the network's own uncertainty", fontsize=6.4, pad=4)
    panel_label(ax0, "(a)", dx=-0.34, dy=1.13)

    # ---- (b) where the components disagree ---------------------------------
    spread = np.asarray(d["component_mean_spread_deg_per_joint"], dtype=float)
    bars = ax1.bar(x, spread, color=cols, edgecolor="black", linewidth=0.5)
    bars[unobs].set_hatch("///")
    for xi, v in zip(x, spread):
        ax1.text(xi, v + spread.max() * 0.03, f"{v:.0f}", ha="center", fontsize=5.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(JOINT_NAMES, rotation=38, ha="right", fontsize=6.2)
    ax1.set_ylabel(r"Disagreement between the $K$" "\n" r"components (deg)")
    ax1.set_ylim(0, spread.max() * 1.24)
    ax1.set_title("where the candidates differ", fontsize=6.4, pad=4)
    panel_label(ax1, "(b)", dx=-0.34, dy=1.13)

    # ---- (c) is the mixture used at all? -----------------------------------
    K = float(d["n_mixtures_declared"])
    eff = float(d["effective_n_components_mean"])
    near = d.get("n_components_within_5.0mm")
    rows = [("components the model declares ($K$)", K, "0.85", ".."),
            ("components carrying real weight\n" r"(perplexity $e^{H(\pi)}$)", eff,
             OKABE["blue"], ""),
            ]
    if near is not None:
        rows.append(("components landing within 5 mm\nof the requested position",
                     float(near), OKABE["green"], ""))
    y = np.arange(len(rows))
    b = ax2.barh(y, [v for _, v, _, _ in rows],
                 color=[c for _, _, c, _ in rows],
                 edgecolor="black", linewidth=0.5, height=0.6)
    for bar, (_, _, _, h) in zip(b, rows):
        if h:
            bar.set_hatch(h)
    for yi, (_, v, _, _) in zip(y, rows):
        ax2.text(v + K * 0.02, yi, f"{v:.1f}", va="center", fontsize=6.5,
                 fontweight="bold")
    # labels sit inside the axes rather than as tick labels, which on a
    # three-panel row would otherwise overrun the neighbouring plot
    for yi, (lbl, _, _, _) in zip(y, rows):
        ax2.text(K * 0.02, yi - 0.42, lbl, va="bottom", ha="left", fontsize=5.9)
    ax2.set_yticks([])
    ax2.set_ylim(len(rows) - 0.35, -0.85)
    ax2.set_xlim(0, K * 1.20)
    ax2.set_xlabel("Number of mixture components")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("a collapsed mixture would read 1 and 1", fontsize=6.4, pad=4)
    panel_label(ax2, "(c)", dx=-0.14, dy=1.13)

    save(fig, "fig7_mdn_structure")


# =============================================================================
# FIGURE 8  --  the operating plane
# =============================================================================

def fig8_pareto(R: Results, plt) -> None:
    """Claim: the hybrid dominates in the region that matters, and the cost of
    each additional millimetre of accuracy is visible.

    Marker area encodes the refinement budget, so accuracy, latency and
    iteration count appear together. A staircase marks the non-dominated set so
    the reader does not have to infer it.
    """
    methods = ["mlp", "mdn_argmax_pi", "mdn_fk_best", "mlp_nr_iter1",
               "mlp_nr_iter2", "mlp_nr_iter3", "mlp_nr", "mdn_nr",
               "nr_midrange_nr", "nn_lookup_nr"]

    pts = []
    for m in methods:
        rs = R.rows(m)
        if not rs:
            continue
        err = float(np.mean([r["task_space"]["position_error_mm"]["mean"] for r in rs]))
        lat = None
        for r in rs:
            t = r.get("timing", {})
            lat = (t.get("numpy_single_ms", {}).get("mean")
                   or t.get("call_single_ms", {}).get("mean"))
            if lat:
                break
        nrt = rs[0].get("nr", {}).get("timing", {}).get("single_ms", {}).get("mean")
        if lat is None and nrt is None:
            continue
        total = (lat or 0.0) + (nrt or 0.0)
        iters = rs[0].get("nr", {}).get("mean_iterations") or 0.0
        pts.append((m, total, err, iters))
    if not pts:
        print("  [skip] fig8: no timing data")
        return

    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    for m, lat, err, iters in pts:
        s = style_for(m)
        ax.scatter(lat, err, s=18 + 26 * iters, color=s["c"], marker=s["m"],
                   edgecolor="black", linewidth=0.45, alpha=0.9, zorder=3)
        ax.annotate(s["label"], xy=(lat, err), xytext=(4, 4),
                    textcoords="offset points", fontsize=5.8)

    # non-dominated staircase: cheaper and more accurate than everything left of it
    order = sorted(pts, key=lambda p: p[1])
    front, best = [], np.inf
    for m, lat, err, _ in order:
        if err < best:
            front.append((lat, err))
            best = err
    if len(front) > 1:
        fx = [p[0] for p in front]
        fy = [p[1] for p in front]
        ax.step(fx, fy, where="post", color="0.45", lw=0.8, ls=(0, (4, 2)),
                zorder=1, label="non-dominated set")
        ax.legend(fontsize=6, loc="upper right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latency per query (ms)")
    ax.set_ylabel("Mean position error (mm)")
    ax.annotate("marker area $\\propto$ NR iterations",
                xy=(0.02, 0.03), xycoords="axes fraction", fontsize=5.8, color="0.35")
    save(fig, "fig8_pareto")


# =============================================================================
# FIGURE 9  --  training dynamics  (kept, tightened)
# =============================================================================

def fig9_training(R: Results, plt) -> None:
    """Claim: Muon reaches the same MLP optimum in roughly a third fewer epochs,
    but pays for it in wall-clock time and does not help the MDN.

    The previous version of this figure was already sound. What it lacked was
    the marker for the epoch actually selected by early stopping, and the
    wall-clock panel that turns the epoch saving into a cost the reader can act
    on.
    """
    curves = R.history.get("curves", {})
    summary = R.history.get("summary", {})
    if not curves:
        print("  [skip] fig9: no training history")
        return

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3),
                             gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.85],
                                              wspace=0.40))
    for ax, arch, ttl, ylab in ((axes[0], "mlp", "MLP", "Validation MSE"),
                                (axes[1], "mdn", "MDN", "Validation NLL")):
        for o in OPTIMIZERS:
            k = f"{arch}_{o}"
            if k not in curves:
                continue
            m = np.asarray(curves[k]["mean"], dtype=float)
            s = np.asarray(curves[k]["std"], dtype=float)
            xx = np.arange(len(m))
            st = OPT_STYLE[o]
            ax.plot(xx, m, color=st["c"], ls=st["ls"],
                    label=f"{st['label']} ($n$={curves[k]['n_runs']})")
            ax.fill_between(xx, m - s, m + s, color=st["c"], alpha=0.15, linewidth=0)
            be = summary.get(k, {}).get("best_epoch", {}).get("mean")
            if be is not None and be < len(m):
                ax.plot([be], [m[int(be)]], marker=st["m"], color=st["c"],
                        markersize=5, markeredgecolor="black", markeredgewidth=0.5,
                        zorder=5, linestyle="none")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=7, pad=3)
        ax.legend(fontsize=6.0)
    axes[0].annotate("marker: mean epoch selected\nby early stopping",
                     xy=(0.30, 0.72), xycoords="axes fraction", fontsize=5.8,
                     color="0.35")
    panel_label(axes[0], "(a)", dx=-0.28)
    panel_label(axes[1], "(b)", dx=-0.28)

    # ---- (c) the cost of those epochs ---------------------------------------
    ax = axes[2]
    # panel (c) must respect the same exclusion as panels (a) and (b)
    keys = [k for k in summary
            if not k.startswith("_") and k.split("_", 1)[1] in OPTIMIZERS]
    keys.sort()
    y = np.arange(len(keys))
    mins = [summary[k]["train_time_sec"]["mean"] / 60 for k in keys]
    cols = [OPT_STYLE[k.split("_", 1)[1]]["c"] for k in keys]
    bars = ax.barh(y, mins, color=cols, edgecolor="black", linewidth=0.5, height=0.66)
    for b, k in zip(bars, keys):
        if k.startswith("mdn"):
            b.set_hatch("///")
    for yi, v in zip(y, mins):
        ax.text(v + 1.5, yi, f"{v:.0f}", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels([k.replace("_", "/")
                        for k in keys], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Training time (min)")
    ax.set_xlim(0, max(mins) * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_title("hatched: MDN", fontsize=6.6, pad=3)
    panel_label(ax, "(c)", dx=-0.60)

    save(fig, "fig9_training")


# =============================================================================
# GREYSCALE PROOF
# =============================================================================

def grayscale_check() -> None:
    """Re-render every PNG in luminance so the IEEE greyscale test can be seen.

    IEEE asks that a figure remain interpretable on a monochrome printout. This
    writes a `grayscale/` folder next to the figures; if two series become
    indistinguishable there, the line style or marker has to change, not the
    colour.
    """
    try:
        from PIL import Image
    except Exception:
        print("  [skip] greyscale check needs Pillow")
        return
    dst = os.path.join(PATHS["fig_dir"], "grayscale")
    os.makedirs(dst, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(PATHS["fig_dir"], "*.png"))):
        Image.open(p).convert("L").save(
            os.path.join(dst, os.path.basename(p)))
    print(f"  wrote {dst}/  -- inspect these before submitting")


# =============================================================================
# ENTRY POINT
# =============================================================================

FIGURES = {
    1: ("success rate vs tolerance", fig1_success_vs_tolerance),
    2: ("MDN mixture selection", fig2_mdn_selection),
    3: ("seed quality vs NR iterations", fig3_seed_quality),
    4: ("optimizer comparison", fig4_optimizers),
    5: ("spatial error structure", fig5_spatial),
    6: ("Cartesian path tracking", fig6_path_tracking),
    7: ("MDN structural diagnostics", fig7_mdn_structure),
    8: ("accuracy-latency Pareto", fig8_pareto),
    9: ("training dynamics", fig9_training),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Publication figures for the neural inverse-kinematics "
                    "manuscript. Reads only the artefacts in PAPER_DATA/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Run it from the project directory, or point --out-dir at the "
               "PAPER_DATA folder from anywhere.")
    ap.add_argument("--only", type=int, nargs="+", default=None,
                    choices=sorted(FIGURES), help="render a subset")
    ap.add_argument("--out-dir", default=PATHS["out_dir"],
                    help="folder holding per_model_full.json, baselines.json, "
                         "statistics.json, trajectory.json, audit.json, "
                         "training_history.json and per_sample/")
    ap.add_argument("--fig-dir", default=None,
                    help="where figures are written (default: OUT_DIR/figures)")
    ap.add_argument("--raw-dir", default=None,
                    help="per-sample error vectors (default: OUT_DIR/per_sample)")
    ap.add_argument("--data-npz", default=PATHS["data_npz"],
                    help="processed dataset, needed only by figure 5")
    ap.add_argument("--scaler", default=PATHS["scaler"],
                    help="fitted input scaler, needed only by figure 5")
    ap.add_argument("--font-size", type=float, default=8.0)
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"],
                    help="pdf is the vector format IEEE prefers")
    ap.add_argument("--grayscale-check", action="store_true",
                    help="also emit luminance-only copies for the print test")
    ap.add_argument("--list", action="store_true",
                    help="list the figures and exit")
    args = ap.parse_args(argv)

    if args.list:
        for k, (name, _) in sorted(FIGURES.items()):
            print(f"  {k}  {name}")
        return 0

    PATHS["out_dir"] = os.path.expanduser(args.out_dir)
    PATHS["fig_dir"] = os.path.expanduser(
        args.fig_dir or os.path.join(PATHS["out_dir"], "figures"))
    PATHS["raw_dir"] = os.path.expanduser(
        args.raw_dir or os.path.join(PATHS["out_dir"], "per_sample"))
    PATHS["data_npz"] = os.path.expanduser(args.data_npz)
    PATHS["scaler"] = os.path.expanduser(args.scaler)

    global OUT_DIR, FIG_DIR, RAW_DIR
    OUT_DIR, FIG_DIR, RAW_DIR = (PATHS["out_dir"], PATHS["fig_dir"],
                                 PATHS["raw_dir"])

    if not os.path.isdir(PATHS["out_dir"]):
        print(f"error: {PATHS['out_dir']} does not exist.")
        print("Run extract_paper_data.py first, or pass --out-dir with the "
              "path to your PAPER_DATA folder.")
        return 2
    os.makedirs(PATHS["fig_dir"], exist_ok=True)

    plt = setup(args.font_size)
    R = Results()

    todo = args.only or sorted(FIGURES)
    made, failed = 0, []
    for k in todo:
        name, fn = FIGURES[k]
        print(f"[{k}] {name}")
        try:
            fn(R, plt)
            made += 1
        except Exception as exc:
            import traceback

            print(f"  FAILED: {exc}")
            traceback.print_exc()
            failed.append(k)

    if args.grayscale_check:
        grayscale_check()
    print(f"\n{made} figure(s) in {os.path.abspath(PATHS['fig_dir'])}")
    if failed:
        print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())