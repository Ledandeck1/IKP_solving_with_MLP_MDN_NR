#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
figures_3d.py  --  the figures that show the manipulator
--------------------------------------------------------------------------------
Run after paper_assets.py --stage multimodality.

    python3 figures_3d.py

--------------------------------------------------------------------------------
WHY THESE REPLACE WHAT THEY REPLACE
--------------------------------------------------------------------------------
The earlier figure set argued entirely in the space of statistics: boxes, bars,
paired lines, a Pareto frontier. That vocabulary belongs to a benchmark paper.
This is a robotics paper, and the robotics literature on multi-valued inverse
kinematics has shown the arm since the earliest work on the subject, because a
reader judges a posture by looking at it.

  fig_postures      replaces nothing; it is the figure the manuscript did not
                    have. One Cartesian target, several arm configurations that
                    all reach it, drawn as skeletons. Multimodality stops being
                    a number and becomes a picture.

  fig_workspace_3d  replaces fig5. Two flat projections and a colourbar cannot
                    convey a shell-shaped workspace. A point cloud in three
                    dimensions with the arm drawn for scale can.

  fig_optimizers    replaces fig4. The forest plot carried three quantities the
                    caption already stated. One panel with the paired runs and
                    the equivalence band is enough.

  fig_path          replaces fig6. Panel (a), the commanded joint angle, was the
                    only one carrying the argument; the other two restated
                    Table I. It is kept, enlarged, and paired with the arm drawn
                    at the waypoints where the MDN jumps.

  fig8 (Pareto)     deleted. Accuracy against latency is Table I with the axes
                    swapped, and the non-dominated staircase asks the reader to
                    learn a convention to read two columns of a table.

Author: Gustavo Henrique Germano Ledandeck (UFABC)
License: MIT
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PATHS: Dict[str, str] = {
    "out_dir": "PAPER_DATA",
    "fig_dir": os.path.join("PAPER_DATA", "figures"),
    "raw_dir": os.path.join("PAPER_DATA", "per_sample"),
    "data_npz": "dataset_processed.npz",
    "scaler": "scaler_X.pkl",
}

COL1, COL2 = 3.5, 7.16
RNG_SEED = 20260908
IK_JOINTS = [0, 1, 2, 3, 4]
JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist pitch", "Wrist roll", "Gripper"]
OPTIMIZERS = ["adamw", "muon"]
OPT_DISPLAY = {"adamw": "AdamW", "muon": "Muon"}

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
         "vermil": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
         "yellow": "#F0E442", "black": "#000000", "grey": "#999999"}

DH_TABLE = np.array([[6.5, 0.0, np.pi / 2], [0.0, 11.0, 0.0], [0.0, 15.0, 0.0],
                     [0.0, 0.0, np.pi / 2], [0.0, 0.0, -np.pi / 2],
                     [18.0, 0.0, 0.0]], dtype=np.float64)


def log(m: str = "", lvl: int = 0) -> None:
    print("  " * lvl + m, flush=True)


def load_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


# ------------------------------------------------------------- kinematics ----

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


def link_polyline(theta: np.ndarray) -> np.ndarray:
    """Origins of every frame, base to tool: the arm as a polyline, shape (7, 3).

    Drawing the frame origins rather than a CAD model keeps the figure honest
    about what the model actually is, a Denavit-Hartenberg chain, and keeps the
    file small enough for a vector figure.
    """
    return fk_frames(np.asarray(theta).reshape(1, 6))[0, :, :3, 3]


def wrap_deg(d: np.ndarray) -> np.ndarray:
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------- plotting ----

def setup(fontsize: float = 8.0):
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
        "grid.color": "#BBBBBB", "grid.alpha": 0.4, "grid.linewidth": 0.35,
        "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6,
        "lines.linewidth": 1.3, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })
    return plt


def save(fig, name: str) -> None:
    os.makedirs(PATHS["fig_dir"], exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PATHS["fig_dir"], f"{name}.{ext}"))
    import matplotlib.pyplot as plt

    plt.close(fig)
    log(f"wrote {os.path.join(PATHS['fig_dir'], name)}.pdf|.png", 1)


def panel(ax, text, dx=-0.10, dy=1.03):
    ax.text2D(dx, dy, text, transform=ax.transAxes, fontweight="bold",
              va="top", ha="left", fontsize=9) if hasattr(ax, "text2D") else \
        ax.text(dx, dy, text, transform=ax.transAxes, fontweight="bold",
                va="top", ha="left", fontsize=9)


def draw_arm(ax, theta, color, lw=1.6, alpha=1.0, joints=True, label=None,
             zorder=3):
    """Draw one configuration as a polyline with markers at the joint origins."""
    P = link_polyline(theta)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round", label=label, zorder=zorder)
    if joints:
        ax.scatter(P[1:-1, 0], P[1:-1, 1], P[1:-1, 2], s=7, color=color,
                   alpha=alpha, edgecolor="none", zorder=zorder + 1)
    return P


def style_3d(ax, lim=None, elev=22, azim=-58):
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("$x$ (cm)", labelpad=-8)
    ax.set_ylabel("$y$ (cm)", labelpad=-8)
    ax.set_zlabel("$z$ (cm)", labelpad=-8)
    ax.tick_params(labelsize=6.0, pad=-3)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.04)
        pane.pane.set_edgecolor("0.85")
    ax.grid(True, alpha=0.18, linewidth=0.3)
    if lim is not None:
        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])
        ax.set_zlim(lim[2])
    try:
        ax.set_box_aspect((1, 1, 0.92), zoom=1.28)
    except Exception:
        pass


# =============================================================================
# FIGURE  --  MULTIMODALITY, DRAWN
# =============================================================================

def fig_postures(plt, n_panels: int = 3, max_postures: int = 5) -> None:
    """One target, several arms that reach it.

    This is the claim of the paper in a single image. For a requested Cartesian
    position, the mixture density network emits K candidate configurations; the
    ones drawn here all place the tool within a few millimetres of the same
    point, through postures that differ by tens of degrees at the shoulder and
    elbow. A deterministic regressor must choose one number per joint and, when
    trained on a squared loss, converges toward the average of these arms, which
    is itself not one of them.

    The panels are chosen among targets where at least two components are
    genuinely distinct, so the figure illustrates the phenomenon rather than
    claiming it is universal; how often it occurs is reported in Table II and in
    the tolerance sweep, not here.
    """
    path = os.path.join(PATHS["raw_dir"], "postures_mdn.npz")
    if not os.path.exists(path):
        log("[skip] postures: run paper_assets.py --stage multimodality first", 1)
        return
    z = np.load(path)
    mu, pi, cerr = z["mu_rad"], z["pi"], z["comp_err_mm"]
    targets, y_true = z["targets_cm"], z["y_true_rad"]
    if mu.shape[0] == 0:
        log("[skip] postures: no target had two distinct valid components", 1)
        return

    # rank the stored targets by how many genuinely distinct postures they admit
    def distinct_set(i: int, tol_mm: float = 5.0, tau_deg: float = 30.0):
        ok = np.where(cerr[i] <= tol_mm)[0]
        ok = ok[np.argsort(cerr[i][ok])]
        keep: List[int] = []
        for c in ok:
            if all(np.abs(wrap_deg(np.rad2deg(
                    mu[i, c, IK_JOINTS] - mu[i, k, IK_JOINTS]))).max() > tau_deg
                   for k in keep):
                keep.append(int(c))
            if len(keep) >= max_postures:
                break
        return keep

    ranked = sorted(range(mu.shape[0]), key=lambda i: -len(distinct_set(i)))
    chosen = [i for i in ranked if len(distinct_set(i)) >= 2][:n_panels]
    if not chosen:
        log("[skip] postures: no target with two distinct postures", 1)
        return

    palette = [OKABE["blue"], OKABE["vermil"], OKABE["green"],
               OKABE["purple"], OKABE["orange"]]

    fig = plt.figure(figsize=(COL2, 3.5))
    axes = [fig.add_subplot(1, len(chosen), k + 1, projection="3d")
            for k in range(len(chosen))]
    # tight margins: a 3D axes wastes most of its box on empty corners, so the
    # panels are pushed outward until the arms fill the space they are given
    fig.subplots_adjust(left=-0.02, right=1.02, top=1.10, bottom=0.02,
                        wspace=0.02)

    for k, (ax, i) in enumerate(zip(axes, chosen)):
        comps = distinct_set(i)
        spread = 0.0
        for j, c in enumerate(comps):
            draw_arm(ax, mu[i, c], palette[j % len(palette)], lw=1.9,
                     alpha=0.95, zorder=3 + j)
            for k2 in comps[:j]:
                spread = max(spread, float(np.abs(wrap_deg(np.rad2deg(
                    mu[i, c, IK_JOINTS] - mu[i, k2, IK_JOINTS]))).max()))
        t = targets[i]
        ax.scatter([t[0]], [t[1]], [t[2]], marker="*", s=150, color="black",
                   depthshade=False, zorder=20)
        # each panel is framed on its own arms; a shared frame would shrink
        # every posture to fit the widest one and lose the detail
        Pk = np.vstack([link_polyline(mu[i, c]) for c in comps] + [t[None, :]])
        ctr = Pk.mean(axis=0)
        half = float(np.abs(Pk - ctr).max()) * 1.02
        style_3d(ax, [(ctr[0] - half, ctr[0] + half),
                      (ctr[1] - half, ctr[1] + half),
                      (ctr[2] - half, ctr[2] + half)])
        ax.set_title(f"({chr(97 + k)}) {len(comps)} postures, "
                     f"up to {spread:.0f}$^\\circ$ apart",
                     fontsize=6.8, pad=-14, fontweight="bold")

    handles = [plt.Line2D([], [], color=palette[j % len(palette)], lw=1.8,
                          label=f"solution {j + 1}")
               for j in range(max(len(distinct_set(i)) for i in chosen))]
    handles.append(plt.Line2D([], [], color="black", marker="*", lw=0,
                              markersize=9, label="requested position"))
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               fontsize=6.6, bbox_to_anchor=(0.5, 0.005))
    fig.text(0.5, -0.055,
             "Every arm shown places the tool within 5 mm of the star. "
             "A network trained on a squared loss in joint space must return a "
             "single number per joint and converges toward the average of these "
             "configurations, which is not itself a solution.",
             ha="center", fontsize=6.2, color="0.35")
    save(fig, "fig_postures")


# =============================================================================
# FIGURE  --  WORKSPACE IN THREE DIMENSIONS
# =============================================================================

def fig_workspace_3d(plt, n_points: int = 12000) -> None:
    """The reachable set, coloured by the error the MLP makes there.

    The two flat projections this replaces could not show that the workspace is
    a shell with a hollow interior, and the reader had to reassemble the shape
    from two silhouettes. Here the cloud is drawn once, the arm is drawn inside
    it at a mid-range posture so the scale is legible, and the second panel
    slices the same cloud so the interior structure stays readable in print.
    """
    from matplotlib.colors import LogNorm

    try:
        import joblib

        d = np.load(PATHS["data_npz"])
        scaler = joblib.load(PATHS["scaler"])
        P = scaler.inverse_transform(d["X_test"])
        y_test = d["y_test"]
    except Exception as exc:
        log(f"[skip] workspace: {exc}", 1)
        return

    err = None
    for cand in ("parity_mlp.npz",):
        p = os.path.join(PATHS["raw_dir"], cand)
        if os.path.exists(p):
            z = np.load(p)
            err = z["task_error_mm"]
            break
    if err is None:
        log("[skip] workspace: parity_mlp.npz missing", 1)
        return
    n = min(len(err), len(P))
    P, err = P[:n], err[:n]

    rng = np.random.default_rng(RNG_SEED)
    sel = rng.choice(n, min(n_points, n), replace=False)
    Ps, es = P[sel], err[sel]
    order = np.argsort(es)          # accurate points drawn first, under the rest

    fig = plt.figure(figsize=(COL2, 3.0))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2)

    norm = LogNorm(vmin=max(np.percentile(es, 2), 1e-1),
                   vmax=np.percentile(es, 98))
    sc = ax0.scatter(Ps[order, 0], Ps[order, 1], Ps[order, 2], c=es[order],
                     norm=norm, cmap="viridis", s=1.2, linewidths=0,
                     alpha=0.55, depthshade=False, rasterized=True)

    mid = np.deg2rad([90.0, 65.0, -45.0, -45.0, 0.0, 45.0])
    draw_arm(ax0, mid, "black", lw=2.0, zorder=20)
    ax0.scatter([0], [0], [0], s=26, color="black", marker="s",
                depthshade=False, zorder=21)
    style_3d(ax0, elev=20, azim=-62)
    ax0.set_title("reachable positions in the test split,\n"
                  "arm drawn at a mid-range posture", fontsize=6.4, pad=-2)
    panel(ax0, "(a)", dx=0.0, dy=1.02)

    cax = fig.add_axes([0.10, -0.05, 0.30, 0.022])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("MLP position error (mm)", fontsize=6.4, labelpad=1)
    cb.ax.tick_params(labelsize=5.8, pad=1)

    # (b) a vertical slice, where the shell structure is unambiguous
    from scipy import stats as sstats

    band = np.abs(Ps[:, 1]) <= 4.0
    if band.sum() < 200:
        band = np.abs(Ps[:, 1] - np.median(Ps[:, 1])) <= 6.0
    H, xe, ze, _ = sstats.binned_statistic_2d(
        Ps[band, 0], Ps[band, 2], es[band], statistic="median", bins=44)
    C, _, _, _ = sstats.binned_statistic_2d(
        Ps[band, 0], Ps[band, 2], es[band], statistic="count", bins=[xe, ze])
    H = np.where(C >= 3, H, np.nan)
    ax1.pcolormesh(xe, ze, H.T, norm=norm, cmap="viridis", shading="auto",
                   rasterized=True)
    Pm = link_polyline(mid)
    ax1.plot(Pm[:, 0], Pm[:, 2], color="black", lw=1.8, zorder=5)
    ax1.scatter(Pm[1:-1, 0], Pm[1:-1, 2], s=10, color="black", zorder=6)
    ax1.scatter([0], [0], s=30, color="black", marker="s", zorder=6)
    ax1.set_xlabel("$x$ (cm)")
    ax1.set_ylabel("$z$ (cm)")
    ax1.set_aspect("equal")
    ax1.grid(False)
    ax1.set_title("slice through $|y| \\leq 4$ cm", fontsize=6.4, pad=3)
    panel(ax1, "(b)", dx=-0.20, dy=1.06)

    save(fig, "fig_workspace_3d")


# =============================================================================
# FIGURE  --  OPTIMIZERS, ONE PANEL
# =============================================================================

def fig_optimizers(plt) -> None:
    """AdamW against Muon, on one axis.

    The version this replaces spent a forest plot on three numbers the caption
    already gave. Here each seed is a line between the two arms, so the reader
    sees the consistency that produces the small p-value, and the inset states
    the effect against the margin of practical relevance. Nothing else is
    needed: the corrected p-values live in Table IV.
    """
    import pandas as pd

    csv = os.path.join(PATHS["out_dir"], "per_model.csv")
    if not os.path.exists(csv):
        log("[skip] optimizers: per_model.csv missing", 1)
        return
    df = pd.read_csv(csv)
    df = df[df["optimizer"].isin(OPTIMIZERS)]
    ost = load_json(os.path.join(PATHS["out_dir"], "optimizer_stats.json"), {})

    methods = [("mlp", "MLP"), ("mdn_fk_best", "MDN, FK selection"),
               ("mlp_nr", "MLP + NR")]
    methods = [(m, l) for m, l in methods if m in set(df["method"])]

    fig, axes = plt.subplots(1, len(methods), figsize=(COL2, 2.4),
                             gridspec_kw=dict(wspace=0.45))
    axes = np.atleast_1d(axes)

    pw = {r["method"]: r for r in ost.get("pairwise", [])}
    x = np.array([0.0, 1.0])
    for ax, (method, label) in zip(axes, methods):
        sub = df[df["method"] == method]
        piv = sub.pivot_table(index="seed", columns="optimizer",
                              values="err_mean_mm").dropna()
        if piv.empty:
            continue
        for _, row in piv.iterrows():
            v = [row[o] for o in OPTIMIZERS]
            up = v[1] > v[0]
            ax.plot(x, v, color=OKABE["vermil"] if up else OKABE["blue"],
                    lw=0.8, alpha=0.65, marker="o", markersize=2.6, zorder=2)
        for i, o in enumerate(OPTIMIZERS):
            m = piv[o].mean()
            ax.plot([x[i] - 0.16, x[i] + 0.16], [m, m], color="black", lw=1.8,
                    zorder=4)
            ax.annotate(f"{m:.3g}", xy=(x[i], m), xytext=(0, 9 if i == 0 else 9),
                        textcoords="offset points", ha="center", fontsize=6.2,
                        fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([OPT_DISPLAY[o] for o in OPTIMIZERS], fontsize=7)
        ax.set_xlim(-0.42, 1.42)
        ax.set_title(label, fontsize=6.8, pad=10)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="y", labelsize=6.2)

        r = pw.get(method)
        if r:
            eq = r.get("tost", {}).get("equivalent")
            ax.annotate(
                f"$\\Delta$ = {r['delta_mm']:+.3g} mm\n"
                f"$p$ = {r['p_raw']:.3f}" +
                ("\nequivalent at $\\pm$0.5 mm" if eq else ""),
                xy=(0.5, 0.03), xycoords="axes fraction", ha="center",
                va="bottom", fontsize=5.9, color="0.28")
    axes[0].set_ylabel("Mean position error (mm)")
    fig.text(0.5, -0.16,
             "One line per seed. Blue: Muon improved that seed; orange: AdamW "
             "did. Bars are the arm means.",
             ha="center", fontsize=6.0, color="0.35")
    save(fig, "fig_optimizers")


# =============================================================================
# FIGURE  --  PATH TRACKING, WITH THE ARM
# =============================================================================

def fig_path(plt) -> None:
    """A branch switch, as an angle and as two arms.

    Panel (a) is the commanded angle of the joint that moves most between
    adjacent waypoints; panel (b) draws the arm on either side of the largest
    jump. The reason for the pairing is that the height of a spike in (a) is
    hard to translate into a physical motion, and (b) does the translation.
    """
    tj = load_json(os.path.join(PATHS["out_dir"], "trajectory.json"), {})
    paths = tj.get("paths", {})
    if not paths:
        log("[skip] path: trajectory.json missing", 1)
        return
    pname = "circle_xy" if "circle_xy" in paths else next(iter(paths))
    entry = paths[pname]

    series = [("mlp", "MLP", OKABE["vermil"], "-"),
              ("mdn_fk_best", "MDN, FK selection", OKABE["green"], "-."),
              ("mlp_nr", "MLP + NR", OKABE["blue"], "-"),
              ("nr_warmstart_sequential", "NR, warm start", OKABE["grey"], "--")]
    series = [s for s in series if s[0] in entry]
    if not series:
        log("[skip] path: no comparable methods", 1)
        return

    def jseries(key: str) -> np.ndarray:
        return np.asarray(entry[key]["joint_series_deg"], dtype=float)

    jumps = np.zeros(6)
    for key, *_ in series:
        jumps = np.maximum(jumps, np.abs(np.diff(jseries(key), axis=0)).max(0))
    j_idx = int(np.argmax(jumps))

    worst = max(series, key=lambda t: np.abs(np.diff(jseries(t[0])[:, j_idx])).max())
    Jw = jseries(worst[0])
    k = int(np.argmax(np.abs(np.diff(Jw[:, j_idx]))))
    delta = float(abs(Jw[k + 1, j_idx] - Jw[k, j_idx]))

    fig = plt.figure(figsize=(COL2, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1], projection="3d")

    for key, lbl, c, ls in series:
        ax.plot(jseries(key)[:, j_idx], color=c, ls=ls, lw=1.1, label=lbl)
    ax.axvspan(k - 3, k + 3, color=worst[2], alpha=0.12, zorder=0)
    ax.annotate(f"{delta:.0f}$^\\circ$ between waypoints {k} and {k + 1}",
                xy=(k, Jw[k:k + 2, j_idx].mean()), xytext=(0.32, 0.07),
                textcoords="axes fraction", fontsize=6.2, color=worst[2],
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=worst[2]))
    ax.set_xlabel("Waypoint index")
    ax.set_ylabel(f"Commanded {JOINT_NAMES[j_idx].lower()} angle (deg)")
    ax.set_title(f"{pname.replace('_', ' ')} path: the commanded trajectory",
                 fontsize=6.6, pad=4)
    ax.legend(fontsize=6.2, loc="upper left", ncol=2)
    panel(ax, "(a)", dx=-0.13, dy=1.10)

    th_a = np.deg2rad(Jw[k])
    th_b = np.deg2rad(Jw[k + 1])
    draw_arm(ax3, th_a, OKABE["blue"], lw=1.8, label=f"waypoint {k}")
    draw_arm(ax3, th_b, worst[2], lw=1.8, label=f"waypoint {k + 1}")
    pa, pb = fk_batch(th_a.reshape(1, 6))[0], fk_batch(th_b.reshape(1, 6))[0]
    ax3.scatter([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], marker="*",
                s=70, color="black", depthshade=False, zorder=20)
    allP = np.vstack([link_polyline(th_a), link_polyline(th_b)])
    ctr = allP.mean(0)
    half = float(np.abs(allP - ctr).max()) * 1.08
    style_3d(ax3, [(ctr[0] - half, ctr[0] + half),
                   (ctr[1] - half, ctr[1] + half),
                   (ctr[2] - half, ctr[2] + half)], elev=20, azim=-58)
    ax3.set_title(f"the same two waypoints,\n"
                  f"{np.linalg.norm(pa - pb) * 10:.1f} mm apart in space",
                  fontsize=6.4, pad=-2)
    ax3.legend(fontsize=6.0, loc="upper left", bbox_to_anchor=(-0.05, 0.96))
    panel(ax3, "(b)", dx=0.0, dy=1.02)

    save(fig, "fig_path_branch_switch")


# ================================================================== main =====

FIGURES = {
    "postures": ("multimodality drawn as arms", fig_postures),
    "workspace": ("workspace in 3D", fig_workspace_3d),
    "optimizers": ("AdamW vs Muon, one panel", fig_optimizers),
    "path": ("branch switch on a Cartesian path", fig_path),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("---")[0])
    ap.add_argument("--only", nargs="+", default=None, choices=sorted(FIGURES))
    ap.add_argument("--out-dir", default="PAPER_DATA")
    ap.add_argument("--data-npz", default="dataset_processed.npz")
    ap.add_argument("--scaler", default="scaler_X.pkl")
    ap.add_argument("--font-size", type=float, default=8.0)
    ap.add_argument("--n-panels", type=int, default=3,
                    help="targets shown in the posture figure")
    ap.add_argument("--max-postures", type=int, default=5,
                    help="arms drawn per target")
    args = ap.parse_args(argv)

    PATHS["out_dir"] = os.path.expanduser(args.out_dir)
    PATHS["fig_dir"] = os.path.join(PATHS["out_dir"], "figures")
    PATHS["raw_dir"] = os.path.join(PATHS["out_dir"], "per_sample")
    PATHS["data_npz"] = os.path.expanduser(args.data_npz)
    PATHS["scaler"] = os.path.expanduser(args.scaler)
    os.makedirs(PATHS["fig_dir"], exist_ok=True)

    plt = setup(args.font_size)
    todo = args.only or sorted(FIGURES)
    for key in todo:
        name, fn = FIGURES[key]
        log(f"[{key}] {name}")
        try:
            if key == "postures":
                fn(plt, n_panels=args.n_panels, max_postures=args.max_postures)
            else:
                fn(plt)
        except Exception as exc:
            import traceback

            log(f"FAILED: {exc}", 1)
            traceback.print_exc()
    log(f"\nfigures in {os.path.abspath(PATHS['fig_dir'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
