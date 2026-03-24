#!/usr/bin/env python3
"""
hpo_plots.py – Generate publication-quality HPO visualization plots.

Reads hpo_results.json and creates:
  1. Optimization History (best-so-far curves for both stages)
  2. Hyperparameter Importance (parallel coordinate plots)
  3. Trial Duration vs. Value scatter
  4. Per-Fold Variance of the best trial

Usage:
    .venv/bin/python scripts/hpo_plots.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "hpo_results.json"
FIG_DIR = SCRIPT_DIR.parent / "figures" / "hpo"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "text.color": "#c9d1d9",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "savefig.facecolor": "#0d1117",
    "savefig.dpi": 180,
})

ACCENT_S1 = "#58a6ff"   # Blue for Stage 1
ACCENT_S2 = "#f778a2"   # Pink for Stage 2
ACCENT_GREEN = "#3fb950"
ACCENT_ORANGE = "#d29922"
ACCENT_PURPLE = "#bc8cff"


def load_data():
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
# Plot 1: Optimization History (Best-So-Far)
# ═══════════════════════════════════════════════════════════════════════
def plot_optimization_history(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("HPO Optimization History", fontsize=16, fontweight="bold", y=1.02)

    for stage_key, ax, color, title, ylabel in [
        ("stage1", ax1, ACCENT_S1, "Stage 1: Change Detector", "F1-Score (Binary)"),
        ("stage2", ax2, ACCENT_S2, "Stage 2: Land-Cover Predictor", "Macro F1-Score"),
    ]:
        trials = data[stage_key]["trials"]
        complete = [t for t in trials if t["state"] == "COMPLETE"]
        pruned = [t for t in trials if t["state"] == "PRUNED"]

        # Complete trials
        nums_c = [t["number"] for t in complete]
        vals_c = [t["value"] for t in complete]

        # Best-so-far line
        best_so_far = []
        best = -float("inf")
        for t in sorted(complete, key=lambda x: x["number"]):
            if t["value"] > best:
                best = t["value"]
            best_so_far.append(best)
        nums_sorted = sorted(nums_c)

        # Scatter: all complete
        ax.scatter(nums_c, vals_c, alpha=0.25, s=12, color=color, zorder=2, label="Complete")

        # Scatter: pruned (muted)
        if pruned:
            nums_p = [t["number"] for t in pruned]
            vals_p = [t["value"] for t in pruned if t["value"] is not None]
            nums_p_valid = [t["number"] for t in pruned if t["value"] is not None]
            ax.scatter(nums_p_valid, vals_p, alpha=0.10, s=8, color="#8b949e", zorder=1,
                       marker="x", label=f"Pruned ({len(pruned)})")

        # Best-so-far line
        ax.plot(nums_sorted, best_so_far, color=color, lw=2.5, zorder=3, label="Best so far")

        # Best marker
        best_val = data[stage_key]["best_value"]
        best_trial_num = None
        for t in complete:
            if abs(t["value"] - best_val) < 1e-6:
                best_trial_num = t["number"]
                break
        if best_trial_num is not None:
            ax.scatter([best_trial_num], [best_val], s=120, color=ACCENT_GREEN, zorder=5,
                       edgecolors="white", linewidths=1.5, marker="*", label=f"Best: {best_val:.4f}")

        ax.set_title(title)
        ax.set_xlabel("Trial Number")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        n_complete = data[stage_key]["n_trials_complete"]
        n_pruned = data[stage_key]["n_trials_pruned"]
        ax.text(0.02, 0.98, f"{n_complete} complete, {n_pruned} pruned",
                transform=ax.transAxes, fontsize=9, va="top", color="#8b949e")

    fig.tight_layout()
    out = FIG_DIR / "optimization_history.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  ✅ {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Plot 2: Hyperparameter Importance (Parallel Coordinates)
# ═══════════════════════════════════════════════════════════════════════
def plot_param_vs_value(data):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Hyperparameter vs. Objective Value", fontsize=16, fontweight="bold", y=1.02)

    # Stage 1 params
    s1_trials = [t for t in data["stage1"]["trials"] if t["state"] == "COMPLETE"]
    s1_params_to_plot = [
        ("s1_n_estimators", "n_estimators"),
        ("s1_max_depth", "max_depth"),
        ("s1_max_features_frac", "max_features (frac)"),
    ]

    for i, (param_key, label) in enumerate(s1_params_to_plot):
        ax = axes[0, i]
        vals = []
        scores = []
        for t in s1_trials:
            if param_key in t["params"]:
                vals.append(t["params"][param_key])
                scores.append(t["value"])
        if vals:
            sc = ax.scatter(vals, scores, alpha=0.4, s=18, c=scores,
                           cmap="cool", edgecolors="none")
            ax.set_xlabel(label)
            ax.set_ylabel("F1-Score" if i == 0 else "")
            ax.set_title(f"S1: {label}", fontsize=11)
            ax.grid(True, alpha=0.2)

    # Stage 2 params
    s2_trials = [t for t in data["stage2"]["trials"] if t["state"] == "COMPLETE"]
    s2_params_to_plot = [
        ("s2_n_estimators", "n_estimators"),
        ("s2_max_depth", "max_depth"),
        ("pred_threshold", "pred_threshold"),
    ]

    for i, (param_key, label) in enumerate(s2_params_to_plot):
        ax = axes[1, i]
        vals = []
        scores = []
        for t in s2_trials:
            if param_key in t["params"]:
                vals.append(t["params"][param_key])
                scores.append(t["value"])
        if vals:
            sc = ax.scatter(vals, scores, alpha=0.4, s=18, c=scores,
                           cmap="RdYlGn", edgecolors="none")
            ax.set_xlabel(label)
            ax.set_ylabel("Macro F1" if i == 0 else "")
            ax.set_title(f"S2: {label}", fontsize=11)
            ax.grid(True, alpha=0.2)

    fig.tight_layout()
    out = FIG_DIR / "param_vs_value.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  ✅ {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Plot 3: Trial Duration vs. Value
# ═══════════════════════════════════════════════════════════════════════
def plot_duration_vs_value(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Trial Duration vs. Objective Value", fontsize=16, fontweight="bold", y=1.02)

    for stage_key, ax, color, title in [
        ("stage1", ax1, ACCENT_S1, "Stage 1: Change Detector"),
        ("stage2", ax2, ACCENT_S2, "Stage 2: Predictor"),
    ]:
        trials = data[stage_key]["trials"]
        complete = [t for t in trials if t["state"] == "COMPLETE" and t.get("duration_s")]
        pruned = [t for t in trials if t["state"] == "PRUNED" and t.get("duration_s")]

        if complete:
            durations = [t["duration_s"] for t in complete]
            values = [t["value"] for t in complete]
            ax.scatter(durations, values, alpha=0.4, s=18, color=color, label="Complete")

        if pruned:
            durations_p = [t["duration_s"] for t in pruned]
            values_p = [t["value"] for t in pruned if t["value"] is not None]
            durations_p_valid = [t["duration_s"] for t in pruned if t["value"] is not None]
            ax.scatter(durations_p_valid, values_p, alpha=0.15, s=10, color="#8b949e",
                       marker="x", label="Pruned")

        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("Objective Value")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "duration_vs_value.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  ✅ {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Plot 4: Per-Fold Variance of Best Trials
# ═══════════════════════════════════════════════════════════════════════
def plot_fold_variance(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Best Trial: Per-Fold Performance Variance", fontsize=16, fontweight="bold", y=1.02)

    # Stage 1
    s1_folds = data["stage1"]["best_user_attrs"].get("fold_f1s", [])
    if s1_folds:
        folds = list(range(1, len(s1_folds) + 1))
        bars = ax1.bar(folds, s1_folds, color=ACCENT_S1, width=0.5, edgecolor="#30363d", linewidth=0.5)
        ax1.axhline(y=np.mean(s1_folds), color=ACCENT_GREEN, linestyle="--", lw=1.5,
                    label=f"Mean: {np.mean(s1_folds):.4f}")
        ax1.set_xlabel("Spatial Fold")
        ax1.set_ylabel("F1-Score")
        ax1.set_title("Stage 1: Change Detector")
        ax1.set_xticks(folds)
        ax1.set_xticklabels(["North", "Mid-North", "Mid-South", "South"])
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.2, axis="y")

        # Annotate
        for bar, val in zip(bars, s1_folds):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="#c9d1d9")

    # Stage 2: accuracy and macro f1
    s2_accs = data["stage2"]["best_user_attrs"].get("fold_accs", [])
    s2_f1s = data["stage2"]["best_user_attrs"].get("fold_macro_f1s", [])
    if s2_accs and s2_f1s:
        folds = list(range(1, len(s2_accs) + 1))
        width = 0.3
        x = np.array(folds)

        bars1 = ax2.bar(x - width/2, s2_accs, width, color=ACCENT_PURPLE, label="Accuracy",
                       edgecolor="#30363d", linewidth=0.5)
        bars2 = ax2.bar(x + width/2, s2_f1s, width, color=ACCENT_S2, label="Macro F1",
                       edgecolor="#30363d", linewidth=0.5)

        ax2.axhline(y=np.mean(s2_accs), color=ACCENT_PURPLE, linestyle="--", lw=1, alpha=0.6)
        ax2.axhline(y=np.mean(s2_f1s), color=ACCENT_S2, linestyle="--", lw=1, alpha=0.6)

        ax2.set_xlabel("Spatial Fold")
        ax2.set_ylabel("Score")
        ax2.set_title("Stage 2: Land-Cover Predictor")
        ax2.set_xticks(folds)
        ax2.set_xticklabels(["North", "Mid-North", "Mid-South", "South"])
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.2, axis="y")

        for bar, val in zip(bars1, s2_accs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=ACCENT_PURPLE)
        for bar, val in zip(bars2, s2_f1s):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=ACCENT_S2)

    fig.tight_layout()
    out = FIG_DIR / "fold_variance.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  ✅ {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Plot 5: Summary Comparison Card
# ═══════════════════════════════════════════════════════════════════════
def plot_summary_card(data):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    meta = data["metadata"]
    s1 = data["stage1"]
    s2 = data["stage2"]
    baseline = data["baseline"]

    lines = [
        ("HPO Summary", "", 16, "bold", ACCENT_GREEN),
        ("", "", 6, "normal", "#c9d1d9"),
        (f"Runtime: {meta['total_runtime_hours']:.1f}h", f"Pixels: {meta['n_pixels']:,}  |  Features: {meta['n_features']}  |  Folds: {meta['n_folds']}", 11, "normal", "#8b949e"),
        ("", "", 10, "normal", "#c9d1d9"),
        ("── Stage 1: Change Detector ──", "", 13, "bold", ACCENT_S1),
        (f"  Best F1: {s1['best_value']:.4f}",
         f"  Baseline: n={baseline['stage1']['n_estimators']}, d={baseline['stage1']['max_depth']}", 11, "normal", "#c9d1d9"),
        (f"  Optimized: n={s1['best_params']['s1_n_estimators']}, d={s1['best_params']['s1_max_depth']}, "
         f"frac={s1['best_params'].get('s1_max_features_frac', 'N/A'):.2f}",
         f"  Trials: {s1['n_trials_complete']} complete + {s1['n_trials_pruned']} pruned = {len(s1['trials'])} total",
         11, "normal", "#c9d1d9"),
        ("", "", 8, "normal", "#c9d1d9"),
        ("── Stage 2: Land-Cover Predictor ──", "", 13, "bold", ACCENT_S2),
        (f"  Best Macro F1: {s2['best_value']:.4f}  |  Accuracy: {s2['best_user_attrs'].get('mean_accuracy', 'N/A')}",
         f"  Baseline: n={baseline['stage2']['n_estimators']}, d={baseline['stage2']['max_depth']}, t={baseline['stage2']['pred_threshold']}", 11, "normal", "#c9d1d9"),
        (f"  Optimized: n={s2['best_params']['s2_n_estimators']}, d={s2['best_params']['s2_max_depth']}, "
         f"t={s2['best_params']['pred_threshold']:.3f}",
         f"  Trials: {s2['n_trials_complete']} complete + {s2['n_trials_pruned']} pruned = {len(s2['trials'])} total",
         11, "normal", "#c9d1d9"),
    ]

    y = 0.95
    for left, right, size, weight, color in lines:
        ax.text(0.05, y, left, transform=ax.transAxes, fontsize=size,
                fontweight=weight, color=color, va="top", fontfamily="monospace")
        if right:
            ax.text(0.55, y, right, transform=ax.transAxes, fontsize=size,
                    fontweight="normal", color="#8b949e", va="top", fontfamily="monospace")
        y -= (size + 4) / 200

    fig.tight_layout()
    out = FIG_DIR / "summary_card.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  ✅ {out}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 50)
    print("  HPO Visualization Plots")
    print("=" * 50)

    data = load_data()
    print(f"\n  Data: {data['metadata']['total_runtime_hours']:.1f}h run,")
    print(f"  Stage 1: {len(data['stage1']['trials'])} trials (best F1={data['stage1']['best_value']:.4f})")
    print(f"  Stage 2: {len(data['stage2']['trials'])} trials (best Macro F1={data['stage2']['best_value']:.4f})")
    print(f"\n  Output: {FIG_DIR}/\n")

    plot_optimization_history(data)
    plot_param_vs_value(data)
    plot_duration_vs_value(data)
    plot_fold_variance(data)
    plot_summary_card(data)

    print(f"\n  ✅ All plots saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
