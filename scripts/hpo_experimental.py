#!/usr/bin/env python3
"""
hpo_experimental.py – Hyperparameter Optimization for the 2-Stage Experimental Model.

Uses Optuna (TPE sampler + MedianPruner) to optimize hyperparameters for:
  - Stage 1: Binary Change Detector (RandomForest)
  - Stage 2: Multiclass Land-Cover Predictor (RandomForest)

This script imports data-loading functions from experimental_nuremberg.py.
All HPO logic is self-contained here.

Usage:
    .venv/bin/python scripts/hpo_experimental.py                  # Full 6h run
    .venv/bin/python scripts/hpo_experimental.py --hours 0.1      # Quick smoke test (6 min)
    .venv/bin/python scripts/hpo_experimental.py --hours 6 --hard-limit 8  # 6h target, 8h hard stop

Output:
    scripts/hpo_results.json   – Structured results with best params, trial history, metrics.
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# ---------------------------------------------------------------------------
# Import data-loading from the existing experimental script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from experimental_nuremberg import (
    load_and_align,
    build_boundary_mask,
    build_feature_matrix,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_PATH = SCRIPT_DIR / "hpo_results.json"
NUM_FOLDS = 4


def create_spatial_folds(y_row, num_folds=NUM_FOLDS):
    """Create fold IDs using latitudinal banding (identical to production)."""
    n = len(y_row)
    sort_idx = np.argsort(y_row)
    fold_ids = np.zeros(n, dtype=int)
    fold_ids[sort_idx] = np.arange(n) * num_folds // n
    fold_ids = np.clip(fold_ids, 0, num_folds - 1)
    return fold_ids


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 Objective: Binary Change Detection
# ═══════════════════════════════════════════════════════════════════════════
def stage1_objective(trial, X, is_changed, y_row, fold_ids, start_time, time_limit_s):
    """Optimize Stage 1 (change detector) hyperparameters."""
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > time_limit_s:
        raise optuna.exceptions.OptunaError("Time budget exceeded for Stage 1")

    # Sample hyperparameters
    n_estimators = trial.suggest_int("s1_n_estimators", 50, 500, step=50)
    max_depth = trial.suggest_int("s1_max_depth", 5, 40)
    min_samples_split = trial.suggest_int("s1_min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("s1_min_samples_leaf", 1, 10)
    max_features_type = trial.suggest_categorical("s1_max_features_type", ["sqrt", "log2", "fraction"])
    if max_features_type == "fraction":
        max_features = trial.suggest_float("s1_max_features_frac", 0.3, 0.9)
    else:
        max_features = max_features_type
    class_weight = trial.suggest_categorical("s1_class_weight", ["balanced", "balanced_subsample", "none"])
    if class_weight == "none":
        class_weight = None

    fold_f1s = []

    for f_idx in range(NUM_FOLDS):
        # Time check per fold
        if time.time() - start_time > time_limit_s:
            raise optuna.exceptions.OptunaError("Time budget exceeded mid-fold")

        train_mask = fold_ids != f_idx
        test_mask = fold_ids == f_idx

        X_train, X_test = X[train_mask], X[test_mask]
        y_train_chg = is_changed[train_mask]
        y_test_chg = is_changed[test_mask]

        # Balanced sampling for training
        chg_mask = y_train_chg == 1
        n_chg = chg_mask.sum()
        if n_chg == 0:
            fold_f1s.append(0.0)
            continue

        stb_indices = np.where(~chg_mask)[0]
        rng = np.random.default_rng(42 + f_idx)
        stb_sample = rng.choice(stb_indices, size=min(n_chg, len(stb_indices)), replace=False)
        bal_indices = np.concatenate([np.where(chg_mask)[0], stb_sample])

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_train[bal_indices], y_train_chg[bal_indices])

        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test_chg, y_pred, pos_label=1, zero_division=0)
        fold_f1s.append(f1)

        # Report intermediate value for pruning (Hyperband-style)
        trial.report(np.mean(fold_f1s), f_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    mean_f1 = np.mean(fold_f1s)
    trial.set_user_attr("fold_f1s", [round(f, 4) for f in fold_f1s])
    return mean_f1


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 Objective: Multiclass Land-Cover Prediction
# ═══════════════════════════════════════════════════════════════════════════
def stage2_objective(trial, X, y_2020, y_2021, is_changed, y_row, fold_ids,
                     best_s1_params, start_time, time_limit_s):
    """Optimize Stage 2 (multiclass predictor) hyperparameters.

    Uses the best Stage 1 params to generate change predictions, then
    optimizes Stage 2 on the end-to-end pipeline accuracy.
    """
    elapsed = time.time() - start_time
    if elapsed > time_limit_s:
        raise optuna.exceptions.OptunaError("Time budget exceeded for Stage 2")

    # Sample Stage 2 hyperparameters
    n_estimators = trial.suggest_int("s2_n_estimators", 50, 500, step=50)
    max_depth = trial.suggest_int("s2_max_depth", 5, 50)
    min_samples_split = trial.suggest_int("s2_min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("s2_min_samples_leaf", 1, 10)
    max_features_type = trial.suggest_categorical("s2_max_features_type", ["sqrt", "log2", "fraction"])
    if max_features_type == "fraction":
        max_features = trial.suggest_float("s2_max_features_frac", 0.3, 0.9)
    else:
        max_features = max_features_type
    class_weight = trial.suggest_categorical("s2_class_weight", ["balanced", "balanced_subsample", "none"])
    if class_weight == "none":
        class_weight = None
    pred_threshold = trial.suggest_float("pred_threshold", 0.50, 0.95)

    # Reconstruct Stage 1 params
    s1_max_features = best_s1_params.get("s1_max_features_frac", best_s1_params.get("s1_max_features_type", "sqrt"))
    s1_class_weight = best_s1_params.get("s1_class_weight")
    if s1_class_weight == "none":
        s1_class_weight = None

    fold_accs = []
    fold_macro_f1s = []

    for f_idx in range(NUM_FOLDS):
        if time.time() - start_time > time_limit_s:
            raise optuna.exceptions.OptunaError("Time budget exceeded mid-fold")

        train_mask = fold_ids != f_idx
        test_mask = fold_ids == f_idx

        X_train, X_test = X[train_mask], X[test_mask]
        y_train_chg = is_changed[train_mask]
        y_train_cls = y_2021[train_mask]
        y_test_true = y_2021[test_mask]
        y_test_2020 = y_2020[test_mask]

        # Stage 1: Use best params
        chg_mask = y_train_chg == 1
        n_chg = chg_mask.sum()
        if n_chg == 0:
            fold_accs.append(0.0)
            fold_macro_f1s.append(0.0)
            continue

        stb_indices = np.where(~chg_mask)[0]
        rng = np.random.default_rng(42 + f_idx)
        stb_sample = rng.choice(stb_indices, size=min(n_chg, len(stb_indices)), replace=False)
        bal_indices = np.concatenate([np.where(chg_mask)[0], stb_sample])

        clf1 = RandomForestClassifier(
            n_estimators=best_s1_params.get("s1_n_estimators", 100),
            max_depth=best_s1_params.get("s1_max_depth", 15),
            min_samples_split=best_s1_params.get("s1_min_samples_split", 2),
            min_samples_leaf=best_s1_params.get("s1_min_samples_leaf", 1),
            max_features=s1_max_features,
            class_weight=s1_class_weight,
            n_jobs=-1,
            random_state=42,
        )
        clf1.fit(X_train[bal_indices], y_train_chg[bal_indices])

        # Stage 2: Use trial params
        chg_indices = np.where(chg_mask)[0]
        clf2 = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=42,
        )
        clf2.fit(X_train[chg_indices], y_train_cls[chg_indices])

        # Predict
        probs = clf1.predict_proba(X_test)
        chg_col = list(clf1.classes_).index(1) if 1 in clf1.classes_ else -1
        chg_probs = probs[:, chg_col] if chg_col >= 0 else np.zeros(len(X_test))

        pred_changed = chg_probs > pred_threshold
        fold_preds = y_test_2020.copy()
        if pred_changed.any():
            stage2_preds = clf2.predict(X_test[pred_changed])
            fold_preds[pred_changed] = stage2_preds

        acc = accuracy_score(y_test_true, fold_preds)
        macro_f1 = f1_score(y_test_true, fold_preds, average="macro", zero_division=0)
        fold_accs.append(acc)
        fold_macro_f1s.append(macro_f1)

        # Report for pruning
        trial.report(np.mean(fold_macro_f1s), f_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    mean_acc = np.mean(fold_accs)
    mean_macro_f1 = np.mean(fold_macro_f1s)
    trial.set_user_attr("fold_accs", [round(a, 4) for a in fold_accs])
    trial.set_user_attr("fold_macro_f1s", [round(f, 4) for f in fold_macro_f1s])
    trial.set_user_attr("mean_accuracy", round(mean_acc, 4))
    return mean_macro_f1


# ═══════════════════════════════════════════════════════════════════════════
# Progress Callback
# ═══════════════════════════════════════════════════════════════════════════
class ProgressCallback:
    """Print a live progress line after each trial."""

    def __init__(self, stage_name, start_time, time_limit_s):
        self.stage_name = stage_name
        self.start_time = start_time
        self.time_limit_s = time_limit_s
        self.best_value = -float("inf")

    def __call__(self, study, trial):
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        remaining = max(0, self.time_limit_s - elapsed)
        remaining_str = str(timedelta(seconds=int(remaining)))

        if trial.value is not None and trial.value > self.best_value:
            self.best_value = trial.value
            marker = " ★ NEW BEST"
        else:
            marker = ""

        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

        value_str = f"{trial.value:.4f}" if trial.value is not None else "PRUNED"
        print(
            f"  [{self.stage_name}] Trial {trial.number:3d} | "
            f"Value: {value_str} | Best: {self.best_value:.4f} | "
            f"Done: {n_complete} | Pruned: {n_pruned} | "
            f"Elapsed: {elapsed_str} | Remaining: {remaining_str}"
            f"{marker}",
            flush=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="HPO for 2-Stage Experimental Model")
    parser.add_argument("--hours", type=float, default=6.0, help="Target runtime in hours (default: 6)")
    parser.add_argument("--hard-limit", type=float, default=8.0, help="Hard stop after this many hours (default: 8)")
    args = parser.parse_args()

    target_seconds = args.hours * 3600
    hard_limit_seconds = args.hard_limit * 3600

    # Register SIGALRM for hard stop
    def hard_stop_handler(signum, frame):
        print(f"\n{'='*60}")
        print(f"HARD TIME LIMIT ({args.hard_limit}h) REACHED — forcing stop.")
        print(f"{'='*60}")
        # Save whatever we have so far
        sys.exit(0)

    signal.signal(signal.SIGALRM, hard_stop_handler)
    signal.alarm(int(hard_limit_seconds))

    global_start = time.time()
    print("=" * 70)
    print(f"  HPO for 2-Stage Experimental Model")
    print(f"  Target: {args.hours}h | Hard limit: {args.hard_limit}h")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Load Data ──────────────────────────────────────────────────────
    print("\nLoading data...")
    wc2020, wc2021, s2_2020, stats, std_layer = load_and_align()
    boundary_mask = build_boundary_mask()
    X, y_2020, y_2021, y_row, flat_idx, is_changed, feature_names, valid = \
        build_feature_matrix(wc2020, wc2021, s2_2020, stats, std_layer, boundary_mask)

    print(f"  Data loaded: {X.shape[0]:,} pixels × {X.shape[1]} features")
    print(f"  Changed pixels: {is_changed.sum():,} ({is_changed.mean()*100:.2f}%)")

    # ── Create Folds ───────────────────────────────────────────────────
    fold_ids = create_spatial_folds(y_row)
    for f in range(NUM_FOLDS):
        n_f = (fold_ids == f).sum()
        print(f"  Fold {f}: {n_f:,} pixels")

    data_load_time = time.time() - global_start
    print(f"\n  Data loading took {data_load_time:.0f}s")

    # Allocate time: 40% Stage 1, 60% Stage 2
    remaining_time = target_seconds - data_load_time
    s1_time = remaining_time * 0.4
    s2_time = remaining_time * 0.6

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1 HPO
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  STAGE 1: Change Detector HPO")
    print(f"  Time budget: {s1_time/3600:.1f}h ({s1_time:.0f}s)")
    print(f"  Objective: F1-score on binary change class")
    print(f"{'='*70}\n")

    s1_start = time.time()
    s1_study = optuna.create_study(
        study_name="stage1_detector",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    s1_callback = ProgressCallback("S1", s1_start, s1_time)

    def s1_obj(trial):
        return stage1_objective(trial, X, is_changed, y_row, fold_ids, s1_start, s1_time)

    try:
        s1_study.optimize(
            s1_obj,
            timeout=s1_time,
            callbacks=[s1_callback],
            show_progress_bar=True,
        )
    except Exception as e:
        print(f"\n  Stage 1 stopped: {e}")

    s1_elapsed = time.time() - s1_start
    s1_best = s1_study.best_trial
    print(f"\n  ✅ Stage 1 complete in {s1_elapsed/60:.1f} min")
    print(f"  Best F1: {s1_best.value:.4f}")
    print(f"  Best params: {json.dumps(s1_best.params, indent=2)}")
    print(f"  Trials: {len(s1_study.trials)} total, "
          f"{len([t for t in s1_study.trials if t.state == optuna.trial.TrialState.COMPLETE])} complete, "
          f"{len([t for t in s1_study.trials if t.state == optuna.trial.TrialState.PRUNED])} pruned")

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2 HPO
    # ══════════════════════════════════════════════════════════════════
    # Recalculate remaining time
    total_elapsed = time.time() - global_start
    s2_time = max(60, target_seconds - total_elapsed)  # At least 1 min

    print(f"\n{'='*70}")
    print(f"  STAGE 2: Land-Cover Predictor HPO")
    print(f"  Time budget: {s2_time/3600:.1f}h ({s2_time:.0f}s)")
    print(f"  Objective: Macro F1-score (end-to-end pipeline)")
    print(f"  Using best Stage 1 params from above")
    print(f"{'='*70}\n")

    s2_start = time.time()
    s2_study = optuna.create_study(
        study_name="stage2_predictor",
        direction="maximize",
        sampler=TPESampler(seed=123),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    s2_callback = ProgressCallback("S2", s2_start, s2_time)

    def s2_obj(trial):
        return stage2_objective(
            trial, X, y_2020, y_2021, is_changed, y_row, fold_ids,
            s1_best.params, s2_start, s2_time
        )

    try:
        s2_study.optimize(
            s2_obj,
            timeout=s2_time,
            callbacks=[s2_callback],
            show_progress_bar=True,
        )
    except Exception as e:
        print(f"\n  Stage 2 stopped: {e}")

    s2_elapsed = time.time() - s2_start
    s2_best = s2_study.best_trial
    print(f"\n  ✅ Stage 2 complete in {s2_elapsed/60:.1f} min")
    print(f"  Best Macro F1: {s2_best.value:.4f}")
    print(f"  Best params: {json.dumps(s2_best.params, indent=2)}")
    print(f"  Trials: {len(s2_study.trials)} total, "
          f"{len([t for t in s2_study.trials if t.state == optuna.trial.TrialState.COMPLETE])} complete, "
          f"{len([t for t in s2_study.trials if t.state == optuna.trial.TrialState.PRUNED])} pruned")

    # ══════════════════════════════════════════════════════════════════
    # Save Results
    # ══════════════════════════════════════════════════════════════════
    total_time = time.time() - global_start

    # Build trial histories
    def trial_to_dict(t):
        return {
            "number": t.number,
            "value": round(t.value, 4) if t.value is not None else None,
            "state": t.state.name,
            "params": t.params,
            "user_attrs": t.user_attrs,
            "duration_s": round(t.duration.total_seconds(), 1) if t.duration else None,
        }

    results = {
        "metadata": {
            "script": "hpo_experimental.py",
            "started": datetime.fromtimestamp(global_start).isoformat(),
            "finished": datetime.now().isoformat(),
            "total_runtime_hours": round(total_time / 3600, 2),
            "target_hours": args.hours,
            "hard_limit_hours": args.hard_limit,
            "n_pixels": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_folds": NUM_FOLDS,
        },
        "baseline": {
            "stage1": {"n_estimators": 100, "max_depth": 15},
            "stage2": {"n_estimators": 100, "max_depth": 20, "pred_threshold": 0.80},
        },
        "stage1": {
            "objective": "F1-score (binary change detection)",
            "best_value": round(s1_best.value, 4),
            "best_params": s1_best.params,
            "best_user_attrs": s1_best.user_attrs,
            "n_trials_complete": len([t for t in s1_study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_trials_pruned": len([t for t in s1_study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "runtime_minutes": round(s1_elapsed / 60, 1),
            "trials": [trial_to_dict(t) for t in s1_study.trials],
        },
        "stage2": {
            "objective": "Macro F1-score (end-to-end pipeline)",
            "best_value": round(s2_best.value, 4),
            "best_params": s2_best.params,
            "best_user_attrs": s2_best.user_attrs,
            "n_trials_complete": len([t for t in s2_study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_trials_pruned": len([t for t in s2_study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "runtime_minutes": round(s2_elapsed / 60, 1),
            "trials": [trial_to_dict(t) for t in s2_study.trials],
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ══════════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  HPO COMPLETE")
    print(f"{'='*70}")
    print(f"  Total runtime: {total_time/3600:.2f}h ({total_time/60:.0f} min)")
    print(f"\n  ┌─ Stage 1 (Change Detector) ─────────────────────────┐")
    print(f"  │ Best F1: {s1_best.value:.4f}                          │")
    print(f"  │ Trials:  {len(s1_study.trials):4d} total                         │")
    for k, v in s1_best.params.items():
        print(f"  │   {k}: {v}")
    print(f"  └────────────────────────────────────────────────────┘")
    print(f"\n  ┌─ Stage 2 (Land-Cover Predictor) ──────────────────┐")
    print(f"  │ Best Macro F1: {s2_best.value:.4f}                     │")
    print(f"  │ Trials:  {len(s2_study.trials):4d} total                         │")
    for k, v in s2_best.params.items():
        print(f"  │   {k}: {v}")
    if "mean_accuracy" in s2_best.user_attrs:
        print(f"  │   Overall Accuracy: {s2_best.user_attrs['mean_accuracy']:.4f}")
    print(f"  └────────────────────────────────────────────────────┘")
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Compare with baseline: S1(n=100, d=15), S2(n=100, d=20, t=0.80)")

    # Cancel alarm
    signal.alarm(0)


if __name__ == "__main__":
    main()
