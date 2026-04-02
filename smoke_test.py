"""Smoke test for the student-signal public API.

Runs the full prepare → train → rank flow using demo data.
If this script completes without errors and the output looks sensible,
the library is wired together correctly.

Usage:
    uv run python smoke_test.py
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.svm import SVC

import student_signal
from student_signal.evaluate import load_settings

TRAIN_PATH = Path("data/01-raw/demo/synth_data_train.csv")
PRED_PATH = Path("data/01-raw/demo/synth_data_pred.csv")
TARGET_COL = "Dropout"
ID_COL = "Studentnummer"
SEP = "\t"


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def check(label: str, condition: bool) -> None:
    status = "✓" if condition else "✗ FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise AssertionError(f"Check failed: {label}")


# ── 1. Load demo data ────────────────────────────────────────────
section("1. Load demo data")

train_df = pd.read_csv(TRAIN_PATH, sep=SEP, engine="python")
pred_df = pd.read_csv(PRED_PATH, sep=SEP, engine="python")

print(f"  Train: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
print(f"  Pred:  {pred_df.shape[0]} rows, {pred_df.shape[1]} columns")

check("Train file loaded", len(train_df) > 0)
check("Pred file loaded", len(pred_df) > 0)
check("Target column present in train", TARGET_COL in train_df.columns)
check("ID column present in train", ID_COL in train_df.columns)


# ── 2. Load settings ─────────────────────────────────────────────
section("2. Load settings")

settings = load_settings()
print(f"  Config keys: {list(settings.keys())}")

check("imputation settings present", "imputation" in settings)
check("hyperparameters present", "hyperparameters" in settings)
check("evaluation settings present", "evaluation" in settings)


# ── 3. prepare() ─────────────────────────────────────────────────
section("3. prepare()")

prepared = student_signal.prepare(
    train_df,
    pred_df,
    target_col=TARGET_COL,
    id_col=ID_COL,
)

print(f"  X_train:        {prepared.X_train.shape}")
print(f"  X_train_scaled: {prepared.X_train_scaled.shape}")
print(f"  X_pred:         {prepared.X_pred.shape}")
print(f"  X_pred_scaled:  {prepared.X_pred_scaled.shape}")
print(f"  y_train dropouts: {prepared.y_train.sum()} / {len(prepared.y_train)}")

check("X_train has rows", len(prepared.X_train) > 0)
check("X_pred has rows", len(prepared.X_pred) > 0)
check("Scaler returned", prepared.scaler is not None)
check("Scaled shape matches unscaled", prepared.X_train_scaled.shape == prepared.X_train.shape)
check("No NaN in X_train", prepared.X_train.isna().sum().sum() == 0)
check("No NaN in X_pred", prepared.X_pred.isna().sum().sum() == 0)
check(
    "Scaled values in [0, 1]",
    prepared.X_train_scaled.min().min() >= -0.01
    and prepared.X_train_scaled.max().max() <= 1.01,
)


# ── 4. Train models ──────────────────────────────────────────────
section("4. Train models (small grids for speed)")

hp = settings["hyperparameters"]
seed = hp["random_seed"]

# Use small grids so this runs in seconds
rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=seed)
rf_model.fit(prepared.X_train, prepared.y_train)

lasso_model = Lasso(alpha=0.001, random_state=seed)
lasso_model.fit(prepared.X_train_scaled, prepared.y_train)

svm_model = SVC(kernel="rbf", C=1, probability=True, random_state=seed)
svm_model.fit(prepared.X_train_scaled, prepared.y_train)

print("  Random Forest  trained ✓")
print("  Lasso          trained ✓")
print("  SVM            trained ✓")


# ── 5. rank() ────────────────────────────────────────────────────
section("5. rank()")

rf_ranked = student_signal.rank(rf_model, prepared, use_scaled=False)
lasso_ranked = student_signal.rank(lasso_model, prepared, use_scaled=True)
svm_ranked = student_signal.rank(svm_model, prepared, use_scaled=True)

for name, ranked in [("Random Forest", rf_ranked), ("Lasso", lasso_ranked), ("SVM", svm_ranked)]:
    check(f"{name}: correct row count", len(ranked) == len(pred_df))
    check(f"{name}: has ranking column", "ranking" in ranked.columns)
    check(f"{name}: has ID column", ID_COL in ranked.columns)
    check(f"{name}: ranked from 1", ranked["ranking"].min() == 1)
    check(f"{name}: scores are not all identical", ranked["voorspelling"].nunique() > 1)

print(f"\n  Top 5 students by RF dropout risk:")
print(rf_ranked.head(5).to_string(index=False))


# ── 6. Summary ───────────────────────────────────────────────────
section("Smoke test complete — all checks passed")
print()
print("  Note: evaluate() is intentionally not tested here.")
print("  See KNOWN_ISSUES.md #2 — no train/validation split yet.")
print("  Evaluation metrics are not meaningful until that is fixed.")
print()
