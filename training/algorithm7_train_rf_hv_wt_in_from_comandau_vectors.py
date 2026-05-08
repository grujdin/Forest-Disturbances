"""
Algorithm #7. Train Random Forest classifier for HV / WT / IN from Algorithm #6 vectors.

Purpose
-------
This script trains a Random Forest model from the EO training vectors generated
by Algorithm #6. It is designed for the Comandau training database workflow.

It can run while labelling is still in progress. If one of the target classes is
missing, the script writes a warning and trains a draft model only on the classes
present, provided that at least two classes are available.

Main outputs
------------
- rf_hv_wt_in_model.joblib
- classification_report_holdout.txt
- confusion_matrix_holdout.csv
- feature_importances.csv
- training_vectors_with_rf_predictions.csv
- training_label_counts.csv
- rf_training_metadata.json
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# =============================================================================
# HARD-CODED CONFIG
# =============================================================================

TRAINING_VECTOR_CSV = Path("D:/Forest_Disturbance/outputs/rf_training_vectors_comandau_algorithm6/eo_rf_training_vectors_comandau.csv")
OUTPUT_ROOT = Path("D:/Forest_Disturbance/outputs/rf_model_hv_wt_in_comandau_algorithm7")

TARGET_FIELD = "rf_label"
TARGET_CLASSES = ["HV", "WT", "IN"]
ALLOW_PARTIAL_CLASSES_FOR_DRAFT_MODEL = True
MIN_CLASSES_TO_TRAIN = 2
MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT = 2

# Exclude identifiers, labels, diagnostics, geometry fields, and leakage-prone fields.
EXCLUDE_EXACT = {
    "training_object_id",
    "source_layer",
    "source_id",
    "pre_date",
    "post_date",
    "interval",
    "label_raw",
    "label_status",
    "rf_label",
    "geometry",
}
EXCLUDE_PREFIXES = (
    "m_eo_s2_pre_scene",
    "m_eo_s2_post_scene",
    "m_eo_s1_pre_scene",
    "m_eo_s1_post_scene",
    "m_semantic_pre_date",
    "m_semantic_post_date",
)
# Keep missingness numeric fields such as m_*_valid_frac and m_*_offset_days.

RANDOM_STATE = 42
N_ESTIMATORS = 600
MAX_FEATURES = "sqrt"
MIN_SAMPLES_LEAF = 2
N_JOBS = -1
TEST_SIZE = 0.25
CV_FOLDS_MAX = 5

# =============================================================================
# Helpers
# =============================================================================

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def is_excluded_feature(col: str) -> bool:
    if col in EXCLUDE_EXACT:
        return True
    return any(col.startswith(p) for p in EXCLUDE_PREFIXES)


def select_features(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    candidate_cols = [c for c in df.columns if not is_excluded_feature(c)]
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for c in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            # Exclude columns with all missing values.
            if df[c].notna().sum() > 0:
                numeric_cols.append(c)
        else:
            # Keep only low-cardinality useful categoricals, if any.
            nunique = df[c].astype(str).nunique(dropna=True)
            if 1 < nunique <= 30:
                categorical_cols.append(c)
    return candidate_cols, numeric_cols, categorical_cols


def probability_frame(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(X)
    classes = list(model.named_steps["rf"].classes_)
    out = pd.DataFrame(index=X.index)
    for i, cls in enumerate(classes):
        out[f"rf_probability_{cls}"] = proba[:, i]
    out["rf_predicted_class"] = model.predict(X)
    sorted_proba = np.sort(proba, axis=1)
    out["rf_probability_max"] = sorted_proba[:, -1]
    out["rf_probability_margin"] = sorted_proba[:, -1] - sorted_proba[:, -2] if proba.shape[1] >= 2 else np.nan
    return out

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log("Starting Algorithm #7: RF training for HV / WT / IN")

    if not TRAINING_VECTOR_CSV.exists():
        raise FileNotFoundError(f"Training-vector CSV not found: {TRAINING_VECTOR_CSV}. Run Algorithm #6 first.")

    df = pd.read_csv(TRAINING_VECTOR_CSV)
    if TARGET_FIELD not in df.columns:
        raise ValueError(f"Missing target field {TARGET_FIELD!r} in {TRAINING_VECTOR_CSV}")

    df = df[df[TARGET_FIELD].isin(TARGET_CLASSES)].copy()
    label_counts = df[TARGET_FIELD].value_counts().rename_axis("rf_label").reset_index(name="n_samples")
    label_counts.to_csv(OUTPUT_ROOT / "training_label_counts.csv", index=False)
    present_classes = sorted(df[TARGET_FIELD].unique().tolist())
    missing_classes = [c for c in TARGET_CLASSES if c not in present_classes]

    if len(present_classes) < MIN_CLASSES_TO_TRAIN:
        msg = (
            f"Not enough classes to train. Present classes: {present_classes}; required at least {MIN_CLASSES_TO_TRAIN}. "
            "Continue labelling the GDB and rerun Algorithm #6/#7."
        )
        (OUTPUT_ROOT / "rf_training_not_run.txt").write_text(msg, encoding="utf-8")
        raise RuntimeError(msg)

    if missing_classes:
        warning = (
            f"WARNING: missing target classes in the current training vectors: {missing_classes}. "
            f"Present classes: {present_classes}. "
            "A draft model will be trained only on the present classes because "
            f"ALLOW_PARTIAL_CLASSES_FOR_DRAFT_MODEL={ALLOW_PARTIAL_CLASSES_FOR_DRAFT_MODEL}."
        )
        (OUTPUT_ROOT / "missing_class_warning.txt").write_text(warning, encoding="utf-8")
        log(warning)
        if not ALLOW_PARTIAL_CLASSES_FOR_DRAFT_MODEL:
            raise RuntimeError(warning)

    _, numeric_cols, categorical_cols = select_features(df)
    if not numeric_cols and not categorical_cols:
        raise RuntimeError("No usable predictor columns found.")

    X = df[numeric_cols + categorical_cols].copy()
    y = df[TARGET_FIELD].copy()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        max_features=MAX_FEATURES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=N_JOBS,
        oob_score=True,
    )
    model = Pipeline(steps=[("preprocess", preprocess), ("rf", rf)])

    # Holdout only if every class has enough samples.
    min_count = y.value_counts().min()
    holdout_possible = len(present_classes) >= 2 and min_count >= MIN_SAMPLES_PER_CLASS_FOR_HOLDOUT
    if holdout_possible:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, X.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, labels=present_classes, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=present_classes)
        pd.DataFrame(cm, index=[f"true_{c}" for c in present_classes], columns=[f"pred_{c}" for c in present_classes]).to_csv(
            OUTPUT_ROOT / "confusion_matrix_holdout.csv"
        )
        holdout_summary = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        (OUTPUT_ROOT / "classification_report_holdout.txt").write_text(report, encoding="utf-8")
        pd.DataFrame([holdout_summary]).to_csv(OUTPUT_ROOT / "holdout_summary.csv", index=False)
    else:
        model.fit(X, y)
        holdout_summary = {"holdout_used": False, "reason": "too few samples per class"}
        (OUTPUT_ROOT / "classification_report_holdout.txt").write_text("Holdout split not used: too few samples per class.\n", encoding="utf-8")

    # Cross-validation, when possible.
    cv_scores = []
    if len(present_classes) >= 2 and min_count >= 2:
        n_splits = min(CV_FOLDS_MAX, int(min_count))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(model, X, y, cv=cv, scoring="balanced_accuracy", n_jobs=N_JOBS)
            cv_scores = scores.tolist()
            pd.DataFrame({"fold": np.arange(1, len(scores) + 1), "balanced_accuracy": scores}).to_csv(
                OUTPUT_ROOT / "cross_validation_scores.csv", index=False
            )

    # Fit final model on all currently labelled data.
    final_model = Pipeline(steps=[("preprocess", preprocess), ("rf", rf)])
    final_model.fit(X, y)
    joblib.dump(final_model, OUTPUT_ROOT / "rf_hv_wt_in_model.joblib")

    # Predictions on training vectors for QA.
    pred_df = probability_frame(final_model, X)
    out_pred = pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    out_pred.to_csv(OUTPUT_ROOT / "training_vectors_with_rf_predictions.csv", index=False)

    # Feature importances.
    try:
        feature_names = list(final_model.named_steps["preprocess"].get_feature_names_out())
        importances = final_model.named_steps["rf"].feature_importances_
        pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False).to_csv(
            OUTPUT_ROOT / "feature_importances.csv", index=False
        )
    except Exception as exc:
        (OUTPUT_ROOT / "feature_importance_warning.txt").write_text(repr(exc), encoding="utf-8")

    metadata = {
        "training_vector_csv": str(TRAINING_VECTOR_CSV),
        "target_field": TARGET_FIELD,
        "target_classes_requested": TARGET_CLASSES,
        "present_classes": present_classes,
        "missing_classes": missing_classes,
        "n_samples": int(len(df)),
        "n_numeric_features": len(numeric_cols),
        "n_categorical_features": len(categorical_cols),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "max_features": MAX_FEATURES,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "class_weight": "balanced_subsample",
        "holdout_summary": holdout_summary,
        "cross_validation_balanced_accuracy": cv_scores,
        "oob_score": float(getattr(final_model.named_steps["rf"], "oob_score_", np.nan)),
        "draft_model_due_to_missing_classes": bool(missing_classes),
    }
    (OUTPUT_ROOT / "rf_training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log(f"Done. RF outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
