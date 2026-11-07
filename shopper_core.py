"""Code shared by the training script and the Streamlit app.

Keeping the column definitions and metric calculation here means the numbers
in the app are computed in exactly the same way as the numbers in the README.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"
METRICS_JSON = MODEL_DIR / "training_summary.json"

TARGET = "Revenue"

# Counts, durations and web-analytics rates: heavily right-skewed and never negative.
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
# OperatingSystems, Browser, Region and TrafficType are integers in the file, but
# those integers are ID codes with no order, so they are one-hot encoded.
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

_BOOL_MAP = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}


def to_bool(column: pd.Series) -> pd.Series:
    """Read TRUE/FALSE style values, however the CSV tool happened to write them."""
    return column.astype(str).str.strip().str.lower().map(_BOOL_MAP).astype("boolean")


def clean_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Validate a frame against the training schema.

    Returns the cleaned frame and an empty list, or None and a list of problems
    written for the person who uploaded the file.
    """
    raw = raw.rename(columns=lambda c: str(c).strip())
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        return None, [f"Missing column(s): {', '.join(missing)}"]
    if raw.empty:
        return None, ["The file has a header but no data rows."]

    frame = raw[required].copy()
    problems = []

    for col in (TARGET, "Weekend"):
        frame[col] = to_bool(frame[col])
        if frame[col].isna().any():
            problems.append(f"{col}: {int(frame[col].isna().sum())} value(s) are not TRUE/FALSE.")

    for col in NUMERIC_FEATURES + ["OperatingSystems", "Browser", "Region", "TrafficType"]:
        converted = pd.to_numeric(frame[col], errors="coerce")
        if converted.isna().any():
            problems.append(f"{col}: {int(converted.isna().sum())} value(s) are blank or not numeric.")
        elif col in NUMERIC_FEATURES and (converted < 0).any():
            problems.append(f"{col}: contains negative values, which this feature cannot have.")
        frame[col] = converted

    for col in ("Month", "VisitorType"):
        if frame[col].isna().any():
            problems.append(f"{col}: {int(frame[col].isna().sum())} value(s) are blank.")
        frame[col] = frame[col].astype(str).str.strip()

    if problems:
        return None, problems

    frame["Weekend"] = frame["Weekend"].astype(bool)
    frame[TARGET] = frame[TARGET].astype(bool)
    return frame, []


def ranking_score(model, X: pd.DataFrame) -> np.ndarray:
    """Continuous score for AUC. SVC is not probability-calibrated, so it uses its margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float | None]:
    """The six metrics required by the brief. 'Purchase' is the positive class."""
    # AUC is undefined when the file holds only one class, so it is reported as None.
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else None
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": None if auc is None else float(auc),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }
