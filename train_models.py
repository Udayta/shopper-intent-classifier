"""Train the five classifiers and save everything the Streamlit app needs.

Run from the project folder:  python train_models.py

Produces:
    model/*.joblib                 one fitted pipeline per classifier
    model/training_summary.json    chosen hyperparameters and test-set metrics
    test_data.csv                  the held-out rows, target column included
"""
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from shopper_core import (
    CATEGORICAL_FEATURES,
    FEATURES,
    METRIC_NAMES,
    METRICS_JSON,
    MODEL_DIR,
    MODEL_FILES,
    NUMERIC_FEATURES,
    ROOT,
    TARGET,
    TEST_CSV,
    clean_frame,
    compute_metrics,
    ranking_score,
)

RAW_CSV = ROOT / "data" / "online_shoppers_intention.csv"
SEED = 2026
TEST_SHARE = 0.20


def build_preprocessor() -> ColumnTransformer:
    # log1p tames the long right tails before scaling, which matters most for the
    # distance-based models (KNN, SVM) and for Gaussian Naive Bayes.
    numeric = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())
    # Unseen categories in an uploaded file become all zeros rather than an error.
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


# Weighted and unweighted classes are both searched because only about 15% of
# sessions end in a purchase.
SEARCH_SPACE = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, random_state=SEED),
        {"clf__C": [0.001, 0.003, 0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=SEED),
        {
            "clf__max_depth": [1, 2, 3, 4, 6, 8, 12, None],
            "clf__min_samples_leaf": [1, 10, 30],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [5, 11, 21, 41], "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 0.3, 1.0, 3.0]},
    ),
    "Support Vector Machine": (
        SVC(kernel="rbf", random_state=SEED),
        {
            # An earlier grid with C=10 and gamma=0.03 took 54 minutes, so those are left out.
            "clf__C": [0.1, 1, 3],
            "clf__gamma": ["scale", 0.01, 0.003, 0.001],
            "clf__class_weight": [None, "balanced"],
        },
    ),
}


def pick_best_candidate(cv_results: dict) -> int:
    """Highest mean CV F1; candidates tied at 4 decimals are separated by mean CV AUC."""
    f1 = np.round(cv_results["mean_test_f1"], 4)
    auc = cv_results["mean_test_roc_auc"]
    return int(np.lexsort((-auc, -f1))[0])


def main() -> None:
    raw = pd.read_csv(RAW_CSV)
    data, problems = clean_frame(raw)
    if problems:
        raise SystemExit("Raw data failed validation: " + "; ".join(problems))

    # Duplicates are removed before splitting, otherwise one copy could land in
    # training and its twin in the test set, which would inflate the test scores.
    rows_before = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    print(f"Rows: {rows_before} -> {len(data)} after removing duplicates")

    train, test = train_test_split(data, test_size=TEST_SHARE, stratify=data[TARGET], random_state=SEED)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    print(f"Train: {len(train)}  Test: {len(test)}  Purchase rate in test: {y_test.mean():.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    test.to_csv(TEST_CSV, index=False)

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    results = {}
    for name, (estimator, grid) in SEARCH_SPACE.items():
        started = time.perf_counter()
        pipeline = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        # Tuned on training folds only, so the test set stays unseen until the end.
        search = GridSearchCV(
            pipeline, grid, scoring=["f1", "roc_auc"], refit=pick_best_candidate, cv=folds, n_jobs=-1
        )
        search.fit(X_train, y_train)

        best = search.best_estimator_
        joblib.dump(best, MODEL_DIR / MODEL_FILES[name], compress=3)

        y_pred = best.predict(X_test)
        results[name] = {
            "best_params": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
            "cv_f1": float(search.cv_results_["mean_test_f1"][search.best_index_]),
            "cv_auc": float(search.cv_results_["mean_test_roc_auc"][search.best_index_]),
            "train_f1": float(compute_metrics(y_train, best.predict(X_train), ranking_score(best, X_train))["F1"]),
            "test": compute_metrics(y_test, y_pred, ranking_score(best, X_test)),
        }
        print(f"{name}: done in {time.perf_counter() - started:.1f}s, best {results[name]['best_params']}", flush=True)

    summary = {
        "seed": SEED,
        "rows_after_dedup": len(data),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_purchase_rate": float(y_test.mean()),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "models": results,
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table = pd.DataFrame({n: r["test"] for n, r in results.items()}).T[METRIC_NAMES]
    print("\nHeld-out test set results")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
