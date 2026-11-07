"""Streamlit front end: scores an uploaded test CSV with the five saved models.

Nothing is trained here. Each model is loaded from model/ and only ever calls
predict / predict_proba / decision_function on the uploaded rows.
"""
import io

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from data_explorer import render_data_explorer
from shopper_core import (
    FEATURES,
    METRIC_NAMES,
    MODEL_DIR,
    MODEL_FILES,
    ROOT,
    TARGET,
    TEST_CSV,
    clean_frame,
    compute_metrics,
    ranking_score,
)

MUTED_BAR = "#cbd5e1"
# One fixed colour per model, so a model keeps its colour wherever it appears.
MODEL_COLORS = {
    "Logistic Regression": "#2563eb",
    "Decision Tree": "#ea580c",
    "KNN": "#0d9488",
    "Naive Bayes": "#ca8a04",
    "Support Vector Machine": "#db2777",
}
MODEL_ICONS = {
    "Logistic Regression": "📈",
    "Decision Tree": "🌳",
    "KNN": "📍",
    "Naive Bayes": "🎲",
    "Support Vector Machine": "🧭",
}
CLASS_LABELS = ["No purchase", "Purchase"]

METRIC_HELP = {
    "Accuracy": "Share of all sessions classified correctly. Flattering here, because most sessions do not buy.",
    "AUC": "How well the model ranks buyers above non-buyers, across every possible cut-off.",
    "Precision": "Of the sessions flagged as purchases, the share that really purchased.",
    "Recall": "Of the sessions that really purchased, the share the model caught.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Correlation between predicted and actual classes, from -1 to 1. Robust to class imbalance.",
}

st.set_page_config(page_title="Shopper Intent Lab", page_icon="🛒", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading saved models...")
def load_models() -> dict:
    return {name: joblib.load(MODEL_DIR / filename) for name, filename in MODEL_FILES.items()}


@st.cache_data(show_spinner="Scoring the file with all five models...")
def score_file(csv_bytes: bytes) -> tuple[pd.DataFrame | None, dict, list[str], dict]:
    """Run every saved model over the file. Cached on the file's contents."""
    try:
        raw = pd.read_csv(io.BytesIO(csv_bytes))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        return None, {}, [f"This file could not be read as a CSV ({error})."], {}

    info = {"extra_columns": [str(c).strip() for c in raw.columns if str(c).strip() not in FEATURES + [TARGET]]}
    frame, problems = clean_frame(raw)
    if problems:
        return None, {}, problems, info

    y_true = frame[TARGET]
    scored = {}
    for name, model in load_models().items():
        y_pred = model.predict(frame[FEATURES])
        y_score = ranking_score(model, frame[FEATURES])
        scored[name] = {
            "y_pred": y_pred,
            "y_score": y_score,
            "metrics": compute_metrics(y_true, y_pred, y_score),
        }
    return frame, scored, [], info


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def confusion_chart(y_true, y_pred) -> alt.LayerChart:
    matrix = confusion_matrix(y_true, y_pred, labels=[False, True])
    cells = pd.DataFrame(
        [
            {"Actual": CLASS_LABELS[a], "Predicted": CLASS_LABELS[p], "Sessions": int(matrix[a, p])}
            for a in range(2)
            for p in range(2)
        ]
    )
    cells["Share"] = cells["Sessions"] / max(cells["Sessions"].sum(), 1)
    # Ink flips to white on the darker half of the scale so counts stay readable.
    cells["dark_cell"] = cells["Sessions"] > cells["Sessions"].max() / 2

    base = alt.Chart(cells).encode(
        x=alt.X("Predicted:N", sort=CLASS_LABELS, axis=alt.Axis(labelAngle=0, orient="top", ticks=False, domain=False)),
        y=alt.Y("Actual:N", sort=CLASS_LABELS, axis=alt.Axis(ticks=False, domain=False)),
        tooltip=["Actual", "Predicted", "Sessions", alt.Tooltip("Share:Q", format=".1%")],
    )
    tiles = base.mark_rect(cornerRadius=4, stroke="transparent", strokeWidth=2).encode(
        color=alt.Color("Sessions:Q", scale=alt.Scale(range=["#ede9fe", "#5b21b6"]), legend=None)
    )
    counts = base.mark_text(fontSize=22, fontWeight="bold").encode(
        text="Sessions:Q",
        color=alt.condition("datum.dark_cell", alt.value("#ffffff"), alt.value("#0b0b0b")),
    )
    return (tiles + counts).properties(height=300)


def gap_to_best(table: pd.DataFrame, model: str, metric: str) -> str | None:
    """Delta text for a metric tile: how far this model is from the best of the five."""
    value, best = table.loc[model, metric], table[metric].max()
    if pd.isna(value):
        return None
    return "Best of 5" if value >= best else f"{value - best:+.4f}"


def roc_chart(y_true, scored: dict, selected: str) -> alt.Chart:
    curves = []
    for name, result in scored.items():
        fpr, tpr, _ = roc_curve(y_true, result["y_score"])
        auc = result["metrics"]["AUC"]
        curves.append(pd.DataFrame({"Model": name, "Legend": f"{name} (AUC {auc:.3f})", "FPR": fpr, "TPR": tpr}))
    data = pd.concat(curves, ignore_index=True)
    data["is_selected"] = data["Model"] == selected
    legend_order = [f"{n} (AUC {scored[n]['metrics']['AUC']:.3f})" for n in MODEL_COLORS]

    chance = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], color=MUTED_BAR, strokeWidth=1
    ).encode(x="x:Q", y="y:Q")
    lines = alt.Chart(data).mark_line(interpolate="step-after").encode(
        x=alt.X("FPR:Q", title="False positive rate (non-buyers wrongly flagged)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("TPR:Q", title="True positive rate (buyers caught)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "Legend:N",
            sort=legend_order,
            scale=alt.Scale(domain=legend_order, range=list(MODEL_COLORS.values())),
            legend=alt.Legend(title=None, orient="bottom", columns=2, labelLimit=260),
        ),
        strokeWidth=alt.condition("datum.is_selected", alt.value(3), alt.value(1.5)),
        detail="Model:N",
        tooltip=["Model", alt.Tooltip("FPR:Q", format=".3f"), alt.Tooltip("TPR:Q", format=".3f")],
    )
    return (chance + lines).properties(height=420)


def metric_bars(table: pd.DataFrame, metric: str, selected: str) -> alt.LayerChart:
    data = table[[metric]].rename(columns={metric: "value"}).dropna().reset_index(names="Model")
    data["is_selected"] = data["Model"] == selected
    lower = min(0.0, float(data["value"].min())) if len(data) else 0.0

    base = alt.Chart(data, title=metric).encode(
        y=alt.Y("Model:N", sort=list(MODEL_FILES), title=None, axis=alt.Axis(ticks=False, domain=False, labelLimit=170)),
        x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[lower, 1.15]), axis=None),
        tooltip=["Model", alt.Tooltip("value:Q", title=metric, format=".4f")],
    )
    bars = base.mark_bar(size=14, cornerRadiusEnd=4).encode(
        color=alt.condition("datum.is_selected", alt.value(MODEL_COLORS[selected]), alt.value(MUTED_BAR))
    )
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("value:Q", format=".3f"))
    return (bars + labels).properties(height=170)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Choose the data")
    upload = st.file_uploader("Upload a test CSV", type="csv", help="Must hold all 17 input features and the Revenue column.")
    st.download_button(
        "Download sample test_data.csv",
        data=TEST_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        width="stretch",
    )
    st.caption("No file uploaded? The app uses the held-out test set from the repository.")

    st.header("2. Choose a model")
    selected = st.selectbox("Model", list(MODEL_FILES), format_func=lambda m: f"{MODEL_ICONS[m]}  {m}", key="model_choice")

    with st.expander("Required columns"):
        st.write(", ".join(f"`{c}`" for c in FEATURES + [TARGET]))

# ---------------------------------------------------------------- main page
st.markdown(
    """
    <div class="hero">
      <div class="kicker">UCI · Online Shoppers Purchasing Intention</div>
      <h1>🛒 Shopper Intent Lab</h1>
      <p>Will an online shopping session end in a purchase? Upload a test CSV and see how five
      classifiers, trained in advance, score it. The models are never retrained on your data.</p>
      <div class="chips"><span>5 saved models</span><span>6 evaluation metrics</span>
      <span>Confusion matrix &amp; report</span><span>Data explorer</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

csv_bytes = upload.getvalue() if upload is not None else TEST_CSV.read_bytes()
source_name = upload.name if upload is not None else "test_data.csv (bundled)"
frame, scored, problems, file_info = score_file(csv_bytes)

if problems:
    st.error("**This file can't be scored yet. Please fix the following and upload it again:**")
    for problem in problems:
        st.markdown(f"- {problem}")
    st.stop()

y_true = frame[TARGET]
purchase_rate = y_true.mean()
st.caption(f"Data: **{source_name}** · {len(frame):,} sessions · {purchase_rate:.1%} ended in a purchase")
if y_true.nunique() < 2:
    st.warning("This file contains only one class, so AUC can't be calculated and the other metrics are less meaningful.")

result = scored[selected]
comparison = pd.DataFrame({name: s["metrics"] for name, s in scored.items()}).T[METRIC_NAMES].astype(float)

tab_model, tab_compare, tab_data, tab_rows = st.tabs(
    ["🎯 Selected model", "🏆 All five compared", "📊 Data explorer", "🔍 Row-level predictions"]
)

with tab_model:
    st.subheader(f"{MODEL_ICONS[selected]} {selected}")
    for column, metric in zip(st.columns(6), METRIC_NAMES):
        with column, st.container(key=f"tile-{metric}"):
            st.metric(
                metric,
                format_metric(result["metrics"][metric]),
                delta=gap_to_best(comparison, selected, metric),
                delta_color="off",
                help=METRIC_HELP[metric],
            )
    st.caption("Under each value: the gap to the best of the five models on this file.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Confusion matrix")
        st.altair_chart(confusion_chart(y_true, result["y_pred"]), width="stretch")
    with right:
        st.markdown("##### Classification report")
        report = classification_report(
            y_true,
            result["y_pred"],
            labels=[False, True],
            target_names=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        )
        report_table = pd.DataFrame(report).T.rename(columns=str.capitalize)
        # sklearn returns accuracy as a single number, which pandas repeats down the
        # whole row. Keep it in the F1 column only, as sklearn's text report does.
        report_table.loc["accuracy", ["Precision", "Recall"]] = float("nan")
        report_table.loc["accuracy", "Support"] = len(y_true)
        # Formatted as text because st.dataframe displays empty number cells as "None".
        for column in ("Precision", "Recall", "F1-score"):
            report_table[column] = report_table[column].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        report_table["Support"] = report_table["Support"].map(lambda v: f"{int(v):,}")
        st.dataframe(report_table, width="stretch")

with tab_compare:
    st.subheader("How the five models score on this file")
    st.dataframe(
        comparison.style.format("{:.4f}", na_rep="n/a").highlight_max(axis=0, props="font-weight: bold"),
        width="stretch",
    )
    st.caption(f"Bold marks the best model for each metric. In the charts below, the coloured bar is {selected}.")

    for start in (0, 3):
        for column, metric in zip(st.columns(3), METRIC_NAMES[start : start + 3]):
            column.altair_chart(metric_bars(comparison, metric, selected), width="stretch")

    st.markdown("##### ROC curves")
    if y_true.nunique() < 2:
        st.info("ROC curves need both classes in the file, so they are not shown for this upload.")
    else:
        st.caption(
            f"Each curve shows the trade-off as the decision threshold moves. A curve closer to the top-left corner "
            f"ranks buyers better; the dashed line is random guessing. {selected} is drawn thicker."
        )
        st.altair_chart(roc_chart(y_true, scored, selected), width="stretch")

with tab_data:
    render_data_explorer(frame, file_info)

with tab_rows:
    st.subheader(f"What {selected} predicted for each session")
    rows = frame[FEATURES].copy()
    rows.insert(0, "Actual", y_true.map({True: "Purchase", False: "No purchase"}))
    rows.insert(1, "Predicted", pd.Series(result["y_pred"], index=frame.index).map({True: "Purchase", False: "No purchase"}))
    rows.insert(2, "Correct", rows["Actual"] == rows["Predicted"])
    # SVC was trained without probability calibration, so it only offers its margin.
    has_proba = hasattr(load_models()[selected], "predict_proba")
    score_column = "Purchase probability" if has_proba else "Decision score"
    rows.insert(3, score_column, result["y_score"])
    only_errors = st.toggle("Show only the sessions it got wrong")
    shown = rows[~rows["Correct"]] if only_errors else rows
    st.caption(f"Showing {len(shown):,} of {len(rows):,} sessions")
    if has_proba:
        st.caption("Purchase probability: the model's estimated chance that the session ends in a purchase (0 to 1).")
    else:
        st.caption(
            "Decision score: this SVM gives no probabilities, only a distance from its decision boundary. "
            "Above 0 means it predicts a purchase; the further from 0, the more confident it is."
        )
    st.dataframe(
        shown,
        width="stretch",
        height=420,
        hide_index=True,
        column_config={
            score_column: st.column_config.NumberColumn(format="%.3f" if has_proba else "%+.3f"),
        },
    )
    st.download_button(
        "Download these predictions as CSV",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{selected.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
