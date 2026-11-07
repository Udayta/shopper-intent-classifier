"""The Data explorer tab: describes whatever CSV the app is currently scoring."""
import html

import altair as alt
import pandas as pd
import streamlit as st

from shopper_core import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

CLASS_COLORS = {"No purchase": "#94a3b8", "Purchase": "#db2777"}
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Meanings follow the UCI dataset documentation.
COLUMN_INFO = {
    "Administrative": "Number of account-management pages visited (login, profile, settings).",
    "Administrative_Duration": "Total time spent on account-management pages, in seconds.",
    "Informational": "Number of information pages visited (about us, contact, delivery info).",
    "Informational_Duration": "Total time spent on information pages, in seconds.",
    "ProductRelated": "Number of product pages visited.",
    "ProductRelated_Duration": "Total time spent on product pages, in seconds.",
    "BounceRates": "Average bounce rate of the pages visited: share of visitors who land on a page and leave at once.",
    "ExitRates": "Average exit rate of the pages visited: share of page views that were the last of a session.",
    "PageValues": "Average value of the pages visited before an online purchase (Google Analytics page value).",
    "SpecialDay": "How close the visit was to a special day such as Mother's Day or Valentine's Day (0 to 1).",
    "Month": "Month of the visit. The dataset has no January or April sessions.",
    "OperatingSystems": "Operating system of the visitor, as an anonymised code.",
    "Browser": "Browser of the visitor, as an anonymised code.",
    "Region": "Region the visitor came from, as an anonymised code.",
    "TrafficType": "How the visitor arrived (banner, SMS, direct and so on), as an anonymised code.",
    "VisitorType": "Returning visitor, new visitor, or other.",
    "Weekend": "TRUE if the visit happened on a weekend.",
    "Revenue": "Target column: TRUE if the session ended in a purchase.",
}


def stat_cards(cards: list[tuple[str, str, str, str]]) -> None:
    """cards: (label, value, note, colour). Rendered as one responsive HTML row."""
    items = "".join(
        f'<div class="stat" style="--c:{color}"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div><div class="note">{html.escape(note)}</div></div>'
        for label, value, note, color in cards
    )
    st.markdown(f'<div class="stat-row">{items}</div>', unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def column_dictionary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURES + [TARGET]:
        values = frame[col]
        if col in NUMERIC_FEATURES:
            kind = "Numeric"
            summary = f"min {values.min():,.4g} · median {values.median():,.4g} · max {values.max():,.4g}"
        else:
            kind = "Target" if col == TARGET else "Categorical"
            top = values.astype(str).value_counts(normalize=True).head(3)
            summary = ", ".join(f"{k} ({v:.0%})" for k, v in top.items())
        rows.append(
            {
                "Column": col,
                "Type": kind,
                "Meaning": COLUMN_INFO[col],
                "Distinct values": int(values.nunique()),
                "Summary": summary,
            }
        )
    return pd.DataFrame(rows)


def quick_insights(frame: pd.DataFrame) -> list[str]:
    rate = frame[TARGET].mean()
    notes = [f"**{rate:.1%}** of sessions in this file ended in a purchase."]

    numeric = frame[NUMERIC_FEATURES].assign(**{TARGET: frame[TARGET].astype(int)})
    if frame[TARGET].nunique() > 1:
        corr = numeric.corr()[TARGET].drop(TARGET).dropna()
        if len(corr):
            top = corr.abs().idxmax()
            notes.append(f"**{top}** has the strongest link with buying (correlation {corr[top]:+.2f}).")

    if (frame["PageValues"] > 0).any() and (frame["PageValues"] == 0).any():
        with_value = frame.loc[frame["PageValues"] > 0, TARGET].mean()
        without = frame.loc[frame["PageValues"] == 0, TARGET].mean()
        notes.append(
            f"Sessions with a page value above 0 bought **{with_value:.1%}** of the time, "
            f"against **{without:.1%}** for sessions with no page value."
        )

    by_month = frame.groupby("Month")[TARGET].agg(["mean", "size"])
    by_month = by_month[by_month["size"] >= 30]
    if len(by_month) > 1:
        best = by_month["mean"].idxmax()
        notes.append(f"**{best}** has the highest purchase rate among months with at least 30 sessions ({by_month.loc[best, 'mean']:.1%}).")

    by_visitor = frame.groupby("VisitorType")[TARGET].mean()
    if {"New_Visitor", "Returning_Visitor"} <= set(by_visitor.index):
        notes.append(
            f"New visitors bought **{by_visitor['New_Visitor']:.1%}** of the time, "
            f"returning visitors **{by_visitor['Returning_Visitor']:.1%}**."
        )
    return notes


def class_balance_chart(outcome: pd.Series) -> alt.LayerChart:
    counts = outcome.value_counts().reindex(list(CLASS_COLORS), fill_value=0).rename_axis("Outcome").reset_index(name="Sessions")
    counts["Share"] = counts["Sessions"] / max(counts["Sessions"].sum(), 1)
    base = alt.Chart(counts).encode(
        y=alt.Y("Outcome:N", sort=list(CLASS_COLORS), title=None, axis=alt.Axis(ticks=False, domain=False)),
        x=alt.X("Sessions:Q", axis=None, scale=alt.Scale(domain=[0, counts["Sessions"].max() * 1.25 or 1])),
        tooltip=["Outcome", "Sessions", alt.Tooltip("Share:Q", format=".1%")],
    )
    bars = base.mark_bar(size=34, cornerRadiusEnd=4).encode(
        color=alt.Color("Outcome:N", scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())), legend=None)
    )
    labels = base.mark_text(align="left", dx=6, fontWeight="bold").encode(
        text=alt.Text("label:N")
    ).transform_calculate(label="format(datum.Sessions, ',') + '  (' + format(datum.Share, '.1%') + ')'")
    return (bars + labels).properties(height=150)


def numeric_distribution(data: pd.DataFrame, col: str, trim: bool, hide_zeros: bool) -> alt.Chart:
    if hide_zeros:
        data = data[data[col] != 0]
    if trim and data[col].nunique() > 1:
        data = data[data[col] <= data[col].quantile(0.99)]
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(field=col, type="quantitative", bin=alt.Bin(maxbins=40), title=col),
            y=alt.Y("count():Q", title="Sessions", stack=True),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["Outcome", alt.Tooltip("count():Q", title="Sessions")],
        )
        .properties(height=320)
    )


def category_charts(data: pd.DataFrame, col: str) -> tuple[alt.Chart, alt.Chart]:
    grouped = data.groupby(col).agg(Sessions=(TARGET, "size"), Rate=(TARGET, "mean")).reset_index()
    grouped[col] = grouped[col].astype(str)
    if col == "Month":
        order = [m for m in MONTH_ORDER if m in set(grouped[col])] + sorted(set(grouped[col]) - set(MONTH_ORDER))
    elif col in ("OperatingSystems", "Browser", "Region", "TrafficType"):
        order = sorted(grouped[col], key=lambda v: float(v))
    else:
        order = grouped.sort_values("Sessions", ascending=False)[col].tolist()

    stacked = data.assign(**{col: data[col].astype(str)})
    sessions = (
        alt.Chart(stacked, title="Sessions per value")
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count():Q", title="Sessions"),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[col, "Outcome", alt.Tooltip("count():Q", title="Sessions")],
        )
        .properties(height=300)
    )

    overall = data[TARGET].mean()
    rate_bars = (
        alt.Chart(grouped, title="Purchase rate per value")
        .mark_bar(color="#7c3aed", cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(f"{col}:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Rate:Q", title="Purchase rate", axis=alt.Axis(format="%")),
            tooltip=[col, alt.Tooltip("Rate:Q", format=".1%"), "Sessions"],
        )
    )
    average = (
        alt.Chart(pd.DataFrame({"Rate": [overall]}))
        .mark_rule(color="#db2777", strokeDash=[5, 4], strokeWidth=2)
        .encode(y="Rate:Q", tooltip=[alt.Tooltip("Rate:Q", title="Overall rate", format=".1%")])
    )
    return sessions, (rate_bars + average).properties(height=300)


def correlation_heatmap(frame: pd.DataFrame) -> alt.LayerChart:
    numeric = frame[NUMERIC_FEATURES].assign(**{TARGET: frame[TARGET].astype(int)})
    corr = numeric.corr().round(2)
    order = list(corr.columns)
    long = corr.reset_index(names="A").melt(id_vars="A", var_name="B", value_name="r").dropna()
    long["strong"] = long["r"].abs() >= 0.6

    base = alt.Chart(long).encode(
        x=alt.X(
            "B:N", sort=order, title=None,
            axis=alt.Axis(labelAngle=-40, labelOverlap=False, labelLimit=200, ticks=False, domain=False),
        ),
        y=alt.Y("A:N", sort=order, title=None, axis=alt.Axis(labelLimit=200, ticks=False, domain=False)),
        tooltip=[alt.Tooltip("A:N", title="Feature"), alt.Tooltip("B:N", title="With"), alt.Tooltip("r:Q", format="+.2f")],
    )
    cells = base.mark_rect(stroke="#ffffff", strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color(
            "r:Q",
            scale=alt.Scale(domain=[-1, 0, 1], range=["#2563eb", "#f3f1f8", "#db2777"]),
            legend=alt.Legend(title="Correlation", orient="right"),
        )
    )
    text = base.mark_text(fontSize=10).encode(
        text=alt.Text("r:Q", format=".2f"),
        color=alt.condition("datum.strong", alt.value("#ffffff"), alt.value("#1e1b3a")),
    )
    return (cells + text).properties(height=460)


def render_data_explorer(frame: pd.DataFrame, info: dict) -> None:
    outcome = frame[TARGET].map({True: "Purchase", False: "No purchase"})
    data = frame.assign(Outcome=outcome)
    extras = info.get("extra_columns", [])

    st.subheader("What's inside this CSV")
    stat_cards(
        [
            ("Sessions (rows)", f"{len(frame):,}", "one row per browsing session", "#4f46e5"),
            ("Columns used", f"{len(FEATURES) + 1}", f"{len(NUMERIC_FEATURES)} numeric · {len(CATEGORICAL_FEATURES)} categorical · 1 target", "#0d9488"),
            ("Purchases", f"{int(frame[TARGET].sum()):,}", f"{frame[TARGET].mean():.1%} of sessions", "#db2777"),
            ("Duplicate rows", f"{int(frame.duplicated().sum()):,}", "identical sessions", "#ea580c"),
            ("Extra columns", f"{len(extras)}", "ignored by the models" if extras else "none found", "#ca8a04"),
        ]
    )

    section("Quick insights")
    for note in quick_insights(frame):
        st.markdown(f'<div class="insight">💡 {_md_bold(note)}</div>', unsafe_allow_html=True)
    if extras:
        st.caption("Ignored columns: " + ", ".join(extras))

    section("Column dictionary")
    st.dataframe(
        column_dictionary(frame),
        width="stretch",
        hide_index=True,
        column_config={
            "Meaning": st.column_config.TextColumn(width="large"),
            "Summary": st.column_config.TextColumn(width="medium"),
        },
    )

    section("Class balance")
    st.altair_chart(class_balance_chart(outcome), width="stretch")
    st.caption("Purchases are the minority class, which is why accuracy alone can look better than a model really is.")

    section("Explore one feature")
    left, right = st.columns([2, 1])
    feature = left.selectbox("Feature", FEATURES, index=FEATURES.index("PageValues"))
    left.caption(COLUMN_INFO[feature])
    if feature in NUMERIC_FEATURES:
        zero_share = (data[feature] == 0).mean()
        trim = right.toggle("Hide the top 1% of values", value=True, help="These features have very long tails.")
        hide_zeros = right.toggle(
            "Hide zero values",
            value=zero_share > 0.5 and zero_share < 1,
            help="Switched on when most sessions are 0, so the rest of the shape is visible.",
        )
        right.markdown(f"**{zero_share:.0%}** of sessions have a value of 0 for this feature.")
        if hide_zeros and zero_share == 1:
            st.info("Every session has 0 for this feature, so there is nothing left to plot.")
        else:
            st.altair_chart(numeric_distribution(data, feature, trim, hide_zeros), width="stretch")
        summary = data.groupby("Outcome")[feature].agg(["mean", "median", "max"]).reindex(list(CLASS_COLORS))
        summary.columns = ["Mean", "Median", "Max"]
        st.dataframe(summary.style.format("{:,.3f}"), width="stretch")
    else:
        sessions, rates = category_charts(data, feature)
        a, b = st.columns(2)
        a.altair_chart(sessions, width="stretch")
        b.altair_chart(rates, width="stretch")
        st.caption("The dashed pink line is the overall purchase rate for this file.")

    section("How the numeric features relate")
    if frame[TARGET].nunique() > 1:
        st.altair_chart(correlation_heatmap(frame), width="stretch")
        st.caption("Pink means two columns rise together, blue means one rises as the other falls. The last row is the target.")
    else:
        st.info("The correlation chart needs both classes in the file.")

    section("Numeric summary")
    st.dataframe(frame[NUMERIC_FEATURES].describe().T.style.format("{:,.3f}"), width="stretch")

    section("Raw data preview")
    st.dataframe(frame.head(100), width="stretch", height=320)
    st.caption(f"First {min(100, len(frame))} of {len(frame):,} rows, after validation.")


def _md_bold(text: str) -> str:
    """Escape text for HTML, then turn **bold** into <b>."""
    parts = html.escape(text).split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))
