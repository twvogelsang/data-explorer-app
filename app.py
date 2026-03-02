import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Exam Analytics Dashboard", layout="wide")

st.markdown(
    """
<style>
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 0rem;
}
h1 { font-size: 1.5rem !important; margin-bottom: 0.4rem; }
h2 { font-size: 1.1rem !important; margin-bottom: 0.25rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Exam Analytics Dashboard")
st.markdown("Interactive analysis of exam activity by category and semester.")

# -----------------------------
# Load & Clean Data
# -----------------------------
df = pd.read_csv("exam.csv")

df["Category"] = df["Category"].str.strip()
df = df[df["Attribute"] != "26WN"]

df["Year"] = df["Attribute"].str[:2].astype(int) + 2000
df["Term"] = df["Attribute"].str[2:]

term_order_map = {"WN": 1, "FL": 2}
df["Term_Order"] = df["Term"].map(term_order_map)

semester_order_df = (
    df[["Attribute", "Year", "Term_Order"]]
    .drop_duplicates()
    .sort_values(["Year", "Term_Order"])
)

ordered_semesters = semester_order_df["Attribute"].tolist()

df["Attribute"] = pd.Categorical(
    df["Attribute"], categories=ordered_semesters, ordered=True
)

df = df.sort_values("Attribute")

# -----------------------------
# Filters
# -----------------------------
with st.expander("Filters", expanded=False):
    col_f1, col_f2, col_f3 = st.columns(
        [2, 2, 1]
    )  # wider main columns, smaller third column

    with col_f1:
        categories = st.multiselect(
            "Categories",
            options=sorted(df["Category"].unique()),
            default=sorted(df["Category"].unique()),
        )

    with col_f2:
        semesters = st.multiselect(
            "Semesters",
            options=ordered_semesters,
            default=ordered_semesters,
        )

    with col_f3:
        show_trendlines = st.checkbox("Trendlines", value=False)

filtered_df = df[
    (df["Category"].isin(categories)) & (df["Attribute"].isin(semesters))
].sort_values("Attribute")

# -----------------------------
# Metrics
# -----------------------------
total_selected = filtered_df["Value"].sum()
mean_selected = filtered_df.groupby("Category")["Value"].mean().mean()
max_selected = filtered_df["Value"].max()
min_selected = filtered_df["Value"].min()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total", f"{total_selected:,}")
col2.metric("Average", f"{mean_selected:,.1f}")
col3.metric("Max", f"{max_selected:,}")
col4.metric("Min", f"{min_selected:,}")

# -----------------------------
# Trendline warning if too many categories selected
# # -----------------------------
if show_trendlines and len(categories) > 3:
    st.warning(
        "Trendlines are disabled to reduce visual clutter. To display trendlines, select 3 or fewer categories."
    )

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(14, 4.2))

palette = sns.color_palette("tab10", n_colors=df["Category"].nunique())
category_colors = dict(zip(sorted(df["Category"].unique()), palette))

semester_index_map = {sem: i for i, sem in enumerate(ordered_semesters)}

for category in filtered_df["Category"].unique():
    cat_df = filtered_df[filtered_df["Category"] == category]

    ax.plot(
        cat_df["Attribute"].astype(str),
        cat_df["Value"],
        marker="o",
        label=category,
        color=category_colors[category],
        linewidth=2,
    )

    if show_trendlines and len(categories) <= 3 and len(cat_df) > 1:
        x_vals = np.array([semester_index_map[str(s)] for s in cat_df["Attribute"]])
        y_vals = cat_df["Value"].values
        coeffs = np.polyfit(x_vals, y_vals, deg=1)
        trend = np.polyval(coeffs, x_vals)

        ax.plot(
            cat_df["Attribute"].astype(str),
            trend,
            linestyle="--",
            color=category_colors[category],
            alpha=0.35,
            linewidth=1.2,
        )

ax.set_ylabel("Exam Count")
plt.xticks(rotation=45)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
plt.tight_layout()
st.pyplot(fig)


# -----------------------------
# Explicit Note
# -----------------------------
st.info(
    "All metrics and tables reflect only the categories and semesters currently selected in the filters."
)

# -----------------------------
# Category Performance Summary
# -----------------------------
with st.expander("Category Performance Summary", expanded=False):
    if not filtered_df.empty:
        overall_total = filtered_df["Value"].sum()
        summary_list = []

        for category in filtered_df["Category"].unique():
            cat_df = filtered_df[filtered_df["Category"] == category].sort_values(
                "Attribute"
            )

            total = cat_df["Value"].sum()
            mean = cat_df["Value"].mean()
            abs_change = cat_df["Value"].iloc[-1] - cat_df["Value"].iloc[0]

            pct_change = (
                abs_change / cat_df["Value"].iloc[0]
                if cat_df["Value"].iloc[0] != 0
                else 0
            )

            pct_of_total = total / overall_total if overall_total != 0 else 0

            summary_list.append(
                {
                    "Category": category,
                    "Total": f"{total:,.0f}",
                    "Average": f"{mean:,.1f}",
                    "% of Selected Total": f"{pct_of_total * 100:.1f}%",
                    "Absolute Change": f"{abs_change:+,.0f}",
                    "% Change (Selected Filters)": f"{pct_change * 100:+.1f}%",
                }
            )

        summary_df = pd.DataFrame(summary_list).sort_values("Total", ascending=False)
        summary_df = summary_df.reset_index(drop=True)

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# -----------------------------
# Show Data
# -----------------------------
with st.expander("Show Data"):
    display_df = filtered_df[["Category", "Attribute", "Value"]].copy()
    display_df.rename(
        columns={"Attribute": "Semester", "Value": "Exam Count"},
        inplace=True,
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)
