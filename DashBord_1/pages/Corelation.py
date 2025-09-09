import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.load_data import load_data

df = load_data()

st.title("📊 5.Correlations, Drivers & Slice-and-Dice")

# Pre-compute correlations
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# -------------------------
# KPIs
# -------------------------
target_corr = corr_matrix["TARGET"].drop("TARGET").sort_values()

top_pos_corr = target_corr.tail(5)
top_neg_corr = target_corr.head(5)

most_corr_income = corr_matrix["AMT_INCOME_TOTAL"].drop("AMT_INCOME_TOTAL").abs().idxmax()
most_corr_credit = corr_matrix["AMT_CREDIT"].drop("AMT_CREDIT").abs().idxmax()

corr_income_credit = corr_matrix.loc["AMT_INCOME_TOTAL", "AMT_CREDIT"]
corr_age_target = corr_matrix.loc["AGE_YEARS", "TARGET"]
corr_emp_target = corr_matrix.loc["DAYS_EMPLOYED", "TARGET"] if "DAYS_EMPLOYED" in corr_matrix else np.nan
corr_fam_target = corr_matrix.loc["CNT_FAM_MEMBERS", "TARGET"] if "CNT_FAM_MEMBERS" in corr_matrix else np.nan

top5_var_explained = target_corr.abs().nlargest(5).sum()
num_corr_gt_05 = (target_corr.abs() > 0.5).sum()

# Display KPIs
st.title("📊 Correlations, Drivers & Slice-and-Dice")

col1, col2 = st.columns(2)
with col1:
    st.metric("Top +Corr with TARGET", ", ".join(top_pos_corr.index.tolist()))
    st.metric("Top −Corr with TARGET", ", ".join(top_neg_corr.index.tolist()))
    st.metric("Most correlated with Income", most_corr_income)
    st.metric("Most correlated with Credit", most_corr_credit)
    st.metric("Corr(Income, Credit)", f"{corr_income_credit:.2f}")
with col2:
    st.metric("Corr(Age, TARGET)", f"{corr_age_target:.2f}")
    st.metric("Corr(Employment Years, TARGET)", f"{corr_emp_target:.2f}")
    st.metric("Corr(Family Size, TARGET)", f"{corr_fam_target:.2f}")
    st.metric("Variance explained by Top 5", f"{top5_var_explained:.2f}")
    st.metric("# Features with |corr| > 0.5", num_corr_gt_05)


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Correlation Graph Options")

chart = st.sidebar.selectbox(
    "Select graph",
    (
        "Heatmap — Correlation (selected numerics)",
        "Bar — |Correlation| vs TARGET",
        "Scatter — Age vs Credit",
        "Scatter — Age vs Income",
        "Scatter — Employment vs TARGET",
        "Boxplot — Credit by Education",
        "Boxplot — Income by Family Status",
        "Filtered Bar — Default Rate by Gender",
        "Filtered Bar — Default Rate by Education",
    )
)

# -------------------------
# Plot helpers (matplotlib only)
# -------------------------
def apply_rotation(ax, xrot=25, yrot=25):
    for tick in ax.get_xticklabels():
        tick.set_rotation(xrot)
        tick.set_ha("right")
    for tick in ax.get_yticklabels():
        tick.set_rotation(yrot)

def plot_heatmap(cols):
    fig, ax = plt.subplots(figsize=(10, 5))
    data = numeric_df[cols].corr()
    cax = ax.matshow(data, cmap="coolwarm")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="left")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, rotation=25)
    ax.set_title("Correlation Heatmap", pad=20)
    st.pyplot(fig)

def plot_corr_bar():
    fig, ax = plt.subplots(figsize=(10, 5))
    data = target_corr.abs().sort_values(ascending=False).head(20)
    data.plot.bar(ax=ax, color="#1f77b4", alpha=1, label="|Correlation| with TARGET")
    ax.legend()
    apply_rotation(ax)
    st.pyplot(fig)

def plot_scatter(x, y, hue="TARGET"):
    fig, ax = plt.subplots(figsize=(10, 5))
    if hue in df:
        colors = df[hue].map({0: "#1f77b4", 1: "#ff7f0e"})
        scatter = ax.scatter(df[x], df[y], c=colors, alpha=0.7, label=None)
        for val, col in {0: "#1f77b4", 1: "#ff7f0e"}.items():
            ax.scatter([], [], c=col, alpha=0.7, label=f"{hue}={val}")
    else:
        ax.scatter(df[x], df[y], color="#1f77b4", alpha=0.7, label=f"{x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    apply_rotation(ax)
    st.pyplot(fig)

def plot_boxplot(x, y):
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = df.groupby(x)[y].apply(list)
    ax.boxplot(groups, labels=groups.index)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend([y], loc="upper right")
    apply_rotation(ax)
    st.pyplot(fig)

def plot_filtered_bar(group_col):
    df_group = df.groupby(group_col)["TARGET"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    df_group.plot.bar(ax=ax, color="#1f77b4", alpha=1, label="Default Rate (%)")
    ax.set_xlabel(group_col)
    ax.set_ylabel("Default Rate (%)")
    ax.legend()
    apply_rotation(ax)
    st.pyplot(fig)

# -------------------------
# Chart rendering
# -------------------------
st.subheader("📈 Graphs")

if chart == "Heatmap — Correlation (selected numerics)":
    selected_cols = st.multiselect(
        "Select numeric columns",
        numeric_df.columns.tolist(),
        ["TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AGE_YEARS"],
    )
    if len(selected_cols) > 1:
        plot_heatmap(selected_cols)
    else:
        st.warning("Select at least 2 columns.")

elif chart == "Bar — |Correlation| vs TARGET":
    plot_corr_bar()

elif chart == "Scatter — Age vs Credit":
    plot_scatter("AGE_YEARS", "AMT_CREDIT")

elif chart == "Scatter — Age vs Income":
    plot_scatter("AGE_YEARS", "AMT_INCOME_TOTAL")

elif chart == "Scatter — Employment vs TARGET":
    if "DAYS_EMPLOYED" in df:
        plot_scatter("DAYS_EMPLOYED", "TARGET")
    else:
        st.warning("DAYS_EMPLOYED not available.")

elif chart == "Boxplot — Credit by Education":
    plot_boxplot("NAME_EDUCATION_TYPE", "AMT_CREDIT")

elif chart == "Boxplot — Income by Family Status":
    plot_boxplot("NAME_FAMILY_STATUS", "AMT_INCOME_TOTAL")

elif chart == "Filtered Bar — Default Rate by Gender":
    plot_filtered_bar("CODE_GENDER")

elif chart == "Filtered Bar — Default Rate by Education":
    plot_filtered_bar("NAME_EDUCATION_TYPE")

# -------------------------
# Narrative
# -------------------------
st.subheader("📝 Insights & Policy Ideas")
st.markdown("""
- Strongest **drivers of default** show up in correlations with employment, credit, and family size.  
- Income and credit are **strongly related**; LTI (loan-to-income) caps could be applied.  
- Default rate differs by **education level and gender**, useful for segmentation.  
- Policy candidates:  
  - Set **minimum income floors** for high-risk applicants.  
  - Apply **LTV / LTI caps** where risk is higher.  
  - Use **family size and employment years** as risk-based pricing levers.  
""")