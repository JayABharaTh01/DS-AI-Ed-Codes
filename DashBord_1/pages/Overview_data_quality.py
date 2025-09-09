import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.load_data import load_data

df = load_data()

st.title("📊 1.Overview & Data Quality")

# ===================== KPIs =====================
total_applicants = df["SK_ID_CURR"].nunique()
default_rate = df["TARGET"].mean() * 100
repaid_rate = 100 - default_rate
total_features = df.shape[1]
avg_missing_per_feature = df.isnull().mean().mean() * 50
num_features = df.select_dtypes(include=[np.number]).shape[1]
cat_features = df.select_dtypes(exclude=[np.number]).shape[1]
median_age = df["AGE_YEARS"].median()
median_income = df["AMT_INCOME_TOTAL"].median()
avg_credit = df["AMT_CREDIT"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Applicants", f"{total_applicants:,}")
col2.metric("Default Rate (%)", f"{default_rate:.2f}%")
col3.metric("Repaid Rate (%)", f"{repaid_rate:.2f}%")

col4, col5, col6 = st.columns(3)
col4.metric("Total Features", total_features)
col5.metric("Num Features", num_features)
col6.metric("Cat Features", cat_features)

col7, col8, col9 = st.columns(3)
col7.metric("Avg Missing per Feature (%)", f"{avg_missing_per_feature:.2f}%")
col8.metric("Median Age (Years)", f"{median_age:.0f}")
col9.metric("Median Annual Income", f"{median_income:,.0f}")

st.metric("Average Credit Amount", f"{avg_credit:,.0f}")

# Sidebar
#use to display sidebar for chart options
st.sidebar.header("Chart Options")
chart = st.sidebar.selectbox(
    "Select chart",
    (
        "Target Distribution",
        "Missing Values (Top N)",
        "Histogram — AGE_YEARS",
        "Histogram — AMT_INCOME_TOTAL",
        "Histogram — AMT_CREDIT",
        "Bar — Categorical (CODE_GENDER / FAMILY / EDUCATION)"
    )
)

if "Histogram" in chart:
    bins = st.sidebar.slider("Bins", 20, 100, 10)
elif chart == "Missing Values (Top N)":
    top_n = st.sidebar.slider("Top N features", 3, 10, 2)
elif chart == "Target Distribution":
    display_mode = st.sidebar.radio("Display as", ("Bar", "Pie"))
elif chart == "Bar — Categorical":
    cat_col = st.sidebar.selectbox("Categorical column", ["CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"])
    top_k = st.sidebar.slider("Top K categories to show", 3, 30, 10)
    xlabel = st.sidebar.text_input("X label", cat_col)
    ylabel = st.sidebar.text_input("Y label", "Count")

# Plot functions
def plot_histogram(series, bins, xlabel, ylabel):
    series_clean = series.dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(series_clean, bins=bins, alpha=1, color="#1f77b4", label=xlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.xticks(rotation=25)
    plt.yticks(rotation=25)
    st.pyplot(fig)

def plot_bar_from_series(series_counts, xlabel, ylabel, horizontal=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(len(series_counts))
    if horizontal:
        ax.barh(indices, series_counts.values, alpha=1, color="#1f77b4", label=xlabel)
        ax.set_yticks(indices)
        ax.set_yticklabels(series_counts.index, rotation=25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.bar(indices, series_counts.values, alpha=1, color="#1f77b4", label=xlabel)
        ax.set_xticks(indices)
        ax.set_xticklabels(series_counts.index, rotation=25, ha="right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    ax.legend()
    plt.xticks(rotation=25)
    plt.yticks(rotation=25)
    st.pyplot(fig)

# Chart rendering
st.subheader("📊 Chart")
if chart == "Target Distribution":
    target_dist = df["TARGET"].value_counts().sort_index()
    labels = [str(x) for x in target_dist.index]
    if display_mode == "Bar":
        plot_bar_from_series(target_dist, xlabel="TARGET", ylabel="Count")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.pie(
            target_dist.values,
            labels=labels,
            autopct="%1.1f%%",
            colors=["#1f77b4", "#ff7f0e"],
        )
        ax.legend(labels, title="TARGET")
        st.pyplot(fig)

elif chart == "Missing Values (Top N)":
    missing_vals = (df.isnull().mean() * 100).sort_values(ascending=False).head(top_n)
    # Fixed labels
    plot_bar_from_series(missing_vals[::-1], xlabel="Missing %", ylabel="Feature", horizontal=True)

elif chart == "Histogram — AGE_YEARS":
    plot_histogram(df["AGE_YEARS"], bins=bins, xlabel='Age (Years)', ylabel='Count')

elif chart == "Histogram — AMT_INCOME_TOTAL":
    plot_histogram(df["AMT_INCOME_TOTAL"], bins=bins, xlabel='Annual Income', ylabel='Count')

elif chart == "Histogram — AMT_CREDIT":
    plot_histogram(df["AMT_CREDIT"], bins=bins, xlabel='Credit Amount', ylabel='Count')

elif chart == "Bar — Categorical":
    counts = df[cat_col].fillna("MISSING").value_counts().head(top_k)
    plot_bar_from_series(counts, xlabel=cat_col, ylabel='Count', horizontal=False)

# Insights
st.subheader("📝 Insights")
st.markdown(f"""
- Default rate: **{default_rate:.2f}%** → dataset is imbalanced.  
- Median age: **{median_age:.0f} years**.  
- Median income: **{median_income:,.0f}**.  
- Avg missing per feature: **{avg_missing_per_feature:.2f}%**.  
- Top missing feature: **{(df.isnull().mean()*100).sort_values(ascending=False).index[0]}**  
""")
