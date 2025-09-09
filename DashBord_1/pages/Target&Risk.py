import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.load_data import load_data

df=load_data()

st.title("📊  2.Target & Risk Segmentation")


# -----------------------------
# KPIs
# -----------------------------
total_defaults = int(df["TARGET"].sum())
default_rate = df["TARGET"].mean() * 100

# Group-wise default rates
def_rate_gender = df.groupby("CODE_GENDER")["TARGET"].mean() * 100
def_rate_edu = df.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean() * 100
def_rate_family = df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean() * 100
def_rate_housing = df.groupby("NAME_HOUSING_TYPE")["TARGET"].mean() * 100

# Averages among defaulters
df_def = df[df["TARGET"] == 1]
avg_income_def = df_def["AMT_INCOME_TOTAL"].mean()
avg_credit_def = df_def["AMT_CREDIT"].mean()
avg_annuity_def = df_def["AMT_ANNUITY"].mean()
avg_emp_def = df_def["EMPLOYMENT_YEARS"].mean()

# Show KPIs
st.subheader("🔑 Key Risk Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Defaults", f"{total_defaults:,}")
col2.metric("Default Rate (%)", f"{default_rate:.2f}%")
col3.metric("Avg Income (Defaulters)", f"{avg_income_def:,.0f}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Credit (Defaulters)", f"{avg_credit_def:,.0f}")
col5.metric("Avg Annuity (Defaulters)", f"{avg_annuity_def:,.0f}")
col6.metric("Avg Employment Years (Defaulters)", f"{avg_emp_def:.1f}")

col7, col8, col9 = st.columns(3)
col7.metric("Default Rate by Gender (%)", f"{def_rate_gender.mean():.2f}%")
col8.metric("Default Rate by Education (%)", f"{def_rate_edu.mean():.2f}%")
col9.metric("Default Rate by Family Status (%)", f"{def_rate_family.mean():.2f}%")

st.metric("Default Rate by Housing Type (%)", f"{def_rate_housing.mean():.2f}%")
st.markdown("---")

# Safe column presence checks and fallback handling
def safe_col(col, fill=np.nan):
    return df[col] if col in df.columns else pd.Series([fill] * len(df), index=df.index)

TARGET = safe_col("TARGET", 0).astype(int)
CODE_GENDER = safe_col("CODE_GENDER", "Unknown")
NAME_EDUCATION_TYPE = safe_col("NAME_EDUCATION_TYPE", "Unknown")
NAME_FAMILY_STATUS = safe_col("NAME_FAMILY_STATUS", "Unknown")
AMT_INCOME_TOTAL = pd.to_numeric(safe_col("AMT_INCOME_TOTAL", np.nan), errors="coerce")
AMT_CREDIT = pd.to_numeric(safe_col("AMT_CREDIT", np.nan), errors="coerce")
AMT_ANNUITY = pd.to_numeric(safe_col("AMT_ANNUITY", np.nan), errors="coerce")
DAYS_EMPLOYED = safe_col("DAYS_EMPLOYED", np.nan)
# handle placeholder 365243 (common in this dataset) -> treat as NaN
DAYS_EMPLOYED = DAYS_EMPLOYED.replace({365243: np.nan})
EMP_YEARS = (-pd.to_numeric(DAYS_EMPLOYED, errors="coerce") / 365).replace([np.inf, -np.inf], np.nan)
NAME_HOUSING_TYPE = safe_col("NAME_HOUSING_TYPE", "Unknown")
NAME_CONTRACT_TYPE = safe_col("NAME_CONTRACT_TYPE", "Unknown")
AGE_YEARS = None
if "DAYS_BIRTH" in df.columns:
    AGE_YEARS = (-pd.to_numeric(df["DAYS_BIRTH"], errors="coerce") / 365).astype("float").apply(np.floor)

# Add columns back into a working df for grouping convenience
work = pd.DataFrame({
    "TARGET": TARGET,
    "CODE_GENDER": CODE_GENDER,
    "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
    "NAME_FAMILY_STATUS": NAME_FAMILY_STATUS,
    "AMT_INCOME_TOTAL": AMT_INCOME_TOTAL,
    "AMT_CREDIT": AMT_CREDIT,
    "AMT_ANNUITY": AMT_ANNUITY,
    "EMP_YEARS": EMP_YEARS,
    "NAME_HOUSING_TYPE": NAME_HOUSING_TYPE,
    "NAME_CONTRACT_TYPE": NAME_CONTRACT_TYPE,
})
if AGE_YEARS is not None:
    work["AGE_YEARS"] = AGE_YEARS
    
# plotting defaults
PLOT_COLOR_1 = "#1f77b4"
PLOT_COLOR_2 = "#ff7f0e"
PLOT_COLOR_3 = "#2ca02c"
ALPHA = 1
FIGSIZE = (10, 5)
XT_ROT = 25
YT_ROT = 25
SHOW_GRID = False  # explicitly removed grid
st.subheader("📈 Graphs — Target & Risk")

# 1) Bar — Counts: Default vs Repaid
st.write("1) Counts: Default vs Repaid")
work["TARGET"].value_counts().sort_index().plot(
    kind="bar", color=[PLOT_COLOR_1, PLOT_COLOR_2], figsize=FIGSIZE)
plt.ylabel("Count"); plt.xticks(rotation=XT_ROT)
st.pyplot(plt.gcf()); plt.clf()

# 2) Default % by Gender
st.write("2) Default % by Gender")
(work.groupby("CODE_GENDER")["TARGET"].mean()*100).plot(
    kind="bar", color=PLOT_COLOR_1, figsize=FIGSIZE)
plt.ylabel("Default Rate (%)"); plt.xticks(rotation=XT_ROT)
st.pyplot(plt.gcf()); plt.clf()

# 3) Default % by Education
st.write("3) Default % by Education")
(work.groupby("NAME_EDUCATION_TYPE")["TARGET"].mean().sort_values(ascending=False)*100).plot(
    kind="bar", color=PLOT_COLOR_3, figsize=FIGSIZE)
plt.ylabel("Default Rate (%)"); plt.xticks(rotation=XT_ROT, ha="right")
st.pyplot(plt.gcf()); plt.clf()

# 4) Default % by Family Status
st.write("4) Default % by Family Status")
(work.groupby("NAME_FAMILY_STATUS")["TARGET"].mean().sort_values(ascending=False)*100).plot(
    kind="bar", color=PLOT_COLOR_2, figsize=FIGSIZE)
plt.ylabel("Default Rate (%)"); plt.xticks(rotation=XT_ROT, ha="right")
st.pyplot(plt.gcf()); plt.clf()

# 5) Default % by Housing Type
st.write("5) Default % by Housing Type")
(work.groupby("NAME_HOUSING_TYPE")["TARGET"].mean().sort_values(ascending=False)*100).plot(
    kind="bar", color=PLOT_COLOR_1, figsize=FIGSIZE)
plt.ylabel("Default Rate (%)"); plt.xticks(rotation=XT_ROT, ha="right")
st.pyplot(plt.gcf()); plt.clf()

# 6) Income by Target
st.write("6) Income by Target")
work.boxplot(column="AMT_INCOME_TOTAL", by="TARGET", figsize=FIGSIZE)
plt.ylabel("Income"); plt.suptitle(""); plt.title("")
st.pyplot(plt.gcf()); plt.clf()

# 7) Credit by Target
st.write("7) Credit by Target")
work.boxplot(column="AMT_CREDIT", by="TARGET", figsize=FIGSIZE)
plt.ylabel("Credit"); plt.suptitle(""); plt.title("")
st.pyplot(plt.gcf()); plt.clf()

# 8) Age vs Target
if "AGE_YEARS" in work.columns:
    st.write("8) Age vs Target")
    work.boxplot(column="AGE_YEARS", by="TARGET", figsize=FIGSIZE)
    plt.ylabel("Age (Years)"); plt.suptitle(""); plt.title("")
    st.pyplot(plt.gcf()); plt.clf()

# 9) Employment Years Histogram
st.write("9) Employment Years by Target")
work.groupby("TARGET")["EMP_YEARS"].plot(kind="hist", bins=30, alpha=0.6, figsize=FIGSIZE)
plt.xlabel("Employment Years"); plt.ylabel("Count"); plt.legend(["Repaid (0)", "Default (1)"])
st.pyplot(plt.gcf()); plt.clf()

# 10) Contract Type vs Target
st.write("10) Contract Type vs Target")
work.groupby(["NAME_CONTRACT_TYPE","TARGET"]).size().unstack(fill_value=0).plot(
    kind="bar", stacked=True, figsize=FIGSIZE, color=[PLOT_COLOR_1, PLOT_COLOR_2])
plt.ylabel("Count"); plt.xticks(rotation=XT_ROT, ha="right")
st.pyplot(plt.gcf()); plt.clf()


st.markdown("---")
st.subheader("Narrative / Next hypotheses")
st.markdown("""
- Use the Default % by X charts above to identify the 2–3 segments with the *highest* and *lowest* default rates (for example check education/family/housing charts).  
- Example hypotheses to test further with multivariate models:  
  - **Low income + high LTI** increases default probability.  
  - **Short employment history (low EMP_YEARS)** is associated with higher default.  
  - **Specific family statuses or housing types** may correlate with higher stress (investigate top 2–3 categories by default %).  
- Next steps: run logistic regression / tree-based models including LTI, DTI, EMP_YEARS and interaction terms to validate.  
""")