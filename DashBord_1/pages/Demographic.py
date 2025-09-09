import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.load_data import load_data

df = load_data()

st.title("📊 3.Demographics & Household Profile")


# -----------------------------
# KPIs
# -----------------------------
df["AGE_YEARS"] = -(df["DAYS_BIRTH"] / 365).astype(int)
df["EMP_YEARS"] = -(df["DAYS_EMPLOYED"] / 365).replace({365243: np.nan})  # handle placeholder for unemployed

pct_male = (df["CODE_GENDER"].eq("M").mean() * 100)
pct_female = (df["CODE_GENDER"].eq("F").mean() * 100)
avg_age_def = df.loc[df["TARGET"] == 1, "AGE_YEARS"].mean()
avg_age_nondef = df.loc[df["TARGET"] == 0, "AGE_YEARS"].mean()
pct_with_children = (df["CNT_CHILDREN"].gt(0).mean() * 100)
avg_family_size = df["CNT_FAM_MEMBERS"].mean()
pct_married = (df["NAME_FAMILY_STATUS"].str.contains("Married").mean() * 100)
pct_single = (df["NAME_FAMILY_STATUS"].str.contains("Single|Separated|Widow|Widower|Divorced").mean() * 100)
pct_higher_edu = df["NAME_EDUCATION_TYPE"].isin(["Higher education", "Academic degree"]).mean() * 100
pct_with_parents = (df["NAME_HOUSING_TYPE"] == "With parents").mean() * 100
pct_working = df["OCCUPATION_TYPE"].ne("Other").mean() * 100  # crude proxy
avg_emp_years = df["EMPLOYMENT_YEARS"].mean()

# KPI Display
st.subheader("🔑 Key Demographic Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("% Male", f"{pct_male:.1f}%")
col2.metric("% Female", f"{pct_female:.1f}%")
col3.metric("Avg Age — Defaulters", f"{avg_age_def:.1f}")

col4, col5, col6 = st.columns(3)
col4.metric("Avg Age — Non-Defaulters", f"{avg_age_nondef:.1f}")
col5.metric("% With Children", f"{pct_with_children:.1f}%")
col6.metric("Avg Family Size", f"{avg_family_size:.1f}")

col7, col8, col9 = st.columns(3)
col7.metric("% Married", f"{pct_married:.1f}%")
col8.metric("% Single/Other", f"{pct_single:.1f}%")
col9.metric("% Higher Education", f"{pct_higher_edu:.1f}%")

col10 = st.columns(1)[0]
col10.metric("% Living With Parents", f"{pct_with_parents:.1f}%")
st.metric("Avg Employment Years", f"{avg_emp_years:.1f}")
st.markdown("---")

# -----------------------------
# -------------------------
# Charts
# -------------------------
st.subheader("📊 Demographics & Household Distributions")

# Histogram — Age distribution
st.write("### Age Distribution (All Applicants)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["AGE_YEARS"].dropna(), bins=50, color="#1f77b4", alpha=1, label="Age")
ax.set_xlabel("Age (Years)")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# Histogram — Age by Target (overlay)
st.write("### Age Distribution by Target")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df[df["TARGET"] == 0]["AGE_YEARS"], bins=50, alpha=0.6, label="Non-Defaulters (0)")
ax.hist(df[df["TARGET"] == 1]["AGE_YEARS"], bins=50, alpha=0.6, label="Defaulters (1)")
ax.set_xlabel("Age (Years)")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# Bar — Gender distribution
st.write("### Gender Distribution")
gender_counts = df["CODE_GENDER"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(gender_counts.index, gender_counts.values, color="#1f77b4", alpha=1)
ax.set_xlabel("Gender")
ax.set_ylabel("Count")
st.pyplot(fig)

# Bar — Family Status distribution
st.write("### Family Status Distribution")
family_counts = df["NAME_FAMILY_STATUS"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(family_counts.index, family_counts.values, color="#ff7f0e", alpha=1)
ax.set_xlabel("Family Status")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=25)
st.pyplot(fig)

# Bar — Education distribution
st.write("### Education Distribution")
edu_counts = df["NAME_EDUCATION_TYPE"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(edu_counts.index, edu_counts.values, color="#2ca02c", alpha=1)
ax.set_xlabel("Education Type")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=25)
st.pyplot(fig)

# Bar — Occupation distribution (top 10)
st.write("### Occupation Distribution (Top 10)")
occ_counts = df["OCCUPATION_TYPE"].fillna("MISSING").value_counts().head(10)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(occ_counts.index, occ_counts.values, color="#9467bd", alpha=1)
ax.set_xlabel("Occupation Type")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=25)
st.pyplot(fig)

# Pie — Housing Type distribution
st.write("### Housing Type Distribution")
housing_counts = df["NAME_HOUSING_TYPE"].value_counts()
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(housing_counts.values, labels=housing_counts.index, autopct="%1.1f%%", startangle=90)
st.pyplot(fig)

# Countplot — CNT_CHILDREN
st.write("### Children Count Distribution")
children_counts = df["CNT_CHILDREN"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(children_counts.index, children_counts.values, color="#8c564b", alpha=1)
ax.set_xlabel("Number of Children")
ax.set_ylabel("Count")
st.pyplot(fig)

# Boxplot — Age vs Target
st.write("### Age vs Target")
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot([df[df["TARGET"] == 0]["AGE_YEARS"], df[df["TARGET"] == 1]["AGE_YEARS"]],
           labels=["Repaid (0)", "Default (1)"])
ax.set_ylabel("Age (Years)")
st.pyplot(fig)

# Heatmap — Correlations
st.write("### Correlation Heatmap (Demographic Variables)")
demo_vars = df[["AGE_YEARS", "CNT_CHILDREN", "CNT_FAM_MEMBERS", "TARGET"]]
corr = demo_vars.corr()
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(corr, cmap="coolwarm")
fig.colorbar(cax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=25)
ax.set_yticklabels(corr.columns)
st.pyplot(fig)

# -------------------------
# Narrative
# -------------------------
st.subheader("📝 Insights")
st.markdown("""
- **Gender split** shows slightly higher share of females.  
- **Age patterns**: Defaulters skew younger on average than non-defaulters.  
- **Children and family size** increase household complexity, which may affect repayment.  
- **Education**: Higher education applicants form a smaller subset, potentially linked with better repayment.  
- **Living with parents** and employment gaps reveal dependence factors.  
""")