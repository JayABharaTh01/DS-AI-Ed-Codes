import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# -------------------------
# Home / Intro Page
# -------------------------
st.title("📊 Credit Risk Dashboard")

st.markdown("""
Welcome! This dashboard provides an **end-to-end view of loan applicants, risk segmentation, and financial health**.  
It is organized into 5 pages, each focusing on a different aspect of portfolio risk and applicant characteristics.

### 🔎 Navigation
- **Page 1 — Overview & Data Quality**  
- **Page 2 — Target & Risk Segmentation**  
- **Page 3 — Demographics & Household Profile**  
- **Page 4 — Financial Health & Affordability**  
- **Page 5 — Correlations & Drivers**

---
""")

np.random.seed(42)
df = pd.DataFrame({
    "TARGET": np.random.choice([0, 1], size=500, p=[0.8, 0.2]),
    "AGE_YEARS": np.random.randint(20, 70, size=500),
    "AMT_INCOME_TOTAL": np.random.randint(50000, 500000, size=500),
    "AMT_CREDIT": np.random.randint(100000, 1000000, size=500),
})

# -------------------------
# Graphs
# -------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Target Distribution")
    st.bar_chart(df["TARGET"].value_counts())

with col2:
    st.subheader("Age Distribution")
    st.bar_chart(df["AGE_YEARS"].value_counts().sort_index())

col3, col4 = st.columns(2)

with col3:
    st.subheader("Income Distribution")
    st.bar_chart(df["AMT_INCOME_TOTAL"].value_counts().sort_index().head(50))

with col4:
    st.subheader("Credit Distribution")
    st.bar_chart(df["AMT_CREDIT"].value_counts().sort_index().head(50))
