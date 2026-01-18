import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Interactive Data Dashboard",
    layout="wide"
)

st.title("📊 Interactive Dashboard")
st.write("Summary of findings from Assignments 1 & 2")

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

# -------------------------
# Sidebar Widgets
# -------------------------
st.sidebar.header("🔧 Filters")

# Widget 1: Select column for analysis
numeric_cols = df.select_dtypes(include="number").columns.tolist()
selected_col = st.sidebar.selectbox(
    "Select a numeric column",
    numeric_cols
)

# Widget 2: Slider filter
min_val, max_val = float(df[selected_col].min()), float(df[selected_col].max())
value_range = st.sidebar.slider(
    "Select value range",
    min_val,
    max_val,
    (min_val, max_val)
)

# Widget 3: Checkbox
show_raw = st.sidebar.checkbox("Show raw data")

filtered_df = df[
    (df[selected_col] >= value_range[0]) &
    (df[selected_col] <= value_range[1])
]

# -------------------------
# Optional Raw Data Display
# -------------------------
if show_raw:
    st.subheader("📄 Raw Data Preview")
    st.dataframe(filtered_df.head(100))

# -------------------------
# Layout for Plots
# -------------------------
col1, col2 = st.columns(2)

# -------------------------
# Plot 1: Histogram
# -------------------------
with col1:
    st.subheader("Histogram")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------
# Plot 2: Boxplot
# -------------------------
with col2:
    st.subheader("Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(x=filtered_df[selected_col], ax=ax)
    st.pyplot(fig)

# -------------------------
# Plot 3: Correlation Heatmap
# -------------------------
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    filtered_df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)
st.pyplot(fig)

# -------------------------
# Plot 4: Line Chart
# -------------------------
st.subheader("Line Chart")
st.line_chart(filtered_df[selected_col])

# -------------------------
# Plot 5: Bar Chart (Top 10 values)
# -------------------------
st.subheader("Top 10 Values")
top_values = filtered_df[selected_col].value_counts().head(10)
st.bar_chart(top_values)

# -------------------------
# Key Insights Section
# -------------------------
st.subheader("📌 Key Insights")
st.markdown("""
- Interactive filters allow exploration of numeric features  
- Distribution and spread visible via histogram and boxplot  
- Strong correlations can be identified from the heatmap  
- Dashboard supports dynamic exploration of the dataset  
""")
