import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize dataset-related session state (if not already set by app.py)
for key, default in {
    "df": None,
    "dataset_name": None,
    "selected_dataset": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Constants
CLEANING_STEPS = [
    ("1. Data Inspection", "Examine data types, structure, and basic statistics", "High"),
    ("2. Missing Value Detection", "Identify columns with missing or null values", "Critical"),
    ("3. Data Type Conversion", "Ensure correct data types for each column", "High"),
    ("4. Outlier Detection", "Identify extreme values that may affect model performance", "Medium"),
    ("5. Feature Encoding", "Convert categorical variables to numeric representations", "Critical"),
    ("6. Feature Scaling", "Normalize numeric features to similar ranges", "High")
]

IMPORTANCE_COLORS = {"Critical": "#B60015", "High": "#722F37", "Medium": "#f0ad4e"}


@st.cache_data
def clean_dataset(df: pd.DataFrame):
    """Apply cleaning operations to dataset (cached)."""
    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns

    # Impute missing values
    for col in numeric_cols:
        if col != 'result' and df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

    for col in categorical_cols:
        if col != 'result' and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)

    # Encode categorical
    for col in categorical_cols:
        if col != 'result':
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col])

    return df_clean, len(categorical_cols)


def render_cleaning_step(step, description, importance):
    """Render a single cleaning step card."""
    st.markdown(f"""
    <div class="metric-card">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <h4 style="color: #722F37; margin: 0;">{step}</h4>
          <p style="margin: 0.5rem 0 0 0; color: #441C21;">{description}</p>
        </div>
        <div style="background-color: {IMPORTANCE_COLORS[importance]}; color: white; padding: 0.5rem 1rem;
                    border-radius: 0.5rem; font-weight: bold;">
          {importance}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_card(title, features, exclude='result'):
    """Render feature list card."""
    st.markdown(f"""
    <div class="metric-card">
      <h4 style="color: #722F37;">{title}</h4>
    """, unsafe_allow_html=True)
    for col in features:
        if col != exclude:
            st.write(f"• {col}")
    st.markdown("</div>", unsafe_allow_html=True)


def render():
    """Main render function for data cleaning page."""
    st.markdown('<h1 class="main-header">Data Cleaning</h1>', unsafe_allow_html=True)

    # Guard: no dataset selected yet
    if not st.session_state.get('selected_dataset') or st.session_state.get('df') is None:
        st.markdown("""
        <div class="warning-box">
          <h4>No Dataset Selected</h4>
          <p>Please select a dataset from the <strong>"About the Datasets"</strong> section first.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    dataset_name = st.session_state.dataset_name

    st.markdown(f"""
    <div class="info-box">
      <h3>Understanding Data Cleaning</h3>
      <p>
      Data cleaning is a critical step in the machine learning pipeline. Before we can build accurate
      models, we need to ensure our data is properly formatted, consistent, and ready for analysis.
      </p>
      <p><strong>Current Dataset:</strong> {dataset_name}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Cleaning Pipeline
    st.markdown('<h2 class="sub-header">Cleaning Pipeline</h2>', unsafe_allow_html=True)
    for step, desc, importance in CLEANING_STEPS:
        render_cleaning_step(step, desc, importance)

    st.markdown("---")

    # Step 1: Data Inspection
    st.markdown('<h2 class="sub-header">Step 1: Data Inspection</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Data Shape")
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])

    with col2:
        st.markdown("#### Data Types")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            st.metric(str(dtype), count)

    st.markdown("#### Raw Data Sample (Before Cleaning)")
    st.dataframe(df.head(10), width='stretch')

    st.markdown("---")

    # Step 2: Data Types Analysis
    st.markdown('<h2 class="sub-header">Step 2: Data Types Analysis</h2>', unsafe_allow_html=True)

    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })

    st.dataframe(dtype_df, width='stretch')

    st.markdown("---")

    # Step 3: Missing Value Strategy
    st.markdown('<h2 class="sub-header">Step 3: Missing Value Strategy</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <h4>Imputation Techniques Used</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    techniques = [
        ("Numeric Columns", "Median Imputation", "Robust to outliers and extreme values",
         "Missing value = Median of column",
         ["Less sensitive to outliers than mean", "Better for skewed distributions", "Preserves central tendency"]),
        ("Categorical Columns", "Mode Imputation", "Preserves most common category",
         "Missing value = Most frequent value",
         ["Maintains distribution patterns", "Logical for categorical data", "Minimizes bias introduction"])
    ]

    for col, (title, method, reason, formula, points) in zip([col1, col2], techniques):
        with col:
            points_html = "".join(f"<li>{point}</li>" for point in points)
            st.markdown(f"""
            <div class="metric-card">
              <h4 style="color: #722F37;">{title}</h4>
              <p><strong>Method:</strong> {method}</p>
              <p><strong>Reason:</strong> {reason}</p>
              <p><strong>Formula:</strong> {formula}</p>
              <ul>{points_html}</ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Step 4: Categorical Encoding
    st.markdown('<h2 class="sub-header">Step 4: Categorical Encoding</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <h4>Why Encode Categorical Variables?</h4>
      <p>
      Machine learning algorithms require numeric input. Categorical variables (like 'Pass'/'Fail')
      must be converted to numeric format through encoding.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Label Encoding Method")
    st.code("""
# Example: Label Encoding for 'result' column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['result_encoded'] = le.fit_transform(df['result'])

# Result:
# 'Fail' → 0
# 'Pass' → 1
    """, language="python")

    # Show encoding example
    if 'result' in df.columns:
        encoding_example = df[['result']].copy()
        le = LabelEncoder()
        encoding_example['result_encoded'] = le.fit_transform(df['result'])

        st.markdown("#### Encoding Example")
        st.dataframe(encoding_example.head(10), width='stretch')

    st.markdown("---")

    # Step 5: Feature Scaling
    st.markdown('<h2 class="sub-header">Step 5: Feature Scaling</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <h4>Why Scale Features?</h4>
      <p>
      Features with different scales can dominate the learning process. Scaling ensures all
      features contribute equally to model training.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
          <h4 style="color: #722F37;">StandardScaler</h4>
          <p><strong>Method:</strong> Z-score normalization</p>
          <p><strong>Formula:</strong> z = (x - μ) / σ</p>
          <p>Where:</p>
          <ul>
            <li>x = original value</li>
            <li>μ = mean</li>
            <li>σ = standard deviation</li>
          </ul>
          <p><strong>Result:</strong> Mean = 0, Std Dev = 1</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
          <h4 style="color: #722F37;">When to Use?</h4>
          <ul>
            <li><strong>Use when:</strong> Features have different scales</li>
            <li><strong>Use when:</strong> Using distance-based algorithms</li>
            <li><strong>Use when:</strong> Data is normally distributed</li>
            <li><strong>Be careful:</strong> With outliers (they affect mean/std)</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Complete Cleaning Preview
    st.markdown('<h2 class="sub-header">Cleaned Data Preview</h2>', unsafe_allow_html=True)

    # Use cached cleaning on current df
    df_clean, cat_cols_before = clean_dataset(df)

    st.dataframe(df_clean.head(10), width='stretch')

    cols = st.columns(3)
    metrics = [
        ("Missing Values Before", df.isnull().sum().sum(), "Missing Values After", df_clean.isnull().sum().sum()),
        ("Categorical Columns Before", cat_cols_before, "Categorical Columns After",
         len(df_clean.select_dtypes(include=['object']).columns)),
        ("Data Quality", "Clean", "Ready for Modeling", "Yes")
    ]

    for col, (label1, val1, label2, val2) in zip(cols, metrics):
        with col:
            st.metric(label1, val1)
            st.metric(label2, val2)

    st.markdown("""
    <div class="success-box">
      <h4>Data Cleaning Complete!</h4>
      <p>
      Your dataset is now clean and ready for analysis. <br>
      Next Step: Proceed to the <strong>Missingness</strong>
      section to dive deeper into missing value patterns.
      </p>
    </div>
    """, unsafe_allow_html=True)