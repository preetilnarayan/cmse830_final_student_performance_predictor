import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

# Ensure dataset-related state exists
for key, default in {
    "df": None,
    "dataset_name": None,
    "selected_dataset": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Constants
MISSINGNESS_TYPES = [
    ("MCAR", "Missing Completely At Random",
     ["Missingness is random", "No pattern or relationship", "Least problematic", "Safe to use simple imputation"]),
    ("MAR", "Missing At Random",
     ["Related to observed data", "Pattern can be explained", "Moderate concern", "Use advanced imputation"]),
    ("MNAR", "Missing Not At Random",
     ["Related to unobserved data", "Systematic pattern", "Most problematic", "Requires careful handling"])
]

IMPUTATION_STRATEGIES = [
    ("Mean/Median Imputation", "Numeric data with MCAR", "Simple, fast, preserves mean",
     "Reduces variance, ignores relationships", "Used for numeric columns"),
    ("Mode Imputation", "Categorical data", "Simple, maintains distribution",
     "May introduce bias", "Used for categorical columns"),
    ("Forward/Backward Fill", "Time series data", "Maintains temporal patterns",
     "Not suitable for cross-sectional data", "Not applicable"),
    ("KNN Imputation", "Complex patterns (MAR)", "Uses feature relationships",
     "Computationally expensive", "Not used"),
    ("Multiple Imputation", "High missingness (MNAR)", "Most robust, captures uncertainty",
     "Complex, time-consuming", "Not recommended")
]


@st.cache_data
def analyze_missing_data(df: pd.DataFrame):
    """Analyze missing data patterns - cached for performance."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_percent = (missing_cells / total_cells) * 100
    complete_rows = df.dropna().shape[0]

    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100).round(2),
        'Data Type': df.dtypes.values
    })

    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

    return total_cells, missing_cells, missing_percent, complete_rows, missing_data


@st.cache_data
def compute_missingness_correlation(df: pd.DataFrame):
    """Compute correlation of missing values - cached."""
    missing_matrix = df.isnull().astype(int)
    cols_with_missing = df.columns[df.isnull().any()].tolist()

    if len(cols_with_missing) > 1:
        return missing_matrix[cols_with_missing].corr(), cols_with_missing
    return None, cols_with_missing


@st.cache_data
def impute_missing_values(df: pd.DataFrame):
    """Apply imputation to dataset - cached."""
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    categorical_cols = df_imputed.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        if df_imputed[col].isnull().sum() > 0:
            df_imputed[col].fillna(df_imputed[col].median(), inplace=True)

    for col in categorical_cols:
        if df_imputed[col].isnull().sum() > 0:
            mode_val = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else 'Unknown'
            df_imputed[col].fillna(mode_val, inplace=True)

    return df_imputed


def plot_missing_values(missing_data: pd.DataFrame):
    """Create bar chart for missing values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(missing_data['Column'], missing_data['Missing %'], color='#722F37', alpha=0.8)
    ax.set_xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold', color='#441C21')
    ax.set_ylabel('Column', fontsize=12, fontweight='bold', color='#441C21')
    ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold', color='#441C21')
    ax.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.,
            f'{width:.1f}%',
            ha='left',
            va='center',
            fontweight='bold',
            fontsize=10,
            color='#441C21'
        )

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(missing_corr: pd.DataFrame):
    """Create correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        missing_corr,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Correlation of Missing Values', fontsize=14, fontweight='bold', color='#441C21')
    plt.tight_layout()
    return fig


def render_severity_cards(missing_data: pd.DataFrame):
    """Render severity classification cards."""
    st.markdown("#### Severity Classification")

    severity_map = [
        (5, "Low", "#28a745"),
        (20, "Medium", "#ff9800"),
        (float('inf'), "High", "#722F37")
    ]

    for _, row in missing_data.iterrows():
        pct = row['Missing %']
        severity, color = next((s, c) for threshold, s, c in severity_map if pct < threshold)

        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 0.5rem;
                    border-radius: 0.3rem; margin: 0.5rem 0;">
          <strong>{row['Column']}</strong><br>
          {pct:.1f}% - {severity}
        </div>
        """, unsafe_allow_html=True)


def render_missingness_type_cards():
    """Render missingness mechanism cards."""
    cols = st.columns(3)

    for col, (title, subtitle, points) in zip(cols, MISSINGNESS_TYPES):
        with col:
            points_html = "".join(f"<li>{point}</li>" for point in points)
            st.markdown(f"""
            <div class="metric-card">
              <h4 style="color: #722F37;">{title}</h4>
              <p><strong>{subtitle}</strong></p>
              <ul>{points_html}</ul>
            </div>
            """, unsafe_allow_html=True)


def render_strategy_cards():
    """Render imputation strategy cards."""
    for method, use_case, pros, cons, choice in IMPUTATION_STRATEGIES:
        choice_color = "#28a745" if choice.lower().startswith("used") else "#CF0018"

        st.markdown(f"""
        <div class="metric-card">
          <div style="display: flex; justify-content: space-between; align-items: start;">
            <div style="flex: 1;">
              <h4 style="color: #722F37; margin: 0 0 0.5rem 0;">{method}</h4>
              <p style="margin: 0.3rem 0;"><strong>Use Case:</strong> {use_case}</p>
              <p style="margin: 0.3rem 0;"><strong>Pros:</strong> {pros}</p>
              <p style="margin: 0.3rem 0;"><strong>Cons:</strong> {cons}</p>
            </div>
            <div style="background-color: {choice_color}; color: white; padding: 0.5rem 1rem;
                        border-radius: 0.5rem; margin-left: 1rem; white-space: nowrap;">
              {choice}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)


def render():
    """Main render function for missingness analysis page."""
    st.markdown('<h1 class="main-header">Missing Value Analysis</h1>', unsafe_allow_html=True)

    # Guard: ensure dataset is selected and df exists in state
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
      <h3>Understanding Missing Data</h3>
      <p>
      Missing data is a common challenge in real-world datasets. Understanding patterns of missingness
      helps us choose appropriate imputation strategies and assess potential biases.
      </p>
      <p><strong>Current Dataset:</strong> {dataset_name}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Overall Missing Data Summary
    st.markdown('<h2 class="sub-header">Missing Data Overview</h2>', unsafe_allow_html=True)

    total_cells, missing_cells, missing_percent, complete_rows, missing_data = analyze_missing_data(df)

    cols = st.columns(4)
    metrics = [
        ("Total Cells", f"{total_cells:,}"),
        ("Missing Cells", f"{missing_cells:,}"),
        ("Missing %", f"{missing_percent:.2f}%"),
        ("Complete Rows", f"{complete_rows:,}")
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    st.markdown("---")

    # Missing Values by Column
    st.markdown('<h2 class="sub-header">Missing Values by Column</h2>', unsafe_allow_html=True)

    if len(missing_data) > 0:
        st.dataframe(missing_data, width='stretch')

        st.markdown("#### Missing Values Visualization")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = plot_missing_values(missing_data)
            st.pyplot(fig)
            plt.close()

        with col2:
            render_severity_cards(missing_data)
    else:
        st.markdown("""
        <div class="success-box">
          <h4>No Missing Values Detected</h4>
          <p>This dataset has complete data for all features.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Missing Data Patterns
    st.markdown('<h2 class="sub-header">Missing Data Patterns</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <h4>Understanding Missingness Mechanisms</h4>
    </div>
    """, unsafe_allow_html=True)

    render_missingness_type_cards()

    st.markdown("---")

    # Correlation of Missingness
    if missing_cells > 0:
        st.markdown('<h2 class="sub-header">Missingness Correlation</h2>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
          <p>
          This heatmap shows whether missing values in one column are correlated with missing
          values in another column. High correlation might indicate systematic missingness.
          </p>
        </div>
        """, unsafe_allow_html=True)

        missing_corr, cols_with_missing = compute_missingness_correlation(df)

        if missing_corr is not None:
            fig = plot_correlation_heatmap(missing_corr)
            st.pyplot(fig)
            plt.close()

            st.info("""
            Interpretation:
            - Values close to 1: Missing values in these columns occur together
            - Values close to 0: Missing values are independent
            - Values close to -1: When one is missing, the other is present
            """)
        else:
            st.info("Only one column has missing values, correlation analysis not applicable.")

    st.markdown("---")

    # Imputation Strategies
    st.markdown('<h2 class="sub-header">Imputation Strategies</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <h4>Choosing the Right Imputation Method</h4>
      <p>Different types of missing data require different handling approaches.</p>
    </div>
    """, unsafe_allow_html=True)

    render_strategy_cards()

    st.markdown("---")

    # Before and After Imputation
    st.markdown('<h2 class="sub-header">Before and After Imputation</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Before Imputation")
        st.dataframe(df.head(10), width='stretch')
        st.metric("Rows with Missing Values", len(df) - df.dropna().shape[0])

    with col2:
        st.markdown("#### After Imputation")
        df_imputed = impute_missing_values(df)
        st.dataframe(df_imputed.head(10), width='stretch')
        st.metric("Rows with Missing Values", len(df_imputed) - df_imputed.dropna().shape[0])

    st.markdown("""
    <div class="success-box">
      <h4>Imputation Complete</h4>
      <p>
      All missing values have been handled using appropriate strategies. The dataset is now
      ready for exploratory data analysis and model training.
      </p>
      <p>
      Next Step: Proceed to <strong>Prediction Models</strong> to start building ML models.
      </p>
    </div>
    """, unsafe_allow_html=True)