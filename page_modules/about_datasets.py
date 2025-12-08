import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Set matplotlib style once
plt.style.use('seaborn-v0_8-darkgrid')

# Dataset configurations
DATASETS = {
    1: {
        "name": "Mathematics Performance",
        "subject": "Mathematics",
        "records": "~395",
        "focus": "Study habits & grades",
        "features": "7",
    },
    2: {
        "name": "Portuguese Language Performance",
        "subject": "Portuguese",
        "records": "~649",
        "focus": "Demographics & behavior",
        "features": "9",
    },
    3: {
        "name": "Comprehensive Academic Profile",
        "subject": "Combined",
        "records": "~500",
        "focus": "Support systems",
        "features": "10",
    }
}

# Initialize dataset-related state if not present
for key, default in {
    "df": None,
    "dataset_name": None,
    "selected_dataset": None,
    "last_clicked_dataset": None  # for styling/logic if you ever need it
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def create_dataset_card(dataset_id, config):
    """Create HTML for dataset selection card."""
    return f"""
    <div class="metric-card">
      <h4 style="color: #722F37;">Dataset {dataset_id}: {config['name']}</h4>
      <ul style="font-size: 0.9rem;">
        <li><strong>Subject:</strong> {config['subject']}</li>
        <li><strong>Records:</strong> {config['records']}</li>
        <li><strong>Focus:</strong> {config['focus']}</li>
        <li><strong>Features:</strong> {config['features']}</li>
      </ul>
    </div>
    """


def load_dataset(dataset_id):
    """Load dataset based on ID and persist in session_state."""
    from utils.data_loader import load_dataset_1, load_dataset_2, load_dataset_3

    loaders = {1: load_dataset_1, 2: load_dataset_2, 3: load_dataset_3}

    # If the same dataset is already loaded, do nothing
    if st.session_state.selected_dataset == dataset_id and st.session_state.df is not None:
        return

    with st.spinner(f"Loading {DATASETS[dataset_id]['name']}..."):
        st.session_state.df = loaders[dataset_id]()
        st.session_state.selected_dataset = dataset_id
        st.session_state.dataset_name = DATASETS[dataset_id]["name"]
        st.session_state.last_clicked_dataset = dataset_id

    # Explicit rerun ensures other pages depending on df see updated state immediately
    st.rerun()  # safe pattern for refreshing after a load[web:58][web:72]


def render_dataset_selection():
    """Render dataset selection buttons and cards."""
    st.markdown('<h2 class="sub-header">Select a Dataset</h2>', unsafe_allow_html=True)

    cols = st.columns(3)

    for col, (dataset_id, config) in zip(cols, DATASETS.items()):
        with col:
            label = f"{config['icon']} {config['subject']} Performance"
            if st.button(label, key=f"ds{dataset_id}"):
                load_dataset(dataset_id)
            st.markdown(create_dataset_card(dataset_id, config), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        <div class="warning-box">
          <h4>No Dataset Selected</h4>
          <p>Please select a dataset using the buttons above (or from the sidebar page) to continue with the analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_balance_status(ratio):
    """Determine balance status and color based on ratio."""
    if ratio > 0.8:
        return "Well Balanced", "#71d488"
    elif ratio > 0.5:
        return "Moderately Balanced", "#ffc061"
    else:
        return "Imbalanced", "#D62D41"


def render_class_distribution(df):
    """Render class distribution visualization."""
    st.markdown('<h3 class="section-header">Target Variable Distribution</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 8))
        result_counts = df["result"].value_counts()
        colors = ["#28a745" if x == "Pass" else "#722F37" for x in result_counts.index]

        wedges, texts, autotexts = ax.pie(
            result_counts.values,
            labels=result_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 14, "weight": "bold"},
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontsize(14)
            autotext.set_weight("bold")

        ax.set_title("Pass vs Fail Distribution", fontsize=16, fontweight="bold", color="#441C21")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Class Balance Analysis")

        pass_count = (df["result"] == "Pass").sum()
        fail_count = (df["result"] == "Fail").sum()

        st.metric("Pass Count", pass_count)
        st.metric("Fail Count", fail_count)

        balance_ratio = min(pass_count, fail_count) / max(pass_count, fail_count)
        balance_status, balance_color = get_balance_status(balance_ratio)

        st.markdown(
            f"""
            <div style="background-color: {balance_color}; color: white; padding: 1rem;
                        border-radius: 0.5rem; text-align: center; margin-top: 1rem;">
              <h4 style="margin: 0;">{balance_status}</h4>
              <p style="margin: 0.5rem 0 0 0;">Ratio: {balance_ratio:.2f}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dataset_overview(df, dataset_name):
    """Render complete dataset overview."""
    st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)

    # Success message
    st.markdown(
        f"""
        <div class="success-box">
          <h3>Currently Selected: {dataset_name}</h3>
          <p>Dataset loaded successfully! You can now proceed with the analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key Metrics
    pass_rate = (df["result"] == "Pass").sum() / len(df) * 100
    missing_count = df.isnull().sum().sum()

    cols = st.columns(5)
    metrics = [
        ("Total Records", len(df)),
        ("Features", len(df.columns) - 1),
        ("Pass Rate", f"{pass_rate:.1f}%"),
        ("Fail Rate", f"{100 - pass_rate:.1f}%"),
        ("Missing Values", missing_count),
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    st.markdown("---")

    # Data Source
    st.markdown('<h3 class="section-header">Data Source</h3>', unsafe_allow_html=True)
    st.info(
        """
        **Source:** UCI Machine Learning Repository  
        **Origin:** Student performance data from two Portuguese secondary schools  
        **Collection Period:** 2005-2006 academic year  
        **Citation:** P. Cortez and A. Silva (2008)
        """
    )

    # Sample Data
    st.markdown('<h3 class="section-header">Sample Data Preview</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(10), width="stretch")

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Dataset",
        data=csv,
        file_name=f"{dataset_name.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Statistical Summary
    st.markdown('<h3 class="section-header">Statistical Summary</h3>', unsafe_allow_html=True)
    st.dataframe(df.describe(), width="stretch")

    # Class Distribution
    render_class_distribution(df)

    st.markdown("---")

    # Feature Overview
    st.markdown('<h3 class="section-header">Feature Overview</h3>', unsafe_allow_html=True)

    feature_info = [
        {
            "Feature": col,
            "Type": "Numeric" if df[col].dtype in ["int64", "float64"] else "Categorical",
            "Missing": df[col].isnull().sum(),
            "Unique Values": df[col].nunique(),
        }
        for col in df.columns
        if col != "result"
    ]

    st.dataframe(pd.DataFrame(feature_info), width="stretch")

    # Next steps
    st.markdown(
        """
        <div class="success-box">
          <h4>Next Steps</h4>
          <p>
          Now that you've selected a dataset, proceed to the <strong>EDA</strong> section to explore patterns and relationships in the data.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    """Main render function for about_datasets page."""
    df = st.session_state.get("df")
    dataset_name = st.session_state.get("dataset_name")
    selected_dataset = st.session_state.get("selected_dataset")

    # Header
    header_text = (
        f"About the Dataset </span>"
        if df is not None and dataset_name
        else "About the Datasets"
    )
    st.markdown(f'<h1 class="main-header">{header_text}</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Render appropriate content
    if df is None or selected_dataset is None:
        render_dataset_selection()
    else:
        render_dataset_overview(df, dataset_name)