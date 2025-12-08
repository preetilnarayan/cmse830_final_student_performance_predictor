import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Ensure dataset-related state exists
for key, default in {
    "df": None,
    "dataset_name": None,
    "selected_dataset": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

plt.style.use('seaborn-v0_8-darkgrid')

# Visualization colors
COLORS = {
    'pass': '#28a745',
    'fail': '#722F37',
    'text': '#441C21'
}


@st.cache_data
def prepare_eda_data(df: pd.DataFrame):
    """Clean and prepare data for EDA - cached for performance."""
    df = df.copy()

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)

    return df


@st.cache_data
def encode_dataframe(df: pd.DataFrame):
    """Encode categorical variables for correlation analysis."""
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = le.fit_transform(df_encoded[col])
    return df_encoded


def plot_distributions(df, numeric_features):
    """Create histogram plots for feature distributions."""
    num_cols_plot = min(len(numeric_features), 6)
    n_rows = (num_cols_plot + 1) // 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, col in enumerate(numeric_features[:num_cols_plot]):
        ax = axes[idx]

        # Separate by result
        pass_data = df[df['result'] == 'Pass'][col].dropna()
        fail_data = df[df['result'] == 'Fail'][col].dropna()

        ax.hist(pass_data, bins=20, alpha=0.6, label='Pass',
                color=COLORS['pass'], edgecolor='black')
        ax.hist(fail_data, bins=20, alpha=0.6, label='Fail',
                color=COLORS['fail'], edgecolor='black')

        ax.set_xlabel(col, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'Distribution of {col}', fontsize=12,
                     fontweight='bold', color=COLORS['text'])
        ax.legend()
        ax.grid(alpha=0.3)

    # Hide empty subplots
    for idx in range(num_cols_plot, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_boxplots(df, numeric_features):
    """Create box plots comparing Pass vs Fail."""
    num_cols_plot = min(len(numeric_features), 6)
    n_rows = (num_cols_plot + 1) // 2

    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, col in enumerate(numeric_features[:num_cols_plot]):
        ax = axes[idx]

        data_to_plot = [
            df[df['result'] == 'Fail'][col].dropna(),
            df[df['result'] == 'Pass'][col].dropna()
        ]

        bp = ax.boxplot(data_to_plot, labels=['Fail', 'Pass'], patch_artist=True)

        # Color boxes
        colors = [COLORS['fail'], COLORS['pass']]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel(col, fontsize=11, fontweight='bold')
        ax.set_title(f'{col} by Result', fontsize=12,
                     fontweight='bold', color=COLORS['text'])
        ax.grid(axis='y', alpha=0.3)

    # Hide empty subplots
    for idx in range(num_cols_plot, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df_encoded):
    """Create correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df_encoded.corr()

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title('Feature Correlation Matrix', fontsize=16,
                fontweight='bold', color=COLORS['text'])
    plt.tight_layout()
    return fig, correlation_matrix


def plot_pairwise_relationship(df, feature_x, feature_y):
    """Create scatter plot for two features."""
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_specs = [
        ('Pass', COLORS['pass'], 'o'),
        ('Fail', COLORS['fail'], 's')
    ]

    for result, color, marker in plot_specs:
        mask = df['result'] == result
        ax.scatter(
            df[mask][feature_x],
            df[mask][feature_y],
            c=color,
            label=result,
            alpha=0.6,
            s=50,
            marker=marker,
            edgecolors='black',
            linewidth=0.5
        )

    ax.set_xlabel(feature_x, fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_y, fontsize=12, fontweight='bold')
    ax.set_title(f'{feature_x} vs {feature_y}', fontsize=14,
                 fontweight='bold', color=COLORS['text'])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def render_correlation_insights(target_corr):
    """Render positive and negative correlations."""
    col1, col2 = st.columns(2)

    correlations = [
        (col1, "Positive Correlations", "(Higher values → Higher pass rate)",
         target_corr[target_corr > 0].head(5)),
        (col2, "Negative Correlations", "(Higher values → Lower pass rate)",
         target_corr[target_corr < 0].tail(5))
    ]

    for col, title, subtitle, corr_data in correlations:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <h4 style="color: #722F37;">{title}</h4>
              <p><em>{subtitle}</em></p>
            </div>
            """, unsafe_allow_html=True)

            for feature, corr_value in corr_data.items():
                st.metric(feature, f"{corr_value:.3f}")


def generate_insights(df, target_corr):
    """Generate key insights from the data."""
    pass_rate = (df['result'] == 'Pass').sum() / len(df) * 100
    insights = []

    # Pass rate insight
    if pass_rate > 70:
        insights.append((
            "High Pass Rate",
            f"<div class=\"success-box\">{pass_rate:.1f}% of students pass. The dataset shows generally positive outcomes.</div>"
        ))
    elif pass_rate > 50:
        insights.append((
            "Moderate Pass Rate",
            f"<div class=\"warning-box\">{pass_rate:.1f}% of students pass. There's room for improvement.</div>"
        ))
    else:
        insights.append((
            "Low Pass Rate",
            f"<div class=\"error-box\">Only {pass_rate:.1f}% of students pass. Significant intervention needed.</div>"
        ))

    # Top correlations
    if len(target_corr[target_corr > 0]) > 0:
        top_positive = target_corr[target_corr > 0].index[0]
        insights.append((
            "Strongest Positive Predictor",
            f"{top_positive} shows the strongest positive correlation with passing."
        ))

    if len(target_corr[target_corr < 0]) > 0:
        top_negative = target_corr[target_corr < 0].index[-1]
        insights.append((
            "Strongest Negative Predictor",
            f"{top_negative} shows the strongest negative correlation with passing."
        ))

    # Class balance
    pass_count = (df['result'] == 'Pass').sum()
    fail_count = (df['result'] == 'Fail').sum()
    balance_ratio = min(pass_count, fail_count) / max(pass_count, fail_count)

    if balance_ratio > 0.8:
        insights.append((
            "Well Balanced Dataset",
            "The dataset has a good balance between Pass and Fail outcomes."
        ))
    else:
        insights.append((
            "Imbalanced Dataset",
            "Consider using techniques like SMOTE or class weights during modeling."
        ))

    return insights


def render():
    """Main render function for EDA page."""
    st.markdown('<h1 class="main-header">Exploratory Data Analysis</h1>', unsafe_allow_html=True)

    # Guard: ensure a dataset is selected and df exists
    if not st.session_state.get('selected_dataset') or st.session_state.get('df') is None:
        st.markdown("""
        <div class="warning-box">
          <h4>No Dataset Selected</h4>
          <p>Please select a dataset from the <strong>"About the Datasets"</strong> section first.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    dataset_name = st.session_state.dataset_name
    df_raw = st.session_state.df
    df = prepare_eda_data(df_raw)

    st.markdown(f"""
    <div class="info-box">
      <h3>Exploring {dataset_name}</h3>
      <p>
      Exploratory Data Analysis (EDA) helps us understand the underlying patterns, relationships,
      and distributions in our data before building predictive models.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Statistical Summary
    st.markdown('<h2 class="sub-header">Statistical Summary</h2>', unsafe_allow_html=True)
    st.dataframe(df.describe(), width='stretch')
    st.markdown("---")

    # Get numeric features (excluding target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_cols if col not in ['result', 'final_grade']]

    # Feature Distributions
    if len(numeric_features) > 0:
        st.markdown('<h2 class="sub-header">Feature Distributions</h2>', unsafe_allow_html=True)
        fig = plot_distributions(df, numeric_features)
        st.pyplot(fig)
        plt.close()
        st.markdown("---")

        # Box Plots
        st.markdown('<h2 class="sub-header">Feature Comparison: Pass vs Fail</h2>', unsafe_allow_html=True)
        fig = plot_boxplots(df, numeric_features)
        st.pyplot(fig)
        plt.close()
        st.markdown("---")

    # Correlation Analysis
    st.markdown('<h2 class="sub-header">Correlation Analysis</h2>', unsafe_allow_html=True)

    df_encoded = encode_dataframe(df)
    fig, correlation_matrix = plot_correlation_heatmap(df_encoded)
    st.pyplot(fig)
    plt.close()

    # Top correlations
    target_corr = correlation_matrix['result'].drop('result').sort_values(ascending=False)
    st.markdown("#### Top Features Correlated with Pass/Fail")
    render_correlation_insights(target_corr)
    st.markdown("---")

    # Pairwise Relationships
    st.markdown('<h2 class="sub-header">Pairwise Relationships</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <p>Select two features to explore their relationship and how it relates to student outcomes.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Use stable keys so selections persist between reruns and do not conflict with other pages. [web:75][web:82]
    with col1:
        feature_x = st.selectbox(
            "Select X-axis feature",
            numeric_features,
            key="eda_pair_x"
        )
    with col2:
        feature_y = st.selectbox(
            "Select Y-axis feature",
            [f for f in numeric_features if f != feature_x],
            key="eda_pair_y"
        )

    if feature_x and feature_y:
        fig = plot_pairwise_relationship(df, feature_x, feature_y)
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Key Insights
    st.markdown('<h2 class="sub-header">Key Insights</h2>', unsafe_allow_html=True)

    insights = generate_insights(df, target_corr)
    for title, description in insights:
        st.markdown(f"""
        <div class="metric-card">
          <h4 style="color: #722F37;">{title}</h4>
          <p style="margin: 0.5rem 0 0 0;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="success-box">
      <h4>EDA Complete!</h4>
      <p>
      You've gained valuable insights into the data patterns and relationships. <br>
      Next Step: Proceed to <strong>Cleaning</strong> section to learn about data preprocessing
      </p>
    </div>
    """, unsafe_allow_html=True)
