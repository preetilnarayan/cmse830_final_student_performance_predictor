import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from utils.preprocessing import preprocess_data
from utils.models import train_all_models

plt.style.use('seaborn-v0_8-darkgrid')

# Initialize session state keys used on this page
for key, default in {
    "df": None,
    "dataset_name": None,
    "selected_dataset": None,
    "model_results": None,
    "scaler": None,
    "X_train": None,
    "X_test": None,
    "X_features": None,
    "label_encoders": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Constants
METRICS_TO_PLOT = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
METRIC_COLORS = [
    "#01AF49",
    "#F1C40F",
    "#E67E22",
    "#E74C3C"
]


def plot_performance_bars(metrics_df):
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics_df['Model']))
    width = 0.2

    for i, (metric, color) in enumerate(zip(METRICS_TO_PLOT, METRIC_COLORS)):
        ax.bar(x + i * width, metrics_df[metric], width, label=metric, alpha=0.8, color=color)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', color='#441C21')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='RdYlGn',
        xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'],
        cbar_kws={'label': 'Count'}, ax=ax, linewidths=2, linecolor='#441C21',
        annot_kws={'size': 16, 'weight': 'bold'}
    )

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold', color='#441C21')

    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_pred_proba):
    fig, ax = plt.subplots(figsize=(7, 6))

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color='#722F37', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#AA8287', lw=2, linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#722F37')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold', color='#441C21')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names, importances, model_name):
    feature_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feature_df['Feature'], feature_df['Importance'], color='#722F37', alpha=0.8)

    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold', color='#441C21')
    ax.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width, bar.get_y() + bar.get_height() / 2., f'{width:.3f}',
            ha='left', va='center', fontweight='bold', fontsize=9, color='#441C21'
        )

    plt.tight_layout()
    return fig, feature_df


def plot_explained_variance(explained_var, n_components, method_name):
    var_df = pd.DataFrame({
        'Component': [f'{method_name}{i+1}' for i in range(n_components)],
        'Explained Variance (%)': explained_var * 100,
        'Cumulative Variance (%)': np.cumsum(explained_var) * 100
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(var_df))
    ax.bar(x, var_df['Explained Variance (%)'], 0.6, label='Individual', color='#722F37', alpha=0.8)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        var_df['Cumulative Variance (%)'],
        'o-',
        color='#AA8287',
        linewidth=3,
        markersize=10,
        label='Cumulative'
    )

    ax.set_xlabel(f'{method_name} Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Individual Variance (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold', color='#AA8287')
    ax.set_title(f'{method_name} Explained Variance', fontsize=14, fontweight='bold', color='#441C21')
    ax.set_xticks(x)
    ax.set_xticklabels(var_df['Component'])
    ax.grid(alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return fig, var_df


def generate_model_insights(model_data):
    insights = []

    acc = model_data['accuracy']
    if acc > 0.85:
        insights.append(("High Accuracy", f"The model achieves {acc:.1%} accuracy, indicating strong predictive power."))
    elif acc > 0.75:
        insights.append(("Good Accuracy", f"The model achieves {acc:.1%} accuracy, which is solid but has room for improvement."))
    else:
        insights.append(("Moderate Accuracy", f"The model achieves {acc:.1%} accuracy. Consider feature engineering or different algorithms."))

    if model_data['precision'] > model_data['recall']:
        insights.append(("High Precision", "The model is conservative in predicting Pass, making fewer false positive errors."))
    elif model_data['recall'] > model_data['precision']:
        insights.append(("High Recall", "The model is aggressive in predicting Pass, capturing more true positives but with more false positives."))
    else:
        insights.append(("Balanced Precision/Recall", "The model maintains a good balance between precision and recall."))

    if model_data['f1'] > 0.80:
        insights.append(("Strong F1 Score", f"F1-Score of {model_data['f1']:.3f} indicates excellent overall classification performance."))

    return insights


def render_model_details(selected_model, model_data, feature_names):
    cols = st.columns(4)
    metrics = [
        ("Accuracy", model_data['accuracy']),
        ("Precision", model_data['precision']),
        ("Recall", model_data['recall']),
        ("F1-Score", model_data['f1'])
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, f"{value:.3f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        fig = plot_confusion_matrix(model_data['confusion_matrix'])
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### ROC Curve")
        fig = plot_roc_curve(model_data['y_test'], model_data['y_pred_proba'])
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    st.markdown("#### Classification Report")
    st.text(model_data['classification_report'])

    if 'feature_importance' in model_data:
        st.markdown('<h2 class="sub-header">Feature Importance</h2>', unsafe_allow_html=True)
        fig, feature_df = plot_feature_importance(feature_names, model_data['feature_importance'], selected_model)
        st.pyplot(fig)
        plt.close()
        st.dataframe(feature_df, width='stretch')

    elif 'explained_variance' in model_data:
        st.markdown('<h2 class="sub-header">Dimensionality Reduction Analysis</h2>', unsafe_allow_html=True)

        method_name = "PCA" if "PCA" in selected_model else "SVD"
        explained_var = model_data['explained_variance']
        n_components = model_data['n_components']

        st.info(
            f"{method_name} Components: {n_components}\n"
            f"Total Variance Explained: {explained_var.sum() * 100:.2f}%"
        )

        fig, var_df = plot_explained_variance(explained_var, n_components, method_name)
        st.pyplot(fig)
        plt.close()
        st.dataframe(var_df, width='stretch')


def render():
    st.markdown('<h1 class="main-header">Prediction Models</h1>', unsafe_allow_html=True)

    # Guard: dataset must be selected and df must exist
    if not st.session_state.get('selected_dataset') or st.session_state.get('df') is None:
            st.markdown("""
            <div class="warning-box">
            <h4>No Dataset Selected</h4>
            <p>Please select a dataset from the "About the Datasets" section first.</p>
            </div>
            """, unsafe_allow_html=True)
            return

    df = st.session_state.df
    dataset_name = st.session_state.dataset_name

    st.markdown(f"""
    <div class="info-box">
      <h3>Training Machine Learning Models</h3>
      <p>
      Multiple machine learning models will now be trained on the {dataset_name} dataset.
      Their performance will be compared to identify the best predictor of student outcomes.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Train button: stores results and other artifacts in session_state and reruns. [web:58][web:102]
    if st.button("Train All Models", type="primary", use_container_width=True, key="train_all_models_btn"):
        with st.spinner("Training models..."):
            X, y, df_processed, missing_info, missing_percent, label_encoders = preprocess_data(df)
            results, scaler, X_train, X_test, y_train, y_test = train_all_models(X, y)
            st.session_state.model_results = results
            st.session_state.scaler = scaler
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.X_features = X
            st.session_state.label_encoders = label_encoders
        st.success("All models trained successfully.")
        st.rerun()

    # Use stateful guard instead of `'model_results' in st.session_state`
    model_results = st.session_state.get("model_results", None)
    if model_results is not None:
        results = st.session_state.model_results
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', unsafe_allow_html=True)

        metrics_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results.keys()],
            'Precision': [results[m]['precision'] for m in results.keys()],
            'Recall': [results[m]['recall'] for m in results.keys()],
            'F1-Score': [results[m]['f1'] for m in results.keys()]
        }).sort_values('Accuracy', ascending=False)

        st.dataframe(
            metrics_df.style.highlight_max(
                axis=0,
                subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                color="#8df0a4"
            ),
            width='stretch'
        )

        # Best model
        best_model = metrics_df.iloc[0]['Model']
        best_accuracy = metrics_df.iloc[0]['Accuracy']

        st.markdown(f"""
        <div class="success-box">
          <h4>Best Performing Model: {best_model}</h4>
          <p>Accuracy: <strong>{best_accuracy:.2%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<h2 class="sub-header">Performance Visualization</h2>', unsafe_allow_html=True)
        fig = plot_performance_bars(metrics_df)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown('<h2 class="sub-header">Detailed Model Analysis</h2>', unsafe_allow_html=True)

        # Stable key so the selected model persists nicely across reruns. [web:82][web:95]
        selected_model = st.selectbox(
            "Select a model for detailed analysis:",
            list(results.keys()),
            key="selected_model_for_details"
        )
        model_data = results[selected_model]
        feature_names = st.session_state.X_features.columns if st.session_state.X_features is not None else []

        render_model_details(selected_model, model_data, feature_names)

        st.markdown("---")
        st.markdown('<h2 class="sub-header">Key Insights</h2>', unsafe_allow_html=True)

        insights = generate_model_insights(model_data)
        for title, description in insights:
            st.markdown(f"""
            <div class="metric-card">
              <h4 style="color: #722F37;">{title}</h4>
              <p style="margin: 0.5rem 0 0 0;">{description}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="success-box">
          <h4>Model Training Complete</h4>
          <p>
          You have trained and compared multiple machine learning models. <br>
          Next Step: Proceed to the <strong>Student Predictor</strong> to use these models for real-time predictions.
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown("""
        <div class="warning-box">
          <h4>No Models Trained Yet</h4>
          <p>Click the "Train All Models" button above to begin training.</p>
        </div>
        """, unsafe_allow_html=True)