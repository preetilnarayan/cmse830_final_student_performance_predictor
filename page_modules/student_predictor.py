import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

plt.style.use('seaborn-v0_8-darkgrid')

# Ensure state keys exist
for key, default in {
    "df": None,
    "dataset_name": None,
    "selected_dataset": None,
    "model_results": None,
    "scaler": None,
    "X_features": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Constants
CONFIDENCE_LEVELS = [
    (80, "Very High", "#28a745", "The model is very confident in this prediction."),
    (60, "High", "#52c77c", "The model has good confidence in this prediction."),
    (50, "Moderate", "#ff9800", "The model has moderate confidence. Consider additional factors."),
    (0, "Low", "#722F37", "The model has low confidence. The outcome is uncertain.")
]

RECOMMENDATIONS = {
    0: {  # Fail
        "title": "At-Risk Student Identified",
        "class": "warning-box",
        "items": [
            "<strong>Early Intervention:</strong> Provide additional academic support immediately",
            "<strong>Tutoring:</strong> Consider one-on-one or group tutoring sessions",
            "<strong>Study Skills:</strong> Help develop better study habits and time management",
            "<strong>Counseling:</strong> Explore any underlying issues affecting performance",
            "<strong>Parent Engagement:</strong> Communicate with parents about concerns and action plan"
        ]
    },
    1: {  # Pass
        "title": "Student on Track for Success",
        "class": "success-box",
        "items": [
            "<strong>Maintain Momentum:</strong> Encourage continued good performance",
            "<strong>Challenge:</strong> Consider advanced or enrichment opportunities",
            "<strong>Peer Support:</strong> Student could mentor struggling classmates",
            "<strong>Goal Setting:</strong> Help set higher academic goals",
            "<strong>Recognition:</strong> Acknowledge achievements to boost motivation"
        ]
    }
}


def get_confidence_level(confidence):
    """Get confidence level description."""
    for threshold, level, color, desc in CONFIDENCE_LEVELS:
        if confidence > threshold:
            return level, color, desc
    return CONFIDENCE_LEVELS[-1][1:]


def plot_probability_bars(prediction_proba):
    """Create probability bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Fail', 'Pass']
    probabilities = [prediction_proba[0] * 100, prediction_proba[1] * 100]
    colors = ['#722F37', '#28a745']

    bars = ax.barh(
        categories,
        probabilities,
        color=colors,
        alpha=0.8,
        edgecolor='#441C21',
        linewidth=2
    )

    ax.set_xlabel('Probability (%)', fontsize=13, fontweight='bold')
    ax.set_title('Prediction Confidence Breakdown', fontsize=15, fontweight='bold', color='#441C21')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)

    for bar, prob in zip(bars, probabilities):
        ax.text(
            prob + 2,
            bar.get_y() + bar.get_height() / 2.,
            f'{prob:.1f}%',
            ha='left',
            va='center',
            fontweight='bold',
            fontsize=14,
            color='#441C21'
        )

    ax.axvline(x=50, color='#441C21', linestyle='--', linewidth=2, label='Decision Threshold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    return fig


def plot_top_features(feature_names, importances, top_n=5):
    """Create top features importance chart."""
    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = [importances[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features, top_importances, color='#722F37', alpha=0.8)

    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold', color='#441C21')
    ax.grid(axis='x', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2.,
            f'{width:.3f}',
            ha='left',
            va='center',
            fontweight='bold',
            fontsize=10,
            color='#441C21'
        )

    plt.tight_layout()
    return fig


def prepare_input_data(input_data, df, feature_cols):
    """Prepare and encode input data for prediction."""
    input_df = pd.DataFrame([input_data])

    # Encode categorical variables
    for col_name in input_df.columns:
        if input_df[col_name].dtype == 'object':
            le = LabelEncoder()
            le.fit(df[col_name].dropna())
            input_df[col_name] = le.transform(input_df[col_name])

    # Ensure same columns and order as training
    return input_df.reindex(columns=feature_cols)


def apply_dimensionality_reduction(input_scaled, model_choice, model_data):
    """Apply PCA/SVD if needed."""
    if 'PCA' in model_choice and 'pca' in model_data:
        return model_data['pca'].transform(input_scaled)
    elif 'SVD' in model_choice and 'svd' in model_data:
        return model_data['svd'].transform(input_scaled)
    return input_scaled


def render_result_card(prediction, confidence):
    """Render main prediction result card."""
    result_text = "Pass" if prediction == 1 else "Fail"
    result_color = "#28a745" if prediction == 1 else "#722F37"
    result_gradient = "#52c77c" if prediction == 1 else "#AA8287"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {result_color} 0%, {result_gradient} 100%);
                padding: 3rem; border-radius: 1rem; text-align: center;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0;">
      <h1 style="color: white; margin: 0; font-size: 3rem;">Predicted: {result_text}</h1>
      <h2 style="color: white; margin-top: 1rem; font-size: 2rem;">Confidence: {confidence:.1f}%</h2>
    </div>
    """, unsafe_allow_html=True)


def render_probability_cards(prediction_proba, prediction):
    """Render probability cards for fail/pass."""
    col1, col2 = st.columns(2)

    probs = [
        (col1, "Fail", 0, "#722F37"),
        (col2, "Pass", 1, "#28a745")
    ]

    for col, label, idx, color in probs:
        with col:
            bg_color = '#d4edda' if prediction == idx else '#f8d7da'
            st.markdown(f"""
            <div class="metric-card" style="background-color: {bg_color};">
              <h3 style="color: {color}; text-align: center;">{label} Probability</h3>
              <h1 style="color: {color}; text-align: center; margin: 0;">{prediction_proba[idx]*100:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)


def render_recommendations(prediction):
    """Render recommendations based on prediction."""
    rec = RECOMMENDATIONS[prediction]
    items_html = "".join(f"<li>{item}</li>" for item in rec["items"])

    st.markdown(f"""
    <div class="{rec['class']}">
      <h4>{rec['title']}</h4>
      <ul>{items_html}</ul>
    </div>
    """, unsafe_allow_html=True)


def render_input_form(df, feature_cols):
    """Render interactive inputs for each feature."""
    inputs = {}
    for feature in feature_cols:
        if feature not in df.columns:
            # Skip features not present in this dataset
            continue

        if df[feature].dtype == 'object':
            options = sorted(df[feature].dropna().unique())
            inputs[feature] = st.selectbox(
                feature,
                options,
                key=f"predict_input_{feature}"
            )
        else:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            default_val = float(df[feature].median())
            inputs[feature] = st.slider(
                feature,
                min_val,
                max_val,
                default_val,
                key=f"predict_slider_{feature}"
            )

    return inputs


def render():
    """Main render function for student predictor page."""
    st.markdown('<h1 class="main-header">Student Predictor</h1>', unsafe_allow_html=True)

    # Check prerequisites
    if not st.session_state.get('selected_dataset') or st.session_state.get('df') is None:
        st.markdown("""
        <div class="warning-box">
          <h4>No Dataset Selected</h4>
          <p>Please select a dataset from the <strong>"Dataset & Model Prediction"</strong> page first.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    model_results = st.session_state.get("model_results", None)
    if model_results is None:
        st.markdown("---")
        st.markdown("""
        <div class="warning-box">
        <h4>No Models Trained Yet</h4>
        <p>Please go to the <strong>Prediction Models</strong> tab and click "Train All Models" first.</p>
        </div>
        """, unsafe_allow_html=True)
        return


    df = st.session_state.df
    dataset_name = st.session_state.dataset_name
    results = st.session_state.model_results
    scaler = st.session_state.scaler
    X_features = st.session_state.X_features

    st.markdown(f"""
    <div class="info-box">
      <h3>Make Individual Predictions</h3>
      <p>
      Use trained models to predict whether a student will pass or fail based on their
      characteristics. Enter student information below to get a prediction.
      </p>
      <p><strong>Dataset:</strong> {dataset_name}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model Selection
    st.markdown('<h2 class="sub-header">Select Prediction Model</h2>', unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Choose a model for prediction:",
        list(results.keys()),
        key="predict_model_choice",
        help="Select the machine learning model you want to use for prediction"
    )

    selected_model_data = results[model_choice]

    # Display model performance
    cols = st.columns(3)
    metrics = [
        ("Model Accuracy", f"{selected_model_data['accuracy']:.1%}"),
        ("Precision", f"{selected_model_data['precision']:.3f}"),
        ("Recall", f"{selected_model_data['recall']:.3f}")
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

    st.markdown("---")

    # Input Features
    st.markdown('<h2 class="sub-header">Enter Student Information</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      <p>Fill in the student's information below. The model will use these features to predict their outcome.</p>
    </div>
    """, unsafe_allow_html=True)

    feature_cols = list(X_features.columns) if X_features is not None else []
    input_data = render_input_form(df, feature_cols)

    st.markdown("---")

    # Predict Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "Predict Student Outcome",
            type="primary",
            use_container_width=True,
            key="predict_student_outcome_btn"
        )

    if predict_button and feature_cols:
        # Prepare and transform input
        input_df = prepare_input_data(input_data, df, feature_cols)
        input_scaled = scaler.transform(input_df)
        input_scaled = apply_dimensionality_reduction(input_scaled, model_choice, selected_model_data)

        # Make prediction
        model = selected_model_data['model']
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        confidence = prediction_proba[prediction] * 100

        # Display results
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Prediction Result</h2>', unsafe_allow_html=True)

        render_result_card(prediction, confidence)

        # Detailed probabilities
        st.markdown("#### Prediction Probabilities")
        render_probability_cards(prediction_proba, prediction)

        # Probability visualization
        fig = plot_probability_bars(prediction_proba)
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # Interpretation
        st.markdown('<h2 class="sub-header">Interpretation</h2>', unsafe_allow_html=True)

        confidence_level, confidence_color, confidence_desc = get_confidence_level(confidence)

        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {confidence_color};">
          <h4 style="color: {confidence_color};">Confidence Level: {confidence_level}</h4>
          <p>{confidence_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations
        st.markdown("#### Recommendations")
        render_recommendations(prediction)

        st.markdown("---")

        # Feature Contributions
        if 'feature_importance' in selected_model_data and X_features is not None:
            st.markdown("#### Key Factors Influencing This Prediction")

            fig = plot_top_features(X_features.columns, selected_model_data['feature_importance'])
            st.pyplot(fig)
            plt.close()

            st.info("These features had the most influence on the model's decision across all predictions.")

        st.markdown("---")

        # Model Information
        st.markdown("#### Model Information")

        st.markdown(f"""
        <div class="info-box">
          <p><strong>Model Used:</strong> {model_choice}</p>
          <p><strong>Model Accuracy:</strong> {selected_model_data['accuracy']:.1%}</p>
          <p><strong>Dataset:</strong> {dataset_name}</p>
          <p><strong>Prediction Threshold:</strong> 50%</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-box">
          <h4>Ready to Predict</h4>
          <p>
          Enter all the student information above and click the <strong>"Predict Student Outcome"</strong>
          button to get a prediction.
          </p>
        </div>
        """, unsafe_allow_html=True)
