import streamlit as st
from utils.data_loader import load_dataset_1, load_dataset_2, load_dataset_3
from page_modules import home, about_datasets, cleaning, missingness, eda, prediction_models, student_predictor, documentation
from utils.styles import apply_custom_styles

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Apply custom CSS once
apply_custom_styles()

# Initialize session state with defaults
default_state = {
    "selected_dataset": None,
    "df": None,
    "dataset_name": None,
    "selected_sidebar_page": "Home",       # which main page (radio)
    "selected_dataset_tab": "About Dataset",  # which inner tab
    "model_results": None,        
    "trained_model": None, 
}
for key, default in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Dataset configuration mapping
DATASET_CONFIG = {
    "Mathematics Performance": {"loader": load_dataset_1, "id": 1},
    "Portuguese Language Performance": {"loader": load_dataset_2, "id": 2},
    "Comprehensive Academic Profile": {"loader": load_dataset_3, "id": 3}
}

def reset_dataset_state():
    """Reset all dataset-related state variables when dataset changes"""
    keys_to_reset = [
        # Model-related states
        "trained_model",
        "model_metrics",
        "selected_model",
        "best_model",
        "model_results",
        "prediction_result",
        "last_prediction",
        
        # Data processing states
        "cleaned_df",
        "processed_df",
        "train_test_split_done",
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        
        # Analysis states
        "eda_cache",
        "missingness_analysis",
        "cleaning_applied",
        
        # User inputs for predictor
        "predictor_inputs",
        "feature_values",
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=200)

    page = st.radio(
        "Go to:",
        ["Home", "Dataset & Model Prediction", "Documentation"],
        index=["Home", "Dataset & Model Prediction", "Documentation"]
        .index(st.session_state.selected_sidebar_page),
        key="sidebar_page_radio"
    )
    # persist sidebar page in state
    st.session_state.selected_sidebar_page = page

    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This application predicts student performance using Machine Learning"
    )

# HEADER
st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# PAGE RENDERING

if page == "Home":
    home.render()

elif page == "Dataset & Model Prediction":
    st.markdown("## Dataset & Model Prediction")

    # Info box
    st.markdown(
        "<div class='info-box'>"
        "<h3>Dataset Information</h3>"
        "<p>The datasets are derived from the UCI Machine Learning Repository and contain "
        "real student performance data from Portuguese secondary schools.</p>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Dataset selector
    dataset_option = st.selectbox(
        "Select a dataset to work with:",
        ["Select a dataset...", *DATASET_CONFIG.keys()],
        index=0
    )

    # Load dataset if selected
    if dataset_option in DATASET_CONFIG:
        config = DATASET_CONFIG[dataset_option]

        # Load dataset only if it changed
        if st.session_state.dataset_name != dataset_option:
            st.session_state.df = config["loader"]()
            st.session_state.selected_dataset = config["id"]
            st.session_state.dataset_name = dataset_option
            # reset active inner tab when dataset changes
            st.session_state.selected_dataset_tab = "About Dataset"
            # Reset all model predictions and functionality
            reset_dataset_state()

        tab_labels = [
            "About Dataset",
            "EDA",
            "Cleaning",
            "Missingness",
            "Prediction Models",
            "Student Predictor"
        ]

        # figure out which tab should be initially active
        try:
            default_tab_index = tab_labels.index(st.session_state.selected_dataset_tab)
        except ValueError:
            default_tab_index = 0

        tabs = st.tabs(tab_labels)

        tab_modules = [about_datasets, eda, cleaning, missingness, prediction_models, student_predictor]

        for i, (tab, module) in enumerate(zip(tabs, tab_modules)):
            with tab:
                # when this tab's content is rendered, mark it active
                if i == default_tab_index:
                    st.session_state.selected_dataset_tab = tab_labels[i]
                module.render()

    else:
        st.markdown(
            "<div class='warning-box'>Please select a dataset from the dropdown to continue.</div>",
            unsafe_allow_html=True
        )

elif page == "Documentation":
    documentation.render()

# FOOTER
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#AA8287; padding: 1rem 0;'>"
    "Student Performance Prediction System | Powered by Machine Learning"
    "</div>",
    unsafe_allow_html=True
)