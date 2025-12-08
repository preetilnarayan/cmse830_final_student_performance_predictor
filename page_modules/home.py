import streamlit as st

# Constants
JOURNEY_STEPS = [
    ("1. About the Datasets", "Explore and select from three real-world student performance datasets"),
    ("2. Cleaning", "Learn about data preprocessing and transformation techniques"),
    ("3. Missingness", "Analyze missing data patterns and imputation strategies"),
    ("4. EDA", "Perform exploratory data analysis with interactive visualizations"),
    ("5. Prediction Models", "Compare multiple machine learning models and their performance"),
    ("6. Student Predictor", "Make real-time predictions for individual students")
]

FEATURES = {
    "Datasets": [
        "Mathematics Performance",
        "Portuguese Language Performance",
        "Comprehensive Academic Profile"
    ],
    "ML Models": [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "PCA & SVD Analysis"
    ],
    "Analytics": [
        "Missing Value Analysis",
        "Feature Importance",
        "Interactive Visualizations",
        "Real-time Predictions"
    ]
}

USERS = [
    ("School Administrators", "School administrators can use model outputs and dashboards to understand how academic performance, attendance, and behavioral indicators are distributed across grades, schools, or programs."),
    ("Teachers & Educators", "Teachers and classroom instructors can review risk indicators and prediction results to spot students who are trending toward failure before grades are finalized"),
    ("Academic Counselors", "Academic and guidance counselors can combine individual predictions with contextual features (such as prior failures, absences, or support needs) to design highly personalized intervention plans."),
    ("Education Researchers", "Education researchers can use the integrated datasets, model explanations, and feature importance views to study how demographic, behavioral, and academic variables jointly influence success.")
]

STATS = [
    ("Datasets Available", "3", "Three comprehensive student performance datasets"),
    ("ML Models", "5", "Five different machine learning algorithms"),
    ("Total Records", "1,544", "Combined student records across all datasets"),
    ("Features Analyzed", "25+", "Various academic and behavioral indicators")
]


def create_box(box_type, title, content):
    """Create a styled info/warning/success box"""
    return f"""
    <div class="{box_type}">
    <h3>{title}</h3>
    <p style="font-size: 1.1rem;">{content}</p>
    </div>
    """


def create_metric_card(title, items):
    """Create a metric card with list items"""
    items_html = "".join(f"<li>{item}</li>" for item in items)
    return f"""
    <div class="metric-card">
    <h3 style="color: #722F37;">{title}</h3>
    <ul>{items_html}</ul>
    </div>
    """


def create_user_card(title, desc):
    """Create a user benefit card"""
    return f"""
    <div class="metric-card" style="text-align: left; min-height: 210px;">
        <h3 style="color: #722F37; margin-top: 0;">
            {title}
        </h3>
        <p style="font-size: 0.95rem; margin-bottom: 0; color: #000000;">{desc}</p>
    </div>
    """


def render():
    """Render the home page"""
    # Welcome
    st.markdown(create_box(
        "info-box",
        "Welcome to the Student Performance Predictor!",
        "This application uses advanced machine learning algorithms to predict whether a student will "
        "pass or fail based on various performance metrics and behavioral factors. Follow the storyline "
        "through the interactive analysis journey!"
    ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Features
    st.markdown('<h2 class="sub-header">Key Features</h2>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (title, items) in zip(cols, FEATURES.items()):
        with col:
            st.markdown(create_metric_card(title, items), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Problem Statement
    st.markdown('<h2 class="sub-header"> Problem Statement</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-box">
        <h4>The Challenge</h4>
        <p>Educational institutions face a critical challenge identifying at-risk students early 
        enough to provide meaningful interventions. Traditional methods often rely on subjective 
        assessments or react only after students have already failed.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box">
        <h4>Our Solution</h4>
        <p>A data-driven machine learning system that analyzes student performance metrics, 
        behavioral patterns, and demographic factors to predict outcomes with high accuracy, 
        enabling proactive support and intervention strategies.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Target Users
    st.markdown('<h2 class="sub-header">Who Benefits?</h2>', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    # First row: 0, 1
    with col_left:
        st.markdown(create_user_card(*USERS[0]), unsafe_allow_html=True)
    with col_right:
        st.markdown(create_user_card(*USERS[1]), unsafe_allow_html=True)

    # Second row: 2, 3
    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.markdown(create_user_card(*USERS[2]), unsafe_allow_html=True)
    with col_right2:
        st.markdown(create_user_card(*USERS[3]), unsafe_allow_html=True)

    
    st.markdown("---")
    
    # Get Started
    st.markdown('<h2 class="sub-header">Ready to Begin your Analysis Journey?</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <h4>Start by exploring our datasets!</h4>
    <p style="font-size: 1.1rem;">
    Navigate to <strong>"Dataset & Model Prediction"</strong> in the sidebar to select a dataset 
    and begin your analysis journey. Each dataset offers unique insights into student performance 
    from different academic contexts.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    for title, description in JOURNEY_STEPS:
        col_title, col_desc = st.columns([3, 6])   # unpack them properly

        with col_title:
            st.markdown(f"<h4 style='color:#722F37;'>{title}</h4>", unsafe_allow_html=True)

        with col_desc:
            st.markdown(
                f"<p style='margin-top:0.3rem; color:#441C21;'>{description}</p>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Quick Stats
    st.markdown('<h2 class="sub-header">Quick Stats</h2>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (label, value, help_text) in zip(cols, STATS):
        with col:
            st.metric(label, value, help=help_text)