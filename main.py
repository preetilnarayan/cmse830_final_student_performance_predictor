import streamlit as st
import pandas as pd
import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load real datasets with intentional missing values
@st.cache_data
def load_dataset_1():
    """Dataset 1: UCI Student Performance - Mathematics (with missing values)"""
    try:
        # Load from UCI repository
        url = "https://raw.githubusercontent.com/uciml/student-performance-dataset/master/student-mat.csv"
        df = pd.read_csv(url, sep=';')
        
        # Select relevant columns and rename
        df_subset = df[['age', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']].copy()
        df_subset.columns = ['age', 'study_time_weekly', 'past_failures', 'absences', 
                            'period1_grade', 'period2_grade', 'final_grade']
        
        # Introduce missing values strategically (10-15% missing)
        np.random.seed(42)
        for col in ['study_time_weekly', 'absences', 'period1_grade']:
            missing_idx = np.random.choice(df_subset.index, size=int(len(df_subset) * 0.12), replace=False)
            df_subset.loc[missing_idx, col] = np.nan
        
        # Create pass/fail based on final grade (>=10 is pass in Portuguese system)
        df_subset['result'] = df_subset['final_grade'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
        
        return df_subset
    except:
        # Fallback to synthetic data if download fails
        return generate_fallback_dataset_1()

@st.cache_data
def load_dataset_2():
    """Dataset 2: UCI Student Performance - Portuguese (with missing values)"""
    try:
        url = "https://raw.githubusercontent.com/uciml/student-performance-dataset/master/student-por.csv"
        df = pd.read_csv(url, sep=';')
        
        # Select different columns for variety
        df_subset = df[['age', 'Medu', 'Fedu', 'traveltime', 'famrel', 'goout', 'health', 'absences', 'G3']].copy()
        df_subset.columns = ['age', 'mother_education', 'father_education', 'travel_time', 
                            'family_relationship', 'social_outings', 'health_status', 'absences', 'final_grade']
        
        # Introduce missing values (8-12% missing)
        np.random.seed(123)
        for col in ['mother_education', 'health_status', 'social_outings', 'travel_time']:
            missing_idx = np.random.choice(df_subset.index, size=int(len(df_subset) * 0.10), replace=False)
            df_subset.loc[missing_idx, col] = np.nan
        
        df_subset['result'] = df_subset['final_grade'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
        
        return df_subset
    except:
        return generate_fallback_dataset_2()

@st.cache_data
def load_dataset_3():
    """Dataset 3: Extended Student Performance with Multiple Subjects (with missing values)"""
    try:
        # Load both math and portuguese, combine for comprehensive dataset
        url_math = "https://raw.githubusercontent.com/uciml/student-performance-dataset/master/student-mat.csv"
        url_por = "https://raw.githubusercontent.com/uciml/student-performance-dataset/master/student-por.csv"
        
        df_math = pd.read_csv(url_math, sep=';')
        df_por = pd.read_csv(url_por, sep=';')
        
        # Take subset from each
        df_math_subset = df_math[['age', 'studytime', 'failures', 'schoolsup', 'famsup', 
                                   'paid', 'activities', 'higher', 'internet', 'G3']].head(250)
        df_por_subset = df_por[['age', 'studytime', 'failures', 'schoolsup', 'famsup', 
                                'paid', 'activities', 'higher', 'internet', 'G3']].head(250)
        
        df_combined = pd.concat([df_math_subset, df_por_subset], ignore_index=True)
        
        df_combined.columns = ['age', 'study_time_weekly', 'past_failures', 'school_support', 
                              'family_support', 'paid_classes', 'extracurricular', 
                              'higher_ed_aspiration', 'internet_access', 'final_grade']
        
        # Convert yes/no to 1/0
        for col in ['school_support', 'family_support', 'paid_classes', 'extracurricular', 
                    'higher_ed_aspiration', 'internet_access']:
            df_combined[col] = df_combined[col].map({'yes': 1, 'no': 0})
        
        # Introduce missing values (15-20% missing in some columns)
        np.random.seed(456)
        high_missing_cols = ['paid_classes', 'extracurricular', 'study_time_weekly']
        moderate_missing_cols = ['school_support', 'internet_access']
        
        for col in high_missing_cols:
            missing_idx = np.random.choice(df_combined.index, size=int(len(df_combined) * 0.18), replace=False)
            df_combined.loc[missing_idx, col] = np.nan
        
        for col in moderate_missing_cols:
            missing_idx = np.random.choice(df_combined.index, size=int(len(df_combined) * 0.10), replace=False)
            df_combined.loc[missing_idx, col] = np.nan
        
        df_combined['result'] = df_combined['final_grade'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
        
        return df_combined
    except:
        return generate_fallback_dataset_3()

# Fallback synthetic datasets (in case download fails)
def generate_fallback_dataset_1():
    """Fallback for Dataset 1 with missing values"""
    np.random.seed(42)
    n = 395
    
    df = pd.DataFrame({
        'age': np.random.randint(15, 23, n),
        'study_time_weekly': np.random.randint(1, 5, n),
        'past_failures': np.random.randint(0, 4, n),
        'absences': np.random.randint(0, 40, n),
        'period1_grade': np.random.randint(0, 20, n),
        'period2_grade': np.random.randint(0, 20, n),
        'final_grade': np.random.randint(0, 20, n)
    })
    
    # Add missing values
    for col in ['study_time_weekly', 'absences', 'period1_grade']:
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.12), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    df['result'] = df['final_grade'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
    return df

def generate_fallback_dataset_2():
    """Fallback for Dataset 2 with missing values"""
    np.random.seed(123)
    n = 649
    
    df = pd.DataFrame({
        'age': np.random.randint(15, 23, n),
        'mother_education': np.random.randint(0, 5, n),
        'father_education': np.random.randint(0, 5, n),
        'travel_time': np.random.randint(1, 5, n),
        'family_relationship': np.random.randint(1, 6, n),
        'social_outings': np.random.randint(1, 6, n),
        'health_status': np.random.randint(1, 6, n),
        'absences': np.random.randint(0, 40, n),
        'final_grade': np.random.randint(0, 20, n)
    })
    
    # Add missing values
    for col in ['mother_education', 'health_status', 'social_outings', 'travel_time']:
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.10), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    df['result'] = df['final_grade'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
    return df

def generate_fallback_dataset_3():
    """Fallback for Dataset 3 with missing values"""
    np.random.seed(456)
    n = 500
    
    df = pd.DataFrame({
        'age': np.random.randint(15, 23, n),
        'study_time_weekly': np.random.randint(1, 5, n),
        'past_failures': np.random.randint(0, 4, n),
        'school_support': np.random.randint(0, 2, n),
        'family_support': np.random.randint(0, 2, n),
        'paid_classes': np.random.randint(0, 2, n),
        'extracurricular': np.random.randint(0, 2, n),
        'higher_ed_aspiration': np.random.randint(0, 2, n),
        'internet_access': np.random.randint(0, 2, n),
        'final_grade': np.random.randint(0, 20, n)
    })
    
    # Add missing values
    for col in ['paid_classes', 'extracurricular', 'study_time_weekly']:
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.18), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    for col in ['school_support', 'internet_access']:
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.10), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    df['result'] = df['final_grade'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
    return df

def preprocess_data(df, target_col='result'):
    """Preprocess the dataset for modeling with comprehensive missing value handling"""
    df_processed = df.copy()
    
    # Store original missing value info for display
    missing_info = df_processed.isnull().sum()
    missing_percent = (missing_info / len(df_processed) * 100).round(2)
    
    # Handle missing values with multiple strategies
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # For numeric columns: use median imputation (more robust to outliers)
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            median_value = df_processed[col].median()
            df_processed[col].fillna(median_value, inplace=True)
    
    # For categorical columns: use mode imputation
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
            df_processed[col].fillna(mode_value, inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    label_encoders = {}
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object' and col != target_col:
            label_encoders[col] = LabelEncoder()
            df_processed[col] = label_encoders[col].fit_transform(df_processed[col])
    
    # Separate features and target
    X = df_processed.drop(target_col, axis=1)
    y = le.fit_transform(df_processed[target_col])
    
    return X, y, df_processed, missing_info, missing_percent

def train_models(X, y):
    """Train multiple models and return results"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    return results, scaler, X_train, X_test, y_train, y_test

# Initialize session state
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = 'Dataset 1'

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=80)
    st.title("🎓 Navigation")
    
    page = st.radio(
        "Select Section:",
        ["🏠 Home", "📊 Product Overview", "🔬 Data Science Analysis", "🎯 Model Prediction", "📈 Evaluation Metrics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Selection")
    dataset_choice = st.selectbox(
        "Choose Dataset:",
        ["Dataset 1: UCI Math Performance", 
         "Dataset 2: UCI Portuguese Performance", 
         "Dataset 3: Comprehensive Academic (Combined)"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This app predicts student pass/fail outcomes using machine learning.
    
    **Features:**
    - 3 Different datasets
    - Multiple ML models
    - Interactive visualizations
    - Real-time predictions
    """)

# Load selected dataset
if "Dataset 1" in dataset_choice:
    with st.spinner("Loading UCI Student Performance Dataset (Math)..."):
        df = load_dataset_1()
    dataset_name = "UCI Mathematics Performance"
    dataset_source = "UCI Machine Learning Repository"
elif "Dataset 2" in dataset_choice:
    with st.spinner("Loading UCI Student Performance Dataset (Portuguese)..."):
        df = load_dataset_2()
    dataset_name = "UCI Portuguese Performance"
    dataset_source = "UCI Machine Learning Repository"
else:
    with st.spinner("Loading Comprehensive Student Dataset..."):
        df = load_dataset_3()
    dataset_name = "Comprehensive Academic Profile"
    dataset_source = "UCI ML Repository (Combined)"

# Main content
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🎓 Student Performance Prediction System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="success-box">
        <h3>Welcome to the Student Performance Predictor!</h3>
        <p>This application uses advanced machine learning algorithms to predict whether a student will pass or fail based on various performance metrics and behavioral factors.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Multiple Datasets")
        st.write("""
        - **Dataset 1**: Study hours and attendance patterns
        - **Dataset 2**: Demographics and behavioral factors
        - **Dataset 3**: Comprehensive academic scores
        """)
    
    with col2:
        st.markdown("### 🤖 ML Models")
        st.write("""
        - Random Forest Classifier
        - Gradient Boosting
        - Logistic Regression
        - Model comparison tools
        """)
    
    with col3:
        st.markdown("### 📈 Analytics")
        st.write("""
        - Feature importance analysis
        - Interactive visualizations
        - Performance metrics
        - ROC curves and confusion matrices
        """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **Select a Dataset** from the sidebar
    2. **Navigate** to different sections using the menu
    3. **Explore** the data in Product Overview
    4. **Analyze** features in Data Science Analysis
    5. **Make Predictions** in Model Prediction
    6. **Review Metrics** in Evaluation Metrics
    """)

elif page == "📊 Product Overview":
    st.markdown(f'<h1 class="main-header">📊 Product Overview: {dataset_name}</h1>', unsafe_allow_html=True)
    
    # Problem Definition
    st.markdown("### 🎯 Problem Statement")
    st.markdown("""
    <div class="warning-box">
    <strong>Business Problem:</strong> Educational institutions need to identify at-risk students early to provide timely interventions and improve overall student success rates.
    <br><br>
    <strong>Solution:</strong> A machine learning-based prediction system that analyzes student performance metrics to forecast pass/fail outcomes.
    <br><br>
    <strong>Target Users:</strong> School administrators, counselors, and educators who want to proactively support struggling students.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Overview
    st.markdown("### 📋 Dataset Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        pass_rate = (df['result'] == 'Pass').sum() / len(df) * 100
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col4:
        st.metric("Fail Rate", f"{100-pass_rate:.1f}%")
    with col5:
        missing_count = df.isnull().sum().sum()
        st.metric("Missing Values", missing_count)
    
    # Data source information
    st.info(f"**📚 Data Source:** {dataset_source} | **Dataset:** {dataset_name}")
    
    st.markdown("---")
    
    # Display dataset
    st.markdown("### 📊 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Dataset",
        data=csv,
        file_name=f"{dataset_name.replace(' ', '_')}.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Basic Statistics
    st.markdown("### 📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Class Distribution
    st.markdown("### 🎯 Target Variable Distribution")
    fig, ax = plt.subplots()
    df['result'].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%',
        colors=['#28a745', '#dc3545'],
        startangle=90,
        ax=ax
    )
    ax.set_ylabel('')
    ax.set_title('Pass vs Fail Distribution')
    st.pyplot(fig)


elif page == "🔬 Data Science Analysis":
    st.markdown(f'<h1 class="main-header">🔬 Data Science Analysis: {dataset_name}</h1>', unsafe_allow_html=True)
    
    # Data Processing
    with st.expander("🔧 Data Processing & Cleaning", expanded=True):
        # Show missing values BEFORE cleaning
        st.markdown("#### 📊 Missing Values Analysis (Before Cleaning)")
        missing_before = df.isnull().sum()
        missing_percent_before = (missing_before / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing_before.index,
            'Missing Count': missing_before.values,
            'Missing %': missing_percent_before.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            # Visualize missing values
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(missing_df['Column'], missing_df['Missing %'], color='red')
            ax.set_title('Missing Values by Column (%)')
            ax.set_xlabel('Column')
            ax.set_ylabel('Missing %')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.success("✅ No missing values detected!")
        
        st.markdown("---")
        
        st.markdown("""
        **Data Cleaning Steps Applied:**
        1. ✅ **Missing Value Detection**: Identified columns with missing data
        2. ✅ **Imputation Strategy**: 
           - Numeric columns: Filled with **median** (robust to outliers)
           - Categorical columns: Filled with **mode** (most frequent value)
        3. ✅ **Categorical Encoding**: Used Label Encoding for categorical variables
        4. ✅ **Feature Scaling**: Applied StandardScaler for model training
        5. ✅ **Train-Test Split**: 80% training, 20% testing with stratification
        
        **Why Median over Mean?**
        - Median is less affected by extreme outliers
        - More representative of central tendency in skewed distributions
        - Better for handling missing values in educational data
        """)
        
        # Show data AFTER cleaning
        st.markdown("#### ✨ Data After Cleaning")
        df_display = df.copy()
        
        # Apply same cleaning for display
        numeric_cols = df_display.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_display[col].isnull().sum() > 0:
                df_display[col].fillna(df_display[col].median(), inplace=True)
        
        categorical_cols = df_display.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_display[col].isnull().sum() > 0:
                df_display[col].fillna(df_display[col].mode()[0], inplace=True)
        
        st.dataframe(df_display.head(10), use_container_width=True)
        st.success(f"✅ All {len(df)} records cleaned and ready for analysis!")
    
    # Feature Analysis
    st.markdown("---")
    st.markdown("### 📊 Feature Analysis")
    
    # Correlation heatmap
    numeric_df = df.copy()
    le = LabelEncoder()
    for col in numeric_df.columns:
        if numeric_df[col].dtype == 'object':
            numeric_df[col] = le.fit_transform(numeric_df[col])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    plt.title('Feature Correlation Matrix')
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Distribution plots
    st.markdown("### 📊 Feature Distributions")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    col1, col2 = st.columns(2)
    for idx, col in enumerate(numeric_cols):
        with col1 if idx % 2 == 0 else col2:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=col, hue='result', kde=False, ax=ax)
            ax.set_title(f'Distribution of {col}')
            st.pyplot(fig)
    
    # Box plots
    st.markdown("### 📦 Feature Comparison (Pass vs Fail)")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='result', y=col, ax=ax)
        ax.set_title(f'{col} by Result')
        st.pyplot(fig)

elif page == "🎯 Model Prediction":
    st.markdown(f'<h1 class="main-header">🎯 Model Training & Prediction</h1>', unsafe_allow_html=True)
    
    # Train models
    with st.spinner("Training models... Please wait."):
        X, y, df_processed, missing_info, missing_percent = preprocess_data(df)
        results, scaler, X_train, X_test, y_train, y_test = train_models(X, y)
    
    st.success("✅ Models trained successfully!")
    
    # Model selection
    st.markdown("### 🤖 Select Model for Prediction")
    model_choice = st.selectbox("Choose a model:", list(results.keys()))
    
    selected_model = results[model_choice]['model']
    
    st.markdown("---")
    
    # Feature Importance
    st.markdown("### 📊 Feature Importance Analysis")
    
    if hasattr(selected_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': selected_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title=f'Feature Importance - {model_choice}',
                    color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")
    
    st.markdown("---")
    
    # Make Predictions
    st.markdown("### 🎯 Make Individual Predictions")
    
    st.markdown("""
    <div class="warning-box">
    <strong>Instructions:</strong> Enter student information below to predict pass/fail outcome.
    </div>
    """, unsafe_allow_html=True)
    
    # Create input fields based on dataset
    input_data = {}
    
    cols = st.columns(3)
    for idx, col in enumerate(X.columns):
        with cols[idx % 3]:
            if df[col].dtype == 'object':
                unique_vals = df[col].unique()
                input_data[col] = st.selectbox(f"{col}", unique_vals)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_data[col] = st.slider(f"{col}", min_val, max_val, mean_val)
    
    if st.button("🔮 Predict Result", type="primary"):
        # Prepare input
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical
        le = LabelEncoder()
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                # Fit on original data
                le.fit(df[col])
                input_df[col] = le.transform(input_df[col])
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = selected_model.predict(input_scaled)[0]
        probability = selected_model.predict_proba(input_scaled)[0]
        
        # Display result
        result_text = "Pass" if prediction == 1 else "Fail"
        result_color = "#28a745" if prediction == 1 else "#dc3545"
        confidence = probability[prediction] * 100
        
        st.markdown("---")
        st.markdown("### 🎉 Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="background-color: {result_color}; padding: 2rem; border-radius: 1rem; text-align: center;">
                <h1 style="color: white; margin: 0;">Result: {result_text}</h1>
                <h3 style="color: white; margin-top: 1rem;">Confidence: {confidence:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability breakdown
        st.markdown("### 📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Outcome': ['Fail', 'Pass'],
            'Probability': probability * 100
        })
        
        fig = px.bar(prob_df, x='Outcome', y='Probability',
                    title='Prediction Confidence',
                    color='Outcome',
                    color_discrete_map={'Pass': '#28a745', 'Fail': '#dc3545'})
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Evaluation Metrics":
    st.markdown(f'<h1 class="main-header">📈 Model Evaluation Metrics</h1>', unsafe_allow_html=True)
    
    # Train models
    with st.spinner("Calculating metrics... Please wait."):
        X, y, df_processed, missing_info, missing_percent = preprocess_data(df)
        results, scaler, X_train, X_test, y_train, y_test = train_models(X, y)
    
    st.success("✅ Evaluation complete!")
    
    # Model Comparison
    st.markdown("### 🏆 Model Performance Comparison")
    
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results.keys()],
        'Precision': [results[m]['precision'] for m in results.keys()],
        'Recall': [results[m]['recall'] for m in results.keys()],
        'F1-Score': [results[m]['f1'] for m in results.keys()]
    })
    
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']), 
                 use_container_width=True)
    
    # Visualize metrics
    fig = px.bar(metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                x='Model', y='Score', color='Metric', barmode='group',
                title='Model Performance Metrics Comparison')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed metrics for each model
    st.markdown("### 📊 Detailed Model Analysis")
    
    selected_model_eval = st.selectbox("Select model for detailed analysis:", list(results.keys()))
    
    model_results = results[selected_model_eval]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{model_results['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{model_results['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{model_results['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{model_results['f1']:.3f}")
    
    # Confusion Matrix
    st.markdown("### 🎯 Confusion Matrix")
    
    cm = model_results['confusion_matrix']
    fig = px.imshow(cm, text_auto=True, 
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Fail', 'Pass'], y=['Fail', 'Pass'],
                   title=f'Confusion Matrix - {selected_model_eval}',
                   color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curve
    st.markdown("### 📈 ROC Curve")
    
    fpr, tpr, _ = roc_curve(model_results['y_test'], model_results['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {roc_auc:.3f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', 
                            line=dict(dash='dash', color='gray')))
    fig.update_layout(title=f'ROC Curve - {selected_model_eval}',
                     xaxis_title='False Positive Rate',
                     yaxis_title='True Positive Rate')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Insights
    st.markdown("### 💡 Key Insights")
    
    best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
    best_accuracy = metrics_df['Accuracy'].max()
    
    st.markdown(f"""
    <div class="success-box">
    <strong>Best Performing Model:</strong> {best_model} with {best_accuracy:.1%} accuracy
    <br><br>
    <strong>Dataset:</strong> {dataset_name}
    <br>
    <strong>Total Samples:</strong> {len(df)} ({len(X_train)} training, {len(X_test)} testing)
    <br>
    <strong>Class Distribution:</strong> Pass: {(y == 1).sum()} | Fail: {(y == 0).sum()}
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem 0;">
    <p>🎓 Student Performance Prediction System | Built with Streamlit & Scikit-learn</p>
    <p>For educational purposes - Machine Learning Demo Application</p>
</div>
""", unsafe_allow_html=True)