import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
        color: #000000;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        color: #000000;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        color: #000000;
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
    """Train multiple models including dimensionality reduction techniques"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # ============= LINEAR REGRESSION =============
    # Treating as regression problem first, then threshold for classification
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, y_train)
    y_pred_continuous = lin_reg.predict(X_test_scaled)
    
    # Convert continuous predictions to binary (0.5 threshold)
    y_pred_linreg = (y_pred_continuous >= 0.5).astype(int)
    y_pred_linreg = np.clip(y_pred_linreg, 0, 1)  # Ensure binary
    
    # Calculate probabilities (normalized predictions)
    y_pred_proba_linreg = np.clip(y_pred_continuous, 0, 1)
    
    results['Linear Regression'] = {
        'model': lin_reg,
        'accuracy': accuracy_score(y_test, y_pred_linreg),
        'precision': precision_score(y_test, y_pred_linreg, zero_division=0),
        'recall': recall_score(y_test, y_pred_linreg, zero_division=0),
        'f1': f1_score(y_test, y_pred_linreg, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_linreg),
        'y_test': y_test,
        'y_pred': y_pred_linreg,
        'y_pred_proba': y_pred_proba_linreg,
        'mse': mean_squared_error(y_test, y_pred_continuous),
        'mae': mean_absolute_error(y_test, y_pred_continuous),
        'r2': r2_score(y_test, y_pred_continuous),
        'y_continuous': y_pred_continuous
    }
    
    # ============= PCA + LINEAR REGRESSION =============
    # Apply PCA for dimensionality reduction
    n_components = min(5, X_train_scaled.shape[1])  # Use 5 components or less
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train Linear Regression on PCA-transformed data
    lin_reg_pca = LinearRegression()
    lin_reg_pca.fit(X_train_pca, y_train)
    y_pred_continuous_pca = lin_reg_pca.predict(X_test_pca)
    
    # Convert to binary
    y_pred_pca = (y_pred_continuous_pca >= 0.5).astype(int)
    y_pred_pca = np.clip(y_pred_pca, 0, 1)
    y_pred_proba_pca = np.clip(y_pred_continuous_pca, 0, 1)
    
    results['PCA + Linear Regression'] = {
        'model': lin_reg_pca,
        'pca': pca,
        'accuracy': accuracy_score(y_test, y_pred_pca),
        'precision': precision_score(y_test, y_pred_pca, zero_division=0),
        'recall': recall_score(y_test, y_pred_pca, zero_division=0),
        'f1': f1_score(y_test, y_pred_pca, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_pca),
        'y_test': y_test,
        'y_pred': y_pred_pca,
        'y_pred_proba': y_pred_proba_pca,
        'mse': mean_squared_error(y_test, y_pred_continuous_pca),
        'mae': mean_absolute_error(y_test, y_pred_continuous_pca),
        'r2': r2_score(y_test, y_pred_continuous_pca),
        'y_continuous': y_pred_continuous_pca,
        'explained_variance': pca.explained_variance_ratio_,
        'n_components': n_components
    }
    
    # ============= SVD + LINEAR REGRESSION =============
    # Apply Truncated SVD for dimensionality reduction
    n_components_svd = min(5, X_train_scaled.shape[1])
    svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
    X_train_svd = svd.fit_transform(X_train_scaled)
    X_test_svd = svd.transform(X_test_scaled)
    
    # Train Linear Regression on SVD-transformed data
    lin_reg_svd = LinearRegression()
    lin_reg_svd.fit(X_train_svd, y_train)
    y_pred_continuous_svd = lin_reg_svd.predict(X_test_svd)
    
    # Convert to binary
    y_pred_svd = (y_pred_continuous_svd >= 0.5).astype(int)
    y_pred_svd = np.clip(y_pred_svd, 0, 1)
    y_pred_proba_svd = np.clip(y_pred_continuous_svd, 0, 1)
    
    results['SVD + Linear Regression'] = {
        'model': lin_reg_svd,
        'svd': svd,
        'accuracy': accuracy_score(y_test, y_pred_svd),
        'precision': precision_score(y_test, y_pred_svd, zero_division=0),
        'recall': recall_score(y_test, y_pred_svd, zero_division=0),
        'f1': f1_score(y_test, y_pred_svd, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred_svd),
        'y_test': y_test,
        'y_pred': y_pred_svd,
        'y_pred_proba': y_pred_proba_svd,
        'mse': mean_squared_error(y_test, y_pred_continuous_svd),
        'mae': mean_absolute_error(y_test, y_pred_continuous_svd),
        'r2': r2_score(y_test, y_pred_continuous_svd),
        'y_continuous': y_pred_continuous_svd,
        'explained_variance': svd.explained_variance_ratio_,
        'n_components': n_components_svd
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
        - Linear Regression
        - PCA + Linear Regression
        - SVD + Linear Regression
        - Dimensionality reduction analysis
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
    
    fig, ax = plt.subplots(figsize=(8, 6))
    result_counts = df['result'].value_counts()
    colors = ['#28a745' if x == 'Pass' else '#dc3545' for x in result_counts.index]
    
    wedges, texts, autotexts = ax.pie(result_counts.values, labels=result_counts.index, 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_weight('bold')
    
    ax.set_title('Pass vs Fail Distribution', fontsize=14, fontweight='bold')
    st.pyplot(fig)
    plt.close()

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
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(missing_df['Column'], missing_df['Missing %'], color='#e74c3c', alpha=0.7)
            ax.set_xlabel('Column', fontsize=12, fontweight='bold')
            ax.set_ylabel('Missing %', fontsize=12, fontweight='bold')
            ax.set_title('Missing Values by Column (%)', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
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
    
    num_cols_to_plot = len(numeric_cols)
    n_rows = (num_cols_to_plot + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if num_cols_to_plot > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Separate data by result
        pass_data = df[df['result'] == 'Pass'][col].dropna()
        fail_data = df[df['result'] == 'Fail'][col].dropna()
        
        ax.hist(pass_data, bins=20, alpha=0.6, label='Pass', color='#28a745')
        ax.hist(fail_data, bins=20, alpha=0.6, label='Fail', color='#dc3545')
        
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Hide empty subplots
    for idx in range(num_cols_to_plot, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Box plots
    st.markdown("### 📦 Feature Comparison (Pass vs Fail)")
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if num_cols_to_plot > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        data_to_plot = [df[df['result'] == 'Fail'][col].dropna(),
                       df[df['result'] == 'Pass'][col].dropna()]
        
        bp = ax.boxplot(data_to_plot, labels=['Fail', 'Pass'], patch_artist=True)
        
        # Color boxes
        bp['boxes'][0].set_facecolor('#dc3545')
        bp['boxes'][1].set_facecolor('#28a745')
        
        for box in bp['boxes']:
            box.set_alpha(0.6)
        
        ax.set_ylabel(col, fontsize=10)
        ax.set_title(f'{col} by Result', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    # Hide empty subplots
    for idx in range(num_cols_to_plot, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

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
    
    if 'PCA' in model_choice:
        # PCA Analysis
        pca = results[model_choice]['pca']
        explained_var = results[model_choice]['explained_variance']
        n_components = results[model_choice]['n_components']
        
        st.markdown(f"**PCA Components**: {n_components}")
        st.markdown(f"**Total Variance Explained**: {explained_var.sum()*100:.2f}%")
        
        # Explained variance chart
        var_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(n_components)],
            'Explained Variance (%)': explained_var * 100,
            'Cumulative Variance (%)': np.cumsum(explained_var) * 100
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(var_df))
        width = 0.6
        
        bars = ax.bar(x, var_df['Explained Variance (%)'], width, label='Individual', 
                     color='lightblue', alpha=0.8)
        
        ax2 = ax.twinx()
        line = ax2.plot(x, var_df['Cumulative Variance (%)'], 'ro-', linewidth=2, 
                       markersize=8, label='Cumulative')
        
        ax.set_xlabel('Principal Components', fontsize=12, fontweight='bold')
        ax.set_ylabel('Individual Variance Explained (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold', color='red')
        ax.set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(var_df['Component'])
        ax.grid(alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Component loadings
        st.markdown("**PCA Component Loadings (Top Features)**")
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=X.columns
        )
        
        # Show top 3 features for each component
        for i in range(n_components):
            col = f'PC{i+1}'
            top_features = loadings[col].abs().nlargest(3)
            st.write(f"**{col}**: {', '.join(top_features.index.tolist())}")
        
        st.dataframe(loadings.style.background_gradient(cmap='coolwarm', axis=None), 
                    use_container_width=True)
        
    elif 'SVD' in model_choice:
        # SVD Analysis
        svd = results[model_choice]['svd']
        explained_var = results[model_choice]['explained_variance']
        n_components = results[model_choice]['n_components']
        
        st.markdown(f"**SVD Components**: {n_components}")
        st.markdown(f"**Total Variance Explained**: {explained_var.sum()*100:.2f}%")
        
        # Explained variance chart
        var_df = pd.DataFrame({
            'Component': [f'SV{i+1}' for i in range(n_components)],
            'Explained Variance (%)': explained_var * 100,
            'Cumulative Variance (%)': np.cumsum(explained_var) * 100
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(var_df))
        width = 0.6
        
        bars = ax.bar(x, var_df['Explained Variance (%)'], width, label='Individual', 
                     color='lightgreen', alpha=0.8)
        
        ax2 = ax.twinx()
        line = ax2.plot(x, var_df['Cumulative Variance (%)'], 'o-', color='orange', 
                       linewidth=2, markersize=8, label='Cumulative')
        
        ax.set_xlabel('Singular Vectors', fontsize=12, fontweight='bold')
        ax.set_ylabel('Individual Variance Explained (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold', color='orange')
        ax.set_title('SVD Explained Variance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(var_df['Component'])
        ax.grid(alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Component loadings
        st.markdown("**SVD Component Loadings (Top Features)**")
        loadings = pd.DataFrame(
            svd.components_.T,
            columns=[f'SV{i+1}' for i in range(n_components)],
            index=X.columns
        )
        
        # Show top 3 features for each component
        for i in range(n_components):
            col = f'SV{i+1}'
            top_features = loadings[col].abs().nlargest(3)
            st.write(f"**{col}**: {', '.join(top_features.index.tolist())}")
        
        st.dataframe(loadings.style.background_gradient(cmap='coolwarm', axis=None),
                    use_container_width=True)
        
    else:
        # Linear Regression coefficients
        st.markdown("**Linear Regression Coefficients**")
        
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': selected_model.coef_,
            'Abs_Coefficient': np.abs(selected_model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#d62728' if c < 0 else '#2ca02c' for c in coef_df['Coefficient']]
        bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
        
        ax.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title('Feature Coefficients (Linear Regression)', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.dataframe(coef_df[['Feature', 'Coefficient']], use_container_width=True)
        
        st.info("""
        **Interpreting Coefficients:**
        - **Positive coefficients** (green): Features that increase pass probability
        - **Negative coefficients** (red): Features that decrease pass probability
        - **Magnitude**: Larger absolute values = stronger influence
        """)
    
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
        
        # Apply dimensionality reduction if needed
        if 'PCA' in model_choice:
            pca = results[model_choice]['pca']
            input_scaled = pca.transform(input_scaled)
        elif 'SVD' in model_choice:
            svd = results[model_choice]['svd']
            input_scaled = svd.transform(input_scaled)
        
        # Predict
        prediction_continuous = selected_model.predict(input_scaled)[0]
        prediction = 1 if prediction_continuous >= 0.5 else 0
        
        # Calculate confidence
        confidence = prediction_continuous if prediction == 1 else (1 - prediction_continuous)
        confidence = np.clip(confidence * 100, 0, 100)
        
        # Display result
        result_text = "Pass" if prediction == 1 else "Fail"
        result_color = "#28a745" if prediction == 1 else "#dc3545"
        
        st.markdown("---")
        st.markdown("### 🎉 Prediction Result")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="background-color: {result_color}; padding: 2rem; border-radius: 1rem; text-align: center;">
                <h1 style="color: white; margin: 0;">Result: {result_text}</h1>
                <h3 style="color: white; margin-top: 1rem;">Confidence: {confidence:.1f}%</h3>
                <p style="color: white; margin-top: 0.5rem;">Raw Score: {prediction_continuous:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability breakdown
        st.markdown("### 📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Outcome': ['Fail', 'Pass'],
            'Probability': [(1 - prediction_continuous) * 100, prediction_continuous * 100]
        })
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#dc3545', '#28a745']
        bars = ax.bar(prob_df['Outcome'], prob_df['Probability'], color=colors, alpha=0.8)
        
        ax.set_xlabel('Outcome', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Show model type
        st.info(f"**Model Used**: {model_choice} | **Threshold**: 0.5")

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
        'F1-Score': [results[m]['f1'] for m in results.keys()],
        'MSE': [results[m]['mse'] for m in results.keys()],
        'MAE': [results[m]['mae'] for m in results.keys()],
        'R² Score': [results[m]['r2'] for m in results.keys()]
    })
    
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'R² Score'])
                                    .highlight_min(axis=0, subset=['MSE', 'MAE']), 
                 use_container_width=True)
    
    st.markdown("---")
    
    # Visualize classification metrics
    st.markdown("#### Classification Metrics")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics_df['Model']))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, metrics_df[metric], width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: Classification Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_df['Model'])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Visualize regression metrics
    st.markdown("#### Regression Metrics")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # MSE and MAE
    x = np.arange(len(metrics_df['Model']))
    width = 0.35
    
    ax1.bar(x - width/2, metrics_df['MSE'], width, label='MSE', color='lightcoral', alpha=0.8)
    ax1.bar(x + width/2, metrics_df['MAE'], width, label='MAE', color='lightskyblue', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Error Value', fontsize=12, fontweight='bold')
    ax1.set_title('Error Metrics (Lower is Better)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_df['Model'], rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # R² Score
    bars = ax2.bar(metrics_df['Model'], metrics_df['R² Score'], color='steelblue', alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('R² Score (Higher is Better)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Add value labels on R² bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
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
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fail', 'Pass'], 
                yticklabels=['Fail', 'Pass'],
                cbar_kws={'label': 'Count'},
                ax=ax, linewidths=1, linecolor='gray')
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {selected_model_eval}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # ROC Curve
    st.markdown("### 📈 ROC Curve")
    
    fpr, tpr, _ = roc_curve(model_results['y_test'], model_results['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC Curve - {selected_model_eval}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Model Insights
    st.markdown("### 💡 Key Insights")
    
    best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
    best_accuracy = metrics_df['Accuracy'].max()
    best_r2 = metrics_df['R² Score'].max()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="success-box">
        <strong>Best Performing Model (Classification):</strong> {best_model}
        <br>
        <strong>Accuracy:</strong> {best_accuracy:.1%}
        <br><br>
        <strong>Dataset:</strong> {dataset_name}
        <br>
        <strong>Total Samples:</strong> {len(df)} ({len(X_train)} training, {len(X_test)} testing)
        <br>
        <strong>Class Distribution:</strong> Pass: {(y == 1).sum()} | Fail: {(y == 0).sum()}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="warning-box">
        <strong>Model Comparison Insights:</strong>
        <br>
        • <strong>Linear Regression:</strong> Direct relationship modeling
        <br>
        • <strong>PCA + Linear Regression:</strong> Variance explained: {results['PCA + Linear Regression']['explained_variance'].sum()*100:.1f}%
        <br>
        • <strong>SVD + Linear Regression:</strong> Variance explained: {results['SVD + Linear Regression']['explained_variance'].sum()*100:.1f}%
        <br><br>
        <strong>Best R² Score:</strong> {best_r2:.3f}
        </div>
        """, unsafe_allow_html=True)
    
    # Dimensionality Reduction Comparison
    st.markdown("---")
    st.markdown("### 🔍 Dimensionality Reduction Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PCA Analysis**")
        pca_var = results['PCA + Linear Regression']['explained_variance']
        pca_n = results['PCA + Linear Regression']['n_components']
        st.metric("Components Used", pca_n)
        st.metric("Variance Explained", f"{pca_var.sum()*100:.2f}%")
        
        pca_data = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(pca_n)],
            'Variance': pca_var * 100
        })
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(pca_data['Component'], pca_data['Variance'], color='steelblue', alpha=0.8)
        ax.set_xlabel('Component', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
        ax.set_title('PCA Variance per Component', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("**SVD Analysis**")
        svd_var = results['SVD + Linear Regression']['explained_variance']
        svd_n = results['SVD + Linear Regression']['n_components']
        st.metric("Components Used", svd_n)
        st.metric("Variance Explained", f"{svd_var.sum()*100:.2f}%")
        
        svd_data = pd.DataFrame({
            'Component': [f'SV{i+1}' for i in range(svd_n)],
            'Variance': svd_var * 100
        })
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(svd_data['Component'], svd_data['Variance'], color='mediumseagreen', alpha=0.8)
        ax.set_xlabel('Component', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
        ax.set_title('SVD Variance per Component', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem 0;">
    <p>🎓 Student Performance Prediction System | Built with Streamlit & Scikit-learn</p>
    <p>For educational purposes - Machine Learning Demo Application</p>
</div>
""", unsafe_allow_html=True)