import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder


@st.cache_data
def preprocess_data(df, target_col='result'):
    """Preprocess the dataset for modeling with comprehensive missing value handling"""
    df_processed = df.copy()
    
    # Store original missing value info for display
    missing_info = df_processed.isnull().sum()
    missing_percent = (missing_info / len(df_processed) * 100).round(2)
    
    # Separate column types
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from feature lists
    numeric_cols = [col for col in numeric_cols if col != target_col]
    categorical_cols = [col for col in categorical_cols if col != target_col]
    
    # Vectorized imputation for numeric columns
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Imputation for categorical columns
    for col in categorical_cols:
        if df_processed[col].isnull().any():
            mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'Unknown'
            df_processed[col].fillna(mode_value, inplace=True)
    
    # Encode categorical variables (excluding target)
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df_processed[col] = label_encoders[col].fit_transform(df_processed[col])
    
    # Separate features and target
    X = df_processed.drop(target_col, axis=1)
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df_processed[target_col])
    
    return X, y, df_processed, missing_info, missing_percent, label_encoders