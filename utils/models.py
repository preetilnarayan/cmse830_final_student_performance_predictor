import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)

# Model configurations
MODEL_CONFIGS = {
    'Logistic Regression': {
        'class': LogisticRegression,
        'params': {'random_state': 42, 'max_iter': 1000},
        'has_importance': False
    },
    'Random Forest': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 10},
        'has_importance': True
    },
    'Gradient Boosting': {
        'class': GradientBoostingClassifier,
        'params': {'n_estimators': 100, 'random_state': 42, 'max_depth': 5},
        'has_importance': True
    }
}


def calculate_metrics(y_test, y_pred, y_pred_proba):
    """Calculate all metrics for a model"""
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'classification_report': classification_report(y_test, y_pred)
    }


def train_single_model(model_class, params, X_train, X_test, y_train, y_test, has_importance=False):
    """Train a single model and return results"""
    model = model_class(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    results = calculate_metrics(y_test, y_pred, y_pred_proba)
    results['model'] = model
    
    if has_importance:
        results['feature_importance'] = model.feature_importances_
    
    return results


def train_dimensionality_reduction_model(X_train, X_test, y_train, y_test, method='pca'):
    """Train model with dimensionality reduction (PCA or SVD)"""
    n_components = min(5, X_train.shape[1])
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:  # svd
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
    
    X_train_reduced = reducer.fit_transform(X_train)
    X_test_reduced = reducer.transform(X_test)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_reduced, y_train)
    
    y_pred = model.predict(X_test_reduced)
    y_pred_proba = model.predict_proba(X_test_reduced)[:, 1]
    
    results = calculate_metrics(y_test, y_pred, y_pred_proba)
    results['model'] = model
    results[method] = reducer
    results['explained_variance'] = reducer.explained_variance_ratio_
    results['n_components'] = n_components
    
    return results


@st.cache_data
def train_all_models(X, y):
    """Train all models and return results"""
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Train standard models
    for model_name, config in MODEL_CONFIGS.items():
        results[model_name] = train_single_model(
            config['class'], 
            config['params'], 
            X_train_scaled, 
            X_test_scaled, 
            y_train, 
            y_test,
            config['has_importance']
        )
    
    # Train PCA + Logistic Regression
    results['PCA + Logistic Regression'] = train_dimensionality_reduction_model(
        X_train_scaled, X_test_scaled, y_train, y_test, method='pca'
    )
    
    # Train SVD + Logistic Regression
    results['SVD + Logistic Regression'] = train_dimensionality_reduction_model(
        X_train_scaled, X_test_scaled, y_train, y_test, method='svd'
    )
    
    return results, scaler, X_train, X_test, y_train, y_test