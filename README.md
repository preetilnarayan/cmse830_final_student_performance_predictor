# Student Performance Prediction System

An interactive application that predicts student pass/fail outcomes using real-world datasets from the UCI Student Performance dataset.  
The system implements Logistic Regression, Random Forest, Gradient Boosting, PCA + Logistic Regression, and SVD + Logistic Regression, with extensive data exploration, preprocessing transparency, and model evaluation.  

---

## Project Structure

```

├── app.py
├── page_modules/
│   ├── about_datasets.py        # Dataset descriptions and basic stats
│   ├── cleaning.py              # Data cleaning and preprocessing overview
│   ├── documentation.py         # In-app project documentation
│   ├── eda.py                   # Exploratory data analysis (EDA)
│   ├── home.py                  # Landing page and guided workflow
│   ├── missingness.py           # Missing data patterns and handling
│   ├── prediction_models.py     # Model training results and metrics
│   └── student_predictor.py     # Individual student prediction interface
├── README.md
├── requirements.txt
└── utils/
├── data_loader.py           # Data loading utilities
├── models.py                # Model configuration and training logic
├── preprocessing.py         # Imputation, encoding, scaling
└── styles.py                # Custom CSS and visual styling

```

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Datasets](#datasets)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Machine Learning Models](#machine-learning-models)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Technical Implementation](#technical-implementation)  
- [Educational Value](#educational-value)  
- [Future Enhancements](#future-enhancements)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  
- [References](#references)  

---

## Overview

This application addresses a critical challenge in education: **identifying at-risk students early** so that institutions can provide timely interventions and improve overall student success rates.  
Using machine learning techniques, the system analyzes various performance, behavioral, and contextual metrics to forecast pass/fail outcomes.

### Problem Statement

Educational institutions need proactive tools to detect struggling students before failure occurs.  
Traditional reactive approaches often miss early warning signs, leading to higher failure rates and lower student retention.

### Solution

A machine learning–based prediction system that:

- Analyzes multiple student performance indicators.  
- Provides real-time pass/fail predictions with associated scores.  
- Offers interpretable outputs for educators, counselors, and researchers.  
- Visualizes key patterns in student data, preprocessing steps, and model behavior.  

### Target Users

- **School administrators** – for resource allocation and planning.  
- **Teachers & educators** – to identify struggling students early.  
- **Academic counselors** – to design targeted intervention plans.  
- **Education researchers** – to analyze drivers of student success at scale.  

---

## Features

### Core Functionality

- End-to-end workflow across **About the Datasets**, **EDA**, **Missingness**, **Cleaning**, **Prediction Models**, **Dataset & Model Prediction**, and **Student Predictor** pages.  
- Multiple model families: Logistic Regression, Random Forest, Gradient Boosting, PCA + Logistic Regression, and SVD + Logistic Regression.  
- Dedicated pages for cohort-level analysis (**Dataset & Model Prediction**) and individual “what-if” analysis (**Student Predictor**).  
- Consistent train/test split, stratification by target, and feature scaling to evaluate all models fairly.  
- Rich outputs including metrics, confusion matrices, feature importance, and dimensionality reduction summaries.  

### Analysis Capabilities

- **Dataset-level exploration** via **About the Datasets** and **EDA**: distributions, class balance, and correlations between academic and behavioral features.  
- **Missing data diagnostics** on the **Missingness** page: where values are missing and how this may affect models.  
- **Preprocessing inspection** on **Cleaning**: imputation, encoding, and scaling strategies used prior to training.  
- **Model performance comparison**: accuracy, precision, recall, F1-score, confusion matrices, and classification reports across all classifiers.  
- **Model behavior insights**: feature importance for tree-based models and explained variance / components for PCA and SVD pipelines.  

### User Experience

- Clean, structured interface with sidebar navigation across **Home**, **Dataset & Model Prediction**, **About the Datasets**, **EDA**, **Cleaning**, **Missingness**, **Prediction Models**, **Student Predictor**, and **Documentation**.  
- Page-specific guidance so users know how to move from data understanding, through preprocessing and modeling, to individual predictions.  
- Consistent visual styling for tables, plots, and metrics to clearly separate analysis views and prediction outputs.  
- Workflow designed for both educators and data practitioners, supporting risk identification, analysis, and reporting.  

---

## Datasets

The application uses three derived datasets based on the UCI **Student Performance** data (Math and Portuguese tracks).

### Dataset 1: UCI Mathematics Performance

**Source**: UCI Student Performance Dataset (Math)  
**Samples**: 395 students  

Example features:

- `age`: Student age (15–22 years).  
- `study_time_weekly`: Weekly study time (1–4 scale).  
- `past_failures`: Number of past class failures (0–4).  
- `absences`: Number of school absences.  
- `period1_grade`, `period2_grade`, `final_grade`: Grades 0–20.  
- `result`: Binary pass/fail derived from `final_grade`.  

Includes moderate missing values in selected numeric features.

### Dataset 2: UCI Portuguese Performance

**Source**: UCI Student Performance Dataset (Portuguese)  
**Samples**: 649 students.  

Example features:

- `age`: Student age (15–22 years).  
- `mother_education`, `father_education`: Parental education levels.  
- `travel_time`: Home-to-school travel time (1–4 scale).  
- `family_relationship`: Family relationship quality.  
- `social_outings`, `health_status`: Social and health indicators.  
- `absences`, `final_grade`, and binary `result`.  

Contains missing values in several socio-demographic attributes.

### Dataset 3: Comprehensive Academic Profile

**Source**: Combined data constructed from Math and Portuguese tracks.  
**Samples**: ~500 students.  

Example features:

- `study_time_weekly`, `past_failures`, `absences`.  
- Academic support indicators (school and family support, paid classes, extracurriculars).  
- Higher education aspiration and home internet access.  
- `final_grade` and derived pass/fail `result`.  

Includes missing values in support- and behavior-related features.

---

## Installation

### Prerequisites

- Python 3.8 or higher.  
- `pip` package manager.  

### Step 1: Clone the Repository

```

git clone [https://github.com/yourusername/student-performance-predictor.git](https://github.com/yourusername/student-performance-predictor.git)
cd student-performance-predictor

```

### Step 2: Create Virtual Environment (Recommended)

Create and activate a virtual environment using `venv`, `conda`, or another tool of your choice.

### Step 3: Install Dependencies

```

pip install -r requirements.txt

```

### Step 4: Run the Application

```

streamlit run app.py

```

By default, Streamlit runs at `http://localhost:8501`.  

---

## Requirements

`requirements.txt` includes at least:

```

streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

```

---

## Usage

### Quick Start Guide

1. **Launch the Application**

```

streamlit run app.py

```

2. **Navigate Through Pages**

- **Home** – Overview, goals, and recommended workflow.  
- **About the Datasets** – Dataset descriptions, feature summaries, and basic stats.  
- **EDA** – Distributions, correlations, and class balance plots.  
- **Missingness** – Missing value patterns and summaries.  
- **Cleaning** – Description of preprocessing, imputation, and scaling steps.  
- **Prediction Models** – Trained models, metrics, feature importance, and dimensionality reduction.  
- **Dataset & Model Prediction** – Select dataset and model, explore predictions at a cohort level.  
- **Student Predictor** – Make individual student predictions with interactive inputs.  
- **Documentation** – In-app documentation, methodology, and references.  

3. **Make Predictions**

- Open **Dataset & Model Prediction** or **Student Predictor**.  
- Select a dataset and model (Logistic Regression, Random Forest, Gradient Boosting, PCA + Logistic Regression, SVD + Logistic Regression).  
- Adjust sliders/dropdowns for student features.  
- Click **Predict Result** to obtain pass/fail prediction and scores.  

4. **Analyze Results**

- Review accuracy, precision, recall, and F1-score for each model.  
- Inspect confusion matrices and classification reports.  
- Examine feature importance for tree-based models and explained variance for PCA/SVD pipelines.  

### Example Use Cases

#### For Educators

Goal: Identify at-risk students and intervene early.

1. Visit **About the Datasets** and **EDA** to understand current cohorts.  
2. Use **Prediction Models** to see which models perform best.  
3. Go to **Student Predictor**, enter a student’s current metrics, and review the pass/fail prediction and scores.  
4. Use these insights to plan targeted support or adjustments in teaching.

#### For Data Scientists

Goal: Understand model performance and behavior.

1. Select a dataset in **Dataset & Model Prediction**.  
2. Explore **EDA**, **Missingness**, and **Cleaning** to understand data quality and preprocessing.  
3. Use **Prediction Models** to compare Logistic Regression, Random Forest, Gradient Boosting, PCA + LR, and SVD + LR.  
4. Analyze metrics, confusion matrices, feature importance, and dimensionality reduction outputs.

---

## Machine Learning Models

### Logistic Regression

Models the probability of pass/fail as a function of student features using a linear decision boundary in feature space.  
Serves as a fast, interpretable baseline; coefficients indicate how each feature affects the odds of passing.

### Random Forest

An ensemble of decision trees that aggregates many randomized trees to produce robust predictions.  
Captures non-linear relationships and interactions between features; exposes feature importance scores.

### Gradient Boosting

Sequentially builds shallow trees where each tree corrects errors made by previous ones.  
Often achieves strong performance on structured data and can model subtle patterns in student outcomes.

### PCA + Logistic Regression

Applies Principal Component Analysis (PCA) to reduce the feature space to a small number of orthogonal components.  
Trains Logistic Regression on these components, reducing multicollinearity and noise while keeping the model relatively interpretable.

### SVD + Logistic Regression

Uses Truncated Singular Value Decomposition (SVD) to project data into a low-dimensional latent space.  
Logistic Regression is trained on latent features, which can be effective for high-dimensional or sparse inputs and provides explained variance information.

---

## Evaluation Metrics

The app focuses on classification metrics for pass/fail prediction.

### Classification Metrics

- **Accuracy** – Proportion of correctly classified students.  
- **Precision** – Among students predicted to pass, the fraction who actually pass.  
- **Recall** – Among students who actually pass, the fraction correctly identified.  
- **F1-Score** – Harmonic mean of precision and recall, balancing both.  

The app also computes:

- **Confusion Matrix** – Counts of true positives, true negatives, false positives, and false negatives.  
- **Classification Report** – Class-wise precision, recall, F1-score, and support for pass and fail classes.  

### Model-Specific Insights

- **Random Forest** and **Gradient Boosting**: feature importance scores showing which variables drive predictions.  
- **PCA + Logistic Regression** and **SVD + Logistic Regression**: number of components and explained variance ratios, indicating how much information is preserved.  

---

## Technical Implementation

### Data Processing Pipeline

1. **Data Loading** – Load datasets from local files or UCI-derived sources.  
2. **Missing Value Handling** – Detect missing values and apply median/mode or similar strategies for imputation.  
3. **Feature Engineering** – Encode categorical variables, scale numeric features (e.g., StandardScaler), and derive a binary pass/fail target.  
4. **Train–Test Split** – Split into training and test sets with stratification to preserve class balance.  

### Model Training Pipeline

1. **Scaling** – Standardize features before model training.  
2. **Base Models** – Train Logistic Regression, Random Forest, and Gradient Boosting on scaled data with shared splits.  
3. **Dimensionality Reduction Models** – Apply PCA and SVD to obtain a reduced feature representation and train Logistic Regression on each.  
4. **Evaluation** – Compute metrics, confusion matrices, classification reports, and any model-specific diagnostics.  

### Key Technologies

- **Frontend**: Streamlit.  
- **Data Processing**: pandas, NumPy.  
- **Machine Learning**: scikit-learn (Logistic Regression, RandomForest, GradientBoosting, PCA, TruncatedSVD, metrics).  
- **Visualization**: Matplotlib, Seaborn.  
- **Data Source**: UCI Machine Learning Repository (Student Performance dataset).  

---

## Educational Value

### Technical Skills

- Data preprocessing and cleaning on real educational datasets.  
- Handling missing values, encoding, and scaling.  
- Applying ensemble methods and dimensionality reduction in classification tasks.  
- Building and deploying a multipage Streamlit data app.  

### Data Science Best Practices

- Exploratory data analysis (EDA) and visualization for insight.  
- Model interpretability via feature importance and component analysis.  
- Use of multiple evaluation metrics and clear documentation of the pipeline.  

---

## Future Enhancements

- Add additional models (e.g., XGBoost, calibrated classifiers).  
- Implement cross-validation and hyperparameter optimization.  
- Add feature selection and fairness/robustness analysis.  
- Provide an API endpoint for batch predictions.  
- Support custom dataset uploads.  
- Export detailed PDF/HTML reports.  
- Add temporal tracking of cohorts over multiple terms.  

---

## License

This project is released under an MIT-style open-source license.  
Users may use, modify, and redistribute the software provided that the required copyright
notice and license terms are included in derivative works.

---

## Acknowledgments

- **UCI Machine Learning Repository** for providing the Student Performance dataset.  
- **P. Cortez and A. Silva** – Original dataset creators (2008).  

---

## References

### Dataset Citation

Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.*  
In A. Brito & J. Teixeira (Eds.), **Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008)**, Porto, Portugal, EUROSIS.  

### Dataset URL

- UCI Student Performance Dataset: https://archive.ics.uci.edu/dataset/320/student+performance