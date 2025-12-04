# Student Performance Prediction System

An application that predicts student pass/fail outcomes using real-world datasets from the UCI Machine Learning Repository. This project implements Linear Regression, PCA, and SVD for student performance analysis with extensive data visualization and model evaluation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
- [Technical Implementation](#technical-implementation)
- [License](#license)

## Overview

This application addresses a critical challenge in education: **identifying at-risk students early** to provide timely interventions and improve overall student success rates. Using machine learning techniques, the system analyzes various student performance metrics to forecast pass/fail outcomes with high accuracy.

### Problem Statement
Educational institutions need proactive tools to identify struggling students before it's too late. Traditional reactive approaches often miss early warning signs, leading to higher failure rates and lower student retention.

### Solution
A machine learning-based prediction system that:
- Analyzes multiple student performance indicators
- Provides real-time predictions with confidence scores
- Offers interpretable results for educators
- Visualizes key patterns in student data

### Target Users
- School administrators
- Academic counselors
- Educators and teachers
- Educational data analysts

## Features

### Core Functionality
- **3 Real-World Datasets**: UCI Machine Learning Repository datasets with 395-649 samples each
- **Missing Value Handling**: Comprehensive data cleaning with median/mode imputation
- **Multiple ML Models**: Linear Regression, PCA + Linear Regression, SVD + Linear Regression
- **Interactive Predictions**: Real-time student outcome predictions with confidence scores
- **Advanced Visualizations**: 15+ matplotlib/seaborn charts for data analysis

### Analysis Capabilities
- Feature correlation analysis
- Distribution analysis by outcome (Pass/Fail)
- Dimensionality reduction visualization
- Feature importance/coefficient analysis
- Model performance comparison
- ROC curves and confusion matrices

### User Experience
- Clean, intuitive interface with sidebar navigation
- 5 main sections: Home, Product Overview, Data Science Analysis, Model Prediction, Evaluation Metrics
- Professional color scheme (green for pass, red for fail)
- Responsive design with clear instructions
- Download dataset functionality

## Datasets

The application uses three real-world datasets from the UCI Machine Learning Repository:

### Dataset 1: UCI Mathematics Performance
**Source**: UCI Student Performance Dataset (Math)  
**Samples**: 395 students  
**Features**:
- `age`: Student age (15-22 years)
- `study_time_weekly`: Weekly study time (1-4 scale)
- `past_failures`: Number of past class failures (0-4)
- `absences`: Number of school absences (0-93)
- `period1_grade`: First period grade (0-20)
- `period2_grade`: Second period grade (0-20)
- `final_grade`: Final grade (0-20)
- `result`: Pass (≥10) or Fail (<10)

**Missing Values**: 10-15% in study_time, absences, period1_grade

### Dataset 2: UCI Portuguese Performance
**Source**: UCI Student Performance Dataset (Portuguese)  
**Samples**: 649 students  
**Features**:
- `age`: Student age (15-22 years)
- `mother_education`: Mother's education level (0-4 scale)
- `father_education`: Father's education level (0-4 scale)
- `travel_time`: Home to school travel time (1-4 scale)
- `family_relationship`: Family relationship quality (1-5 scale)
- `social_outings`: Going out with friends (1-5 scale)
- `health_status`: Current health status (1-5 scale)
- `absences`: Number of school absences (0-75)
- `final_grade`: Final grade (0-20)
- `result`: Pass (≥10) or Fail (<10)

**Missing Values**: 8-12% in mother_education, health_status, social_outings, travel_time

### Dataset 3: Comprehensive Academic Profile
**Source**: UCI ML Repository (Combined Math + Portuguese)  
**Samples**: 500 students  
**Features**:
- `age`: Student age (15-22 years)
- `study_time_weekly`: Weekly study time (1-4 scale)
- `past_failures`: Number of past class failures (0-4)
- `school_support`: Extra educational support (0/1)
- `family_support`: Family educational support (0/1)
- `paid_classes`: Extra paid classes (0/1)
- `extracurricular`: Extra-curricular activities (0/1)
- `higher_ed_aspiration`: Wants higher education (0/1)
- `internet_access`: Internet access at home (0/1)
- `final_grade`: Final grade (0-20)
- `result`: Pass (≥10) or Fail (<10)

**Missing Values**: 15-20% in paid_classes, extracurricular, study_time_weekly; 10% in school_support, internet_access

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/student-performance-predictor.git
cd student-performance-predictor
```

### Step 2: Create Virtual Environment (Recommended)

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Usage

### Quick Start Guide

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

2. **Select a Dataset**
   - Use the sidebar dropdown to choose from 3 datasets
   - Each dataset loads automatically from UCI repository

3. **Navigate Through Sections**
   - **Home**: Overview and quick start
   - **Product Overview**: Explore dataset statistics and distributions
   - **Data Science Analysis**: View correlations, missing values, and feature analysis
   - **Model Prediction**: Make individual student predictions
   - **Evaluation Metrics**: Compare model performance

4. **Make Predictions**
   - Go to "Model Prediction" section
   - Select a model (Linear Regression, PCA, or SVD)
   - Adjust sliders/dropdowns for student features
   - Click "Predict Result" to get pass/fail prediction

5. **Analyze Results**
   - View confidence scores and probabilities
   - Examine feature importance/coefficients
   - Compare model performance metrics

### Example Use Cases

#### For Educators
```
Goal: Identify at-risk students
1. Navigate to "Product Overview"
2. Review class distribution and statistics
3. Go to "Model Prediction"
4. Input student's current metrics
5. Review prediction and confidence score
6. Take proactive intervention if needed
```

#### For Data Scientists
```
Goal: Understand model performance
1. Select dataset from sidebar
2. Go to "Data Science Analysis"
3. Review correlation matrix and distributions
4. Navigate to "Evaluation Metrics"
5. Compare accuracy, precision, recall, F1-score
6. Analyze ROC curves and confusion matrices
```

## Machine Learning Models

### 1. Linear Regression
**Purpose**: Direct relationship modeling between features and outcome

**Implementation**:
- Treats pass/fail as continuous problem (0/1)
- Uses 0.5 threshold for binary classification
- Provides interpretable coefficients

**Advantages**:
- Simple and interpretable
- Fast training and prediction
- Shows direct feature impact

**Metrics**:
- Classification: Accuracy, Precision, Recall, F1
- Regression: MSE, MAE, R²

### 2. PCA + Linear Regression
**Purpose**: Dimensionality reduction using Principal Component Analysis

**Implementation**:
- Reduces features to 5 principal components
- Captures maximum variance in data
- Trains Linear Regression on transformed features

**Advantages**:
- Reduces multicollinearity
- Handles high-dimensional data
- Removes noise and redundancy

**Key Outputs**:
- Explained variance ratio per component
- Component loadings (feature contributions)
- Cumulative variance explained

### 3. SVD + Linear Regression
**Purpose**: Matrix factorization for feature extraction

**Implementation**:
- Applies Truncated SVD (5 components)
- Decomposes feature matrix
- Trains Linear Regression on singular vectors

**Advantages**:
- Works with sparse data
- Optimal low-rank approximation
- Identifies latent patterns

**Key Outputs**:
- Singular values
- Explained variance per component
- Feature loadings in latent space

## Evaluation Metrics
### Classification Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Accuracy** | Overall correctness | (TP + TN) / (TP + TN + FP + FN) |
| **Precision** | Positive prediction accuracy | TP / (TP + FP) |
| **Recall** | True positive rate | TP / (TP + FN) |
| **F1-Score** | Harmonic mean of precision/recall | 2 × (Precision × Recall) / (Precision + Recall) |

### Regression Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MSE** | Mean Squared Error | Lower is better |
| **MAE** | Mean Absolute Error | Lower is better |
| **R² Score** | Coefficient of determination | Higher is better (0-1) |

### Visualizations Included

1. **Confusion Matrix**: True/False Positives and Negatives
2. **ROC Curve**: True Positive Rate vs False Positive Rate
3. **Feature Importance**: Coefficient/loading visualizations
4. **Variance Explained**: PCA/SVD component analysis
5. **Distribution Plots**: Feature distributions by outcome
6. **Correlation Heatmap**: Feature relationships

## Technical Implementation
### Data Processing Pipeline

```python
1. Data Loading
   ├── Load from UCI repository URLs
   ├── Fallback to synthetic data if unavailable
   └── Cache with @st.cache_data

2. Missing Value Handling
   ├── Detection: Identify columns with missing values
   ├── Imputation:
   │   ├── Numeric: Median (robust to outliers)
   │   └── Categorical: Mode (most frequent)
   └── Validation: Verify no missing values remain

3. Feature Engineering
   ├── Label Encoding: Categorical -> Numeric
   ├── Feature Scaling: StandardScaler
   └── Target Creation: Grade -> Pass/Fail (threshold: 10)

4. Train-Test Split
   ├── 80% Training
   ├── 20% Testing
   └── Stratified by target (maintain class balance)
```

### Model Training Pipeline

```python
1. Scale Features
   └── StandardScaler: Mean=0, Std=1

2. Train Models
   ├── Linear Regression
   │   ├── Fit on scaled training data
   │   └── Threshold predictions at 0.5
   │
   ├── PCA + Linear Regression
   │   ├── PCA: Extract 5 components
   │   ├── Fit Linear Regression on components
   │   └── Store explained variance
   │
   └── SVD + Linear Regression
       ├── TruncatedSVD: Extract 5 components
       ├── Fit Linear Regression on components
       └── Store explained variance

3. Evaluate
   ├── Classification metrics
   ├── Regression metrics
   └── Generate visualizations
```

### Key Technologies

- **Frontend**: Streamlit (interactive web interface)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Data Source**: UCI Machine Learning Repository

## Educational Value

This project demonstrates:

### Technical Skills
- Data preprocessing and cleaning  
- Handling missing values with multiple strategies  
- Feature engineering and encoding  
- Dimensionality reduction (PCA, SVD)  
- Regression-to-classification conversion  
- Model evaluation and comparison  
- Interactive web application development  

### Data Science Best Practices
- Exploratory Data Analysis (EDA)  
- Data visualization for insights  
- Model interpretability  
- Cross-validation and train-test split  
- Multiple evaluation metrics  
- Documentation and code organization  

### Future Enhancements
- Add more ML models (Random Forest, XGBoost)
- Implement cross-validation
- Add feature selection algorithms
- Create API endpoint for predictions
- Add user authentication
- Support for custom dataset uploads
- Export detailed PDF reports
- Add temporal analysis (tracking over time)

## License
This project is licensed under the MIT License based libraries & softwares
```
MIT License
Copyright (c) 2025 Student Performance Predictor
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments
- **UCI Machine Learning Repository** for providing the Student Performance datasets
- **P. Cortez and A. Silva** - Original dataset creators (2008)

## References
### Dataset Citation
```
P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. 
In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference 
(FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7.
```

### Dataset URL
- [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
