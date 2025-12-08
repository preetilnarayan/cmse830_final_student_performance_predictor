# about_datasets.py
import streamlit as st

def render():
    # OVERVIEW
    st.header("Overview")
    st.markdown(
        """
        An interactive application that predicts student pass/fail outcomes using real-world datasets from the UCI Machine Learning Repository.  
The system implements Linear Regression with PCA and SVD variants, supports rich data visualization, and reports model evaluation metrics.
This project focuses on early identification of at-risk students so that educators can intervene before failure occurs.  
The app analyzes student performance metrics and forecasts pass/fail outcomes to support data-informed decision making.
"""
    )

    st.subheader("Problem Statement")
    st.markdown(
        """
Educational institutions often detect struggling students only after they fail, which leads to higher failure rates and poor retention.  
A proactive, data-driven tool can highlight early warning signs and support timely interventions.
"""
    )

    st.subheader("Solution")
    st.markdown(
        """
The application provides a machine-learning based prediction system that:

- Uses multiple student performance indicators as model inputs.  
- Generates real-time pass/fail predictions and confidence scores.  
- Presents interpretable outputs for educators and analysts.  
- Visualizes key patterns and trends in student data.
"""
    )

    st.subheader("Target Users")
    st.markdown(
        """
- School administrators  
- Academic counselors  
- Educators and teachers  
- Educational data analysts  
"""
    )

    st.markdown("---")

    # FEATURES
    st.header("Features")

    st.subheader("Core Functionality")
    st.markdown(
        """
- End-to-end workflow spanning data understanding, EDA, missingness analysis, data cleaning, model training, and student-level prediction.​
- Multiple model families implemented: Logistic Regression, Random Forest, Gradient Boosting, PCA + Logistic Regression, and SVD + Logistic Regression for robust pass/fail prediction.​
- Dedicated pages for Dataset & Model Prediction and Student Predictor so users can experiment with cohort-level scenarios and individual “what-if” analyses.​
- Consistent use of a train/test split, feature scaling, and stratification to ensure fair evaluation of all models on the same data.​
- Rich evaluation outputs including accuracy, precision, recall, F1-score, confusion matrices, classification reports, and dimensionality reduction summaries.
"""
    )

    st.subheader("Analysis Capabilities")
    st.markdown(
        """
- **Dataset-level exploration** via **About the Datasets** and **EDA**, including distributions, class balance, and correlations between key academic features.  
- **Missing data diagnostics** on the **Missingness** page, showing where values are absent and how this may affect downstream models.  
- **Preprocessing inspection** on the **Cleaning** page, documenting imputation strategies, scaling, and feature transformations used before training.  
- **Model performance comparison** using accuracy, precision, recall, F1-score, confusion matrices, and classification reports across all classifiers.  
- **Model behavior insights** through feature importance (for tree-based models) and explained variance / components (for PCA + Logistic Regression and SVD + Logistic Regression).
"""
    )

    st.subheader("User Experience")
    st.markdown(
        """
- Clean, intuitive interface with sidebar navigation across all analysis and prediction pages.
- Guided workflow through dedicated pages: Home, Dataset & Model Prediction, About the Datasets, EDA, Cleaning, Missingness, Prediction Models, Student Predictor, and Documentation.
- Consistent visual design for plots and results, helping distinguish analysis views, preprocessing summaries, and prediction outputs at a glance.
- Clear, stepwise instructions on each page so users know how to explore data, inspect preprocessing, choose models, and run individual student predictions.
- Layout optimized for readability on typical desktop screens, keeping navigation and content tightly integrated for educational use cases.
"""
    )

    st.markdown("---")

    # DATASETS
    st.header("Datasets")
    st.markdown(
        """
The app uses three derived datasets based on the UCI Student Performance data, focusing on grades and selected demographic or behavioral features.
"""
    )

    st.subheader("Dataset 1: Mathematics Performance")
    st.markdown(
        """
- Source: UCI Student Performance (Math track).  
- Approx. 395 student samples.  
- Example features: age, weekly study time, past failures, absences, period grades, final grade, and binary pass/fail result.  
- Includes moderate levels of missing values in selected numeric features.
"""
    )

    st.subheader("Dataset 2: Portuguese Performance")
    st.markdown(
        """
- Source: UCI Student Performance (Portuguese track).  
- Approx. 649 student samples.  
- Example features: age, parental education, travel time, family relationship quality, social outings, health status, absences, final grade, and pass/fail label.  
- Contains missing values in several socio-demographic attributes.
"""
    )

    st.subheader("Dataset 3: Comprehensive Academic Profile")
    st.markdown(
        """
- Source: Combined data constructed from Math and Portuguese tracks.  
- Approx. 500 student samples.  
- Example features: study time, past failures, academic support, family support, paid classes, extracurriculars, higher education aspiration, internet access, final grade, and pass/fail outcome.  
- Contains missing values in support- and behavior-related fields.
"""
    )

    st.markdown("---")

    # USER GUIDE
    st.header("Application User Guide")

    st.subheader("App Pages / Modules")
    st.markdown(
        """
- **Home**: For High-level description and quick links.  
- **Dataset & Model Prediction**: Select datasets, explore features, and make predictions.
- **About the Datasets**: Detailed dataset descriptions and statistics.
- **EDA**: Visualize distributions and correlations.
- **Cleaning**: Understand preprocessing steps, imputation methods.  
- **Missingness**: Analyze missing data patterns and strategies.
- **Prediction Models**: Explore implemented ML pipelines and their behavior.  
- **Student Predictor**: Input individual student data to obtain pass/fail predictions.
- **Documentation**: Comprehensive project documentation, methodology, and references.
"""
    )

    st.markdown("---")

    # MACHINE LEARNING MODELS
    st.header("Machine Learning Models")

    st.subheader("1. Logistic Regression")
    st.markdown(
        """
Logistic Regression models the probability that a student will pass or fail based on input features such as grades, attendance, and study habits.
It is a simple, fast, and interpretable baseline model, providing clear insights into how each feature affects the odds of passing.
"""
    )

    st.subheader("2. Random Forest")
    st.markdown(
        """
Random Forest is an ensemble of decision trees that combines many weak learners to make a robust classification.
It can capture non-linear relationships and complex interactions between features, often improving accuracy and robustness compared to linear models.
"""
    )

    st.subheader("3. Gradient Boosting")
    st.markdown(
        """
        Gradient Boosting builds trees sequentially, where each new tree corrects the errors of the previous ones.
This model is powerful for capturing subtle patterns in the data and can provide strong performance when tuned properly, especially on imbalanced or noisy educational datasets.
"""
    )
    
    st.subheader("4. PCA + Logistic Regression")
    st.markdown(
        """
The PCA + Logistic Regression pipeline first applies Principal Component Analysis to reduce the feature space into a smaller set of orthogonal components.
Logistic Regression is then trained on these components, which helps reduce multicollinearity, noise, and overfitting while still keeping the model relatively interpretable.
"""
    )
    
    st.subheader("5. SVD + Logistic Regression")
    st.markdown(
        """
The SVD + Logistic Regression pipeline uses Singular Value Decomposition to project the data into a low-dimensional latent space.
Logistic Regression is trained on these latent features, allowing the model to leverage compressed representations that capture key structure in the data, which can be useful when dealing with high-dimensional or sparse inputs.
"""
    )

    st.markdown("---")

    # EVALUATION METRICS
    st.header("Evaluation Metrics")

    st.subheader("Core Classification Metrics")
    st.markdown(
        """
The app evaluates all models using standard binary classification metrics:

- **Accuracy**: Proportion of correctly predicted students out of all students.  
- **Precision**: Fraction of predicted “pass” students who actually pass, helping assess false positives.  
- **Recall**: Fraction of actual “pass” students correctly identified by the model, highlighting missed positives.  
- **F1-Score**: Harmonic mean of precision and recall, providing a single score that balances both.
"""
    )

    st.subheader("Confusion Matrix")
    st.markdown(
        """
For each model, a **confusion matrix** summarizes prediction outcomes as:

- True Positives (TP): Students correctly predicted to pass.  
- True Negatives (TN): Students correctly predicted to fail.  
- False Positives (FP): Students predicted to pass but actually fail.  
- False Negatives (FN): Students predicted to fail but actually pass.  

This view helps understand the types of errors made by each model and whether it is conservative or aggressive in predicting passes.
"""
    )

    st.subheader("Classification Report")
    st.markdown(
        """
The app also produces a **classification report**, which breaks down precision, recall, F1-score, and support for each class (pass/fail).  
This detailed summary allows side-by-side inspection of how well models treat both passing and failing students.
"""
    )

    st.subheader("Feature Importance and Dimensionality Reduction")
    st.markdown(
        """
- **Random Forest** and **Gradient Boosting** expose **feature importance** values, indicating which features contribute most to the predictions.  
- **PCA + Logistic Regression** and **SVD + Logistic Regression** report **explained variance ratios** and the number of components used, showing how much information is preserved in the reduced feature space.
"""
    )

    st.markdown("---")

    # FUTURE ENHANCEMENTS
    st.header("Future Enhancements")
    st.markdown(
        """
- Add more algorithms such as Random Forest and gradient-boosting models.  
- Incorporate cross-validation and model selection workflows.  
- Implement feature selection strategies.  
- Provide an API endpoint for programmatic predictions.  
- Add authentication and support for custom dataset uploads.  
- Export detailed PDF reports and enable temporal trend analysis.
"""
    )

    st.markdown("---")

    # LICENSE & ACKNOWLEDGMENTS
    st.header("License & Acknowledgments")

    st.subheader("Acknowledgments")
    st.markdown(
        """
- UCI Machine Learning Repository for providing the Student Performance dataset.  
- Original dataset authors: P. Cortez and A. Silva (2008).  
"""
    )

    st.subheader("Dataset Reference")
    st.markdown(
        """
Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.*  
In A. Brito and J. Teixeira (Eds.), **Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008)**, Porto, Portugal, EUROSIS.
"""
    )
