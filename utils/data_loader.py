import pandas as pd
import numpy as np
import streamlit as st

# Dataset configurations
DATASET_CONFIGS = {
    1: {
        'url': "https://raw.githubusercontent.com/uciml/student-performance-dataset/master/student-mat.csv",
        'columns': ['age', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3'],
        'column_names': ['age', 'study_time_weekly', 'past_failures', 'absences',
                         'period1_grade', 'period2_grade', 'final_grade'],
        'missing_cols': ['study_time_weekly', 'absences', 'period1_grade'],
        'missing_rate': 0.12,
        'seed': 42,
        'n_fallback': 395
    },
    2: {
        'url': "https://raw.githubusercontent.com/uciml/student-performance-dataset/master/student-por.csv",
        'columns': ['age', 'Medu', 'Fedu', 'traveltime', 'famrel', 'goout', 'health', 'absences', 'G3'],
        'column_names': ['age', 'mother_education', 'father_education', 'travel_time',
                         'family_relationship', 'social_outings', 'health_status', 'absences', 'final_grade'],
        'missing_cols': ['mother_education', 'health_status', 'social_outings', 'travel_time'],
        'missing_rate': 0.10,
        'seed': 123,
        'n_fallback': 649
    },
    3: {
        'urls': {
            'math': "https://raw.githubusercontent.com/uciml/student-performance-dataset/master/student-mat.csv",
            'por': "https://raw.githubusercontent.com/uciml/student-performance-dataset/master/student-por.csv"
        },
        'columns': ['age', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
                    'activities', 'higher', 'internet', 'G3'],
        'column_names': ['age', 'study_time_weekly', 'past_failures', 'school_support',
                         'family_support', 'paid_classes', 'extracurricular',
                         'higher_ed_aspiration', 'internet_access', 'final_grade'],
        'binary_cols': ['school_support', 'family_support', 'paid_classes', 'extracurricular',
                        'higher_ed_aspiration', 'internet_access'],
        'missing_high': ['paid_classes', 'extracurricular', 'study_time_weekly'],
        'missing_moderate': ['school_support', 'internet_access'],
        'missing_rate_high': 0.18,
        'missing_rate_moderate': 0.10,
        'seed': 456,
        'n_fallback': 500,
        'rows_per_source': 250
    }
}

# Fallback data ranges
FALLBACK_RANGES = {
    1: {
        'age': (15, 23),
        'study_time_weekly': (1, 5),
        'past_failures': (0, 4),
        'absences': (0, 40),
        'period1_grade': (0, 20),
        'period2_grade': (0, 20),
        'final_grade': (0, 20)
    },
    2: {
        'age': (15, 23),
        'mother_education': (0, 5),
        'father_education': (0, 5),
        'travel_time': (1, 5),
        'family_relationship': (1, 6),
        'social_outings': (1, 6),
        'health_status': (1, 6),
        'absences': (0, 40),
        'final_grade': (0, 20)
    },
    3: {
        'age': (15, 23),
        'study_time_weekly': (1, 5),
        'past_failures': (0, 4),
        'school_support': (0, 2),
        'family_support': (0, 2),
        'paid_classes': (0, 2),
        'extracurricular': (0, 2),
        'higher_ed_aspiration': (0, 2),
        'internet_access': (0, 2),
        'final_grade': (0, 20)
    }
}


def add_missing_values(df, columns, missing_rate, seed):
    """Add missing values to specified columns."""
    np.random.seed(seed)
    for col in columns:
        missing_idx = np.random.choice(df.index, size=int(len(df) * missing_rate), replace=False)
        df.loc[missing_idx, col] = np.nan
    return df


def add_result_column(df):
    """Add Pass/Fail result column based on final grade."""
    df['result'] = df['final_grade'].apply(lambda x: 'Pass' if x >= 10 else 'Fail')
    return df


@st.cache_data
def load_dataset_1():
    """Dataset 1: UCI Student Performance - Mathematics."""
    config = DATASET_CONFIGS[1]

    try:
        df = pd.read_csv(config['url'], sep=';')
        df_subset = df[config['columns']].copy()
        df_subset.columns = config['column_names']

        df_subset = add_missing_values(
            df_subset, config['missing_cols'],
            config['missing_rate'], config['seed']
        )
        return add_result_column(df_subset)
    except Exception:
        return generate_fallback_dataset(1)


@st.cache_data
def load_dataset_2():
    """Dataset 2: UCI Student Performance - Portuguese."""
    config = DATASET_CONFIGS[2]

    try:
        df = pd.read_csv(config['url'], sep=';')
        df_subset = df[config['columns']].copy()
        df_subset.columns = config['column_names']

        df_subset = add_missing_values(
            df_subset, config['missing_cols'],
            config['missing_rate'], config['seed']
        )
        return add_result_column(df_subset)
    except Exception:
        return generate_fallback_dataset(2)


@st.cache_data
def load_dataset_3():
    """Dataset 3: Extended Student Performance with Multiple Subjects."""
    config = DATASET_CONFIGS[3]

    try:
        df_math = pd.read_csv(config['urls']['math'], sep=';')
        df_por = pd.read_csv(config['urls']['por'], sep=';')

        df_math_subset = df_math[config['columns']].head(config['rows_per_source'])
        df_por_subset = df_por[config['columns']].head(config['rows_per_source'])

        df_combined = pd.concat([df_math_subset, df_por_subset], ignore_index=True)
        df_combined.columns = config['column_names']

        # Convert binary columns from yes/no to 1/0
        for col in config['binary_cols']:
            df_combined[col] = df_combined[col].map({'yes': 1, 'no': 0})

        # Add missing values with different rates
        np.random.seed(config['seed'])
        df_combined = add_missing_values(
            df_combined, config['missing_high'],
            config['missing_rate_high'], config['seed']
        )
        df_combined = add_missing_values(
            df_combined, config['missing_moderate'],
            config['missing_rate_moderate'], config['seed'] + 1
        )

        return add_result_column(df_combined)
    except Exception:
        return generate_fallback_dataset(3)


def generate_fallback_dataset(dataset_id):
    """Generate fallback dataset with missing values."""
    config = DATASET_CONFIGS[dataset_id]
    ranges = FALLBACK_RANGES[dataset_id]

    np.random.seed(config['seed'])
    n = config['n_fallback']

    # Generate base data
    df = pd.DataFrame({
        col: np.random.randint(low, high, n)
        for col, (low, high) in ranges.items()
    })

    # Add missing values based on dataset type
    if dataset_id == 3:
        df = add_missing_values(
            df, config['missing_high'],
            config['missing_rate_high'], config['seed']
        )
        df = add_missing_values(
            df, config['missing_moderate'],
            config['missing_rate_moderate'], config['seed'] + 1
        )
    else:
        df = add_missing_values(
            df, config['missing_cols'],
            config['missing_rate'], config['seed']
        )

    return add_result_column(df)