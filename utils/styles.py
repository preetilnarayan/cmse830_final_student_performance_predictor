import streamlit as st


def apply_custom_styles():
    """Apply custom CSS styling with a wine-themed UI"""
    st.markdown("""
        <style>
        /* Palette:
           Background: #FFF6F8 (soft blush)
           Primary (wine):   #7B1E3A
           Secondary:        #B3475F
           Dark:             #3D0C1C
           Success:          #1E8E3E (green)
           Warning:          #FFD54F (bright yellow)
           Error:            #D32F2F (red)
        */

        /* App background + base text */
        .stApp {
            background-color: #FFF6F8;
            color: #241017;
        }

        body, p, span, label, div, h1, h2, h3, h4, h5, h6 {
            color: #241017;
        }

        /* Headers */
        .main-header {
            font-size: 2.5rem;
            color: #7B1E3A;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 800;
        }

        .sub-header {
            font-size: 1.8rem;
            color: #3D0C1C;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 700;
        }

        .section-header {
            font-size: 1.3rem;
            color: #7B1E3A;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        /* Generic cards */
        .metric-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #FBE1E8 100%);
            padding: 1.5rem;
            border-radius: 0.9rem;
            margin: 0.5rem 0;
            border-left: 5px solid #7B1E3A;
            color: #241017;
            box-shadow: 0 2px 6px rgba(61, 12, 28, 0.12);
        }

        /* Status boxes */
        .success-box {
            background-color: #E8F5E9;
            border-left: 5px solid #1E8E3E;
            padding: 1.5rem;
            margin: 1rem 0;
            color: #241017;
            border-radius: 0.7rem;
        }

        .warning-box {
            background-color: #FFF9E0;
            border-left: 5px solid #FFC107;
            padding: 1.5rem;
            margin: 1rem 0;
            color: #241017;
            border-radius: 0.7rem;
        }

        .info-box {
            background-color: #FBE1E8;
            border-left: 5px solid #7B1E3A;
            padding: 1.5rem;
            margin: 1rem 0;
            color: #241017;
            border-radius: 0.7rem;
        }

        /* Dataset highlight cards */
        .dataset-card {
            background: linear-gradient(135deg, #7B1E3A 0%, #3D0C1C 100%);
            padding: 2rem;
            border-radius: 1.1rem;
            margin: 1rem;
            color: #FFFFFF;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 6px 14px rgba(33, 2, 10, 0.35);
        }

        .dataset-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 18px rgba(33, 2, 10, 0.45);
        }

        .dataset-card h3 {
            color: #FFFFFF;
            margin-top: 0;
        }

        /* Primary buttons */
        .stButton>button {
            background: linear-gradient(135deg, #7B1E3A 0%, #3D0C1C 100%);
            color: #FFFFFF;
            border-radius: 0.7rem;
            padding: 0.5rem 2rem;
            font-weight: 600;
            border: none;
            transition: transform 0.1s ease, box-shadow 0.1s ease, opacity 0.1s ease;
        }

        .stButton>button:hover {
            opacity: 0.93;
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(61, 12, 28, 0.35);
        }

        .stButton>button:active {
            transform: translateY(0);
            box-shadow: none;
        }

        /* Download button */
        .stDownloadButton>button {
            background: linear-gradient(135deg, #3D0C1C 0%, #7B1E3A 100%);
            color: #FFFFFF !important;
            border-radius: 0.7rem;
            border: none;
            font-weight: 600;
        }

        .stDownloadButton>button * {
            color: #FFFFFF !important;
        }

        .stDownloadButton>button:hover {
            opacity: 0.93;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #3D0C1C 0%, #5A1428 50%, #7B1E3A 100%);
        }

        [data-testid="stSidebar"] * {
            color: #FFEFF3 !important;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #7B1E3A;
            font-weight: 700;
        }

        [data-testid="stMetricLabel"] {
            color: #5A1428;
        }

        /* Selectbox / widget labels */
        .stSelectbox label,
        .stRadio label,
        .stSlider label {
            color: #5A1428;
            font-weight: 600;
        }

        /* Selectbox body text */
        div[data-baseweb="select"] * {
            color: #2B131B !important;
        }

        /* =====================
           Tabs (wine styling)
           ===================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            padding-bottom: 0.3rem;
            border-bottom: 2px solid rgba(123, 30, 58, 0.25);
        }

        .stTabs [data-baseweb="tab"] {
            color: #5A1428;
            font-weight: 600;
            border-radius: 0.7rem 0.7rem 0 0;
            padding: 0.4rem 1.1rem;
            background-color: rgba(251, 225, 232, 0.7);
            transition: background-color 0.15s ease, color 0.15s ease, transform 0.1s ease;
            border: 1px solid transparent;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(248, 204, 215, 0.9);
            transform: translateY(-1px);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #7B1E3A !important;
            border-radius: 0.7rem 0.7rem 0 0;
            border: 1px solid #5A1428;
            box-shadow: 0 3px 8px rgba(61, 12, 28, 0.35);
            transform: translateY(-1px);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] * {
            color: #FFFFFF !important;
        }

        .stTabs [data-baseweb="tab-panel"] {
            border-radius: 0 0.7rem 0.7rem 0.7rem;
            border: 1px solid rgba(123, 30, 58, 0.15);
            padding: 1rem;
            margin-top: -1px;
            background-color: #FFFFFF;
        }

        /* =====================
           TABLE HEADERS
           ===================== */
        .dataframe thead tr th {
            background-color: #7B1E3A !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border-color: #5A1428 !important;
        }

        [data-testid="stDataFrame"] thead tr th {
            background-color: #7B1E3A !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        [data-testid="stDataFrame"] th {
            background-color: #7B1E3A !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        div[data-testid="stDataFrame"] div[role="columnheader"] {
            background-color: #7B1E3A !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        .stDataFrame thead {
            background-color: #7B1E3A !important;
        }

        table thead th,
        table thead td {
            background-color: #7B1E3A !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }

        .row_heading {
            background-color: #7B1E3A !important;
            color: #FFFFFF !important;
        }

        .col_heading {
            background-color: #7B1E3A !important;
            color: #FFFFFF !important;
        }

        .dataframe {
            border: 2px solid #7B1E3A !important;
            border-radius: 0.5rem;
            overflow: hidden;
        }

        /* Alerts */
        .stAlert.st-success {
            background-color: #E8F5E9;
            border-left: 5px solid #1E8E3E;
        }

        .stAlert.st-warning {
            background-color: #FFF9E0;
            border-left: 5px solid #FFC107;
        }

        .stAlert.st-error {
            background-color: #FFEBEE;
            border-left: 5px solid #D32F2F;
        }

        /* Code / JSON blocks */
        pre, code {
            background-color: #FBE1E8 !important;
            color: #2B131B !important;
        }

        /* Force white text */
        button[kind="primary"],
        button[kind="primary"] * {
            color: #FFFFFF !important;
        }

        /* Wine theme background for primary buttons */
        button[kind="primary"] {
            background: linear-gradient(135deg, #7B1E3A 0%, #3D0C1C 100%) !important;
            border: none !important;
            font-weight: 600 !important;
        }

        </style>
    """, unsafe_allow_html=True)
