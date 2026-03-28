# -*- coding: utf-8 -*-
"""
Industrial Production Systems - Streamlit Dashboard
Converted from Colab notebook
"""

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # non-interactive
import matplotlib.pyplot as plt

import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industrial Production Systems Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 4px 0 0; opacity: 0.85; font-size: 0.9rem; }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2c3e50;
        border-left: 5px solid #667eea;
        padding-left: 12px;
        margin: 20px 0 10px;
    }
    .insight-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING AND PREPROCESSING ───────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_and_preprocess():
    url = "https://raw.githubusercontent.com/Zuckmo/Production-System/refs/heads/main/production_data_processed.csv"
    df = pd.read_csv(url)

    # Timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Fill NaN with column mean
    df = df.fillna(df.mean(numeric_only=True))

    # Winsorization (IQR capping) — skip units_produced & defect_count
    numerical_cols = [
        "temperature", "vibration_level", "power_consumption", "pressure",
        "material_flow_rate", "cycle_time", "error_rate", "machine_age_hours",
        "last_maintenance_hours", "oil_level", "downtime",
        "ambient_temperature", "humidity", "noise_level_db"
    ]
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # ── Machine-level Clustering ──────────────────────────────────────────
    clustering_features = [
        'temperature', 'vibration_level', 'power_consumption', 'pressure',
        'oil_level', 'machine_age_hours', 'error_rate', 'noise_level_db'
    ]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[clustering_features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_efficiency_machine'] = kmeans.fit_predict(df_scaled)

    # ── Shift-level Clustering ────────────────────────────────────────────
    df['date'] = df['timestamp'].dt.date
    df['shift_instance_id'] = df['date'].astype(str) + '_' + df['shift']

    grouped = df.groupby(['shift_instance_id', 'machine_id', 'line_id']).agg(
        units_produced=('units_produced', 'sum'),
        downtime=('downtime', 'sum'),
        error_rate=('error_rate', 'mean'),
        cycle_time=('cycle_time', 'mean')
    ).reset_index()

    grouped['efficiency_error_rate']     = 1 - grouped['error_rate']
    grouped['efficiency_downtime']       = 1 / (1 + grouped['downtime'])
    grouped['efficiency_cycle_time']     = 1 / (1 + grouped['cycle_time'])
    grouped['efficiency_units_produced'] = grouped['units_produced']

    eff_cols = ['efficiency_error_rate','efficiency_downtime',
                'efficiency_cycle_time','efficiency_units_produced']
    scaler2 = StandardScaler()
    grouped[eff_cols] = scaler2.fit_transform(grouped[eff_cols])

    kmeans2 = KMeans(n_clusters=2, random_state=42, n_init='auto')
    grouped['shift_efficiency_cluster'] = kmeans2.fit_predict(grouped[eff_cols])

    df = df.merge(
        grouped[['shift_instance_id','machine_id','line_id','shift_efficiency_cluster']],
        on=['shift_instance_id','machine_id','line_id'], how='left'
    )

    return df, df_scaled, clustering_features

df, df_scaled, clustering_features = load_and_preprocess()

# ─── SIDEBAR NAVIGATION ──────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/factory.png", width=72)
st.sidebar.title("🏭 Industrial Production")
st.sidebar.markdown("---")

pages = [
    "Overview & Dataset",
    " Exploratory Data Analysis",
    " KPI & Performance Metrics",
    "Machine Condition Clustering",
    "Shift Efficiency Clustering",
    "Temperature Impact Analysis",
    "Defect & Quality Analysis",
    "Bottleneck Identification",]
    
page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset rows: **{len(df)}** | Columns: **{len(df.columns)}**") 

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & DATASET
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview & Dataset":
    st.title("Industrial Production Systems Dashboard")
    st.markdown(" comprehensive analysis of production data from machine sensors to clustering" )
    
    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    total_units = int(df['units_produced'].sum())
    total_defects = int(df['defect_count'].sum())
    avg_oee_avail = (100)
    
    