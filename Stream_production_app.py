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
    page_title="Industrial Production Systems",
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

# ─── DATA LOADING ────────────────────────────────────────────────────────────
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
    "📊 Overview & Dataset",
    "🔍 Exploratory Data Analysis",
    "📈 KPI & Performance Metrics",
    "🤖 Machine Condition Clustering",
    "⏱️ Shift Efficiency Clustering",
    "🌡️ Temperature Impact Analysis",
    "⚠️ Defect & Quality Analysis",
    "🚦 Bottleneck Analysis",
    "🧬 Predictive Modeling",
]
page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset rows: **{len(df):,}** | Columns: **{df.shape[1]}**")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & DATASET
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & Dataset":
    st.title("🏭 Industrial Production Systems Dashboard")
    st.markdown("Analisis komprehensif data produksi industri — dari sensor mesin hingga prediksi kegagalan.")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    total_units = int(df['units_produced'].sum())
    total_defects = int(df['defect_count'].sum())
    avg_oee_avail = (10080 - 3204) / 10080
    total_downtime = df['downtime'].sum()
    avg_error = df['error_rate'].mean()

    for col, val, label in zip(
        [col1, col2, col3, col4, col5],
        [f"{total_units:,}", f"{total_defects:,}", f"{avg_oee_avail:.1%}",
         f"{total_downtime:,.0f}", f"{avg_error:.3f}"],
        ["Total Units Produced", "Total Defects", "Availability (OEE)",
         "Total Downtime (min)", "Avg Error Rate"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <h2>{val}</h2><p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Dataset description
    st.markdown('<div class="section-header">📋 Dataset Description</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Head", "Describe", "Info"])
    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
    with tab2:
        st.dataframe(df.describe().T.style.background_gradient(cmap='Blues'), use_container_width=True)
    with tab3:
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype": df.dtypes.values,
            "Non-Null": df.notnull().sum().values,
            "Null": df.isnull().sum().values,
            "Unique": df.nunique().values
        })
        st.dataframe(info_df, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🔢 Categorical Value Counts</div>', unsafe_allow_html=True)
    cat_cols = ["machine_type","line_id","shift","product_type","machine_id","lubrication_status"]
    sel = st.selectbox("Pilih kolom", cat_cols)
    vc = df[sel].value_counts().reset_index()
    vc.columns = [sel, "count"]
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=vc, x=sel, y="count", palette="viridis", ax=ax)
    ax.set_title(f"Value Counts: {sel}")
    ax.tick_params(axis='x', rotation=30)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")

    numeric_cols = [
        "temperature","vibration_level","power_consumption","pressure",
        "material_flow_rate","cycle_time","error_rate","machine_age_hours",
        "last_maintenance_hours","oil_level","units_produced","downtime",
        "ambient_temperature","humidity","noise_level_db","defect_count"
    ]

    tab1, tab2, tab3 = st.tabs(["📦 Boxplots", "🔥 Correlation Matrix", "📉 Distributions"])

    with tab1:
        st.markdown("#### Boxplots — Deteksi Outlier (setelah Winsorization)")
        n_cols = 3
        n_rows = int(np.ceil(len(numeric_cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
        axes = axes.flatten()
        for i, col in enumerate(numeric_cols):
            sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
            axes[i].set_title(col, fontsize=10)
            axes[i].grid(axis='y', linestyle='--', alpha=0.6)
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        st.markdown("#### Correlation Matrix Heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",
                    linewidths=.5, ax=ax, annot_kws={"size": 7})
        ax.set_title("Correlation Matrix of Industrial Sensor Data")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown("""
        <div class="insight-box">
        <b>💡 Key Correlations:</b><br>
        • <b>machine_age_hours ↔ oil_level (0.65)</b> — Terkuat: Mesin lebih tua membutuhkan lebih banyak pelumas<br>
        • <b>error_rate</b> berkorelasi moderat dengan vibration_level (0.34), power_consumption (0.34), temperature (0.25)<br>
        • pressure, material_flow_rate, cycle_time menunjukkan korelasi lemah — kemungkinan fitur independen
        </div>""", unsafe_allow_html=True)

    with tab3:
        sel_col = st.selectbox("Pilih kolom untuk distribusi", numeric_cols, key="dist_sel")
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.histplot(df[sel_col], kde=True, bins=30, color='steelblue', ax=ax)
        ax.set_title(f"Distribution of {sel_col.replace('_',' ').title()}")
        ax.set_xlabel(sel_col.replace('_',' ').title())
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — KPI & PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 KPI & Performance Metrics":
    st.title("📈 KPI & Performance Metrics (OEE)")

    total_units = df['units_produced'].sum()
    total_defects = df['defect_count'].sum()
    good_count = total_units - total_defects

    availability  = (10080 - 3204) / 10080
    performance   = (120 * total_units) / (10080 - 3204)
    quality       = good_count / total_units
    oee           = availability * performance * quality

    col1, col2, col3, col4 = st.columns(4)
    for col, val, label, color in zip(
        [col1, col2, col3, col4],
        [f"{availability:.2%}", f"{performance:.4f}", f"{quality:.2%}", f"{oee:.4f}"],
        ["Availability", "Performance", "Quality", "OEE"],
        ["#3498db","#2ecc71","#e67e22","#9b59b6"]
    ):
        col.markdown(f"""
        <div style="background:{color};border-radius:10px;padding:18px;color:white;text-align:center;">
            <h2 style="margin:0;font-size:1.8rem">{val}</h2>
            <p style="margin:4px 0 0;opacity:.85">{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["⏱ Cycle Time", "⬇️ Downtime", "👷 Operator"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            avg_machine = df.groupby('machine_id')['cycle_time'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(data=avg_machine, x='machine_id', y='cycle_time', palette='Blues_d', ax=ax)
            ax.set_title("Avg Cycle Time per Machine")
            ax.set_ylabel("Seconds")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col_b:
            avg_shift = df.groupby('shift')['cycle_time'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(data=avg_shift, x='shift', y='cycle_time', palette='Greens_d', ax=ax)
            ax.set_title("Avg Cycle Time per Shift")
            ax.tick_params(axis='x', rotation=25)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with tab2:
        dt_total = df.groupby('machine_id')['downtime'].sum().reset_index()
        dt_freq  = df.groupby('machine_id')['downtime'].apply(lambda x: (x>0).sum()).reset_index()
        dt_freq.columns = ['machine_id','frequency']

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(data=dt_total, x='machine_id', y='downtime', palette='Reds_d', ax=ax)
            ax.set_title("Total Downtime per Machine")
            ax.set_ylabel("Minutes")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col_b:
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(data=dt_freq, x='machine_id', y='frequency', palette='Oranges_d', ax=ax)
            ax.set_title("Downtime Frequency per Machine")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("""
        <div class="insight-box">
        ⚠️ <b>M001</b> memiliki total downtime tertinggi dan frekuensi downtime terbanyak — mesin paling bermasalah.
        </div>""", unsafe_allow_html=True)

    with tab3:
        err_op = df.groupby('operator_id')['error_rate'].sum().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots(figsize=(10,4))
        sns.barplot(data=err_op, x='operator_id', y='error_rate', palette='rocket', ax=ax)
        ax.set_title("Total Error Rate per Operator")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Defect % per operator
        def_op = df.groupby('operator_id').agg(
            total_units=('units_produced','sum'),
            total_defects=('defect_count','sum')
        )
        def_op['defect_pct'] = (def_op['total_defects'] / def_op['total_units']).fillna(0)*100
        st.dataframe(def_op.sort_values('defect_pct', ascending=False).round(2), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MACHINE CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Machine Condition Clustering":
    st.title("🤖 Machine Condition Clustering (K-Means, k=3)")

    tab1, tab2, tab3 = st.tabs(["📌 PCA Scatter", "📊 Cluster Centroids", "📉 Elbow & Metrics"])

    with tab1:
        pca = PCA(n_components=2, random_state=42)
        pca_components = pca.fit_transform(df_scaled)
        df_pca = pd.DataFrame(pca_components, columns=['PCA1','PCA2'])
        df_pca['Cluster'] = df['cluster_efficiency_machine'].values

        fig, ax = plt.subplots(figsize=(9,6))
        palette = {0:'#e74c3c', 1:'#3498db', 2:'#2ecc71'}
        labels_map = {0:'Cluster 0 (Degraded/Critical)', 1:'Cluster 1 (Moderate/Low Oil)', 2:'Cluster 2 (Older, Stable & Healthy)'}
        for cid, grp in df_pca.groupby('Cluster'):
            ax.scatter(grp['PCA1'], grp['PCA2'], c=palette[cid], label=labels_map[cid],
                       s=25, alpha=0.6, edgecolors='none')
        ax.set_title("PCA of Machine Condition Clusters")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""
        <div class="insight-box">
        <b>Interpretasi Kluster:</b><br>
        🔴 <b>Cluster 0 – Degraded/Critical</b>: Temperatur tinggi (~86°C), vibrasi tinggi, error_rate ~0.98. Perlu perhatian segera.<br>
        🔵 <b>Cluster 1 – Moderate, Low Oil</b>: Mesin lebih muda, oil_level rendah (~56%). Risiko keausan akselerasi.<br>
        🟢 <b>Cluster 2 – Older, Stable, Healthy</b>: Mesin tertua tapi well-maintained, oil_level tinggi (~79%).
        </div>""", unsafe_allow_html=True)

    with tab2:
        centroids = df.groupby('cluster_efficiency_machine')[clustering_features].mean().round(2)
        st.dataframe(centroids.T.style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)

        # Distribution count
        cluster_counts = df['cluster_efficiency_machine'].value_counts().sort_index().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        cluster_counts['Cluster'] = cluster_counts['Cluster'].map({
            0:'Cluster 0\n(Degraded/Critical)',
            1:'Cluster 1\n(Moderate/Low Oil)',
            2:'Cluster 2\n(Older, Stable)'
        })
        fig, ax = plt.subplots(figsize=(7,4))
        colors = ['#e74c3c','#3498db','#2ecc71']
        ax.bar(cluster_counts['Cluster'], cluster_counts['Count'], color=colors)
        ax.set_title("Distribusi Kluster Mesin")
        ax.set_ylabel("Jumlah Observasi")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab3:
        st.markdown("#### Evaluasi Jumlah Kluster Optimal")
        inertia_list, sil_list, db_list = [], [], []
        k_range = range(2, 10)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(df_scaled)
            inertia_list.append(km.inertia_)
            sil_list.append(silhouette_score(df_scaled, lbl))
            db_list.append(davies_bouldin_score(df_scaled, lbl))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(k_range, inertia_list, 'o-', color='blue'); axes[0].set_title("Elbow Method"); axes[0].set_xlabel("k"); axes[0].grid(True)
        axes[1].plot(k_range, sil_list, 'o-', color='green'); axes[1].set_title("Silhouette Score"); axes[1].set_xlabel("k"); axes[1].grid(True)
        axes[2].plot(k_range, db_list, 'o-', color='red'); axes[2].set_title("Davies-Bouldin Index"); axes[2].set_xlabel("k"); axes[2].grid(True)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SHIFT EFFICIENCY CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⏱️ Shift Efficiency Clustering":
    st.title("⏱️ Shift Efficiency Clustering (K-Means, k=2)")

    # Re-compute shift aggregated
    df['date'] = df['timestamp'].dt.date
    df['shift_instance_id'] = df['date'].astype(str) + '_' + df['shift']

    grouped = df.groupby(['shift_instance_id','machine_id','line_id']).agg(
        units_produced=('units_produced','sum'),
        downtime=('downtime','sum'),
        error_rate=('error_rate','mean'),
        cycle_time=('cycle_time','mean')
    ).reset_index()

    grouped['efficiency_error_rate']     = 1 - grouped['error_rate']
    grouped['efficiency_downtime']       = 1 / (1 + grouped['downtime'])
    grouped['efficiency_cycle_time']     = 1 / (1 + grouped['cycle_time'])
    grouped['efficiency_units_produced'] = grouped['units_produced']

    eff_cols = ['efficiency_error_rate','efficiency_downtime','efficiency_cycle_time','efficiency_units_produced']
    scaler2 = StandardScaler()
    grouped[eff_cols] = scaler2.fit_transform(grouped[eff_cols])

    kmeans2 = KMeans(n_clusters=2, random_state=42, n_init='auto')
    grouped['shift_efficiency_cluster'] = kmeans2.fit_predict(grouped[eff_cols])

    centroids_shift = grouped.groupby('shift_efficiency_cluster')[
        ['units_produced','downtime','error_rate','cycle_time']].mean().round(2)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Cluster Centroids (Original Values)")
        st.dataframe(centroids_shift.T.style.background_gradient(cmap='Blues', axis=1), use_container_width=True)
    with col_b:
        cnt = grouped['shift_efficiency_cluster'].value_counts().sort_index().reset_index()
        cnt.columns = ['Cluster','Count']
        cnt['Cluster'] = cnt['Cluster'].map({0:'Cluster 0\n(Higher Efficiency)', 1:'Cluster 1\n(Lower Efficiency)'})
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(cnt['Cluster'], cnt['Count'], color=['#2ecc71','#e74c3c'])
        ax.set_title("Distribusi Kluster Shift")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("""
    <div class="insight-box">
    <b>Interpretasi:</b><br>
    🟢 <b>Cluster 0 – Higher Efficiency</b>: Produksi ~33 unit/shift, downtime lebih tinggi namun output jauh lebih besar<br>
    🔴 <b>Cluster 1 – Lower Efficiency</b>: Produksi ~14 unit/shift, error rate sedikit lebih tinggi meski downtime lebih rendah
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Distribusi Shift per Efisiensi Kluster")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for cid, color, label in [(0,'#2ecc71','Higher Efficiency'), (1,'#e74c3c','Lower Efficiency')]:
        merged = df.merge(grouped[['shift_instance_id','machine_id','line_id','shift_efficiency_cluster']],
                          on=['shift_instance_id','machine_id','line_id'], how='left', suffixes=('','_new'))
        col_name = 'shift_efficiency_cluster_new' if 'shift_efficiency_cluster_new' in merged.columns else 'shift_efficiency_cluster'
        sub = merged[merged[col_name] == cid]
        vc = sub['shift'].value_counts()
        axes[cid].bar(vc.index, vc.values, color=color)
        axes[cid].set_title(f"Cluster {cid}: {label} — Shift Distribution")
        axes[cid].tick_params(axis='x', rotation=20)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — TEMPERATURE IMPACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌡️ Temperature Impact Analysis":
    st.title("🌡️ Temperature Impact Analysis")

    Q1_t = df['temperature'].quantile(0.25)
    Q3_t = df['temperature'].quantile(0.75)

    df_low    = df[df['temperature'] <= Q1_t]
    df_normal = df[(df['temperature'] > Q1_t) & (df['temperature'] < Q3_t)]
    df_high   = df[df['temperature'] >= Q3_t]

    metrics = ['power_consumption','error_rate','downtime','units_produced']
    scenarios = ['Low Temp', 'Normal Temp', 'High Temp']
    data_scenarios = [df_low, df_normal, df_high]

    summary = pd.DataFrame({
        s: d[metrics].mean() for s, d in zip(scenarios, data_scenarios)
    }).T.round(3)

    col_a, col_b = st.columns([1,1.5])
    with col_a:
        st.markdown(f"**Threshold:** Q1={Q1_t:.1f}°C | Q3={Q3_t:.1f}°C")
        st.dataframe(summary.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    with col_b:
        corr_f = ['temperature','power_consumption','error_rate','downtime','units_produced']
        corr_vec = df[corr_f].corr()['temperature'].drop('temperature').round(3)
        fig, ax = plt.subplots(figsize=(5,3))
        colors = ['#e74c3c' if v>0 else '#3498db' for v in corr_vec.values]
        ax.barh(corr_vec.index, corr_vec.values, color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title("Correlation with Temperature")
        ax.set_xlabel("Pearson r")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Impact of Temperature Scenarios on Key Metrics", fontsize=13)
    colors_bar = ['#3498db','#f39c12','#e74c3c']
    for i, metric in enumerate(metrics):
        means = [d[metric].mean() for d in data_scenarios]
        axes[i].bar(scenarios, means, color=colors_bar)
        axes[i].set_title(metric.replace('_',' ').title())
        axes[i].set_ylabel("Average Value")
        axes[i].tick_params(axis='x', rotation=20)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class="insight-box">
    <b>💡 Findings:</b> Temperatur tinggi → power consumption naik, error rate naik, units produced turun.
    Pertahankan temperatur operasi di bawah Q3 untuk efisiensi optimal.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — DEFECT & QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Defect & Quality Analysis":
    st.title("⚠️ Defect & Quality Analysis")

    total_units  = df['units_produced'].sum()
    total_defects = df['defect_count'].sum()
    good_count   = total_units - total_defects

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Units", f"{int(total_units):,}")
    col2.metric("Good Units",  f"{int(good_count):,}", delta=f"{good_count/total_units:.1%}")
    col3.metric("Defective",   f"{int(total_defects):,}", delta=f"-{total_defects/total_units:.1%}", delta_color="inverse")

    tab1, tab2, tab3 = st.tabs(["🥧 Pie Charts", "📊 By Category", "🔬 By Machine Cluster"])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(5,5))
            ax.pie([good_count, total_defects], labels=['Good','Defective'],
                   colors=['#66b3ff','#ff9999'], autopct='%1.1f%%',
                   explode=(0.05,0), shadow=True, startangle=140)
            ax.set_title("Good vs Defective Units")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col_b:
            dc_prod = df.groupby('product_type')['defect_count'].sum()
            fig, ax = plt.subplots(figsize=(5,5))
            ax.pie(dc_prod.values, labels=dc_prod.index, autopct='%1.1f%%',
                   startangle=90, colors=sns.color_palette('viridis', len(dc_prod)))
            ax.set_title("Defect % per Product Type")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            


    with tab2:
        cat_choice = st.selectbox("Lihat defect rate berdasarkan", ["machine_type","machine_id","lubrication_status","product_type"])
        dr = df.groupby(cat_choice)['defect_count'].mean().sort_values(ascending=False).reset_index()
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=dr, x=cat_choice, y='defect_count', palette='plasma', ax=ax)
        ax.set_title(f"Average Defect Rate by {cat_choice.replace('_',' ').title()}")
        ax.set_ylabel("Avg Defect Count")
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab3:
        dr_cluster = df.groupby('cluster_efficiency_machine')['defect_count'].mean().reset_index()
        dr_cluster['Cluster'] = dr_cluster['cluster_efficiency_machine'].map({
            0:'Cluster 0\n(Degraded)', 1:'Cluster 1\n(Moderate)', 2:'Cluster 2\n(Stable)'})
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(dr_cluster['Cluster'], dr_cluster['defect_count'], color=['#e74c3c','#3498db','#2ecc71'])
        ax.set_title("Avg Defect Rate by Machine Condition Cluster")
        ax.set_ylabel("Avg Defect Count")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — BOTTLENECK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚦 Bottleneck Analysis":
    st.title("🚦 Bottleneck Analysis")

    total_dt_line  = df.groupby('line_id')['downtime'].sum()
    freq_dt_line   = df.groupby('line_id')['downtime'].apply(lambda x: (x>0).sum())
    avg_err_line   = df.groupby('line_id')['error_rate'].mean()
    total_up_line  = df.groupby('line_id')['units_produced'].sum()

    # Summary table
    bottleneck_df = pd.DataFrame({
        'Total Downtime': total_dt_line,
        'Downtime Freq': freq_dt_line,
        'Avg Error Rate': avg_err_line.round(3),
        'Total Units': total_up_line
    })
    st.dataframe(bottleneck_df.style.background_gradient(cmap='Reds', subset=['Total Downtime','Downtime Freq'])
                               .background_gradient(cmap='Greens', subset=['Total Units']), use_container_width=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    titles = ['Total Downtime per Line','Downtime Frequency per Line',
              'Average Error Rate per Line','Total Units Produced per Line']
    datasets = [total_dt_line, freq_dt_line, avg_err_line, total_up_line]
    ylabels  = ['Minutes','Occurrences','Avg Error Rate','Units']
    palettes = ['viridis','magma','cividis','plasma']

    for ax, title, data, ylabel, pal in zip(axes, titles, datasets, ylabels, palettes):
        sns.barplot(x=data.index, y=data.values, ax=ax, palette=pal)
        ax.set_title(title)
        ax.set_xlabel("Line ID")
        ax.set_ylabel(ylabel)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    <div class="insight-box">
    <b>⚡ Bottleneck Findings:</b><br>
    1. <b>Machine M001</b> — Downtime tertinggi (25,861 min) & frekuensi terbanyak (960 kejadian) → prioritas maintenance<br>
    2. <b>Cluster 1 (Lower Efficiency Shifts)</b> — Produksi 14 unit/shift vs 33 unit/shift, error rate lebih tinggi meski downtime rendah → perlu investigasi proses
    </div>""", unsafe_allow_html=True)


# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🏭 Industrial Production Systems Dashboard | Data: production_data_processed.csv")