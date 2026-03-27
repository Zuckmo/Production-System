import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    print('XGBoost available ✓')
except ImportError:
    print('XGBoost not installed — install with: pip install xgboost')

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    print('imbalanced-learn available ✓')
except ImportError:
    print('imbalanced-learn not installed — install with: pip install imbalanced-learn')

sns.set_theme(style='whitegrid', palette='viridis')
plt.rcParams['figure.dpi'] = 100
print('All libraries imported ✓')


url = 'https://raw.githubusercontent.com/Zuckmo/Production-System/refs/heads/main/production_data_processed.csv'
df = pd.read_csv(url)
print(f'Dataset shape: {df.shape}')
df.head()

# Statistik deskriptif
df.describe().T.style.background_gradient(cmap='Blues')

df.info()

# Cek missing values
missing = df.isnull().sum()
print('Missing Values per Column:')
print(missing[missing > 0])

# Value counts kolom kategorikal
cat_cols = ['machine_type','line_id','shift','operator_id','product_type','machine_id','lubrication_status']
for col in cat_cols:
    print(f'{col}:\n{df[col].value_counts()}\n')
    
## 4. Data Preprocessing

### 4.1 Convert Timestamp

df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f'Timestamp range: {df["timestamp"].min()} → {df["timestamp"].max()}')

### 4.2 Handle Missing Values

print('Missing values sebelum imputasi:')
print(df.isnull().sum()[df.isnull().sum() > 0])

df = df.fillna(df.mean(numeric_only=True))
print('\nMissing values setelah imputasi:', df.isnull().sum().sum())

### 4.3 Check Duplicates

print(f'Duplicate rows: {df.duplicated().sum()}')

### 4.4 Outlier Detection (IQR Method)

print('Checking for outliers using IQR method:')
for column in df.select_dtypes(include=['number']).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    if not outliers.empty:
        print(f"  '{column}': {len(outliers)} outliers")
    else:
        print(f"  '{column}': No outliers")

### 4.5 Visualisasi Outlier — Boxplots

columns_for_boxplot = [
    'temperature','vibration_level','power_consumption','pressure',
    'material_flow_rate','cycle_time','error_rate','machine_age_hours',
    'last_maintenance_hours','oil_level','units_produced','downtime',
    'ambient_temperature','humidity','noise_level_db'
]

n_cols = 3
n_rows = int(np.ceil(len(columns_for_boxplot) / n_cols))
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 4*n_rows))
axes = axes.flatten()

for i, col in enumerate(columns_for_boxplot):
    sns.boxplot(y=df[col], ax=axes[i], color='lightcoral')
    axes[i].set_title(col, fontsize=9)
    axes[i].grid(axis='y', linestyle='--', alpha=0.6)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Boxplots — Before Winsorization', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

### 4.6 Winsorization (Percentile Capping)

numerical_cols = [
    'temperature','vibration_level','power_consumption','pressure',
    'material_flow_rate','cycle_time','error_rate','machine_age_hours',
    'last_maintenance_hours','oil_level','downtime',
    'ambient_temperature','humidity','noise_level_db'
]

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

print('Winsorization selesai ✓')

## 5. Correlation Matrix

numeric_cols = [
    'temperature','vibration_level','power_consumption','pressure',
    'material_flow_rate','cycle_time','error_rate','machine_age_hours',
    'last_maintenance_hours','oil_level','units_produced','downtime',
    'ambient_temperature','humidity','noise_level_db','defect_count'
]

correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation_matrix,
    annot=True, cmap='coolwarm', fmt='.2f',
    linewidths=.5, annot_kws={'size': 7}
)
plt.title('Correlation Matrix of Industrial Sensor Data', fontsize=14)
plt.tight_layout()
plt.show()

### Interpretasi Correlation Matrix

total_units_produced = df['units_produced'].sum()
total_defect_count = df['defect_count'].sum()
good_count = total_units_produced - total_defect_count

# Availability: planned=10080 min, downtime rows=3204
availability = (10080 - 3204) / 10080

# Performance: ideal cycle time = 120s
performance = (120 * total_units_produced) / (10080 - 3204)

# Quality
quality = good_count / total_units_produced

# OEE
OEE = availability * performance * quality

print(f'Total Units Produced : {total_units_produced:,.0f}')
print(f'Good Units           : {good_count:,.0f}')
print(f'Defective Units      : {total_defect_count:,.0f}')
print()
print(f'Availability  : {availability:.4f} ({availability:.2%})')
print(f'Performance   : {performance:.4f}')
print(f'Quality       : {quality:.4f} ({quality:.2%})')
print(f'OEE           : {OEE:.4f}')

### 6.1 Average Cycle Time per Machine / Shift / Line

avg_cycle_time_per_machine = df.groupby('machine_id')['cycle_time'].mean()
avg_cycle_time_per_shift   = df.groupby('shift')['cycle_time'].mean()
avg_cycle_time_per_line    = df.groupby('line_id')['cycle_time'].mean()

print('Average Cycle Time per Machine (seconds):')
print(avg_cycle_time_per_machine.round(2), '\n')

print('Average Cycle Time per Shift:')
print(avg_cycle_time_per_shift.round(2), '\n')

print('Average Cycle Time per Line:')
print(avg_cycle_time_per_line.round(2))

### 6.2 Total Downtime & Frequency per Machine

total_downtime_per_machine    = df.groupby('machine_id')['downtime'].sum()
downtime_frequency_per_machine = df.groupby('machine_id')['downtime'].apply(lambda x: (x>0).sum())

print('Total Downtime per Machine:')
print(total_downtime_per_machine)

print('\nDowntime Frequency per Machine:')
print(downtime_frequency_per_machine)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(x=total_downtime_per_machine.index, y=total_downtime_per_machine.values, ax=axes[0], palette='Reds_d')
axes[0].set_title('Total Downtime per Machine')
axes[0].set_ylabel('Minutes')

sns.barplot(x=downtime_frequency_per_machine.index, y=downtime_frequency_per_machine.values, ax=axes[1], palette='Oranges_d')
axes[1].set_title('Downtime Frequency per Machine')
axes[1].set_ylabel('Occurrences')

plt.tight_layout()
plt.show()

## 7. Cycle Time Optimal Analysis

plt.figure(figsize=(10, 5))
sns.histplot(df['cycle_time'], kde=True, bins=30, color='skyblue')
plt.title('Distribusi Waktu Siklus (Cycle Time)')
plt.xlabel('Waktu Siklus (detik)')
plt.ylabel('Frekuensi')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

df['cycle_time_bin'] = pd.cut(df['cycle_time'], bins=5, precision=0)

cycle_time_analysis = df.groupby('cycle_time_bin').agg(
    avg_defect_count=('defect_count','mean'),
    avg_error_rate=('error_rate','mean'),
    total_units_produced=('units_produced','sum')
).reset_index()

print('Analisis Cycle Time vs Defect & Error Rate:')
print(cycle_time_analysis.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.barplot(x='cycle_time_bin', y='avg_defect_count', data=cycle_time_analysis, ax=axes[0], palette='viridis')
axes[0].set_title('Avg Defect Count per Cycle Time Bin')
axes[0].tick_params(axis='x', rotation=30)

sns.barplot(x='cycle_time_bin', y='avg_error_rate', data=cycle_time_analysis, ax=axes[1], palette='plasma')
axes[1].set_title('Avg Error Rate per Cycle Time Bin')
axes[1].tick_params(axis='x', rotation=30)

sns.barplot(x='cycle_time_bin', y='total_units_produced', data=cycle_time_analysis, ax=axes[2], palette='cividis')
axes[2].set_title('Total Units per Cycle Time Bin')
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()

## 8. Machine Condition Clustering

clustering_features = [
    'temperature','vibration_level','power_consumption','pressure',
    'oil_level','machine_age_hours','error_rate','noise_level_db'
]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[clustering_features])

df_scaled_df = pd.DataFrame(df_scaled, columns=clustering_features)
print('Scaled data preview:')
df_scaled_df.head()

### 8.1 Elbow Method

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(K_range, inertia, marker='o', color='steelblue')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(K_range)
plt.grid(True)
plt.show()

### 8.2 Silhouette & Davies-Bouldin Evaluation

scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    sil = silhouette_score(df_scaled, labels)
    db  = davies_bouldin_score(df_scaled, labels)
    scores.append((k, sil, db))

k_vals, sil_vals, db_vals = zip(*scores)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(k_vals, sil_vals, 'o-', color='green')
axes[0].set_title('Silhouette Score'); axes[0].set_xlabel('k'); axes[0].grid(True)

axes[1].plot(k_vals, db_vals, 'o-', color='red')
axes[1].set_title('Davies-Bouldin Index'); axes[1].set_xlabel('k'); axes[1].grid(True)

plt.tight_layout()
plt.show()

for k, sil, db in scores:
    print(f'k={k}: Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}')

### 8.3 Apply K-Means (k=3)

kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_efficiency_machine'] = kmeans_final.fit_predict(df_scaled)

cluster_centroids = df.groupby('cluster_efficiency_machine')[clustering_features].mean()
print('Cluster Centroids:')
print(cluster_centroids.round(2))

print('\nCluster Distribution:')
print(df['cluster_efficiency_machine'].value_counts().sort_index())

### 8.4 Visualisasi PCA

pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(pca_components, columns=['PCA1','PCA2'])
df_pca['cluster_efficiency_machine'] = df['cluster_efficiency_machine'].values

plt.figure(figsize=(9, 6))
palette = {0:'#e74c3c', 1:'#3498db', 2:'#2ecc71'}
labels_map = {
    0:'Cluster 0 (Degraded/Critical)',
    1:'Cluster 1 (Moderate/Low Oil)',
    2:'Cluster 2 (Older, Stable & Healthy)'
}
for cid, grp in df_pca.groupby('cluster_efficiency_machine'):
    plt.scatter(grp['PCA1'], grp['PCA2'], c=palette[cid],
                label=labels_map[cid], s=20, alpha=0.6)

plt.title('PCA of Machine Condition Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(fontsize=9)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

### Karakteristik Kluster Mesin


## 9. Shift Efficiency Clustering

df['date'] = df['timestamp'].dt.date
df['shift_instance_id'] = df['date'].astype(str) + '_' + df['shift']

grouped_data = df.groupby(['shift_instance_id','machine_id','line_id']).agg(
    units_produced=('units_produced','sum'),
    downtime=('downtime','sum'),
    error_rate=('error_rate','mean'),
    cycle_time=('cycle_time','mean')
).reset_index()

print(f'Jumlah shift instance unik: {grouped_data["shift_instance_id"].nunique()}')
grouped_data.head()

# Buat efficiency scores
grouped_data['efficiency_error_rate']     = 1 - grouped_data['error_rate']
grouped_data['efficiency_downtime']       = 1 / (1 + grouped_data['downtime'])
grouped_data['efficiency_cycle_time']     = 1 / (1 + grouped_data['cycle_time'])
grouped_data['efficiency_units_produced'] = grouped_data['units_produced']

eff_cols = ['efficiency_error_rate','efficiency_downtime',
            'efficiency_cycle_time','efficiency_units_produced']

scaler2 = StandardScaler()
grouped_data[eff_cols] = scaler2.fit_transform(grouped_data[eff_cols])

print('Efficiency scores (scaled) preview:')
grouped_data[eff_cols].head()

# Evaluasi optimal K
k_range_eval = range(2, 10)
sil_scores, db_scores = [], []

for k in k_range_eval:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    lbl = km.fit_predict(grouped_data[eff_cols])
    sil_scores.append(silhouette_score(grouped_data[eff_cols], lbl))
    db_scores.append(davies_bouldin_score(grouped_data[eff_cols], lbl))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(k_range_eval, sil_scores, 'o-', color='green'); axes[0].set_title('Silhouette Score — Shift Clustering'); axes[0].grid(True)
axes[1].plot(k_range_eval, db_scores, 'o-', color='red'); axes[1].set_title('Davies-Bouldin — Shift Clustering'); axes[1].grid(True)
plt.tight_layout()
plt.show()

# Apply optimal k=2
kmeans_shift = KMeans(n_clusters=2, random_state=42, n_init='auto')
grouped_data['shift_efficiency_cluster'] = kmeans_shift.fit_predict(grouped_data[eff_cols])

# Restore original scale for interpretasi
temp_orig = df.groupby(['shift_instance_id','machine_id','line_id']).agg(
    units_produced=('units_produced','sum'),
    downtime=('downtime','sum'),
    error_rate=('error_rate','mean'),
    cycle_time=('cycle_time','mean')
).reset_index()
temp_orig['shift_efficiency_cluster'] = grouped_data['shift_efficiency_cluster'].values

centroids_shift = temp_orig.groupby('shift_efficiency_cluster')[
    ['units_produced','downtime','error_rate','cycle_time']].mean()

print('Cluster Centroids (Original Values):')
print(centroids_shift.round(2))

# Merge back ke df utama
df = df.merge(
    grouped_data[['shift_instance_id','machine_id','line_id','shift_efficiency_cluster']],
    on=['shift_instance_id','machine_id','line_id'], how='left'
)
print('\nShift cluster merged to main DataFrame ✓')

## 10. Bottleneck & Line Analysis

total_dt_line  = df.groupby('line_id')['downtime'].sum()
freq_dt_line   = df.groupby('line_id')['downtime'].apply(lambda x: (x>0).sum())
avg_err_line   = df.groupby('line_id')['error_rate'].mean()
total_up_line  = df.groupby('line_id')['units_produced'].sum()

bottleneck_summary = pd.DataFrame({
    'Total Downtime (min)': total_dt_line,
    'Downtime Freq':        freq_dt_line,
    'Avg Error Rate':       avg_err_line.round(3),
    'Total Units':          total_up_line
})
print(bottleneck_summary)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

datasets = [total_dt_line, freq_dt_line, avg_err_line, total_up_line]
titles   = ['Total Downtime per Line','Downtime Frequency','Avg Error Rate','Total Units Produced']
ylabels  = ['Minutes','Count','Error Rate','Units']
palettes = ['viridis','magma','cividis','plasma']

for ax, data, title, ylabel, pal in zip(axes, datasets, titles, ylabels, palettes):
    sns.barplot(x=data.index, y=data.values, ax=ax, palette=pal)
    ax.set_title(title); ax.set_ylabel(ylabel)

plt.tight_layout()
plt.show()

## 11. Defect & Quality Analysis

total_units_produced = df['units_produced'].sum()
total_defect_count   = df['defect_count'].sum()
good_count           = total_units_produced - total_defect_count

good_pct   = (good_count / total_units_produced) * 100
defect_pct = (total_defect_count / total_units_produced) * 100

print(f'Total Units Produced  : {total_units_produced:,.0f}')
print(f'Good Units            : {good_count:,.0f} ({good_pct:.2f}%)')
print(f'Defective Units       : {total_defect_count:,.0f} ({defect_pct:.2f}%)')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Pie: Good vs Defective
axes[0].pie([good_count, total_defect_count], labels=['Good','Defective'],
            colors=['#66b3ff','#ff9999'], autopct='%1.1f%%',
            explode=(0.05,0), shadow=True, startangle=140)
axes[0].set_title('Good vs Defective Units')

# Bar: Defect per product type
dc_prod = df.groupby('product_type')['defect_count'].sum().sort_values(ascending=False)
axes[1].bar(dc_prod.index, dc_prod.values, color=sns.color_palette('viridis', len(dc_prod)))
axes[1].set_title('Total Defect Count per Product Type')
axes[1].set_ylabel('Total Defects')
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()

# Defect rate by machine type, machine ID, lubrication status, cluster
dr_machine_type  = df.groupby('machine_type')['defect_count'].mean().sort_values(ascending=False)
dr_machine_id    = df.groupby('machine_id')['defect_count'].mean().sort_values(ascending=False)
dr_lubr_status   = df.groupby('lubrication_status')['defect_count'].mean().sort_values(ascending=False)
dr_cluster       = df.groupby('cluster_efficiency_machine')['defect_count'].mean().sort_values(ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, data, title, pal in zip(
    axes,
    [dr_machine_type, dr_machine_id, dr_lubr_status, dr_cluster],
    ['Avg Defect by Machine Type','Avg Defect by Machine ID',
     'Avg Defect by Lubrication Status','Avg Defect by Machine Cluster'],
    ['viridis','magma','plasma','cividis']
):
    sns.barplot(x=data.index.astype(str), y=data.values, ax=ax, palette=pal)
    ax.set_title(title)
    ax.set_ylabel('Avg Defect Count')
    ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()

## 12. Temperature Impact Analysis

# Korelasi dengan temperature
corr_features = ['temperature','power_consumption','error_rate','downtime','units_produced']
print('Correlations with temperature:')
print(df[corr_features].corr()['temperature'].drop('temperature').round(3))

Q1_t = df['temperature'].quantile(0.25)
Q3_t = df['temperature'].quantile(0.75)

df_low    = df[df['temperature'] <= Q1_t]
df_normal = df[(df['temperature'] > Q1_t) & (df['temperature'] < Q3_t)]
df_high   = df[df['temperature'] >= Q3_t]

metrics = ['power_consumption','error_rate','downtime','units_produced']
summary = pd.DataFrame({
    f'Low (<={Q1_t:.1f}°C)':     df_low[metrics].mean(),
    f'Normal ({Q1_t:.1f}–{Q3_t:.1f}°C)': df_normal[metrics].mean(),
    f'High (>={Q3_t:.1f}°C)':    df_high[metrics].mean()
}).T.round(3)

print('Average Metrics per Temperature Scenario:')
print(summary)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Impact of Temperature Scenarios on Key Metrics', fontsize=13)
scenarios = ['Low Temp','Normal Temp','High Temp']
colors = ['#3498db','#f39c12','#e74c3c']

for i, metric in enumerate(metrics):
    means = [df_low[metric].mean(), df_normal[metric].mean(), df_high[metric].mean()]
    axes[i].bar(scenarios, means, color=colors)
    axes[i].set_title(metric.replace('_',' ').title())
    axes[i].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()

## 13. Operator Performance Analysis

error_per_operator = df.groupby('operator_id')['error_rate'].sum().sort_values(ascending=False)
print('Error Rate per Operator (tertinggi → terendah):')
print(error_per_operator.round(2))

# Defect percentage per operator
defect_count_per_operator  = df.groupby('operator_id')['defect_count'].sum()
units_per_operator         = df.groupby('operator_id')['units_produced'].sum()

operator_perf = pd.DataFrame({
    'total_units':   units_per_operator,
    'total_defects': defect_count_per_operator
})
operator_perf['defect_pct'] = (operator_perf['total_defects'] / operator_perf['total_units']).fillna(0) * 100

print('\nDefect Percentage per Operator:')
print(operator_perf.sort_values('defect_pct', ascending=False).round(2))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x=error_per_operator.index, y=error_per_operator.values, ax=axes[0], palette='rocket')
axes[0].set_title('Total Error Rate per Operator')
axes[0].tick_params(axis='x', rotation=45)

sorted_op = operator_perf.sort_values('defect_pct', ascending=False)
axes[1].bar(sorted_op.index, sorted_op['defect_pct'], color=sns.color_palette('magma', len(sorted_op)))
axes[1].set_title('Defect % per Operator')
axes[1].set_ylabel('%')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

## 14. Predictive Modeling — Defect Classification


df['has_defect'] = (df['defect_count'] > 0).astype(int)
print(f'Class distribution:\n{df["has_defect"].value_counts()}')
print(f'\nClass ratio: {df["has_defect"].mean():.2%} positive')

feature_cols = [
    'temperature','vibration_level','power_consumption','pressure',
    'material_flow_rate','cycle_time','error_rate','machine_age_hours',
    'last_maintenance_hours','oil_level','downtime','ambient_temperature',
    'humidity','noise_level_db','hour','day_of_week','is_weekend',
    'cluster_efficiency_machine'
]
if 'shift_efficiency_cluster' in df.columns:
    feature_cols.append('shift_efficiency_cluster')

X = df[feature_cols].dropna()
y = df.loc[X.index, 'has_defect']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler_m = StandardScaler()
X_train_s = scaler_m.fit_transform(X_train)
X_test_s  = scaler_m.transform(X_test)

print(f'Train: {X_train.shape}, Test: {X_test.shape}')

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42,
                             n_jobs=-1, class_weight='balanced')
rf.fit(X_train_s, y_train)
y_pred_rf = rf.predict(X_test_s)
y_prob_rf = rf.predict_proba(X_test_s)[:,1]

print('=== Random Forest ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}')
print(f'F1 Score : {f1_score(y_test, y_pred_rf):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, y_prob_rf):.4f}')
print()
print(classification_report(y_test, y_pred_rf, target_names=['No Defect','Defect']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Defect','Defect'],
            yticklabels=['No Defect','Defect'])
ax.set_title('Confusion Matrix — Random Forest')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()

# Feature Importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(9, 5))
sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
plt.title('Top 15 Feature Importances — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# XGBoost (jika tersedia)
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=100, random_state=42, eval_metric='logloss',
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
    )
    xgb.fit(X_train_s, y_train)
    y_pred_xgb = xgb.predict(X_test_s)
    y_prob_xgb = xgb.predict_proba(X_test_s)[:,1]

    print('=== XGBoost ===')
    print(f'Accuracy : {accuracy_score(y_test, y_pred_xgb):.4f}')
    print(f'F1 Score : {f1_score(y_test, y_pred_xgb):.4f}')
    print(f'ROC-AUC  : {roc_auc_score(y_test, y_prob_xgb):.4f}')
    print()
    print(classification_report(y_test, y_pred_xgb, target_names=['No Defect','Defect']))
except ImportError:
    print('XGBoost tidak tersedia. Install: pip install xgboost')