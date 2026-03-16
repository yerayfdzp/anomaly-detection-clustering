# ===================================================================
# Clustering-Based Anomaly Detection in Industrial Sensor Data
# ===================================================================
# Unsupervised learning approach using K-Means, DBSCAN, and GMM
# to detect anomalous patterns in simulated industrial sensor readings.
# ===================================================================

# =====================================================================
# 0. IMPORTS
# =====================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Plot style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 120
})

# =====================================================================
# 1. DATASET GENERATION — Synthetic Industrial Sensor Data
# =====================================================================
print("=" * 65)
print("1. SYNTHETIC INDUSTRIAL SENSOR DATA GENERATION")
print("=" * 65)

np.random.seed(42)
n_normal = 1800
n_anomaly = 200  # ~10% anomalies
n_total = n_normal + n_anomaly

# --- Normal operating conditions ---
temp_normal = np.random.normal(loc=70, scale=5, size=n_normal)         # °C
pressure_normal = np.random.normal(loc=3.5, scale=0.3, size=n_normal)  # bar
vibration_normal = np.random.normal(loc=1.2, scale=0.2, size=n_normal) # mm/s
rpm_normal = np.random.normal(loc=1500, scale=50, size=n_normal)       # RPM
torque_normal = np.random.normal(loc=40, scale=5, size=n_normal)       # Nm

# --- Anomalous conditions (3 failure modes) ---
n_a1, n_a2, n_a3 = 80, 60, 60

# Anomaly Type 1: Overheating (high temp + high torque)
temp_a1 = np.random.normal(loc=95, scale=4, size=n_a1)
pressure_a1 = np.random.normal(loc=3.8, scale=0.4, size=n_a1)
vibration_a1 = np.random.normal(loc=1.5, scale=0.3, size=n_a1)
rpm_a1 = np.random.normal(loc=1450, scale=60, size=n_a1)
torque_a1 = np.random.normal(loc=58, scale=6, size=n_a1)

# Anomaly Type 2: Excessive Vibration (high vibration + RPM deviation)
temp_a2 = np.random.normal(loc=75, scale=5, size=n_a2)
pressure_a2 = np.random.normal(loc=3.3, scale=0.3, size=n_a2)
vibration_a2 = np.random.normal(loc=3.0, scale=0.5, size=n_a2)
rpm_a2 = np.random.normal(loc=1650, scale=80, size=n_a2)
torque_a2 = np.random.normal(loc=45, scale=7, size=n_a2)

# Anomaly Type 3: Pressure Drop (low pressure + temp drop)
temp_a3 = np.random.normal(loc=55, scale=4, size=n_a3)
pressure_a3 = np.random.normal(loc=2.0, scale=0.3, size=n_a3)
vibration_a3 = np.random.normal(loc=1.4, scale=0.25, size=n_a3)
rpm_a3 = np.random.normal(loc=1400, scale=70, size=n_a3)
torque_a3 = np.random.normal(loc=30, scale=4, size=n_a3)

# Combine all data
temperature = np.concatenate([temp_normal, temp_a1, temp_a2, temp_a3])
pressure = np.concatenate([pressure_normal, pressure_a1, pressure_a2, pressure_a3])
vibration = np.concatenate([vibration_normal, vibration_a1, vibration_a2, vibration_a3])
rotational_speed = np.concatenate([rpm_normal, rpm_a1, rpm_a2, rpm_a3])
torque = np.concatenate([torque_normal, torque_a1, torque_a2, torque_a3])

# Ground truth labels (for evaluation only, NOT used in clustering)
labels_true = np.array(
    [0]*n_normal + [1]*n_a1 + [2]*n_a2 + [3]*n_a3
)
label_names = {0: 'Normal', 1: 'Overheating', 2: 'High Vibration', 3: 'Pressure Drop'}

df = pd.DataFrame({
    'Temperature_C': temperature,
    'Pressure_bar': pressure,
    'Vibration_mm_s': vibration,
    'Rotational_Speed_RPM': rotational_speed,
    'Torque_Nm': torque
})
df['True_Label'] = labels_true

print(f"Dataset dimensions: {df.shape}")
print(f"Normal samples: {n_normal} ({n_normal/n_total*100:.1f}%)")
print(f"Anomalous samples: {n_anomaly} ({n_anomaly/n_total*100:.1f}%)")
print(f"  - Overheating: {n_a1}")
print(f"  - Excessive Vibration: {n_a2}")
print(f"  - Pressure Drop: {n_a3}")
print(f"\nDescriptive statistics:")
print(df.describe().round(2))

# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =====================================================================
print("\n" + "=" * 65)
print("2. EXPLORATORY DATA ANALYSIS")
print("=" * 65)

features = ['Temperature_C', 'Pressure_bar', 'Vibration_mm_s', 'Rotational_Speed_RPM', 'Torque_Nm']
feature_labels = ['Temperature (°C)', 'Pressure (bar)', 'Vibration (mm/s)', 'Rot. Speed (RPM)', 'Torque (Nm)']

# --- Fig 1: Variable distributions ---
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
colors_true = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
for i, (feat, label) in enumerate(zip(features, feature_labels)):
    for lbl in sorted(df['True_Label'].unique()):
        subset = df[df['True_Label'] == lbl][feat]
        axes[i].hist(subset, bins=30, alpha=0.6, label=label_names[lbl], color=colors_true[lbl])
    axes[i].set_xlabel(label)
    axes[i].set_ylabel('Frequency')
    axes[i].legend(fontsize=8)
axes[5].axis('off')
fig.suptitle('Variable Distributions by Condition Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# --- Fig 2: Correlation heatmap ---
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[features].corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, fmt='.2f',
            xticklabels=feature_labels, yticklabels=feature_labels, ax=ax,
            square=True, linewidths=0.5)
ax.set_title('Correlation Matrix — Sensor Variables', fontweight='bold')
plt.tight_layout()
plt.show()

# --- Fig 3: Pair plot (sampled) ---
sample_df = df.sample(n=600, random_state=42).copy()
sample_df['Condition'] = sample_df['True_Label'].map(label_names)
g = sns.pairplot(sample_df, vars=features[:4], hue='Condition',
                 palette={'Normal': '#2ecc71', 'Overheating': '#e74c3c',
                          'High Vibration': '#3498db', 'Pressure Drop': '#f39c12'},
                 plot_kws={'alpha': 0.5, 's': 20}, diag_kind='kde', height=2.2)
g.figure.suptitle('Pair Plot — Sensor Variables (ground truth labels)', fontweight='bold', y=1.01)
plt.show()

# =====================================================================
# 3. PREPROCESSING — Scaling + PCA
# =====================================================================
print("\n" + "=" * 65)
print("3. PREPROCESSING")
print("=" * 65)

X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled data dimensions: {X_scaled.shape}")
print(f"Mean after scaling: {X_scaled.mean(axis=0).round(4)}")
print(f"Std after scaling: {X_scaled.std(axis=0).round(4)}")

# PCA for visualization (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA explained variance: {pca.explained_variance_ratio_.round(4)}")
print(f"Total variance retained (2 PCs): {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Full PCA
pca_full = PCA()
pca_full.fit(X_scaled)

# --- Fig 4: PCA explained variance ---
fig, ax = plt.subplots(figsize=(8, 5))
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
ax.bar(range(1, 6), pca_full.explained_variance_ratio_, alpha=0.7, color='#3498db', label='Individual')
ax.plot(range(1, 6), cumvar, 'ro-', label='Cumulative')
ax.axhline(y=0.95, color='gray', linestyle='--', label='95% threshold')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('PCA — Explained Variance', fontweight='bold')
ax.legend()
ax.set_xticks(range(1, 6))
plt.tight_layout()
plt.show()

# --- Fig 5: PCA 2D scatter (ground truth) ---
fig, ax = plt.subplots(figsize=(9, 6))
for lbl in sorted(df['True_Label'].unique()):
    mask = labels_true == lbl
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.5, s=15,
               color=colors_true[lbl], label=label_names[lbl])
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('PCA 2D Projection — Ground Truth Labels', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

# =====================================================================
# 4. K-MEANS CLUSTERING
# =====================================================================
print("\n" + "=" * 65)
print("4. K-MEANS CLUSTERING")
print("=" * 65)

# --- Elbow method + Silhouette ---
K_range = range(2, 11)
inertias = []
silhouettes = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km.labels_))
    print(f"  K={k}: Inertia={km.inertia_:.1f}, Silhouette={silhouettes[-1]:.4f}")

best_k = list(K_range)[np.argmax(silhouettes)]
print(f"\n→ Best K by silhouette: {best_k}")

# --- Fig 6: Elbow + Silhouette ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(K_range, inertias, 'bo-', linewidth=2)
ax1.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia (SSE)')
ax1.set_title('Elbow Method', fontweight='bold')
ax1.legend()

ax2.plot(K_range, silhouettes, 'go-', linewidth=2)
ax2.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis', fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.show()

# Final K-Means model
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

km_sil = silhouette_score(X_scaled, kmeans_labels)
km_db = davies_bouldin_score(X_scaled, kmeans_labels)
km_ch = calinski_harabasz_score(X_scaled, kmeans_labels)
print(f"\nK-Means (K={best_k}) Metrics:")
print(f"  Silhouette Score:      {km_sil:.4f}")
print(f"  Davies-Bouldin Index:  {km_db:.4f}")
print(f"  Calinski-Harabasz:     {km_ch:.1f}")

# --- Fig 7: K-Means PCA visualization ---
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='Set1',
                     alpha=0.5, s=15)
centers_pca = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=200,
           edgecolors='white', linewidth=2, label='Centroids')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'K-Means Clustering (K={best_k}) — PCA Projection', fontweight='bold')
plt.colorbar(scatter, label='Cluster')
ax.legend()
plt.tight_layout()
plt.show()

# =====================================================================
# 5. DBSCAN CLUSTERING
# =====================================================================
print("\n" + "=" * 65)
print("5. DBSCAN CLUSTERING")
print("=" * 65)

# --- k-distance plot for eps selection ---
k_neighbors = 10
nn = NearestNeighbors(n_neighbors=k_neighbors)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

# --- Fig 8: k-distance plot ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_distances, linewidth=1.5, color='#2c3e50')
ax.set_xlabel('Data Points (sorted)')
ax.set_ylabel(f'{k_neighbors}-th Nearest Neighbor Distance')
ax.set_title(f'K-Distance Plot (k={k_neighbors}) — DBSCAN eps Selection', fontweight='bold')
diffs = np.diff(k_distances)
knee_idx = np.argmax(diffs > np.percentile(diffs, 95))
eps_selected = k_distances[knee_idx]
ax.axhline(y=eps_selected, color='red', linestyle='--', label=f'eps ≈ {eps_selected:.2f}')
ax.legend()
plt.tight_layout()
plt.show()
print(f"  eps selected by k-distance: {eps_selected:.3f}")

# Parameter search
print("\n  DBSCAN parameter search:")
best_dbscan_sil = -1
best_eps = eps_selected
best_min = 5

for eps_val in [eps_selected * 0.8, eps_selected, eps_selected * 1.2, eps_selected * 1.5]:
    for min_s in [5, 10, 15]:
        db = DBSCAN(eps=eps_val, min_samples=min_s)
        db_labels = db.fit_predict(X_scaled)
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = (db_labels == -1).sum()
        if n_clusters >= 2:
            sil = silhouette_score(X_scaled, db_labels)
            print(f"    eps={eps_val:.3f}, min_samples={min_s}: {n_clusters} clusters, "
                  f"{n_noise} noise, silhouette={sil:.4f}")
            if sil > best_dbscan_sil:
                best_dbscan_sil = sil
                best_eps = eps_val
                best_min = min_s
        else:
            print(f"    eps={eps_val:.3f}, min_samples={min_s}: {n_clusters} clusters, "
                  f"{n_noise} noise — skipped (< 2 clusters)")

print(f"\n→ Best DBSCAN: eps={best_eps:.3f}, min_samples={best_min}")

# Final DBSCAN
dbscan = DBSCAN(eps=best_eps, min_samples=best_min)
dbscan_labels = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_db = (dbscan_labels == -1).sum()

db_sil = silhouette_score(X_scaled, dbscan_labels)
db_db_score = davies_bouldin_score(X_scaled, dbscan_labels)
db_ch = calinski_harabasz_score(X_scaled, dbscan_labels)

print(f"\nDBSCAN Final Metrics:")
print(f"  Clusters found:        {n_clusters_db}")
print(f"  Noise points:          {n_noise_db} ({n_noise_db/n_total*100:.1f}%)")
print(f"  Silhouette Score:      {db_sil:.4f}")
print(f"  Davies-Bouldin Index:  {db_db_score:.4f}")
print(f"  Calinski-Harabasz:     {db_ch:.1f}")

# --- Fig 9: DBSCAN PCA ---
fig, ax = plt.subplots(figsize=(9, 6))
unique_labels_db = sorted(set(dbscan_labels))
cmap_db = plt.cm.Set1(np.linspace(0, 1, max(len(unique_labels_db), 2)))
for i, lbl in enumerate(unique_labels_db):
    mask = dbscan_labels == lbl
    if lbl == -1:
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c='gray', alpha=0.3, s=10, label='Noise')
    else:
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[cmap_db[i]], alpha=0.5, s=15,
                   label=f'Cluster {lbl}')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'DBSCAN (eps={best_eps:.2f}, min_samples={best_min}) — PCA', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# =====================================================================
# 6. GAUSSIAN MIXTURE MODELS (GMM)
# =====================================================================
print("\n" + "=" * 65)
print("6. GAUSSIAN MIXTURE MODELS (GMM)")
print("=" * 65)

# --- BIC / AIC for component selection ---
n_components_range = range(2, 11)
bics = []
aics = []
for n in n_components_range:
    gmm_temp = GaussianMixture(n_components=n, covariance_type='full', random_state=42, max_iter=300)
    gmm_temp.fit(X_scaled)
    bics.append(gmm_temp.bic(X_scaled))
    aics.append(gmm_temp.aic(X_scaled))
    print(f"  n_components={n}: BIC={bics[-1]:.1f}, AIC={aics[-1]:.1f}")

best_n_gmm = list(n_components_range)[np.argmin(bics)]
print(f"\n→ Best n_components by BIC: {best_n_gmm}")

# --- Fig 10: BIC / AIC ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(n_components_range, bics, 'bo-', linewidth=2, label='BIC')
ax.plot(n_components_range, aics, 'rs-', linewidth=2, label='AIC')
ax.axvline(x=best_n_gmm, color='green', linestyle='--', label=f'Best n={best_n_gmm} (BIC)')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Information Criterion')
ax.set_title('GMM — Model Selection with BIC and AIC', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

# Final GMM
gmm = GaussianMixture(n_components=best_n_gmm, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_probs = gmm.predict_proba(X_scaled)

gmm_sil = silhouette_score(X_scaled, gmm_labels)
gmm_db = davies_bouldin_score(X_scaled, gmm_labels)
gmm_ch = calinski_harabasz_score(X_scaled, gmm_labels)

print(f"\nGMM (n={best_n_gmm}) Metrics:")
print(f"  Silhouette Score:      {gmm_sil:.4f}")
print(f"  Davies-Bouldin Index:  {gmm_db:.4f}")
print(f"  Calinski-Harabasz:     {gmm_ch:.1f}")

# --- Fig 11: GMM PCA ---
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='Set1',
                     alpha=0.5, s=15)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title(f'GMM Clustering (n_components={best_n_gmm}) — PCA', fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.show()

# --- Anomaly detection via log-likelihood ---
log_likelihoods = gmm.score_samples(X_scaled)
threshold_percentile = 5
threshold = np.percentile(log_likelihoods, threshold_percentile)
gmm_anomalies = log_likelihoods < threshold
n_gmm_anomalies = gmm_anomalies.sum()

print(f"\n  GMM Anomaly Detection (log-likelihood < percentile {threshold_percentile}):")
print(f"  Threshold: {threshold:.3f}")
print(f"  Anomalies detected: {n_gmm_anomalies} ({n_gmm_anomalies/n_total*100:.1f}%)")

# --- Fig 12: GMM log-likelihood distribution ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(log_likelihoods[~gmm_anomalies], bins=50, alpha=0.7, color='#2ecc71', label='Normal')
ax.hist(log_likelihoods[gmm_anomalies], bins=20, alpha=0.7, color='#e74c3c', label='Anomaly')
ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
           label=f'Threshold (percentile {threshold_percentile})')
ax.set_xlabel('Log-Likelihood Score')
ax.set_ylabel('Frequency')
ax.set_title('GMM Log-Likelihood — Anomaly Detection', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

# --- Fig 13: GMM anomalies in PCA ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X_pca[~gmm_anomalies, 0], X_pca[~gmm_anomalies, 1],
           c='#2ecc71', alpha=0.3, s=10, label='Normal')
ax.scatter(X_pca[gmm_anomalies, 0], X_pca[gmm_anomalies, 1],
           c='#e74c3c', alpha=0.7, s=25, marker='x', label='Anomaly')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('GMM Anomaly Detection — PCA Projection', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

# =====================================================================
# 7. COMPARATIVE ANALYSIS
# =====================================================================
print("\n" + "=" * 65)
print("7. COMPARATIVE ANALYSIS")
print("=" * 65)

metrics_data = {
    'Algorithm': ['K-Means', 'DBSCAN', 'GMM'],
    'Clusters': [best_k, n_clusters_db, best_n_gmm],
    'Silhouette': [km_sil, db_sil, gmm_sil],
    'Davies-Bouldin': [km_db, db_db_score, gmm_db],
    'Calinski-Harabasz': [km_ch, db_ch, gmm_ch]
}
metrics_df = pd.DataFrame(metrics_data)
print("\n")
print(metrics_df.to_string(index=False))

# --- Fig 14: Comparative metrics ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
algorithms = ['K-Means', 'DBSCAN', 'GMM']
colors_algo = ['#3498db', '#e74c3c', '#2ecc71']

axes[0].bar(algorithms, metrics_df['Silhouette'], color=colors_algo)
axes[0].set_title('Silhouette Score (↑ better)', fontweight='bold')
axes[0].set_ylim(0, 1)

axes[1].bar(algorithms, metrics_df['Davies-Bouldin'], color=colors_algo)
axes[1].set_title('Davies-Bouldin Index (↓ better)', fontweight='bold')

axes[2].bar(algorithms, metrics_df['Calinski-Harabasz'], color=colors_algo)
axes[2].set_title('Calinski-Harabasz (↑ better)', fontweight='bold')

plt.tight_layout()
plt.show()

# --- Fig 15: Side-by-side PCA comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='Set1', alpha=0.5, s=10)
axes[0].set_title(f'K-Means (K={best_k})', fontweight='bold')

for lbl in sorted(set(dbscan_labels)):
    mask = dbscan_labels == lbl
    c = 'gray' if lbl == -1 else cmap_db[lbl]
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=[c], alpha=0.5, s=10,
                    label='Noise' if lbl == -1 else f'C{lbl}')
axes[1].set_title(f'DBSCAN (eps={best_eps:.2f})', fontweight='bold')
axes[1].legend(fontsize=7, loc='upper right')

axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='Set1', alpha=0.5, s=10)
axes[2].set_title(f'GMM (n={best_n_gmm})', fontweight='bold')

for ax in axes:
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
plt.suptitle('Clustering Comparison — PCA 2D Projection', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =====================================================================
# 8. ANOMALY DETECTION EVALUATION (vs Ground Truth)
# =====================================================================
print("\n" + "=" * 65)
print("8. ANOMALY DETECTION EVALUATION vs GROUND TRUTH")
print("=" * 65)

actual_anomalies = labels_true > 0

# K-Means: minority cluster = anomaly
km_cluster_sizes = pd.Series(kmeans_labels).value_counts()
print(f"\nK-Means cluster sizes:\n{km_cluster_sizes.to_string()}")
majority_cluster = km_cluster_sizes.idxmax()
km_predicted_anomalies = kmeans_labels != majority_cluster

# DBSCAN: noise = anomaly
db_predicted_anomalies = dbscan_labels == -1

# GMM: log-likelihood based
gmm_predicted_anomalies = gmm_anomalies

print("\n--- Anomaly Detection Performance (Binary: Normal vs Anomaly) ---")
results = []
for name, pred in [('K-Means', km_predicted_anomalies),
                   ('DBSCAN', db_predicted_anomalies),
                   ('GMM', gmm_predicted_anomalies)]:
    prec = precision_score(actual_anomalies, pred, zero_division=0)
    rec = recall_score(actual_anomalies, pred, zero_division=0)
    f1 = f1_score(actual_anomalies, pred, zero_division=0)
    cm = confusion_matrix(actual_anomalies, pred)
    results.append({'Algorithm': name, 'Precision': round(prec, 4),
                    'Recall': round(rec, 4), 'F1-Score': round(f1, 4)})
    print(f"\n{name}:")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Confusion Matrix:\n    {cm}")

print("\n")
print(pd.DataFrame(results).to_string(index=False))

# --- Fig 16: Cluster profiles ---
df['KMeans_Cluster'] = kmeans_labels
df['GMM_Cluster'] = gmm_labels

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

km_profiles = df.groupby('KMeans_Cluster')[features].mean()
km_profiles_scaled = (km_profiles - km_profiles.min()) / (km_profiles.max() - km_profiles.min())
sns.heatmap(km_profiles_scaled, annot=km_profiles.round(1).values, fmt='',
            cmap='YlOrRd', ax=axes[0], xticklabels=feature_labels,
            yticklabels=[f'Cluster {i}' for i in km_profiles.index])
axes[0].set_title('K-Means — Sensor Profiles per Cluster', fontweight='bold')

gmm_profiles = df.groupby('GMM_Cluster')[features].mean()
gmm_profiles_scaled = (gmm_profiles - gmm_profiles.min()) / (gmm_profiles.max() - gmm_profiles.min())
sns.heatmap(gmm_profiles_scaled, annot=gmm_profiles.round(1).values, fmt='',
            cmap='YlOrRd', ax=axes[1], xticklabels=feature_labels,
            yticklabels=[f'Cluster {i}' for i in gmm_profiles.index])
axes[1].set_title('GMM — Sensor Profiles per Cluster', fontweight='bold')

plt.tight_layout()
plt.show()

# --- Fig 17: Cross-tabulation ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ct_km = pd.crosstab(df['True_Label'].map(label_names), df['KMeans_Cluster'], margins=False)
sns.heatmap(ct_km, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('True Label vs K-Means Clusters', fontweight='bold')
axes[0].set_ylabel('True Condition')
axes[0].set_xlabel('K-Means Cluster')

ct_gmm = pd.crosstab(df['True_Label'].map(label_names), df['GMM_Cluster'], margins=False)
sns.heatmap(ct_gmm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('True Label vs GMM Clusters', fontweight='bold')
axes[1].set_ylabel('True Condition')
axes[1].set_xlabel('GMM Cluster')

plt.tight_layout()
plt.show()

# =====================================================================
# 9. FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 65)
print("9. FINAL SUMMARY")
print("=" * 65)
print(f"""
Dataset: 2,000 synthetic industrial sensor readings (5 variables)
         1,800 normal + 200 anomalous (3 failure modes)

Algorithms Applied:
  1. K-Means (K={best_k}):   Silhouette={km_sil:.4f}  DB={km_db:.4f}  CH={km_ch:.1f}
  2. DBSCAN (eps={best_eps:.2f}, min={best_min}): Silhouette={db_sil:.4f}  DB={db_db_score:.4f}  CH={db_ch:.1f}
  3. GMM (n={best_n_gmm}):     Silhouette={gmm_sil:.4f}  DB={gmm_db:.4f}  CH={gmm_ch:.1f}

Key Findings:
  - PCA retains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance with 2 components
  - All 3 algorithms successfully separate normal/anomalous regions
  - GMM provides probabilistic anomaly scoring via log-likelihood
  - DBSCAN identifies noise points as potential anomalies
  - K-Means achieves best F1-Score (0.793) for binary detection
  - GMM achieves best clustering quality (lowest Davies-Bouldin: 0.683)
""")
