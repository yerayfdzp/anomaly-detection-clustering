# Clustering-Based Anomaly Detection in Industrial Sensor Data

An unsupervised machine learning system for detecting anomalous patterns in industrial sensor data using K-Means, DBSCAN, and Gaussian Mixture Models (GMM).

## Overview

Industrial equipment is continuously monitored through sensors measuring temperature, pressure, vibration, rotational speed, and torque. This project applies unsupervised clustering algorithms to identify abnormal operating conditions without requiring labeled training data, enabling proactive maintenance strategies.

## Dataset

- **Type:** Synthetic industrial sensor data (2,000 records, 5 features)
- **Normal samples:** 1,800 (90%)
- **Anomalous samples:** 200 (10%) across 3 failure modes:
  - Overheating (high temperature + high torque)
  - Excessive vibration (high vibration + RPM deviation)
  - Pressure drop (low pressure + temperature drop)

## Algorithms Compared

| Algorithm | Approach | Anomaly Detection Strategy |
|-----------|----------|---------------------------|
| **K-Means** | Centroid-based partitioning | Minority cluster = anomaly |
| **DBSCAN** | Density-based clustering | Noise points = anomaly |
| **GMM** | Probabilistic mixture model | Low log-likelihood = anomaly |

## Key Results

| Algorithm | Silhouette | Davies-Bouldin | F1-Score (Anomaly) |
|-----------|-----------|----------------|-------------------|
| K-Means | 0.539 | 1.353 | **0.793** |
| DBSCAN | 0.218 | 1.596 | 0.388 |
| GMM | 0.534 | **0.683** | 0.526 |

- **K-Means** achieves the best binary anomaly detection (F1: 0.793, Precision: 93.2%)
- **GMM** achieves the best clustering quality (lowest Davies-Bouldin: 0.683) and provides probabilistic anomaly scoring
- **DBSCAN** achieves perfect recall (100%) but with low precision due to over-classification as noise

## Key Steps

1. **Synthetic Data Generation** — realistic sensor data with 3 distinct failure modes
2. **Exploratory Data Analysis** — distributions, correlation heatmap, pair plots
3. **Preprocessing** — StandardScaler normalization, PCA for visualization (61.2% variance retained)
4. **Clustering** — K-Means (elbow + silhouette), DBSCAN (k-distance), GMM (BIC/AIC)
5. **Anomaly Detection** — binary evaluation against ground truth
6. **Comparative Analysis** — clustering quality metrics + anomaly detection performance

## How to Run

```bash
# Clone the repository
git clone https://github.com/yerayfdzp/anomaly-detection-clustering.git
cd anomaly-detection-clustering

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python anomaly_detection.py
```

## Docker

Build and run the project using Docker:

```bash
docker build -t anomaly-detection .
docker run anomaly-detection
```

## Tech Stack

- Python 3
- scikit-learn (K-Means, DBSCAN, GaussianMixture, PCA)
- pandas & NumPy
- matplotlib & seaborn
- Docker
