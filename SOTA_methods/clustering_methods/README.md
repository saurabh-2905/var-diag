# Anomaly Detection using Clustering (KNN) Methods

This repository provides a comprehensive implementation for anomaly detection in time series data using **clustering-based techniques** (specifically, K-Nearest Neighbors). It primarily focuses on the below scenario:

- **EventID + Timestamp Difference Model**: Anomaly detection using sequences of event IDs and corresponding timestamp differences as combined features.

Significant accuracy variations were observed, with the **ID model outperforming the ID-Timestamp model** in most cases.

---

## 📊 Features

- **Data Loading & Preprocessing**:
  - Handles trace data consisting of event IDs and timestamps.
  - Extracts sequential patterns (ID sequences, timestamp differences) and performs feature extraction using `seglearn`.
  - Normalization with `StandardScaler` ensures compatibility with clustering algorithms.

- **Feature Engineering**:
  - Extracts a wide range of time series features (mean, variance, skewness, etc.) from traces using `seglearn`.

- **KNN-based Anomaly Detection**:
  - KNN is fitted on training feature vectors.
  - During testing, the model checks if a test point's nearest neighbors belong to a single cluster. If not, the point is flagged as an anomaly.

- **Threshold-free** detection:
  - No manual threshold setting; anomaly detection is purely based on neighbor cluster consistency.

- **Evaluation Metrics**:
  - Precision, Recall, F1 Score, and Confusion Matrix visualization included.

- **Inference Time Reporting**:
  - Provides time taken per test sample for anomaly detection.

---

## 📦 Prerequisites

Ensure the following libraries are installed:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Seglearn
- Matplotlib

Install required packages using:

```bash
pip install numpy pandas scikit-learn seglearn matplotlib
```

## 📂 Repository Structure
```bash
clustering_methods/
├── libraries/
│   ├── anomaly_detection.py         # KNN-based anomaly detection functions.
│   └── utils.py                     # Helper functions (read_trace, load_data, etc.)
│
├── scalers/
│   └── scaler.pkl                   # Pre-fitted scaler for feature normalization.
│
├── trace_data/                      # Raw trace data directory.
├── train_data_processed/            # Processed trace data directory.
│
├── trained_model/
│   ├── kmeans_model.pkl             # Saved KMeans model.
│   ├── train_clusters.pkl           # Cluster labels for training data.
│   ├── train_features.pkl           # Training features used for clustering.
│
├── clustered_features_seglearn.csv  # Feature CSV for training data.
├── clustering_methods.ipynb         # Main notebook for training and evaluation.
└── README.md                        
```

## 🚀 How to run
- *Clone the repository*
    1. Copy the gitlink repo https link and open git bash in local folder.
    2. Run the command 'git clone 'copied link'' and it will fetch all the files from the repository to the local folder.
```bash
    git clone https://github.com/user_id/Anomaly_Detection.git
    cd Anomaly_Detection
```
- *Preparing the data*
    1. Place the zipped trace file inside the project directory.
    2. Extract the data and structure it as per the required formats.

- *Now run the notebooks*
    1. Open the Jupyter Notebooks:
```bash
    jupyter notebook clustering_methods.ipynb
```
    2. Also, we can directly open the notebook in a VS code.
    3. Once, the notebook is opened the runs the cells sequentially to Preprocess the trace data, extract features using Seglearn, train the KNN model and generate cluster labels, and evaluate the model and detect anomalies.


- *Saving the trained_model and scalers*
    1. The following files will be saved in the trained_model and scalers directories for later use.
        1. kmeans_model.pkl
        2. train_clusters.pkl
        3. train_features.pkl
        4. scaler.pkl


## 📈 Evaluate the Model
The following metrics will be computed and visualized.
1. Precision
2. Recall
3. F1 score
4. Confusion matrix
5. Inference time is also included to have an idea about the time taken.

## Observations
1. The event_id + timestamp difference model gives robust anomaly detection when combined with KNN and feature extraction techniques.


