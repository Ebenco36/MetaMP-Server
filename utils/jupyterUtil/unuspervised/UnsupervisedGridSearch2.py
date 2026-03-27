import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, Birch, MeanShift, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from jupyterUtil.GetFilePathIntoDict import getFiles

# Define your hyperparameter grids
kmeans_param_grid = {
    "n_clusters": range(2, 6),
    "n_init": [10],
}

dbscan_param_grid = {
    "eps": [0.3, 0.4, 0.5],
    "min_samples": [5, 10, 15, 20],
}

birch_param_grid = {
    "branching_factor": [50, 100, 200],
    "threshold": [0.1, 0.2, 0.3],
}

mean_shift_param_grid = {}

mini_batch_kmeans_param_grid = {
    "n_clusters": range(2, 6),
    "n_init": [10],
}

agg_clustering_param_grid = {
    "n_clusters": range(2, 6),
}

gmm_param_grid = {
    "n_components": range(2, 6),
}

optics_param_grid = {}

# Create a dictionary to map clustering algorithm names to their respective parameter grids
clustering_algorithms = {
    "K-Means": (KMeans(), kmeans_param_grid),
    "DBSCAN": (DBSCAN(), dbscan_param_grid),
    "BIRCH": (Birch(), birch_param_grid),
    "Mean Shift": (MeanShift(), mean_shift_param_grid),
    "MiniBatchKMeans": (MiniBatchKMeans(), mini_batch_kmeans_param_grid),
    "Agglomerative Clustering": (AgglomerativeClustering(), agg_clustering_param_grid),
    "Gaussian Mixture Models": (GaussianMixture(), gmm_param_grid),
    "OPTICS": (OPTICS(), optics_param_grid),
}

def preprocess_data(data):
    """Preprocess the input data (e.g., scaling)"""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# Create a directory to save results
os.makedirs("clustering_results", exist_ok=True)

# Evaluate cluster quality using Silhouette Score
def evaluate_cluster_quality(model, data_scaled):
    labels = model.fit_predict(data_scaled)
    if (len(set(labels)) > 1):
      silhouette = silhouette_score(data_scaled, labels)
      return silhouette
    else:
      return 0

data_paths = getFiles()

# Loop over your datasets
for data_path in data_paths:
    # Load or generate your data here
    data = pd.read_csv(data_paths[data_path])
    
    # Preprocess the data if needed (scaling, etc.)
    data_scaled = data[data.columns[1:].to_list()]
    for algorithm_name, (algorithm, param_grid) in clustering_algorithms.items():
        # Create a pipeline with the clustering algorithm
        model = GridSearchCV(algorithm, param_grid, cv=5, scoring=evaluate_cluster_quality)
        model.fit(data_scaled)
        
        # Evaluate cluster quality for the best model
        best_model = model.best_estimator_
        silhouette_score = evaluate_cluster_quality(best_model, data_scaled)
        
        results = {
            "Dataset": data_path,
            "Algorithm": algorithm_name,
            "Best Parameters": model.best_params_,
            "Silhouette Score": silhouette_score,
        }
    
        # Save results to a file
        results_filename = os.path.join("clustering_results", f"results_{os.path.basename(data_path)}_{algorithm_name}.json")
        pd.DataFrame(results).to_json(results_filename, orient='records')
        
        print(f"Results for {algorithm_name} on {data_path} saved to {results_filename}")
    
        # Create a bar chart for each algorithm
        plt.figure(figsize=(10, 6))
        plt.barh(["Best Silhouette Score"], [silhouette_score], color='skyblue')
        plt.xlabel('Silhouette Score')
        plt.title(f'Clustering Performance Chart ({algorithm_name})')
        plt.xlim(-1, 1)  # Set the x-axis range to match silhouette score range
        plt.grid(axis='x', linestyle='--', alpha=0.6)
    
        # Save the chart with high quality
        chart_filename = os.path.join("clustering_results", f"chart_{os.path.basename(data_path)}_{algorithm_name}.png")
        plt.savefig(chart_filename, dpi=300)
        plt.clf()  # Clear the current figure
    
        print(f"Chart for {algorithm_name} on {data_path} saved to {chart_filename}")
