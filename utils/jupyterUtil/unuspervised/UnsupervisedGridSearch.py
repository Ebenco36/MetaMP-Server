import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans, DBSCAN, Birch, MeanShift, MiniBatchKMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from jupyterUtil.GetFilePathIntoDict import getFiles
from jupyterUtil.Helpers import create_dir
from kmodes.kmodes import KModes

import matplotlib
matplotlib.use('Agg')

# Define your hyperparameter ranges
k_values = range(2, 10)
eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
branching_factor_values = [10, 50, 100, 200]
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5]
bandwidth_values = [0.1, 0.5, 1.0, 2.0, 3.0]
min_samples_values = [5, 10, 15, 20, 30, 40]
xi_values = [0.05, 0.1, 0.2, 0.3, 0.4]
metric_values = ['euclidean']
min_samples_values = [5, 10, 15, 20, 25, 30]
n_init_values = [5, 10, 15, 20, 25, 30]

mean_shift_params = {
    'bandwidth': [0.1, 0.5, 1.0, 2.0, 3.0]
}
optics_params = {
    'min_samples': min_samples_values,
    'xi': [0.05, 0.1, 0.2, 0.3, 0.4]
}

kmodes_params = {
    'n_clusters': k_values,  # Adjust the number of clusters (k)
    'init': ['Huang', 'Cao'] # Consider different initialization methods
}
agg_clustering_params = {
    'n_clusters': k_values,
    'metric': ['euclidean'],
    'linkage': ['ward', 'complete', 'average', 'single']
}

# Create a directory to save results
os.makedirs("clustering_results", exist_ok=True)

# Evaluate cluster quality using Silhouette Score
def silhouette_evaluate_cluster_quality(model, data_scaled):
    if model:
        labels = model.fit_predict(data_scaled)
        if (len(set(labels)) > 1):
          silhouette = silhouette_score(data_scaled, labels)
          return silhouette
        else:
          return 0
    return -1
    

# Evaluate cluster quality using Silhouette Score
def calinski_evaluate_cluster_quality(model, data_scaled):
    if model:
        labels = model.fit_predict(data_scaled)
        if (len(set(labels)) > 1):
          calinski_harabasz = calinski_harabasz_score(data_scaled, labels)
          
          return calinski_harabasz
        else:
          return 0
    return -1
    
    
# Evaluate cluster quality using Silhouette Score
def davies_evaluate_cluster_quality(model, data_scaled):
    if model:
        labels = model.fit_predict(data_scaled)
        if (len(set(labels)) > 1):
          davies_bouldin = davies_bouldin_score(data_scaled, labels)
          return davies_bouldin
        else:
          return 0
    return -1
    
def perform_grid_search(model, param_grid, data_scaled):
    grid_search = GridSearchCV(model, param_grid, scoring=silhouette_evaluate_cluster_quality, n_jobs=-1)
    grid_search.fit(data_scaled)
    best_model = grid_search
    return best_model
    
def fit_to_label(best_model, X, model_name, dataset_name, best_score, best_params):
    create_dir("./clustering_prediction_dataset/")
    data = X.copy()
    labels = best_model.fit_predict(X)
    data[model_name] = labels
    data.to_csv("./clustering_prediction_dataset/" + model_name + "_" + dataset_name + ".csv", index=False)
    saveChart(data, model_name, dataset_name, best_score, best_params)
    
def saveChart(X, labels, dataset_name, best_score, best_params):
    create_dir("./clustering_prediction_charts/")
    X_ = X.columns.to_list()
    plt.figure(figsize=(10, 6))
    plt.scatter(X[X_[0]], X[X_[1]], c=X[labels], cmap='viridis')
    title_font = {'fontsize': 16, 'fontweight': 'bold'}
    plt.title(labels + " on " + dataset_name + " \n  with the score: " + str(best_score) + " \n parameters: " + str(best_params), fontdict=title_font)
    # Add axis labels and adjust layout
    plt.xlabel(f'Feature {X_[0]}')
    plt.ylabel(f'Feature {X_[1]}')
    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the chart with high quality
    plt.savefig("./clustering_prediction_charts/" + f'{labels}_{dataset_name}_scatter_plot.png', dpi=300)
    plt.close()  # Close the current figure
    

def saveJson(my_dict, file_path = "my_dict.json"):
    # Save the dictionary to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(my_dict, json_file)


data_paths = getFiles()

# Loop over your datasets
for data_path in data_paths:
    print(data_path)
    create_dir("./clustering_results/")
    # Load or generate your data here
    data = pd.read_csv(data_paths[data_path])
    
    # Preprocess the data if needed (scaling, etc.)
    data_scaled = data[data.columns.to_list()]
    
    # Define a dictionary to store results
    results = {"Dataset": data_path, "Silhouette Scores": {}}
    
    # Define models for hyperparameter tuning
    clustering_algorithms = [
        ("K-Means", KMeans(), {"n_init": n_init_values, "n_clusters": k_values}),
        ("DBSCAN", DBSCAN(), {"eps": eps_values, "min_samples": min_samples_values}),
        ("BIRCH", Birch(), {"branching_factor": branching_factor_values, "threshold": threshold_values}),
        ("Mean Shift", MeanShift(), mean_shift_params),
        ("MiniBatchKMeans", MiniBatchKMeans(), {"n_init": n_init_values, "n_clusters": k_values}),
        ("Agglomerative Clustering", AgglomerativeClustering(), agg_clustering_params),
        ("Gaussian Mixture Models", GaussianMixture(), {"n_components": k_values}),
        ("OPTICS", OPTICS(), optics_params),
        ("KModes", KModes(), kmodes_params)
    ]

    # Iterate over clustering algorithms
    for algorithm_name, model, param_grid in clustering_algorithms:
        grid_search = perform_grid_search(model, param_grid, data_scaled)

        # Get the best model and its silhouette score
        best_model = grid_search.best_estimator_
        best_silhouette = grid_search.best_score_
        best_params = grid_search.best_params_
        
        fit_to_label(best_model, data_scaled, algorithm_name, data_path, best_silhouette, best_params)

        # Store results
        results["Silhouette Scores"][algorithm_name] = best_silhouette

        # Save the best model to a file (if needed)
        # best_model_filename = os.path.join("clustering_results", f"best_model_{data_path}_{algorithm_name}.pkl")
        # joblib.dump(best_model, best_model_filename)
        # Create a bar chart for each dataset
    
    # Save results to a file
    # print(results)
    results_filename = os.path.join("clustering_results", f"results_{os.path.basename(data_path)}.json")
    #pd.DataFrame(results).to_json(results_filename, orient='records')
    saveJson(results, results_filename)

    # Create a bar chart for each dataset
    plt.figure(figsize=(10, 6))
    silhouettes = results["Silhouette Scores"]
    plt.barh(list(silhouettes.keys()), list(silhouettes.values()), color='skyblue')
    plt.xlabel('Silhouette Score')
    plt.title('Clustering Performance Chart')
    plt.xlim(-1, 1)  # Set the x-axis range to match silhouette score range
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Save the chart with high quality
    plt.savefig("./clustering_results/" + f'{data_path}_scatter_plot.png', dpi=300)
    plt.clf()  # Clear the current figure
    plt.close()  # Close the current figure

    print(f"Chart for {data_path} saved to {data_path}_scatter_plot.png")
