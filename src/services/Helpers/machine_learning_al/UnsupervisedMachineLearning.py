import numpy as np
import joblib
from sklearn.cluster import (
    DBSCAN, MeanShift, 
    AgglomerativeClustering, 
    OPTICS, AffinityPropagation, 
    SpectralClustering, KMeans, 
    SpectralClustering
)
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score
)
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram
class MachineLearning:

    def __init__(
        self, X = [], eps=0.3, min_samples=None, 
        n_clusters=2, n_components=2, 
        UOT = False, save_path = None,
        linkage = "ward", metric = "euclidean",
        distance_threshold = None
    ):
        """
            if UOT is True then we only want to train separately first before predicting
        """
        self.X = X
        self.eps = eps
        self.UOT = UOT
        self.save_path = save_path
        self.n_clusters = n_clusters
        self.min_samples = min_samples
        self.n_components = n_components
        self.distance_threshold = distance_threshold
        # for agglomerative
        self.metric = metric
        self.linkage = linkage
        self.dendogram = None
        self.silhouette = None
        self.calinski_harabasz = None
            
    
    def run_clustering_algorithm(self, algorithm, **kwargs):
        clustering = algorithm(**kwargs)
        if (self.UOT is True):
            path = self.save_path if(self.save_path) else "./public/data_sessions/"
            complete_path = path + "/_" + clustering.__class__.__name__ + "_.pkl"
            clustering = clustering.fit(self.X)
            labels = clustering.fit_predict(self.X)
            joblib.dump(labels, complete_path)
            params = clustering.get_params(deep=True)
            # return clustering, labels, params, complete_path
        else:
            complete_path = "We are not saving anything"
            clustering = clustering.fit(self.X)
            labels = clustering.fit_predict(self.X)
            params = clustering.get_params(deep=True)
            
        self.silhouette = self.silhouette_evaluate_cluster_quality(labels, self.X)
        self.calinski_harabasz = self.calinski_evaluate_cluster_quality(labels, self.X)
        
        return clustering, labels, params, complete_path
        

    # DBSCAN
    def dbscan_clustering(self):
        model, labels, params, path = self.run_clustering_algorithm(DBSCAN, eps=self.eps, min_samples=self.min_samples)
        self.X['clustering'] = labels
        return self.X, params, path, self.dendogram, self.silhouette, self.calinski_harabasz

    # Mean Shift
    def mean_shift_clustering(self):
        model, labels, params, path = self.run_clustering_algorithm(MeanShift)
        self.X['clustering'] = labels
        return self.X, params, path, self.dendogram, self.silhouette, self.calinski_harabasz
    
    # Agglomerative Clustering
    def agglomerative_clustering(self):
        # print(self.distance_threshold, self.n_clusters, self.linkage, self.metric)
        if((isinstance(self.n_clusters, int) and self.n_clusters >= 2)):
            self.distance_threshold = None
        else:
            self.n_clusters = None
            
        model, labels, params, path = self.run_clustering_algorithm(
            AgglomerativeClustering,
            distance_threshold = self.distance_threshold,
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            metric=self.metric, 
        )
        self.X['clustering'] = labels
        #Not needed for now
        if(self.distance_threshold == False):
            self.dendogram = self.plot_dendrogram(model, truncate_mode="level", p=3)
        return self.X, params, path, self.dendogram, self.silhouette, self.calinski_harabasz
    """
    # OPTICS
    def optics_clustering(self):
        labels, params, path = self.run_clustering_algorithm(OPTICS, min_samples=self.min_samples)
        self.X['clustering'] = labels
        return self.X, params, path

    # Affinity Propagation
    def affinity_propagation_clustering(self):
        labels, param, paths = self.run_clustering_algorithm(AffinityPropagation)
        self.X['clustering'] = labels
        return self.X, param, paths

    # SpectralClustering Propagation
    def spectral_clustering(self):
        labels, params, path = self.run_clustering_algorithm(SpectralClustering)
        self.X['clustering'] = labels
        return self.X, params, path

    """
    # KMeans Propagation
    def kMeans_clustering(self):
        print(self.n_clusters)
        model, labels, params, path = self.run_clustering_algorithm(KMeans, n_init="auto", n_clusters=self.n_clusters)
        self.X['clustering'] = labels
        return self.X, params, path, self.dendogram, self.silhouette, self.calinski_harabasz

    def gaussian_clustering(self):
        # Create an instance of the Gaussian Mixture Model algorithm
        model, labels, params, path = self.run_clustering_algorithm(GaussianMixture, n_components=self.n_components)
        self.X['clustering'] = labels
        return self.X, params, path, self.dendogram, self.silhouette, self.calinski_harabasz

    @staticmethod
    def make_predictions(model_path, new_data):
        # Load the model from the file
        loaded_model = joblib.load(model_path)
        # Predict clusters for new data using the loaded model
        new_predictions = loaded_model.predict(new_data)

        return new_predictions
    
    @staticmethod
    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # Create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        # Construct the linkage matrix with distances and counts
        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        return dendrogram(linkage_matrix, **kwargs)
    
    
    # Evaluate cluster quality using Silhouette Score
    def silhouette_evaluate_cluster_quality(self, labels, data_scaled):
        if (len(set(labels)) > 1):
            self.silhouette = silhouette_score(data_scaled, labels)
            return self.silhouette
        else:
            return 0
        return -1
        

    # Evaluate cluster quality using Silhouette Score
    def calinski_evaluate_cluster_quality(self, labels, data_scaled):
        if (len(set(labels)) > 1):
            self.calinski_harabasz = calinski_harabasz_score(data_scaled, labels)
            
            return self.calinski_harabasz
        else:
            return 0
        return -1