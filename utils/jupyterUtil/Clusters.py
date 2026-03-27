# ML for clustering

from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS, SpectralClustering, MeanShift, DBSCAN
from sklearn.mixture import GaussianMixture


def kmeans_AL(data):
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(data)
    labels = kmeans.labels_
    data['output'] = labels
    # Get the cluster centers
    centers = kmeans.cluster_centers_
    return data, centers, labels

def agglomerative_clustering_AL(data):
    hierarchical = AgglomerativeClustering(n_clusters=3)
    labels = hierarchical.fit_predict(data)
    data['output'] = labels
    return data, labels

def DBSCAN_AL(data):
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(data)
    data['output'] = labels
    return data, labels

def mean_shift(data):
    meanshift = MeanShift()
    labels = meanshift.fit_predict(data)
    data['output'] = labels
    return data

def gaussian_AL(data):
    gmm = GaussianMixture(n_components=3)
    labels = gmm.fit_predict(data)
    data['labels'] = labels
    return data, labels

def spectral_AL(data):
    spectral = SpectralClustering(n_clusters=3)
    labels = spectral.fit_predict(data)
    data['output'] = labels
    return data, labels

def optics_AL(data):
    optics = OPTICS(min_samples=5, xi=0.05)
    labels = optics.fit_predict(data)
    data['output'] = labels
    return data, labels
