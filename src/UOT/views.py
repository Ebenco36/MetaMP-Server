import json
from src.UOT.services import MachineLearningService, get_use_cases
from flask import request, send_file
from flask_restful import Resource
from flask import jsonify
import altair as alt
from src.utils.response import ApiResponse

class MachineLearningView(Resource):
    def __init__(self):
        self.MLservice = MachineLearningService()
    def get(self, dataset, action):
        return jsonify('Welcome')
    
    def post(self, dataset, action):
        data = request.get_json()
        dimension_method = data.get('dimension_method', "tsne")
        
        eps = data.get('eps', 0.3)
        metric = data.get('metric', "euclidean")
        n_clusters = data.get('n_clusters', 2)
        n_components = data.get('n_components', 2)
        n_init = data.get('n_init', "auto")
        algorithm = data.get('algorithm', "auto")
        max_iter = data.get('max_iter', 300)
        min_samples = data.get('min_samples', 15)
        method = data.get('method', "barnes_hut")
        n_neighbors = data.get('n_neighbors', 20)
        min_dist = data.get('min_dist', 0.1)
        
        distance_threshold = data.get('distance_threshold', 0)
        linkage = data.get('linkage', "ward")
        
        perplexity = data.get('perplexity', 20)
        early_exaggeration = data.get('early_exaggeration', 12)
        learning_rate = data.get('learning_rate', "auto")
        
        # set dataset
        self.service = self.MLservice.setDataset(dataset)
        
        if action == 'data_view':
            return self.post_dataview()
        elif action == 'impute':
            return self.post_impute()
        elif action == 'normalize':
            return self.post_normalize()
        elif action == 'apply_pca':
            return self.post_apply_pca(n_components=n_components)
        elif action == 'apply_tsne':
            return self.post_apply_tsne(
                n_components=n_components, learning_rate=learning_rate, 
                metric=metric, perplexity=perplexity, early_exaggeration=early_exaggeration
            )
        elif action == 'apply_umap':
            return self.post_apply_UMAP(
                n_neighbors=n_neighbors, min_dist=min_dist, 
                n_components=n_components, metric=metric, method=method
            )
        elif action == 'apply_kmeans':
            return self.post_apply_kmeans(
                n_clusters=n_clusters, max_iter=max_iter, 
                n_init=n_init, dimension_method=dimension_method, 
                n_neighbors=n_neighbors, min_dist=min_dist, 
                n_components=n_components, method=method,
                learning_rate=learning_rate,  perplexity=perplexity, 
                early_exaggeration=early_exaggeration, metric=metric
            )
        elif action == 'apply_dbscan':
            return self.post_apply_dbscan(
                eps=eps, min_samples=min_samples, metric=metric, 
                algorithm=algorithm, dimension_method=dimension_method, 
                n_neighbors=n_neighbors, min_dist=min_dist, 
                n_components=n_components, method=method,
                learning_rate=learning_rate,  perplexity=perplexity, 
                early_exaggeration=early_exaggeration
            )
        elif action == 'apply_agglomerative':
            return self.post_apply_agglomerative(
                n_clusters=n_clusters, distance_threshold=distance_threshold, 
                linkage=linkage, dimension_method=dimension_method, 
                n_neighbors=n_neighbors, min_dist=min_dist, 
                n_components=n_components, method=method,
                learning_rate=learning_rate,  perplexity=perplexity, 
                early_exaggeration=early_exaggeration, metric=metric
            )
        else:
            return jsonify({"error": "Invalid action"})

    def post_dataview(self):
        data_view = self.service.dataview()
        return ApiResponse.success(data_view.to_dict(orient='records'))
    
    def post_impute(self):
        imputed_data = self.service.impute_data()
        return ApiResponse.success(imputed_data.to_dict(orient='records'))

    def post_normalize(self):
        normalized_data = self.service.normalize_data()
        return ApiResponse.success(normalized_data.to_dict(orient='records'))

    def post_apply_pca(self, n_components=2):
        pca_result = self.service.apply_pca(n_components=n_components)
        chart = self.generate_chart(
            df = pca_result, x = "Feature1", 
            y = "Feature2", color="target", title="Dimensionality Reduction Plot")
        data = {
            "data": pca_result.to_dict(orient='records'), 
            "chart": chart
        }
        return ApiResponse.success(data, "Fetch successfully", 200)

    def post_apply_tsne(self, n_components=2, learning_rate='auto', metric='euclidean', perplexity=15, early_exaggeration=12):
        tsne_result = self.service.apply_tsne(
            n_components=n_components, learning_rate=learning_rate, 
            metric=metric, perplexity=perplexity, early_exaggeration=early_exaggeration
        )
        chart = self.generate_chart(
            df = tsne_result, x = "Feature1", 
            y = "Feature2", color="target", title="Dimensionality Reduction Plot")
        data = {
            "data": tsne_result.to_dict(orient='records'), 
            "chart": chart
        }
        return ApiResponse.success(data, "Fetch successfully", 200)
    
    def post_apply_UMAP(self, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', method="barnes_hut"):
        umap_result = self.service.apply_umap(
            n_neighbors=n_neighbors, min_dist=min_dist, 
            n_components=n_components, metric=metric, method=method
        )
        chart = self.generate_chart(
            df = umap_result, x = "Feature1", 
            y = "Feature2", color="target", title="Dimensionality Reduction Plot")
        data = {
            "data": umap_result.to_dict(orient='records'), 
            "chart": chart
        }
        return ApiResponse.success(data, "Fetch successfully", 200)

    def post_apply_kmeans(
        self, n_clusters=2, max_iter=300, n_init="auto", dimension_method="tsne",
        n_neighbors=15, min_dist=0.1, n_components=2, method="barnes_hut",
        learning_rate='auto',  perplexity=15, early_exaggeration=12, metric=None
    ):
        kmeans_result, scores = self.service.apply_kmeans(
            n_clusters=n_clusters, max_iter=max_iter, 
            n_init=n_init, dimension_method=dimension_method, 
            n_neighbors=n_neighbors, min_dist=min_dist, 
            n_components=n_components, method=method,
            learning_rate=learning_rate,  perplexity=perplexity, 
            early_exaggeration=early_exaggeration, metric=metric
        )
        chart = self.generate_chart(
            df = kmeans_result, x = "Feature1", 
            y = "Feature2", color="labels", title="Clustering Plot")
        data = {
            "data": kmeans_result.to_dict(orient='records'), 
            "chart": chart,
            "scores": scores
        }
        return ApiResponse.success(data, "Fetch successfully", 200)

    def post_apply_dbscan(
        self, eps=0.5, min_samples=5, metric=None, algorithm=None, dimension_method="tsne",
        n_neighbors=15, min_dist=0.1, n_components=2, method="barnes_hut",
        learning_rate='auto',  perplexity=15, early_exaggeration=12
    ):
        dbscan_result, scores = self.service.apply_dbscan(
            eps=eps, min_samples=min_samples, metric=metric, 
            algorithm=algorithm, dimension_method=dimension_method, 
            n_neighbors=n_neighbors, min_dist=min_dist, 
            n_components=n_components, method=method,
            learning_rate=learning_rate,  perplexity=perplexity, 
            early_exaggeration=early_exaggeration
        )
        chart = self.generate_chart(
            df = dbscan_result, x = "Feature1", 
            y = "Feature2", color="labels", title="Clustering Plot")
        data = {
            "data": dbscan_result.to_dict(orient='records'), 
            "chart": chart,
            "scores": scores
        }
        return ApiResponse.success(data, "Fetch successfully", 200)
    
    def post_apply_agglomerative(
        self, n_clusters=2, distance_threshold=0, linkage="ward", dimension_method="tsne",
        n_neighbors=15, min_dist=0.1, n_components=2, method="barnes_hut",
        learning_rate='auto',  perplexity=15, early_exaggeration=12, metric=None
    ):
        agglomerative_result, scores = self.service.apply_agglomerative(
            n_clusters=n_clusters, distance_threshold=distance_threshold, 
            linkage=linkage, dimension_method=dimension_method, 
            n_neighbors=n_neighbors, min_dist=min_dist, 
            n_components=n_components, method=method,
            learning_rate=learning_rate,  perplexity=perplexity, 
            early_exaggeration=early_exaggeration, metric=metric
        )
        chart = self.generate_chart(
            df = agglomerative_result, x = "Feature1", 
            y = "Feature2", color="labels", title="Clustering Plot")
        data = {
            "data": agglomerative_result.to_dict(orient='records'), 
            "chart": chart,
            "scores": scores
        }
        return ApiResponse.success(data, "Fetch successfully", 200)

    @staticmethod
    def generate_chart(df=None, x=None, y=None, color=None, title=None):

        chart = alt.Chart(df).mark_circle().encode(
            x=x,
            y=y,
            color=color + ':N',
            tooltip=[alt.Tooltip(tooltip, title=tooltip.capitalize()) for tooltip in [x, y, "target"]]
        ).properties(
            title=title,
            width="container",
        ).interactive().configure_legend(orient='bottom', direction = 'vertical')
        
        return chart.to_dict()
    
    
    
class UseCases(Resource):
    def __init__(self):
        pass
    
    def get(self):
        data = {
            "em_features": [
                'emt_molecular_weight', 
                'reconstruction_num_particles',
                'processed_resolution'
            ],
            "x_ray_features": [
                "cell_length_a", 
                "cell_length_b", 
                "cell_length_c", 
                "crystal_density_matthews",
                "molecular_weight", 
                "processed_resolution"
            ],
        }
        
        return ApiResponse.success(data, "Fetch successfully")
    
    def post(self):
        data = request.get_json()
        use_case = data.get("use_case", "summary_statistics")
        chart_type = data.get("chart_type", "bar")
        groupby = data.get("category", "group")
        chart_width = data.get("chart_width", 800)
        chart_trend = data.get("chart_trend", "No")
        outlier_detection_by_method = data.get("outlier_detection_by_method", "EM")
        
        #### For dection #####
        if "features" in data and len(data.get("features")) == 0:
            del data["features"]
        if outlier_detection_by_method == "EM":
            features = data.get("features", [
                'emt_molecular_weight', 
                'reconstruction_num_particles',
                'processed_resolution'
            ])
        else:
            features = data.get("features", [
                'cell_length_a', 
                'cell_length_b',
                'cell_length_c'
            ])
        
        
        payload = {
            "groupby": groupby,
            "features": features,
            "use_case": use_case,
            "chart_type": chart_type,
            "chart_width": chart_width,
            "chart_trend": chart_trend,
            "outlier_detection_by_method": outlier_detection_by_method
        }
        response = get_use_cases(payload)
        
        return ApiResponse.success(response, "Fetch successfully")