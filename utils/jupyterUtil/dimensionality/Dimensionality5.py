from jupyterUtil.Imputers import KNN_AL, soft_imputer_regressor, simple_regressor
from jupyterUtil.Helpers import create_dir
import pandas as pd
import altair as alt
import os
# import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

KNN_AL = pd.read_csv("./new/New_dataframe_X-ray.csv_Processed.csv")
soft_imputer_regressor = pd.read_csv("./imputers/soft_imputer_regressor.csv")
# iterative_imputer_regressor = pd.read_csv("./imputers/iterative_imputer_regressor.csv")
simple_regressor = pd.read_csv("./imputers/simple_regressor.csv")
# Load example datasets

datasets = {
    "NORMAL_KNN_AL": KNN_AL, 
    'NORMAL_soft_imputer_regressor': soft_imputer_regressor , 
    'NORMAL_simple_regressor': simple_regressor
}

# Initialize dimensionality reduction techniques
dim_reduction_techniques = {
    'PCA': PCA(n_components=2),
    't-SNE': TSNE(n_components=2),
    # 'UMAP': umap.UMAP(n_components=2),
}



# Iterate through datasets
for i, dataset in enumerate(datasets):
    create_dir("dimension_reduction_charts/charts")
    # Initialize empty lists to store Altair charts
    charts = []
    format_dataset_name = dataset
    for technique_name, technique in dim_reduction_techniques.items():
        check_file = "dimension_reduction_datasets/"+ technique_name+dataset+".csv"
        if os.path.exists(check_file):
            reduced_data = pd.read_csv(check_file)
        else:
            create_dir("dimension_reduction_datasets/")
            reduced_data = technique.fit_transform(datasets[dataset])
        df = pd.DataFrame(reduced_data, columns=['PCA1', 'PCA2'])
        df.to_csv("dimension_reduction_datasets/"+ technique_name+dataset+".csv", index=False)
        chart = alt.Chart(df).mark_point().encode(
            x='PCA1:Q',
            y='PCA2:Q',
        ).properties(title=f'{technique_name} ===>>> {format_dataset_name[1]} imputer and {format_dataset_name[0]}')
        charts.append(chart)

    # Combine all charts into a single Altair visualization
    combine_chart = alt.hconcat(*charts).configure_title(
        fontSize=16
    )
    combine_chart
    combine_chart.save("dimension_reduction_charts/charts/"+dataset + 'HIGH.png', engine="vl-convert", ppi=200, format='png')
