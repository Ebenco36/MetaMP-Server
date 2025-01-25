import pandas as pd
import altair as alt
import os
# import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from jupyterUtil.Imputers import KNN_AL
from jupyterUtil.Normalization import minmax, maxAbs
from jupyterUtil.Helpers import create_dir

def perform_dimensionality_reduction(dataset, technique_name, technique, parameters, dataset_name):
    check_file = f"dimension_reduction_datasetsss/{technique_name}_{dataset_name}.csv"
    if os.path.exists(check_file):
        reduced_data = pd.read_csv(check_file)
    else:
        create_dir("dimension_reduction_datasets_testsss/")
        reduced_data = technique(**parameters).fit_transform(dataset)
        df = pd.DataFrame(reduced_data, columns=[f'{technique_name}1', f'{technique_name}2'])
        df.to_csv(check_file, index=False)
    return reduced_data, technique_name, dataset_name

# Load data
df = pd.read_csv("./new/New_dataframe_X-ray.csv_Processed.csv")
imputed_data_KNN = KNN_AL(df)
imputed_data_KNN.to_csv("./imputers/KNN_AL.csv", index=False)

# Load example datasets
create_dir("dimension_reduction_datasetsss/normalized_dataset/")

minmax_data = minmax(imputed_data_KNN)
minmax_data.to_csv("./dimension_reduction_datasetsss/normalized_dataset/KNN_AL_minmax.csv", index=False)

maxabs_data = maxAbs(imputed_data_KNN)
maxabs_data.to_csv("./dimension_reduction_datasetsss/normalized_dataset/KNN_AL_maxAbs.csv", index=False)

datasets = {
    'minmaxScaler-KNN': minmax_data,
    'maxabsScaler-KNN': maxabs_data,
}

# Parameter grids for t-SNE and UMAP
tsne_params = [
    {'n_components': 2},
    {'n_components': 2, 'perplexity': 50},
    {'n_components': 2, 'learning_rate': 200},
]

umap_params = [
    {'n_components': 2},
    {'n_components': 2, 'n_neighbors': 5},
    {'n_components': 2, 'min_dist': 0.1},
]

# Iterate through datasets
for dataset_name, dataset in datasets.items():
    create_dir("dimension_reduction_chartsss/charts")
    charts = []
    
    # Perform dimensionality reduction for t-SNE with different parameters
    for params in tsne_params:
        reduced_data, technique_name, dataset_name = perform_dimensionality_reduction(dataset, 't-SNE', TSNE, params, dataset_name)
        chart = alt.Chart(pd.DataFrame(reduced_data, columns=[f'{technique_name}1', f'{technique_name}2'])).mark_point().encode(
            x=f'{technique_name}1:Q',
            y=f'{technique_name}2:Q',
        ).properties(title=f'{technique_name} ===>>> {dataset_name}' + str(params))
        charts.append(chart)
    
    # Perform dimensionality reduction for UMAP with different parameters
    for params in umap_params:
        reduced_data, technique_name, dataset_name = perform_dimensionality_reduction(dataset, 'UMAP', umap.UMAP, params, dataset_name)
        chart = alt.Chart(pd.DataFrame(reduced_data, columns=[f'{technique_name}1', f'{technique_name}2'])).mark_point().encode(
            x=f'{technique_name}1:Q',
            y=f'{technique_name}2:Q',
        ).properties(title=f'{technique_name} ===>>> {dataset_name}')
        charts.append(chart)

    # Combine all charts into a single Altair visualization
    combine_chart = alt.hconcat(*charts)
    combine_chart.save(f"dimension_reduction_chartsss/charts/{dataset_name}_HIGH.png", engine="vl-convert", ppi=200, format='png')
