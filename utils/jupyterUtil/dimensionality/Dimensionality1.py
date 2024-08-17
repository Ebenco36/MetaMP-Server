from jupyterUtil.Imputers import KNN_AL
from jupyterUtil.Normalization import minmax, maxAbs
from jupyterUtil.Helpers import create_dir
import pandas as pd
import altair as alt
import os
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv("./new/New_dataframe_X-ray.csv_Processed.csv")
imputed_data_KNN = KNN_AL(df)
imputed_data_KNN.to_csv("./imputers/KNN_AL.csv", index=False)

# Load example datasets
create_dir("dimension_reduction_datasets/normalized_dataset/")

minmax_data = minmax(imputed_data_KNN)
minmax_data.to_csv("./dimension_reduction_datasets/normalized_dataset/KNN_AL_minmax.csv", index=False)

maxabs_data = maxAbs(imputed_data_KNN)
maxabs_data.to_csv("./dimension_reduction_datasets/normalized_dataset/KNN_AL_maxAbs.csv", index=False)



datasets = { 
    'minmax-KNN': minmax_data, 
    'maxabs-KNN': maxabs_data,
}

# Initialize dimensionality reduction techniques
dim_reduction_techniques = {
    'PCA': PCA(n_components=2),
    't-SNE': TSNE(n_components=2),
    'UMAP': umap.UMAP(n_components=2),
}



# Iterate through datasets
for i, dataset in enumerate(datasets):
    create_dir("dimension_reduction_charts/charts")
    # Initialize empty lists to store Altair charts
    charts = []
    format_dataset_name = dataset.split("-")
    for technique_name, technique in dim_reduction_techniques.items():
        check_file = "dimension_reduction_datasets/"+ technique_name+dataset+".csv"
        if os.path.exists(check_file):
            reduced_data = pd.read_csv(check_file)
        else:
            create_dir("dimension_reduction_datasets/")
            reduced_data = technique.fit_transform(datasets[dataset])
        df = pd.DataFrame(reduced_data, columns=[f'{technique_name}1', f'{technique_name}2'])
        df.to_csv("dimension_reduction_datasets/"+ technique_name+dataset+".csv", index=False)
        chart = alt.Chart(df).mark_point().encode(
            x=f'{technique_name}1:Q',
            y=f'{technique_name}2:Q',
        ).properties(title=f'{technique_name} ===>>> {format_dataset_name[1]} imputer and {format_dataset_name[0]}')
        charts.append(chart)
    # Combine all charts into a single Altair visualization
    combine_chart = alt.hconcat(*charts).configure_title(
        fontSize=16
    )
    combine_chart
    combine_chart.save("dimension_reduction_charts/charts/"+dataset + 'HIGH.png', engine="vl-convert", ppi=200, format='png')
