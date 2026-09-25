from jupyterUtil.Imputers import simple_regressor
from jupyterUtil.Normalization import minmax, maxAbs
from jupyterUtil.Helpers import create_dir
import pandas as pd
import altair as alt
import os
# import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv("./new/New_dataframe_X-ray.csv_Processed.csv")
imputed_data_SR = simple_regressor(df)
# save imputer
imputed_data_SR.to_csv("./imputers/simple_regressor.csv", index=False)

# Load example datasets
create_dir("dimension_reduction_datasets/normalized_dataset/")

minmax_data_sr = minmax(imputed_data_SR)
minmax_data_sr.to_csv("./dimension_reduction_datasets/normalized_dataset/simple_regressor_minmax.csv", index=False)

maxabs_data_sr  = maxAbs(imputed_data_SR)
maxabs_data_sr.to_csv("./dimension_reduction_datasets/normalized_dataset/simple_regressor_maxAbs.csv", index=False)

datasets = {
    'minmax-simpleImputer': minmax_data_sr , 
    'maxabs-simpleImputer': maxabs_data_sr,
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
