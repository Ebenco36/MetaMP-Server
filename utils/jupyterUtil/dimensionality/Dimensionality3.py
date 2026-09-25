from jupyterUtil.Imputers import KNN_AL, soft_imputer_regressor, iterative_imputer_regressor, simple_regressor
from jupyterUtil.Normalization import standardScaler, minmax, robustScaler, maxAbs, yeo_johnson, quantile_transform, unit_vector
from jupyterUtil.Helpers import create_dir
import pandas as pd
import altair as alt
import os
# import umap
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap, MDS


df = pd.read_csv("./dataSpace/New_dataframe_EM.csv_Processed.csv")
print("We are here ***")
imputed_data_IIR = iterative_imputer_regressor(df)
imputed_data_IIR.to_csv("./imputers/iterative_imputer_regressor.csv", index=False)
print("We are here 0")
create_dir("dimension_reduction_datasets/normalized_dataset/")

minmax_data_iir = minmax(imputed_data_IIR)
minmax_data_iir.to_csv("./dimension_reduction_datasets/normalized_dataset/iterative_imputer_regressor_minmax.csv", index=False)


maxabs_data_iir  = maxAbs(imputed_data_IIR)
maxabs_data_iir.to_csv("./dimension_reduction_datasets/normalized_dataset/iterative_imputer_regressor_maxAbs.csv", index=False)


datasets = {
    'minmaxScaler-iterativeImputerRegressor': minmax_data_iir , 
    'maxabsScaler-iterativeImputerRegressor': maxabs_data_iir,
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
            print("Training to fit")
            reduced_data = technique.fit_transform(datasets[dataset])
        df = pd.DataFrame(reduced_data, columns=[f'{technique_name}1', f'{technique_name}2'])
        df.to_csv("dimension_reduction_datasets/"+ technique_name+dataset+".csv", index=False)
        chart = alt.Chart(df).mark_point().encode(
            x=f'{technique_name}1:Q',
            y=f'{technique_name}2:Q',
        ).properties(title=f'{technique_name} ===>>> {format_dataset_name[1]} imputer and {format_dataset_name[0]}')
        charts.append(chart)

    # Combine all charts into a single Altair visualization
    combine_chart = alt.hconcat(*charts)
    combine_chart
    combine_chart.save("dimension_reduction_charts/charts/"+dataset + 'HIGH.png', engine="vl-convert", ppi=200, format='png')
