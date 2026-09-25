
import os
import joblib
import warnings
import numpy as np
import pandas as pd
import altair as alt
try:
    from umap import UMAP
except ImportError:  # pragma: no cover - optional dependency for UMAP views
    UMAP = None
# import tensorflow as tf
import matplotlib.pyplot as plt
try:
    import shap
except ImportError:  # pragma: no cover - optional dependency for explainability plots
    shap = None
from sklearn.utils import shuffle
from sklearn.svm import SVC, OneClassSVM
from sklearn.impute import (
    SimpleImputer, KNNImputer
)
from sklearn.decomposition import (
    PCA, IncrementalPCA
)
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, 
    RobustScaler, OneHotEncoder, LabelEncoder
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, 
    ParameterGrid, GridSearchCV
)
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering
)
from sklearn.manifold import TSNE
os.environ["NUMBA_CACHE_DIR"] = "/tmp"
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score, recall_score, 
    f1_score, silhouette_score, mean_squared_error, roc_auc_score, roc_curve,
    confusion_matrix
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest, RandomForestRegressor
)
from sklearn.semi_supervised import SelfTrainingClassifier


def drop_id_columns(dataframe):
    """
    Drops columns from a DataFrame that contain '_id' or 'id_' in their names.

    Parameters:
    dataframe (DataFrame): The DataFrame from which to drop columns.

    Returns:
    DataFrame: DataFrame with columns containing '_id' or 'id_' dropped.

    """
    columns_to_drop = [col for col in dataframe.columns if '_id' in col or 'id_' in col]
    return dataframe.drop(columns=columns_to_drop)

def drop_columns(dataframe, columns_to_drop=[]):
    """
    Safely drops specified columns from a DataFrame, ignoring non-existent columns.

    Parameters:
    - dataframe (DataFrame): The input DataFrame from which to drop columns.
    - columns_to_drop (list): List of column names to drop.

    Returns:
    - DataFrame: A new DataFrame with the specified columns removed.
    """
    # Filter only columns that exist in the DataFrame
    valid_columns_to_drop = [col for col in columns_to_drop if col in dataframe.columns]

    # Drop only valid columns
    return dataframe.drop(columns=valid_columns_to_drop)


def separate_numerical_categorical(data):
    """
    Separate the columns of a DataFrame into numerical and categorical.

    Parameters:
    data (DataFrame): The DataFrame to separate.

    Returns:
    tuple: A tuple containing two lists, one for numerical columns and one for categorical columns.

    """
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=['number']).columns.tolist()
    
    return numerical_cols, categorical_cols

def onehot_encoder(data, columns=None, drop_first=None):
    """
    One-hot encodes categorical columns in a DataFrame or a list of categorical columns.

    Parameters:
    data (DataFrame or list): The DataFrame containing the categorical columns or a list of categorical column names.
    columns (list or None): List of columns to one-hot encode. If None and data is a DataFrame, all object or category dtype columns will be encoded.
    drop_first (str, array-like, or None): Strategy to use for dropping one of the categories. Options are 'if_binary', 'first', or None. Default is None.

    Returns:
    DataFrame: DataFrame with one-hot encoded columns.

    """
    if isinstance(data, pd.DataFrame):
        # If columns are not specified, select object or category dtype columns
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        data_to_encode = data[columns]
    elif isinstance(data, list):
        data_to_encode = pd.DataFrame(data, columns=['Category'])  # Create a DataFrame with a single column
        columns = ['Category']  # Assign a column name for consistency
    else:
        raise ValueError("Input data must be a DataFrame or a list of categorical column names.")

    # Create OneHotEncoder object
    encoder = OneHotEncoder(drop=drop_first, sparse_output=False)

    # Fit and transform the data
    encoded_data = encoder.fit_transform(data_to_encode)

    # Create column names for the encoded features
    encoded_columns = encoder.get_feature_names_out(columns)

    # Create DataFrame with encoded columns
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data_to_encode.index)

    return encoded_df

def impute_knn_best_hyperparameter(dataframe, n_neighbors_options=[1, 3, 5, 7, 10], default_k=3, cv=5):
    """Impute missing values using KNN with the best hyper-parameter for the number of neighbors."""
    numeric_data = dataframe.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        raise ValueError("No numeric data available for imputation.")

    best_score = float('inf')
    best_k = None
    
    X_train, _ = train_test_split(numeric_data, test_size=0.2, random_state=42)  # Split data for cross-validation

    for k in n_neighbors_options:
        imputer = KNNImputer(n_neighbors=k)
        try:
            imputed_scores = cross_val_score(imputer, X_train, cv=cv, scoring='neg_mean_squared_error', error_score=np.nan)
            mean_score = -np.mean(imputed_scores)
        except Exception as e:
            print(f"Error occurred during cross-validation for k={k}: {e}")
            continue

        if mean_score < best_score:
            best_score = mean_score
            best_k = k

    if best_k is None:
        print("No optimal k found; using default k =", default_k)
        best_k = default_k

    best_imputer = KNNImputer(n_neighbors=best_k)
    imputed_data = best_imputer.fit_transform(numeric_data)
    imputed_df = pd.DataFrame(imputed_data, index=numeric_data.index, columns=numeric_data.columns)
    
    return imputed_df, best_k

class ImputationComparison:
    def __init__(self, dataframe, target_column, missing_fraction=0.1, cv=5):
        self.dataframe = dataframe
        self.target_column = target_column
        self.missing_fraction = missing_fraction
        self.cv = cv
        self.imputers = {
            'KNNImputer': KNNImputer(),
            'SimpleImputer_mean': SimpleImputer(strategy='mean'),
            'SimpleImputer_median': SimpleImputer(strategy='median')
        }
        self.best_imputer = None
        self.best_score = float('inf')
        self.best_name = None

    def preprocess_data(self):
        # Drop rows with missing values in the target column
        self.dataframe.dropna(subset=[self.target_column], inplace=True)
        # Drop columns with more than 50% missing values
        threshold = 0.5 * len(self.dataframe)
        self.dataframe.dropna(axis=1, thresh=threshold, inplace=True)
        
        numeric_data = self.dataframe.select_dtypes(include=np.number)
        categorical_data = self.dataframe.select_dtypes(exclude=np.number)
        
        # Fill missing values in numeric columns using SimpleImputer with mean strategy
        if not numeric_data.empty:
            imputer_numeric = SimpleImputer(strategy='mean')
            self.dataframe[numeric_data.columns] = imputer_numeric.fit_transform(numeric_data)
        
        # Fill missing values in categorical columns using SimpleImputer with most frequent strategy
        if not categorical_data.empty:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            self.dataframe[categorical_data.columns] = imputer_categorical.fit_transform(categorical_data)

    def simulate_missing_data(self, data):
        data_missing = data.copy()
        n_missing = int(np.floor(self.missing_fraction * data_missing.size))
        missing_indices = np.random.choice(data_missing.size, n_missing, replace=False)
        data_missing.values.ravel()[missing_indices] = np.nan
        return data_missing

    def evaluate_imputers(self):
        self.preprocess_data()
        numeric_data = self.dataframe.select_dtypes(include=[np.number])
        if numeric_data.empty:
            raise ValueError("No numeric data available for imputation.")

        X_train, _ = train_test_split(numeric_data, test_size=0.2, random_state=42)
        X_val_missing = self.simulate_missing_data(X_train)

        scores = []
        for name, imputer in self.imputers.items():
            try:
                imputer.fit(X_train)
                X_val_imputed = imputer.transform(X_val_missing)
                mse = mean_squared_error(X_train, X_val_imputed)
                scores.append((name, mse))
                if mse < self.best_score:
                    self.best_score = mse
                    self.best_imputer = imputer
                    self.best_name = name
            except Exception as e:
                print(f"Error occurred with {name}: {e}")

        if self.best_imputer is None:
            raise ValueError("All imputers failed. No imputer was successfully fitted.")

        return scores

    def impute(self):
        if self.best_imputer is None:
            raise ValueError("No best imputer set. Ensure evaluate_imputers() is called first.")
        
        numeric_data = self.dataframe.select_dtypes(include=[np.number])
        imputed_data = self.best_imputer.fit_transform(numeric_data)
        imputed_df = pd.DataFrame(imputed_data, index=numeric_data.index, columns=numeric_data.columns)
        return imputed_df

    def plot_results(self, X_train, X_val_imputed, scores):
        # Scatter plot of original vs imputed values
        scatter_data = pd.DataFrame({
            'Original': X_train.values.ravel(),
            'Imputed': X_val_imputed.ravel()
        })
        
        scatter_plot = alt.Chart(scatter_data).mark_point().encode(
            x='Original',
            y='Imputed',
            tooltip=['Original', 'Imputed']
        ).properties(
            title='Original vs Imputed Values',
            width=600,
            height=400
        ) + alt.Chart(scatter_data).mark_line(color='red').encode(
            x='Original',
            y='Original'
        )

        # Bar plot for error metrics
        scores_df = pd.DataFrame(scores, columns=['Imputer', 'MSE'])
        
        bar_plot = alt.Chart(scores_df).mark_bar().encode(
            x='Imputer',
            y='MSE',
            tooltip=['Imputer', 'MSE']
        ).properties(
            title='Imputer Mean Squared Error',
            width=600,
            height=400
        )
        
        return scatter_plot & bar_plot

def evaluate_imputation(original_data, imputed_data):
    """Evaluate the effect of imputation on the dataset."""
    print("\nEvaluation of Imputation:")
    for col in original_data.columns:
        original_mean = original_data[col].mean()
        imputed_mean = imputed_data[col].mean()
        original_var = original_data[col].var()
        imputed_var = imputed_data[col].var()
        print(f"{col} - Original Mean: {original_mean:.2f}, Imputed Mean: {imputed_mean:.2f}, "
              f"Original Variance: {original_var:.2f}, Imputed Variance: {imputed_var:.2f}")

def evaluate_dimensionality_reduction(data, methods_params):
    """
    Evaluate dimensionality reduction techniques visually using Altair for plotting.

    Parameters:
    data (array-like): Input data.
    methods_params (dict): Dictionary containing lists of parameter dictionaries to try for each method.

    Returns:
    dict: Reduced data for each technique.
    list: List of DataFrames containing reduced data suitable for Altair plotting.

    """
    reduced_data = {}
    plot_data = []
    
    if isinstance(data, pd.DataFrame):
        prepared_data = data.copy()
        prepared_data.columns = prepared_data.columns.map(str)
    else:
        prepared_data = pd.DataFrame(data)
        prepared_data.columns = prepared_data.columns.map(str)

    if prepared_data.empty or prepared_data.shape[0] < 2 or prepared_data.shape[1] == 0:
        return reduced_data, plot_data

    prepared_data = prepared_data.apply(pd.to_numeric, errors="coerce")
    prepared_data = prepared_data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    non_constant_columns = prepared_data.columns[
        prepared_data.nunique(dropna=False) > 1
    ].tolist()
    if non_constant_columns:
        prepared_data = prepared_data[non_constant_columns]
    if prepared_data.empty or prepared_data.shape[1] == 0:
        return reduced_data, plot_data

    # Standardize the data after removing degenerate columns that can make
    # PCA/t-SNE numerically unstable without adding information.
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(prepared_data.to_numpy(dtype=float, copy=False))
    data_scaled = np.nan_to_num(data_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    column_variances = np.nanvar(data_scaled, axis=0)
    valid_variance_mask = np.isfinite(column_variances) & (column_variances > 0)
    if valid_variance_mask.any():
        data_scaled = data_scaled[:, valid_variance_mask]
    if data_scaled.shape[1] == 0:
        return reduced_data, plot_data

    unique_row_count = int(np.unique(np.round(data_scaled, decimals=12), axis=0).shape[0])
    has_variation = bool(np.any(np.nanvar(data_scaled, axis=0) > 0))

    def build_fallback_embedding():
        x = np.linspace(-1.0, 1.0, num=len(prepared_data), dtype=float)
        y = np.zeros(len(prepared_data), dtype=float)
        return np.column_stack([x, y])

    use_fallback_embedding = unique_row_count < 2 or not has_variation
    
    for method, param_list in methods_params.items():
        param_str = "; ".join([f"{key}: {value}" for key, value in param_list.items()])
        method_key = f"{method}_{param_str.replace('; ', '_').replace(': ', '_')}"

        if use_fallback_embedding:
            reduced_data[method_key] = build_fallback_embedding()
        else:
            if method == 'PCA':
                if data_scaled.shape[1] < 2:
                    reduced_data[method_key] = build_fallback_embedding()
                    plot_df = pd.DataFrame(reduced_data[method_key], columns=['Component 1', 'Component 2'])
                    plot_df['Method'] = method
                    plot_df['Parameter'] = param_str
                    plot_data.append(plot_df)
                    continue
                reducer = PCA(**param_list)
            elif method == 't-SNE':
                if len(data_scaled) <= 3:
                    reduced_data[method_key] = build_fallback_embedding()
                    plot_df = pd.DataFrame(reduced_data[method_key], columns=['Component 1', 'Component 2'])
                    plot_df['Method'] = method
                    plot_df['Parameter'] = param_str
                    plot_data.append(plot_df)
                    continue
                tsne_params = dict(param_list)
                tsne_params.setdefault("init", "random")
                tsne_params.setdefault("learning_rate", "auto")
                tsne_params["perplexity"] = min(
                    int(tsne_params.get("perplexity", 30)),
                    len(data_scaled) - 1,
                )
                reducer = TSNE(**tsne_params, random_state=42)
            elif method == 'UMAP':
                if UMAP is None:
                    raise ImportError("UMAP is not installed.")
                reducer = UMAP(**param_list)
            else:
                raise ValueError(f"Unsupported method: {method}")
        """
        elif method == 'Autoencoder':
            # Build and train Autoencoder
            autoencoder, encoder = build_autoencoder(data_scaled.shape[1])
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=32, verbose=0)
            autoencoder_result = encoder.predict(data_scaled)
            reduced_data[method_key] = autoencoder_result
            # Convert reduced data to DataFrame for Altair plotting
            plot_df = pd.DataFrame(autoencoder_result, columns=['Component 1', 'Component 2'])
            plot_df['Method'] = method
            plot_df['Parameter'] = param_str
            plot_data.append(plot_df)
            continue  # Skip further processing for Autoencoder
        """
        if not use_fallback_embedding:
            if method == 'PCA':
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="invalid value encountered in divide",
                        category=RuntimeWarning,
                    )
                    reduced_data[method_key] = reducer.fit_transform(data_scaled)
            else:
                reduced_data[method_key] = reducer.fit_transform(data_scaled)
        
        # Convert reduced data to DataFrame for Altair plotting
        plot_df = pd.DataFrame(reduced_data[method_key], columns=['Component 1', 'Component 2'])
        plot_df['Method'] = method
        plot_df['Parameter'] = param_str
        plot_data.append(plot_df)

    return reduced_data, plot_data

def grid_search_clustering(X, param_grid, scoring_metric):
    best_score = -np.inf
    best_model = None
    best_params = None
    
    for param_set in param_grid:
        estimator = param_set['estimator']
        params = param_set['params']
        
        for param_values in ParameterGrid(params):
            est = estimator(**param_values)
            est.fit(X)
            
            # Evaluate the clustering using the specified metric
            score = scoring_metric(X, est.labels_) if scoring_metric == silhouette_score else scoring_metric(X, est.labels_, metric='euclidean')
            
            # Update best model and parameters if current score is better
            if score > best_score:
                best_score = score
                best_model = est
                best_params = param_values
    
    return best_model, best_params, best_score

class ClassifierComparison:
    def __init__(
        self,
        X,
        y,
        test_size=0.2,
        random_state=42,
        X_test=None,
        y_test=None,
    ):
        if X_test is None or y_test is None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        else:
            self.X_train = pd.DataFrame(X).copy()
            self.X_test = pd.DataFrame(X_test).copy()
            self.y_train = pd.Series(y).copy()
            self.y_test = pd.Series(y_test).copy()
        self.results_df = None
        self.models = {}  # Dictionary to store trained models
        self.imputer = SimpleImputer(strategy='median')
        self.random_state = random_state
        self.class_labels = sorted(pd.Series(self.y_train).astype(str).unique().tolist())
        self.predictions_by_classifier = {}
        self.classification_reports = {}
        self.confusion_matrices = {}

        self.X_train = self.imputer.fit_transform(self.X_train)
        self.X_test = self.imputer.transform(self.X_test)
    
    def train_and_evaluate(self):
        classifiers = {
            "Logistic Regression": LogisticRegression(
                random_state=self.random_state, max_iter=50000
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(random_state=self.random_state),
            "KNeighbors Classifier": KNeighborsClassifier(
                n_neighbors=5
            ),
            "Gradient Boosting Classifier": GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100, 
                learning_rate=0.1
            ),
            "SVM": SVC(probability=True, random_state=self.random_state)
        }

        results = {"Classifier": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-score": []}

        for clf_name, clf in classifiers.items():
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            results["Classifier"].append(clf_name)
            results["Accuracy"].append(round(accuracy, 3))
            results["Precision"].append(round(precision, 3))
            results["Recall"].append(round(recall, 3))
            results["F1-score"].append(round(f1, 3))
            
            # Save trained model
            self.models[clf_name] = clf
            self.predictions_by_classifier[clf_name] = y_pred
            self.classification_reports[clf_name] = classification_report(
                self.y_test,
                y_pred,
                zero_division=0,
            )
            self.confusion_matrices[clf_name] = confusion_matrix(
                self.y_test,
                y_pred,
                labels=self.class_labels,
            )

        self.results_df = pd.DataFrame(results)
    
    def plot_performance_comparison(self, save_filename="default"):
        if self.results_df is None:
            print("No results to plot. Please run train_and_evaluate method first.")
            return

        # Convert results DataFrame to Altair-friendly format
        melted_df = self.results_df.melt(id_vars='Classifier', var_name='Metric', value_name='Score')
        
        # Calculate the mean score per classifier for sorting purposes
        classifier_means = melted_df.groupby('Classifier')['Score'].mean().reset_index()
        classifier_means = classifier_means.sort_values(by='Score', ascending=False)

        # Use the sorted classifier names as a list for ordering
        sorted_classifiers = classifier_means['Classifier'].tolist()

        # Create the chart using Altair
        chart = alt.Chart(melted_df).mark_bar().encode(
            x=alt.X('Classifier:N', title='Classifier', sort=sorted_classifiers),
            y=alt.Y('Score:Q', title='Score'),
            color='Metric:N',
            tooltip=melted_df.columns.to_list()
        ).properties(
            width="container",
            title='Performance Comparison of Classifiers'
        ).configure_axis(
            labelAngle=-45
        ).configure_legend(
            orient='top'
        )
        # Display the chart
        chart.save('models/supervisedChart' + save_filename + '.png', scale_factor=2.0)
        # return chart.to_dict()
        
        
    def save_models(self, save_path='models/', save_filename="default"):
        for clf_name, clf in self.models.items():
            joblib.dump(clf, f'{save_path}{clf_name}__{save_filename}.joblib')

    def save_diagnostics(self, save_path='models/', save_filename="default"):
        if self.results_df is None or self.results_df.empty:
            return

        diagnostics_dir = os.path.join(save_path, "diagnostics")
        os.makedirs(diagnostics_dir, exist_ok=True)

        best_row = self.results_df.sort_values(by="F1-score", ascending=False).iloc[0]
        best_classifier = best_row["Classifier"]
        safe_name = best_classifier.lower().replace(" ", "_")

        report_path = os.path.join(
            diagnostics_dir,
            f"{save_filename}_{safe_name}_classification_report.txt",
        )
        with open(report_path, "w") as fh:
            fh.write(self.classification_reports[best_classifier])

        confusion_df = pd.DataFrame(
            self.confusion_matrices[best_classifier],
            index=self.class_labels,
            columns=self.class_labels,
        )
        confusion_df.to_csv(
            os.path.join(
                diagnostics_dir,
                f"{save_filename}_{safe_name}_confusion_matrix.csv",
            )
        )

        confusion_long = confusion_df.reset_index().melt(
            id_vars="index",
            var_name="Predicted",
            value_name="Count",
        ).rename(columns={"index": "Actual"})
        heatmap = alt.Chart(confusion_long).mark_rect().encode(
            x=alt.X("Predicted:N", title="Predicted Class"),
            y=alt.Y("Actual:N", title="Actual Class"),
            color=alt.Color("Count:Q", title="Count"),
            tooltip=["Actual", "Predicted", "Count"],
        ).properties(
            width=420,
            height=320,
            title=f"Confusion Matrix: {best_classifier}",
        )
        heatmap.save(
            os.path.join(
                diagnostics_dir,
                f"{save_filename}_{safe_name}_confusion_matrix.png",
            ),
            scale_factor=2.0,
        )

        if hasattr(self.models[best_classifier], "predict_proba"):
            probabilities = self.models[best_classifier].predict_proba(self.X_test)
            y_test_bin = label_binarize(self.y_test, classes=self.class_labels)
            roc_rows = []
            for index, class_label in enumerate(self.class_labels):
                if y_test_bin[:, index].sum() == 0:
                    continue
                fpr, tpr, _ = roc_curve(y_test_bin[:, index], probabilities[:, index])
                auc_value = auc(fpr, tpr)
                for false_positive_rate, true_positive_rate in zip(fpr, tpr):
                    roc_rows.append(
                        {
                            "class_label": class_label,
                            "false_positive_rate": false_positive_rate,
                            "true_positive_rate": true_positive_rate,
                            "auc": auc_value,
                        }
                    )

            if roc_rows:
                roc_df = pd.DataFrame(roc_rows)
                roc_df.to_csv(
                    os.path.join(
                        diagnostics_dir,
                        f"{save_filename}_{safe_name}_roc_curves.csv",
                    ),
                    index=False,
                )
                roc_chart = alt.Chart(roc_df).mark_line().encode(
                    x=alt.X("false_positive_rate:Q", title="False Positive Rate"),
                    y=alt.Y("true_positive_rate:Q", title="True Positive Rate"),
                    color=alt.Color("class_label:N", title="Class"),
                    tooltip=["class_label", "auc"],
                ).properties(
                    width=420,
                    height=320,
                    title=f"ROC Curves: {best_classifier}",
                )
                roc_chart.save(
                    os.path.join(
                        diagnostics_dir,
                        f"{save_filename}_{safe_name}_roc_curves.png",
                    ),
                    scale_factor=2.0,
                )

class ClassifierComparisonSemiSupervisedXX:
    def __init__(self, X_labeled, y_labeled, X_unlabeled, test_size=0.2, random_state=42):
        # Split labeled data for training and evaluation
        self.X_train_labeled, self.X_test, self.y_train_labeled, self.y_test = train_test_split(
            X_labeled, y_labeled, test_size=test_size, random_state=random_state)
        self.X_unlabeled = X_unlabeled
        self.results_df = None
        self.models = {}  # Dictionary to store trained models
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_labeled = self.scaler.fit_transform(self.X_train_labeled)
        self.X_test = self.scaler.transform(self.X_test)
        self.X_unlabeled = self.scaler.transform(self.X_unlabeled)
    
    def train_and_evaluate(self):
        # random_state=42,
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=50000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "KNeighbors Classifier": KNeighborsClassifier(n_neighbors=5),
            "Gradient Boosting Classifier": GradientBoostingClassifier( n_estimators=100, learning_rate=0.1),
            "SVM": SVC(probability=True)  # Ensure probability=True for SVM
        }

        results = {"Classifier": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-score": []}

        # Combine labeled and unlabeled data
        X_combined = np.vstack((self.X_train_labeled, self.X_unlabeled))
        
        # Ensure y_combined is 1-dimensional by converting DataFrame to NumPy array and then raveling
        y_combined = np.concatenate((self.y_train_labeled.ravel(), -1 * np.ones(len(self.X_unlabeled), dtype=int)))

        for clf_name, clf in classifiers.items():
            # Semi-supervised learning with SelfTrainingClassifier
            self_training_clf = SelfTrainingClassifier(clf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self_training_clf.fit(X_combined, y_combined)
            y_pred = self_training_clf.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            results["Classifier"].append(clf_name)
            results["Accuracy"].append(round(accuracy, 3))
            results["Precision"].append(round(precision, 3))
            results["Recall"].append(round(recall, 3))
            results["F1-score"].append(round(f1, 3))
            
            # Save trained model
            self.models[clf_name] = self_training_clf
        self.results_df = pd.DataFrame(results)
        
    def plot_performance_comparison(self, save_filename="default"):
        if self.results_df is None:
            print("No results to plot. Please run train_and_evaluate method first.")
            return
        
        # Convert results DataFrame to Altair-friendly format
        melted_df = self.results_df.melt(id_vars='Classifier', var_name='Metric', value_name='Score')

        # Create the chart using Altair
        chart = alt.Chart(melted_df).mark_line().encode(
            x=alt.X('Classifier:N', title='Classifier', sort=alt.EncodingSortField(field='Score', order='descending')),
            y=alt.Y('Score:Q', title='Score'),
            color='Metric:N'
        ).properties(
            width="container",
            title='Performance Comparison of Classifiers'
        ).configure_axis(
            labelAngle=-45
        ).configure_legend(
            orient='top'
        )

        # Display the chart
        chart.save('models/semi-supervised/' + save_filename + '.png', scale_factor=2.0)
    
    def save_models(self, save_path='models/semi-supervised/', save_filename="default"):
        for clf_name, clf in self.models.items():
            joblib.dump(clf, f'{save_path}{clf_name}__{save_filename}.joblib')

 
 
import warnings
from sklearn.exceptions import UndefinedMetricWarning

class ClassifierComparisonSemiSupervised:
    def __init__(self, X_labeled, y_labeled, X_unlabeled, test_size=0.2, random_state=42):
        # Split labeled data for training and evaluation
        self.X_train_labeled, self.X_test, self.y_train_labeled, self.y_test = train_test_split(
            X_labeled, y_labeled, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_labeled,
        )
        self.X_unlabeled = X_unlabeled
        self.results_df = None
        self.models = {}  # Dictionary to store trained models
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoder = LabelEncoder()
        self.random_state = random_state

        self.y_train_labeled = self.label_encoder.fit_transform(np.asarray(self.y_train_labeled))
        self.y_test = self.label_encoder.transform(np.asarray(self.y_test))
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_labeled = self.imputer.fit_transform(self.X_train_labeled)
        self.X_test = self.imputer.transform(self.X_test)
        self.X_unlabeled = self.imputer.transform(self.X_unlabeled)

        self.X_train_labeled = self.scaler.fit_transform(self.X_train_labeled)
        self.X_test = self.scaler.transform(self.X_test)
        self.X_unlabeled = self.scaler.transform(self.X_unlabeled)
    
    def train_and_evaluate(self):
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=50000, random_state=self.random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "KNeighbors Classifier": KNeighborsClassifier(n_neighbors=5),
            "Gradient Boosting Classifier": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=self.random_state,
            ),
            "SVM": SVC(probability=True, max_iter=10000, random_state=self.random_state)
        }

        results = {"Classifier": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-score": []}

        # Combine labeled and unlabeled data
        X_combined = np.vstack((self.X_train_labeled, self.X_unlabeled))
        
        # Ensure y_combined is 1-dimensional by converting DataFrame to NumPy array and then raveling
        y_combined = np.concatenate((self.y_train_labeled.ravel(), -1 * np.ones(len(self.X_unlabeled), dtype=int)))
        X_combined, y_combined = shuffle(
            X_combined,
            y_combined,
            random_state=self.random_state,
        )

        # Check if there are unlabeled samples
        if np.sum(y_combined == -1) == 0:
            raise ValueError("Semi-supervised training requires at least one unlabeled sample.")

        for clf_name, clf in classifiers.items():
            # Semi-supervised learning with SelfTrainingClassifier
            self_training_clf = SelfTrainingClassifier(clf)  # Adjust parameters as needed
            self_training_clf.fit(X_combined, y_combined)
            # Predict on the test set
            y_pred = self_training_clf.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            
            # Store results
            results["Classifier"].append(clf_name)
            results["Accuracy"].append(round(accuracy, 3))
            results["Precision"].append(round(precision, 3))
            results["Recall"].append(round(recall, 3))
            results["F1-score"].append(round(f1, 3))
            # Save trained model
            self.models[clf_name] = self_training_clf
        self.results_df = pd.DataFrame(results)
    
    def plot_performance_comparison(self, save_filename="default"):
        if self.results_df is None:
            print("No results to plot. Please run train_and_evaluate method first.")
            return
        
        # Convert results DataFrame to Altair-friendly format
        melted_df = self.results_df.melt(id_vars='Classifier', var_name='Metric', value_name='Score')

        # Create the chart using Altair
        chart = alt.Chart(melted_df).mark_line().encode(
            x=alt.X('Classifier:N', title='Classifier', sort=alt.EncodingSortField(field='Score', order='descending')),
            y=alt.Y('Score:Q', title='Score'),
            color='Metric:N'
        ).properties(
            width="container",
            title='Performance Comparison of Classifiers'
        ).configure_axis(
            labelAngle=-45
        ).configure_legend(
            orient='top'
        )
        
        self.results_df.to_csv('models/semi-supervised/' + save_filename + '_main.csv')
        # Display the chart
        chart.save('models/semi-supervised/' + save_filename + '.png', scale_factor=2.0)
    
    def save_models(self, save_path='models/semi-supervised/', save_filename="default"):
        for clf_name, clf in self.models.items():
            joblib.dump(clf, f'{save_path}{clf_name}__{save_filename}.joblib')


        
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score
def plot_multi_class_roc(clf, X_test, y_test, classes):
    """
    Plots the ROC curve for multi-class classification.
    
    Parameters:
    - clf: Trained classifier
    - X_test: Test feature set
    - y_test: True labels for test set
    - classes: List of class labels
    """
    # Binarize the output for ROC calculation
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)
    
    # Get probabilities for the positive class
    y_score = clf.predict_proba(X_test)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'navy', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Class Classification')
    plt.legend(loc="lower right")
    plt.show()


class DataImputation:
    def __init__(self, strategy='mean', n_neighbors=5):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        if strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=n_neighbors)
        else:
            self.imputer = SimpleImputer(strategy=strategy)
    
    def fit_transform(self, X):
        return self.imputer.fit_transform(X)


class ClassifierComparisonSemiSupervisedOutlier:
    def __init__(self, X_labeled, y_labeled, X_unlabeled, test_size=0.2, random_state=42):
        self.X_train_labeled, self.X_test, self.y_train_labeled, self.y_test = train_test_split(
            X_labeled, y_labeled, test_size=test_size, random_state=random_state)
        self.X_unlabeled = X_unlabeled
        self.results_df = None
        self.models = {}  # Dictionary to store trained models
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_labeled = self.scaler.fit_transform(self.X_train_labeled)
        self.X_test = self.scaler.transform(self.X_test)
        self.X_unlabeled = self.scaler.transform(self.X_unlabeled)
    
    def train_and_evaluate(self):
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=50000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "KNeighbors Classifier": KNeighborsClassifier(n_neighbors=5),
            "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),
            "SVM": SVC(probability=True)  # Ensure probability=True for SVM
        }

        results = {"Classifier": [], "Accuracy": [], "Precision": [], "Recall": [], "F1-score": []}

        # Combine labeled and unlabeled data
        X_combined = np.vstack((self.X_train_labeled, self.X_unlabeled))
        
        # Ensure y_combined is 1-dimensional by converting DataFrame to NumPy array and then raveling
        y_combined = np.concatenate((self.y_train_labeled.to_numpy().ravel(), -1 * np.ones(len(self.X_unlabeled), dtype=int)))
        
        # Debugging: Verify the combined dataset
        print(f"X_combined shape: {X_combined.shape}")
        print(f"y_combined shape: {y_combined.shape}")
        print(f"Number of unlabeled samples in y_combined: {np.sum(y_combined == -1)}")

        for clf_name, clf in classifiers.items():
            # Semi-supervised learning with SelfTrainingClassifier
            self_training_clf = SelfTrainingClassifier(clf)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self_training_clf.fit(X_combined, y_combined)
            y_pred = self_training_clf.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred, average='weighted', zero_division=0)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

            results["Classifier"].append(clf_name)
            results["Accuracy"].append(round(accuracy, 3))
            results["Precision"].append(round(precision, 3))
            results["Recall"].append(round(recall, 3))
            results["F1-score"].append(round(f1, 3))
            
            # Save trained model
            self.models[clf_name] = self_training_clf

        self.results_df = pd.DataFrame(results)
    
    def detect_anomalies(self, method="isolation_forest", X=None):
        if method == "isolation_forest":
            model = IsolationForest(contamination=0.1, random_state=42)
        elif method == "one_class_svm":
            model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        elif method == "lof":
            model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
        
        model.fit(self.X_train_labeled)  # Fit on labeled (normal) data
        
        if X is None:
            X = self.X_unlabeled
        
        if method == "lof":
            anomalies = model.predict(X)
        else:
            anomalies = model.predict(X)
        
        anomaly_indices = np.where(anomalies == -1)[0]
        return anomaly_indices, anomalies
    
    def evaluate_anomaly_detection(self, ground_truth_anomalies):
        methods = ["isolation_forest", "one_class_svm", "lof"]
        results = {"Method": [], "Precision": [], "Recall": [], "F1-score": []}
        
        for method in methods:
            anomaly_indices, anomalies = self.detect_anomalies(method=method)
            
            # Create ground truth labels for the evaluation
            y_true = np.isin(range(len(self.X_unlabeled)), ground_truth_anomalies).astype(int)
            y_pred = np.isin(range(len(self.X_unlabeled)), anomaly_indices).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results["Method"].append(method)
            results["Precision"].append(round(precision, 3))
            results["Recall"].append(round(recall, 3))
            results["F1-score"].append(round(f1, 3))
            
            print(f"Anomaly detection method: {method}")
            print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
            print(classification_report(y_true, y_pred, zero_division=0))
        
        self.anomaly_results_df = pd.DataFrame(results)
        print(self.anomaly_results_df)
    
    def plot_performance_comparison(self, save_filename="default"):
        if self.results_df is None:
            print("No results to plot. Please run train_and_evaluate method first.")
            return
        
        # Convert results DataFrame to Altair-friendly format
        melted_df = self.results_df.melt(id_vars='Classifier', var_name='Metric', value_name='Score')

        # Create the chart using Altair
        chart = alt.Chart(melted_df).mark_line().encode(
            x=alt.X('Classifier:N', title='Classifier', sort=alt.EncodingSortField(field='Score', order='descending')),
            y=alt.Y('Score:Q', title='Score'),
            color='Metric:N'
        ).properties(
            width="container",
            title='Performance Comparison of Classifiers'
        ).configure_axis(
            labelAngle=-45
        ).configure_legend(
            orient='top'
        )

        # Display the chart
        chart.save(f'models/semi-supervised/{save_filename}.png', scale_factor=2.0)
    
    def save_models(self, save_path='models/semi-supervised/', save_filename="default"):
        for clf_name, clf in self.models.items():
            joblib.dump(clf, f'{save_path}{clf_name}__{save_filename}.joblib')


class DataNormalization:
    def __init__(self, method='min_max'):
        self.method = method
        if method == 'min_max':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
    
    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

class ColumnRemoval:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def fit_transform(self, X):
        cols_to_remove = X.columns[X.isnull().mean() > self.threshold]
        return X.drop(columns=cols_to_remove)

class DimensionalityReduction:
    def __init__(self, method='pca', n_components=None, **kwargs):
        if method == 'pca':
            self.model = PCA(n_components=n_components, **kwargs)
        elif method == 'incremental_pca':
            self.model = IncrementalPCA(n_components=n_components, **kwargs)
        elif method == 'tsne':
            self.model = TSNE(n_components=n_components, **kwargs)
        elif method == 'umap':
            self.model = UMAP(n_components=n_components, **kwargs)
    
    def fit_transform(self, X):
        return self.model.fit_transform(X)

class Unsupervised:
    def __init__(self, algorithm='kmeans', param_grid=None, **kwargs):
        if algorithm == 'kmeans':
            self.model = KMeans(**kwargs)
        elif algorithm == 'dbscan':
            self.model = DBSCAN(**kwargs)
        elif algorithm == 'agglomerative':
            self.model = AgglomerativeClustering(**kwargs)
        
        self.param_grid = param_grid
    
    def fit(self, X):
        if self.param_grid:
            grid_search = GridSearchCV(self.model, self.param_grid)
            grid_search.fit(X)
            self.model = grid_search.best_estimator_
        else:
            self.model.fit(X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X):
        if hasattr(self.model, 'labels_'):
            labels = self.model.labels_
        elif hasattr(self.model, 'predict'):
            labels = self.model.predict(X)
        else:
            raise ValueError("Model does not support label prediction.")
        
        if len(set(labels)) < 2:
            return 0  # Return 0 silhouette score if there is only one cluster
        
        return silhouette_score(X, labels)

class Supervised:
    def __init__(self, algorithm='random_forest', param_grid=None, **kwargs):
        if algorithm == 'random_forest':
            self.model = RandomForestClassifier(**kwargs)
        elif algorithm == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**kwargs)
        elif algorithm == 'svm':
            self.model = SVC(**kwargs)
        elif algorithm == 'knn':
            self.model = KNeighborsClassifier(**kwargs)
        
        if param_grid:
            self.model = GridSearchCV(self.model, param_grid)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return {
            "ground_truth": y_true,
            "predicted": y_pred
        }

class ClusterPlotter:
    def __init__(self, data, x_col='x', y_col='y', cluster_col='cluster', tooltip=[], width=400, height=400):
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.cluster_col = cluster_col
        self.tooltip = tooltip
        self.width = width
        self.height = height
    
    def plot(self):
        chart = alt.Chart(self.data).mark_circle().encode(
            x=self.x_col,
            y=self.y_col,
            color=alt.Color(self.cluster_col + ':N', scale=alt.Scale(scheme='category20')),
            tooltip=self.tooltip
        ).properties(
            width=self.width,
            height=self.height
        )
        
        return chart
     
def select_features_using_decision_tree(dataframe, target_column, num_features=None):
    """Select top features using a Decision Tree based on feature importance.
    
    Args:
        dataframe (pd.DataFrame): The input data frame.
        target_column (str): The name of the target column.
        num_features (int, optional): The number of top features to select. If None, select all features.

    Returns:
        pd.DataFrame: Data frame with selected features.
        list: List of selected feature names.
    """
    dataframe = dataframe.copy()
    dataframe.columns = pd.Index([str(column) for column in dataframe.columns], dtype="object")
    target_column = str(target_column)
    X = dataframe.drop(columns=[target_column]).copy()
    y = dataframe[target_column]

    # SQLAlchemy-reflected/pandas-generated frames can carry quoted_name objects
    # alongside plain strings. scikit-learn expects a consistent feature-name type.
    X.columns = pd.Index([str(column) for column in X.columns], dtype="object")
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train.columns = pd.Index([str(column) for column in X_train.columns], dtype="object")
    X_test.columns = pd.Index([str(column) for column in X_test.columns], dtype="object")

    # Train a decision tree classifier
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get feature importances
    feature_importances = model.feature_importances_
    
    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    })
    
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # Select top features
    if num_features is not None:
        top_features = feature_importance_df.head(num_features)['feature'].tolist()
    else:
        top_features = feature_importance_df['feature'].tolist()
        
    
    # Visualize feature importances
    chart_data = feature_importance_df
    feature_mapping = {
        'subunit_segments': 'Subunit Segments',
        'rcsentinfo_polymer_molecular_weight_maximum': 'Polymer Molecular Weight Maximum',
        'rcsentinfo_polymer_molecular_weight_minimum': 'Polymer Molecular Weight Minimum',
        'rcsentinfo_molecular_weight': 'Molecular Weight',
        'thickness': 'Thickness',
        'tilt': 'Tilt'
    }
    chart_data['feature'] = feature_importance_df['feature'].replace(feature_mapping)

    bar_chart = alt.Chart(feature_importance_df).mark_bar().encode(
        x=alt.X('importance', title='Importance Score'),
        y=alt.Y('feature', title='Features', sort='-x'),
        tooltip=['feature', 'importance']
    ).properties(
        width=600,
        height=400,
        title='Top Feature Importances in Decision Tree'
    )
    bar_chart.save('./models/feature_selection.png', scale_factor=2.0)
    if os.getenv("ML_ENABLE_SHAP", "false").lower() == "true":
        plot_shap_values(model, X, filename='./models/shap_bar_plot.png', dpi=300)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig("models/shap_summary_plot.png", bbox_inches='tight', dpi=200)
        plt.close()

    
    selected_features_df = dataframe[top_features]
    
    return selected_features_df, top_features


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_shap_values(model, X, filename='./models/shap_plot.png', dpi=500):
    if shap is None:
        raise ImportError("SHAP is not installed in this environment.")
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # Define a custom colormap using LinearSegmentedColormap
    colors = [(0, "green"), (0.5, "white"), (1, "purple")]  # Define blue-to-white-to-red gradient
    cmap = LinearSegmentedColormap.from_list("blue_white_red", colors)
    plt.figure(figsize=(10, 5))
    # Plotting SHAP values
    shap.summary_plot(
        shap_values.values, 
        features=X, 
        plot_type="dot", 
        feature_names=X.columns,
        cmap=cmap,
        show=False
    )
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"SHAP summary plot saved as {filename}")
    
    
def select_features_using_random_forest(dataframe, target_column, num_features=None):
    """Select top features using a Random Forest based on feature importance.
    
    Args:
        dataframe (pd.DataFrame): The input data frame.
        target_column (str): The name of the target column.
        num_features (int, optional): The number of top features to select. If None, select all features.

    Returns:
        pd.DataFrame: Data frame with selected features.
        list: List of selected feature names.
    """
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    feature_importances = model.feature_importances_
    
    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    })
    
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # Select top features
    if num_features is not None:
        top_features = feature_importance_df.head(num_features)['feature'].tolist()
    else:
        top_features = feature_importance_df['feature'].tolist()
    
    selected_features_df = dataframe[top_features]
    
    return selected_features_df, top_features
