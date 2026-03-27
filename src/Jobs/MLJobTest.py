import os, sys
sys.path.append(os.getcwd())
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
import umap
import altair as alt
from src.Dashboard.services import get_tables_as_dataframe
from app import app

class SupervisedLearning:
    def __init__(self, table_names, model_save_dir='models'):
        self.table_names = table_names
        self.df = None
        self.X = None
        self.y = None
        self.results = {}
        self.label_encoder = LabelEncoder()
        self.model_metrics = {}  # Initialize model metrics dictionary
        self.self_training_metrics = {}  # Separate metrics for SelfTraining
        self.classifiers = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Support Vector Classifier': SVC(random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'KNeighbors Classifier': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)  # Create directory if it doesn't exist

    def load_data(self):
        with app.app_context():
            all_data = get_tables_as_dataframe(self.table_names, "pdb_code")
        self.df = all_data[["thickness", "subunit_segments", "tilt", "pdb_code", "membrane_topology_in", "membrane_topology_out", "group"]]
        self.df = self.df.dropna()
        
        # Filter out specific PDB codes
        # include_pdb_codes = ["1PFO", "1B12"]
        exclude_pdb_codes = [
            "1PFO", "1FDM", "1AFO", "2CPB", "1B12", "1GOS", "1MT5", "1KN9", "1OJA", "1MZT",
            "1O5W", "1PJF", "1UUM", "1T7D", "2BXR", "1YGM", "1ZLL", "2GMH", "2J58", "2HAC",
            "2J7A", "2OLV", "2JO1", "2OQO", "2QCU", "2PRM", "2Z5X", "2VQG", "2JWA", "2K1L",
            "3BKD", "2RLF", "3HYW", "3I65", "2KNC", "3HD7", "2KOG", "2KIH", "2KIX", "3VMA",
            "3JQO", "3NSJ", "2KSJ", "2KS1", "2K9Y", "2L35", "2L0J", "3LBW", "3ML3", "3PRW",
            "3P1L", "3Q7M", "2YH3", "3LIM", "2KPF", "2L9U", "2LJB", "2KYV", "3VMT", "3Q54",
            "2LCX", "2LZL", "2M8R", "2LY0", "2M3B", "4LXJ", "4HSC", "4CDB", "2MFR", "2M59",
            "4TSY", "5EH4", "4WOL", "4QKC", "5B49", "5IMW", "5IMY", "2MIC", "2N2A", "5HK1",
            "5LY6", "5NUO", "5LV6", "5JOO", "6BFG", "6MLU", "6DLW", "6H03", "6F2D", "6HJR",
            "6E10", "6BKK", "6NYF", "6MQU", "6JXR", "6MTI", "6MJH", "6S3S", "6S3R", "7K7A",
            "7BV6", "6NV1", "6PVR", "7K3G", "6Z0G", "7OFM", "7AGX", "7OKN", "7KN0", "7LQ6",
            "6LKD", "7RSL", "8A1D", "7WSO", "7XT6", "7XQ8", "7W2B", "7FJD", "7VU5", "7MPA",
            "8GI1"
        ]
        
        # [
        #     "1PFO", "1B12", "1GOS", "1MT5", "1KN9", "1OJA", "1O5W", "1UUM", "1T7D", "2BXR",
        #     "1YGM", "2GMH", "2OLV", "2OQO", "2QCU", "2PRM", "2Z5X", "2VQG", "3HYW", "3I65",
        #     "3VMA", "3NSJ", "3ML3", "3PRW", "3P1L", "3Q7M", "2YH3", "3LIM", "3VMT", "3Q54",
        #     "2YMK", "2LOU", "4LXJ", "4HSC", "4CDB", "4P6J", "4TSY", "5B49", "5IMW", "5IMY",
        #     "5JYN", "5LY6", "6BFG", "6MLU", "6DLW", "6H03", "6NYF", "6MTI", "7OFM", "7LQ6",
        #     "7RSL", "8A1D", "7QAM"
        # ]
        self.df = self.df[~self.df['pdb_code'].isin(exclude_pdb_codes)]
        print("Hope the real-life test data is not part?")
        print(self.df[self.df["pdb_code"].isin(exclude_pdb_codes)])
        # Encode categorical columns
        self.df['membrane_topology_in'] = self.label_encoder.fit_transform(self.df['membrane_topology_in'])
        self.df['membrane_topology_out'] = self.label_encoder.fit_transform(self.df['membrane_topology_out'])
        self.df['group'] = self.label_encoder.fit_transform(self.df['group'])
        # self.df.to_csv("dataDF.csv")
        # Prepare features (X) and target (y)
        self.X = self.df.drop(columns=['group', 'pdb_code']).values
        self.y = self.df['group'].values


    def save_model(self, model, model_name):
        model_file = os.path.join(self.model_save_dir, f"{model_name.replace(' ', '_')}_model.joblib")
        joblib.dump(model, model_file)
        print(f"Model {model_name} saved to {model_file}")
        
    def semi_supervised_setup(self):
        # Introduce unlabeled data
        y_unlabeled = np.copy(self.y)
        y_unlabeled[:5] = -1  # Mask first 5 labels as "unlabeled"
        return y_unlabeled

    def evaluate_classifiers(self):
        y_unlabeled = self.semi_supervised_setup()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize results storage
        for name in self.classifiers.keys():
            self.results[name] = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

        # Perform cross-validation
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = y_unlabeled[train_index], self.y[test_index]
            
            # Split the data into labeled and unlabeled
            labeled_mask = y_train != -1
            unlabeled_mask = y_train == -1
            
            X_labeled = X_train[labeled_mask]
            y_labeled = y_train[labeled_mask]
            X_unlabeled = X_train[unlabeled_mask]
            
            for name, classifier in self.classifiers.items():
                # Train the model on the labeled data
                classifier.fit(X_labeled, y_labeled)
                
                # Predict pseudo-labels for the unlabeled data
                pseudo_labels = classifier.predict(X_unlabeled)
                
                # Combine labeled data with pseudo-labeled data
                X_combined = np.vstack((X_labeled, X_unlabeled))
                y_combined = np.hstack((y_labeled, pseudo_labels))
                
                # Retrain the model on the combined data
                classifier.fit(X_combined, y_combined)
                
                # Evaluate the model on the test set
                y_pred = classifier.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Store metrics
                self.results[name]['accuracy'].append(accuracy)
                self.results[name]['precision'].append(precision)
                self.results[name]['recall'].append(recall)
                self.results[name]['f1_score'].append(f1)
                # Store metrics in the dictionary
                self.model_metrics[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                
                # Save the trained Self-Training model
                self.save_model(classifier, f"Self_Training_Manual_{name}")
                
    def train_supervised_classifiers(self):
        """ Train classifiers using fully labeled data (supervised learning only). """
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize results storage
        supervised_metrics = {}

        for name in self.classifiers.keys():
            self.results[name] = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}

        # Perform cross-validation
        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]  # Use all labeled data

            for name, classifier in self.classifiers.items():
                # Train the model on the labeled data
                classifier.fit(X_train, y_train)
                
                # Evaluate the model on the test set
                y_pred = classifier.predict(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Store metrics
                self.results[name]['accuracy'].append(accuracy)
                self.results[name]['precision'].append(precision)
                self.results[name]['recall'].append(recall)
                self.results[name]['f1_score'].append(f1)

                # Calculate and store average metrics across folds
                supervised_metrics[name] = {
                    'accuracy': np.mean(self.results[name]['accuracy']),
                    'precision': np.mean(self.results[name]['precision']),
                    'recall': np.mean(self.results[name]['recall']),
                    'f1_score': np.mean(self.results[name]['f1_score'])
                }

                # Save the trained model
                self.save_model(classifier, f"Supervised_{name}")

        # Print the results for supervised learning
        self.display_supervised_results(supervised_metrics)

        # Optionally save metrics to a CSV file
        self.save_supervised_metrics_to_csv(supervised_metrics, './models/supervised_metrics.csv')

    def display_supervised_results(self, supervised_metrics):
        """Display the metrics for supervised classifiers."""
        for name, metrics in supervised_metrics.items():
            print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, "
                  f"Recall: {metrics['recall']:.4f}, "
                  f"F1 Score: {metrics['f1_score']:.4f}")
            
    def display_results(self):
        for name in self.results.keys():
            print(f"{name} - Accuracy: {np.mean(self.results[name]['accuracy']):.4f}, "
                  f"Precision: {np.mean(self.results[name]['precision']):.4f}, "
                  f"Recall: {np.mean(self.results[name]['recall']):.4f}, "
                  f"F1 Score: {np.mean(self.results[name]['f1_score']):.4f}")

    def save_metrics_to_csv(self, file_name):
        metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

        for model_name, metrics in self.model_metrics.items():
            metrics_df = pd.concat([metrics_df, pd.DataFrame([{
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            }])], ignore_index=True)

        # Save to CSV
        metrics_df.to_csv(file_name, index=False)
        print(f"Metrics saved to {file_name}")
        
    def save_supervised_metrics_to_csv(self, supervised_metrics, file_name):
        """Save the supervised learning metrics to a CSV file."""
        metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

        for model_name, metrics in supervised_metrics.items():
            metrics_df = pd.concat([metrics_df, pd.DataFrame([{
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            }])], ignore_index=True)

        # Save to CSV
        metrics_df.to_csv(file_name, index=False)
        print(f"Supervised learning metrics saved to {file_name}")
        
    def save_semi_supervised_metrics_to_csv(self, file_name):
        metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

        for model_name, metrics in self.self_training_metrics.items():
            metrics_df = pd.concat([metrics_df, pd.DataFrame([{
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            }])], ignore_index=True)

        # Save to CSV
        metrics_df.to_csv(file_name, index=False)
        print(f"Semi-supervised metrics saved to {file_name}")
        

    def dimensionality_reduction(self):
        # Perform dimensionality reduction for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(self.X)

        umap_model = umap.UMAP(n_components=2, random_state=42)
        X_umap = umap_model.fit_transform(self.X)

        return X_pca, X_tsne, X_umap

    def visualize_2d(self, X, y, method_name, save_path=None):
        # Create a DataFrame for visualization
        vis_df = pd.DataFrame(X, columns=['Component 1', 'Component 2'])
        vis_df['Group'] = y

        # Create Altair scatter plot
        chart = alt.Chart(vis_df).mark_circle(size=60).encode(
            x='Component 1:Q',
            y='Component 2:Q',
            color=alt.Color('Group:O', scale=alt.Scale(scheme='category10')),
            tooltip=['Group:O']
        ).properties(
            title=f'2D Visualization using {method_name}'
        ).interactive()
        
        # Save chart to specified path if provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directories if they don't exist
            chart.save(save_path)
            print(f"{method_name} chart saved to {save_path}")

        return chart

    def train_self_training_classifier(self):
        y_unlabeled = self.semi_supervised_setup()
        
        # Initialize a new set of metrics for SelfTraining
        self.self_training_metrics = {}
        
        for name, classifier in self.classifiers.items():
            # Create a SelfTrainingClassifier
            self_training_model = SelfTrainingClassifier(classifier)

            # Fit the model
            self_training_model.fit(self.X, y_unlabeled)

            # Predict on the original labels
            y_pred = self_training_model.predict(self.X)

            # Store metrics
            accuracy = accuracy_score(self.y, y_pred)
            precision = precision_score(self.y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y, y_pred, average='weighted', zero_division=0)

            # Store results in the self-training metrics dictionary
            self.self_training_metrics["semi-supervised-function "+name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Save the trained Self-Training model
            self.save_model(self_training_model, f"Self_Training_{name}")

    def train_label_spreading_classifier(self):
        y_unlabeled = self.semi_supervised_setup()
        label_spread = LabelSpreading(kernel='knn', n_neighbors=5)
        # Train Label Spreading with the classifier's predictions
        label_spread.fit(self.X, y_unlabeled)
        
        # Evaluate on the original labels
        y_pred = label_spread.predict(self.X)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y, y_pred)
        precision = precision_score(self.y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y, y_pred, average='weighted', zero_division=0)
        
        # Store metrics
        self.self_training_metrics["label_spread"] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        # Save the trained Label Spreading model
        self.save_model(label_spread, f"Label_Spreading")

# Usage
table_names = ['membrane_proteins', 'membrane_protein_opm']
sl_model = SupervisedLearning(table_names)
sl_model.load_data()
sl_model.evaluate_classifiers()
sl_model.display_results()
sl_model.save_metrics_to_csv("./models/model_metrics.csv")

# Supervised Learning
sl_model.train_supervised_classifiers()

# Train SelfTrainingClassifier
# sl_model.train_self_training_classifier()
# Train LabelSpreading
# sl_model.train_label_spreading_classifier()
# sl_model.display_results()
# sl_model.save_semi_supervised_metrics_to_csv("./models/self_training_metrics.csv")

# Dimensionality reduction for visualization
X_pca, X_tsne, X_umap = sl_model.dimensionality_reduction()

# Visualizations
model_folder = './models/'
pca_save_path = os.path.join(model_folder, 'pca_chart.json')
tsne_save_path = os.path.join(model_folder, 'tsne_chart.json')
umap_save_path = os.path.join(model_folder, 'umap_chart.json')

# Visualizations and save charts
pca_chart = sl_model.visualize_2d(X_pca, sl_model.y, 'PCA', pca_save_path)
tsne_chart = sl_model.visualize_2d(X_tsne, sl_model.y, 't-SNE', tsne_save_path)
umap_chart = sl_model.visualize_2d(X_umap, sl_model.y, 'UMAP', umap_save_path)

# Display charts (if in a Jupyter notebook or similar environment)
# pca_chart.show()
# tsne_chart.show()
# umap_chart.show()

