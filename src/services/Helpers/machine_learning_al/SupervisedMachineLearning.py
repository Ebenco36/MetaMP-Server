from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import shap

class SupervisedLearningAlgorithms:
    def __init__(self):
        pass

    def run_classification_algorithm(self, algorithm, **kwargs):
        clustering = algorithm(**kwargs)
        complete_path = ""
        labels = clustering.fit_predict(self.X)
        params = clustering.get_params(deep=True)
        return labels, params, complete_path
        

    def train_decision_tree(self, X_train, y_train):
        labels, params, path = self.run_classification_algorithm(DecisionTreeClassifier, x = X_train, y = y_train)
        return self.X, params, path

    def train_random_forest(self, X_train, y_train):
        labels, params, path = self.run_classification_algorithm(RandomForestClassifier, x = X_train, y = y_train)
        return self.X, params, path

    def train_svm(self, X_train, y_train):
        labels, params, path = self.run_classification_algorithm(SVC, x = X_train, y = y_train)
        return self.X, params, path

    def train_logistic_regression(self, X_train, y_train):
        labels, params, path = self.run_classification_algorithm(LogisticRegression, x = X_train, y = y_train)
        return self.X, params, path

    def train_gradient_boosting(self, X_train, y_train):
        labels, params, path = self.run_classification_algorithm(GradientBoostingClassifier, x = X_train, y = y_train)
        return self.X, params, path

    def explain(self, X, model):
        if self.explainer is None:
            self.explainer = shap.Explainer(model)  # You can choose any trained model here
        
        return self.explainer(X)


# Perform predictions and explanations
# predictions_rf = algorithms.random_forest.predict(X_test)
# explanations_rf = algorithms.explain(X_test)  # Using SHAP for Random Forest explanations
