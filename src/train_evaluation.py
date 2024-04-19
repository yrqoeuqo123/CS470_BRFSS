import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss, f1_score, cohen_kappa_score


# Load the sub-dataset
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

# Define models to train
models = {
    'LogisticRegression': LogisticRegression(max_iter=10000),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(),
    'XGBoost': GradientBoostingClassifier(),
    'MLP': MLPClassifier(max_iter=1000)
}

# Sub-datasets files
subdatasets = ['Sn', 'Sh', 'St', 'Sht']

# Store the best parameters for each model on each dataset
best_params = {
    'Sn': {
        'LogisticRegression': {'C': 0.001, 'penalty': 'l2'},
        'KNN': {'metric': 'manhattan', 'n_neighbors': 9, 'weights': 'uniform'},
        'DecisionTree': {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 5},
        'RandomForest': {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 200},
        'SVM': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
        'XGBoost': {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 150},
        'MLP': {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50,), 'solver': 'adam'}
    },
    'Sh': {
        'LogisticRegression': {'C': 0.01, 'penalty': 'l2'},
        'KNN': {'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'uniform'},
        'DecisionTree': {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10},
        'RandomForest': {'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200},
        'SVM': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'},
        'XGBoost': {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100},
        'MLP': {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'solver': 'adam'}

    },
    'St': {
        'LogisticRegression': {'C': 0.1, 'penalty': 'l2'},
        'KNN': {'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'uniform'},
        'DecisionTree': {'max_depth': 30, 'min_samples_leaf': 4, 'min_samples_split': 2},
        'RandomForest': {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100},
        'SVM': {'C': 10, 'gamma': 'scale', 'kernel': 'linear'},
        'XGBoost': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 150},
        'MLP': {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (50,), 'solver': 'adam'}

    },
    'Sht': {
        'LogisticRegression': {'C': 0.01, 'penalty': 'l2'},
        'KNN': {'metric': 'euclidean', 'n_neighbors': 9, 'weights': 'uniform'},
        'DecisionTree': {'max_depth': 20, 'min_samples_leaf': 4, 'min_samples_split': 10},
        'RandomForest': {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200},
        'SVM': {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'},
        'XGBoost': {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 150},
        'MLP': {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (100,), 'solver': 'adam'}
    }
}

# Train and evaluate models with best parameters
for dataset_name in subdatasets:
    print(f"\nProcessing {dataset_name} subdataset...")
    train_file = f"{dataset_name}_train.csv"
    test_file = f"{dataset_name}_test.csv"
    
    # Load train and test data
    train_data, test_data = load_data(train_file, test_file)
    
    # Extract features and target variable
    X_train = train_data.drop(columns=['Diabetes_012'])
    y_train = train_data['Diabetes_012']
    X_test = test_data.drop(columns=['Diabetes_012'])
    y_test = test_data['Diabetes_012']
    
    # Train and evaluate each model with best parameters
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")
        model_params = best_params[dataset_name][model_name]
        model.set_params(**model_params)
        model.fit(X_train, y_train)
        
        # Predict probabilities for cross-entropy loss
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        # Predictions for F1-score
        y_pred = model.predict(X_test)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        multiclass_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted') if y_proba is not None else None
        cohen_kappa = cohen_kappa_score(y_test, y_pred)
        categorical_loss = log_loss(y_test, y_proba) if y_proba is not None else None

        # Print metrics
        print(f"Test Accuracy: {test_accuracy}")
        print("Classification Report:")
        print(report)
        print(f"Weighted F1-Score: {weighted_f1}")
        print(f"Macro F1-Score: {macro_f1}")
        print(f"Micro F1-Score: {micro_f1}")
        if multiclass_auc is not None:
            print(f"Multiclass AUC-ROC: {multiclass_auc}")
        print(f"Cohen's Kappa: {cohen_kappa}")
        if categorical_loss is not None:
            print(f"Categorical Cross-Entropy Loss: {categorical_loss}")

print("All models have been trained and evaluated.")