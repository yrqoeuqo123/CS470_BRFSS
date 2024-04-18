import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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
results = []

# Train and evaluate models with grid search
def train_and_evaluate_with_gridsearch(model_name, model, param_grid, X_train, X_test, y_train, y_test):
    # Split the training data into a smaller subset for grid search
    X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42)
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_subset, y_train_subset)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

    
    results.append({
        'Dataset': dataset_name,
        'Model': model_name,
        'Best Parameters': best_params,
        'Best Training Accuracy': best_score,
        'Test Accuracy': test_accuracy,
        'Classification Report': report
    })

# Define hyperparameter grids for each model
param_grids = {
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l2']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'DecisionTree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale']
    },
    'XGBoost': {
        'n_estimators': [100, 150],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'solver': ['adam']
    }
}

# Loop to process each sub-dataset
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
    
    # Train and evaluate each model with grid search
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")
        param_grid = param_grids[model_name]
        train_and_evaluate_with_gridsearch(model_name, model, param_grid, X_train, X_test, y_train, y_test)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv('model_results.csv', index=False)

print("All models have been trained and evaluated.")
