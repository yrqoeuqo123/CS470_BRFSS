import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.layers import concatenate, GlobalMaxPooling1D
from sklearn.metrics import mean_squared_error, f1_score


def apply_pca(X, n_components=0.95):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

def generate_feature_interactions(X):
    # Calculate the number of two-way and three-way interactions
    n_features = X.shape[1]
    two_way_combinations = n_features * (n_features - 1) // 2
    three_way_combinations = n_features * (n_features - 1) * (n_features - 2) // 6

    # Allocate space for both two-way and three-way interactions
    interactions = np.zeros((X.shape[0], two_way_combinations + three_way_combinations))
    
    index = 0  # Initialize index for storing interactions
    # Calculate two-way interactions
    for i in range(n_features):
        for j in range(i + 1, n_features):
            interactions[:, index] = X[:, i] * X[:, j]
            index += 1
    
    # Calculate three-way interactions
    for i in range(n_features):
        for j in range(i + 1, n_features):
            for k in range(j + 1, n_features):
                interactions[:, index] = X[:, i] * X[:, j] * X[:, k]
                index += 1

    return interactions

def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def learn_vector_representations(model, X):
    return model.predict(X)

def convolutional_layers(input_layer):
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    flattened = Flatten()(pool2)
    return flattened

def build_convolutional_base(input_shape):
    input_layer = Input(shape=(input_shape, 1))

    # Define multiple branches with different kernel sizes
    conv_branches = []
    kernel_sizes = [2, 4, 8]  # Example kernel sizes
    for kernel_size in kernel_sizes:
        conv = Conv1D(filters=64, kernel_size=kernel_size, activation='relu', padding='same')(input_layer)
        pool = MaxPooling1D(pool_size=2)(conv)
        conv_branches.append(pool)

    # Concatenate all convolutional branches
    concatenated = concatenate(conv_branches, axis=-1)

    # Additional convolutional layers after concatenation
    conv1 = Conv1D(filters=128, kernel_size=3, activation='relu')(concatenated)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv2)

    # A final global max pooling to summarize features into a flat form
    global_max_pool = GlobalMaxPooling1D()(pool1)

    return Model(inputs=input_layer, outputs=global_max_pool)


def build_classification_model(conv_base_output_shape, num_classes):
    input_layer = Input(shape=(conv_base_output_shape,))
    x = Dense(256, activation='relu')(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


subdatasets = ['Sn', 'Sh', 'St', 'Sht']
results = {}
scaler = StandardScaler()

# Adjust your training and evaluation loop
for dataset_name in subdatasets:
    print(f"\nProcessing {dataset_name} subdataset...")
    train_file = f"{dataset_name}_train.csv"
    test_file = f"{dataset_name}_test.csv"
    
    train_data, test_data = load_data(train_file, test_file)
    X_train = train_data.drop(columns=['Diabetes_012']).values
    y_train = train_data['Diabetes_012'].values
    X_test = test_data.drop(columns=['Diabetes_012']).values
    y_test = test_data['Diabetes_012'].values

    # Normalize data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_interactions = generate_feature_interactions(X_train)
    X_test_interactions = generate_feature_interactions(X_test)

    # Apply PCA
    X_train_reduced, pca = apply_pca(X_train_interactions)
    X_test_reduced = pca.transform(X_test_interactions)

    conv_base = build_convolutional_base(X_train_reduced.shape[1])
    train_representations = learn_vector_representations(conv_base, X_train_reduced.reshape(-1, X_train_reduced.shape[1], 1))
    test_representations = learn_vector_representations(conv_base, X_test_reduced.reshape(-1, X_test_reduced.shape[1], 1))

    classification_model = build_classification_model(train_representations.shape[1], len(np.unique(y_train)))
    history = classification_model.fit(train_representations, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.1)

    predictions = classification_model.predict(test_representations)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, predicted_classes)
    # Compute MSE
    mse = mean_squared_error(y_test, predicted_classes)
    # Compute F1-Score (assuming a multiclass, average as 'macro' or 'weighted')
    f1 = f1_score(y_test, predicted_classes, average='macro')
    # Get loss value
    loss_value = history.history['loss'][-1]  # Get last loss value from the training history
    
    results[dataset_name] = {
        'accuracy': accuracy,
        'mse': mse,
        'f1_score': f1,
        'final_loss': loss_value
    }
    print(f"Metrics for {dataset_name}: Accuracy: {accuracy:.2f}, MSE: {mse:.2f}, F1-Score: {f1:.2f}, Final Loss: {loss_value:.2f}")

print("\nFinal Results:", results)
