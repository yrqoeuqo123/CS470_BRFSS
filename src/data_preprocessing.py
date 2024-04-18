import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Path to the dataset
file_path = 'diabetes_012_health_indicators_BRFSS2015.csv'

# Read the dataset into a pandas DataFrame
health_indicators = pd.read_csv(file_path)

# Define the conditions for the subcohorts
conditions = {
    'Sn': ((health_indicators['HeartDiseaseorAttack'] == 0) & 
           (health_indicators['Stroke'] == 0)),
    'Sh': ((health_indicators['HeartDiseaseorAttack'] == 1) & 
           (health_indicators['Stroke'] == 0)),
    'St': ((health_indicators['HeartDiseaseorAttack'] == 0) & 
           (health_indicators['Stroke'] == 1)),
    'Sht': ((health_indicators['HeartDiseaseorAttack'] == 1) & 
            (health_indicators['Stroke'] == 1))
}

# Create subdatasets for each subcohort and remove 'HeartDiseaseorAttack' and 'Stroke' columns
subcohorts_data = {}
for name, condition in conditions.items():
    subcohorts_data[name] = health_indicators[condition].drop(['HeartDiseaseorAttack', 'Stroke'], axis=1)

    # Standardize features except for the target
    scaler = StandardScaler()
    features = subcohorts_data[name].drop(columns=['Diabetes_012'])
    scaled_features = scaler.fit_transform(features)
    
    # Convert scaled features back to DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
    
    # Update the original DataFrame with scaled features
    subcohorts_data[name].update(scaled_df)

    # Split data into training and testing sets
    X = subcohorts_data[name].drop(columns=['Diabetes_012'])  # Features
    y = subcohorts_data[name]['Diabetes_012']  # Target variable
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save training and testing sets
    subcohorts_data[name + '_train'] = pd.concat([xTrain, yTrain], axis=1)
    subcohorts_data[name + '_test'] = pd.concat([xTest, yTest], axis=1)

    # Save each subdataset to a new CSV file
    subcohorts_data[name + '_train'].to_csv(f"{name}_train.csv", index=False)
    subcohorts_data[name + '_test'].to_csv(f"{name}_test.csv", index=False)
    print(f"Subcohort dataset {name} saved as CSV files: {name}_train.csv and {name}_test.csv")
