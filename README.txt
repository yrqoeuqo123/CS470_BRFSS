Towards Precision Public Health: Diabetes Risk Prediction
---------------------------------------------------------
This repository hosts the code and methodology used in the study "Towards Precision Public Health: Benchmarking Predictive Models for Diabetes Risk Using BRFSS Behavioral Data"

Dataset Overview
----------------
The dataset encompasses 253,680 participants and includes 21 features related to health behaviors, chronic conditions, and preventive services. 
Data has been preprocessed to facilitate analysis, including discretization and normalization.

Retrieve from the Following Link: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

Getting Started
---------------
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
-------------
What things you need to install the software and how to install them:
python>=3.6
numpy
pandas
scikit-learn
xgboost
tensorflow
keras

Repository Structure
--------------------
project-root/
│
├── datasets/
│ ├── diabetes_012_health_indicators_BRFSS2015.csv
│ ├── diabetes_binary_5050split_health_indicators_BRFSS2015.csv
│ ├── diabetes_binary_health_indicators_BRFSS2015.csv
│ ├── Sh_test/
│ ├── Sh_train/
│ ├── Sht_test/
│ ├── Sht_train/
│ ├── Sn_test/
│ ├── Sn_train/
│ ├── St_test/
│ └── St_train/
│
├── src/
│ ├── data_exploration_cohort.ipynb
│ ├── data_preprocessing.py
│ ├── DK_CNN.py
│ ├── finding_best_paras.py
│ └── train_evaluation.py
│
├── model_results.csv
└── README.md

Data Description
----------------
- `diabetes_012_health_indicators_BRFSS2015.csv`: Original dataset of 253,680 responses with detailed health indicators.
- `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`: A balanced subset of the data for comparative analysis.
- `diabetes_binary_health_indicators_BRFSS2015.csv`: Another subset with a binary target variable.
  
We focus on `diabetes_012_health_indicators_BRFSS2015.csv` for our main study due to its comprehensive data on diabetes indicators.

Source Code Files
-----------------
- `data_exploration_cohort.ipynb`: Jupyter notebook containing preliminary data exploration and cohort analysis.
- `data_preprocessing.py`: Script for data preprocessing and splitting into training and testing datasets for different subcohorts.
- `DK_CNN.py`: The Domain Knowledge-infused Convolutional Neural Network model implementation.
- `finding_best_paras.py`: Script for hyperparameter tuning across different models.
- `train_evaluation.py`: Script for training and evaluating models with optimal parameters found.

Results
-------
After running the `finding_best_paras.py` script, the optimal parameters are saved into `model_results.csv` in the project root.

To Run the Code
---------------
Pull the dataset, `diabetes_012_health_indicators_BRFSS2015.csv`, to the same folder as all the code scripts. Or just pull both code scripts and dataset to the main folder.

Models Included
---------------
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Trees
Random Forest
Support Vector Machines (SVM)
XGBoost
Multilayer Perceptron (MLP)
Domain Knowledge-Infused Convolutional Neural Network (DK-CNN)

Authors
-------
Tianyi Zhang - Dept. of Quantitative Theory & Methods
Brian Lin - Dept. of Computer Science
Ben DiGennaro - Dept. of Computer Science
