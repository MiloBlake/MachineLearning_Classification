"""
AdaBoost Classifier for Predicting Wildfires

This script implements an AdaBoost classifier to predict the occurrence of wildfires.
The model is trained on the wildfires_training.csv dataset and evaluated on the wildfires_test.csv dataset.
It first runs the model with its default parameters and then tunes two of the hyperparameters to find the best combination.

Author: Milo Blake
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# -----SETUP-----

# File paths
test_file = 'Datasets/wildfires_test.csv'
training_file = 'Datasets/wildfires_training.csv'

# Load datasets
print("Loading datasets...")
train_data = pd.read_csv(training_file)
test_data = pd.read_csv(test_file)


# ----- PRE-PROCESSING-----

# Remove any rows with missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Feature Engineering
# Temp humidity - combines temp and humidity
train_data['temp_and_humidity'] = train_data['temp'] * (1 - train_data['humidity']/100)
test_data['temp_and_humidity'] = test_data['temp'] * (1 - test_data['humidity']/100)

# Drought Buildup Index - combines drought code and buildup index
train_data['drought_and_buildup_index'] = train_data['drought_code'] * train_data['buildup_index']
test_data['drought_and_buildup_index'] = test_data['drought_code'] * test_data['buildup_index']

# Wind-Drought Risk - interaction between wind and drought conditions
train_data['wind_and_drought'] = train_data['wind_speed'] * train_data['drought_code']
test_data['wind_and_drought'] = test_data['wind_speed'] * test_data['drought_code']

# Dry hot - low rainfall with high temp
train_data['temp_and_dry'] = train_data['temp'] / (train_data['rainfall'] + 1) # add 1 to avoid dividing by zero
test_data['temp_and_dry'] = test_data['temp'] / (test_data['rainfall'] + 1)

# Separate features and target
X_train = train_data.drop('fire', axis=1)
y_train = train_data['fire']
X_test = test_data.drop('fire', axis=1)
y_test = test_data['fire']



# -----MODEL TRAINING-----

# Default AdaBoost hyperparameters
print("\n**********Training AdaBoost with Default Hyperparameters**********")

default_model = AdaBoostClassifier(random_state=42)
default_model.fit(X_train, y_train)

y_train_pred_default = default_model.predict(X_train)
y_test_pred_default = default_model.predict(X_test)


# Custom hyperparameters 
print("\n**********Training AdaBoost with Custom Hyperparameters**********")
highest_accuracy = 0
best_params = (None, None)
best_model = None

n_estimators_values = [int(x) for x in np.arange(11, 1012, 50)] 
learning_rate_values = [round(x, 1) for x in np.arange(0.1, 2.1, 0.2)]

for n_estimators in n_estimators_values:
    for learning_rate in learning_rate_values:
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"n_estimators: {n_estimators}, "
                f"learning_rate: {learning_rate} => "
                f"Training Accuracy: {train_accuracy:.4f}, "
                f"Test Accuracy: {test_accuracy:.4f}")

        if test_accuracy > highest_accuracy:
            highest_accuracy = test_accuracy
            best_params = (n_estimators, learning_rate)
            best_model = model


# -----RESULTS-----


print("\n**********Results of Default Model**********")

print(f"\nDefault Hyperparameters: n_estimators: {default_model.n_estimators} learning_rate: {default_model.learning_rate}")

print(f"\nTraining Set Accuracy: {accuracy_score(y_train, y_train_pred_default):.4f}")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_test_pred_default):.4f}")

print("\nTest Set Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_default))



print("\n\n**********Results of Tuned Model**********")

y_train_pred_tuned = best_model.predict(X_train)
y_test_pred_tuned = best_model.predict(X_test)

print(f"\nBest Hyperparameters: n_estimators={best_params[0]}, learning_rate={best_params[1]}")
print(f"\nTraining Set Accuracy: {accuracy_score(y_train, y_train_pred_tuned):.4f}")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_test_pred_tuned):.4f}")

print("\nTest Set Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_tuned))


# Feature Importance Analysis
import matplotlib.pyplot as plt

importances = best_model.feature_importances_
feature_names = X_train.columns
feature_importance_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

features, importances_vals = zip(*feature_importance_pairs)
plt.bar(features, importances_vals)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()