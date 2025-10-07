import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

####################
###    SETUP     ###
####################

# File paths
test_file = 'Datasets/wildfires_test.csv'
training_file = 'Datasets/wildfires_training.csv'

# Load datasets
print("Loading datasets...")
train_data = pd.read_csv(training_file)
test_data = pd.read_csv(test_file)


####################
## PRE-PROCESSING ##
####################

# Remove any rows with missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Separate features and target
X_train = train_data.drop('fire', axis=1)
y_train = train_data['fire']
X_test = test_data.drop('fire', axis=1)
y_test = test_data['fire']
####################
## MODEL TRAINING ##
####################

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
results = []

n_estimators_values = [int(x) for x in np.arange(11, 1012, 100)] 
learning_rate_values = [round(x, 1) for x in np.arange(0.1, 2.1, 0.2)]

for n_estimators in n_estimators_values:
    for learning_rate in learning_rate_values:
        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        results.append((n_estimators, learning_rate, train_accuracy, test_accuracy))
        print(f"n_estimators: {n_estimators}, learning_rate: {learning_rate} => Training set Acc: {train_accuracy:.4f}, Test set Acc: {test_accuracy:.4f}")

        if test_accuracy > highest_accuracy:
            highest_accuracy = test_accuracy
            best_params = (n_estimators, learning_rate)
            best_model = model


####################
###   RESULTS    ###
####################

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
