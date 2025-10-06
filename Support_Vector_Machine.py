import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

# Separate features and target
X_train = train_data.drop('fire', axis=1)
y_train = train_data['fire']
X_test = test_data.drop('fire', axis=1)
y_test = test_data['fire']


####################
## PRE-PROCESSING ##
####################

# Remove any rows with missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


####################
## MODEL TRAINING ##
####################

# Run with sklearn's default SVC hyperparameters
print("\nTraining SVM model with default hyperparameters...")
default_svm = SVC()
default_svm.fit(X_train_scaled, y_train)

print("\n**********Results of Default Model**********")

y_train_pred_default = default_svm.predict(X_train_scaled)
y_test_pred_default = default_svm.predict(X_test_scaled)

print(f"\nDefault Hyperparameters: C={default_svm.C}, gamma={default_svm.gamma}")
print(f"\nTraining Set Accuracy: {accuracy_score(y_train, y_train_pred_default):.4f}")
print(f"Test Set Accuracy: {accuracy_score(y_test, y_test_pred_default):.4f}")

print("Confusion Matrix for test set:")
print(confusion_matrix(y_test, y_test_pred_default))


# Find the best hyperparameters
print("\n**********Training SVM model with custom hyperparameters**********")

highest_accuracy = 0
best_params = (None, None)
best_model = None
results = []

gamma_values = [round(x, 2) for x in np.arange(0.01, 1.01, 0.05)]
c_values = [round(x, 1) for x in np.arange(0.1, 100.1, 5)]

for C in c_values:
    for gamma in gamma_values:
        custom_svm = SVC(C=C, gamma=gamma)
        custom_svm.fit(X_train_scaled, y_train)

        # Make predictions
        y_train_pred = custom_svm.predict(X_train_scaled)
        y_test_pred = custom_svm.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Store train and test accuracy of each hyperparameter combination
        results.append({
            'C': C,
            'gamma': gamma,
            'train_acc': train_acc,
            'test_acc': test_acc
        })

        print(f"C={C:>5}, gamma={str(gamma):>5} | Training set Accuracy: {train_acc:.4f} | Test set Accuracy: {test_acc:.4f}")

        if test_acc > highest_accuracy:
            highest_accuracy = test_acc
            best_params = (C, gamma)
            best_model = custom_svm


print("\n**********Results of Best Model**********")
y_train_pred_best = best_model.predict(X_train_scaled)
y_test_pred_best = best_model.predict(X_test_scaled)

print(f"\nBest Hyperparameters: C={best_params[0]}, gamma={best_params[1]}")
print(f"\nTraining Set Accuracy: {accuracy_score(y_train, y_train_pred_best):.4f}")
print("\nHighest Test Set Accuracy: {:.4f}".format(accuracy_score(y_test, y_test_pred_best)))

print("Test set Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_best))
