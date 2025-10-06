# This file is used to visualise the effect of feature scaling on the dataset
# I plot temp vs drought_code, however any two features can be used

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data
train_data = pd.read_csv('Datasets/wildfires_training.csv')

# Separate features and target
x_train = train_data.drop('fire', axis=1)
y_train = train_data['fire']

# Scale data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)

# Recreate train_data with scaled features
train_data_scaled = x_train_scaled_df.copy()
train_data_scaled['fire'] = y_train.values

# Separate by class
fire_yes = train_data[train_data['fire'] == 'yes']
fire_no = train_data[train_data['fire'] == 'no']

fire_yes_scaled = train_data_scaled[train_data_scaled['fire'] == 'yes']
fire_no_scaled = train_data_scaled[train_data_scaled['fire'] == 'no']

# Features to be visualised
feature_x = 'temp'
feature_y = 'drought_code'

# Create side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

######################
### BEFORE SCALING ###
######################
ax1.scatter(fire_no[feature_x], fire_no[feature_y], c='red', label='No Fire', 
           alpha=0.7, edgecolors='black', s=60, linewidths=0.5)
ax1.scatter(fire_yes[feature_x], fire_yes[feature_y], c='blue', label='Fire', 
           alpha=0.7, edgecolors='black', s=60, linewidths=0.5)
ax1.set_title('Original Data', fontsize=12)
ax1.legend(fontsize=12, loc='best')
ax1.grid(True, alpha=0.3)

######################
### AFTER SCALING ###
######################
ax2.scatter(fire_no_scaled[feature_x], fire_no_scaled[feature_y], c='red', label='No Fire', 
           alpha=0.7, edgecolors='black', s=60, linewidths=0.5)
ax2.scatter(fire_yes_scaled[feature_x], fire_yes_scaled[feature_y], c='blue', label='Fire', 
           alpha=0.7, edgecolors='black', s=60, linewidths=0.5)
ax2.set_title('Scaled Data', fontsize=12)
ax2.legend(fontsize=12, loc='best')
ax2.grid(True, alpha=0.3)

plt.show()



