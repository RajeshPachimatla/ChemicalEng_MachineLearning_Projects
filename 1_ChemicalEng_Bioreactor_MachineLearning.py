# -*- coding: utf-8 -*-
"""
@author: G Mouktika and Pachimatla Rajesh

-Prediction of yield o gluconic acid production in a bioreactor
-first by linear regression
-second by ANN

"""
##### 1. Import the necessary libraries ###
# import dependencies
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# upload the data
bioreactor_data = pd.read_csv('P1_Bioreactor_ML.csv')

bioreactor_data.shape
#(60,4)

## data preprocessing
bioreactor_data.describe()
bioreactor_data.info()
#We can check the columns unique features
bioreactor_data['glucose_conc'].unique()
bioreactor_data['biomass_conc'].unique()
bioreactor_data['Dissolved_O2'].unique()
bioreactor_data['gluconic_acid_yield'].unique()
# Replace ? by NaN
bioreactor_data.replace('?', np.nan, inplace=True)
# drop all rows containing nan
bioreactor_data.dropna(inplace = True)

## To remove the outliers
# Define a threshhold for considering values as outliers
# Z = (X - μ) / σ
#y_column = bioreactor_data['gluconic_acid_yield']
y_column = bioreactor_data['gluconic_acid_yield'].astype(float)

#Lets calculate the standard deviation, mean and z_score to the data distribution
#hence find the outliers
std_y = y_column.std()
mean_y = y_column.mean()
z_scores = np.abs(stats.zscore(y_column))
z_scores.sort_values()
#by looking at the values, take threshold value as 0.17
y_threshold = 0.17
# create a boolean mask for rows with outliers
outlier_mask = (z_scores > y_threshold)

# remove rows with outliers
bioreactor_data_clean = bioreactor_data[~outlier_mask]
bioreactor_data_clean.reset_index(inplace=True, drop=True)

#### Method 1: Linear Regression ####
# Splitting the data into 80% training and 20% testing
X = bioreactor_data[['glucose_conc', 'biomass_conc', 'Dissolved_O2']]
y = bioreactor_data['gluconic_acid_yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Create a linear regression model
model = LinearRegression()
# Train the model using the training data
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ",mse)
print("R-squared: ",r2)

#Note: We can not use here accuracy.score, since it is used for multiclass classification

###Data Visualisation: Plot predicted vs actual
# alpha sets the transparency of the points to make overlapping points more visible
plt.figure(figsize = (8,6))
plt.scatter(y_test, y_pred, alpha = 0.5)
w = np.linspace(0, 150, 300)
z = w
plt.plot(z, w, linestyle='--', color='red')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.show()

# Plot a residual plot

# Calculate the residuals (the difference between actual and predicted values)
residuals = y_test.astype(float) - y_pred.astype(float)

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residual Plot')
plt.xlabel('Predicted Values(y_pred)')
plt.ylabel('Residuals')
# Mark a horizontal line at y = 0 for reference
plt.axhline(y=0, color = 'r', linestyle='--')
plt.show()

###### Method 2: ANN ####
# import relevant libraries

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Build the ANN model
model = keras.Sequential([
    # Create an input layer with 3 features
    keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    # Create a hidden layer with 32 neurons and ReLU activation
    keras.layers.Dense(32, activation = 'relu'),
    # Output layer with one neuron
    keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.summary()
# Train the model
model.fit(X_train, y_train, epochs = 150 , batch_size=32, validation_data=(X_test, y_test))

# Make predictions on the test set
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ",mse)
print("R-squared: ",r2)

# Create a scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
w = np.linspace(0, 150, 300)
z = w
plt.plot(z, w, linestyle='--', color='red')
plt.title('Actual vs. Predicted Values (ANN)')
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.show()

print(y_test.shape)
print(y_pred.shape)

if y_test.ndim > 1:
    y_test = y_test.ravel()
#y_test.ravel(): If y_test is a multi-dimensional array, this code is attempting 
#to flatten it. The ravel() function in NumPy is used to convert a 
#multi-dimensional array into a one-dimensional array. 
#It essentially takes all the elements of the multi-dimensional array and 
#puts them into a single one-dimensional array.

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residual Plot (ANN)')
plt.xlabel('Predicted Values (y_pred)')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0 for reference
plt.show()

# Relation between yield and x1

plt.figure(figsize=(8, 6))
plt.scatter(X1, y_pred, alpha=0.5)
plt.title('y vs. x1')
plt.xlabel('x1')
plt.ylabel('y_pred')
plt.show()

# Relation between yield and x2

plt.figure(figsize=(8, 6))
plt.scatter(X2, y_pred, alpha=0.5)
plt.title('y vs. x2')
plt.xlabel('x2')
plt.ylabel('y_pred')
plt.show()

# Relation between yield and x3

plt.figure(figsize=(8, 6))
plt.scatter(X3, y_pred, alpha=0.5)
plt.title('y vs. x3')
plt.xlabel('x3')
plt.ylabel('y_pred')
plt.show()

# Relation between the batch number and yield

plt.figure(figsize=(8, 6))
plt.scatter(batch_number, y, alpha=0.5)
plt.title('y vs. batch number')
plt.xlabel('batch number')
plt.ylabel('y')
plt.show()

