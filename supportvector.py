# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:57:42 2018

@author: Atul Chauhan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data/data.csv')

X = dataset.iloc[:, 1:19].values
y = dataset.iloc[:, 20].values

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)

y_pred2 = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
a=mean_squared_error(y_test, y_pred2)
print("Mean Sqaured Error:")
print(a)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error:")
b=mean_absolute_error(y_test, y_pred2)
print(b)

from sklearn.metrics import explained_variance_score
c=explained_variance_score(y_test, y_pred2)
print("Variance:")
print(c)

from sklearn.metrics import median_absolute_error
d= median_absolute_error(y_test, y_pred2)
print("Median Absolute Error:")
print(d)

from sklearn.metrics import r2_score
e=r2_score(y_test, y_pred2)
print(e)

errors = abs(y_test - y_pred2)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / y_pred2))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')