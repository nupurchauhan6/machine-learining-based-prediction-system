
import numpy as np
import pandas as pd
import pickle
import requests
import json
import sklearn

dataset = pd.read_csv('data/data.csv')

X = dataset.iloc[:,0:20].values
y = dataset.iloc[:, 20].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Saving model to disk
pickle.dump(regressor, open('data/mlrmodel.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('data/mlrmodel.pkl','rb'))

print('Coefficients: \n', regressor.coef_)

from sklearn.metrics import mean_squared_error
a=mean_squared_error(y_test, y_pred)
print("Mean Sqaured Error:")
print(a)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error:")
b=mean_absolute_error(y_test, y_pred)
print(b)

from sklearn.metrics import explained_variance_score
c=explained_variance_score(y_test, y_pred)
print("Variance:")
print(c)

from sklearn.metrics import median_absolute_error
d= median_absolute_error(y_test, y_pred)
print("Median Absolute Error:")
print(d)

from sklearn.metrics import r2_score
e=r2_score(y_test, y_pred)
print(e)


errors = abs(y_test - y_pred)

mape = np.mean(100 * (errors / y_pred))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')
