
import numpy as np
import pandas as pd
import pickle
import requests
import json

dataset = pd.read_csv('data/data.csv')

X = dataset.iloc[:,1:19].values
y = dataset.iloc[:, 19].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)


# Predicting a new result

y_pred1 = regressor.predict(X_test)


# Saving model to disk
pickle.dump(regressor, open('data/dtrmodel.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('data/dtrmodel.pkl','rb'))

from sklearn.metrics import mean_squared_error
a=mean_squared_error(y_test, y_pred1)
print("Mean Sqaured Error:")
print(a)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error:")
b=mean_absolute_error(y_test, y_pred1)
print(b)

from sklearn.metrics import explained_variance_score
c=explained_variance_score(y_test, y_pred1)
print("Variance:")
print(c)

from sklearn.metrics import median_absolute_error
d= median_absolute_error(y_test, y_pred1)
print("Median Absolute Error:")
print(d)

from sklearn.metrics import r2_score
e=r2_score(y_test, y_pred1)
print(e)

errors = abs(y_test - y_pred1)

# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / y_pred1))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred1))
