#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 21:54:16 2019

@author: iancorlett
"""


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets
dataset = pd.read_csv("Position_Salaries.csv")
#dataset
X = dataset.iloc[:, 1:2].values
#X
y = dataset.iloc[:,2].values
#y

#split the dataset into training set and test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train"""


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))
#y = sc_y.fit_transform(y)

# fitting SVR model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y.reshape(-1,1))

#predicting the results with SVR model
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])) )



#visualising the SVR model results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#visualising the SVR model results (high resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or bluff (SVR Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
