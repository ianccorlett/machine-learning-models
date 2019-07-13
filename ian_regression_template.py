#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 21:29:01 2019

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



# fitting regression model to the dataset


#predicting the results with regression model
y_pred = regressor.predict([[6.5]])



#visualising the regression model results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#visualising the regression model results (high resolution)
X_grid = np.arrange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

