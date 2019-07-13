#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:40:36 2019

@author: iancorlett
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets
dataset = pd.read_csv("50_Startups.csv")
#dataset
X = dataset.iloc[:, :-1].values
#X
y = dataset.iloc[:,4]
#y

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap
X=X[:,1:]


#split the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#X_train


# fitting multiple linear regressiont othe training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1 )
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLX = sm.OLS( endog = y, exog = X_opt ).fit()
regressor_OLX.summary()
# stats summary show index 2 is above 5% significance level, so remove
X_opt = X[:, [0,1,3,4,5]]
regressor_OLX = sm.OLS( endog = y, exog = X_opt ).fit()
regressor_OLX.summary()
# stats summary shows index 1 is above 5% significance level, so remove
X_opt = X[:, [0,3,4,5]]
regressor_OLX = sm.OLS( endog = y, exog = X_opt ).fit()
regressor_OLX.summary()
# stats summary shows index 4 is above 5% significance level, so remove
X_opt = X[:, [0,3,5]]
regressor_OLX = sm.OLS( endog = y, exog = X_opt ).fit()
regressor_OLX.summary()
# stats summary shows index 5 is above 5% significance level, so remove
X_opt = X[:, [0,3]]
regressor_OLX = sm.OLS( endog = y, exog = X_opt ).fit()
regressor_OLX.summary()
