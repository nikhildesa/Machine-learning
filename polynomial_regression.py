# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:48:16 2019

@author: Nikhil
"""

# Polynomial regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Polynomial_dataset.csv")

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# No train test is required as dataset is very 
# create linear regression model 1
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X,y)
y_pred = linear_model.predict(X)

# create polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_model = PolynomialFeatures(degree = 2)

# converting x into x poly matrix required for polynomial regression
X_poly = poly_model.fit_transform(X)

#create linear regression model 2 which fits x poly and y
linear_model2 = LinearRegression()
linear_model2.fit(X_poly,y)
y_pred2 = linear_model2.predict(X_poly)

# visualize the result for linear regression model 1
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred, color = 'blue')
plt.title('Prediction by linear model')
plt.xlabel('levels')
plt.ylabel('salary')
plt.show()

# visualize the result for polynomial regression model
plt.scatter(X, y, color = 'red')
plt.plot(X, y_pred2, color = 'green')
plt.title('Prediction by polynomial model')
plt.xlabel('levels')
plt.ylabel('salary')
plt.show()


