# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:10:13 2022

@author: Kordian Czyzewski

Real estate price prediction in Singapore, by different model
of linear regression. We use classic Linear Regression, Lasso, Ridge and
Elastic Net models. Nature of data requires from us using polynomial features.
Hyperparameters of Lasso, Ridge and Elastic Net is choosing by CV method.
Model comparision use MAE, RMSE and R2 metrics.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV

result = []
models = {0: 'LinearRegression', 1: 'Lasso', 2: 'ElasticNet', 3: 'Ridge'}
# EDA
data = pd.read_csv('Real_estate.csv')

sns.heatmap(data.corr())
sns.pairplot(data)
# We see nonlinear relationship. So pipeline with polynomial feature will be
# more adequate than simply linear regression

# Data preparation

X = data[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station',
          'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
y = data['Y house price of unit area']

scaler = StandardScaler()
scaler.fit(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# LinearRegression model

model_linreg = make_pipeline(PolynomialFeatures(2),
                             LinearRegression())
model_linreg.fit(X_train, y_train)
y_linreg_pred = model_linreg.predict(X_test)

R2 = model_linreg.score(X_test, y_test)
MAE = mean_absolute_error(y_test, y_linreg_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_linreg_pred))

linreg_result = {'R2': R2, 'MAE': MAE, 'RMSE': RMSE}

result.append(linreg_result)

# Lasso model

model_lasso = make_pipeline(PolynomialFeatures(2),
                            LassoCV(eps=0.0000001, n_alphas=1000))
model_lasso.fit(X_train, y_train)
y_lasso_pred = model_lasso.predict(X_test)

R2 = model_lasso.score(X_test, y_test)
MAE = mean_absolute_error(y_test, y_lasso_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_lasso_pred))

lasso_result = {'R2': R2, 'MAE': MAE, 'RMSE': RMSE}

result.append(lasso_result)

# ElasticNet model

model_elasticnet = make_pipeline(PolynomialFeatures(2),
                                 ElasticNetCV(eps=0.000001, n_alphas=1000))
model_elasticnet.fit(X_train, y_train)
y_elasticnet_pred = model_elasticnet.predict(X_test)

R2 = model_elasticnet.score(X_test, y_test)
MAE = mean_absolute_error(y_test, y_elasticnet_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_elasticnet_pred))

elasticnet_result = {'R2': R2, 'MAE': MAE, 'RMSE': RMSE}

result.append(elasticnet_result)

# Ridge model

alphas = np.linspace(0.01, 100, num=10000)
model_ridge = make_pipeline(PolynomialFeatures(1), RidgeCV(alphas))
model_ridge.fit(X_train, y_train)
y_ridge_pred = model_ridge.predict(X_test)

R2 = model_ridge.score(X_test, y_test)
MAE = mean_absolute_error(y_test, y_linreg_pred)
RMSE = np.sqrt(mean_squared_error(y_test, y_ridge_pred))

ridge_result = {'R2': R2, 'MAE': MAE, 'RMSE': RMSE}

result.append(ridge_result)

# Comparison

comparison_df = pd.DataFrame(result)
comparison_df['Model'] = pd.Series(models)
comparison_df.set_index(keys='Model', inplace=True)
print(comparison_df)
