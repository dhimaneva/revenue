# -*- coding: utf-8 -*-
"""EvaSoni_P44075.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16Q7g0xngGYqFzE-v8Kc7hU4NSs8-A6bi
"""

# prompt: import pandas, matplotlib, sklearn

import pandas as pd
import matplotlib.pyplot as plt
import sklearn

# prompt: import/content/Eva Soni - revolutioncart_data - Eva Soni - revolutioncart_data.csv

import pandas as pd
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv('/content/Eva Soni - revolutioncart_data - Eva Soni - revolutioncart_data.csv')
print(df.head())

df

# prompt: consider monthly_revenue as y and other X

# Define X and y
X = df.drop('monthly_revenue', axis=1)  # Assuming 'monthly_revenue' is the column name
y = df['monthly_revenue']

print(X.head())
print(y.head())



# prompt: split X , y into train test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# prompt: make linear regression model for X_train and y_train

from sklearn.linear_model import LinearRegression

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# prompt: show coff and intercept

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# prompt: do the cross validation

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)  # You can change the number of folds (cv)

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())

# prompt: dump model

import pickle

# Save the model to a file
filename = 'linear_regression_model.pkl'
pickle.dump(model, open(filename, 'wb'))

print(f"Model saved to {filename}")

new_data = pd.DataFrame({'average_page_load_time': [2.5],
                        'average_product_rating': [4.2],
                        'average_shipping_time': [3.1],
                        'competitor_price_index': [0.9],
                        'consumer_confidence_index': [75],
                        # ... add other columns and values as needed
                       })

# Assuming 'df' is your original training data
missing_cols = set(df.columns) - set(new_data.columns)

for col in missing_cols:
    # Fill missing columns with 0 (or appropriate value)
    new_data[col] = 0

# prompt: make lasso regression for X_train and y_train

from sklearn.linear_model import Lasso

# Create a Lasso Regression model
lasso_model = Lasso(alpha=0.1)  # You can adjust the alpha parameter

# Train the model on the training data
lasso_model.fit(X_train, y_train)

# Print the coefficients and intercept
print("Lasso Coefficients:", lasso_model.coef_)
print("Lasso Intercept:", lasso_model.intercept_)

# Perform cross-validation (optional)
cv_scores_lasso = cross_val_score(lasso_model, X, y, cv=5)
print("Lasso Cross-validation scores:", cv_scores_lasso)
print("Lasso Average cross-validation score:", cv_scores_lasso.mean())

# prompt: lasso_model.score(X_test, y_test)

lasso_model_score = lasso_model.score(X_test, y_test)
print("Lasso Model R-squared on test data:", lasso_model_score)

# prompt: make ridge regression with X_train and y_train

from sklearn.linear_model import Ridge

# Create a Ridge Regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust the alpha parameter

# Train the model on the training data
ridge_model.fit(X_train, y_train)

# Print the coefficients and intercept
print("Ridge Coefficients:", ridge_model.coef_)
print("Ridge Intercept:", ridge_model.intercept_)

# Perform cross-validation (optional)
cv_scores_ridge = cross_val_score(ridge_model, X, y, cv=5)
print("Ridge Cross-validation scores:", cv_scores_ridge)
print("Ridge Average cross-validation score:", cv_scores_ridge.mean())

ridge_model_score = ridge_model.score(X_test, y_test)
print("Ridge Model R-squared on test data:", ridge_model_score)

# prompt: make elasticnet on X_train and y_train

from sklearn.linear_model import ElasticNet

# Create an ElasticNet Regression model
elasticnet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # You can adjust alpha and l1_ratio

# Train the model on the training data
elasticnet_model.fit(X_train, y_train)

# Print the coefficients and intercept
print("ElasticNet Coefficients:", elasticnet_model.coef_)
print("ElasticNet Intercept:", elasticnet_model.intercept_)

# Perform cross-validation (optional)
cv_scores_elasticnet = cross_val_score(elasticnet_model, X, y, cv=5)
print("ElasticNet Cross-validation scores:", cv_scores_elasticnet)
print("ElasticNet Average cross-validation score:", cv_scores_elasticnet.mean())

elasticnet_model_score = elasticnet_model.score(X_test, y_test)
print("ElasticNet Model R-squared on test data:", elasticnet_model_score)

# prompt: do cross validation on elastic net model

# Perform cross-validation on the ElasticNet model
cv_scores_elasticnet = cross_val_score(elasticnet_model, X, y, cv=5)

# Print the cross-validation scores
print("ElasticNet Cross-validation scores:", cv_scores_elasticnet)
print("ElasticNet Average cross-validation score:", cv_scores_elasticnet.mean())

# prompt: make random forest regressor model for the X_train and y_train

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Perform cross-validation (optional)
cv_scores_rf = cross_val_score(rf_model, X, y, cv=5)
print("Random Forest Cross-validation scores:", cv_scores_rf)
print("Random Forest Average cross-validation score:", cv_scores_rf.mean())

rf_model_score = rf_model.score(X_test, y_test)
print("Random Forest Model R-squared on test data:", rf_model_score)

# prompt: please do random forest hyperparamter optimization with methhurestic

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define the parameter grid for Random Forest
param_dist = {
    'n_estimators': randint(50, 200),  # Number of trees in the forest
    'max_depth': randint(5, 20),      # Maximum depth of the tree
    'min_samples_split': randint(2, 10),  # Minimum number of samples required to split an internal node
    'min_samples_leaf': randint(1, 5),   # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]          # Whether bootstrap samples are used when building trees
}

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter settings that are sampled
    cv=5,      # Number of cross-validation folds
    scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available CPU cores
)

# Fit the RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", random_search.best_params_)
print("Best score (negative MSE): ", random_search.best_score_)

# Evaluate the best model on the test data
best_rf_model = random_search.best_estimator_
best_rf_model_score = best_rf_model.score(X_test, y_test)
print("Best Random Forest Model R-squared on test data:", best_rf_model_score)

# prompt: by passing best parameters please make random forest classifi

# Create a Random Forest Regressor model with the best parameters
best_rf_model = RandomForestRegressor(
    n_estimators=random_search.best_params_['n_estimators'],
    max_depth=random_search.best_params_['max_depth'],
    min_samples_split=random_search.best_params_['min_samples_split'],
    min_samples_leaf=random_search.best_params_['min_samples_leaf'],
    bootstrap=random_search.best_params_['bootstrap'],
    random_state=42
)

# Train the model on the training data
best_rf_model.fit(X_train, y_train)

# Evaluate the model on the test data
best_rf_model_score = best_rf_model.score(X_test, y_test)
print("Best Random Forest Model R-squared on test data:", best_rf_model_score)

# prompt: do cross val for the best_rf_model

# Perform cross-validation on the best Random Forest model
cv_scores_best_rf = cross_val_score(best_rf_model, X, y, cv=5)

# Print the cross-validation scores
print("Best Random Forest Cross-validation scores:", cv_scores_best_rf)
print("Best Random Forest Average cross-validation score:", cv_scores_best_rf.mean())

# prompt: make KNN model for X_train and y_train

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create a KNN Regressor model
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors

# Train the model on the training data
knn_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_knn = knn_model.predict(X_test)

# Evaluate the model
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print("KNN Model MSE:", mse_knn)
print("KNN Model R-squared:", r2_knn)

# Perform cross-validation (optional)
cv_scores_knn = cross_val_score(knn_model, X, y, cv=5)
print("KNN Cross-validation scores:", cv_scores_knn)
print("KNN Average cross-validation score:", cv_scores_knn.mean())

# prompt: pip install github

!pip install PyGithub

# prompt: !pip install deap

!pip install deap

# prompt: check score with X_test and Y_test

# Assuming 'best_rf_model' is your trained model
y_pred = best_rf_model.predict(X_test)
score = best_rf_model.score(X_test, y_test)

print("R-squared score on test data:", score)

# You can also calculate other metrics like MSE or MAE
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test data:", mse)

# prompt: do cross val for KNN model

# Perform cross-validation on the KNN model
cv_scores_knn = cross_val_score(knn_model, X, y, cv=5)

# Print the cross-validation scores
print("KNN Cross-validation scores:", cv_scores_knn)
print("KNN Average cross-validation score:", cv_scores_knn.mean())

# prompt: K VALUE IDENTIFY WITH GRIDSEARCH

from sklearn.model_selection import GridSearchCV

# Create a KNN Regressor model
knn_model = KNeighborsRegressor()

# Define the parameter grid for KNN
param_grid = {
    'n_neighbors': list(range(1, 31))  # Try values from 1 to 30 for n_neighbors
}

# Create a GridSearchCV object
grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=param_grid,
    cv=5,  # Number of cross-validation folds
    scoring='neg_mean_squared_error',  # Use negative MSE as the scoring metric
    verbose=2,
    n_jobs=-1  # Use all available CPU cores
)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters found: ", grid_search.best_params_)
print("Best score (negative MSE): ", grid_search.best_score_)

# Evaluate the best model on the test data
best_knn_model = grid_search.best_estimator_
best_knn_model_score = best_knn_model.score(X_test, y_test)
print("Best KNN Model R-squared on test data:", best_knn_model_score)

# You can now use best_knn_model for predictions

