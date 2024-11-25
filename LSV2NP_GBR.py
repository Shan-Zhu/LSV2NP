import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# Read data
input_data = pd.read_csv("LSV2NP_database.csv", sep=',')

# Determine feature and target variables
features = input_data.drop('Particle_size (nm)', axis=1).drop('DOI', axis=1).drop('Composition', axis=1)
target = input_data['Particle_size (nm)']

# Split training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Select the model and fit it
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Organize the prediction results and actual results into DataFrames
train_results = pd.DataFrame({'Actual': y_train, 'Predicted': train_predictions})
test_results = pd.DataFrame({'Actual': y_test, 'Predicted': test_predictions})

# Calculate the R² metric
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

# Calculate the MAE metric
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)

# Calculate the MSE metric
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

# Calculate the RMSE metric
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

print("Train R²:", train_r2)
print("Test R²:", test_r2)
print("Train MAE:", train_mae)
print("Test MAE:", test_mae)
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Define 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Lists to store the Mean Squared Error (MSE) and R² values for each cross-validation
mse_scores = []
r2_scores = []

# Perform 5-fold cross-validation
for train_index, test_index in kf.split(X_train):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    # Train the model on the training set
    model.fit(X_train_fold, y_train_fold)

    # Make predictions on the test set
    y_pred_fold = model.predict(X_test_fold)

    # Calculate the Mean Squared Error and add it to the list
    mse = np.mean((y_pred_fold - y_test_fold) ** 2)
    mse_scores.append(mse)

    # Calculate the R² value and add it to the list
    r2 = r2_score(y_test_fold, y_pred_fold)
    r2_scores.append(r2)

# Output the Mean Squared Error results for each cross-validation
for i, mse in enumerate(mse_scores):
    print(f"Mean Squared Error of the {i + 1}-th fold cross-validation: {mse}")

# Output the R² results for each cross-validation
for i, r2 in enumerate(r2_scores):
    print(f"R² value of the {i + 1}-th fold cross-validation: {r2}")

# Calculate the average Mean Squared Error and average R² value
average_mse = np.mean(mse_scores)
average_r2 = np.mean(r2_scores)
print(f"Average Mean Squared Error: {average_mse}")
print(f"Average R² value: {average_r2}")

print("OK")