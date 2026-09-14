# ================================================================
# EXPERIMENT NO. 6
# PREDICTIVE ANALYTICS USING LINEAR REGRESSION
# AND MODEL PERFORMANCE EVALUATION
# ================================================================

# Name       : Dev Sharma
# UID        : CU24250269
# Course     : B.Tech CSE
# Section    : CSE "A"
# Roll No.   : 17

# ================================================================
# AIM
# ================================================================

### To develop a Linear Regression model for predicting Sales using
### advertising expenditure on TV, Radio and Newspaper and evaluate
### the model using appropriate regression performance metrics.

# ================================================================
# STEP 1: IMPORT LIBRARIES
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================================================
# STEP 2: LOAD DATASET
# ================================================================

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select Advertising Budget and Sales Dataset",
    filetypes=[
        ("CSV Files", "*.csv"),
        ("All Files", "*.*")
    ]
)

if not file_path:
    print("No file selected.")
    exit()

df = pd.read_csv(file_path)

# ================================================================
# STEP 3: DATA PREPROCESSING
# ================================================================

# Remove unnecessary index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Handle missing values if present
df = df.dropna()

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ================================================================
# STEP 4: SELECT FEATURES AND TARGET
# ================================================================

X = df[
    [
        "TV Ad Budget ($)",
        "Radio Ad Budget ($)",
        "Newspaper Ad Budget ($)"
    ]
]

y = df["Sales ($)"]

# ================================================================
# STEP 5: SPLIT DATASET INTO TRAINING AND TESTING SETS
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ================================================================
# STEP 6: TRAIN LINEAR REGRESSION MODEL
# ================================================================

model = LinearRegression()

model.fit(X_train, y_train)

# ================================================================
# STEP 7: PREDICT SALES
# ================================================================

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred
})

print("\nActual vs Predicted Sales:")
print(results.head(10))

# ================================================================
# STEP 8: MODEL PERFORMANCE EVALUATION
# ================================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))

# ================================================================
# STEP 9: REGRESSION COEFFICIENTS
# ================================================================

print("\nRegression Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", round(coefficient, 4))

print("Intercept:", round(model.intercept_, 4))

# ================================================================
# STEP 10: REGRESSION EQUATION
# ================================================================

print("\nRegression Equation:")

print(
    f"Sales = {model.intercept_:.4f} "
    f"+ ({model.coef_[0]:.4f} × TV) "
    f"+ ({model.coef_[1]:.4f} × Radio) "
    f"+ ({model.coef_[2]:.4f} × Newspaper)"
)

# ================================================================
# STEP 11: ACTUAL VS PREDICTED SALES
# ================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=2
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Actual Sales ($)")
plt.ylabel("Predicted Sales ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================================================
# STEP 12: TV ADVERTISING VS SALES
# ================================================================

tv_model = LinearRegression()

tv_model.fit(
    df[["TV Ad Budget ($)"]],
    y
)

tv_pred = tv_model.predict(
    df[["TV Ad Budget ($)"]]
)

sort_index = np.argsort(
    df["TV Ad Budget ($)"].values
)

plt.figure(figsize=(8, 6))

plt.scatter(
    df["TV Ad Budget ($)"],
    y,
    alpha=0.7,
    label="Actual Data"
)

plt.plot(
    df["TV Ad Budget ($)"].values[sort_index],
    tv_pred[sort_index],
    linewidth=2,
    label="Regression Line"
)

plt.title("TV Advertising vs Sales")
plt.xlabel("TV Ad Budget ($)")
plt.ylabel("Sales ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================================================
# STEP 13: ACTUAL VS PREDICTED COMPARISON
# ================================================================

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
}).reset_index(drop=True)

comparison = comparison.head(20)

plt.figure(figsize=(10, 6))

plt.plot(
    comparison.index,
    comparison["Actual"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    comparison.index,
    comparison["Predicted"],
    marker="x",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales Comparison")
plt.xlabel("Test Sample")
plt.ylabel("Sales ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ================================================================
# STEP 14: REGRESSION COEFFICIENT VISUALIZATION
# ================================================================

plt.figure(figsize=(8, 6))

plt.bar(
    ["TV", "Radio", "Newspaper"],
    model.coef_
)

plt.title("Advertising Variables and Their Impact on Sales")
plt.xlabel("Advertising Medium")
plt.ylabel("Regression Coefficient")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# ================================================================
# STEP 15: MODEL INTERPRETATION
# ================================================================

coefficient_series = pd.Series(
    model.coef_,
    index=X.columns
)

most_influential = coefficient_series.abs().idxmax()

print("\nMost influential feature:", most_influential)

if r2 >= 0.80:
    print("The model has strong predictive performance.")
elif r2 >= 0.50:
    print("The model has moderate predictive performance.")
else:
    print("The model has limited predictive performance.")

# ================================================================
# OBSERVATIONS
# ================================================================

# 1. The Advertising Budget and Sales dataset contains advertising
#    expenditure data for TV, Radio and Newspaper along with Sales.
#
# 2. The unnecessary index column was removed during preprocessing.
#
# 3. TV Ad Budget, Radio Ad Budget and Newspaper Ad Budget were
#    selected as independent variables.
#
# 4. Sales was selected as the dependent variable.
#
# 5. The dataset was divided into 80% training data and 20%
#    testing data.
#
# 6. A Multiple Linear Regression model was trained using the
#    training dataset.
#
# 7. The trained model was used to predict Sales for the testing
#    dataset.
#
# 8. MAE, MSE, RMSE and R² were calculated to evaluate the
#    performance of the regression model.
#
# 9. Regression coefficients were analyzed to understand the
#    influence of different advertising media on Sales.
#
# 10. Actual and predicted Sales were visualized to evaluate the
#     prediction performance of the model.

# ================================================================
# QUESTIONS AND ANSWERS
# ================================================================

# Q1. What is Predictive Analytics, and how is it used in
#     real-world applications?
#
# Answer:
# Predictive Analytics is the process of using historical data,
# statistical methods and machine learning algorithms to predict
# future outcomes.
#
# It is used in:
# - Sales forecasting
# - Demand prediction
# - Customer behavior prediction
# - Risk analysis
# - Financial forecasting
#
# In this experiment, Predictive Analytics is used to predict
# Sales based on advertising expenditure.


# ----------------------------------------------------------------

# Q2. Explain the working principle of the Linear Regression
#     algorithm.
#
# Answer:
# Linear Regression establishes a mathematical relationship between
# independent variables and a dependent variable.
#
# The general equation is:
#
# Y = b0 + b1X1 + b2X2 + b3X3
#
# In this experiment:
#
# Sales = b0 + b1(TV) + b2(Radio) + b3(Newspaper)
#
# The model learns the coefficients from the training data and
# uses them to predict Sales.


# ----------------------------------------------------------------

# Q3. Differentiate between dependent and independent variables
#     with suitable examples.
#
# Answer:
# Independent variables are the variables used to make predictions.
#
# The dependent variable is the variable that is being predicted.
#
# In this experiment:
#
# Independent Variables:
# - TV Ad Budget
# - Radio Ad Budget
# - Newspaper Ad Budget
#
# Dependent Variable:
# - Sales


# ----------------------------------------------------------------

# Q4. Why is it necessary to split the dataset into training
#     and testing sets?
#
# Answer:
# The training dataset is used to train the model, while the
# testing dataset is used to evaluate the model on unseen data.
#
# This helps determine whether the model can generalize to
# new observations.
#
# In this experiment, 80% of the data is used for training and
# 20% is used for testing.


# ----------------------------------------------------------------

# Q5. What is the significance of the R² Score in regression
#     analysis?
#
# Answer:
# R², or the Coefficient of Determination, measures how much
# variation in the dependent variable is explained by the
# regression model.
#
# A higher R² generally indicates that the model explains a
# larger proportion of the variation in the target variable.
#
# An R² value closer to 1 indicates stronger explanatory
# performance.


# ----------------------------------------------------------------

# Q6. Differentiate between MAE, MSE, and RMSE. Which metric
#     is more sensitive to large prediction errors?
#
# Answer:
# MAE (Mean Absolute Error) calculates the average absolute
# difference between actual and predicted values.
#
# MSE (Mean Squared Error) calculates the average squared
# prediction error.
#
# RMSE (Root Mean Squared Error) is the square root of MSE.
#
# MSE and RMSE are more sensitive to large prediction errors
# because the errors are squared.


# ----------------------------------------------------------------

# Q7. What assumptions should be satisfied before applying
#     Linear Regression?
#
# Answer:
# The major assumptions are:
#
# 1. There should be a linear relationship between variables.
#
# 2. Observations should be independent.
#
# 3. The variance of errors should be approximately constant.
#
# 4. Errors should be approximately normally distributed for
#    statistical inference.
#
# 5. Independent variables should not have severe multicollinearity.


# ----------------------------------------------------------------

# Q8. How can overfitting and underfitting affect the performance
#     of a regression model?
#
# Answer:
# Overfitting occurs when the model learns the training data too
# closely, including noise. It may perform poorly on unseen data.
#
# Underfitting occurs when the model is too simple and fails to
# capture important patterns in the dataset.
#
# A good model should achieve a suitable balance between
# complexity and generalization.


# ----------------------------------------------------------------

# Q9. Mention any three real-world applications of Linear
#     Regression in business or industry.
#
# Answer:
#
# 1. Sales forecasting
#    Predicting future sales using advertising expenditure.
#
# 2. House price prediction
#    Predicting house prices using area, location and other
#    numerical features.
#
# 3. Demand forecasting
#    Predicting future product demand using historical data.
#
# Other applications include revenue prediction, cost estimation
# and financial forecasting.


# ----------------------------------------------------------------

# Q10. How can feature selection improve the accuracy and
#      interpretability of a predictive model?
#
# Answer:
# Feature selection removes irrelevant or redundant variables
# from the dataset.
#
# It can:
# - Reduce model complexity
# - Reduce noise
# - Improve generalization
# - Reduce computation time
# - Improve interpretability
#
# In this experiment, selecting TV, Radio and Newspaper advertising
# budgets provides relevant variables for predicting Sales.


# ================================================================
# END OF EXPERIMENT 6
# ================================================================

print("\nExperiment 6 completed successfully!")