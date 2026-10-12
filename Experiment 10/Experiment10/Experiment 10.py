# ================================================================
# EXPERIMENT NO. 10
# End-to-End Data Analytics Project:
# Business Intelligence and Predictive Analytics
# ================================================================

# NAME       : DEV SHARMA
# UID        : CU24250269
# COURSE     : BTECH CSE
# SECTION    : B
# ROLL NO.   : 17

# ================================================================
# AIM
# ================================================================
# To implement a complete data analytics workflow on a real-world
# dataset by performing data preprocessing, exploratory data analysis,
# statistical analysis, visualization, predictive modeling, and
# presentation of actionable business insights.

# ================================================================
# STEP 1: IMPORT LIBRARIES
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk

from tkinter import filedialog

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================================================
# STEP 2: SELECT DATASET
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
    raise ValueError("No dataset selected.")

df = pd.read_csv(file_path)

# ================================================================
# STEP 3: DISPLAY BASIC DATASET INFORMATION
# ================================================================

print("Dataset Shape:", df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

# ================================================================
# STEP 4: DATA PREPROCESSING
# ================================================================

# Remove unnecessary index column if present
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert required columns to numeric
numeric_columns = [
    "TV Ad Budget ($)",
    "Radio Ad Budget ($)",
    "Newspaper Ad Budget ($)",
    "Sales ($)"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Handle missing values
df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

print("\nDuplicate Rows After Preprocessing:", df.duplicated().sum())

# ================================================================
# STEP 5: DESCRIPTIVE STATISTICS
# ================================================================

print("\nDescriptive Statistics:")
print(df.describe())

# ================================================================
# STEP 6: OUTLIER ANALYSIS
# ================================================================

plt.figure(figsize=(10, 6))

sns.boxplot(data=df)

plt.title("Boxplot of Advertising and Sales Variables")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ================================================================
# STEP 7: DISTRIBUTION ANALYSIS
# ================================================================

for column in numeric_columns:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column],
        kde=True
    )

    plt.title("Distribution of " + column)
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# ================================================================
# STEP 8: CORRELATION ANALYSIS
# ================================================================

correlation_matrix = df[numeric_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ================================================================
# STEP 9: ADVERTISING BUDGET VS SALES
# ================================================================

advertising_columns = [
    "TV Ad Budget ($)",
    "Radio Ad Budget ($)",
    "Newspaper Ad Budget ($)"
]

for column in advertising_columns:

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x=column,
        y="Sales ($)"
    )

    plt.title(column + " vs Sales")
    plt.xlabel(column)
    plt.ylabel("Sales ($)")

    plt.tight_layout()
    plt.show()

# ================================================================
# STEP 10: STATISTICAL ANALYSIS
# ================================================================

print("\n========== PEARSON CORRELATION TEST ==========")

for column in advertising_columns:

    correlation, p_value = pearsonr(
        df[column],
        df["Sales ($)"]
    )

    print("\n", column)
    print("Correlation:", round(correlation, 4))
    print("P-value:", round(p_value, 6))

# ================================================================
# STEP 11: PREPARE DATA FOR MACHINE LEARNING
# ================================================================

X = df[advertising_columns]
y = df["Sales ($)"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ================================================================
# STEP 12: BUILD LINEAR REGRESSION MODEL
# ================================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

# ================================================================
# STEP 13: MAKE PREDICTIONS
# ================================================================

y_pred = model.predict(X_test)

# ================================================================
# STEP 14: MODEL EVALUATION
# ================================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n========== MODEL PERFORMANCE ==========")
print("Mean Absolute Error (MAE):", round(mae, 4))
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))
print("R² Score:", round(r2, 4))

# ================================================================
# STEP 15: MODEL COEFFICIENTS
# ================================================================

print("\n========== MODEL COEFFICIENTS ==========")

for column, coefficient in zip(
    advertising_columns,
    model.coef_
):
    print(column, ":", round(coefficient, 4))

print("\nIntercept:", round(model.intercept_, 4))

# ================================================================
# STEP 16: REGRESSION EQUATION
# ================================================================

print("\n========== REGRESSION EQUATION ==========")

print(
    "Sales =",
    round(model.intercept_, 4),
    "+",
    round(model.coef_[0], 4),
    "* TV",
    "+",
    round(model.coef_[1], 4),
    "* Radio",
    "+",
    round(model.coef_[2], 4),
    "* Newspaper"
)

# ================================================================
# STEP 17: ACTUAL VS PREDICTED SALES
# ================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.show()

# ================================================================
# STEP 18: RESIDUAL ANALYSIS
# ================================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Analysis")

plt.tight_layout()
plt.show()

# ================================================================
# STEP 19: FEATURE IMPORTANCE
# ================================================================

feature_importance = pd.Series(
    model.coef_,
    index=advertising_columns
)

plt.figure(figsize=(8, 5))

feature_importance.sort_values().plot(
    kind="barh"
)

plt.title("Advertising Feature Coefficients")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.show()

# ================================================================
# STEP 20: BUSINESS INSIGHTS
# ================================================================

print("\n========== BUSINESS INSIGHTS ==========")

strongest_feature = feature_importance.abs().idxmax()

print(
    "1. The advertising variable with the largest absolute "
    "regression coefficient is:",
    strongest_feature
)

print(
    "2. The model explains approximately",
    round(r2 * 100, 2),
    "% of the variation in sales."
)

print(
    "3. Advertising budgets can be analyzed to understand "
    "their relationship with sales."
)

print(
    "4. Actual and predicted sales can be compared to evaluate "
    "the usefulness of the predictive model."
)

print(
    "5. Correlation and regression analysis help identify "
    "important advertising factors."
)

# ================================================================
# STEP 21: BUSINESS RECOMMENDATIONS
# ================================================================

print("\n========== BUSINESS RECOMMENDATIONS ==========")

print(
    "1. Use the regression model to estimate expected sales "
    "from advertising budgets."
)

print(
    "2. Monitor advertising channels that show stronger "
    "relationships with sales."
)

print(
    "3. Allocate advertising budgets using historical "
    "performance and predictive analysis."
)

print(
    "4. Periodically retrain the model using new sales data."
)

print(
    "5. Combine predictive analytics with dashboards for "
    "continuous business monitoring."
)

# ================================================================
# STEP 22: FINAL OBSERVATIONS
# ================================================================

# OBSERVATIONS:
#
# 1. The real-world advertising dataset was successfully loaded.
# 2. Unnecessary columns and duplicate records were removed.
# 3. Missing numerical values were handled using median values.
# 4. Descriptive statistics were calculated.
# 5. Distributions and outliers were analyzed using visualizations.
# 6. Correlation analysis identified relationships between advertising
#    budgets and sales.
# 7. Pearson correlation testing was performed.
# 8. A Multiple Linear Regression model was trained.
# 9. The model was evaluated using MAE, MSE, RMSE, and R².
# 10. Actual and predicted sales were visualized.
# 11. Regression coefficients were analyzed.
# 12. Business insights and recommendations were generated.

# ================================================================
# QUESTIONS AND ANSWERS
# ================================================================

# Q1. What are the major stages involved in an end-to-end data
# analytics project?
#
# Answer:
# The major stages are data collection, data preprocessing,
# exploratory data analysis, statistical analysis, visualization,
# predictive modeling, model evaluation, interpretation of results,
# and communication of business insights.

# Q2. Why is data preprocessing considered the foundation of successful
# analytics and machine learning?
#
# Answer:
# Data preprocessing improves data quality by handling missing values,
# duplicates, incorrect data types, and unnecessary variables.
# High-quality data helps analytical and machine learning models
# produce more reliable results.

# Q3. How does Exploratory Data Analysis (EDA) contribute to model
# development?
#
# Answer:
# EDA helps understand the structure and characteristics of data.
# It identifies patterns, relationships, outliers, and possible
# problems that can affect model development.

# Q4. What factors should be considered while selecting a machine
# learning algorithm for a business problem?
#
# Answer:
# The type of problem, size and structure of the dataset, target
# variable, number of features, interpretability requirements,
# computational resources, and required performance should be considered.

# Q5. Why is model evaluation essential before deploying a predictive
# model?
#
# Answer:
# Model evaluation determines how well a model performs on unseen data.
# It helps identify errors, overfitting, underfitting, and whether the
# model is suitable for practical use.

# Q6. Explain how Business Intelligence tools complement machine
# learning in data analytics projects.
#
# Answer:
# Business Intelligence tools provide dashboards, interactive charts,
# and reports for monitoring business performance. Machine learning
# provides predictive and analytical capabilities. Together they help
# organizations understand current performance and support future
# decision-making.

# Q7. What challenges are commonly encountered while working with
# real-world datasets?
#
# Answer:
# Common challenges include missing values, duplicate records,
# inconsistent data, outliers, incorrect data types, large datasets,
# categorical variables, and noisy information.

# Q8. How can analytical insights be converted into actionable business
# recommendations?
#
# Answer:
# Analytical findings should be connected with business objectives.
# Important trends, opportunities, risks, and performance gaps can be
# converted into specific actions such as budget allocation, resource
# planning, customer targeting, and process improvement.

# Q9. Why is effective presentation and data storytelling important
# in analytics projects?
#
# Answer:
# Effective presentation makes complex analytical findings easier to
# understand. Data storytelling connects results with business problems
# and helps stakeholders make informed decisions.

# Q10. Suggest future improvements or advanced techniques that could
# enhance the accuracy and effectiveness of the developed analytics
# solution.
#
# Answer:
# Future improvements may include collecting more recent data,
# feature engineering, cross-validation, hyperparameter tuning,
# regularization, ensemble models, advanced regression techniques,
# interactive dashboards, and periodic model retraining.

# ================================================================
# RESULT
# ================================================================

# A complete end-to-end data analytics workflow was successfully
# implemented on the Advertising Budget and Sales dataset.
# Data preprocessing, EDA, statistical analysis, visualization,
# predictive modeling, model evaluation, business insights, and
# recommendations were successfully performed.