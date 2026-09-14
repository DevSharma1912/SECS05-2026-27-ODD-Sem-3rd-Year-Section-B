# ================================================================
#                    EXPERIMENT NO. 3
#          STATISTICAL ANALYSIS AND HYPOTHESIS TESTING
#                         USING PYTHON
# ================================================================

# Student Name : Dev Sharma
# UID          : CU24250269
# Course       : B.Tech CSE
# Section      : CSE "A"
# Roll No.     : 17

# ================================================================
# AIM
# ================================================================

# To perform statistical analysis and hypothesis testing on a
# real-world dataset using Python in order to identify relationships
# between variables, validate assumptions, and support
# data-driven decision-making.

# ================================================================
# DATASET
# ================================================================

# IBM HR Employee Attrition Dataset
#
# The dataset contains employee information such as:
# Age, Attrition, Job Role, Monthly Income, Job Satisfaction,
# Years at Company, Job Level, etc.

# ================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm

import tkinter as tk
from tkinter import filedialog


print("\n============================================================")
print("       STATISTICAL ANALYSIS AND HYPOTHESIS TESTING")
print("============================================================")

print("\nLibraries imported successfully!")


# ================================================================
# STEP 2: SELECT AND LOAD DATASET FROM COMPUTER
# ================================================================

print("\n========== SELECT DATASET ==========")

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select IBM HR Employee Attrition Dataset",
    filetypes=[
        ("CSV Files", "*.csv"),
        ("All Files", "*.*")
    ]
)

if not file_path:
    print("No file selected.")
    print("Program terminated.")
    exit()

print("\nSelected File:")
print(file_path)

# Load CSV file
df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

print("Dataset Shape:", df.shape)


# ================================================================
# STEP 3: DISPLAY DATASET INFORMATION
# ================================================================

print("\n============================================================")
print("                 DATASET INFORMATION")
print("============================================================")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicate rows:", df.duplicated().sum())


# ================================================================
# STEP 4: DESCRIPTIVE STATISTICS
# ================================================================

print("\n============================================================")
print("              DESCRIPTIVE STATISTICS")
print("============================================================")

# Numerical columns required for analysis
numeric_columns = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "JobSatisfaction",
    "JobLevel"
]

# Check which columns are available
numeric_columns = [
    column for column in numeric_columns
    if column in df.columns
]

print("\nNumerical columns used:")
print(numeric_columns)


# Calculate descriptive statistics
for column in numeric_columns:

    print("\n------------------------------------------------------------")
    print("Column:", column)
    print("------------------------------------------------------------")

    print("Mean               :", df[column].mean())
    print("Median             :", df[column].median())
    print("Mode               :", df[column].mode()[0])
    print("Variance           :", df[column].var())
    print("Standard Deviation :", df[column].std())


# Display complete statistical summary
print("\n========== COMPLETE DESCRIPTIVE SUMMARY ==========")

print(df[numeric_columns].describe())


# ================================================================
# STEP 5: PEARSON CORRELATION ANALYSIS
# ================================================================

print("\n============================================================")
print("             PEARSON CORRELATION ANALYSIS")
print("============================================================")

# Calculate Pearson correlation matrix
correlation_matrix = df[numeric_columns].corr(method="pearson")

print("\n========== PEARSON CORRELATION MATRIX ==========")

print(correlation_matrix)


# Plot correlation heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.show()


# ================================================================
# STEP 6: FORMULATE NULL AND ALTERNATIVE HYPOTHESIS
# ================================================================

print("\n============================================================")
print("                 HYPOTHESIS FORMULATION")
print("============================================================")

print("\nBusiness Problem:")
print("Is there a significant difference in Monthly Income")
print("between employees who leave the company and employees")
print("who stay?")

print("\nNull Hypothesis (H0):")
print("There is no significant difference in Monthly Income")
print("between employees who stay and employees who leave.")

print("\nAlternative Hypothesis (H1):")
print("There is a significant difference in Monthly Income")
print("between employees who stay and employees who leave.")


# ================================================================
# STEP 7: INDEPENDENT SAMPLE T-TEST
# ================================================================

print("\n============================================================")
print("             INDEPENDENT SAMPLE T-TEST")
print("============================================================")

# Check required columns
if "Attrition" in df.columns and "MonthlyIncome" in df.columns:

    # Create two groups
    stayed = df[
        df["Attrition"] == "No"
    ]["MonthlyIncome"].dropna()

    left = df[
        df["Attrition"] == "Yes"
    ]["MonthlyIncome"].dropna()

    # Calculate group means
    stayed_mean = stayed.mean()
    left_mean = left.mean()

    print("\nNumber of employees who stayed:", len(stayed))
    print("Number of employees who left  :", len(left))

    print("\nMean Monthly Income - Stayed:")
    print(stayed_mean)

    print("\nMean Monthly Income - Left:")
    print(left_mean)

    # Independent sample t-test
    t_statistic, p_value = stats.ttest_ind(
        stayed,
        left,
        equal_var=False
    )

    print("\nT-Statistic:", t_statistic)
    print("P-Value    :", p_value)

    # Significance level
    alpha = 0.05

    print("\nSignificance Level (Alpha):", alpha)

    # Decision
    if p_value < alpha:

        print("\nDecision:")
        print("Reject the Null Hypothesis (H0).")

        print("\nConclusion:")
        print(
            "There is a statistically significant difference "
            "in Monthly Income between employees who stayed "
            "and employees who left."
        )

    else:

        print("\nDecision:")
        print("Fail to Reject the Null Hypothesis (H0).")

        print("\nConclusion:")
        print(
            "There is no statistically significant difference "
            "in Monthly Income between employees who stayed "
            "and employees who left."
        )

else:

    print("Required columns are not available.")


# ================================================================
# STEP 8: ONE-WAY ANOVA
# ================================================================

print("\n============================================================")
print("                    ONE-WAY ANOVA")
print("============================================================")

# Check required columns
if "JobRole" in df.columns and "JobSatisfaction" in df.columns:

    # Create groups based on JobRole
    job_roles = df["JobRole"].dropna().unique()

    groups = []

    valid_roles = []

    for role in job_roles:

        satisfaction_values = df[
            df["JobRole"] == role
        ]["JobSatisfaction"].dropna()

        # ANOVA requires at least two observations
        if len(satisfaction_values) >= 2:

            groups.append(satisfaction_values)
            valid_roles.append(role)

    print("\nJob Roles used in ANOVA:")
    print(valid_roles)

    # Perform One-Way ANOVA
    anova_statistic, anova_p_value = stats.f_oneway(
        *groups
    )

    print("\nF-Statistic:", anova_statistic)
    print("P-Value    :", anova_p_value)

    # Significance level
    alpha = 0.05

    print("\nSignificance Level (Alpha):", alpha)

    # Decision
    if anova_p_value < alpha:

        print("\nDecision:")
        print("Reject the Null Hypothesis (H0).")

        print("\nConclusion:")
        print(
            "There is a statistically significant difference "
            "in Job Satisfaction among the different Job Roles."
        )

    else:

        print("\nDecision:")
        print("Fail to Reject the Null Hypothesis (H0).")

        print("\nConclusion:")
        print(
            "There is no statistically significant difference "
            "in Job Satisfaction among the different Job Roles."
        )

else:

    print("Required columns are not available.")


# ================================================================
# STEP 9: SIMPLE LINEAR REGRESSION
# ================================================================

print("\n============================================================")
print("                SIMPLE LINEAR REGRESSION")
print("============================================================")

# We use:
#
# Independent Variable (X) = YearsAtCompany
# Dependent Variable   (Y) = MonthlyIncome

if "YearsAtCompany" in df.columns and "MonthlyIncome" in df.columns:

    # Select required columns
    regression_data = df[
        ["YearsAtCompany", "MonthlyIncome"]
    ].dropna()

    # Independent variable
    X = regression_data["YearsAtCompany"]

    # Dependent variable
    Y = regression_data["MonthlyIncome"]

    # Add constant/intercept
    X_with_constant = sm.add_constant(X)

    # Create regression model
    model = sm.OLS(
        Y,
        X_with_constant
    ).fit()

    # Display model summary
    print("\n========== REGRESSION MODEL SUMMARY ==========")

    print(model.summary())

    # Get coefficients
    intercept = model.params["const"]
    slope = model.params["YearsAtCompany"]

    # R-squared
    r_squared = model.rsquared

    print("\n========== REGRESSION VALUES ==========")

    print("Intercept :", intercept)
    print("Slope     :", slope)
    print("R-Squared :", r_squared)

else:

    print("Required columns are not available.")


# ================================================================
# STEP 10: REGRESSION EQUATION
# ================================================================

print("\n============================================================")
print("                 REGRESSION EQUATION")
print("============================================================")

if "model" in locals():

    print(
        f"\nMonthlyIncome = "
        f"{intercept:.2f} + "
        f"({slope:.2f} × YearsAtCompany)"
    )

    print("\nInterpretation of Slope:")

    if slope > 0:

        print(
            "Monthly Income tends to increase as "
            "Years at Company increases."
        )

    elif slope < 0:

        print(
            "Monthly Income tends to decrease as "
            "Years at Company increases."
        )

    else:

        print(
            "There is no linear relationship between "
            "Years at Company and Monthly Income."
        )


# ================================================================
# STEP 11: CONFIDENCE INTERVALS
# ================================================================

print("\n============================================================")
print("                  CONFIDENCE INTERVALS")
print("============================================================")

if "model" in locals():

    print(
        "\n95% Confidence Intervals for Regression Parameters:"
    )

    confidence_intervals = model.conf_int()

    confidence_intervals.columns = [
        "Lower Bound",
        "Upper Bound"
    ]

    print(confidence_intervals)


# ================================================================
# STEP 12: REGRESSION VISUALIZATION
# ================================================================

print("\n============================================================")
print("                REGRESSION VISUALIZATION")
print("============================================================")

if "model" in locals():

    plt.figure(figsize=(10, 6))

    sns.regplot(
        x="YearsAtCompany",
        y="MonthlyIncome",
        data=regression_data,
        scatter_kws={"alpha": 0.5}
    )

    plt.title(
        "Years at Company vs Monthly Income"
    )

    plt.xlabel("Years at Company")
    plt.ylabel("Monthly Income")

    plt.tight_layout()
    plt.show()


# ================================================================
# STEP 13: FINAL STATISTICAL SUMMARY
# ================================================================

print("\n")
print("============================================================")
print("               FINAL STATISTICAL SUMMARY")
print("============================================================")

print("\nDataset Shape:")
print(df.shape)


# ---------------- T-TEST SUMMARY ----------------

print("\n========== T-TEST SUMMARY ==========")

if "t_statistic" in locals():

    print("T-Statistic:", t_statistic)
    print("P-Value    :", p_value)

    if p_value < 0.05:

        print("Result: Statistically Significant")
        print("H0 is rejected.")

    else:

        print("Result: Not Statistically Significant")
        print("H0 is not rejected.")


# ---------------- ANOVA SUMMARY ----------------

print("\n========== ANOVA SUMMARY ==========")

if "anova_statistic" in locals():

    print("F-Statistic:", anova_statistic)
    print("P-Value    :", anova_p_value)

    if anova_p_value < 0.05:

        print("Result: Statistically Significant")
        print("H0 is rejected.")

    else:

        print("Result: Not Statistically Significant")
        print("H0 is not rejected.")


# ---------------- REGRESSION SUMMARY ----------------

print("\n========== REGRESSION SUMMARY ==========")

if "model" in locals():

    print("Intercept :", intercept)
    print("Slope     :", slope)
    print("R-Squared :", r_squared)

    print("\nRegression Equation:")
    print(
        f"MonthlyIncome = "
        f"{intercept:.2f} + "
        f"({slope:.2f} × YearsAtCompany)"
    )


# ---------------- CORRELATION SUMMARY ----------------

print("\n========== CORRELATION SUMMARY ==========")

print(correlation_matrix)


# ================================================================
# STEP 14: OBSERVATIONS
# ================================================================

print("\n============================================================")
print("                       OBSERVATIONS")
print("============================================================")

print("""
1. Descriptive statistics were used to summarize the numerical
   variables using mean, median, mode, variance and standard
   deviation.

2. Pearson correlation was used to identify the strength and
   direction of linear relationships between numerical variables.

3. A correlation heatmap was created to visualize relationships
   between variables.

4. An independent sample t-test was performed to compare the
   Monthly Income of employees who stayed and employees who left.

5. The p-value was compared with the significance level of 0.05
   to make the statistical decision.

6. One-Way ANOVA was performed to compare Job Satisfaction
   among different Job Roles.

7. Simple Linear Regression was used to study the relationship
   between YearsAtCompany and MonthlyIncome.

8. The regression coefficient indicates how Monthly Income
   changes with YearsAtCompany.

9. R-Squared indicates the proportion of variation in the
   dependent variable explained by the regression model.

10. Statistical analysis helps organizations make
    evidence-based and data-driven decisions.
""")


# ================================================================
#                    QUESTIONS AND ANSWERS
# ================================================================

# Q1. What is the difference between descriptive statistics
#     and inferential statistics?
#
# Answer:
# Descriptive statistics summarize and describe the data using
# measures such as mean, median, mode, variance and standard
# deviation.
#
# Inferential statistics use sample data to make conclusions
# or predictions about a larger population.
#
# Example:
# Finding the average salary of employees is descriptive
# statistics.
#
# Testing whether two employee groups have significantly
# different salaries is inferential statistics.


# ----------------------------------------------------------------

# Q2. Explain the concepts of the Null Hypothesis (H0) and
#     Alternative Hypothesis (H1).
#
# Answer:
# The Null Hypothesis (H0) states that there is no significant
# difference or relationship between the variables.
#
# The Alternative Hypothesis (H1) states that there is a
# significant difference or relationship between the variables.
#
# In this experiment:
#
# H0:
# There is no significant difference in Monthly Income between
# employees who stay and employees who leave.
#
# H1:
# There is a significant difference in Monthly Income between
# employees who stay and employees who leave.


# ----------------------------------------------------------------

# Q3. What is a p-value? How is it used to make statistical
#     decisions?
#
# Answer:
# A p-value indicates how compatible the observed result is
# with the Null Hypothesis.
#
# At a significance level of 0.05:
#
# If p-value < 0.05:
#     Reject H0.
#
# If p-value >= 0.05:
#     Fail to reject H0.
#
# Therefore, the p-value helps determine whether the result is
# statistically significant.


# ----------------------------------------------------------------

# Q4. Differentiate between a t-test and ANOVA. In which
#     situations is each test applied?
#
# Answer:
# A t-test is generally used to compare the means of two groups.
#
# Example:
# Comparing Monthly Income between employees who stayed and
# employees who left.
#
# ANOVA is used to compare the means of three or more groups.
#
# Example:
# Comparing Job Satisfaction among different Job Roles.
#
# Therefore:
#
# t-test -> Used for comparing two groups.
# ANOVA   -> Used for comparing three or more groups.


# ----------------------------------------------------------------

# Q5. What does the Pearson correlation coefficient indicate?
#     What are its possible values?
#
# Answer:
# Pearson correlation coefficient measures the strength and
# direction of a linear relationship between two numerical
# variables.
#
# Its possible values range from -1 to +1.
#
# +1 = Perfect positive correlation
#  0 = No linear correlation
# -1 = Perfect negative correlation
#
# Positive correlation means that the variables generally
# increase together.
#
# Negative correlation means that when one variable increases,
# the other generally decreases.


# ----------------------------------------------------------------

# Q6. Explain the significance of R² (Coefficient of
#     Determination) in Linear Regression.
#
# Answer:
# R² represents the proportion of variation in the dependent
# variable that is explained by the independent variable in
# the regression model.
#
# R² ranges from 0 to 1.
#
# For example:
#
# R² = 0.70
#
# This means that approximately 70% of the variation in the
# dependent variable is explained by the regression model.
#
# A higher R² generally indicates a better fit, although R²
# should not be used alone to judge a model.


# ----------------------------------------------------------------

# Q7. Why is statistical analysis important before applying
#     machine learning algorithms?
#
# Answer:
# Statistical analysis helps us understand the dataset before
# applying machine learning algorithms.
#
# It can help identify:
#
# - Relationships between variables
# - Outliers
# - Missing values
# - Data distributions
# - Important features
# - Statistical significance
#
# Therefore, statistical analysis can help in selecting
# appropriate features, preprocessing techniques and models.


# ----------------------------------------------------------------

# Q8. What assumptions should be satisfied before performing
#     a t-test or ANOVA?
#
# Answer:
# Important assumptions include:
#
# 1. Observations should be independent.
#
# 2. The dependent variable should generally be numerical.
#
# 3. Data within groups should be approximately normally
#    distributed, especially for small samples.
#
# 4. For the standard versions of these tests, group variances
#    should be reasonably similar.
#
# If these assumptions are strongly violated, suitable
# alternative statistical methods may be considered.


# ----------------------------------------------------------------

# Q9. How can regression analysis help organizations in
#     forecasting and decision-making?
#
# Answer:
# Regression analysis helps organizations understand the
# relationship between variables and make predictions.
#
# It can be used for:
#
# - Sales forecasting
# - Revenue prediction
# - Demand forecasting
# - Cost estimation
# - Employee analytics
# - Business planning
#
# For example, an organization can analyze the relationship
# between employee experience and Monthly Income.
#
# Regression therefore supports data-driven decision-making.


# ----------------------------------------------------------------

# Q10. Give two real-world applications where hypothesis
#      testing is commonly used in data analytics.
#
# Answer:
#
# 1. BUSINESS:
#    A company can test whether a new marketing campaign
#    significantly increases sales compared with an old
#    marketing campaign.
#
# 2. HEALTHCARE:
#    Researchers can test whether a new treatment produces
#    significantly different results compared with an
#    existing treatment.
#
# Other applications include finance, manufacturing,
# education, customer analytics and quality control.


# ================================================================
#                       END OF EXPERIMENT 3
# ================================================================

print("\n")
print("============================================================")
print("           EXPERIMENT 3 COMPLETED SUCCESSFULLY!")
print("============================================================")