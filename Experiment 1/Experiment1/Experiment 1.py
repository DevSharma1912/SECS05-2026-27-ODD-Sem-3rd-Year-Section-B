# Name :- Dev Sharma
# CU24250269
# BTech CSE 3rd Year SECTION - "A"
# Roll No. :- 17

# ============================================================
# Experiment No. 1
# Experiment Name
# Real-World Data Collection, Cleaning and Preprocessing
# using Python
# ============================================================

# Aim:
# To collect a real-world dataset from publicly available
# sources and perform data preprocessing by handling missing
# values, duplicate records, inconsistent formats, categorical
# variables, and outliers using Python.


# ============================================================
# Step 1: Import Required Libraries
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# Step 2: Upload / Select Dataset from Computer
# ============================================================

root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select Dataset CSV File",
    filetypes=[
        ("CSV Files", "*.csv"),
        ("All Files", "*.*")
    ]
)

if not file_path:
    print("No dataset selected.")
    exit()

# Load dataset
df = pd.read_csv(file_path)

print("Dataset loaded successfully.")
print("Selected Dataset:", file_path)


# ============================================================
# Step 3: Display First Five Records
# ============================================================

print("\n========== FIRST FIVE RECORDS ==========")

print(df.head())


# ============================================================
# Step 4: Display Dataset Information
# ============================================================

print("\n========== DATASET INFORMATION ==========")

print(df.info())


# ============================================================
# Step 5: Display Statistical Description
# ============================================================

print("\n========== STATISTICAL DESCRIPTION ==========")

print(df.describe(include="all"))


# ============================================================
# Step 6: Check Dataset Shape
# ============================================================

print("\n========== DATASET SHAPE ==========")

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# ============================================================
# Step 7: Identify Missing Values
# ============================================================

print("\n========== MISSING VALUES ==========")

missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# Step 8: Handle Missing Values
# ============================================================

# Numerical columns -> Median
numerical_columns = df.select_dtypes(
    include=np.number
).columns

for column in numerical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(
            df[column].median()
        )

# Categorical columns -> Mode
categorical_columns = df.select_dtypes(
    exclude=np.number
).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        mode_value = df[column].mode()

        if len(mode_value) > 0:
            df[column] = df[column].fillna(
                mode_value[0]
            )

print("\n========== MISSING VALUE HANDLING ==========")

print("Numerical missing values filled using Median.")
print("Categorical missing values filled using Mode.")

print("\nRemaining Missing Values:")

print(df.isnull().sum().sum())


# ============================================================
# Step 9: Detect Duplicate Records
# ============================================================

duplicate_count = df.duplicated().sum()

print("\n========== DUPLICATE RECORDS ==========")

print("Number of Duplicate Records:", duplicate_count)

# Remove duplicates
df = df.drop_duplicates()

print("Duplicate Records Removed Successfully.")
print("Remaining Rows:", len(df))


# ============================================================
# Step 10: Standardize Categorical Values
# ============================================================

print("\n========== CATEGORICAL DATA ==========")

categorical_columns = df.select_dtypes(
    exclude=np.number
).columns

for column in categorical_columns:

    # Remove unnecessary spaces
    df[column] = df[column].astype(str).str.strip()

    # Convert text values to consistent lowercase format
    df[column] = df[column].str.lower()

print("Categorical values standardized successfully.")


# ============================================================
# Step 11: Encode Categorical Variables
# ============================================================

# Create a copy for encoding
encoded_df = df.copy()

categorical_columns = encoded_df.select_dtypes(
    exclude=np.number
).columns

label_encoder = LabelEncoder()

for column in categorical_columns:

    encoded_df[column] = label_encoder.fit_transform(
        encoded_df[column].astype(str)
    )

print("Categorical variables encoded successfully.")
print("Encoding Method: Label Encoding")


# ============================================================
# Step 12: Detect Outliers using IQR Method
# ============================================================

print("\n========== OUTLIER DETECTION ==========")

numeric_columns = encoded_df.select_dtypes(
    include=np.number
).columns

outlier_summary = {}

for column in numeric_columns:

    Q1 = encoded_df[column].quantile(0.25)
    Q3 = encoded_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = encoded_df[
        (encoded_df[column] < lower_bound) |
        (encoded_df[column] > upper_bound)
    ]

    outlier_summary[column] = len(outliers)

    print(
        column,
        "-> Outliers:",
        len(outliers)
    )


# ============================================================
# Step 13: Visualize Outliers using Box Plots
# ============================================================

# Select numerical columns for visualization
boxplot_columns = encoded_df.select_dtypes(
    include=np.number
).columns

if len(boxplot_columns) > 0:

    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=encoded_df[boxplot_columns]
    )

    plt.title("Box Plot for Outlier Detection")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

print("Box Plot generated successfully.")


# ============================================================
# Step 14: Treat Outliers using IQR Capping
# ============================================================

for column in numeric_columns:

    Q1 = encoded_df[column].quantile(0.25)
    Q3 = encoded_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    encoded_df[column] = encoded_df[column].clip(
        lower_bound,
        upper_bound
    )

print("Outliers treated using IQR Capping.")


# ============================================================
# Step 15: Normalize / Standardize Numerical Attributes
# ============================================================

numeric_columns = encoded_df.select_dtypes(
    include=np.number
).columns

scaler = StandardScaler()

if len(numeric_columns) > 0:

    encoded_df[numeric_columns] = scaler.fit_transform(
        encoded_df[numeric_columns]
    )

print("\n========== FEATURE SCALING ==========")

print("Numerical attributes standardized successfully.")
print("Scaling Method: Standardization")
print("Mean approximately equals 0.")
print("Standard deviation approximately equals 1.")


# ============================================================
# Step 16: Feature Engineering
# ============================================================

# Titanic-specific feature engineering
#
# If the selected dataset contains SibSp and Parch,
# create FamilySize.

if "sibsp" in encoded_df.columns and "parch" in encoded_df.columns:

    encoded_df["family_size"] = (
        encoded_df["sibsp"] +
        encoded_df["parch"] +
        1
    )

    print("\nFeature Created: family_size")


# If Age is available, create AgeGroup
if "age" in df.columns:

    age_values = df["age"]

    encoded_df["age_group"] = pd.cut(
        age_values,
        bins=[0, 12, 18, 35, 60, 100],
        labels=[
            "child",
            "teenager",
            "adult",
            "middle_aged",
            "senior"
        ]
    )

    # Encode Age Group
    encoded_df["age_group"] = LabelEncoder().fit_transform(
        encoded_df["age_group"].astype(str)
    )

    print("Feature Created: age_group")


# ============================================================
# Step 17: Display Preprocessed Dataset
# ============================================================

print("\n========== PREPROCESSED DATASET ==========")

print(encoded_df.head())


# ============================================================
# Step 18: Display Final Dataset Information
# ============================================================

print("\n========== FINAL DATASET INFORMATION ==========")

print("Rows:", encoded_df.shape[0])
print("Columns:", encoded_df.shape[1])

print("\nMissing Values:")
print(encoded_df.isnull().sum().sum())

print("\nDuplicate Records:")
print(encoded_df.duplicated().sum())


# ============================================================
# Step 19: Save Cleaned Dataset
# ============================================================

output_file = "cleaned_preprocessed_dataset.csv"

encoded_df.to_csv(
    output_file,
    index=False
)

print("\n========== DATASET STORAGE ==========")

print(
    "Cleaned Dataset Saved Successfully:",
    output_file
)


# ============================================================
# Step 20: Final Output
# ============================================================

print("\n========== DATA PREPROCESSING ==========")

print("Dataset Collection: Completed")
print("Dataset Loading: Completed")
print("Data Inspection: Completed")
print("Missing Value Handling: Completed")
print("Duplicate Removal: Completed")
print("Categorical Standardization: Completed")
print("Categorical Encoding: Completed")
print("Outlier Detection: Completed")
print("Outlier Treatment: Completed")
print("Feature Standardization: Completed")
print("Feature Engineering: Completed")
print("Cleaned Dataset Export: Completed")

print("\nObservations:")

print("Missing numerical values were handled using the median.")
print("Missing categorical values were handled using the mode.")
print("Duplicate records were removed.")
print("Categorical values were standardized and encoded.")
print("Outliers were detected using the IQR method.")
print("Outliers were treated using IQR capping.")
print("Numerical features were standardized using StandardScaler.")
print("New features were created where applicable.")

print("\nExperiment Completed Successfully.")


# ============================================================
# EXPERIMENT 1 - ANSWERS TO QUESTIONS
# ============================================================


# ------------------------------------------------------------
# QUESTION 1:
# Why is data preprocessing considered one of the most
# important phases in data analytics?
# ------------------------------------------------------------

# Data preprocessing is important because real-world data is
# often incomplete, inconsistent, duplicated and noisy.
#
# Before performing visualization, statistical analysis or
# machine learning, the data must be cleaned and prepared.
#
# Data preprocessing helps to:
#
# 1. Handle missing values.
# 2. Remove duplicate records.
# 3. Correct inconsistent data.
# 4. Detect and treat outliers.
# 5. Convert categorical variables into numerical form.
# 6. Bring numerical features to a common scale.
#
# Proper preprocessing improves the quality and reliability
# of analytical results.


# ------------------------------------------------------------
# QUESTION 2:
# Explain different methods of handling missing values with
# suitable examples.
# ------------------------------------------------------------

# Missing values can be handled using several methods.
#
# 1. Mean:
#    Missing numerical values can be replaced by the mean
#    of the available values.
#
# 2. Median:
#    Missing numerical values can be replaced by the median.
#    Median is useful when the data contains outliers.
#
# 3. Mode:
#    Missing categorical values can be replaced by the most
#    frequently occurring value.
#
# 4. Row Removal:
#    Rows containing missing values can be removed when the
#    number of missing records is small.
#
# 5. Column Removal:
#    A column can be removed when a very large portion of
#    its values is missing.
#
# Example:
#
# If the Age column contains missing values:
#
# df["Age"] = df["Age"].fillna(df["Age"].median())
#
# This replaces missing Age values with the median age.


# ------------------------------------------------------------
# QUESTION 3:
# Differentiate between Label Encoding and One-Hot Encoding.
# ------------------------------------------------------------

# Label Encoding:
#
# Label Encoding converts categories into numerical labels.
#
# Example:
#
# Male   -> 0
# Female -> 1
#
# It is useful when categories have an inherent order or when
# a compact numerical representation is required.
#
#
# One-Hot Encoding:
#
# One-Hot Encoding creates separate binary columns for each
# category.
#
# Example:
#
# Gender
# Male
# Female
#
# becomes:
#
# Male  Female
# 1       0
# 0       1
#
#
# Main Difference:
#
# Label Encoding -> One numerical value per category.
# One-Hot Encoding -> Separate binary column for each category.


# ------------------------------------------------------------
# QUESTION 4:
# What are outliers? How can they affect analytical results?
# ------------------------------------------------------------

# Outliers are observations that are significantly different
# from the majority of values in a dataset.
#
# They can occur because of:
#
# 1. Measurement errors.
# 2. Data entry errors.
# 3. Unusual events.
# 4. Natural variation.
#
# Outliers can affect:
#
# 1. Mean and standard deviation.
# 2. Statistical analysis.
# 3. Correlation.
# 4. Machine learning models.
# 5. Data visualization.
#
# The IQR method can be used to identify outliers.
#
# IQR = Q3 - Q1
#
# Lower Bound = Q1 - 1.5 × IQR
#
# Upper Bound = Q3 + 1.5 × IQR


# ------------------------------------------------------------
# QUESTION 5:
# Explain the difference between normalization and
# standardization.
# ------------------------------------------------------------

# Normalization:
#
# Normalization generally scales values to a fixed range,
# commonly 0 to 1.
#
# A common formula is:
#
# X_normalized = (X - X_min) / (X_max - X_min)
#
#
# Standardization:
#
# Standardization transforms values so that the feature has
# approximately mean 0 and standard deviation 1.
#
# Formula:
#
# X_standardized = (X - Mean) / Standard Deviation
#
#
# Main Difference:
#
# Normalization -> Fixed range, commonly 0 to 1.
# Standardization -> Mean approximately 0 and standard
#                  deviation approximately 1.


# ------------------------------------------------------------
# QUESTION 6:
# Why should duplicate records be removed before analysis?
# ------------------------------------------------------------

# Duplicate records represent repeated observations of the
# same data.
#
# If duplicates are not removed, they can:
#
# 1. Increase the apparent frequency of certain observations.
# 2. Produce biased statistical results.
# 3. Affect averages and distributions.
# 4. Affect machine learning model training.
# 5. Increase the apparent size of the dataset.
#
# Removing duplicate records helps ensure that each observation
# is represented appropriately in the dataset.


# ------------------------------------------------------------
# QUESTION 7:
# What is feature engineering? Give two practical examples.
# ------------------------------------------------------------

# Feature engineering is the process of creating new useful
# attributes from existing data.
#
# It can help represent information in a form that is more
# useful for analysis or machine learning.
#
# Example 1:
#
# Family Size can be created from:
#
# SibSp + Parch + 1
#
# FamilySize = SibSp + Parch + 1
#
#
# Example 2:
#
# Age Group can be created from Age.
#
# For example:
#
# 0-12   -> Child
# 13-18  -> Teenager
# 19-35  -> Adult
# 36-60  -> Middle Aged
# 60+    -> Senior
#
# Feature engineering can provide additional useful
# information without collecting new raw data.


# ------------------------------------------------------------
# QUESTION 8:
# Which preprocessing techniques would you apply to the
# IBM HR Employee Attrition dataset and why?
# ------------------------------------------------------------

# For the IBM HR Employee Attrition dataset, the following
# preprocessing techniques can be applied:
#
# 1. Missing Value Handling:
#    Identify and appropriately handle missing values.
#
# 2. Duplicate Removal:
#    Remove duplicate employee records.
#
# 3. Categorical Encoding:
#    Convert categorical attributes such as department,
#    job role and marital status into numerical form.
#
# 4. Outlier Detection:
#    Use box plots and the IQR method for numerical variables.
#
# 5. Feature Scaling:
#    Standardize numerical features such as age, income and
#    years of experience where appropriate.
#
# 6. Feature Engineering:
#    Create useful attributes such as age groups or income
#    categories.
#
# These techniques improve data quality and prepare the
# employee dataset for statistical analysis and machine
# learning.


# ------------------------------------------------------------
# QUESTION 9:
# How does poor-quality data affect machine learning model
# performance?
# ------------------------------------------------------------

# Poor-quality data can negatively affect machine learning
# models.
#
# Problems such as:
#
# 1. Missing values.
# 2. Duplicate records.
# 3. Incorrect values.
# 4. Outliers.
# 5. Inconsistent categories.
# 6. Different numerical scales.
#
# can cause models to learn incorrect or misleading patterns.
#
# This may result in:
#
# 1. Lower prediction accuracy.
# 2. Poor generalization.
# 3. Biased predictions.
# 4. Unstable models.
# 5. Reduced reliability.
#
# Therefore, high-quality preprocessing is important for
# building reliable machine learning models.


# ------------------------------------------------------------
# QUESTION 10:
# Name any three Python libraries commonly used for data
# preprocessing.
# ------------------------------------------------------------

# Three commonly used Python libraries are:
#
# 1. Pandas:
#    Used for loading, cleaning, manipulating and analyzing
#    structured datasets.
#
# 2. NumPy:
#    Used for numerical operations and array processing.
#
# 3. Scikit-learn:
#    Provides preprocessing tools such as LabelEncoder,
#    StandardScaler and other machine learning utilities.
#
# Other useful libraries include Matplotlib and Seaborn for
# visualization and exploratory data analysis.