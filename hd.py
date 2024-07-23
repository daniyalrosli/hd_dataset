# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
# Replace 'your_dataset.csv' with the path to your actual dataset
df = pd.read_excel('fyp dataset.xlsx')

# Display the first few rows of the dataset
print(df.head())

# Step 1: Data Cleaning
# Handling missing values
# For numerical columns, we can fill missing values with the median
# For categorical columns, we can fill missing values with the most frequent value

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Step 2: Feature Engineering
# Create new features or modify existing ones if necessary

# Example: Creating a new feature 'age_group' based on 'age'
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 70, 80], labels=['<30', '30-39', '40-49', '50-59', '60-69', '70+'])

# Step 3: Data Transformation
# Combining numerical and categorical transformations using ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Step 4: Splitting the Data
# Separate target variable 'target' from features

X = df.drop(['target'], axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the transformations to the training and testing data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Display the transformed data
print(X_train_transformed[:5])
print(X_test_transformed[:5])