import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load dataset
file_path = '/Users/daniyalrosli/Documents/isp610_cleaned.csv'
dataset = pd.read_excel(file_path)

# Handle missing values
dataset.fillna(method='ffill', inplace=True)

# One-hot encoding for categorical variables
dataset = pd.get_dummies(dataset, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

# Feature scaling
scaler = StandardScaler()
numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

# Split the dataset into training and testing sets
X = dataset.drop('HeartDisease', axis=1)
y = dataset['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize the data
sns.pairplot(dataset)
plt.show()
