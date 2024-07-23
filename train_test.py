# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the dataset
# Replace 'your_dataset.csv' with the path to your actual dataset
df = pd.read_csv('your_dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Step 1: Data Cleaning
# Handling missing values
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
# Example: Creating a new feature 'age_group' based on 'age'
if 'age' in df.columns:
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
X = df.drop(['target'], axis=1)  # Replace 'target' with the name of your target column
y = df['target']  # Replace 'target' with the name of your target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the transformations to the training and testing data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Define a function to train and evaluate a model
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    return accuracy, precision, recall, f1, conf_matrix, class_report

# Initialize the models
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Train and evaluate Decision Tree
dt_metrics = train_and_evaluate(dt_model, X_train_transformed, y_train, X_test_transformed, y_test)
print("Decision Tree Metrics:\n")
print(f"Accuracy: {dt_metrics[0]}")
print(f"Precision: {dt_metrics[1]}")
print(f"Recall: {dt_metrics[2]}")
print(f"F1 Score: {dt_metrics[3]}")
print(f"Confusion Matrix:\n {dt_metrics[4]}")
print(f"Classification Report:\n {dt_metrics[5]}")

# Train and evaluate Random Forest
rf_metrics = train_and_evaluate(rf_model, X_train_transformed, y_train, X_test_transformed, y_test)
print("\nRandom Forest Metrics:\n")
print(f"Accuracy: {rf_metrics[0]}")
print(f"Precision: {rf_metrics[1]}")
print(f"Recall: {rf_metrics[2]}")
print(f"F1 Score: {rf_metrics[3]}")
print(f"Confusion Matrix:\n {rf_metrics[4]}")
print(f"Classification Report:\n {rf_metrics[5]}")

# Train and evaluate Logistic Regression
lr_metrics = train_and_evaluate(lr_model, X_train_transformed, y_train, X_test_transformed, y_test)
print("\nLogistic Regression Metrics:\n")
print(f"Accuracy: {lr_metrics[0]}")
print(f"Precision: {lr_metrics[1]}")
print(f"Recall: {lr_metrics[2]}")
print(f"F1 Score: {lr_metrics[3]}")
print(f"Confusion Matrix:\n {lr_metrics[4]}")
print(f"Classification Report:\n {lr_metrics[5]}")

# Comparing Models
models_metrics = {
    "Decision Tree": dt_metrics,
    "Random Forest": rf_metrics,
    "Logistic Regression": lr_metrics
}

best_model = None
best_f1_score = 0

for model_name, metrics in models_metrics.items():
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {metrics[0]}")
    print(f"Precision: {metrics[1]}")
    print(f"Recall: {metrics[2]}")
    print(f"F1 Score: {metrics[3]}")
    
    if metrics[3] > best_f1_score:  # Choose the model with the highest F1 score
        best_f1_score = metrics[3]
        best_model = model_name

print(f"\nBest Model: {best_model} with F1 Score: {best_f1_score}")