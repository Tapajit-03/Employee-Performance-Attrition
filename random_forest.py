import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


file_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

employee_data = pd.read_csv(file_path)


# Remove columns that do not provide useful information
columns_to_remove = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

employee_data = employee_data.drop(columns=columns_to_remove)


# Separate features and target
X = employee_data.drop("Attrition", axis=1)
y = employee_data["Attrition"]


# Convert target values into numbers
y = y.map({
    "No": 0,
    "Yes": 1
})


# Convert categorical variables into numerical variables
categorical_columns = X.select_dtypes(include=["str"]).columns

X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)


# Train the model
model.fit(X_train, y_train)


print("Random Forest training completed successfully!")


# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\nRandom Forest Evaluation")
print("-" * 30)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))