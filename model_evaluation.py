import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


file_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

employee_data = pd.read_csv(file_path)


# Remove unnecessary columns
columns_to_remove = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

employee_data = employee_data.drop(columns=columns_to_remove)


# Separate features and target
X = employee_data.drop("Attrition", axis=1)

y = employee_data["Attrition"].map({
    "No": 0,
    "Yes": 1
})


# Identify numerical and categorical columns
numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["str"]
).columns.tolist()


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)


# Complete machine learning pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# Train the model
model_pipeline.fit(X_train, y_train)

print("Model training completed successfully!")


# Predictions
y_pred = model_pipeline.predict(X_test)

# Probability of attrition
y_probability = model_pipeline.predict_proba(X_test)[:, 1]


# Basic evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_probability)


print("\nModel Evaluation")
print("-" * 35)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))


print("\nClassification Report")
print("-" * 35)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Left"]
    )
)


print("Confusion Matrix")
print("-" * 35)

print(confusion_matrix(y_test, y_pred))