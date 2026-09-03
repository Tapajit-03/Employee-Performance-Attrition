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
    confusion_matrix,
    classification_report
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


# Create training and temporary sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# Create validation and final test sets
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("Dataset split")
print("-" * 30)

print("Training set:", X_train.shape)
print("Validation set:", X_validation.shape)
print("Final test set:", X_test.shape)


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


# Selected model
final_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "model",
            LogisticRegression(
                C=1,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# Combine training and validation data
X_train_final = pd.concat(
    [X_train, X_validation]
)

y_train_final = pd.concat(
    [y_train, y_validation]
)


print("\nTraining final model...")


# Train using training + validation data
final_pipeline.fit(
    X_train_final,
    y_train_final
)


print("Final model training completed successfully!")


# Make predictions on the untouched test set
y_pred = final_pipeline.predict(X_test)

y_probability = final_pipeline.predict_proba(
    X_test
)[:, 1]


# Calculate evaluation metrics
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\nFINAL MODEL EVALUATION")
print("=" * 40)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))


print("\nConfusion Matrix")
print("-" * 30)

print(confusion_matrix(y_test, y_pred))


print("\nClassification Report")
print("-" * 30)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Left"]
    )
)
import joblib

joblib.dump(final_pipeline, "models/final_model.pkl")

print("Final model saved successfully!")