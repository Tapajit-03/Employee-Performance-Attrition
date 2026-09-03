import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


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


# Identify feature types
numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

categorical_columns = X.select_dtypes(include=["str"]).columns.tolist()


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Create preprocessing pipeline
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


# Create model pipeline
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


# Train model
model_pipeline.fit(X_train, y_train)


# Get probability of attrition
y_probability = model_pipeline.predict_proba(X_test)[:, 1]


print("Threshold Analysis")
print("-" * 50)

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

for threshold in thresholds:

    y_pred_threshold = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred_threshold,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )