import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


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


# Create training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(
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


# Logistic Regression model
pipeline = Pipeline(
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


# Train model
pipeline.fit(X_train, y_train)

print("Model trained successfully!")


# Get feature names after preprocessing
feature_names = pipeline.named_steps[
    "preprocessing"
].get_feature_names_out()


# Get model coefficients
coefficients = pipeline.named_steps[
    "model"
].coef_[0]


# Create feature importance table
feature_analysis = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})


feature_analysis["Absolute Coefficient"] = (
    feature_analysis["Coefficient"].abs()
)


# Sort by absolute coefficient
feature_analysis = feature_analysis.sort_values(
    "Absolute Coefficient",
    ascending=False
)


print("\nTop 15 Features Associated With Attrition")
print("-" * 55)

print(
    feature_analysis[
        [
            "Feature",
            "Coefficient",
            "Absolute Coefficient"
        ]
    ].head(15).to_string(index=False)
)


# Select top 10 features for visualization
top_features = feature_analysis.head(10).sort_values(
    "Coefficient"
)


# Create feature importance chart
plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

plt.axvline(
    x=0,
    linewidth=1
)

plt.title(
    "Top Features Associated With Employee Attrition"
)

plt.xlabel(
    "Logistic Regression Coefficient"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()


# Save chart
plt.savefig(
    "visualizations/top_attrition_features.png",
    dpi=300
)

plt.show()


print(
    "\nFeature analysis visualization saved successfully!"
)