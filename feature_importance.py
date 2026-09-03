import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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


# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)


# Train balanced Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)


# Get feature coefficients
feature_coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
})

feature_coefficients["Absolute Coefficient"] = (
    feature_coefficients["Coefficient"].abs()
)

feature_coefficients = feature_coefficients.sort_values(
    "Absolute Coefficient",
    ascending=False
)


print("Top 15 Features Associated With Attrition")
print("-" * 50)

print(
    feature_coefficients
    .head(15)
    .round(4)
    .to_string(index=False)
)