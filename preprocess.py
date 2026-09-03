import pandas as pd
from sklearn.model_selection import train_test_split

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

# Convert the target into numbers
y = y.map({
    "No": 0,
    "Yes": 1
})

# Identify categorical columns
categorical_columns = X.select_dtypes(include=["str"]).columns

# Convert categorical variables into numerical variables
X_encoded = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())