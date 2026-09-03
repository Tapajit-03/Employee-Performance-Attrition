import pandas as pd
import matplotlib.pyplot as plt


file_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

employee_data = pd.read_csv(file_path)


# Create visualization folder
import os

os.makedirs("visualizations", exist_ok=True)


# --------------------------------------------------
# 1. Overall Attrition Distribution
# --------------------------------------------------

attrition_counts = employee_data["Attrition"].value_counts()

plt.figure(figsize=(7, 5))

plt.bar(
    attrition_counts.index,
    attrition_counts.values
)

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "visualizations/attrition_distribution.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 2. Attrition Rate by Department
# --------------------------------------------------

department_attrition = pd.crosstab(
    employee_data["Department"],
    employee_data["Attrition"],
    normalize="index"
) * 100

department_attrition["Yes"].plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Attrition Rate by Department")
plt.xlabel("Department")
plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "visualizations/attrition_by_department.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 3. Attrition Rate by Overtime
# --------------------------------------------------

overtime_attrition = pd.crosstab(
    employee_data["OverTime"],
    employee_data["Attrition"],
    normalize="index"
) * 100

overtime_attrition["Yes"].plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Attrition Rate by Overtime")
plt.xlabel("Overtime")
plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "visualizations/attrition_by_overtime.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 4. Attrition Rate by Job Satisfaction
# --------------------------------------------------

satisfaction_attrition = pd.crosstab(
    employee_data["JobSatisfaction"],
    employee_data["Attrition"],
    normalize="index"
) * 100

satisfaction_attrition["Yes"].plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Attrition Rate by Job Satisfaction")
plt.xlabel("Job Satisfaction Level")
plt.ylabel("Attrition Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "visualizations/attrition_by_satisfaction.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 5. Performance Rating Distribution
# --------------------------------------------------

performance_counts = employee_data[
    "PerformanceRating"
].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    performance_counts.index.astype(str),
    performance_counts.values
)

plt.title("Employee Performance Rating Distribution")
plt.xlabel("Performance Rating")
plt.ylabel("Number of Employees")

plt.tight_layout()

plt.savefig(
    "visualizations/performance_rating_distribution.png",
    dpi=300
)

plt.show()


print("\nAll visualizations created successfully!")

print("\nFiles saved in:")
print("visualizations/")