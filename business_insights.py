import pandas as pd
import os


# Load dataset
file_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

employee_data = pd.read_csv(file_path)


# Create output folder if it does not exist
os.makedirs("output", exist_ok=True)


# --------------------------------------------------
# Overall Attrition
# --------------------------------------------------

total_employees = len(employee_data)

employees_left = (
    employee_data["Attrition"] == "Yes"
).sum()

overall_attrition_rate = (
    employees_left / total_employees
) * 100


# --------------------------------------------------
# Department Analysis
# --------------------------------------------------

department_table = pd.crosstab(
    employee_data["Department"],
    employee_data["Attrition"]
)

department_table["Total Employees"] = (
    department_table.sum(axis=1)
)

department_table["Attrition Rate (%)"] = (
    department_table["Yes"]
    / department_table["Total Employees"]
) * 100

highest_department = (
    department_table["Attrition Rate (%)"]
    .idxmax()
)

highest_department_rate = (
    department_table.loc[
        highest_department,
        "Attrition Rate (%)"
    ]
)


# --------------------------------------------------
# Overtime Analysis
# --------------------------------------------------

overtime_table = pd.crosstab(
    employee_data["OverTime"],
    employee_data["Attrition"]
)

overtime_table["Total Employees"] = (
    overtime_table.sum(axis=1)
)

overtime_table["Attrition Rate (%)"] = (
    overtime_table["Yes"]
    / overtime_table["Total Employees"]
) * 100

overtime_yes_rate = (
    overtime_table.loc[
        "Yes",
        "Attrition Rate (%)"
    ]
)

overtime_no_rate = (
    overtime_table.loc[
        "No",
        "Attrition Rate (%)"
    ]
)


# --------------------------------------------------
# Job Satisfaction Analysis
# --------------------------------------------------

satisfaction_table = pd.crosstab(
    employee_data["JobSatisfaction"],
    employee_data["Attrition"]
)

satisfaction_table["Total Employees"] = (
    satisfaction_table.sum(axis=1)
)

satisfaction_table["Attrition Rate (%)"] = (
    satisfaction_table["Yes"]
    / satisfaction_table["Total Employees"]
) * 100

lowest_satisfaction = (
    satisfaction_table["Attrition Rate (%)"]
    .idxmax()
)

lowest_satisfaction_rate = (
    satisfaction_table.loc[
        lowest_satisfaction,
        "Attrition Rate (%)"
    ]
)


# --------------------------------------------------
# Generate Business Report
# --------------------------------------------------

report = []

report.append(
    "EMPLOYEE ATTRITION ANALYSIS - BUSINESS INSIGHTS"
)

report.append("=" * 55)

report.append(
    f"\nTotal employees analyzed: {total_employees}"
)

report.append(
    f"Employees who left: {employees_left}"
)

report.append(
    f"Overall attrition rate: "
    f"{overall_attrition_rate:.2f}%"
)


# Overtime finding
report.append("\n\n1. OVERTIME RISK")

report.append(
    f"Employees working overtime had an attrition "
    f"rate of {overtime_yes_rate:.2f}%."
)

report.append(
    f"Employees without overtime had an attrition "
    f"rate of {overtime_no_rate:.2f}%."
)

report.append(
    "Recommendation: HR should monitor overtime "
    "workloads and investigate whether excessive "
    "work hours are contributing to employee turnover."
)


# Department finding
report.append("\n\n2. DEPARTMENT RISK")

report.append(
    f"The {highest_department} department had the "
    f"highest observed attrition rate at "
    f"{highest_department_rate:.2f}%."
)

report.append(
    "Recommendation: Management should examine "
    "workload, compensation, career progression, "
    "and employee satisfaction within higher-risk departments."
)


# Satisfaction finding
report.append("\n\n3. JOB SATISFACTION")

report.append(
    f"Employees with job satisfaction level "
    f"{lowest_satisfaction} had an attrition rate "
    f"of {lowest_satisfaction_rate:.2f}%."
)

report.append(
    "Recommendation: HR could conduct employee "
    "feedback surveys and identify workplace factors "
    "associated with low job satisfaction."
)


# ML findings
report.append("\n\n4. MACHINE LEARNING FINDINGS")

report.append(
    "The logistic regression model identified "
    "overtime, business travel, job role, marital "
    "status, job level, and career progression "
    "variables among the strongest model-associated "
    "features."
)

report.append(
    "These factors should be treated as indicators "
    "for further investigation rather than direct "
    "causes of employee attrition."
)


# Final recommendation
report.append("\n\n5. MANAGEMENT RECOMMENDATIONS")

report.append(
    "• Monitor overtime and workload patterns."
)

report.append(
    "• Focus retention analysis on higher-risk departments."
)

report.append(
    "• Investigate low job satisfaction through employee feedback."
)

report.append(
    "• Review career progression and promotion opportunities."
)

report.append(
    "• Use the ML model as a decision-support tool rather "
    "than as the sole basis for HR decisions."
)


# Save report
report_path = "output/business_insights.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("\n".join(report))


print("Business insights report created successfully!")

print(
    f"\nReport saved to: {report_path}"
)