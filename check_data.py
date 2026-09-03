import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

employee_data = pd.read_csv(file_path)

sns.barplot(
    data=employee_data,
    x="PerformanceRating",
    y="PercentSalaryHike",
    estimator="mean"
)

plt.title("Average Salary Hike by Performance Rating")
plt.xlabel("Performance Rating")
plt.ylabel("Average Salary Hike (%)")

plt.tight_layout()
plt.show()