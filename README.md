# Employee Performance & Attrition Analysis

## Project Overview

This project analyzes employee data to identify factors associated with employee attrition and builds machine learning models to predict whether an employee is likely to leave the organization.

The project combines:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature encoding and scaling
- Machine learning
- Model comparison
- Hyperparameter tuning
- Threshold analysis
- Feature interpretation
- Business recommendations

---

## Business Problem

Employee turnover can increase recruitment, training, and operational costs.

The objective of this project is to:

1. Understand employee attrition patterns.
2. Identify factors associated with employee turnover.
3. Build a machine learning model for attrition prediction.
4. Evaluate different models using appropriate classification metrics.
5. Translate the results into actionable HR recommendations.

---

## Dataset

The dataset contains information about employees, including:

- Age
- Department
- Job Role
- Job Satisfaction
- Environment Satisfaction
- Business Travel
- Overtime
- Monthly Income
- Job Level
- Total Working Years
- Years at Company
- Performance Rating
- Work-Life Balance
- And other employee-related features

### Dataset Size

Original dataset:

- Rows: 1,470
- Columns: 35

Target variable:

`Attrition`

- No: 1,233 employees
- Yes: 237 employees

Overall attrition rate:

**16.12%**

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- OpenPyXL

---

## Project Workflow

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Exploratory Data Analysis
     ↓
Data Cleaning
     ↓
Feature Selection
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Train / Validation / Test Split
     ↓
Model Training
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Threshold Analysis
     ↓
Final Model Evaluation
     ↓
Feature Analysis
     ↓
Business Insights