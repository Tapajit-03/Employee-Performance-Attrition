# Employee Performance & Attrition Analysis

## Project Overview

This project analyzes employee data to identify factors associated with employee attrition and develops machine learning models to predict whether an employee is likely to leave an organization.

The project combines exploratory data analysis, data preprocessing, machine learning, model evaluation, feature analysis, and business insights to provide a complete end-to-end employee attrition analysis workflow.

### Key Components

* Exploratory Data Analysis (EDA)
* Data preprocessing and cleaning
* Categorical encoding and feature scaling
* Machine learning model development
* Model comparison
* Hyperparameter tuning
* Classification threshold analysis
* Feature importance and interpretation
* Business insights and HR recommendations

---

## Business Problem

Employee turnover can increase recruitment, training, and operational costs for an organization.

The objective of this project is to:

1. Understand employee attrition patterns.
2. Identify factors associated with employee turnover.
3. Build machine learning models to predict employee attrition.
4. Compare different classification models using appropriate evaluation metrics.
5. Analyze important features associated with employee attrition.
6. Translate machine learning results into actionable HR recommendations.

---

## Dataset

The project uses an employee dataset containing information related to demographics, job characteristics, satisfaction, compensation, work experience, and performance.

### Important Features

* Age
* Department
* Job Role
* Job Satisfaction
* Environment Satisfaction
* Business Travel
* Overtime
* Monthly Income
* Job Level
* Total Working Years
* Years at Company
* Performance Rating
* Work-Life Balance
* And other employee-related features

### Dataset Size

* **Rows:** 1,470
* **Columns:** 35

### Target Variable

`Attrition`

* **No:** 1,233 employees
* **Yes:** 237 employees
* **Overall attrition rate:** 16.12%

The dataset is imbalanced, with significantly fewer employees leaving than staying. Therefore, metrics such as **Precision, Recall, F1 Score, and ROC-AUC** are considered alongside accuracy.

---

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Joblib**

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
```

---

## Machine Learning Models

The project evaluates multiple classification approaches, including:

* Logistic Regression
* Random Forest
* Tuned classification models
* A final prediction pipeline

### Model Comparison

The initial model comparison produced the following results:

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   79.09% |    40.68% | 68.57% |   51.06% |  80.57% |
| Random Forest       |   80.91% |    33.33% | 20.00% |   25.00% |  70.66% |

Although Random Forest achieved higher accuracy, Logistic Regression provided substantially better recall, F1 score, and ROC-AUC in the comparison.

This demonstrates why accuracy alone is not sufficient when evaluating an imbalanced attrition prediction problem.

---

## Threshold Analysis

Classification threshold analysis was performed to investigate the trade-off between precision and recall.

A threshold of **0.60** produced the best F1 score in the evaluated threshold range:

* **Precision:** 45.76%
* **Recall:** 57.45%
* **F1 Score:** 50.94%

The threshold can be adjusted depending on the organization's priorities. For example, an HR team that wants to identify more potentially at-risk employees may prefer a threshold that prioritizes recall.

---

## Key Business Insights

The analysis investigates several employee characteristics that may be associated with attrition, including:

* Overtime
* Job satisfaction
* Environment satisfaction
* Job role
* Monthly income
* Business travel
* Years at company
* Job level
* Work-life balance
* Performance-related factors

The project also analyzes attrition across departments and other employee characteristics to identify patterns that may help HR teams understand potential retention risks.

> **Important:** Correlation or model-based importance does not necessarily imply that a feature directly causes an employee to leave.

---

## Visualizations

The project generates visualizations covering:

* Overall attrition distribution
* Attrition by department
* Attrition by overtime
* Attrition by satisfaction
* Performance rating distribution
* Top features associated with attrition

Generated visualizations are available in the [`visualizations/`](visualizations/) directory.

---

## Business Recommendations

Based on the analysis, organizations could consider:

1. **Monitor overtime patterns** and investigate excessive workloads.
2. **Track employee satisfaction** to identify potential retention risks.
3. **Analyze department and job-role-specific attrition patterns.**
4. **Use employee compensation and career progression data** as part of broader retention analysis.
5. **Use predictive models as decision-support tools**, rather than automatically making decisions about individual employees.
6. **Regularly retrain and evaluate the model** as employee behavior and organizational conditions change.

---

## Project Structure

```text
Employee-Performance-Attrition/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── models/
│   └── final_model.pkl
│
├── output/
│   └── business_insights.txt
│
├── visualizations/
│   ├── attrition_by_department.png
│   ├── attrition_by_overtime.png
│   ├── attrition_by_satisfaction.png
│   ├── attrition_distribution.png
│   ├── performance_rating_distribution.png
│   └── top_attrition_features.png
│
├── business_insights.py
├── check_data.py
├── feature_analysis.py
├── feature_importance.py
├── final_model.py
├── model_comparison.py
├── model_evaluation.py
├── model_tuning.py
├── pipeline_model.py
├── preprocess.py
├── random_forest.py
├── threshold_analysis.py
├── train_model.py
├── visualizations.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Tapajit-03/Employee-Performance-Attrition.git
```

### 2. Navigate to the project directory

```bash
cd Employee-Performance-Attrition
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the required Python scripts

Individual scripts can be executed according to the project workflow. For example:

```bash
python check_data.py
python preprocess.py
python train_model.py
python model_comparison.py
python model_evaluation.py
python feature_analysis.py
python business_insights.py
```

---

## Project Outcome

This project demonstrates an end-to-end machine learning workflow, starting from raw employee data and progressing through preprocessing, exploratory analysis, model development, evaluation, threshold optimization, feature interpretation, and business recommendations.

The project also highlights an important machine learning principle: **the best model should not be selected based solely on accuracy.** For an imbalanced classification problem such as employee attrition, recall, precision, F1 score, and ROC-AUC can provide more useful information about model performance.

---

## Future Improvements

Potential future improvements include:

* Testing additional classification algorithms.
* Applying more advanced hyperparameter optimization.
* Handling class imbalance using techniques such as class weighting or resampling.
* Building an interactive dashboard for HR analysis.
* Deploying the prediction model as a web application or API.
* Monitoring model performance after deployment.
* Performing additional fairness and bias analysis before using predictions in real-world HR decision-making.
