<div align="center">

# 🧠 Employee Job Satisfaction Prediction using Machine Learning

### Predicting employee satisfaction ratings using workplace and demographic data

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=plotly&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-3776AB?style=for-the-badge)](https://seaborn.pydata.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-blueviolet?style=for-the-badge)](#)
[![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-2E8B57?style=for-the-badge)](#)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rizwanahmed786508)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#-license)
[![Repo Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](#)
[![Last Commit](https://img.shields.io/badge/Last%20Commit-2026-informational?style=for-the-badge)](#)

</div>

---

## 📌 Project Overview

This project focuses on predicting **employee job satisfaction ratings** using a supervised Machine Learning approach. Organizations often struggle to understand what drives employee satisfaction, and this project demonstrates how workplace data — such as salary, work-life balance, manager support, experience, and promotion history — can be used to predict how satisfied an employee is likely to be, on a rating scale of 1 to 5.

The project follows a complete, real-world Machine Learning workflow: starting from raw and messy data, moving through data cleaning and exploratory data analysis, then preprocessing the data for modeling, and finally training and evaluating a **Random Forest Classifier**. Every step, from handling missing values to interpreting feature importance, is documented to reflect professional data science practices.

This repository is built as part of an **internship task (NGC)** and is intended to showcase a clean, end-to-end classification pipeline suitable for a Machine Learning / Data Science portfolio.

---

## ✨ Features

- 📂 End-to-end Machine Learning pipeline from raw CSV data to trained model
- 🧹 Robust data cleaning: missing value imputation, duplicate removal, and text standardization
- 📊 Exploratory Data Analysis with count plots and a correlation heatmap
- 🔢 Categorical feature encoding using Label Encoding
- 🤖 Classification model built using **Random Forest Classifier**
- 📈 Full model evaluation: accuracy, classification report, confusion matrix, and feature importance
- 🧾 Clear, well-commented notebook explaining every step in simple terms

---

## 📊 Dataset

- **Original Rows:** 1,030
- **Rows After Cleaning:** 984
- **Total Columns:** 9
- **Target Variable:** `Satisfaction_Rating` (rating scale of 1–5)

**Features used:**

| Column | Description |
|---|---|
| `Employee_ID` | Unique identifier for each employee |
| `Department` | Department the employee belongs to |
| `Age` | Age of the employee |
| `Experience_Years` | Total years of work experience |
| `Salary_PKR` | Employee salary in PKR |
| `Work_Life_Balance` | Work-life balance rating |
| `Manager_Support` | Manager support rating |
| `Promotion_Last_2Y` | Whether the employee was promoted in the last 2 years (Yes/No) |
| `Satisfaction_Rating` | Target variable — employee satisfaction rating (1–5) |

The purpose of the dataset is to capture workplace and demographic factors that can be used to predict how satisfied an employee is with their job.

---

## 🧹 Data Cleaning

The following cleaning steps were performed on the raw dataset:

- **Missing Value Handling**
  - `Department` — missing values filled using **Mode** (categorical feature)
  - `Salary_PKR` — missing values filled using **Median** (numerical feature, less affected by outliers)
  - `Manager_Support` — missing values filled using **Median** (ordinal numerical feature)
- **Duplicate Removal** — 26 duplicate rows identified and removed (1,030 → 1,004 rows)
- **Text Standardization**
  - Department names converted to lowercase
  - Leading/trailing whitespace stripped
  - Inconsistent department names standardized (e.g., "human resources" → "hr", "information technology" → "it", "finance department" → "finance")
- **Invalid Data Removal**
  - Removed rows where `Age` was less than 18
  - Removed rows where `Salary_PKR` was less than or equal to 0
  - Final cleaned dataset: **984 rows**

---

## 📈 Exploratory Data Analysis (EDA)

- **Count Plot** — Visualizes the distribution of employee `Satisfaction_Rating` values (1–5) across the dataset.
- **Department Distribution** — A horizontal count plot showing the number of employees in each department.
- **Correlation Heatmap** — Displays correlation values between all numerical features, helping identify relationships between variables prior to model training.

---

## ⚙️ Data Preprocessing

- **Label Encoding**
  - `Department` converted into numeric values
  - `Promotion_Last_2Y` (Yes/No) converted into numeric values
- **Feature Selection**
  - `X` — all columns except `Satisfaction_Rating`
  - `y` — `Satisfaction_Rating` (target variable)
- **Train-Test Split**
  - Dataset split into **80% training data** and **20% testing data**
  - `random_state=42` used to ensure reproducible results

---

## 🤖 Machine Learning Model

**Random Forest Classifier**

- The model was created using `RandomForestClassifier(random_state=42)`
- Trained on the training set (`X_train`, `y_train`)
- Used to predict `Satisfaction_Rating` on the test set (`X_test`)

---

## 📊 Results

### ✅ Accuracy

```
Accuracy: 83.25%
```

### 📋 Classification Report

| Class | Precision | Recall | F1-score | Support |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.89 | 0.89 | 0.89 | 46 |
| 2 | 0.83 | 0.74 | 0.78 | 46 |
| 3 | 0.77 | 0.86 | 0.81 | 42 |
| 4 | 0.77 | 0.79 | 0.78 | 34 |
| 5 | 0.93 | 0.90 | 0.91 | 29 |

| Metric | Value |
|---|---|
| Accuracy | 0.83 |
| Macro Avg (Precision / Recall / F1) | 0.84 / 0.84 / 0.84 |
| Weighted Avg (Precision / Recall / F1) | 0.83 / 0.83 / 0.83 |
| Total Support | 197 |

### 🔷 Confusion Matrix

A confusion matrix was generated to visualize actual vs. predicted `Satisfaction_Rating` values on the test set (see `Screenshots` section below).

### 🔑 Feature Importance

| Feature | Importance |
|---|---:|
| Salary_PKR | 0.226277 |
| Manager_Support | 0.180701 |
| Work_Life_Balance | 0.173234 |
| Employee_ID | 0.108891 |
| Age | 0.092336 |
| Experience_Years | 0.090801 |
| Department | 0.065780 |
| Promotion_Last_2Y | 0.061978 |

`Salary_PKR`, `Manager_Support`, and `Work_Life_Balance` were the strongest predictors of employee satisfaction.

---

## 📁 Project Structure

```
employee-job-satisfaction-prediction/
│
├── dataset/
│   └── NGC_Employee_Job_Satisfaction_Messy_Dataset_1000Rows.csv
│
├── notebook/
│   └── Employee_Job_Satisfaction_Prediction.ipynb
│
├── presentation/
│   └── index.html
│
├── images/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── feature_correlatiob_heatmap.png
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore                    # Visualizations (EDA, confusion matrix, etc.)
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/rizwanahmed786508/Employee-Job-Satisfaction-Prediction.git

# Navigate to the project directory
cd Employee-Job-Satisfaction-Prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

---

## 🛠️ Technologies Used

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](#)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white)](#)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square)](#)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](#)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](#)

- **Pandas & NumPy** — Data loading, manipulation, and numerical operations
- **Matplotlib & Seaborn** — Data visualization and exploratory analysis
- **Scikit-Learn** — Label encoding, train-test split, model training, and evaluation metrics
- **Random Forest Classifier** — Core machine learning model used for prediction

---

## 🔮 Future Improvements

- Train additional machine learning models such as **Decision Tree** and **XGBoost**
- Perform **Hyperparameter Tuning** to further optimize model performance
- Deploy the model using **Streamlit** for an interactive web application
- Collect real employee survey data for better prediction accuracy

---

## 🖼️ Screenshots



| Section | Preview |
|---|---|
| Dataset Preview | `screenshots/dataset_preview.png` |
| EDA — Satisfaction Rating Distribution | `screenshots/eda_satisfaction_distribution.png` |
| EDA — Department Distribution | `screenshots/eda_department_distribution.png` |
| EDA — Correlation Heatmap | `screenshots/eda_correlation_heatmap.png` |
| Confusion Matrix | `screenshots/confusion_matrix.png` |
| Feature Importance | `screenshots/feature_importance.png` |

---

## 👤 Author

**Rizwan Ahmed**

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/rizwanahmed786508)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rizwanahmed78)

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this project with proper attribution.

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

**Made with 📊 and Machine Learning by Rizwan Ahmed**

</div>
