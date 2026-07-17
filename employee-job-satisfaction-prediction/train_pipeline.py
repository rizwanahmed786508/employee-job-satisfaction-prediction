"""
Training pipeline for the Employee Job Satisfaction Prediction project.

This script reproduces the exact data cleaning, feature engineering, and
model comparison steps performed in the original research notebook
(notebook/employee_satisfication_prediction.ipynb), then persists:
  - the trained model, label encoders, and scaler (model/*.pkl)
  - a metrics bundle used to power the Streamlit Analytics & Model Info
    pages without needing to retrain on every app run (artifacts/metrics.json)
  - static analytics images used on the Analytics page (artifacts/*.png)

Run this once (`python train_pipeline.py`) before launching the app.
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "NGC_Employee_Job_Satisfaction_Messy_Dataset_1000Rows.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

sns.set_theme(style="darkgrid")
plt.rcParams["figure.facecolor"] = "#0b0f1a"
plt.rcParams["axes.facecolor"] = "#0b0f1a"
plt.rcParams["savefig.facecolor"] = "#0b0f1a"
plt.rcParams["text.color"] = "#e5e7eb"
plt.rcParams["axes.labelcolor"] = "#e5e7eb"
plt.rcParams["xtick.color"] = "#9ca3af"
plt.rcParams["ytick.color"] = "#9ca3af"
plt.rcParams["axes.edgecolor"] = "#374151"

metrics = {}

# ---------------------------------------------------------------------------
# 1. Load raw (messy) data
# ---------------------------------------------------------------------------
df_raw = pd.read_csv(DATA_PATH)
metrics["dataset_original_rows"] = int(df_raw.shape[0])
metrics["dataset_original_cols"] = int(df_raw.shape[1])
metrics["missing_before"] = {k: int(v) for k, v in df_raw.isnull().sum().items()}
metrics["duplicates_found"] = int(df_raw.duplicated().sum())

df = df_raw.copy()

# ---------------------------------------------------------------------------
# 2. Missing value handling
# ---------------------------------------------------------------------------
df["Department"] = df["Department"].fillna(df["Department"].mode()[0])
df["Salary_PKR"] = df["Salary_PKR"].fillna(df["Salary_PKR"].median())
df["Manager_Support"] = df["Manager_Support"].fillna(df["Manager_Support"].median())

# ---------------------------------------------------------------------------
# 3. Duplicate removal
# ---------------------------------------------------------------------------
shape_before_dupes = df.shape[0]
df.drop_duplicates(inplace=True)
shape_after_dupes = df.shape[0]

# ---------------------------------------------------------------------------
# 4. Standardize Department strings
# ---------------------------------------------------------------------------
df["Department"] = df["Department"].str.lower().str.strip()
df["Department"] = df["Department"].replace({
    "human resources": "hr",
    "information technology": "it",
    "finance department": "finance",
})

# ---------------------------------------------------------------------------
# 5. Validation filters
# ---------------------------------------------------------------------------
df = df[df["Age"] >= 18]
df = df[df["Salary_PKR"] > 0]

metrics["dataset_cleaned_rows"] = int(df.shape[0])
metrics["dataset_cleaned_cols"] = int(df.shape[1])
metrics["rows_removed_duplicates"] = int(shape_before_dupes - shape_after_dupes)

metrics["satisfaction_distribution"] = {
    str(k): int(v) for k, v in df["Satisfaction_Rating"].value_counts().sort_index().items()
}
metrics["department_distribution"] = {
    str(k): int(v) for k, v in df["Department"].value_counts().items()
}

# ---------------------------------------------------------------------------
# 6. Encode categoricals
# ---------------------------------------------------------------------------
department_encoder = LabelEncoder()
promotion_encoder = LabelEncoder()

df["Department"] = department_encoder.fit_transform(df["Department"])
df["Promotion_Last_2Y"] = promotion_encoder.fit_transform(df["Promotion_Last_2Y"])

metrics["department_classes"] = list(department_encoder.classes_)
metrics["promotion_classes"] = list(promotion_encoder.classes_)

# ---------------------------------------------------------------------------
# 7. Feature / target split
# ---------------------------------------------------------------------------
X = df.drop(["Employee_ID", "Satisfaction_Rating"], axis=1)
y = df["Satisfaction_Rating"]
feature_columns = list(X.columns)
metrics["feature_columns"] = feature_columns

# correlation heatmap data (numeric cols incl target)
corr_df = df.drop(columns=["Employee_ID"]).corr(numeric_only=True)
metrics["correlation_matrix"] = {
    col: {row: float(corr_df.loc[row, col]) for row in corr_df.index} for col in corr_df.columns
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 8. Train & compare models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
}

results = []
trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    trained_models[name] = model
    results.append({
        "model": name,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
    })

results_sorted = sorted(results, key=lambda r: r["accuracy"], reverse=True)
metrics["model_comparison"] = results_sorted

best_model_name = results_sorted[0]["model"]
best_model = trained_models[best_model_name]
metrics["best_model_name"] = best_model_name
metrics["best_model_accuracy"] = results_sorted[0]["accuracy"]

# ---------------------------------------------------------------------------
# 9. Persist model, encoders, scaler
# ---------------------------------------------------------------------------
joblib.dump(best_model, os.path.join(MODEL_DIR, "employee_satisfaction_model.pkl"))
joblib.dump(department_encoder, os.path.join(MODEL_DIR, "department_encoder.pkl"))
joblib.dump(promotion_encoder, os.path.join(MODEL_DIR, "promotion_encoder.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

# ---------------------------------------------------------------------------
# 10. Evaluation artifacts for best model
# ---------------------------------------------------------------------------
y_pred_best = best_model.predict(X_test_scaled)
report = classification_report(y_test, y_pred_best, output_dict=True, zero_division=0)
metrics["classification_report"] = report

cm = confusion_matrix(y_test, y_pred_best)
metrics["confusion_matrix"] = cm.tolist()
metrics["confusion_matrix_labels"] = sorted(y_test.unique().tolist())

if hasattr(best_model, "feature_importances_"):
    importance = sorted(
        zip(feature_columns, best_model.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    metrics["feature_importance"] = [{"feature": f, "importance": round(v, 6)} for f, v in importance]
else:
    metrics["feature_importance"] = []

metrics["train_size"] = int(X_train.shape[0])
metrics["test_size"] = int(X_test.shape[0])
metrics["n_features"] = int(X.shape[1])
metrics["target_variable"] = "Satisfaction_Rating"
metrics["algorithms_tested"] = list(models.keys())

with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Training complete.")
print(f"Best model: {best_model_name} ({metrics['best_model_accuracy']}% accuracy)")
print(f"Artifacts saved to: {MODEL_DIR} and {ARTIFACT_DIR}")
