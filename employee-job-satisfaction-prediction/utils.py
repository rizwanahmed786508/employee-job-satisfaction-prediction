"""
utils.py
Shared helpers: artifact loading (cached), formatting utilities, and the
central design-token palette used across the app.
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PATH = os.path.join(
    BASE_DIR, "dataset", "NGC_Employee_Job_Satisfaction_Messy_Dataset_1000Rows.csv"
)

# ---------------------------------------------------------------------------
# Design tokens — single source of truth for colors used in Python-side
# chart generation (Plotly). CSS-side tokens live in styles.css.
# ---------------------------------------------------------------------------
PALETTE = {
    "bg": "#0b0f1a",
    "surface": "#121826",
    "surface_alt": "#0f1521",
    "border": "rgba(148, 163, 184, 0.14)",
    "text": "#e5e7eb",
    "muted": "#94a3b8",
    "primary": "#38bdf8",     # cyan
    "primary_dark": "#0ea5e9",
    "accent": "#a78bfa",      # violet
    "gold": "#fbbf24",
    "success": "#34d399",
    "warning": "#fbbf24",
    "danger": "#f87171",
    "gradient": ["#38bdf8", "#a78bfa"],
}

RATING_LABELS = {
    1: "Very Dissatisfied",
    2: "Dissatisfied",
    3: "Neutral",
    4: "Satisfied",
    5: "Highly Satisfied",
}

RATING_COLOR = {
    1: PALETTE["danger"],
    2: PALETTE["danger"],
    3: PALETTE["warning"],
    4: PALETTE["success"],
    5: PALETTE["success"],
}

RATING_RISK = {
    1: "High Risk",
    2: "High Risk",
    3: "Moderate Risk",
    4: "Low Risk",
    5: "Low Risk",
}


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    """Load the trained model, encoders, scaler, and feature order."""
    model = joblib.load(os.path.join(MODEL_DIR, "employee_satisfaction_model.pkl"))
    department_encoder = joblib.load(os.path.join(MODEL_DIR, "department_encoder.pkl"))
    promotion_encoder = joblib.load(os.path.join(MODEL_DIR, "promotion_encoder.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return {
        "model": model,
        "department_encoder": department_encoder,
        "promotion_encoder": promotion_encoder,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }


@st.cache_data(show_spinner=False)
def load_metrics():
    """Load the pre-computed metrics bundle produced by train_pipeline.py."""
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_raw_dataset():
    return pd.read_csv(DATA_PATH)


def format_pkr(value: float) -> str:
    return f"Rs. {value:,.0f}"


def rating_label(rating: int) -> str:
    return RATING_LABELS.get(int(rating), "Unknown")


def rating_color(rating: int) -> str:
    return RATING_COLOR.get(int(rating), PALETTE["muted"])


def rating_risk(rating: int) -> str:
    return RATING_RISK.get(int(rating), "Unknown")


def department_display_name(code: str) -> str:
    special = {"hr": "HR", "it": "IT"}
    return special.get(code, code.title())
