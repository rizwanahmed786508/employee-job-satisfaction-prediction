"""
predict.py
Encapsulates the inference logic: takes raw human-readable employee inputs,
runs them through the same encoding + scaling pipeline used at training
time, and returns a structured prediction result.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import rating_color, rating_label, rating_risk


@dataclass
class PredictionResult:
    rating: int
    label: str
    color: str
    risk: str
    confidence: float
    probabilities: Dict[int, float]
    recommendation: str
    reasoning: List[str] = field(default_factory=list)


RECOMMENDATIONS = {
    1: "Immediate HR intervention is advised. Schedule a 1:1 check-in, review "
       "compensation against market rate, and investigate manager-relationship health.",
    2: "Proactive follow-up recommended. Explore workload, growth opportunities, "
       "and manager support before dissatisfaction deepens.",
    3: "Monitor periodically. Satisfaction is neutral — small improvements in "
       "work-life balance or recognition could shift this employee into the satisfied range.",
    4: "Employee is in a healthy zone. Maintain current support levels and continue "
       "regular recognition to sustain satisfaction.",
    5: "Employee is thriving. Consider them for mentorship roles or retention-focused "
       "growth opportunities to preserve this engagement.",
}


def _build_reasoning(inputs: dict, feature_importance: List[dict]) -> List[str]:
    """Produce a short, human-readable explanation using global feature
    importance combined with where this employee's values sit."""
    reasoning = []
    top_features = [f["feature"] for f in feature_importance[:3]]

    if "Salary_PKR" in top_features:
        if inputs["Salary_PKR"] < 60000:
            reasoning.append("Salary is on the lower end, which historically correlates with lower satisfaction.")
        elif inputs["Salary_PKR"] > 120000:
            reasoning.append("Above-average salary is a strong positive contributor to satisfaction.")
        else:
            reasoning.append("Salary sits in a mid-range band with moderate influence on the outcome.")

    if "Work_Life_Balance" in top_features:
        if inputs["Work_Life_Balance"] <= 2:
            reasoning.append("Low work-life balance rating is pulling the prediction toward dissatisfaction.")
        elif inputs["Work_Life_Balance"] >= 4:
            reasoning.append("Strong work-life balance is a key driver pushing this prediction upward.")

    if "Manager_Support" in top_features:
        if inputs["Manager_Support"] <= 2:
            reasoning.append("Weak manager support is one of the strongest negative signals here.")
        elif inputs["Manager_Support"] >= 4:
            reasoning.append("High manager support is reinforcing a positive satisfaction outlook.")

    if "Promotion_Last_2Y" in top_features and inputs["Promotion_Last_2Y"] == "No":
        reasoning.append("No promotion in the last 2 years is a mild negative factor.")

    if not reasoning:
        reasoning.append("Prediction driven by a balanced combination of all input features.")

    return reasoning


def run_prediction(bundle: dict, metrics: dict, inputs: dict) -> PredictionResult:
    """
    inputs: {
        "Department": "hr" (raw lowercase code from department_encoder.classes_),
        "Age": int,
        "Experience_Years": int,
        "Salary_PKR": float,
        "Work_Life_Balance": int (1-5),
        "Manager_Support": int (1-5),
        "Promotion_Last_2Y": "Yes" | "No",
    }
    """
    model = bundle["model"]
    department_encoder = bundle["department_encoder"]
    promotion_encoder = bundle["promotion_encoder"]
    scaler = bundle["scaler"]
    feature_columns = bundle["feature_columns"]

    dept_encoded = department_encoder.transform([inputs["Department"]])[0]
    promo_encoded = promotion_encoder.transform([inputs["Promotion_Last_2Y"]])[0]

    row = {
        "Department": dept_encoded,
        "Age": inputs["Age"],
        "Experience_Years": inputs["Experience_Years"],
        "Salary_PKR": inputs["Salary_PKR"],
        "Work_Life_Balance": inputs["Work_Life_Balance"],
        "Manager_Support": inputs["Manager_Support"],
        "Promotion_Last_2Y": promo_encoded,
    }
    feature_df = pd.DataFrame([row])[feature_columns]
    scaled = scaler.transform(feature_df)

    prediction = int(model.predict(scaled)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(scaled)[0]
        classes = model.classes_
        probabilities = {int(c): float(p) for c, p in zip(classes, proba)}
        confidence = float(max(proba)) * 100
    else:
        probabilities = {prediction: 1.0}
        confidence = 100.0

    reasoning = _build_reasoning(inputs, metrics.get("feature_importance", []))

    return PredictionResult(
        rating=prediction,
        label=rating_label(prediction),
        color=rating_color(prediction),
        risk=rating_risk(prediction),
        confidence=round(confidence, 1),
        probabilities=probabilities,
        recommendation=RECOMMENDATIONS.get(prediction, ""),
        reasoning=reasoning,
    )
