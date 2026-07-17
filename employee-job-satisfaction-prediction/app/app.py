"""
app.py
Employee Job Satisfaction Prediction — premium Streamlit dashboard.

Run with:  streamlit run app.py
(Run `python train_pipeline.py` once first to generate model artifacts.)
"""

import os
import random
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import components as ui
from predict import run_prediction
from report import build_prediction_report
from utils import (
    PALETTE,
    department_display_name,
    load_metrics,
    load_model_bundle,
    load_raw_dataset,
    rating_label,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Employee Satisfaction Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme handling (Dark / Light toggle)
# ---------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "history" not in st.session_state:
    st.session_state.history = []
if "prefill" not in st.session_state:
    st.session_state.prefill = None

LIGHT_OVERRIDES = """
<style>
:root {
    --bg: #f4f6fb !important;
    --surface: #ffffff !important;
    --surface-alt: #eef1f8 !important;
    --border: rgba(15, 23, 42, 0.10) !important;
    --border-strong: rgba(15, 23, 42, 0.18) !important;
    --text: #0f172a !important;
    --muted: #64748b !important;
}
.stApp { background: radial-gradient(circle at 15% 0%, rgba(56,189,248,0.10), transparent 40%),
    radial-gradient(circle at 85% 15%, rgba(167,139,250,0.10), transparent 40%), var(--bg) !important; }
.hero-title { -webkit-text-fill-color: initial !important; background: none !important; color: #0f172a !important; }
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
    background: #ffffff !important; color: #0f172a !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #ffffff 0%, #f1f4fa 100%) !important; }
</style>
"""


def load_css():
    css_path = os.path.join(BASE_DIR, "styles.css")
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    if st.session_state.theme == "light":
        st.markdown(LIGHT_OVERRIDES, unsafe_allow_html=True)


load_css()

# ---------------------------------------------------------------------------
# Load cached artifacts
# ---------------------------------------------------------------------------
try:
    bundle = load_model_bundle()
    metrics = load_metrics()
    artifacts_ready = True
except FileNotFoundError:
    artifacts_ready = False

DEPARTMENTS = metrics["department_classes"] if artifacts_ready else []

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-title">📊 Satisfaction AI</div>
            <div class="brand-sub">Employee Analytics Suite</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Prediction", "📈 Analytics", "🧠 Model Info", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    theme_choice = st.toggle("🌙 Dark Mode", value=(st.session_state.theme == "dark"))
    st.session_state.theme = "dark" if theme_choice else "light"

    if artifacts_ready:
        st.markdown("---")
        st.caption("Model Status")
        st.success(f"{metrics['best_model_name']} · {metrics['best_model_accuracy']}% accuracy")

if not artifacts_ready:
    st.error(
        "Model artifacts not found. Please run `python train_pipeline.py` once "
        "from the project root to train and save the model before launching the app."
    )
    st.stop()

# reload CSS after potential theme flip triggered this run
if st.session_state.theme == "light":
    st.markdown(LIGHT_OVERRIDES, unsafe_allow_html=True)

df_raw = load_raw_dataset()

# ===========================================================================
# HOME
# ===========================================================================
if page == "🏠 Home":
    ui.hero_section(
        "Employee Job Satisfaction<br>Prediction Dashboard",
        "An end-to-end machine learning system that predicts employee satisfaction "
        "(1–5) from workplace and demographic factors — helping HR teams move from "
        "reactive damage control to proactive, data-driven retention decisions.",
        badge="AI-Powered HR Analytics",
    )

    ui.stat_row([
        ui.stat_card("🗂️", f"{metrics['dataset_cleaned_rows']}", "Clean Training Records", "primary"),
        ui.stat_card("🤖", f"{len(metrics['algorithms_tested'])}", "Models Compared", "accent"),
        ui.stat_card("🏆", f"{metrics['best_model_accuracy']}%", f"{metrics['best_model_name']} Accuracy", "gold"),
        ui.stat_card("🎯", "7", "Input Features", "success"),
    ])

    ui.section_title("Why Employee Satisfaction Matters", "The business case for predictive HR analytics")
    c1, c2 = st.columns([3, 2])
    with c1:
        ui.glass_card_open()
        st.markdown(
            """
Dissatisfied employees are more likely to disengage, underperform, or leave —
while satisfied employees tend to be more productive and loyal. This dashboard
turns that insight into a predictive tool: enter an employee's profile and get
an instant, explainable satisfaction forecast.

Built on a real (intentionally messy) HR dataset, the underlying pipeline
demonstrates a complete ML workflow — from data cleaning through model
comparison to deployment — mirroring how such a system could operate inside
an actual HR analytics tool.
            """
        )
        ui.glass_card_close()
    with c2:
        dist = metrics["satisfaction_distribution"]
        fig = px.pie(
            names=[rating_label(int(k)) for k in dist.keys()],
            values=list(dist.values()),
            hole=0.55,
            color_discrete_sequence=[PALETTE["danger"], PALETTE["danger"], PALETTE["warning"], PALETTE["success"], PALETTE["success"]],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color=PALETTE["text"], showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    ui.section_title("Key Features", "What this application offers")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown(ui.feature_card("🔮", "Instant Prediction", "Get a satisfaction rating with confidence score in real time."), unsafe_allow_html=True)
    with f2:
        st.markdown(ui.feature_card("📊", "Visual Analytics", "Explore dataset patterns, correlations, and model performance."), unsafe_allow_html=True)
    with f3:
        st.markdown(ui.feature_card("🧠", "Explainable AI", "Understand which factors are driving each prediction."), unsafe_allow_html=True)
    with f4:
        st.markdown(ui.feature_card("📄", "PDF Reports", "Download a polished report for any prediction you generate."), unsafe_allow_html=True)

    ui.footer()

# ===========================================================================
# PREDICTION
# ===========================================================================
elif page == "🔮 Prediction":
    ui.hero_section(
        "Predict Employee Satisfaction",
        "Fill in the employee profile below to generate an instant, explainable prediction.",
        badge="Live Inference",
    )

    top_l, top_r = st.columns([1, 1])
    with top_l:
        if st.button("🎲 Random Employee", use_container_width=True):
            st.session_state.prefill = {
                "Department": random.choice(DEPARTMENTS),
                "Age": random.randint(20, 58),
                "Experience_Years": random.randint(0, 30),
                "Salary_PKR": float(random.randint(25000, 180000)),
                "Work_Life_Balance": random.randint(1, 5),
                "Manager_Support": random.randint(1, 5),
                "Promotion_Last_2Y": random.choice(["Yes", "No"]),
            }
    with top_r:
        if st.button("♻️ Reset Form", use_container_width=True):
            st.session_state.prefill = None
            st.rerun()

    pf = st.session_state.prefill or {}

    with st.form("prediction_form"):
        ui.section_title("Employee Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            department = st.selectbox(
                "Department",
                options=DEPARTMENTS,
                format_func=department_display_name,
                index=DEPARTMENTS.index(pf["Department"]) if pf.get("Department") in DEPARTMENTS else 0,
            )
            age = st.number_input("Age", min_value=18, max_value=70, value=int(pf.get("Age", 30)))
        with c2:
            experience = st.number_input(
                "Experience (Years)", min_value=0, max_value=45, value=int(pf.get("Experience_Years", 5))
            )
            salary = st.number_input(
                "Salary (PKR)", min_value=10000, max_value=500000, step=1000,
                value=int(pf.get("Salary_PKR", 65000)),
            )
        with c3:
            promotion = st.radio(
                "Promotion in Last 2 Years?", options=["Yes", "No"],
                index=["Yes", "No"].index(pf.get("Promotion_Last_2Y", "No")), horizontal=True,
            )

        st.markdown("&nbsp;", unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        with c4:
            work_life = st.slider("Work-Life Balance", 1, 5, int(pf.get("Work_Life_Balance", 3)))
        with c5:
            manager_support = st.slider("Manager Support", 1, 5, int(pf.get("Manager_Support", 3)))

        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            submitted = st.form_submit_button("🔮 Predict Satisfaction", use_container_width=True)

    if submitted:
        inputs = {
            "Department": department,
            "Age": int(age),
            "Experience_Years": int(experience),
            "Salary_PKR": float(salary),
            "Work_Life_Balance": int(work_life),
            "Manager_Support": int(manager_support),
            "Promotion_Last_2Y": promotion,
        }
        with st.spinner("Running inference through the trained pipeline..."):
            result = run_prediction(bundle, metrics, inputs)

        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "department": department_display_name(department),
            "rating": result.rating,
            "label": result.label,
            "confidence": result.confidence,
        })
        st.session_state.history = st.session_state.history[:10]

        ui.section_title("Prediction Result")
        ui.prediction_result_card(
            result.rating, result.label, result.color, result.risk,
            result.confidence, result.recommendation,
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Probability Distribution**")
            probs = result.probabilities
            fig = go.Figure(go.Bar(
                x=[f"{k}" for k in sorted(probs.keys())],
                y=[probs[k] * 100 for k in sorted(probs.keys())],
                marker_color=PALETTE["primary"],
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color=PALETTE["text"], height=260, margin=dict(t=10, b=10, l=10, r=10),
                yaxis_title="Probability (%)", xaxis_title="Rating",
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Prediction Reasoning**")
            ui.glass_card_open()
            for point in result.reasoning:
                st.markdown(f"- {point}")
            ui.glass_card_close()

        pdf_bytes = build_prediction_report(inputs, result, department_display_name(department))
        st.download_button(
            "📄 Download Prediction Report (PDF)",
            data=pdf_bytes,
            file_name=f"satisfaction_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    if st.session_state.history:
        with st.expander("🕒 Prediction History (this session)"):
            st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)

    with st.expander("📁 Batch Prediction (CSV Upload)"):
        st.caption(
            "Upload a CSV with columns: Department, Age, Experience_Years, Salary_PKR, "
            "Work_Life_Balance, Manager_Support, Promotion_Last_2Y"
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                required_cols = {"Department", "Age", "Experience_Years", "Salary_PKR",
                                  "Work_Life_Balance", "Manager_Support", "Promotion_Last_2Y"}
                missing_cols = required_cols - set(batch_df.columns)
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                else:
                    results_rows = []
                    with st.spinner("Scoring batch..."):
                        for _, row in batch_df.iterrows():
                            row_inputs = {
                                "Department": str(row["Department"]).strip().lower(),
                                "Age": int(row["Age"]),
                                "Experience_Years": int(row["Experience_Years"]),
                                "Salary_PKR": float(row["Salary_PKR"]),
                                "Work_Life_Balance": int(row["Work_Life_Balance"]),
                                "Manager_Support": int(row["Manager_Support"]),
                                "Promotion_Last_2Y": str(row["Promotion_Last_2Y"]).strip().title(),
                            }
                            try:
                                r = run_prediction(bundle, metrics, row_inputs)
                                results_rows.append({**row.to_dict(), "Predicted_Rating": r.rating,
                                                      "Status": r.label, "Confidence_%": r.confidence})
                            except Exception:
                                results_rows.append({**row.to_dict(), "Predicted_Rating": "Error",
                                                      "Status": "Invalid row", "Confidence_%": None})
                    result_df = pd.DataFrame(results_rows)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Could not process file: {e}")

    ui.footer()

# ===========================================================================
# ANALYTICS
# ===========================================================================
elif page == "📈 Analytics":
    ui.hero_section("Visual Analytics", "Dataset insights and model performance, all in one place.", badge="Data & Model Insights")

    tabs = st.tabs(["Dataset Overview", "Model Performance", "Feature Insights"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Satisfaction Rating Distribution**")
            dist = metrics["satisfaction_distribution"]
            fig = px.bar(
                x=[rating_label(int(k)) for k in dist.keys()], y=list(dist.values()),
                color=list(dist.values()), color_continuous_scale=["#f87171", "#fbbf24", "#38bdf8"],
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=PALETTE["text"], showlegend=False, coloraxis_showscale=False,
                               height=340, xaxis_title="", yaxis_title="Employees")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Department Distribution**")
            dept_dist = metrics["department_distribution"]
            fig = px.bar(
                x=list(dept_dist.values()), y=[department_display_name(k) for k in dept_dist.keys()],
                orientation="h", color=list(dept_dist.values()), color_continuous_scale=["#a78bfa", "#38bdf8"],
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=PALETTE["text"], showlegend=False, coloraxis_showscale=False,
                               height=340, xaxis_title="Employees", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Correlation Heatmap**")
        corr = pd.DataFrame(metrics["correlation_matrix"])
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale=["#0b0f1a", "#0ea5e9", "#a78bfa"],
            aspect="auto",
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color=PALETTE["text"], height=440)
        st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Salary Distribution**")
            fig = px.histogram(df_raw, x="Salary_PKR", nbins=30, color_discrete_sequence=[PALETTE["primary"]])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=PALETTE["text"], height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            st.markdown("**Age Distribution**")
            fig = px.histogram(df_raw, x="Age", nbins=20, color_discrete_sequence=[PALETTE["accent"]])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=PALETTE["text"], height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Dataset Snapshot**")
        st.dataframe(df_raw.head(15), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.markdown("**Model Comparison**")
        comp_df = pd.DataFrame(metrics["model_comparison"])
        fig = px.bar(
            comp_df, x="accuracy", y="model", orientation="h", text="accuracy",
            color="accuracy", color_continuous_scale=["#374151", "#38bdf8", "#a78bfa"],
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color=PALETTE["text"], showlegend=False, coloraxis_showscale=False,
                           height=340, xaxis_title="Accuracy (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Confusion Matrix — {metrics['best_model_name']}**")
            cm = metrics["confusion_matrix"]
            labels = metrics["confusion_matrix_labels"]
            fig = px.imshow(
                cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
                text_auto=True, color_continuous_scale=["#0b0f1a", "#0ea5e9", "#38bdf8"],
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=PALETTE["text"], height=360,
                               xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Classification Report**")
            report = metrics["classification_report"]
            rows = []
            for k, v in report.items():
                if isinstance(v, dict):
                    rows.append({"Class": k, "Precision": round(v.get("precision", 0), 2),
                                  "Recall": round(v.get("recall", 0), 2),
                                  "F1-Score": round(v.get("f1-score", 0), 2),
                                  "Support": int(v.get("support", 0))})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=360)

        m1, m2, m3, m4 = st.columns(4)
        best = comp_df.iloc[0]
        m1.metric("Accuracy", f"{best['accuracy']}%")
        m2.metric("Precision", f"{best['precision']}%")
        m3.metric("Recall", f"{best['recall']}%")
        m4.metric("F1 Score", f"{best['f1_score']}%")

    with tabs[2]:
        st.markdown(f"**Feature Importance — {metrics['best_model_name']}**")
        fi_df = pd.DataFrame(metrics["feature_importance"])
        fig = px.bar(
            fi_df, x="importance", y="feature", orientation="h",
            color="importance", color_continuous_scale=["#374151", "#38bdf8", "#a78bfa"],
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color=PALETTE["text"], showlegend=False, coloraxis_showscale=False,
                           height=380, xaxis_title="Importance Score", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "Salary, Work-Life Balance, and Manager Support are the strongest predictors "
            "of employee satisfaction in this dataset — together accounting for the majority "
            "of the model's decision-making weight."
        )

    ui.footer()

# ===========================================================================
# MODEL INFO
# ===========================================================================
elif page == "🧠 Model Info":
    ui.hero_section("Model Information", "Technical details behind the deployed prediction model.", badge="Under the Hood")

    c1, c2 = st.columns(2)
    with c1:
        ui.glass_card_open()
        st.markdown("#### Model Summary")
        st.markdown(f"""
- **Best Algorithm:** {metrics['best_model_name']}
- **Test Accuracy:** {metrics['best_model_accuracy']}%
- **Target Variable:** `{metrics['target_variable']}` (scale of 1–5)
- **Total Features:** {metrics['n_features']}
- **Training Samples:** {metrics['train_size']}
- **Testing Samples:** {metrics['test_size']}
        """)
        ui.glass_card_close()
    with c2:
        ui.glass_card_open()
        st.markdown("#### Dataset Information")
        st.markdown(f"""
- **Original Records:** {metrics['dataset_original_rows']}
- **Cleaned Records:** {metrics['dataset_cleaned_rows']}
- **Duplicates Removed:** {metrics['rows_removed_duplicates']}
- **Missing Values Handled:** Department (mode), Salary (median), Manager Support (median)
- **Feature Scaling:** StandardScaler
- **Train/Test Split:** 80% / 20% (random_state=42)
        """)
        ui.glass_card_close()

    ui.section_title("Algorithms Tested")
    algo_df = pd.DataFrame(metrics["model_comparison"])
    st.dataframe(algo_df, use_container_width=True, hide_index=True)

    ui.section_title("Machine Learning Workflow")
    steps = ["📂 Raw Dataset", "🧹 Data Cleaning", "📊 EDA", "⚙️ Feature Engineering",
             "🤖 Model Training", "📈 Model Evaluation", "🔮 Deployment"]
    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="stat-card accent-primary" style="padding:0.9rem 0.5rem;">'
                f'<div style="font-size:0.82rem;font-weight:600;">{step}</div></div>',
                unsafe_allow_html=True,
            )

    ui.section_title("Feature Columns Used")
    st.code(", ".join(metrics["feature_columns"]), language="text")

    ui.footer()

# ===========================================================================
# ABOUT
# ===========================================================================
elif page == "ℹ️ About":
    ui.hero_section("About This Project", "Problem statement, objectives, and technical background.", badge="Project Documentation")

    ui.glass_card_open()
    st.markdown("""
#### Problem Statement
Organizations often struggle to identify dissatisfied employees before they
disengage or resign. This project builds a Machine Learning pipeline that
predicts an employee's Job Satisfaction Rating (1–5) from workplace and
demographic factors, giving HR teams a proactive tool for retention and
workforce planning.

#### Objectives
- Prepare a realistic, intentionally messy employee survey dataset
- Perform thorough data cleaning and exploratory data analysis
- Engineer features suitable for machine learning
- Train and compare multiple classification algorithms
- Deploy the best-performing model behind an interactive dashboard

#### Applications
- Employee retention risk flagging
- HR analytics and workforce planning
- Compensation and promotion strategy support
- Manager-support and work-life balance interventions

#### Future Improvements
- Incorporate real organizational survey data at scale
- Hyperparameter tuning and cross-validation
- SHAP-based explainability for deeper local interpretability
- Power BI / BI-tool integration for enterprise HR dashboards
- Deep learning approaches for comparison
    """)
    ui.glass_card_close()

    ui.section_title("Technologies Used")
    techs = ["Python", "Streamlit", "Pandas", "NumPy", "Scikit-Learn",
             "Plotly", "Matplotlib", "Seaborn", "Joblib", "FPDF2"]
    st.markdown(
        " ".join([f'<span class="badge badge-primary" style="margin:3px;">{t}</span>' for t in techs]),
        unsafe_allow_html=True,
    )

    ui.section_title("Author")
    ui.glass_card_open()
    st.markdown("""
**Rizwan Ahmed**
Software Engineering Student · Machine Learning Engineer · Data Science Enthusiast

[GitHub](https://github.com/rizwanahmed786508) · [LinkedIn](https://www.linkedin.com/in/rizwanahmed78)
    """)
    ui.glass_card_close()

    ui.footer()
