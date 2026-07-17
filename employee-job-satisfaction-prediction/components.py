"""
components.py
Reusable HTML/CSS-driven UI building blocks rendered via st.markdown.
Keeping these centralized avoids duplicating markup across pages.
"""

import streamlit as st


def hero_section(title: str, subtitle: str, badge: str = None):
    badge_html = f'<div class="hero-badge">{badge}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="hero">
            {badge_html}
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, subtitle: str = ""):
    sub_html = f'<p class="section-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <h2 class="section-title">{title}</h2>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(icon: str, value: str, label: str, accent: str = "primary"):
    return f"""
        <div class="stat-card accent-{accent}">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
    """


def stat_row(cards: list):
    cols = st.columns(len(cards))
    for col, card_html in zip(cols, cards):
        with col:
            st.markdown(card_html, unsafe_allow_html=True)


def feature_card(icon: str, title: str, desc: str):
    return f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
    """


def glass_card_open(extra_class: str = ""):
    st.markdown(f'<div class="glass-card {extra_class}">', unsafe_allow_html=True)


def glass_card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def badge(text: str, kind: str = "primary"):
    st.markdown(f'<span class="badge badge-{kind}">{text}</span>', unsafe_allow_html=True)


def prediction_result_card(rating: int, label: str, color: str, risk: str,
                            confidence: float, recommendation: str):
    risk_kind = "danger" if "High" in risk else ("warning" if "Moderate" in risk else "success")
    st.markdown(
        f"""
        <div class="prediction-card" style="--accent:{color}">
            <div class="prediction-top">
                <div class="prediction-rating">{rating}<span>/5</span></div>
                <div class="prediction-meta">
                    <div class="prediction-label" style="color:{color}">{label}</div>
                    <span class="badge badge-{risk_kind}">{risk}</span>
                </div>
            </div>
            <div class="prediction-confidence">
                <div class="confidence-label">
                    <span>Confidence Score</span><span>{confidence}%</span>
                </div>
                <div class="confidence-track">
                    <div class="confidence-fill" style="width:{confidence}%; background:{color}"></div>
                </div>
            </div>
            <div class="prediction-recommendation">
                <strong>Recommendation:</strong> {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="app-footer">
            <span>Employee Job Satisfaction Prediction &middot; Built by Rizwan Ahmed</span>
            <span class="footer-links">
                <a href="https://github.com/rizwanahmed786508" target="_blank">GitHub</a>
                &middot;
                <a href="https://www.linkedin.com/in/rizwanahmed78" target="_blank">LinkedIn</a>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
