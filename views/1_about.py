import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theme import apply_theme, render_sidebar
apply_theme()
render_sidebar()

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 48px 0 40px;">
    <div style="display:inline-flex; align-items:center; gap:8px; background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3); border-radius:20px; padding:5px 14px; margin-bottom:20px;">
        <div style="width:6px;height:6px;background:#6366f1;border-radius:50%;box-shadow:0 0 8px #6366f1;"></div>
        <span style="font-size:11px;color:#a5b4fc;letter-spacing:0.12em;text-transform:uppercase;font-weight:500;">AI Retention Platform</span>
    </div>
    <div style="font-family:'Syne',sans-serif; font-size:44px; font-weight:800; color:#fff; line-height:1.1; margin-bottom:16px;">
        Predict Churn.<br>
        <span style="background:linear-gradient(135deg,#6366f1,#a855f7,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Retain Customers.</span>
    </div>
    <div style="font-size:16px; color:#6b7280; max-width:580px; line-height:1.7; font-weight:300;">
        ChurnLens is an AI-powered customer retention intelligence platform that helps businesses identify at-risk customers, understand churn drivers, and take data-driven action before it's too late.
    </div>
</div>
""", unsafe_allow_html=True)

# ── STAT ROW ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; gap:16px; margin-bottom:40px; flex-wrap:wrap;">
    <div style="flex:1; min-width:160px; background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:14px; padding:20px 22px;">
        <div style="font-family:'Syne',sans-serif; font-size:32px; font-weight:800; color:#a5b4fc;">83%</div>
        <div style="font-size:12px; color:#6b7280; margin-top:4px; text-transform:uppercase; letter-spacing:0.06em;">ROC-AUC Score</div>
    </div>
    <div style="flex:1; min-width:160px; background:rgba(168,85,247,0.08); border:1px solid rgba(168,85,247,0.2); border-radius:14px; padding:20px 22px;">
        <div style="font-family:'Syne',sans-serif; font-size:32px; font-weight:800; color:#c084fc;">11</div>
        <div style="font-size:12px; color:#6b7280; margin-top:4px; text-transform:uppercase; letter-spacing:0.06em;">Features Used</div>
    </div>
    <div style="flex:1; min-width:160px; background:rgba(236,72,153,0.08); border:1px solid rgba(236,72,153,0.2); border-radius:14px; padding:20px 22px;">
        <div style="font-family:'Syne',sans-serif; font-size:32px; font-weight:800; color:#f9a8d4;">7K+</div>
        <div style="font-size:12px; color:#6b7280; margin-top:4px; text-transform:uppercase; letter-spacing:0.06em;">Training Samples</div>
    </div>
    <div style="flex:1; min-width:160px; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:14px; padding:20px 22px;">
        <div style="font-family:'Syne',sans-serif; font-size:32px; font-weight:800; color:#6ee7b7;">200</div>
        <div style="font-size:12px; color:#6b7280; margin-top:4px; text-transform:uppercase; letter-spacing:0.06em;">Decision Trees</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── PLATFORM FEATURES ─────────────────────────────────────────────────────────
st.markdown('<div style="font-family:\'Syne\',sans-serif; font-size:20px; font-weight:700; color:#fff; margin-bottom:20px;">Platform Features</div>', unsafe_allow_html=True)

features = [
    ("🔮", "Single Prediction", "Input any customer profile and get an instant churn probability score with risk classification and personalized retention recommendations."),
    ("📂", "Bulk Prediction", "Upload a CSV of your entire customer base. Get churn probabilities, risk categories, and recommendations for every customer in one click."),
    ("📈", "Analytics Dashboard", "Explore churn patterns across contract types, tenure bands, revenue segments, and payment methods with interactive visualizations."),
    ("🧠", "Model Insights", "Deep-dive into model performance — confusion matrix, ROC curve, feature importance, precision/recall, and SHAP explanations."),
]

cols = st.columns(2)
for i, (icon, title, desc) in enumerate(features):
    with cols[i % 2]:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:22px; margin-bottom:16px; position:relative; overflow:hidden;">
            <div style="position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,rgba(99,102,241,0.4),transparent);"></div>
            <div style="font-size:28px; margin-bottom:10px;">{icon}</div>
            <div style="font-family:'Syne',sans-serif; font-size:15px; font-weight:700; color:#fff; margin-bottom:8px;">{title}</div>
            <div style="font-size:13px; color:#6b7280; line-height:1.6; font-weight:300;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ── TECH STACK ────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div style="font-family:\'Syne\',sans-serif; font-size:20px; font-weight:700; color:#fff; margin-bottom:20px;">Tech Stack</div>', unsafe_allow_html=True)

stack = [
    ("Frontend", "Streamlit · HTML/CSS · Plotly · Matplotlib"),
    ("ML Model", "Scikit-learn · Random Forest · SHAP"),
    ("Data", "Pandas · NumPy · Telco Churn Dataset"),
    ("Deployment", "Streamlit Cloud · GitHub"),
]

cols2 = st.columns(4)
for i, (layer, tech) in enumerate(stack):
    with cols2[i]:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px; text-align:center;">
            <div style="font-size:11px; color:#6366f1; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">{layer}</div>
            <div style="font-size:12px; color:#9ca3af; line-height:1.6;">{tech}</div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:40px 0 20px; font-size:12px; color:#374151; letter-spacing:0.04em;">
    Built with ❤️ using <span style="color:#6366f1;">Streamlit</span> · Random Forest · Telco Customer Churn Dataset
</div>
""", unsafe_allow_html=True)
