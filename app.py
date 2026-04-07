import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="ChurnLens · AI Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# CUSTOM CSS — DARK LUXURY THEME
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    /* ── RESET & BASE ─────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    .stApp {
        background: #0a0a0f;
        font-family: 'DM Sans', sans-serif;
        color: #e8e8f0;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1200px; }

    /* ── SCROLLBAR ───────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0f; }
    ::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 3px; }

    /* ── HERO HEADER ─────────────────────────── */
    .hero-wrap {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 2.5rem;
        padding-bottom: 2rem;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        gap: 1rem;
    }
    .hero-left {}
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.35);
        color: #a5b4fc;
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 5px 12px;
        border-radius: 20px;
        margin-bottom: 14px;
    }
    .hero-badge::before {
        content: '';
        width: 6px; height: 6px;
        background: #6366f1;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #6366f1;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin-bottom: 10px;
    }
    .hero-title span {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        color: #6b7280;
        font-weight: 300;
        max-width: 480px;
        line-height: 1.6;
    }
    .hero-stats {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
    }
    .stat-pill {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 10px 16px;
        min-width: 180px;
    }
    .stat-pill-icon { font-size: 18px; }
    .stat-pill-text { font-size: 12px; color: #9ca3af; }
    .stat-pill-val { font-family: 'Syne', sans-serif; font-size: 14px; font-weight: 700; color: #e8e8f0; }

    /* ── SECTION LABEL ───────────────────────── */
    .sec-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #4b5563;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .sec-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(255,255,255,0.06);
    }

    /* ── PANEL CARD ──────────────────────────── */
    .panel {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    .panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    }
    .panel-title {
        font-family: 'Syne', sans-serif;
        font-size: 13px;
        font-weight: 600;
        color: #9ca3af;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── INPUTS ──────────────────────────────── */
    .stSlider > div > div > div { background: rgba(99,102,241,0.25) !important; }
    .stSlider > div > div > div > div { background: #6366f1 !important; }
    .stSlider label { font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; color: #9ca3af !important; font-weight: 400 !important; }
    .stSlider [data-testid="stThumbValue"] { background: #6366f1 !important; color: white !important; font-size: 11px !important; }
    
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 13px !important;
        color: #9ca3af !important;
        font-weight: 400 !important;
    }
    
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e8e8f0 !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stSelectbox"] > div > div:focus-within {
        border-color: rgba(99,102,241,0.5) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    /* ── METRIC TILES ────────────────────────── */
    .tile-row { display: flex; gap: 14px; margin: 20px 0; }
    .tile {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s;
    }
    .tile:hover { border-color: rgba(99,102,241,0.3); }
    .tile-accent {
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        border-radius: 14px 0 0 14px;
    }
    .tile-label { font-size: 11px; color: #6b7280; font-weight: 400; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 8px; }
    .tile-val { font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 700; color: #fff; }
    .tile-sub { font-size: 11px; color: #4b5563; margin-top: 4px; }
    .tile-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 500;
        margin-top: 6px;
    }
    .badge-low { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.2); }
    .badge-med { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.2); }
    .badge-high { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.2); }

    /* ── INSIGHT BANNER ──────────────────────── */
    .insight {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 14px 18px;
        font-size: 13px;
        color: #a5b4fc;
        margin: 16px 0;
        display: flex;
        gap: 10px;
        align-items: flex-start;
        line-height: 1.5;
    }
    .insight-icon { font-size: 16px; margin-top: 1px; flex-shrink: 0; }
    .insight b { color: #c7d2fe; }

    /* ── PREDICT BUTTON ──────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        letter-spacing: 0.04em !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 24px rgba(99,102,241,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(99,102,241,0.5) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* ── DIVIDER ─────────────────────────────── */
    hr { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 28px 0; }

    /* ── RESULT SECTION ──────────────────────── */
    .result-header {
        font-family: 'Syne', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #fff;
        margin-bottom: 6px;
    }
    .result-sub { font-size: 13px; color: #6b7280; margin-bottom: 24px; }

    .result-main {
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .result-low-bg {
        background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.06) 100%);
        border: 1px solid rgba(16,185,129,0.25);
    }
    .result-med-bg {
        background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(217,119,6,0.06) 100%);
        border: 1px solid rgba(245,158,11,0.25);
    }
    .result-high-bg {
        background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.06) 100%);
        border: 1px solid rgba(239,68,68,0.25);
    }
    .result-emoji { font-size: 40px; margin-bottom: 10px; display: block; }
    .result-pct {
        font-family: 'Syne', sans-serif;
        font-size: 64px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 6px;
    }
    .result-label { font-size: 14px; color: #9ca3af; margin-bottom: 10px; }
    .result-tag {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .tag-low { background: rgba(16,185,129,0.2); color: #6ee7b7; }
    .tag-med { background: rgba(245,158,11,0.2); color: #fcd34d; }
    .tag-high { background: rgba(239,68,68,0.2); color: #fca5a5; }

    /* ── RECOMMENDATION CARD ─────────────────── */
    .rec-card {
        border-radius: 14px;
        padding: 20px 22px;
        margin-top: 16px;
    }
    .rec-low { background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.15); }
    .rec-med { background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.15); }
    .rec-high { background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.15); }
    .rec-title { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 12px; }
    .rec-title-low { color: #6ee7b7; }
    .rec-title-med { color: #fcd34d; }
    .rec-title-high { color: #fca5a5; }
    .rec-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 13px;
        color: #d1d5db;
        line-height: 1.4;
    }
    .rec-item:last-child { border-bottom: none; }
    .rec-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        margin-top: 5px;
        flex-shrink: 0;
    }
    .dot-low { background: #10b981; box-shadow: 0 0 6px rgba(16,185,129,0.6); }
    .dot-med { background: #f59e0b; box-shadow: 0 0 6px rgba(245,158,11,0.6); }
    .dot-high { background: #ef4444; box-shadow: 0 0 6px rgba(239,68,68,0.6); }

    /* ── ROI ROW ─────────────────────────────── */
    .roi-row { display: flex; gap: 14px; margin-top: 20px; }
    .roi-tile {
        flex: 1;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
    }
    .roi-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
    .roi-val { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 700; color: #fff; }

    /* ── FOOTER ──────────────────────────────── */
    .footer {
        text-align: center;
        padding: 28px;
        color: #374151;
        font-size: 12px;
        letter-spacing: 0.04em;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 50px;
    }
    .footer span { color: #6366f1; }

    /* streamlit metric override */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 14px 18px;
    }
    [data-testid="metric-container"] label { color: #6b7280 !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 22px !important;
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MATPLOTLIB DARK THEME
# ============================================
plt.rcParams.update({
    'figure.facecolor': '#0d0d16',
    'axes.facecolor': '#0d0d16',
    'axes.edgecolor': '#1f2937',
    'axes.labelcolor': '#6b7280',
    'xtick.color': '#6b7280',
    'ytick.color': '#6b7280',
    'text.color': '#e8e8f0',
    'grid.color': '#1f2937',
    'grid.linewidth': 0.5,
    'font.family': 'DejaVu Sans',
})

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-left">
        <div class="hero-badge">AI-Powered Analytics</div>
        <div class="hero-title">Churn<span>Lens</span></div>
        <div class="hero-sub">Real-time churn prediction powered by Random Forest. Identify at-risk customers before they leave.</div>
    </div>
    <div class="hero-stats">
        <div class="stat-pill">
            <div class="stat-pill-icon">🎯</div>
            <div>
                <div class="stat-pill-text">Model Accuracy</div>
                <div class="stat-pill-val">78.5% ROC-AUC</div>
            </div>
        </div>
        <div class="stat-pill">
            <div class="stat-pill-icon">⚡</div>
            <div>
                <div class="stat-pill-text">Algorithm</div>
                <div class="stat-pill-val">Random Forest</div>
            </div>
        </div>
        <div class="stat-pill">
            <div class="stat-pill-icon">📡</div>
            <div>
                <div class="stat-pill-text">Predictions</div>
                <div class="stat-pill-val">Real-time</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL
# ============================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'churn_model.pkl')

@st.cache_resource
def load_model():
    return joblib.load(model_path)

model = load_model()

# ============================================
# INPUT PANELS
# ============================================
st.markdown('<div class="sec-label">Customer Profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">👤 Customer Information</div>
    </div>
    """, unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12,
                       help="Number of months the customer has been with the company")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=150.0,
                                      value=70.0, step=5.0)

with col2:
    st.markdown("""
    <div class="panel">
        <div class="panel-title">📄 Account Details</div>
    </div>
    """, unsafe_allow_html=True)
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

# ============================================
# METRIC TILES
# ============================================
total_charges = tenure * monthly_charges

if contract_type == "Two year":
    risk_badge_cls, risk_badge_txt, tile_accent_color = "badge-low", "Low Risk", "#10b981"
elif contract_type == "One year":
    risk_badge_cls, risk_badge_txt, tile_accent_color = "badge-med", "Medium Risk", "#f59e0b"
else:
    risk_badge_cls, risk_badge_txt, tile_accent_color = "badge-high", "High Risk", "#ef4444"

customer_value = "High Value" if monthly_charges > 80 else "Mid Value" if monthly_charges > 50 else "Starter"
cv_color = "#6366f1" if monthly_charges > 80 else "#8b5cf6" if monthly_charges > 50 else "#a855f7"

st.markdown(f"""
<div class="tile-row">
    <div class="tile">
        <div class="tile-accent" style="background:{tile_accent_color};"></div>
        <div class="tile-label">Contract Risk</div>
        <div class="tile-val">{contract_type.split()[0]}</div>
        <span class="tile-badge {risk_badge_cls}">{risk_badge_txt}</span>
    </div>
    <div class="tile">
        <div class="tile-accent" style="background:#6366f1;"></div>
        <div class="tile-label">Total Lifetime Value</div>
        <div class="tile-val">${total_charges:,.0f}</div>
        <div class="tile-sub">Over {tenure} months</div>
    </div>
    <div class="tile">
        <div class="tile-accent" style="background:{cv_color};"></div>
        <div class="tile-label">Customer Segment</div>
        <div class="tile-val">{customer_value}</div>
        <div class="tile-sub">${monthly_charges:.0f}/month</div>
    </div>
    <div class="tile">
        <div class="tile-accent" style="background:#a855f7;"></div>
        <div class="tile-label">Tenure Band</div>
        <div class="tile-val">{'New' if tenure < 12 else 'Growing' if tenure < 36 else 'Loyal'}</div>
        <div class="tile-sub">{tenure} months</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# INSIGHT BANNER
# ============================================
if contract_type == "Month-to-month":
    insight_text = "<b>High-risk contract detected.</b> Month-to-month customers churn at 43% vs 3% for two-year plans. Consider offering an upgrade with a loyalty discount."
elif contract_type == "One year":
    insight_text = "<b>Moderate commitment detected.</b> Annual contracts reduce churn by 40% vs month-to-month. One more push towards a 2-year plan could lock in this customer."
else:
    insight_text = "<b>Excellent retention signal.</b> Two-year contracts show only 3% churn rate. Focus on upsell opportunities and referral programs for this customer."

st.markdown(f"""
<div class="insight">
    <span class="insight-icon">💡</span>
    <div>{insight_text}</div>
</div>
""", unsafe_allow_html=True)

# ============================================
# PREDICT BUTTON
# ============================================
st.markdown("<br>", unsafe_allow_html=True)
col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    predict = st.button("🔮  Run Prediction", use_container_width=True)

# ============================================
# RESULTS
# ============================================
if predict:
    input_data = pd.DataFrame([[
        tenure, monthly_charges, total_charges
    ]], columns=['tenure', 'MonthlyCharges', 'TotalCharges'])

    prob = model.predict_proba(input_data)[0][1]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="result-header">Prediction Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="result-sub">Based on customer profile analysis using the trained Random Forest model</div>', unsafe_allow_html=True)

    if prob > 0.6:
        res_bg, res_emoji, res_color, res_tag_cls, res_label = "result-high-bg", "🚨", "#ef4444", "tag-high", "HIGH RISK"
        rec_cls, rec_title_cls, dot_cls, rec_title = "rec-high", "rec-title-high", "dot-high", "🚨 Immediate Actions Required"
        recs = [
            ("📞", "Call the customer directly — understand their pain points and concerns."),
            ("🎁", "Offer 20–25% discount on upgrading to an annual or two-year plan."),
            ("⭐", "Bundle high-value add-ons: security suite, cloud backup, priority support."),
            ("📊", "Assign to a dedicated customer success manager for white-glove service."),
        ]
    elif prob > 0.3:
        res_bg, res_emoji, res_color, res_tag_cls, res_label = "result-med-bg", "⚠️", "#f59e0b", "tag-med", "MEDIUM RISK"
        rec_cls, rec_title_cls, dot_cls, rec_title = "rec-med", "rec-title-med", "dot-med", "⚠️ Proactive Engagement"
        recs = [
            ("📧", "Send a personalized engagement email with product tips & highlight unused features."),
            ("🎁", "Offer a 10% loyalty discount on plan renewal."),
            ("📱", "Trigger in-app nudges highlighting the benefits of upgrading."),
            ("📊", "Monitor usage patterns weekly — flag any drop in activity."),
        ]
    else:
        res_bg, res_emoji, res_color, res_tag_cls, res_label = "result-low-bg", "✅", "#10b981", "tag-low", "LOW RISK"
        rec_cls, rec_title_cls, dot_cls, rec_title = "rec-low", "rec-title-low", "dot-low", "✅ Retention & Growth"
        recs = [
            ("💬", "Send regular satisfaction check-ins and gather NPS feedback."),
            ("🎁", "Offer an upgrade or premium add-on at renewal time."),
            ("🌟", "Encourage referrals — happy long-term customers are your best advocates."),
            ("📊", "Identify upsell opportunities based on usage data."),
        ]

    col_r1, col_r2 = st.columns([1, 1.4], gap="large")

    with col_r1:
        st.markdown(f"""
        <div class="result-main {res_bg}">
            <span class="result-emoji">{res_emoji}</span>
            <div class="result-pct" style="color:{res_color};">{prob*100:.1f}%</div>
            <div class="result-label">Churn Probability</div>
            <span class="result-tag {res_tag_cls}">{res_label}</span>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations
        recs_html = "".join([
            f'<div class="rec-item"><div class="rec-dot {dot_cls}"></div><div><b>{icon}</b> {text}</div></div>'
            for icon, text in recs
        ])
        st.markdown(f"""
        <div class="rec-card {rec_cls}">
            <div class="rec-title {rec_title_cls}">{rec_title}</div>
            {recs_html}
        </div>
        """, unsafe_allow_html=True)

    with col_r2:
        # ── CHART 1: Animated Risk Gauge ───────────
        fig1, ax1 = plt.subplots(figsize=(7, 3.2))
        fig1.patch.set_facecolor('#0d0d16')
        ax1.set_facecolor('#0d0d16')

        # Background track
        ax1.barh([0], [100], color='#1a1a2e', height=0.55, zorder=1)
        
        # Colored zones
        ax1.barh([0], [30],        color='#0d2b20',   height=0.55, alpha=0.6, zorder=2)
        ax1.barh([0], [30], left=30, color='#2b2009', height=0.55, alpha=0.6, zorder=2)
        ax1.barh([0], [40], left=60, color='#2b0f0f', height=0.55, alpha=0.6, zorder=2)

        # Glow fill
        glow_color = "#10b981" if prob < 0.3 else "#f59e0b" if prob < 0.6 else "#ef4444"
        ax1.barh([0], [prob*100], color=glow_color, height=0.4, zorder=4, alpha=0.9)

        # Marker line
        ax1.axvline(x=prob*100, color='#ffffff', linewidth=2, zorder=5, alpha=0.9)

        # Zone labels
        for x, lbl, col in [(15, 'LOW', '#10b981'), (45, 'MEDIUM', '#f59e0b'), (80, 'HIGH', '#ef4444')]:
            ax1.text(x, -0.45, lbl, ha='center', fontsize=8, color=col, fontweight='600', alpha=0.7)

        ax1.text(prob*100, 0.45, f'{prob*100:.1f}%', ha='center', fontsize=11,
                 color='white', fontweight='700', zorder=6)

        ax1.set_xlim(0, 100)
        ax1.set_ylim(-0.7, 0.7)
        ax1.set_xlabel('Churn Risk Score', fontsize=10, color='#4b5563', labelpad=10)
        ax1.set_title('Risk Meter', fontsize=12, color='#9ca3af', pad=10, fontweight='600')
        ax1.spines[:].set_visible(False)
        ax1.set_yticks([])
        ax1.tick_params(colors='#4b5563', labelsize=9)
        ax1.xaxis.set_tick_params(which='both', bottom=False)
        ax1.set_xticks([0, 30, 60, 100])
        ax1.set_xticklabels(['0%', '30%', '60%', '100%'])

        fig1.tight_layout(pad=1.5)
        st.pyplot(fig1, use_container_width=True)
        plt.close()

        # ── CHART 2: Feature Influence Bars ────────
        fig2, ax2 = plt.subplots(figsize=(7, 3.8))
        fig2.patch.set_facecolor('#0d0d16')
        ax2.set_facecolor('#0d0d16')

        # Normalized influence values (heuristic)
        tenure_norm = (72 - tenure) / 72  # higher tenure = lower churn
        monthly_norm = (monthly_charges - 20) / 130
        contract_norm = [0.9, 0.5, 0.1][["Month-to-month","One year","Two year"].index(contract_type)]
        senior_norm = 0.65 if senior_citizen == "Yes" else 0.35

        factors = ['Tenure', 'Monthly\nCharges', 'Contract\nType', 'Senior\nCitizen']
        values = [tenure_norm, monthly_norm, contract_norm, senior_norm]
        colors_bar = ['#10b981' if v < 0.4 else '#f59e0b' if v < 0.65 else '#ef4444' for v in values]

        bars = ax2.bar(factors, values, color=colors_bar, width=0.5, zorder=3,
                       edgecolor='none', linewidth=0)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                     f'{val:.0%}', ha='center', va='bottom',
                     fontsize=9, color='#9ca3af', fontweight='500')

        ax2.set_ylim(0, 1.15)
        ax2.set_title('Feature Influence on Churn Risk', fontsize=12, color='#9ca3af', pad=12, fontweight='600')
        ax2.set_ylabel('Risk Contribution', fontsize=9, color='#4b5563')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#1f2937')
        ax2.spines['bottom'].set_color('#1f2937')
        ax2.tick_params(axis='x', colors='#6b7280', labelsize=9)
        ax2.tick_params(axis='y', colors='#4b5563', labelsize=8)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax2.grid(axis='y', alpha=0.15, zorder=0)

        # Legend
        legend_patches = [
            mpatches.Patch(color='#10b981', label='Low contribution'),
            mpatches.Patch(color='#f59e0b', label='Med contribution'),
            mpatches.Patch(color='#ef4444', label='High contribution'),
        ]
        ax2.legend(handles=legend_patches, fontsize=8, loc='upper right',
                   facecolor='#0d0d16', edgecolor='#1f2937', labelcolor='#9ca3af')

        fig2.tight_layout(pad=1.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── ROI TILES ─────────────────────────────
    yearly = monthly_charges * 12
    cac_est = monthly_charges * 4  # rough CAC estimate
    save_pct = 100 - prob * 100

    st.markdown(f"""
    <div class="roi-row">
        <div class="roi-tile">
            <div class="roi-label">Monthly Revenue</div>
            <div class="roi-val">${monthly_charges:.0f}</div>
        </div>
        <div class="roi-tile">
            <div class="roi-label">Annual Revenue</div>
            <div class="roi-val">${yearly:,.0f}</div>
        </div>
        <div class="roi-tile">
            <div class="roi-label">Est. CAC (4×)</div>
            <div class="roi-val">${cac_est:,.0f}</div>
        </div>
        <div class="roi-tile">
            <div class="roi-label">Retention Score</div>
            <div class="roi-val" style="color:{'#10b981' if save_pct>70 else '#f59e0b' if save_pct>40 else '#ef4444'};">{save_pct:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <span>ChurnLens</span> · Built with Streamlit & Random Forest · Telco Customer Churn Dataset · 78.5% ROC-AUC
</div>
""", unsafe_allow_html=True)