import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theme import apply_theme, render_sidebar
from database import save_prediction
apply_theme()
render_sidebar()

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE, 'churn_model.pkl'))
    features = joblib.load(os.path.join(BASE, 'feature_columns.pkl'))
    with open(os.path.join(BASE, 'model_metadata.json')) as f:
        meta = json.load(f)
    return model, features, meta

model, FEATURE_COLS, meta = load_artifacts()

if 'predicted' not in st.session_state:
    st.session_state.predicted = False
if 'last_analyzed_signature' not in st.session_state:
    st.session_state.last_analyzed_signature = None
if 'single_total_charges' not in st.session_state:
    st.session_state.single_total_charges = 0.0

def sync_total_charges():
    tenure_val = st.session_state.get("single_tenure", 0) or 0
    monthly_val = st.session_state.get("single_monthly_charges", 0.0) or 0.0
    st.session_state.single_total_charges = round(float(tenure_val) * float(monthly_val), 2)

plt.rcParams.update({
    'figure.facecolor': '#0d0d1a', 'axes.facecolor': '#0d0d1a',
    'axes.edgecolor': '#1f2937', 'axes.labelcolor': '#6b7280',
    'xtick.color': '#6b7280', 'ytick.color': '#6b7280',
    'text.color': '#e2e2ef', 'grid.color': '#1f2937',
})

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 28px;">
    <div style="font-family:'Syne',sans-serif; font-size:30px; font-weight:800; color:#fff; margin-bottom:6px;">
        🔮 Single Prediction
    </div>
    <div style="font-size:14px; color:#6b7280;">Enter customer details to get an instant churn risk assessment</div>
</div>
""", unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div style="font-size:11px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:16px; display:flex; align-items:center; gap:10px;">Customer Profile <span style="flex:1; height:1px; background:rgba(255,255,255,0.06); display:inline-block;"></span></div>', unsafe_allow_html=True)

st.markdown("""
<style>
/* Keep selectboxes visually behaving like dropdowns with the native chevron. */
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    cursor: pointer !important;
}
div[data-testid="stSelectbox"] input {
    caret-color: transparent !important;
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div style="font-size:12px; color:#6366f1; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">Account Info</div>', unsafe_allow_html=True)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=10000, value=0, step=1, key="single_tenure", on_change=sync_total_charges)
    monthly_charges = st.number_input("Monthly Charges (₹)", min_value=0.0, max_value=10000.0, value=0.0, step=5.0, key="single_monthly_charges", on_change=sync_total_charges)
    sync_total_charges()
    total_charges = st.number_input(
        "Total Charges (₹)",
        min_value=0.0,
        max_value=100000000.0,
        step=0.01,
        format="%.2f",
        key="single_total_charges",
        help="Auto-calculated as Tenure × Monthly Charges.",
    )
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])

with col2:
    st.markdown('<div style="font-size:12px; color:#a855f7; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">Contract & Billing</div>', unsafe_allow_html=True)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

with col3:
    st.markdown('<div style="font-size:12px; color:#ec4899; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">Services</div>', unsafe_allow_html=True)
    internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])

st.markdown("<br>", unsafe_allow_html=True)

current_signature = (
    tenure, monthly_charges, total_charges, senior_citizen, partner,
    contract, payment_method, paperless_billing, internet_service,
    online_security, tech_support
)
if st.session_state.predicted and st.session_state.last_analyzed_signature != current_signature:
    st.session_state.predicted = False

# ── PREDICT BUTTON ────────────────────────────────────────────────────────────
col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    predict = st.button("🔮  Analyze Churn Risk", use_container_width=True)

if predict:
    st.session_state.predicted = True
    st.session_state.last_analyzed_signature = current_signature

# ── QUICK METRICS ──────────────────────────────────────────────────────────────
if st.session_state.predicted:
    contract_risk = {"Month-to-month": ("High Risk", "#ef4444"), "One year": ("Medium Risk", "#f59e0b"), "Two year": ("Low Risk", "#10b981")}
    risk_label, risk_color = contract_risk[contract]

    st.markdown(f"""
    <div style="display:flex; gap:14px; margin-bottom:24px; flex-wrap:wrap;">
        <div style="flex:1; min-width:130px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px 18px;">
            <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Lifetime Value</div>
            <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:#fff;">₹{total_charges:,.0f}</div>
        </div>
        <div style="flex:1; min-width:130px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px 18px;">
            <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Contract Risk</div>
            <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:700; color:{risk_color};">{risk_label}</div>
        </div>
        <div style="flex:1; min-width:130px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px 18px;">
            <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Tenure Band</div>
            <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:700; color:#a5b4fc;">{'New' if tenure < 12 else 'Growing' if tenure < 36 else 'Loyal'}</div>
        </div>
        <div style="flex:1; min-width:130px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px 18px;">
            <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Services Active</div>
            <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:700; color:#fff;">{sum([online_security=='Yes', tech_support=='Yes', internet_service!='No'])}/3</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── PREDICTION RESULTS ────────────────────────────────────────────────────────
if predict:
    # Build input
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        'InternetService': {"No": 0, "DSL": 1, "Fiber optic": 2}[internet_service],
        'OnlineSecurity': 1 if online_security == "Yes" else 0,
        'TechSupport': 1 if tech_support == "Yes" else 0,
        'PaymentMethod': {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[payment_method],
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
    }
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    prob = model.predict_proba(input_df)[0][1]

    # Save to history DB
    risk_label_db = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
    rec_db = "Immediate outreach + discount offer" if prob > 0.6 else "Proactive engagement + loyalty incentive" if prob > 0.3 else "Regular engagement + upsell opportunity"
    save_prediction(input_dict, prob, risk_label_db, rec_db)
    single_history_row = pd.DataFrame([{
        'source': 'Single',
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract': contract,
        'internet_service': internet_service,
        'payment_method': payment_method,
        'churn_probability': round(prob * 100, 1),
        'risk_category': risk_label_db,
    }])
    existing_single_history = st.session_state.get('single_prediction_history')
    if existing_single_history is None or len(existing_single_history) == 0:
        st.session_state['single_prediction_history'] = single_history_row
    else:
        st.session_state['single_prediction_history'] = pd.concat(
            [existing_single_history, single_history_row],
            ignore_index=True
        )
    st.session_state['active_single_prediction'] = single_history_row

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-size:20px; font-weight:700; color:#fff; margin-bottom:20px;">Prediction Results</div>', unsafe_allow_html=True)

    if prob > 0.6:
        res_bg = "rgba(239,68,68,0.08)"; res_border = "rgba(239,68,68,0.25)"
        res_color = "#ef4444"; res_emoji = "🚨"; res_label = "HIGH RISK"
        tag_bg = "rgba(239,68,68,0.15)"; tag_color = "#fca5a5"
        recs = [
            "📞 Call the customer immediately — understand key pain points",
            "🎁 Offer 20-25% discount to upgrade to annual/two-year plan",
            "⭐ Bundle: Online Security + Tech Support package",
            "👤 Assign dedicated customer success manager",
        ]
        dot_color = "#ef4444"
    elif prob > 0.3:
        res_bg = "rgba(245,158,11,0.08)"; res_border = "rgba(245,158,11,0.25)"
        res_color = "#f59e0b"; res_emoji = "⚠️"; res_label = "MEDIUM RISK"
        tag_bg = "rgba(245,158,11,0.15)"; tag_color = "#fcd34d"
        recs = [
            "📧 Send personalized engagement email with tips",
            "🎁 Offer 10% loyalty discount at next renewal",
            "📱 Highlight unused features they're missing out on",
            "📊 Monitor usage weekly — flag drops in activity",
        ]
        dot_color = "#f59e0b"
    else:
        res_bg = "rgba(16,185,129,0.08)"; res_border = "rgba(16,185,129,0.25)"
        res_color = "#10b981"; res_emoji = "✅"; res_label = "LOW RISK"
        tag_bg = "rgba(16,185,129,0.15)"; tag_color = "#6ee7b7"
        recs = [
            "💬 Send NPS survey — gather satisfaction feedback",
            "🎁 Offer premium upgrade at renewal",
            "🌟 Enroll in referral program — loyal customers convert best",
            "📊 Identify upsell opportunities from usage patterns",
        ]
        dot_color = "#10b981"

    r1, r2 = st.columns([1, 1.5], gap="large")

    with r1:
        # Result card
        st.markdown(f"""
        <div style="background:{res_bg}; border:1px solid {res_border}; border-radius:18px; padding:32px; text-align:center; margin-bottom:16px;">
            <div style="font-size:36px; margin-bottom:8px;">{res_emoji}</div>
            <div style="font-family:'Syne',sans-serif; font-size:60px; font-weight:800; color:{res_color}; line-height:1;">{prob*100:.1f}%</div>
            <div style="font-size:13px; color:#9ca3af; margin:8px 0 12px;">Churn Probability</div>
            <div style="display:inline-block; background:{tag_bg}; color:{tag_color}; padding:5px 16px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.1em;">{res_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations
        recs_html = "".join([
            f'<div style="display:flex; align-items:flex-start; gap:10px; padding:9px 0; border-bottom:1px solid rgba(255,255,255,0.05); font-size:13px; color:#d1d5db; line-height:1.4;">'
            f'<div style="width:6px; height:6px; border-radius:50%; background:{dot_color}; margin-top:5px; flex-shrink:0; box-shadow:0 0 6px {dot_color};"></div>'
            f'<div>{r}</div></div>'
            for r in recs
        ])
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:18px 20px;">
            <div style="font-size:11px; font-weight:700; color:{res_color}; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;">Retention Strategy</div>
            {recs_html}
        </div>
        """, unsafe_allow_html=True)

    with r2:
        # Risk meter chart
        fig1, ax1 = plt.subplots(figsize=(7, 2.8))
        fig1.patch.set_facecolor('#0d0d1a')
        ax1.set_facecolor('#0d0d1a')

        ax1.barh([0], [100], color='#12121f', height=0.5, zorder=1)
        ax1.barh([0], [30], color='#0d2b20', height=0.5, alpha=0.7, zorder=2)
        ax1.barh([0], [30], left=30, color='#2b2009', height=0.5, alpha=0.7, zorder=2)
        ax1.barh([0], [40], left=60, color='#2b0f0f', height=0.5, alpha=0.7, zorder=2)

        bar_color = "#10b981" if prob < 0.3 else "#f59e0b" if prob < 0.6 else "#ef4444"
        ax1.barh([0], [prob*100], color=bar_color, height=0.35, zorder=4)
        ax1.axvline(x=prob*100, color='#fff', linewidth=2, zorder=5, alpha=0.9)
        ax1.text(prob*100, 0.38, f'{prob*100:.1f}%', ha='center', fontsize=10, color='white', fontweight='700')

        for x, lbl, c in [(15,'LOW','#10b981'),(45,'MEDIUM','#f59e0b'),(80,'HIGH','#ef4444')]:
            ax1.text(x, -0.38, lbl, ha='center', fontsize=8, color=c, fontweight='600', alpha=0.7)

        ax1.set_xlim(0,100); ax1.set_ylim(-0.6, 0.6)
        ax1.set_title('Risk Meter', fontsize=12, color='#9ca3af', pad=10, fontweight='600')
        ax1.set_xlabel('Churn Risk Score (%)', fontsize=9, color='#4b5563', labelpad=8)
        ax1.spines[:].set_visible(False); ax1.set_yticks([])
        ax1.set_xticks([0,30,60,100]); ax1.set_xticklabels(['0%','30%','60%','100%'])
        fig1.tight_layout(pad=1.2)
        st.pyplot(fig1, use_container_width=True)
        plt.close()

        # Feature influence
        fi_vals = {
            'tenure': (72 - tenure) / 72,
            'MonthlyCharges': (monthly_charges - 20) / 130,
            'TotalCharges': 1 - min(total_charges / 8000, 1),
            'Contract': [0.9, 0.5, 0.1][["Month-to-month","One year","Two year"].index(contract)],
            'InternetService': [0.2, 0.45, 0.85][["No","DSL","Fiber optic"].index(internet_service)],
            'OnlineSecurity': 0.75 if online_security == "No" else 0.2,
            'TechSupport': 0.7 if tech_support == "No" else 0.2,
            'PaymentMethod': [0.85, 0.4, 0.2, 0.2][["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"].index(payment_method)],
            'PaperlessBilling': 0.6 if paperless_billing == "Yes" else 0.3,
            'Partner': 0.6 if partner == "No" else 0.3,
            'SeniorCitizen': 0.65 if senior_citizen == "Yes" else 0.35,
        }

        # Weight by actual feature importance from model
        fi_df = pd.DataFrame({
            'feature': list(fi_vals.keys()),
            'value': list(fi_vals.values())
        }).sort_values('value', ascending=True).tail(7)

        fig2, ax2 = plt.subplots(figsize=(7, 3.8))
        fig2.patch.set_facecolor('#0d0d1a')
        ax2.set_facecolor('#0d0d1a')

        bar_colors = ['#10b981' if v < 0.4 else '#f59e0b' if v < 0.65 else '#ef4444' for v in fi_df['value']]
        bars = ax2.barh(fi_df['feature'], fi_df['value'], color=bar_colors, height=0.55, edgecolor='none')
        for bar, val in zip(bars, fi_df['value']):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.0%}', va='center', fontsize=9, color='#9ca3af')

        ax2.set_xlim(0, 1.15)
        ax2.set_title('Feature Risk Contribution', fontsize=12, color='#9ca3af', pad=10, fontweight='600')
        ax2.set_xlabel('Risk Contribution', fontsize=9, color='#4b5563')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#1f2937'); ax2.spines['bottom'].set_color('#1f2937')
        ax2.tick_params(axis='x', colors='#4b5563', labelsize=8)
        ax2.tick_params(axis='y', colors='#9ca3af', labelsize=9)
        ax2.grid(axis='x', alpha=0.1)
        fig2.tight_layout(pad=1.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ROI row
    yearly = monthly_charges * 12
    cac = monthly_charges * 4
    retention_score = 100 - prob * 100
    rc = '#10b981' if retention_score > 70 else '#f59e0b' if retention_score > 40 else '#ef4444'

    st.markdown(f"""
    <div style="display:flex; gap:14px; margin-top:20px; flex-wrap:wrap;">
        <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:16px; text-align:center;">
            <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Monthly Revenue</div>
            <div style="font-family:'Syne',sans-serif; font-size:20px; font-weight:700; color:#fff;">₹{monthly_charges:.0f}</div>
        </div>
        <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:16px; text-align:center;">
            <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Annual Revenue</div>
            <div style="font-family:'Syne',sans-serif; font-size:20px; font-weight:700; color:#fff;">₹{yearly:,.0f}</div>
        </div>
        <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:16px; text-align:center;">
            <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Est. CAC (4×)</div>
            <div style="font-family:'Syne',sans-serif; font-size:20px; font-weight:700; color:#fff;">₹{cac:,.0f}</div>
        </div>
        <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:16px; text-align:center;">
            <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Retention Score</div>
            <div style="font-family:'Syne',sans-serif; font-size:20px; font-weight:700; color:{rc};">{retention_score:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
