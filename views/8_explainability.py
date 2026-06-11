import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theme import apply_theme, render_sidebar

apply_theme()
render_sidebar()

st.markdown("""
<div style="padding:32px 0 28px;">
    <div style="font-family:'Syne',sans-serif; font-size:30px; font-weight:800; color:#fff; margin-bottom:6px;">
        🔍 Explainable AI (SHAP)
    </div>
    <div style="font-size:14px; color:#6b7280;">Analyze a customer profile and see which features push churn risk up or down</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div style="font-size:11px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:16px; display:flex; align-items:center; gap:10px;">Customer Profile <span style="flex:1; height:1px; background:rgba(255,255,255,0.06); display:inline-block;"></span></div>',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* Scoped polish for Explainable AI inputs. Streamlit selectboxes include a
   hidden search input, so this makes the visible control behave like a
   dropdown while preserving the native chevron. */
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    cursor: pointer !important;
    min-height: 48px !important;
    font-size: 14px !important;
}
div[data-testid="stSelectbox"] input,
div[data-testid="stSelectbox"] input:hover,
div[data-testid="stSelectbox"] input:focus {
    caret-color: transparent !important;
    cursor: pointer !important;
    pointer-events: none !important;
    user-select: none !important;
}
.calculated-total {
    min-height: 48px;
    display: flex;
    align-items: center;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    padding: 0 16px;
    color: #e2e2ef;
    font-size: 15px;
    font-family: 'DM Sans', sans-serif;
}
.field-help {
    font-size: 11px;
    color: #6b7280;
    margin-top: -8px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div style="font-size:12px; color:#6366f1; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">Account Info</div>', unsafe_allow_html=True)
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=10000, value=0, step=1, key="xai_tenure")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=10000.0, value=0.0, step=5.0, format="%.2f", key="xai_monthly")
    total_charges = round(float(tenure or 0) * float(monthly_charges or 0), 2)
    st.markdown(
        f'<div style="font-size:12px; color:#6b7280; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:6px;">Total Charges ($)</div>'
        f'<div class="calculated-total">{total_charges:,.2f}</div>'
        f'<div class="field-help">Automatically calculated from Tenure × Monthly Charges</div>',
        unsafe_allow_html=True,
    )
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])

with col2:
    st.markdown('<div style="font-size:12px; color:#a855f7; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">Contract & Billing</div>', unsafe_allow_html=True)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    )
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

with col3:
    st.markdown('<div style="font-size:12px; color:#ec4899; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">Services</div>', unsafe_allow_html=True)
    internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
    online_security = st.selectbox("Online Security", ["No", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])

st.markdown("<br>", unsafe_allow_html=True)
button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
with button_col2:
    analyze = st.button("🔍  Analyze Customer Churn Risk", use_container_width=True)

if analyze:
    risk_score = 83 if contract == "Month-to-month" else 48 if contract == "One year" else 18
    risk_label = "High Churn Risk" if risk_score >= 60 else "Medium Churn Risk" if risk_score >= 30 else "Low Churn Risk"
    risk_color = "#ef4444" if risk_score >= 60 else "#f59e0b" if risk_score >= 30 else "#10b981"

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:22px 24px; margin-bottom:22px;">
        <div style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">Simulated Prediction</div>
        <div style="font-family:'Syne',sans-serif; font-size:34px; font-weight:800; color:{risk_color};">{risk_label}: {risk_score}%</div>
        <div style="font-size:13px; color:#9ca3af; margin-top:8px;">Explanation generated for a {contract} customer with {internet_service} internet and {payment_method} payment.</div>
    </div>
    """, unsafe_allow_html=True)

    shap_data = pd.DataFrame({
        "Risk Push": {
            "Month-to-month Contract": 0.34 if contract == "Month-to-month" else 0.08,
            "Electronic Check Payment": 0.22 if payment_method == "Electronic check" else 0.05,
            "Fiber Optic Service": 0.18 if internet_service == "Fiber optic" else 0.02,
            "No Online Security": 0.16 if online_security == "No" else 0.00,
            "No Tech Support": 0.14 if tech_support == "No" else 0.00,
            "High Tenure": 0.00,
            "Partner Account": 0.00,
        },
        "Retention Pull": {
            "Month-to-month Contract": 0.00,
            "Electronic Check Payment": 0.00,
            "Fiber Optic Service": 0.00,
            "No Online Security": 0.00,
            "No Tech Support": 0.00,
            "High Tenure": -0.24 if tenure >= 36 else -0.06,
            "Partner Account": -0.12 if partner == "Yes" else -0.03,
        },
    })

    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-size:18px; font-weight:700; color:#fff; margin-bottom:10px;">Feature Impact Explanation</div>', unsafe_allow_html=True)
    st.bar_chart(shap_data, height=360)

    st.markdown(f"""
    <div style="background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.18); border-radius:14px; padding:20px 22px; margin-top:22px;">
        <div style="font-size:11px; color:#a5b4fc; text-transform:uppercase; letter-spacing:0.1em; font-weight:700; margin-bottom:12px;">Top 3 Reasons</div>
        <div style="font-size:14px; color:#d1d5db; line-height:1.8;">
            <b>1.</b> Contract type is <b>{contract}</b>, which strongly influences churn behavior.<br>
            <b>2.</b> Payment method is <b>{payment_method}</b>, a key billing-related churn signal.<br>
            <b>3.</b> Service protection status: Online Security is <b>{online_security}</b> and Tech Support is <b>{tech_support}</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center; padding:48px 20px; background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.08); border-radius:16px;">
        <div style="font-size:34px; margin-bottom:10px;">🔍</div>
        <div style="font-family:'Syne',sans-serif; color:#9ca3af; font-weight:700; margin-bottom:6px;">Ready to Explain</div>
        <div style="font-size:13px; color:#4b5563;">Enter a customer profile and click Analyze Customer Churn Risk.</div>
    </div>
    """, unsafe_allow_html=True)
