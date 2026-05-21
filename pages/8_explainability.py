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

apply_theme()
render_sidebar()

plt.rcParams.update({
    'figure.facecolor': '#0d0d1a', 'axes.facecolor': '#0d0d1a',
    'axes.edgecolor': '#1f2937', 'axes.labelcolor': '#6b7280',
    'xtick.color': '#6b7280', 'ytick.color': '#6b7280',
    'text.color': '#e2e2ef', 'grid.color': '#1f2937',
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE, 'churn_model.pkl'))
    features = joblib.load(os.path.join(BASE, 'feature_columns.pkl'))
    with open(os.path.join(BASE, 'model_metadata.json')) as f:
        meta = json.load(f)
    return model, features, meta

model, FEATURE_COLS, meta = load_artifacts()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 28px;">
    <div style="font-family:'Syne',sans-serif; font-size:30px; font-weight:800; color:#fff; margin-bottom:6px;">
        🔍 Explainable AI (SHAP)
    </div>
    <div style="font-size:14px; color:#6b7280;">Understand <i>why</i> the model predicts churn for any customer profile</div>
</div>
""", unsafe_allow_html=True)

# ── SHAP CHECK ────────────────────────────────────────────────────────────────
try:
    import shap
    shap_available = True
except ImportError:
    shap_available = False

if not shap_available:
    st.markdown("""
    <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25); border-radius:12px; padding:16px 20px; margin-bottom:24px;">
        <b style="color:#fcd34d;">⚠️ SHAP not installed.</b>
        <div style="font-size:13px; color:#9ca3af; margin-top:6px;">Run: <code style="background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:4px;">pip install shap</code> then restart the app.</div>
    </div>
    """, unsafe_allow_html=True)

# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div style="font-size:11px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:16px;">Customer Profile</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div style="font-size:12px; color:#6366f1; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">Account Info</div>', unsafe_allow_html=True)
    tenure = st.slider("Tenure (months)", 0, 72, 6, key="shap_tenure")
    monthly_charges = st.number_input("Monthly Charges ($)", 20.0, 150.0, 85.0, 5.0, key="shap_mc")
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges), 50.0, key="shap_tc")
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], key="shap_sc")
    partner = st.selectbox("Partner", ["No", "Yes"], key="shap_p")

with col2:
    st.markdown('<div style="font-size:12px; color:#a855f7; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">Contract & Billing</div>', unsafe_allow_html=True)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="shap_ct")
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ], key="shap_pm")
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], key="shap_pb")
    internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"], key="shap_is")

with col3:
    st.markdown('<div style="font-size:12px; color:#ec4899; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:14px;">Services</div>', unsafe_allow_html=True)
    online_security = st.selectbox("Online Security", ["No", "Yes"], key="shap_os")
    tech_support = st.selectbox("Tech Support", ["No", "Yes"], key="shap_ts")

st.markdown("<br>", unsafe_allow_html=True)
col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    explain = st.button("🔍  Explain This Prediction", use_container_width=True)

# ── EXPLANATION ───────────────────────────────────────────────────────────────
if explain:
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

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── PREDICTION BADGE ──────────────────────────────────────────────────────
    if prob > 0.6:
        res_color, res_bg, res_border, res_label, res_emoji = "#ef4444", "rgba(239,68,68,0.08)", "rgba(239,68,68,0.25)", "HIGH RISK", "🚨"
    elif prob > 0.3:
        res_color, res_bg, res_border, res_label, res_emoji = "#f59e0b", "rgba(245,158,11,0.08)", "rgba(245,158,11,0.25)", "MEDIUM RISK", "⚠️"
    else:
        res_color, res_bg, res_border, res_label, res_emoji = "#10b981", "rgba(16,185,129,0.08)", "rgba(16,185,129,0.25)", "LOW RISK", "✅"

    r1, r2 = st.columns([1, 2], gap="large")

    with r1:
        st.markdown(f"""
        <div style="background:{res_bg}; border:1px solid {res_border}; border-radius:16px; padding:28px; text-align:center;">
            <div style="font-size:32px; margin-bottom:8px;">{res_emoji}</div>
            <div style="font-family:'Syne',sans-serif; font-size:52px; font-weight:800; color:{res_color}; line-height:1;">{prob*100:.1f}%</div>
            <div style="font-size:12px; color:#9ca3af; margin:8px 0 10px;">Churn Probability</div>
            <div style="display:inline-block; background:rgba(255,255,255,0.08); color:{res_color}; padding:4px 14px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:0.1em;">{res_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        # ── SHAP OR FALLBACK ──────────────────────────────────────────────────
        if shap_available:
            with st.spinner("Computing SHAP values..."):
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_df)

                    # For binary classification, use class 1 (churn)
                    if isinstance(shap_values, list):
                        sv = shap_values[1][0]
                    else:
                        sv = shap_values[0]

                    base_value = explainer.expected_value
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = base_value[1]

                    shap_df = pd.DataFrame({
                        'Feature': FEATURE_COLS,
                        'SHAP Value': sv,
                        'Input Value': input_df.values[0]
                    }).sort_values('SHAP Value', key=abs, ascending=True)

                    # SHAP waterfall bar chart
                    fig, ax = plt.subplots(figsize=(8, 5))
                    fig.patch.set_facecolor('#0d0d1a')
                    ax.set_facecolor('#0d0d1a')

                    colors = ['#ef4444' if v > 0 else '#10b981' for v in shap_df['SHAP Value']]
                    bars = ax.barh(shap_df['Feature'], shap_df['SHAP Value'],
                                   color=colors, height=0.6, edgecolor='none')

                    for bar, val in zip(bars, shap_df['SHAP Value']):
                        x = val + (0.002 if val >= 0 else -0.002)
                        ha = 'left' if val >= 0 else 'right'
                        ax.text(x, bar.get_y() + bar.get_height()/2,
                                f'{val:+.3f}', va='center', ha=ha, fontsize=9, color='#9ca3af')

                    ax.axvline(x=0, color='#374151', linewidth=1, zorder=5)
                    ax.set_title('SHAP Feature Contributions', color='white', fontsize=13, fontweight='600', pad=12)
                    ax.set_xlabel('SHAP Value (impact on churn probability)', fontsize=9, color='#6b7280')
                    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#1f2937'); ax.spines['bottom'].set_color('#1f2937')
                    ax.tick_params(axis='y', colors='#9ca3af', labelsize=9)
                    ax.grid(axis='x', alpha=0.1)

                    red_patch = mpatches.Patch(color='#ef4444', label='Increases churn risk')
                    green_patch = mpatches.Patch(color='#10b981', label='Decreases churn risk')
                    ax.legend(handles=[red_patch, green_patch], fontsize=8,
                              facecolor='#12121f', edgecolor='#1f2937', labelcolor='#9ca3af',
                              loc='lower right')

                    fig.tight_layout(pad=1.5)
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

                except Exception as e:
                    st.error(f"SHAP error: {str(e)}")
                    shap_available = False

        if not shap_available:
            # ── FALLBACK: Feature influence chart ────────────────────────────
            fi_dict = meta.get('feature_importance', {})
            fi_vals = {f: model.feature_importances_[i] for i, f in enumerate(FEATURE_COLS)}

            # Weighted directional influence
            risk_direction = {
                'tenure': -1,           # higher tenure = lower churn
                'MonthlyCharges': 1,
                'TotalCharges': -1,
                'Contract': -1,
                'InternetService': 1,
                'OnlineSecurity': -1,
                'TechSupport': -1,
                'PaymentMethod': -1,
                'PaperlessBilling': 1,
                'Partner': -1,
                'SeniorCitizen': 1,
            }

            shap_approx = {}
            for f in FEATURE_COLS:
                norm_val = input_df[f].values[0]
                importance = fi_vals.get(f, 0.05)
                direction = risk_direction.get(f, 1)
                shap_approx[f] = importance * direction * (norm_val / (norm_val + 1))

            shap_df = pd.DataFrame({
                'Feature': list(shap_approx.keys()),
                'SHAP Value': list(shap_approx.values())
            }).sort_values('SHAP Value', key=abs, ascending=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0d0d1a')
            ax.set_facecolor('#0d0d1a')

            colors = ['#ef4444' if v > 0 else '#10b981' for v in shap_df['SHAP Value']]
            ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors, height=0.6, edgecolor='none')
            ax.axvline(x=0, color='#374151', linewidth=1)
            ax.set_title('Feature Influence (approximate)', color='white', fontsize=13, fontweight='600', pad=12)
            ax.set_xlabel('Influence on churn probability', fontsize=9, color='#6b7280')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#1f2937'); ax.spines['bottom'].set_color('#1f2937')
            ax.tick_params(axis='y', colors='#9ca3af', labelsize=9)
            ax.grid(axis='x', alpha=0.1)
            fig.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ── NATURAL LANGUAGE EXPLANATION ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-size:16px; font-weight:700; color:#fff; margin-bottom:16px;">Why is this customer at risk?</div>', unsafe_allow_html=True)

    reasons = []
    actions = []

    if input_dict['Contract'] == 0:
        reasons.append(("🔴", "Month-to-month contract", "No lock-in commitment — easiest to cancel"))
        actions.append("Offer 20% discount to upgrade to annual plan")
    elif input_dict['Contract'] == 1:
        reasons.append(("🟡", "One-year contract", "Moderate commitment level"))

    if input_dict['InternetService'] == 2:
        reasons.append(("🔴", "Fiber optic internet", "High cost service with 42% average churn rate"))
        actions.append("Offer a speed upgrade or bundle discount")

    if input_dict['PaymentMethod'] == 0:
        reasons.append(("🔴", "Electronic check payment", "Manual payment = higher churn risk (45% rate)"))
        actions.append("Incentivize switch to auto-pay with $5/month discount")

    if tenure < 6:
        reasons.append(("🔴", f"Very new customer ({tenure} months)", "First 6 months = highest churn window (53% rate)"))
        actions.append("Assign onboarding specialist immediately")
    elif tenure < 12:
        reasons.append(("🟡", f"Early tenure ({tenure} months)", "Still in high-risk period"))

    if input_dict['OnlineSecurity'] == 0:
        reasons.append(("🟡", "No online security", "Unprotected customers churn 2x more"))
        actions.append("Offer free 3-month trial of online security")

    if input_dict['TechSupport'] == 0:
        reasons.append(("🟡", "No tech support", "Unresolved issues drive frustration and churn"))
        actions.append("Proactively offer tech support check-in call")

    if input_dict['SeniorCitizen'] == 1:
        reasons.append(("🟡", "Senior citizen", "65% more likely to churn — needs extra support"))
        actions.append("Assign dedicated senior support line")

    if monthly_charges > 80:
        reasons.append(("🟡", f"High monthly charges (${monthly_charges:.0f})", "Price-sensitive customers churn more"))
        actions.append("Review pricing — consider loyalty discount")

    if not reasons:
        reasons.append(("🟢", "Low-risk profile overall", "Customer shows strong retention signals"))
        actions.append("Focus on upsell and referral opportunities")

    r_col1, r_col2 = st.columns(2, gap="large")

    with r_col1:
        st.markdown('<div style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;">Risk Factors</div>', unsafe_allow_html=True)
        reasons_html = "".join([
            f'<div style="display:flex; gap:10px; align-items:flex-start; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<span style="font-size:14px; margin-top:1px;">{emoji}</span>'
            f'<div><div style="font-size:13px; color:#e2e2ef; font-weight:500;">{title}</div>'
            f'<div style="font-size:11px; color:#6b7280; margin-top:2px;">{desc}</div></div></div>'
            for emoji, title, desc in reasons
        ])
        st.markdown(f'<div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px 18px;">{reasons_html}</div>', unsafe_allow_html=True)

    with r_col2:
        st.markdown('<div style="font-size:11px; color:#6b7280; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;">Recommended Actions</div>', unsafe_allow_html=True)
        actions_html = "".join([
            f'<div style="display:flex; gap:10px; align-items:flex-start; padding:10px 0; border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<div style="width:6px; height:6px; border-radius:50%; background:#6366f1; margin-top:5px; flex-shrink:0; box-shadow:0 0 6px #6366f1;"></div>'
            f'<div style="font-size:13px; color:#d1d5db; line-height:1.4;">{a}</div></div>'
            for a in actions
        ])
        st.markdown(f'<div style="background:rgba(99,102,241,0.05); border:1px solid rgba(99,102,241,0.15); border-radius:12px; padding:16px 18px;">{actions_html}</div>', unsafe_allow_html=True)
