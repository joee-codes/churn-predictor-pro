import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theme import apply_theme, render_sidebar
apply_theme()
render_sidebar()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE, 'churn_model.pkl'))
    features = joblib.load(os.path.join(BASE, 'feature_columns.pkl'))
    with open(os.path.join(BASE, 'model_metadata.json')) as f:
        meta = json.load(f)
    return model, features, meta

model, FEATURE_COLS, meta = load_artifacts()

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 28px;">
    <div style="font-family:'Syne',sans-serif; font-size:30px; font-weight:800; color:#fff; margin-bottom:6px;">
        📂 Bulk Prediction
    </div>
    <div style="font-size:14px; color:#6b7280;">Upload a customer CSV and get churn predictions for your entire base at once</div>
</div>
""", unsafe_allow_html=True)

# ── FORMAT INFO ───────────────────────────────────────────────────────────────
with st.expander("📋  Required CSV Format — click to expand", expanded=False):
    st.markdown("""
    <div style="font-size:13px; color:#9ca3af; line-height:1.8;">
    Your CSV must contain these columns (column names are case-sensitive):
    </div>
    """, unsafe_allow_html=True)

    sample = pd.DataFrame([{
        'tenure': 12,
        'MonthlyCharges': 70.0,
        'TotalCharges': 840.0,
        'Contract': 'Month-to-month',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'TechSupport': 'No',
        'PaymentMethod': 'Electronic check',
        'PaperlessBilling': 'Yes',
        'Partner': 'No',
        'SeniorCitizen': 0,
    }, {
        'tenure': 48,
        'MonthlyCharges': 55.0,
        'TotalCharges': 2640.0,
        'Contract': 'Two year',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'TechSupport': 'Yes',
        'PaymentMethod': 'Bank transfer (automatic)',
        'PaperlessBilling': 'No',
        'Partner': 'Yes',
        'SeniorCitizen': 0,
    }])
    st.dataframe(sample, use_container_width=True)

    # Download sample CSV
    csv_sample = sample.to_csv(index=False)
    st.download_button(
        "⬇️  Download Sample CSV",
        data=csv_sample,
        file_name="churnlens_sample.csv",
        mime="text/csv"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your customer CSV here",
    type=['csv'],
    help="CSV with the required columns listed above"
)

def preprocess_bulk(df):
    """Encode raw CSV columns into model-ready features."""
    d = df.copy()

    # Contract
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    d['Contract'] = d['Contract'].map(contract_map)

    # InternetService
    internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
    d['InternetService'] = d['InternetService'].map(internet_map)

    # PaymentMethod
    payment_map = {
        'Electronic check': 0, 'Mailed check': 1,
        'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
    }
    d['PaymentMethod'] = d['PaymentMethod'].map(payment_map)

    # Yes/No columns
    for col in ['OnlineSecurity', 'TechSupport', 'PaperlessBilling', 'Partner']:
        if col in d.columns:
            d[col] = d[col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    d['TotalCharges'] = pd.to_numeric(d['TotalCharges'], errors='coerce').fillna(0)
    d['SeniorCitizen'] = pd.to_numeric(d['SeniorCitizen'], errors='coerce').fillna(0).astype(int)
    d['tenure'] = pd.to_numeric(d['tenure'], errors='coerce').fillna(0).astype(int)
    d['MonthlyCharges'] = pd.to_numeric(d['MonthlyCharges'], errors='coerce').fillna(0)

    # Drop any rows where mapping failed
    d.dropna(subset=['Contract', 'InternetService', 'PaymentMethod'], inplace=True)

    return d[FEATURE_COLS]

def get_recommendation(prob):
    if prob > 0.6:
        return "Immediate outreach + discount offer"
    elif prob > 0.3:
        return "Proactive engagement + loyalty incentive"
    else:
        return "Regular engagement + upsell opportunity"

def get_risk(prob):
    if prob > 0.6: return "High"
    elif prob > 0.3: return "Medium"
    else: return "Low"

def remember_bulk_predictions(df_results):
    bulk_history = pd.DataFrame({
        'source': 'Bulk',
        'tenure': pd.to_numeric(df_results.get('tenure'), errors='coerce'),
        'monthly_charges': pd.to_numeric(df_results.get('MonthlyCharges'), errors='coerce'),
        'total_charges': pd.to_numeric(df_results.get('TotalCharges'), errors='coerce'),
        'contract': df_results.get('Contract'),
        'internet_service': df_results.get('InternetService'),
        'payment_method': df_results.get('PaymentMethod'),
        'churn_probability': pd.to_numeric(df_results.get('Churn_Probability'), errors='coerce'),
        'risk_category': df_results.get('Risk_Category'),
    })

    signature = f"{len(bulk_history)}:{bulk_history['churn_probability'].sum():.4f}:{bulk_history['monthly_charges'].sum():.4f}"
    if st.session_state.get('last_bulk_prediction_signature') == signature:
        return

    existing = st.session_state.get('bulk_prediction_history')
    if existing is None or len(existing) == 0:
        st.session_state['bulk_prediction_history'] = bulk_history
    else:
        st.session_state['bulk_prediction_history'] = pd.concat(
            [existing, bulk_history],
            ignore_index=True
        )
    st.session_state['last_bulk_prediction_signature'] = signature

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:10px; padding:12px 16px; font-size:13px; color:#6ee7b7; margin-bottom:20px;">
            ✅ Loaded <b>{len(df_raw):,}</b> customers · {df_raw.shape[1]} columns detected
        </div>
        """, unsafe_allow_html=True)

        # Check required columns
        missing_cols = [c for c in FEATURE_COLS if c not in df_raw.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()

        # Process & predict
        with st.spinner("Running predictions..."):
            X = preprocess_bulk(df_raw)
            if X.empty:
                st.error("No valid rows found after preprocessing. Check the category values in your CSV.")
                st.stop()
            probs = model.predict_proba(X)[:, 1]

        df_results = df_raw.loc[X.index].copy()
        df_results['Churn_Probability'] = (probs * 100).round(1)
        df_results['Risk_Category'] = [get_risk(p) for p in probs]
        df_results['Recommendation'] = [get_recommendation(p) for p in probs]
        remember_bulk_predictions(df_results)

        # ── SUMMARY KPIs ──────────────────────────────────────────────────────
        total = len(df_results)
        high = (df_results['Risk_Category'] == 'High').sum()
        med  = (df_results['Risk_Category'] == 'Medium').sum()
        low  = (df_results['Risk_Category'] == 'Low').sum()
        avg_prob = probs.mean() * 100

        rev_at_risk = df_results[df_results['Risk_Category'] == 'High']['MonthlyCharges'].sum() if 'MonthlyCharges' in df_results.columns else 0

        st.markdown(f"""
        <div style="display:flex; gap:14px; margin:20px 0; flex-wrap:wrap;">
            <div style="flex:1; min-width:130px; background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:16px 18px; text-align:center;">
                <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Total Analyzed</div>
                <div style="font-family:'Syne',sans-serif; font-size:26px; font-weight:800; color:#a5b4fc;">{total:,}</div>
            </div>
            <div style="flex:1; min-width:130px; background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:12px; padding:16px 18px; text-align:center;">
                <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">High Risk</div>
                <div style="font-family:'Syne',sans-serif; font-size:26px; font-weight:800; color:#fca5a5;">{high:,}</div>
                <div style="font-size:11px; color:#6b7280;">{high/total*100:.1f}% of base</div>
            </div>
            <div style="flex:1; min-width:130px; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); border-radius:12px; padding:16px 18px; text-align:center;">
                <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Medium Risk</div>
                <div style="font-family:'Syne',sans-serif; font-size:26px; font-weight:800; color:#fcd34d;">{med:,}</div>
                <div style="font-size:11px; color:#6b7280;">{med/total*100:.1f}% of base</div>
            </div>
            <div style="flex:1; min-width:130px; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:12px; padding:16px 18px; text-align:center;">
                <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Low Risk</div>
                <div style="font-family:'Syne',sans-serif; font-size:26px; font-weight:800; color:#6ee7b7;">{low:,}</div>
                <div style="font-size:11px; color:#6b7280;">{low/total*100:.1f}% of base</div>
            </div>
            <div style="flex:1; min-width:130px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px 18px; text-align:center;">
                <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Avg Churn Risk</div>
                <div style="font-family:'Syne',sans-serif; font-size:26px; font-weight:800; color:#fff;">{avg_prob:.1f}%</div>
            </div>
            {'<div style="flex:1; min-width:130px; background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.15); border-radius:12px; padding:16px 18px; text-align:center;"><div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Revenue at Risk/mo</div><div style="font-family:Syne,sans-serif; font-size:26px; font-weight:800; color:#fca5a5;">₹' + f"{rev_at_risk:,.0f}" + '</div></div>' if rev_at_risk > 0 else ''}
        </div>
        """, unsafe_allow_html=True)

        # ── RESULTS TABLE ─────────────────────────────────────────────────────
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'Syne\',sans-serif; font-size:16px; font-weight:700; color:#fff; margin-bottom:16px;">Results Table</div>', unsafe_allow_html=True)

        # Filter
        risk_options = ["All", "High", "Medium", "Low"]
        if hasattr(st, "pills"):
            risk_filter = st.pills("Filter by Risk", risk_options, default="All")
        else:
            risk_filter = st.segmented_control("Filter by Risk", risk_options, default="All")

        display_df = df_results if risk_filter == "All" else df_results[df_results['Risk_Category'] == risk_filter]

        # Style the Risk_Category column
        def style_risk(val):
            colors = {'High': 'color: #fca5a5', 'Medium': 'color: #fcd34d', 'Low': 'color: #6ee7b7'}
            return colors.get(val, '')

        cols_show = ['tenure', 'MonthlyCharges', 'Contract', 'Churn_Probability', 'Risk_Category', 'Recommendation']
        cols_show = [c for c in cols_show if c in display_df.columns]

        st.dataframe(
            display_df[cols_show].style.map(style_risk, subset=['Risk_Category']),
            use_container_width=True,
            height=400
        )

        # ── DOWNLOAD ──────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        csv_out = df_results.to_csv(index=False)
        st.download_button(
            label="⬇️  Download Full Results CSV",
            data=csv_out,
            file_name="churnlens_predictions.csv",
            mime="text/csv",
            use_container_width=False
        )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Make sure your CSV has the required columns. Download the sample CSV above for reference.")

else:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.08); border-radius:16px;">
        <div style="font-size:40px; margin-bottom:12px;">📂</div>
        <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:600; color:#6b7280; margin-bottom:6px;">Upload a CSV to get started</div>
        <div style="font-size:13px; color:#374151;">Supports up to 50,000+ customers in one batch</div>
    </div>
    """, unsafe_allow_html=True)
