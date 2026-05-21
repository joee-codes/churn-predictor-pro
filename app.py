import streamlit as st

st.set_page_config(
    page_title="ChurnLens",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background: #07070f !important; font-family: 'DM Sans', sans-serif; color: #e2e2ef; }
[data-testid="stSidebar"] { background: #0d0d1a !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1280px; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #07070f; }
::-webkit-scrollbar-thumb { background: #1e1e30; border-radius: 3px; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] > div > div, div[data-testid="stTextInput"] input, textarea {
    background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.09) !important;
    color: #e2e2ef !important; border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stNumberInput"] label, div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label, .stSlider label {
    font-family: 'DM Sans', sans-serif !important; font-size: 12px !important; color: #6b7280 !important;
    font-weight: 400 !important; letter-spacing: 0.04em !important; text-transform: uppercase !important;
}
.stSlider > div > div > div { background: rgba(99,102,241,0.2) !important; }
.stSlider > div > div > div > div { background: #6366f1 !important; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important;
    font-family: 'Syne', sans-serif !important; font-size: 14px !important; font-weight: 600 !important;
    border: none !important; border-radius: 10px !important; padding: 12px 28px !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important; transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(99,102,241,0.45) !important; }
[data-testid="stFileUploader"] { background: rgba(255,255,255,0.02) !important; border: 1px dashed rgba(99,102,241,0.3) !important; border-radius: 12px !important; }
[data-testid="metric-container"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 12px !important; padding: 16px 20px !important; }
[data-testid="metric-container"] label { color: #6b7280 !important; font-size: 11px !important; text-transform: uppercase !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 24px !important; color: #fff !important; }
[data-testid="stTabs"] button { font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; color: #6b7280 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #a5b4fc !important; border-bottom-color: #6366f1 !important; }
.stAlert { border-radius: 10px !important; }
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 24px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:28px 20px 20px; border-bottom:1px solid rgba(255,255,255,0.06);">
        <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:800; color:#fff; margin-bottom:4px;">
            🔮 Churn<span style="background:linear-gradient(135deg,#6366f1,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Lens</span>
        </div>
        <div style="font-size:10px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px;">AI Retention Platform</div>
    </div>
    <div style="padding:14px 20px 6px; font-size:10px; color:#374151; letter-spacing:0.1em; text-transform:uppercase;">Navigation</div>
    """, unsafe_allow_html=True)

    pages = {
        "🏠  About": "pages/1_about.py",
        "🔮  Single Prediction": "pages/2_single_prediction.py",
        "📂  Bulk Prediction": "pages/3_bulk_prediction.py",
        "📈  Analytics Dashboard": "pages/4_analytics_dashboard.py",
        "🧠  Model Insights": "pages/5_model_insights.py",
    }
    for label in pages:
        st.page_link(pages[label], label=label)

    st.markdown("""
    <div style="margin:20px 20px 0;">
        <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:10px; padding:12px 14px;">
            <div style="font-size:10px; color:#6366f1; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;">Model Status</div>
            <div style="font-size:12px; color:#9ca3af; margin-bottom:3px;">Random Forest · 11 features</div>
            <div style="font-size:12px; color:#9ca3af;">ROC-AUC: <span style="color:#a5b4fc; font-weight:600;">~83%</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── HOME PAGE — split into st.columns to avoid raw HTML bug ──────────────────

# Badge
st.markdown("""
<div style="padding-top:40px; margin-bottom:22px;">
    <div style="display:inline-flex; align-items:center; gap:7px; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3); border-radius:20px; padding:5px 14px;">
        <div style="width:6px;height:6px;background:#6366f1;border-radius:50%;box-shadow:0 0 8px #6366f1;"></div>
        <span style="font-size:11px;color:#a5b4fc;letter-spacing:0.12em;text-transform:uppercase;font-weight:500;">Live · Random Forest Model</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero split into 2 columns
hero_left, hero_right = st.columns([1.1, 0.9], gap="large")

with hero_left:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:46px; font-weight:800; color:#fff; line-height:1.08; margin-bottom:18px; letter-spacing:-0.02em;">
        Know Who's<br>Leaving
        <span style="background:linear-gradient(135deg,#6366f1,#a855f7,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent;"> Before</span><br>They Do.
    </div>
    <div style="font-size:15px; color:#6b7280; line-height:1.75; font-weight:300; max-width:480px; margin-bottom:28px;">
        ChurnLens uses machine learning to predict which customers are about to leave — so your team can act fast, save revenue, and build lasting relationships.
    </div>
    <div style="display:flex; gap:10px; flex-wrap:wrap;">
        <div style="display:flex; align-items:center; gap:8px; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.2); border-radius:8px; padding:8px 14px;">
            <span style="font-size:13px;">✅</span>
            <span style="font-size:12px; color:#6ee7b7; font-weight:500;">83% ROC-AUC Accuracy</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.2); border-radius:8px; padding:8px 14px;">
            <span style="font-size:13px;">⚡</span>
            <span style="font-size:12px; color:#a5b4fc; font-weight:500;">Real-time Predictions</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px; background:rgba(168,85,247,0.1); border:1px solid rgba(168,85,247,0.2); border-radius:8px; padding:8px 14px;">
            <span style="font-size:13px;">📂</span>
            <span style="font-size:12px; color:#c084fc; font-weight:500;">Bulk CSV Support</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:22px; position:relative; overflow:hidden; margin-top:8px;">
        <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#6366f1,#a855f7,#ec4899);"></div>
        <div style="font-size:11px; color:#6366f1; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:16px;">Live Risk Monitor</div>
        <div style="display:flex; flex-direction:column; gap:10px;">
            <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:10px; padding:12px 14px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:12px; color:#fff; font-weight:500;">Customer #4821</div>
                    <div style="font-size:10px; color:#6b7280; margin-top:2px;">Month-to-month · Fiber optic</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:800; color:#ef4444;">78%</div>
                    <div style="font-size:9px; color:#fca5a5; text-transform:uppercase;">High Risk</div>
                </div>
            </div>
            <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); border-radius:10px; padding:12px 14px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:12px; color:#fff; font-weight:500;">Customer #2034</div>
                    <div style="font-size:10px; color:#6b7280; margin-top:2px;">One year · DSL</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:800; color:#f59e0b;">41%</div>
                    <div style="font-size:9px; color:#fcd34d; text-transform:uppercase;">Medium Risk</div>
                </div>
            </div>
            <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:10px; padding:12px 14px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:12px; color:#fff; font-weight:500;">Customer #7719</div>
                    <div style="font-size:10px; color:#6b7280; margin-top:2px;">Two year · DSL</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:800; color:#10b981;">8%</div>
                    <div style="font-size:9px; color:#6ee7b7; text-transform:uppercase;">Low Risk</div>
                </div>
            </div>
        </div>
        <div style="margin-top:16px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.06); display:flex; justify-content:space-between;">
            <div style="text-align:center;">
                <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:800; color:#fca5a5;">1,869</div>
                <div style="font-size:9px; color:#6b7280; text-transform:uppercase; margin-top:2px;">At Risk</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:800; color:#fcd34d;">26.5%</div>
                <div style="font-size:9px; color:#6b7280; text-transform:uppercase; margin-top:2px;">Churn Rate</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:800; color:#a5b4fc;">$139K</div>
                <div style="font-size:9px; color:#6b7280; text-transform:uppercase; margin-top:2px;">Rev at Risk</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)



st.markdown("""
<div style="text-align:center; padding:16px 0 8px; font-size:12px; color:#374151; letter-spacing:0.04em; border-top:1px solid rgba(255,255,255,0.05); margin-top:8px;">
    Built with <span style="color:#6366f1;">Streamlit</span> · Random Forest · Telco Customer Churn Dataset · 7,043 customers
</div>
""", unsafe_allow_html=True)