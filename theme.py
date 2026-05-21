import streamlit as st

def render_sidebar():
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
            "🏠  Home": "app.py",
            "📖  About": "pages/1_about.py",
            "🔮  Single Prediction": "pages/2_single_prediction.py",
            "📂  Bulk Prediction": "pages/3_bulk_prediction.py",
            "📈  Analytics Dashboard": "pages/4_analytics_dashboard.py",
            "🧠  Model Insights": "pages/5_model_insights.py",
            "🕓  Prediction History": "pages/6_history.py",
            "💬  AI Assistant": "pages/7_chat.py",
            "🔍  Explainable AI": "pages/8_explainability.py",
        }
        for label, path in pages.items():
            st.page_link(path, label=label)

        st.markdown("""
        <div style="margin:20px 20px 0;">
            <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:10px; padding:12px 14px;">
                <div style="font-size:10px; color:#6366f1; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;">Model Status</div>
                <div style="font-size:12px; color:#9ca3af; margin-bottom:3px;">Random Forest · 11 features</div>
                <div style="font-size:12px; color:#9ca3af;">ROC-AUC: <span style="color:#a5b4fc; font-weight:600;">~83%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def apply_theme():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background: #07070f !important; font-family: 'DM Sans', sans-serif; color: #e2e2ef; }
[data-testid="stSidebar"] { background: #0d0d1a !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
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
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }
[data-testid="metric-container"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 12px !important; padding: 16px 20px !important; }
[data-testid="metric-container"] label { color: #6b7280 !important; font-size: 11px !important; text-transform: uppercase !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 24px !important; color: #fff !important; }
[data-testid="stTabs"] button { font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; color: #6b7280 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #a5b4fc !important; border-bottom-color: #6366f1 !important; }
.stAlert { border-radius: 10px !important; }
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 24px 0 !important; }
[data-testid="stExpander"] { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)
