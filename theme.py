# -*- coding: utf-8 -*-
from html import escape

import streamlit as st

# â”€â”€ GLOBAL CSS â€” injected once per session via session_state flag â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* â”€â”€ FLASH PREVENTION â€” hide content until fonts/styles load â”€â”€ */
html { visibility: visible !important; }
.stApp { opacity: 1 !important; }

/* â”€â”€ SMOOTH PAGE TRANSITION â”€â”€ */
.main .block-container {
    animation: fadeIn 0.15s ease-in-out;
}
@keyframes fadeIn {
    from { opacity: 0.7; transform: translateY(4px); }
    to   { opacity: 1;   transform: translateY(0);   }
}

/* â”€â”€ BASE â”€â”€ */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: #07070f !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e2ef;
}

/* â”€â”€ SIDEBAR â”€â”€ */
[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    transition: none !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* Keep toggle visible, hide default nav */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    color: white !important;
    background: #1a1a2e !important;
    border-radius: 0 8px 8px 0 !important;
}

/* â”€â”€ LAYOUT â”€â”€ */
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1280px; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #07070f; }
::-webkit-scrollbar-thumb { background: #1e1e30; border-radius: 3px; }

/* â”€â”€ TYPOGRAPHY â”€â”€ */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* â”€â”€ INPUTS â”€â”€ */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stTextInput"] input,
textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #e2e2ef !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label,
.stSlider label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    color: #6b7280 !important;
    font-weight: 400 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.stSlider > div > div > div       { background: rgba(99,102,241,0.2) !important; }
.stSlider > div > div > div > div { background: #6366f1 !important; }

/* â”€â”€ BUTTON â”€â”€ */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.45) !important;
}

/* â”€â”€ MISC COMPONENTS â”€â”€ */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stDataFrame"]    { border-radius: 12px !important; overflow: hidden !important; }
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label {
    color: #6b7280 !important; font-size: 11px !important; text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important; font-size: 24px !important; color: #fff !important;
}
[data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; color: #6b7280 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a5b4fc !important; border-bottom-color: #6366f1 !important;
}
.stAlert  { border-radius: 10px !important; }
hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 24px 0 !important; }
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}

/* â”€â”€ PAGE LINK STYLING â”€â”€ */
[data-testid="stPageLink"] a {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 10px 20px !important;
    font-size: 13px !important;
    color: #9ca3af !important;
    text-decoration: none !important;
    border-radius: 8px !important;
    margin: 2px 8px !important;
    transition: background 0.15s, color 0.15s !important;
}
[data-testid="stPageLink"] a:hover {
    background: rgba(99,102,241,0.12) !important;
    color: #e2e2ef !important;
}
[data-testid="stPageLink"][aria-current="page"] a {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
}

/* â”€â”€ SINGLE-SHELL SIDEBAR NAV â”€â”€ */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 2px !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 10px 20px !important;
    font-size: 13px !important;
    color: #9ca3af !important;
    text-decoration: none !important;
    border-radius: 8px !important;
    margin: 2px 8px !important;
    transition: background 0.15s, color 0.15s !important;
    min-height: 36px !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(99,102,241,0.12) !important;
    color: #e2e2ef !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: inherit !important;
    font-size: 13px !important;
    line-height: 1.2 !important;
    margin: 0 !important;
}


/* Hide Streamlit's native multipage nav across versions. */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"],
[data-testid="stSidebarNavLink"],
[data-testid="stSidebar"] nav,
section[data-testid="stSidebar"] nav,
ul[data-testid="stSidebarNavItems"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* Button-backed router nav; styled to match the original page links. */
.churn-nav-active {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 10px 20px !important;
    font-size: 13px !important;
    color: #a5b4fc !important;
    background: rgba(99,102,241,0.15) !important;
    text-decoration: none !important;
    border-radius: 8px !important;
    margin: 2px 8px !important;
    min-height: 36px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton > button {
    width: calc(100% - 16px) !important;
    min-height: 36px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 10px 20px !important;
    margin: 2px 8px !important;
    background: transparent !important;
    color: #9ca3af !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
    transition: background 0.15s, color 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,0.12) !important;
    color: #e2e2ef !important;
    transform: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:focus,
[data-testid="stSidebar"] .stButton > button:active {
    background: rgba(99,102,241,0.15) !important;
    color: #a5b4fc !important;
    box-shadow: none !important;
    outline: none !important;
}

</style>
"""

_SIDEBAR_HTML = """
<div style="padding:28px 20px 20px; border-bottom:1px solid rgba(255,255,255,0.06);">
    <div style="font-family:'Syne',sans-serif; font-size:22px; font-weight:800; color:#fff; margin-bottom:4px;">
        🔮 Churn<span style="background:linear-gradient(135deg,#6366f1,#a855f7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">Lens</span>
    </div>
    <div style="font-size:10px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin-top:2px;">AI Retention Platform</div>
</div>
<div style="padding:14px 20px 6px; font-size:10px; color:#374151; letter-spacing:0.1em; text-transform:uppercase;">Navigation</div>
"""

_SIDEBAR_STATUS = """
<div style="margin:20px 20px 0;">
    <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:10px; padding:12px 14px;">
        <div style="font-size:10px; color:#6366f1; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;">Model Status</div>
        <div style="font-size:12px; color:#9ca3af; margin-bottom:3px;">Random Forest · 11 features</div>
        <div style="font-size:12px; color:#9ca3af;">ROC-AUC: <span style="color:#a5b4fc; font-weight:600;">83%</span></div>
    </div>
</div>
"""

_PAGES = {
    "🏠 Home":                 "app.py",
    "📖 About":                "views/1_about.py",
    "🧙 Single Prediction":    "views/2_single_prediction.py",
    "📂 Bulk Prediction":      "views/3_bulk_prediction.py",
    "📊 Analytics Dashboard":  "views/4_analytics_dashboard.py",
    "🧠 Model Insights":       "views/5_model_insights.py",
    "🕒 Prediction History":   "views/6_history.py",
    "💬 AI Assistant":         "views/7_chat.py",
    "🔍 Explainable AI":       "views/8_explainability.py",
}

def apply_theme():
    """Inject the app theme once per script run."""
    if st.session_state.get("_router_rendering_page"):
        return
    st.markdown(_CSS, unsafe_allow_html=True)

def render_sidebar():
    """Render the classic Streamlit multipage sidebar for direct page access."""
    if st.session_state.get("_router_rendering_page"):
        return
    with st.sidebar:
        st.markdown(_SIDEBAR_HTML, unsafe_allow_html=True)
        for label, path in _PAGES.items():
            st.page_link(path, label=label)
        st.markdown(_SIDEBAR_STATUS, unsafe_allow_html=True)


def _set_active_route(label):
    st.session_state["active_route_label"] = label


def render_router_sidebar():
    """Render flicker-free in-app navigation and return the selected page path."""
    labels = list(_PAGES.keys())
    current = st.session_state.get("active_route_label", labels[0])
    if current not in _PAGES:
        current = labels[0]
        st.session_state["active_route_label"] = current

    with st.sidebar:
        st.markdown(_SIDEBAR_HTML, unsafe_allow_html=True)
        for i, label in enumerate(labels):
            if label == current:
                st.markdown(f'<div class="churn-nav-active">{escape(label)}</div>', unsafe_allow_html=True)
            else:
                st.button(
                    label,
                    key=f"router_nav_{i}",
                    use_container_width=True,
                    on_click=_set_active_route,
                    args=(label,),
                )
        st.markdown(_SIDEBAR_STATUS, unsafe_allow_html=True)

    return _PAGES[st.session_state.get("active_route_label", current)]

