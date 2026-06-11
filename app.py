import runpy
from pathlib import Path

import streamlit as st

from theme import apply_theme, render_router_sidebar

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ChurnLens",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()
selected_page = render_router_sidebar()


def render_home():
    st.markdown("""
    <div style="padding-top:40px; margin-bottom:22px;">
        <div style="display:inline-flex; align-items:center; gap:7px; background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3); border-radius:20px; padding:5px 14px;">
            <div style="width:6px;height:6px;background:#6366f1;border-radius:50%;box-shadow:0 0 8px #6366f1;"></div>
            <span style="font-size:11px;color:#a5b4fc;letter-spacing:0.12em;text-transform:uppercase;font-weight:500;">Live · Random Forest Model</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        # Customer risk rows — static demo preview
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:22px 22px 14px; position:relative; overflow:hidden; margin-top:8px;">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#6366f1,#a855f7,#ec4899);"></div>
            <div style="font-size:11px; color:#6366f1; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:16px;">Live Risk Monitor Demo Preview</div>
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
            <div style="height:1px; background:rgba(255,255,255,0.06); margin:14px 0 4px;"></div>
        </div>
        """, unsafe_allow_html=True)

        # KPI status cards — rendered via st.columns to avoid raw HTML grid bug
        k1, k2, k3 = st.columns(3, gap="small")
        with k1:
            st.markdown("""
            <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.3); border-radius:10px; padding:12px 6px; text-align:center; position:relative; overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#6366f1,#a855f7);"></div>
                <div style="width:6px;height:6px;border-radius:50%;background:#6366f1;box-shadow:0 0 8px #6366f1;margin:0 auto 6px;"></div>
                <div style="font-family:'Syne',sans-serif;font-size:10px;font-weight:700;color:#a5b4fc;margin-bottom:4px;">AI Monitoring</div>
                <div style="font-size:9px;color:#6ee7b7;font-weight:600;letter-spacing:0.04em;">● Active</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown("""
            <div style="background:rgba(168,85,247,0.08); border:1px solid rgba(168,85,247,0.3); border-radius:10px; padding:12px 6px; text-align:center; position:relative; overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#a855f7,#ec4899);"></div>
                <div style="width:6px;height:6px;border-radius:50%;background:#a855f7;box-shadow:0 0 8px #a855f7;margin:0 auto 6px;"></div>
                <div style="font-family:'Syne',sans-serif;font-size:10px;font-weight:700;color:#c084fc;margin-bottom:4px;">Prediction Engine</div>
                <div style="font-size:9px;color:#fcd34d;font-weight:600;letter-spacing:0.04em;">⚡ Realtime</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown("""
            <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.3); border-radius:10px; padding:12px 6px; text-align:center; position:relative; overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#10b981,#06b6d4);"></div>
                <div style="width:6px;height:6px;border-radius:50%;background:#10b981;box-shadow:0 0 8px #10b981;margin:0 auto 6px;"></div>
                <div style="font-family:'Syne',sans-serif;font-size:10px;font-weight:700;color:#6ee7b7;margin-bottom:4px;">Retention Insights</div>
                <div style="font-size:9px;color:#a5b4fc;font-weight:600;letter-spacing:0.04em;">✦ Enabled</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:24px 0 8px; font-size:12px; color:#374151; letter-spacing:0.04em; border-top:1px solid rgba(255,255,255,0.05); margin-top:24px;">
        Built with <span style="color:#6366f1;">Streamlit</span> · Random Forest · Telco Customer Churn Dataset · 7,043 customers
    </div>
    """, unsafe_allow_html=True)


def render_selected_page(page_path):
    if page_path == "app.py":
        render_home()
        return

    st.session_state["_router_rendering_page"] = True
    try:
        runpy.run_path(str(BASE_DIR / page_path), run_name=f"__churnlens_{page_path.replace('/', '_')}")
    finally:
        st.session_state["_router_rendering_page"] = False


render_selected_page(selected_page)
