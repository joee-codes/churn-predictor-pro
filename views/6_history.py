import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theme import apply_theme, render_sidebar
from database import get_all_predictions, get_kpis, clear_history

apply_theme()
render_sidebar()

plt.rcParams.update({
    'figure.facecolor': '#0d0d1a', 'axes.facecolor': '#0d0d1a',
    'axes.edgecolor': '#1f2937', 'axes.labelcolor': '#6b7280',
    'xtick.color': '#6b7280', 'ytick.color': '#6b7280',
    'text.color': '#e2e2ef', 'grid.color': '#1f2937',
})

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 28px;">
    <div style="font-family:'Syne',sans-serif; font-size:30px; font-weight:800; color:#fff; margin-bottom:6px;">
        🕓 Prediction History
    </div>
    <div style="font-size:14px; color:#6b7280;">Track all predictions made, monitor trends, and export history</div>
</div>
""", unsafe_allow_html=True)

# ── LIVE KPIs ─────────────────────────────────────────────────────────────────
kpis = get_kpis()

st.markdown(f"""
<div style="display:flex; gap:14px; margin-bottom:28px; flex-wrap:wrap;">
    <div style="flex:1; min-width:120px; background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:16px 18px; text-align:center;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Total Predictions</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#a5b4fc;">{kpis['total']}</div>
    </div>
    <div style="flex:1; min-width:120px; background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:12px; padding:16px 18px; text-align:center;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">High Risk</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fca5a5;">{kpis['high_risk']}</div>
    </div>
    <div style="flex:1; min-width:120px; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); border-radius:12px; padding:16px 18px; text-align:center;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Medium Risk</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fcd34d;">{kpis['medium_risk']}</div>
    </div>
    <div style="flex:1; min-width:120px; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:12px; padding:16px 18px; text-align:center;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Low Risk</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#6ee7b7;">{kpis['low_risk']}</div>
    </div>
    <div style="flex:1; min-width:120px; background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.15); border-radius:12px; padding:16px 18px; text-align:center;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Revenue at Risk</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fca5a5;">₹{kpis['revenue_at_risk']:,.0f}</div>
    </div>
    <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px 18px; text-align:center;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Avg Churn Risk</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fff;">{kpis['avg_probability']:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── HISTORY TABLE ─────────────────────────────────────────────────────────────
df = get_all_predictions()

if df.empty:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.08); border-radius:16px;">
        <div style="font-size:36px; margin-bottom:12px;">🕓</div>
        <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:600; color:#6b7280; margin-bottom:6px;">No predictions yet</div>
        <div style="font-size:13px; color:#374151;">Run a prediction on the Single Prediction page to see history here</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── CHARTS ────────────────────────────────────────────────────────────────
    if len(df) >= 3:
        c1, c2 = st.columns(2, gap="large")

        with c1:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            fig.patch.set_facecolor('#0d0d1a')
            ax.set_facecolor('#0d0d1a')

            risk_counts = df['risk_category'].value_counts()
            colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
            bar_colors = [colors.get(r, '#6366f1') for r in risk_counts.index]
            ax.bar(risk_counts.index, risk_counts.values, color=bar_colors, width=0.5, edgecolor='none')
            for i, (cat, val) in enumerate(zip(risk_counts.index, risk_counts.values)):
                ax.text(i, val + 0.1, str(val), ha='center', fontsize=10, color='white', fontweight='600')
            ax.set_title('Risk Distribution', color='white', fontsize=12, fontweight='600', pad=10)
            ax.set_ylabel('Count', fontsize=9, color='#6b7280')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#1f2937'); ax.spines['bottom'].set_color('#1f2937')
            ax.grid(axis='y', alpha=0.1)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            fig.patch.set_facecolor('#0d0d1a')
            ax.set_facecolor('#0d0d1a')

            ax.plot(range(len(df)), df['churn_probability'].values[::-1],
                    color='#6366f1', linewidth=2, marker='o', markersize=4)
            ax.fill_between(range(len(df)), df['churn_probability'].values[::-1],
                            alpha=0.1, color='#6366f1')
            ax.axhline(y=60, color='#ef4444', linestyle='--', alpha=0.4, linewidth=1)
            ax.axhline(y=30, color='#f59e0b', linestyle='--', alpha=0.4, linewidth=1)
            ax.set_title('Churn Probability Trend', color='white', fontsize=12, fontweight='600', pad=10)
            ax.set_ylabel('Churn Probability (%)', fontsize=9, color='#6b7280')
            ax.set_xlabel('Prediction #', fontsize=9, color='#6b7280')
            ax.set_ylim(0, 110)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#1f2937'); ax.spines['bottom'].set_color('#1f2937')
            ax.grid(alpha=0.1)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("<br>", unsafe_allow_html=True)

    # ── TABLE ─────────────────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-size:16px; font-weight:700; color:#fff; margin-bottom:16px;">All Predictions</div>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* Prediction History has one selectbox. Keep it as a chevron dropdown while
       preventing the hidden search input from behaving like an editable field. */
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
        cursor: pointer !important;
        user-select: none !important;
    }
    div[data-testid="stSelectbox"] input,
    div[data-testid="stSelectbox"] input:hover,
    div[data-testid="stSelectbox"] input:focus {
        caret-color: transparent !important;
        cursor: pointer !important;
        pointer-events: none !important;
        user-select: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

    risk_filter = st.selectbox("Filter by Risk", ["All", "High", "Medium", "Low"])

    display_df = df if risk_filter == "All" else df[df['risk_category'] == risk_filter]

    cols_show = ['timestamp', 'tenure', 'monthly_charges', 'contract',
                 'churn_probability', 'risk_category', 'recommendation']
    cols_show = [c for c in cols_show if c in display_df.columns]

    def style_risk(val):
        colors = {'High': 'color: #fca5a5', 'Medium': 'color: #fcd34d', 'Low': 'color: #6ee7b7'}
        return colors.get(val, '')

    st.dataframe(
        display_df[cols_show].style.map(style_risk, subset=['risk_category']),
        use_container_width=True,
        height=400
    )

    # ── EXPORT + CLEAR ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_dl, col_cl, _ = st.columns([1, 1, 2])

    with col_dl:
        csv_export = df.to_csv(index=False)
        st.download_button(
            "⬇️  Export History CSV",
            data=csv_export,
            file_name="churnlens_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col_cl:
        if st.button("🗑️  Clear History", use_container_width=True):
            clear_history()
            st.success("History cleared!")
            st.rerun()
