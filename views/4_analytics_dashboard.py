import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import get_all_predictions
from theme import apply_theme, render_sidebar

apply_theme()
render_sidebar()

plt.rcParams.update({
    'figure.facecolor': '#0d0d1a',
    'axes.facecolor': '#0d0d1a',
    'axes.edgecolor': '#1f2937',
    'axes.labelcolor': '#6b7280',
    'xtick.color': '#6b7280',
    'ytick.color': '#6b7280',
    'text.color': '#e2e2ef',
    'grid.color': '#1f2937',
    'font.family': 'DejaVu Sans',
})

PALETTE = {
    'primary': '#6366f1',
    'secondary': '#a855f7',
    'green': '#10b981',
    'yellow': '#f59e0b',
    'red': '#ef4444',
    'bg': '#0d0d1a',
    'border': '#1f2937',
    'muted': '#6b7280',
}


def make_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])
    for spine in ax.spines.values():
        spine.set_color(PALETTE['border'])
    return fig, ax


def normalize_database_history():
    db_df = get_all_predictions()
    if db_df.empty:
        return pd.DataFrame()

    return pd.DataFrame({
        'source': 'Historical',
        'tenure': pd.to_numeric(db_df.get('tenure'), errors='coerce'),
        'monthly_charges': pd.to_numeric(db_df.get('monthly_charges'), errors='coerce'),
        'total_charges': pd.to_numeric(db_df.get('total_charges'), errors='coerce'),
        'contract': db_df.get('contract'),
        'internet_service': db_df.get('internet_service'),
        'payment_method': db_df.get('payment_method'),
        'churn_probability': pd.to_numeric(db_df.get('churn_probability'), errors='coerce'),
        'risk_category': db_df.get('risk_category'),
    })


def normalize_active_single_prediction():
    df = st.session_state.get('active_single_prediction')
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.copy()


def normalize_bulk_history():
    df = st.session_state.get('bulk_prediction_history')
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.copy()


def clean_history_frame(df):
    if df.empty:
        return df
    df = df.copy()
    df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce').fillna(0)
    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0)
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
    df['churn_probability'] = pd.to_numeric(df['churn_probability'], errors='coerce').fillna(0)
    df['risk_category'] = df['risk_category'].fillna('Low')
    df['churn_binary'] = (df['risk_category'] == 'High').astype(int)
    return df


def combine_history_frames(frames):
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return clean_history_frame(df)


st.markdown("""
<div style="padding:32px 0 24px;">
    <div style="font-family:'Syne',sans-serif; font-size:30px; font-weight:800; color:#fff; margin-bottom:6px;">
        📈 Analytics Dashboard
    </div>
    <div style="font-size:14px; color:#6b7280;">Live churn analytics from prediction history</div>
</div>
""", unsafe_allow_html=True)

historical_df = clean_history_frame(normalize_database_history())
single_df = clean_history_frame(normalize_active_single_prediction())
bulk_df = clean_history_frame(normalize_bulk_history())
base_df = combine_history_frames([historical_df, bulk_df])
if base_df.empty and not single_df.empty:
    base_df = single_df.copy()

data_source = st.radio(
    "Select Data Source",
    ["All Historical Data", "Single Predictions History", "Bulk Predictions History"],
    horizontal=True,
    index=0,
    key="analytics_data_source",
)

if data_source == "Single Predictions History":
    filtered_df = single_df
elif data_source == "Bulk Predictions History":
    filtered_df = bulk_df
else:
    filtered_df = base_df

st.caption(
    f"Using {len(filtered_df):,} records "
    f"(Historical: {len(historical_df):,}, Single active: {len(single_df):,}, Bulk: {len(bulk_df):,})"
)

if filtered_df.empty:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.08); border-radius:16px; margin-top:24px;">
        <div style="font-size:36px; margin-bottom:12px;">📊</div>
        <div style="font-family:'Syne',sans-serif; font-size:16px; font-weight:600; color:#9ca3af; margin-bottom:6px;">No analytics data yet</div>
        <div style="font-size:13px; color:#6b7280;">Run Single Predictions or upload a Bulk Prediction CSV, then return here.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

total = len(filtered_df)
churned = int(filtered_df['churn_binary'].sum())
retained = total - churned
churn_rate = filtered_df['churn_binary'].mean() * 100 if total else 0
avg_monthly = filtered_df['monthly_charges'].mean()
avg_tenure = filtered_df['tenure'].mean()
rev_at_risk = filtered_df.loc[filtered_df['churn_binary'] == 1, 'monthly_charges'].sum()

st.markdown(f"""
<div style="display:flex; gap:14px; margin:24px 0 32px; flex-wrap:wrap;">
    <div style="flex:1; min-width:130px; background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:16px 18px;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Total Customers</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#a5b4fc;">{total:,}</div>
    </div>
    <div style="flex:1; min-width:130px; background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:12px; padding:16px 18px;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Churned</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fca5a5;">{churned:,}</div>
        <div style="font-size:11px; color:#6b7280;">{churn_rate:.1f}% high-risk rate</div>
    </div>
    <div style="flex:1; min-width:130px; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); border-radius:12px; padding:16px 18px;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Avg Monthly ₹</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fcd34d;">₹{avg_monthly:,.0f}</div>
    </div>
    <div style="flex:1; min-width:130px; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:12px; padding:16px 18px;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Avg Tenure</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#6ee7b7;">{avg_tenure:.0f}mo</div>
    </div>
    <div style="flex:1; min-width:130px; background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.15); border-radius:12px; padding:16px 18px;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Revenue at Risk</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fca5a5;">₹{rev_at_risk:,.0f}</div>
        <div style="font-size:11px; color:#6b7280;">per month</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="font-size:11px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:16px; display:flex; align-items:center; gap:10px;">Churn Overview <span style="flex:1; height:1px; background:rgba(255,255,255,0.06); display:inline-block;"></span></div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")

with c1:
    fig, ax = make_fig(6, 3.5)
    labels = ['Retained / Lower Risk', 'High Risk']
    sizes = [retained, churned]
    colors = [PALETTE['green'], PALETTE['red']]
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        wedgeprops={'linewidth': 2, 'edgecolor': PALETTE['bg'], 'width': 0.55},
    )
    ax.set_title('Churn Distribution', color='white', fontsize=13, fontweight='600', pad=10)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with c2:
    fig, ax = make_fig(6, 3.5)
    contract_order = ['Month-to-month', 'One year', 'Two year']
    contract_rates = (filtered_df.groupby('contract')['churn_binary'].mean() * 100).reindex(contract_order).fillna(0)
    bars = ax.bar(
        ['M-to-M', 'One Year', 'Two Year'],
        contract_rates.values,
        color=[PALETTE['red'], PALETTE['yellow'], PALETTE['green']],
        width=0.5,
        edgecolor='none',
        zorder=3,
    )
    for bar, rate in zip(bars, contract_rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', fontsize=10, color='white', fontweight='600')
    ax.set_title('Churn Rate by Contract Type', color='white', fontsize=13, fontweight='600', pad=10)
    ax.set_ylabel('High-Risk Rate (%)', fontsize=9, color=PALETTE['muted'])
    ax.set_ylim(0, max(10, contract_rates.max() * 1.25))
    ax.grid(axis='y', alpha=0.1, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown('<div style="font-size:11px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin:24px 0 16px; display:flex; align-items:center; gap:10px;">Prediction Risk Analysis <span style="flex:1; height:1px; background:rgba(255,255,255,0.06); display:inline-block;"></span></div>', unsafe_allow_html=True)

c3, c4 = st.columns(2, gap="large")

with c3:
    fig, ax = make_fig(6, 3.5)
    risk_counts = filtered_df['risk_category'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
    bars = ax.bar(risk_counts.index, risk_counts.values, color=[PALETTE['red'], PALETTE['yellow'], PALETTE['green']], width=0.5)
    for bar, val in zip(bars, risk_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                f'{int(val):,}', ha='center', fontsize=10, color='white', fontweight='600')
    ax.set_title('Risk Category Distribution', color='white', fontsize=13, fontweight='600', pad=10)
    ax.set_ylabel('Customers', fontsize=9, color=PALETTE['muted'])
    ax.grid(axis='y', alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with c4:
    fig, ax = make_fig(6, 3.5)
    ax.hist(filtered_df['churn_probability'], bins=20, color=PALETTE['primary'], alpha=0.8, edgecolor='none')
    ax.axvline(60, color=PALETTE['red'], linestyle='--', linewidth=1)
    ax.axvline(30, color=PALETTE['yellow'], linestyle='--', linewidth=1)
    ax.set_title('Churn Probability Distribution', color='white', fontsize=13, fontweight='600', pad=10)
    ax.set_xlabel('Churn Probability (%)', fontsize=9, color=PALETTE['muted'])
    ax.set_ylabel('Customers', fontsize=9, color=PALETTE['muted'])
    ax.grid(axis='y', alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

st.markdown(f"""
<div style="text-align:center; padding:24px 0 8px; font-size:12px; color:#374151;">
    Analytics based on {data_source.lower()} · {total:,} prediction records
</div>
""", unsafe_allow_html=True)
