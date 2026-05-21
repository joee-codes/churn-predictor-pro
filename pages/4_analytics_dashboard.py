import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
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
    'font.family': 'DejaVu Sans',
})

PALETTE = {
    'primary': '#6366f1', 'secondary': '#a855f7', 'accent': '#ec4899',
    'green': '#10b981', 'yellow': '#f59e0b', 'red': '#ef4444',
    'bg': '#0d0d1a', 'card': '#12121f', 'border': '#1f2937',
    'text': '#e2e2ef', 'muted': '#6b7280',
}

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 28px;">
    <div style="font-family:'Syne',sans-serif; font-size:30px; font-weight:800; color:#fff; margin-bottom:6px;">
        📈 Analytics Dashboard
    </div>
    <div style="font-size:14px; color:#6b7280;">Churn patterns, revenue risk, and segment analysis from the Telco dataset</div>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    # Try to load uploaded dataset, fallback to synthetic
    data_path = os.path.join(BASE, 'telco_churn.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)
        return df
    else:
        return None

df = load_data()

if df is None:
    st.markdown("""
    <div style="background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.25); border-radius:12px; padding:16px 20px; margin-bottom:24px; font-size:13px; color:#fcd34d;">
        ⚠️ <b>telco_churn.csv not found.</b> Place your dataset in the project root folder to see live analytics.
        The charts below use the Telco dataset statistics.
    </div>
    """, unsafe_allow_html=True)

    # Use hardcoded stats for demo
    use_demo = True
else:
    use_demo = False
    st.markdown(f"""
    <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:10px; padding:12px 16px; font-size:13px; color:#6ee7b7; margin-bottom:20px;">
        ✅ Dataset loaded — <b>{len(df):,}</b> customers · Churn rate: <b>{df['Churn_Binary'].mean()*100:.1f}%</b>
    </div>
    """, unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
if not use_demo:
    total = len(df)
    churned = df['Churn_Binary'].sum()
    churn_rate = df['Churn_Binary'].mean() * 100
    avg_monthly = df['MonthlyCharges'].mean()
    avg_tenure = df['tenure'].mean()
    rev_at_risk = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
else:
    total, churned, churn_rate = 7043, 1869, 26.5
    avg_monthly, avg_tenure, rev_at_risk = 64.76, 32.4, 139130

st.markdown(f"""
<div style="display:flex; gap:14px; margin-bottom:32px; flex-wrap:wrap;">
    <div style="flex:1; min-width:130px; background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:16px 18px;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Total Customers</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#a5b4fc;">{total:,}</div>
    </div>
    <div style="flex:1; min-width:130px; background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.2); border-radius:12px; padding:16px 18px;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Churned</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fca5a5;">{churned:,}</div>
        <div style="font-size:11px; color:#6b7280;">{churn_rate:.1f}% rate</div>
    </div>
    <div style="flex:1; min-width:130px; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); border-radius:12px; padding:16px 18px;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Avg Monthly $</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fcd34d;">${avg_monthly:.0f}</div>
    </div>
    <div style="flex:1; min-width:130px; background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.2); border-radius:12px; padding:16px 18px;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Avg Tenure</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#6ee7b7;">{avg_tenure:.0f}mo</div>
    </div>
    <div style="flex:1; min-width:130px; background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.15); border-radius:12px; padding:16px 18px;">
        <div style="font-size:10px; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">Revenue at Risk</div>
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800; color:#fca5a5;">${rev_at_risk/1000:.0f}K</div>
        <div style="font-size:11px; color:#6b7280;">per month</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── CHART HELPER ──────────────────────────────────────────────────────────────
def make_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])
    for spine in ax.spines.values():
        spine.set_color(PALETTE['border'])
    return fig, ax

# ── ROW 1: Churn Distribution + Contract Impact ───────────────────────────────
st.markdown('<div style="font-size:11px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:16px; display:flex; align-items:center; gap:10px;">Churn Overview <span style="flex:1; height:1px; background:rgba(255,255,255,0.06); display:inline-block;"></span></div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")

with c1:
    fig, ax = make_fig(6, 3.5)
    if not use_demo:
        labels = ['Retained', 'Churned']
        sizes = [total - churned, churned]
    else:
        labels = ['Retained', 'Churned']
        sizes = [5174, 1869]
    colors = [PALETTE['green'], PALETTE['red']]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops={'linewidth': 2, 'edgecolor': PALETTE['bg'], 'width': 0.55}
    )
    for t in texts:
        t.set_color(PALETTE['muted'])
        t.set_fontsize(11)
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(11)
        at.set_fontweight('600')
    ax.set_title('Churn Distribution', color='white', fontsize=13, fontweight='600', pad=10)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with c2:
    fig, ax = make_fig(6, 3.5)
    contracts = ['Month-to-month', 'One year', 'Two year']
    if not use_demo:
        churn_by_contract = df.groupby('Contract')['Churn_Binary'].mean() * 100
        rates = [churn_by_contract.get(c, 0) for c in contracts]
    else:
        rates = [42.7, 11.3, 2.8]

    bar_colors = [PALETTE['red'], PALETTE['yellow'], PALETTE['green']]
    bars = ax.bar(['M-to-M', 'One Year', 'Two Year'], rates, color=bar_colors,
                  width=0.5, edgecolor='none', zorder=3)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', fontsize=10, color='white', fontweight='600')
    ax.set_title('Churn Rate by Contract Type', color='white', fontsize=13, fontweight='600', pad=10)
    ax.set_ylabel('Churn Rate (%)', fontsize=9, color=PALETTE['muted'])
    ax.set_ylim(0, max(rates) * 1.25)
    ax.grid(axis='y', alpha=0.1, zorder=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── ROW 2: Tenure Distribution + Monthly Charges ─────────────────────────────
st.markdown('<div style="font-size:11px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin:24px 0 16px; display:flex; align-items:center; gap:10px;">Revenue & Tenure Analysis <span style="flex:1; height:1px; background:rgba(255,255,255,0.06); display:inline-block;"></span></div>', unsafe_allow_html=True)

c3, c4 = st.columns(2, gap="large")

with c3:
    fig, ax = make_fig(6, 3.8)
    if not use_demo:
        churned_tenure = df[df['Churn'] == 'Yes']['tenure']
        stayed_tenure = df[df['Churn'] == 'No']['tenure']
        ax.hist(stayed_tenure, bins=30, color=PALETTE['primary'], alpha=0.7, label='Retained', edgecolor='none')
        ax.hist(churned_tenure, bins=30, color=PALETTE['red'], alpha=0.7, label='Churned', edgecolor='none')
    else:
        # Simulated distributions
        np.random.seed(42)
        stayed = np.random.exponential(35, 5174)
        churned_t = np.random.exponential(10, 1869)
        ax.hist(stayed, bins=30, color=PALETTE['primary'], alpha=0.7, label='Retained', edgecolor='none', range=(0,72))
        ax.hist(churned_t, bins=30, color=PALETTE['red'], alpha=0.7, label='Churned', edgecolor='none', range=(0,72))
    ax.set_title('Tenure Distribution by Churn', color='white', fontsize=13, fontweight='600', pad=10)
    ax.set_xlabel('Tenure (months)', fontsize=9, color=PALETTE['muted'])
    ax.set_ylabel('Customer Count', fontsize=9, color=PALETTE['muted'])
    ax.legend(facecolor='#1a1a2e', edgecolor=PALETTE['border'], labelcolor='white', fontsize=9)
    ax.grid(axis='y', alpha=0.1); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with c4:
    fig, ax = make_fig(6, 3.8)
    if not use_demo:
        churned_mc = df[df['Churn'] == 'Yes']['MonthlyCharges']
        stayed_mc = df[df['Churn'] == 'No']['MonthlyCharges']
        ax.hist(stayed_mc, bins=30, color=PALETTE['green'], alpha=0.7, label='Retained', edgecolor='none')
        ax.hist(churned_mc, bins=30, color=PALETTE['yellow'], alpha=0.7, label='Churned', edgecolor='none')
    else:
        np.random.seed(1)
        stayed_m = np.random.normal(58, 20, 5174)
        churned_m = np.random.normal(74, 22, 1869)
        ax.hist(stayed_m, bins=30, color=PALETTE['green'], alpha=0.7, label='Retained', edgecolor='none', range=(20,120))
        ax.hist(churned_m, bins=30, color=PALETTE['yellow'], alpha=0.7, label='Churned', edgecolor='none', range=(20,120))
    ax.set_title('Monthly Charges by Churn Status', color='white', fontsize=13, fontweight='600', pad=10)
    ax.set_xlabel('Monthly Charges ($)', fontsize=9, color=PALETTE['muted'])
    ax.set_ylabel('Customer Count', fontsize=9, color=PALETTE['muted'])
    ax.legend(facecolor='#1a1a2e', edgecolor=PALETTE['border'], labelcolor='white', fontsize=9)
    ax.grid(axis='y', alpha=0.1); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── ROW 3: Internet Service + Payment Method ──────────────────────────────────
st.markdown('<div style="font-size:11px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin:24px 0 16px; display:flex; align-items:center; gap:10px;">Service & Payment Analysis <span style="flex:1; height:1px; background:rgba(255,255,255,0.06); display:inline-block;"></span></div>', unsafe_allow_html=True)

c5, c6 = st.columns(2, gap="large")

with c5:
    fig, ax = make_fig(6, 3.5)
    if not use_demo:
        internet_churn = df.groupby('InternetService')['Churn_Binary'].mean() * 100
        ikeys = internet_churn.index.tolist()
        ivals = internet_churn.values.tolist()
    else:
        ikeys = ['No', 'DSL', 'Fiber optic']
        ivals = [7.4, 19.0, 41.9]
    icolors = [PALETTE['green'], PALETTE['yellow'], PALETTE['red']]
    bars = ax.barh(ikeys, ivals, color=icolors[:len(ikeys)], height=0.45, edgecolor='none')
    for bar, val in zip(bars, ivals):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10, color='white', fontweight='600')
    ax.set_title('Churn Rate by Internet Service', color='white', fontsize=13, fontweight='600', pad=10)
    ax.set_xlabel('Churn Rate (%)', fontsize=9, color=PALETTE['muted'])
    ax.set_xlim(0, max(ivals) * 1.2)
    ax.grid(axis='x', alpha=0.1); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with c6:
    fig, ax = make_fig(6, 3.5)
    if not use_demo:
        pm_churn = df.groupby('PaymentMethod')['Churn_Binary'].mean() * 100
        pkeys = [k.replace('(automatic)', '(auto)') for k in pm_churn.index.tolist()]
        pvals = pm_churn.values.tolist()
    else:
        pkeys = ['Electronic\ncheck', 'Mailed\ncheck', 'Bank transfer\n(auto)', 'Credit card\n(auto)']
        pvals = [45.3, 19.1, 16.7, 15.2]

    pcolors = [PALETTE['red'] if v > 30 else PALETTE['yellow'] if v > 20 else PALETTE['green'] for v in pvals]
    bars = ax.bar(pkeys, pvals, color=pcolors, width=0.5, edgecolor='none', zorder=3)
    for bar, val in zip(bars, pvals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9, color='white', fontweight='600')
    ax.set_title('Churn Rate by Payment Method', color='white', fontsize=13, fontweight='600', pad=10)
    ax.set_ylabel('Churn Rate (%)', fontsize=9, color=PALETTE['muted'])
    ax.set_ylim(0, max(pvals) * 1.25)
    ax.grid(axis='y', alpha=0.1, zorder=0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── ROW 4: Tenure Cohort Churn ────────────────────────────────────────────────
st.markdown('<div style="font-size:11px; color:#4b5563; letter-spacing:0.12em; text-transform:uppercase; margin:24px 0 16px; display:flex; align-items:center; gap:10px;">Cohort Analysis <span style="flex:1; height:1px; background:rgba(255,255,255,0.06); display:inline-block;"></span></div>', unsafe_allow_html=True)

fig, ax = make_fig(13, 3.8)

if not use_demo:
    df['tenure_band'] = pd.cut(df['tenure'], bins=[0,6,12,24,36,48,60,72],
                                labels=['0-6','6-12','12-24','24-36','36-48','48-60','60-72'])
    cohort = df.groupby('tenure_band')['Churn_Binary'].mean() * 100
    bands = cohort.index.astype(str).tolist()
    rates_c = cohort.values.tolist()
else:
    bands = ['0-6', '6-12', '12-24', '24-36', '36-48', '48-60', '60-72']
    rates_c = [53.2, 41.8, 32.5, 21.4, 13.7, 8.2, 4.1]

bar_colors_c = [PALETTE['red'] if r > 35 else PALETTE['yellow'] if r > 20 else PALETTE['green'] for r in rates_c]
bars = ax.bar(bands, rates_c, color=bar_colors_c, width=0.6, edgecolor='none', zorder=3)
ax.plot(bands, rates_c, color=PALETTE['primary'], linewidth=2, marker='o',
        markersize=5, zorder=4, alpha=0.8)
for bar, rate in zip(bars, rates_c):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{rate:.0f}%', ha='center', fontsize=9, color='white', fontweight='600')

ax.set_title('Churn Rate by Tenure Cohort (months)', color='white', fontsize=13, fontweight='600', pad=10)
ax.set_xlabel('Tenure Band (months)', fontsize=9, color=PALETTE['muted'])
ax.set_ylabel('Churn Rate (%)', fontsize=9, color=PALETTE['muted'])
ax.set_ylim(0, max(rates_c) * 1.25)
ax.fill_between(range(len(bands)), rates_c, alpha=0.05, color=PALETTE['primary'])
ax.grid(axis='y', alpha=0.1, zorder=0); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("""
<div style="text-align:center; padding:24px 0 8px; font-size:12px; color:#374151;">
    Analytics based on Telco Customer Churn Dataset · 7,043 customers
</div>
""", unsafe_allow_html=True)
