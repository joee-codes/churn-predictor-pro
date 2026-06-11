import streamlit as st
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theme import apply_theme, render_sidebar
from database import get_kpis

apply_theme()
render_sidebar()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 20px;">
    <div style="font-family:'Syne',sans-serif; font-size:30px; font-weight:800; color:#fff; margin-bottom:6px;">
        💬 AI Assistant
    </div>
    <div style="font-size:14px; color:#6b7280;">Ask anything about churn, retention strategy, or your prediction data</div>
</div>
""", unsafe_allow_html=True)

kpis = get_kpis()

# ── GROQ API KEY ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Dark chat input */
[data-testid="stChatInput"] { background: #0d0d1a !important; border-top: 1px solid rgba(255,255,255,0.06) !important; }
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"],
div[class*="stChatInput"] textarea {
    background: #12121f !important;
    color: #e2e2ef !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    caret-color: #6366f1 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #4b5563 !important; }
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    outline: none !important;
}
/* Override any white backgrounds on chat container */
section[data-testid="stBottom"] { background: #07070f !important; }
section[data-testid="stBottom"] > div { background: #07070f !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border-radius: 8px !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── API KEY — hidden after entry ──────────────────────────────────────────────
if 'groq_key' not in st.session_state:
    st.session_state['groq_key'] = ''

if not st.session_state['groq_key']:
    st.markdown("""
    <div style="background:rgba(99,102,241,0.06); border:1px solid rgba(99,102,241,0.2); border-radius:12px; padding:16px 20px; margin-bottom:16px;">
        <div style="font-size:12px; color:#6366f1; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">🔑 Activate AI Assistant</div>
        <div style="font-size:12px; color:#6b7280; margin-bottom:12px;">Get your free Groq API key at <b style="color:#a5b4fc;">console.groq.com</b> — no credit card needed</div>
    </div>
    """, unsafe_allow_html=True)
    key_input = st.text_input("Groq API Key", type="password", placeholder="gsk_...", label_visibility="collapsed")
    if key_input:
        st.session_state['groq_key'] = key_input
        st.rerun()
else:
    # Small indicator in top right that key is active
    st.markdown("""
    <div style="display:flex; justify-content:flex-end; margin-bottom:8px;">
        <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25); border-radius:20px; padding:4px 12px; font-size:11px; color:#6ee7b7; display:flex; align-items:center; gap:6px;">
            <div style="width:5px;height:5px;background:#10b981;border-radius:50%;box-shadow:0 0 6px #10b981;"></div>
            Groq AI Active
        </div>
    </div>
    """, unsafe_allow_html=True)

groq_key = st.session_state.get('groq_key', '')
st.markdown("<hr>", unsafe_allow_html=True)

# ── CONTEXT PILLS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px;">
    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:8px; padding:8px 14px; font-size:12px; color:#9ca3af;">
        📊 <b style="color:#fff;">{kpis['total']}</b> predictions tracked
    </div>
    <div style="background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.15); border-radius:8px; padding:8px 14px; font-size:12px; color:#fca5a5;">
        🚨 <b>{kpis['high_risk']}</b> high risk customers
    </div>
    <div style="background:rgba(239,68,68,0.05); border:1px solid rgba(239,68,68,0.1); border-radius:8px; padding:8px 14px; font-size:12px; color:#fca5a5;">
        💸 <b>₹{kpis['revenue_at_risk']:,.0f}</b> revenue at risk
    </div>
</div>
""", unsafe_allow_html=True)

# ── SUGGESTED PROMPTS ─────────────────────────────────────────────────────────
st.markdown('<div style="font-size:11px; color:#4b5563; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:10px;">Suggested Questions</div>', unsafe_allow_html=True)

suggestions = [
    "Why do month-to-month customers churn more?",
    "What's the best retention strategy for high-risk customers?",
    "How can I reduce revenue at risk?",
    "What features matter most for churn prediction?",
    "How do I interpret the churn probability score?",
    "How does payment method affect churn?",
]

cols = st.columns(3)
for i, s in enumerate(suggestions):
    with cols[i % 3]:
        if st.button(s, key=f"sug_{i}", use_container_width=True):
            st.session_state['prefill'] = s

st.markdown("<br>", unsafe_allow_html=True)

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are ChurnLens AI, an expert customer retention analyst embedded in the ChurnLens platform — an AI-powered churn prediction system built with Random Forest ML.

Platform context:
- Total predictions made: {kpis['total']}
- High risk customers: {kpis['high_risk']}
- Medium risk customers: {kpis['medium_risk']}
- Low risk customers: {kpis['low_risk']}
- Revenue at risk per month: ₹{kpis['revenue_at_risk']:,.0f}
- Average churn probability: {kpis['avg_probability']:.1f}%

Model details:
- Algorithm: Random Forest Classifier (200 trees)
- ROC-AUC: 83%
- Features: tenure, MonthlyCharges, TotalCharges, Contract, InternetService, OnlineSecurity, TechSupport, PaymentMethod, PaperlessBilling, Partner, SeniorCitizen
- Dataset: Telco Customer Churn (7,043 customers, 26.5% churn rate)

Key churn statistics:
- Month-to-month: 42.7% churn | One year: 11.3% | Two year: 2.8%
- Fiber optic: 41.9% churn | DSL: 19% | No internet: 7.4%
- Electronic check: 45.3% churn | Credit card auto: 15.2%
- Tenure 0-6 months: 53% churn | 36+ months: <15%

Be concise, business-focused, and actionable. Use bullet points. Keep responses under 250 words unless complexity requires more."""

# ── CHAT INTERFACE ────────────────────────────────────────────────────────────
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display history
for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; margin-bottom:12px;">
            <div style="background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.25); border-radius:14px 14px 2px 14px; padding:12px 16px; max-width:75%; font-size:13px; color:#e2e2ef; line-height:1.5;">
                {msg['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        formatted = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', msg['content'].replace('\n', '<br>'))
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-start; margin-bottom:12px;">
            <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:14px 14px 14px 2px; padding:14px 18px; max-width:82%; font-size:13px; color:#e2e2ef; line-height:1.7;">
                <span style="font-size:10px; color:#6366f1; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; display:block; margin-bottom:8px;">💬 ChurnLens AI</span>
                {formatted}
            </div>
        </div>
        """, unsafe_allow_html=True)

# Input
prefill = st.session_state.pop('prefill', '')
user_input = st.chat_input("Ask about churn, retention strategy, or your data...")

if user_input or prefill:
    question = user_input or prefill
    st.session_state['messages'].append({'role': 'user', 'content': question})

    if not groq_key:
        st.session_state['messages'].append({
            'role': 'assistant',
            'content': "Please enter your **Groq API key** above to get AI-powered responses. Get it free at console.groq.com — takes 2 minutes!"
        })
    else:
        with st.spinner("Thinking..."):
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)

                messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}]
                for m in st.session_state['messages']:
                    messages_payload.append({"role": m["role"], "content": m["content"]})

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages_payload,
                    max_tokens=500,
                    temperature=0.7,
                )
                reply = response.choices[0].message.content
                if not reply or reply.strip() == "":
                    reply = "Sorry, I couldn't generate a response. Please try again."
                st.session_state['messages'].append({'role': 'assistant', 'content': reply})

            except ImportError:
                st.session_state['messages'].append({'role': 'assistant', 'content': "Run `pip install groq` then restart the app."})
            except Exception as e:
                st.session_state['messages'].append({'role': 'assistant', 'content': f"Error: {str(e)}"})

    st.rerun()

if st.session_state['messages']:
    if st.button("🗑️ Clear Chat"):
        st.session_state['messages'] = []
        st.rerun()
