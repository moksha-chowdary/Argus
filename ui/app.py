"""
ARGUS — Streamlit Chat Interface
Dark-themed conversational UI for chart analysis.
"""
import sys
import io
import time
from pathlib import Path

import numpy as np
import cv2
import streamlit as st

# Allow imports from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from argus_core import ArgusOrchestrator, ArgusResult


# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ARGUS — AI Visual Market Analyst",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Core dark theme */
[data-testid="stAppViewContainer"] {
    background: #0d0d0f;
    color: #e8e6e0;
}
[data-testid="stSidebar"] {
    background: #111115;
    border-right: 1px solid #222228;
}
[data-testid="stChatMessageContent"] {
    background: #16161a !important;
    border: 1px solid #222228;
    border-radius: 10px;
    padding: 12px 16px !important;
}

/* Signal badge colors */
.signal-buy   { background: #0d2b1a; color: #2ecc71; border: 1px solid #2ecc71; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 1.1rem; }
.signal-sell  { background: #2b0d0d; color: #e74c3c; border: 1px solid #e74c3c; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 1.1rem; }
.signal-wait  { background: #2b2506; color: #f39c12; border: 1px solid #f39c12; padding: 4px 12px; border-radius: 6px; font-weight: 700; font-size: 1.1rem; }

/* Summary card */
.summary-card {
    background: #16161a;
    border: 1px solid #222228;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}
.summary-row { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.9rem; }
.label { color: #888; }
.value { color: #e8e6e0; font-weight: 500; }
.value-bull { color: #2ecc71; font-weight: 500; }
.value-bear { color: #e74c3c; font-weight: 500; }
.value-neutral { color: #f39c12; font-weight: 500; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages": [],
        "last_result": None,
        "chart_analyzed": False,
        "argus": None,
        "model": "gemma3",
        "ollama_url": "http://localhost:11434/api",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def get_argus() -> ArgusOrchestrator:
    if (
        st.session_state.argus is None
        or st.session_state.argus.llm_client.model != st.session_state.model
    ):
        st.session_state.argus = ArgusOrchestrator(
            ollama_model=st.session_state.model,
            ollama_url=st.session_state.ollama_url,
        )
    return st.session_state.argus


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 👁️ ARGUS")
    st.markdown("*AI Visual Market Analyst*")
    st.divider()

    st.markdown("### ⚙️ Configuration")
    st.session_state.ollama_url = st.text_input(
        "Ollama URL", value=st.session_state.ollama_url, key="url_input"
    )
    st.session_state.model = st.selectbox(
        "Model",
        ["gemma3", "gemma3:4b", "gemma3:12b", "llama3", "mistral", "phi3"],
        index=0,
    )

    st.divider()

    # Chart upload
    st.markdown("### 📊 Upload Chart")
    uploaded = st.file_uploader(
        "Drop a TradingView screenshot",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        st.image(uploaded, use_container_width=True, caption="Uploaded chart")

    st.divider()

    if st.button("🗑️ New Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.chart_analyzed = False
        if st.session_state.argus:
            st.session_state.argus.reset()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>⚠️ Educational use only. Not financial advice.</small>",
        unsafe_allow_html=True,
    )


# ─── Main area ────────────────────────────────────────────────────────────────

col_chat, col_panel = st.columns([3, 1], gap="medium")

with col_chat:
    st.markdown("### 💬 Chart Analysis")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="👁️" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Chat input
    prompt = st.chat_input(
        "Ask about the chart… (upload a chart first, or ask about the last analysis)"
    )

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        argus = get_argus()

        with st.chat_message("assistant", avatar="👁️"):
            placeholder = st.empty()
            full_response = ""

            if uploaded and not st.session_state.chart_analyzed:
                # Fresh chart analysis
                argus.reset()
                file_bytes = uploaded.read()
                img_array = np.frombuffer(file_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                with st.spinner("Running vision pipeline…"):
                    summary, signal = argus.extract_only(img)
                    st.session_state.last_result = (summary, signal)
                    st.session_state.chart_analyzed = True

                # Stream LLM response
                token_gen = argus.reasoning.analyze_chart(
                    summary, signal, prompt, stream=True
                )
                for token in token_gen:
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            elif st.session_state.chart_analyzed:
                # Follow-up in existing session
                token_gen = argus.chat(prompt, stream=True)
                for token in token_gen:
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            else:
                full_response = (
                    "Please upload a chart screenshot on the left sidebar first, "
                    "then I can analyze it for you."
                )
                placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()


with col_panel:
    st.markdown("### 📈 Analysis Panel")

    result_data = st.session_state.last_result
    if result_data:
        summary, signal = result_data

        # Signal badge
        action = signal.action
        badge_class = {"BUY": "signal-buy", "SELL": "signal-sell", "WAIT": "signal-wait"}.get(action, "signal-wait")
        st.markdown(
            f'<div style="text-align:center;margin-bottom:16px;">'
            f'<span class="{badge_class}">{action}</span>'
            f'<div style="color:#888;font-size:0.85rem;margin-top:4px">'
            f'Conviction: {signal.conviction} | Risk: {signal.risk_rating}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Market summary card
        def color_class(val, pos_vals, neg_vals):
            if val in pos_vals:
                return "value-bull"
            if val in neg_vals:
                return "value-bear"
            return "value-neutral"

        def row(label, value, cls="value"):
            return f'<div class="summary-row"><span class="label">{label}</span><span class="{cls}">{value}</span></div>'

        rows = "".join([
            row("Ticker", summary.ticker or "—"),
            row("Timeframe", summary.timeframe or "—"),
            row("Trend", f"{summary.trend} ({summary.trend_strength})",
                color_class(summary.trend, ["bullish"], ["bearish"])),
            row("RSI", f"{summary.rsi:.1f}" if summary.rsi else "—",
                color_class(summary.rsi_state, ["oversold"], ["overbought"])),
            row("EMA", summary.ema_alignment,
                color_class(summary.ema_alignment, ["bullish"], ["bearish"])),
            row("Volume", summary.volume_behavior,
                color_class(summary.volume_behavior, ["increasing"], ["decreasing"])),
            row("Volatility", summary.volatility),
            row("Momentum", summary.momentum,
                color_class(summary.momentum, ["strong"], ["weakening", "weak"])),
            row("Pattern", summary.chart_pattern or "None"),
            row("Breakout P.", summary.breakout_probability,
                color_class(summary.breakout_probability, ["high"], ["low"])),
            row("Support", f"{summary.support:.2f}" if summary.support else "—"),
            row("Resistance", f"{summary.resistance:.2f}" if summary.resistance else "—"),
        ])

        st.markdown(
            f'<div class="summary-card">{rows}</div>',
            unsafe_allow_html=True,
        )

        # Trade zones
        if signal.entry_zone:
            st.markdown("**📍 Trade Zones**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Entry", signal.entry_zone)
                st.metric("Target", signal.target_zone or "—")
            with col2:
                st.metric("Stop Loss", signal.stop_loss_zone or "—")
                st.metric("R/R", f"{signal.risk_reward_ratio:.1f}x" if signal.risk_reward_ratio else "—")

        # Warnings
        if signal.warnings:
            with st.expander("⚠️ Warnings"):
                for w in signal.warnings:
                    st.markdown(f"• {w}")

        # Confidence
        conf = summary.confidence
        color = "#2ecc71" if conf > 0.6 else "#f39c12" if conf > 0.35 else "#e74c3c"
        st.markdown(
            f'<div style="margin-top:12px;font-size:0.85rem;color:#888">'
            f'Extraction confidence: <span style="color:{color};font-weight:600">{conf:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            '<div style="color:#555;text-align:center;padding:40px 0;font-size:0.9rem">'
            "Upload a chart and ask a question<br>to see the analysis panel."
            "</div>",
            unsafe_allow_html=True,
        )
