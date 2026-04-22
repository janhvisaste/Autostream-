"""
streamlit_app.py - AutoStream Conversational AI interface.

Polished Streamlit UI with:
- Responsive premium layout and branded visual language
- Guided quick-start prompts and clear lead funnel progress
- Existing LangGraph state contract preserved

Run: streamlit run streamlit_app.py
"""

import html as html_module
import io
import os
import re
import sys
import uuid
from contextlib import redirect_stdout

import streamlit as st
from langchain_core.messages import HumanMessage


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


st.set_page_config(
    page_title="AutoStream Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

:root {
    --bg-base: #0f111a;
    --ink-1: #ffffff;
    --ink-2: #d1d5db;
    --ink-muted: #9ca3af;
    --panel: rgba(15, 23, 42, 0.55);
    --panel-strong: rgba(15, 23, 42, 0.85);
    --stroke: rgba(255, 255, 255, 0.12);
    --accent-1: #3b82f6; 
    --accent-2: #8b5cf6;
    --accent-3: #f43f5e;
    --good: #10b981;
    --warn: #f59e0b;
    --info: #3b82f6;
    --shadow-lg: 0 20px 45px rgba(0, 0, 0, 0.4);
    --radius-lg: 18px;
    --radius-md: 12px;
}

html,
body,
.stApp {
    font-family: 'Manrope', sans-serif;
    background:
        radial-gradient(circle at 12% 18%, rgba(59, 130, 246, 0.15) 0, rgba(59, 130, 246, 0) 34%),
        radial-gradient(circle at 88% 10%, rgba(139, 92, 246, 0.12) 0, rgba(139, 92, 246, 0) 28%),
        radial-gradient(circle at 82% 86%, rgba(244, 63, 94, 0.12) 0, rgba(244, 63, 94, 0) 33%),
        linear-gradient(145deg, #090c15 0%, #111827 100%);
    color: var(--ink-2);
}

h1, h2, h3, h4 {
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: -0.01em;
    color: var(--ink-1);
}

#MainMenu,
footer,
[data-testid="stDecoration"],
.stDeployButton {
    display: none !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(175deg, rgba(15, 23, 42, 0.65), rgba(9, 13, 24, 0.76)) !important;
    border-right: 1px solid var(--stroke) !important;
    backdrop-filter: blur(18px);
}

[data-testid="stSidebarContent"] {
    padding-top: 0.2rem;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 999px;
    border: 1px solid var(--stroke);
    padding: 0.58rem 0.85rem;
    font-size: 0.86rem;
    font-weight: 600;
    color: var(--ink-1);
    background: rgba(15, 23, 42, 0.82);
    transition: all 180ms ease;
}

[data-testid="stSidebar"] .stButton > button:hover {
    border-color: rgba(59, 130, 246, 0.5);
    box-shadow: 0 8px 18px rgba(59, 130, 246, 0.16);
    transform: translateY(-1px);
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid var(--stroke);
    background: rgba(15, 23, 42, 0.85);
    color: var(--ink-1);
}

.stButton > button:hover {
    border-color: rgba(59, 130, 246, 0.5);
    color: var(--ink-1);
}

.main .block-container {
    max-width: 1240px;
    padding-top: 1.15rem;
    padding-bottom: 0.6rem;
    height: calc(100vh - 0.7rem);
    display: flex;
    flex-direction: column;
}

section.main > div {
    height: 100%;
}

[data-testid="stToolbar"] {
    background: transparent !important;
}

[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
    position: fixed;
    left: 0.55rem;
    top: 0.62rem;
    z-index: 9999;
}

[data-testid="collapsedControl"] button {
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    background: rgba(15, 23, 42, 0.95) !important;
    color: var(--ink-1) !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
}

[data-testid="collapsedControl"] svg {
    color: var(--ink-1) !important;
    fill: var(--ink-1) !important;
}

[data-testid="collapsedControl"] button:hover {
    border-color: rgba(59, 130, 246, 0.5) !important;
    background: #1e293b !important;
}

.fade-in {
    animation: fadeIn 430ms ease-out forwards;
}

.stagger-1 {
    opacity: 0;
    animation: floatUp 480ms 60ms ease-out forwards;
}

.stagger-2 {
    opacity: 0;
    animation: floatUp 480ms 130ms ease-out forwards;
}

.stagger-3 {
    opacity: 0;
    animation: floatUp 480ms 190ms ease-out forwards;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes floatUp {
    from {
        transform: translateY(8px);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

.hero-shell {
    border: 1px solid var(--stroke);
    border-radius: 24px;
    padding: 1.35rem 1.4rem 1.25rem;
    background: linear-gradient(150deg, rgba(15, 23, 42, 0.85), rgba(9, 13, 24, 0.82));
    box-shadow: var(--shadow-lg);
    margin-bottom: 1rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 999px;
    padding: 0.22rem 0.62rem;
    font-size: 0.71rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--accent-1);
    background: rgba(59, 130, 246, 0.1);
}

.hero-title {
    margin: 0.55rem 0 0.4rem;
    font-size: clamp(1.5rem, 2.6vw, 2.34rem);
    line-height: 1.14;
}

.hero-copy {
    margin: 0;
    max-width: 68ch;
    color: var(--ink-muted);
    line-height: 1.56;
    font-size: 0.95rem;
}

.info-block {
    border: 1px solid var(--stroke);
    background: rgba(15, 23, 42, 0.78);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.45rem;
}

.info-title {
    margin: 0;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    color: var(--ink-1);
}

.info-copy {
    margin: 0.28rem 0 0;
    color: var(--ink-muted);
    font-size: 0.86rem;
    line-height: 1.58;
}

.meta-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.7rem;
    margin-top: 0.95rem;
}

.mini-metric {
    border: 1px solid var(--stroke);
    border-radius: var(--radius-md);
    padding: 0.62rem 0.7rem;
    background: rgba(15, 23, 42, 0.7);
}

.mini-metric .label {
    color: var(--ink-muted);
    font-size: 0.71rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}

.mini-metric .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    color: var(--ink-1);
    font-weight: 700;
}

.panel {
    border: 1px solid var(--stroke);
    background: var(--panel);
    border-radius: var(--radius-lg);
    padding: 1rem;
    backdrop-filter: blur(12px);
}

.panel-title {
    margin: 0;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.92rem;
    color: var(--ink-1);
    letter-spacing: 0.02em;
}

.panel-subtitle {
    margin: 0.24rem 0 0;
    color: var(--ink-muted);
    font-size: 0.8rem;
}

.quick-note {
    margin: 0.66rem 0 0;
    color: var(--ink-muted);
    font-size: 0.79rem;
    line-height: 1.5;
}

.stage-track {
    display: grid;
    gap: 0.34rem;
    margin-top: 0.55rem;
}

.stage-item {
    border-radius: 10px;
    padding: 0.45rem 0.55rem;
    border: 1px solid transparent;
    font-size: 0.77rem;
}

.stage-item.complete {
    border-color: rgba(16, 185, 129, 0.35);
    background: rgba(16, 185, 129, 0.12);
    color: #34d399;
}

.stage-item.current {
    border-color: rgba(59, 130, 246, 0.4);
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    font-weight: 700;
}

.stage-item.pending {
    border-color: var(--stroke);
    background: rgba(255, 255, 255, 0.05);
    color: #9ca3af;
}

[data-testid="stChatInput"] {
    max-width: 980px;
    margin: 0 auto;
}

[data-testid="stChatInput"] > div {
    border-radius: 999px;
    background: var(--panel-strong);
    border: 1px solid var(--stroke);
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.1);
}

[data-testid="stChatInput"] textarea {
    font-size: 0.94rem !important;
    color: var(--ink-1) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #73818f !important;
}

[data-testid="stChatInputSubmitButton"] button {
    border-radius: 999px !important;
    background: linear-gradient(140deg, var(--accent-1), #2563eb) !important;
    border: none !important;
}

[data-testid="stChatInputSubmitButton"] button:hover {
    filter: brightness(1.08);
}

[data-testid="stChatMessage"] {
    background: transparent !important;
}

[data-testid="stChatMessage"] .stMarkdown,
[data-testid="stChatMessage"] .stMarkdown p {
    color: var(--ink-2) !important;
}

[data-testid="chatAvatarIcon-user"] {
    background: var(--accent-2) !important;
    color: #ffffff !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: var(--accent-1) !important;
}

.assistant-meta {
    margin-top: 0.42rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}

.chip {
    border-radius: 999px;
    font-size: 0.68rem;
    padding: 0.22rem 0.56rem;
    border: 1px solid transparent;
    line-height: 1;
}

.chip.intent {
    background: rgba(59, 130, 246, 0.15);
    border-color: rgba(59, 130, 246, 0.3);
    color: #60a5fa;
}

.chip.source {
    background: rgba(139, 92, 246, 0.15);
    border-color: rgba(139, 92, 246, 0.3);
    color: #a78bfa;
}

.success-banner {
    border: 1px solid rgba(16, 185, 129, 0.35);
    background: linear-gradient(160deg, rgba(16, 185, 129, 0.13), rgba(15, 23, 42, 0.7));
    border-radius: 14px;
    padding: 0.84rem 0.92rem;
    margin-top: 0.45rem;
}

.success-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    color: #34d399;
    margin-bottom: 0.16rem;
}

.success-body {
    font-size: 0.84rem;
    color: #d1d5db;
    line-height: 1.55;
}

.small-muted {
    color: #6f7f8b;
    font-size: 0.74rem;
}

[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    overflow: hidden !important;
}

.chat-scroll {
    flex: 1;
    min-height: 0;
    overflow-y: auto !important;
    overflow-x: hidden;
    padding-right: 0.15rem;
    padding-bottom: 0.4rem;
}

.chat-scroll::-webkit-scrollbar {
    width: 8px;
}

.chat-scroll::-webkit-scrollbar-thumb {
    background: rgba(20, 33, 61, 0.22);
    border-radius: 999px;
}

[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottomBlockContainer"],
.stBottomBlockContainer {
    background: transparent !important;
    background-color: transparent !important;
}

[data-testid="stBottom"]::after {
    content: "AutoStream AI can make mistakes. Verify important information before acting.";
    display: block;
    text-align: center;
    color: #6f7f8b;
    font-size: 0.74rem;
    line-height: 1.25;
    padding: 0.22rem 0 0.45rem;
}

.side-metric {
    margin: 0;
    text-transform: uppercase;
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    color: #6b7a86;
    font-weight: 700;
}

.side-value {
    margin: 0.14rem 0 0.65rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    color: var(--ink-1);
    font-weight: 700;
}

@media (max-width: 1024px) {
    .meta-row {
        grid-template-columns: 1fr;
    }
    .hero-shell {
        padding: 1rem;
    }
}

@media (max-width: 780px) {
    .main .block-container {
        padding-top: 0.9rem;
    }
    .hero-title {
        font-size: 1.48rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "msgs": [],
        "graph": None,
        "lead": {
            "capture_stage": "NOT_STARTED",
            "name": None,
            "email": None,
            "platform": None,
        },
        "captured": False,
        "console": "",
        "intent": "",
        "confidence": 0.0,
        "turns": 0,
        "queued_prompt": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


@st.cache_resource(show_spinner=False)
def load_graph():
    from agent import get_graph

    return get_graph()


def esc(value: str) -> str:
    return html_module.escape(str(value or ""))


def split_sources(text: str) -> tuple[str, str]:
    match = re.search(r"\n*\*Source:\s*(.+?)\*\s*$", text)
    if match:
        body = text[: match.start()].rstrip()
        sources = match.group(1).strip()
        return body, sources
    return text, ""


def stage_index(stage: str) -> int:
    order = [
        "NOT_STARTED",
        "COLLECTING_NAME",
        "COLLECTING_EMAIL",
        "COLLECTING_PLATFORM",
        "COMPLETE",
    ]
    if stage in order:
        return order.index(stage)
    return 0


def render_hero() -> None:
    st.markdown(
                """
    <section class="hero-shell fade-in">
      <span class="hero-badge">Live Sales Assistant</span>
      <h1 class="hero-title">Close Leads Faster With Conversational Clarity</h1>
      <p class="hero-copy">
        AutoStream Studio blends grounded product answers with a guided sign-up funnel.
        Ask complex plan questions, qualify intent, and capture buyer details in one smooth conversation.
      </p>
    </section>
    """,
        unsafe_allow_html=True,
    )


def queue_prompt(prompt_text: str) -> None:
    st.session_state.queued_prompt = prompt_text
    st.rerun()


def clear_chat_state() -> None:
    for key in list(st.session_state.keys()):
        if key != "graph":
            del st.session_state[key]
    init_state()
    st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        if st.button("Start New Conversation", use_container_width=True):
            clear_chat_state()

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        
        intent_label = st.session_state.intent.replace("_", " ").title() if st.session_state.intent else "No Intent Yet"
        st.markdown(
            f"""
        <div class="panel fade-in" style="margin-bottom:0.65rem; background: rgba(15, 23, 42, 0.65); padding: 0.8rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
          <p class="side-metric" style="margin: 0; font-size: 0.75rem; color: #9ca3af;">Conversation Turns</p>
          <p class="side-value" style="margin: 0.2rem 0 0; font-size: 1.1rem; font-weight: bold; color: #ffffff;">{st.session_state.turns}</p>
        </div>

        <div class="panel fade-in" style="margin-bottom:0.65rem; background: rgba(15, 23, 42, 0.65); padding: 0.8rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
          <p class="side-metric" style="margin: 0; font-size: 0.75rem; color: #9ca3af;">Latest Intent</p>
          <p class="side-value" style="margin: 0.2rem 0 0; font-size: 1.1rem; font-weight: bold; color: #ffffff;">{esc(intent_label)}</p>
        </div>

        <div class="panel fade-in" style="margin-bottom:0.65rem; background: rgba(15, 23, 42, 0.65); padding: 0.8rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">
          <p class="side-metric" style="margin: 0; font-size: 0.75rem; color: #9ca3af;">Support Coverage</p>
          <p class="side-value" style="margin: 0.2rem 0 0; font-size: 1.1rem; font-weight: bold; color: #ffffff;">Basic: M-F | Pro: 24/7</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_info_block() -> None:
    st.markdown(
        """
    <section class="info-block">
      <p class="info-title">What You Can Ask</p>
      <p class="info-copy">
        Ask about plan pricing, feature gaps, refund and support policy, or jump directly into sign-up.
        The assistant stays grounded in your knowledge base and cites sources when available.
      </p>
    </section>
    """,
        unsafe_allow_html=True,
    )


def render_chat_scroll_start() -> None:
    pass


def render_chat_scroll_end() -> None:
    pass


def render_messages() -> None:
    for item in st.session_state.msgs:
        if item["role"] == "user":
            with st.chat_message("user"):
                st.markdown(item["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(item["content"])
                chips = []
                if item.get("intent"):
                    chips.append(
                        f"<span class='chip intent'>{esc(item['intent'].replace('_', ' ').title())} · {item.get('confidence', 0.0):.0%}</span>"
                    )
                if item.get("sources"):
                    chips.append(f"<span class='chip source'>Source: {esc(item['sources'])}</span>")
                if chips:
                    st.markdown(
                        f"<div class='assistant-meta'>{''.join(chips)}</div>",
                        unsafe_allow_html=True,
                    )

    if st.session_state.captured:
        lead = st.session_state.lead
        st.markdown(
            f"""
        <div class="success-banner">
          <div class="success-title">Lead Captured Successfully</div>
          <div class="success-body">
            Name: {esc(lead.get('name') or 'N/A')}<br>
            Email: {esc(lead.get('email') or 'N/A')}<br>
            Platform: {esc(lead.get('platform') or 'N/A')}
          </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def get_message_to_send() -> str:
    typed = st.chat_input("Ask about plans, pricing, support, or say: I want to sign up")
    if typed and typed.strip():
        return typed.strip()

    queued = st.session_state.queued_prompt
    if queued:
        st.session_state.queued_prompt = ""
        return queued
    return ""


def invoke_agent(user_text: str) -> None:
    st.session_state.msgs.append({"role": "user", "content": user_text})

    if st.session_state.graph is None:
        with st.spinner("Warming up the assistant graph..."):
            st.session_state.graph = load_graph()

    graph = st.session_state.graph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    buf = io.StringIO()
    with redirect_stdout(buf):
        with st.spinner("Thinking through retrieval and intent routing..."):
            state = graph.invoke({"messages": [HumanMessage(content=user_text)]}, config)

    captured_out = buf.getvalue().strip()
    if "LEAD CAPTURED" in captured_out:
        st.session_state.console = captured_out

    st.session_state.intent = state.get("current_intent", "")
    st.session_state.confidence = state.get("intent_confidence", 0.0)
    st.session_state.turns = state.get("turn_count", 0)

    lead_data = state.get("lead_data", {})
    st.session_state.lead = {
        "capture_stage": lead_data.get("capture_stage", "NOT_STARTED"),
        "name": lead_data.get("name"),
        "email": lead_data.get("email"),
        "platform": lead_data.get("platform"),
    }
    st.session_state.captured = state.get("lead_captured", False)

    if state.get("lead_log"):
        st.session_state.console = state["lead_log"]

    ai_messages = [m for m in state.get("messages", []) if getattr(m, "type", "") == "ai"]
    raw_response = ai_messages[-1].content if ai_messages else "I hit a temporary issue. Please try again."
    body, sources = split_sources(raw_response)

    st.session_state.msgs.append(
        {
            "role": "agent",
            "content": body,
            "intent": st.session_state.intent,
            "confidence": st.session_state.confidence,
            "sources": sources,
        }
    )


render_sidebar()

render_hero()

render_info_block()

st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)

render_chat_scroll_start()
chat_shell = st.container(height=350, border=False)
with chat_shell:
    render_messages()
render_chat_scroll_end()


message = get_message_to_send()
if message:
    invoke_agent(message)
    st.rerun()
