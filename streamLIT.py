import streamlit as st
from Rag import FinancialOrchestrator

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fortune 500 Financial Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (purple/black/white theme extensions beyond config.toml)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0A0A0A;
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #1C1C1C;
        color: #E9D5FF !important;
        border: 1px solid #7C3AED;
        border-radius: 8px;
        width: 100%;
        font-size: 0.85rem;
        padding: 0.4rem 0.6rem;
        margin-bottom: 4px;
        transition: background 0.15s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #7C3AED;
        color: #FFFFFF !important;
    }

    /* ── Main header ── */
    .app-header {
        background: linear-gradient(135deg, #7C3AED 0%, #4C1D95 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        color: #FFFFFF;
    }
    .app-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .app-header p  { margin: 0.4rem 0 0; font-size: 0.95rem; opacity: 0.85; }

    /* ── Chat bubbles ── */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.2rem 0;
    }

    /* ── Tool expander ── */
    .tool-expander {
        background: #F5F3FF;
        border: 1px solid #DDD6FE;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .tool-badge {
        display: inline-block;
        background: #7C3AED;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .tool-result-preview {
        color: #4B5563;
        font-family: monospace;
        font-size: 0.78rem;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* ── Welcome card ── */
    .welcome-card {
        background: #F9F7FF;
        border: 1px solid #DDD6FE;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #4B5563;
    }

    /* ── Error box ── */
    .error-box {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #B91C1C;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    # API-format history passed to generate()
    st.session_state.messages = []

if "display_history" not in st.session_state:
    # UI-format history: list of {role, content, tools}
    st.session_state.display_history = []

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Fortune 500 Analyst")
    st.markdown(
        "Ask financial questions about any **Fortune 500** company.\n\n"
        "Powered by **Claude claude-sonnet-4-6** with live financial data."
    )
    st.divider()

    st.markdown("**Quick-select (RAG knowledge base)**")
    quick_companies = {
        "Amazon": "What was Amazon's total revenue in 2023?",
        "Apple": "What was Apple's gross margin in 2023?",
        "Alphabet": "How did Alphabet's operating income change from 2022 to 2023?",
        "Meta": "What was Meta's net income in 2023?",
        "NVIDIA": "What drove NVIDIA's revenue growth in 2023?",
    }
    for company, example_q in quick_companies.items():
        if st.button(company, key=f"quick_{company}"):
            st.session_state["prefill_query"] = example_q

    st.divider()
    st.markdown("**Any Fortune 500 company**")
    st.caption(
        "Try tickers like WMT, JPM, XOM, TSLA, BAC, CVX, HD, LLY, UNH …"
    )

    st.divider()
    st.markdown("**Tips**")
    st.caption(
        "• Ask for YoY growth, margins, or specific ratios\n"
        "• Compare two companies: 'Compare WMT and COST revenue in 2023'\n"
        "• Ask follow-up questions — context is remembered"
    )

    st.divider()
    if st.button("Clear conversation", key="clear"):
        st.session_state.messages = []
        st.session_state.display_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Initialise orchestrator (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_orchestrator():
    return FinancialOrchestrator()


with st.spinner("Loading financial knowledge base…"):
    try:
        if st.session_state.orchestrator is None:
            st.session_state.orchestrator = load_orchestrator()
        orchestrator = st.session_state.orchestrator
        _init_ok = True
    except Exception as _init_err:
        _init_ok = False
        _init_error = str(_init_err)

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📊 Fortune 500 Financial Analyst</h1>
        <p>Instant financial analysis across Fortune 500 companies — powered by live data and AI reasoning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not _init_ok:
    st.markdown(
        f'<div class="error-box">Failed to initialise: {_init_error}</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
TOOL_DISPLAY_NAMES = {
    "rag_search": "Knowledge Base Search",
    "get_financial_statements": "Financial Statements",
    "search_sec_filings": "SEC Filings",
    "web_search": "Web Search",
    "financial_calculator": "Calculator",
    "compare_companies": "Company Comparison",
}

TOOL_ICONS = {
    "rag_search": "🔍",
    "get_financial_statements": "📑",
    "search_sec_filings": "🏛️",
    "web_search": "🌐",
    "financial_calculator": "🧮",
    "compare_companies": "⚖️",
}


def render_tool_calls(tool_calls: list):
    """Render collapsed tool call details inside an expander."""
    if not tool_calls:
        return
    label = f"🔍 Used {len(tool_calls)} tool{'s' if len(tool_calls) > 1 else ''}"
    with st.expander(label, expanded=False):
        for tc in tool_calls:
            icon = TOOL_ICONS.get(tc["tool"], "🔧")
            display_name = TOOL_DISPLAY_NAMES.get(tc["tool"], tc["tool"])
            st.markdown(
                f'<span class="tool-badge">{icon} {display_name}</span>',
                unsafe_allow_html=True,
            )
            # Show key input fields
            inp = tc.get("input", {})
            inp_summary = " · ".join(f"{k}: `{v}`" for k, v in inp.items())
            st.caption(inp_summary)
            if tc.get("result"):
                st.markdown(
                    f'<div class="tool-result-preview">{tc["result"][:400]}'
                    + ("…" if len(tc.get("result", "")) > 400 else "")
                    + "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("---")


if not st.session_state.display_history:
    st.markdown(
        """
        <div class="welcome-card">
            <h3 style="color:#7C3AED; margin-top:0">Welcome! Ask a financial question.</h3>
            <p>Examples:</p>
            <ul style="text-align:left; display:inline-block">
                <li>What was Apple's revenue and net income in 2023?</li>
                <li>Compare Walmart and Costco revenue in 2023</li>
                <li>What's ExxonMobil's debt-to-equity ratio?</li>
                <li>What drove NVIDIA's revenue growth from 2022 to 2023?</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for entry in st.session_state.display_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant":
                render_tool_calls(entry.get("tools", []))

# ---------------------------------------------------------------------------
# Chat input (with optional prefill from sidebar quick-select)
# ---------------------------------------------------------------------------
prefill = st.session_state.pop("prefill_query", "")
user_input = st.chat_input(
    "Ask about any Fortune 500 company…",
    key="chat_input",
) or prefill

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Accumulate tool calls and final answer for this turn
    turn_tool_calls = []
    final_answer = ""

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        status_placeholder = st.empty()

        with st.status("Analysing…", expanded=False) as status_box:
            for event in orchestrator.generate(
                user_input, st.session_state.messages
            ):
                if event["type"] == "tool_call":
                    tool_name = event["tool"]
                    display = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
                    icon = TOOL_ICONS.get(tool_name, "🔧")
                    status_box.update(label=f"{icon} {display}…")
                    turn_tool_calls.append(
                        {"tool": tool_name, "input": event["input"], "result": ""}
                    )

                elif event["type"] == "tool_result":
                    # Attach result to the last matching tool call
                    for tc in reversed(turn_tool_calls):
                        if tc["tool"] == event["tool"] and not tc["result"]:
                            tc["result"] = event["result"]
                            break

                elif event["type"] == "final":
                    final_answer = event["text"]
                    status_box.update(label="Done", state="complete", expanded=False)

                elif event["type"] == "error":
                    final_answer = f"**Error:** {event['text']}"
                    status_box.update(label="Error", state="error")

        answer_placeholder.markdown(final_answer or "_No response generated._")
        render_tool_calls(turn_tool_calls)

    # -----------------------------------------------------------------------
    # Update session state histories
    # -----------------------------------------------------------------------
    # API history: user turn
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    # API history: assistant turn (plain text only — tool calls already consumed)
    if final_answer:
        st.session_state.messages.append(
            {"role": "assistant", "content": final_answer}
        )

    # Display history
    st.session_state.display_history.append(
        {"role": "user", "content": user_input, "tools": []}
    )
    st.session_state.display_history.append(
        {"role": "assistant", "content": final_answer or "_No response._", "tools": turn_tool_calls}
    )
