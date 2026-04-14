import streamlit as st
from Rag import FinancialController

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fortune 500 Financial Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS (purple/black/white theme)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── App header ── */
    .app-header {
        background: linear-gradient(135deg, #7C3AED 0%, #4C1D95 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .app-header h1 { margin: 0; font-size: 2.1rem; font-weight: 700; }
    .app-header p  { margin: 0.4rem 0 0; font-size: 1rem; opacity: 0.85; }

    /* ── Agent pipeline chips ── */
    .agent-chips { margin: 0.5rem 0; }
    .chip-done {
        display: inline-block;
        background: #7C3AED;
        color: white;
        border-radius: 12px;
        padding: 4px 14px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .chip-arrow {
        color: #9CA3AF;
        font-size: 0.85rem;
        margin-right: 4px;
    }

    /* ── Tool badge ── */
    .tool-badge {
        display: inline-block;
        background: #7C3AED;
        color: #FFFFFF;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .tool-result-preview {
        color: #4B5563;
        font-family: monospace;
        font-size: 0.82rem;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* ── Risk badge ── */
    .risk-low    { background:#D1FAE5; border:1px solid #6EE7B7; border-radius:8px; padding:10px 16px; color:#065F46; font-weight:600; font-size:1rem; }
    .risk-medium { background:#FEF3C7; border:1px solid #FCD34D; border-radius:8px; padding:10px 16px; color:#92400E; font-weight:600; font-size:1rem; }
    .risk-high   { background:#FEE2E2; border:1px solid #FCA5A5; border-radius:8px; padding:10px 16px; color:#991B1B; font-weight:600; font-size:1rem; }

    /* ── Chat message text — bigger + more readable ── */
    .stChatMessage .stMarkdown p,
    .stChatMessage .stMarkdown li {
        font-size: 1.05rem;
        line-height: 1.75;
    }
    .stChatMessage .stMarkdown h3 {
        font-size: 1.15rem;
    }
    /* Ensure bold always renders */
    strong { font-weight: 700 !important; }

    /* ── Welcome card ── */
    .welcome-card {
        background: #F9F7FF;
        border: 1px solid #DDD6FE;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #4B5563;
        font-size: 1.05rem;
    }

    /* ── Guide panel company buttons ── */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        background: #F5F3FF;
        color: #5B21B6;
        border: 1px solid #DDD6FE;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0.35rem 1rem;
        transition: all 0.15s;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: #7C3AED;
        color: #FFFFFF;
        border-color: #7C3AED;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # API-format, passed to generate()

if "display_history" not in st.session_state:
    st.session_state.display_history = []   # UI-format, for rendering

if "controller" not in st.session_state:
    st.session_state.controller = None

# ---------------------------------------------------------------------------
# Sidebar — minimal (just clear button; content moved to guide panel)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Fortune 500 Analyst")
    st.divider()
    if st.button("🗑️ Clear conversation", key="clear"):
        st.session_state.messages = []
        st.session_state.display_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Load controller (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_controller():
    return FinancialController()


with st.spinner("Loading financial knowledge base…"):
    try:
        if st.session_state.controller is None:
            st.session_state.controller = load_controller()
        controller = st.session_state.controller
        _init_ok = True
    except Exception as _err:
        _init_ok = False
        _init_error = str(_err)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📊 Fortune 500 Financial Analyst</h1>
        <p>4-agent system: Retrieval → Metrics → Analysis → Risk  ·  Live data + RAG knowledge base</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Guide panel — how to use + quick company buttons
# ---------------------------------------------------------------------------
_guide_open = not bool(st.session_state.display_history)
with st.expander("📖  How to use  ·  Quick companies", expanded=_guide_open):
    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            "**How to use**\n"
            "- Ask about **any Fortune 500 company** by name or ticker\n"
            "- Include the **year** for precise data — e.g. *'Apple revenue 2023'*\n"
            "- Use **'compare X vs Y'** to compare two companies side by side\n"
            "- Use **'trend'** or **'chart'** to get a time series graph\n"
            "- Follow-up questions remember the prior conversation context\n"
            "- Try tickers directly: *WMT, JPM, XOM, TSLA, MSFT, CVX, HD, LLY*"
        )

    with right:
        st.markdown("**Quick questions — famous companies**")
        quick = {
            "Amazon":  "What was Amazon's revenue and operating income in 2023?",
            "Apple":   "How did Apple's gross margin trend from 2021 to 2023?",
            "Alphabet": "What drove Alphabet's revenue growth in 2023?",
            "Meta":    "What was Meta's net income and free cash flow in 2023?",
            "NVIDIA":  "Show me NVIDIA's quarterly revenue trend over the last 2 years",
        }
        cols = st.columns(len(quick))
        for col, (company, example_q) in zip(cols, quick.items()):
            with col:
                if st.button(company, key=f"q_{company}"):
                    st.session_state["prefill_query"] = example_q
                    st.rerun()

if not _init_ok:
    st.error(f"Initialisation failed: {_init_error}")
    st.stop()

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
AGENT_META = {
    "retrieval": ("📥", "Retrieval"),
    "metrics":   ("📊", "Metrics"),
    "analyst":   ("🧠", "Analysis"),
    "risk":      ("⚠️",  "Risk"),
}

TOOL_NAMES = {
    "rag_search":               "Knowledge Base",
    "get_financial_statements": "Financial Statements",
    "search_sec_filings":       "SEC Filings",
    "web_search":               "Web Search",
    "financial_calculator":     "Calculator",
    "compare_companies":        "Comparison",
    "get_time_series":          "Time Series",
}

TOOL_ICONS = {
    "rag_search":               "🔍",
    "get_financial_statements": "📑",
    "search_sec_filings":       "🏛️",
    "web_search":               "🌐",
    "financial_calculator":     "🧮",
    "compare_companies":        "⚖️",
    "get_time_series":          "📈",
}


def render_agent_chips(agents: list) -> None:
    if not agents:
        return
    parts = []
    for i, a in enumerate(agents):
        icon, name = AGENT_META.get(a, ("🤖", a))
        parts.append(f'<span class="chip-done">{icon} {name} ✓</span>')
        if i < len(agents) - 1:
            parts.append('<span class="chip-arrow">→</span>')
    st.markdown(
        f'<div class="agent-chips">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def render_tool_calls(tool_calls: list) -> None:
    if not tool_calls:
        return
    label = f"🔧 {len(tool_calls)} tool call{'s' if len(tool_calls) > 1 else ''}"
    with st.expander(label, expanded=False):
        for tc in tool_calls:
            icon = TOOL_ICONS.get(tc["tool"], "🔧")
            display = TOOL_NAMES.get(tc["tool"], tc["tool"])
            st.markdown(
                f'<span class="tool-badge">{icon} {display}</span>',
                unsafe_allow_html=True,
            )
            inp = tc.get("input", {})
            st.caption(" · ".join(f"{k}: `{v}`" for k, v in inp.items()))
            if tc.get("result"):
                preview = tc["result"][:400] + ("…" if len(tc["result"]) > 400 else "")
                st.markdown(
                    f'<div class="tool-result-preview">{preview}</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("---")


def render_metrics_table(metrics: dict) -> None:
    if not metrics:
        return
    import pandas as pd

    label_map = {
        "revenue_b":             ("Revenue", "B"),
        "revenue_growth_pct":    ("Revenue Growth", "%"),
        "net_income_b":          ("Net Income", "B"),
        "net_income_growth_pct": ("Net Income Growth", "%"),
        "gross_margin_pct":      ("Gross Margin", "%"),
        "operating_margin_pct":  ("Operating Margin", "%"),
        "free_cash_flow_b":      ("Free Cash Flow", "B"),
        "debt_to_equity":        ("Debt / Equity", "x"),
        "year":                  ("Fiscal Year", ""),
    }
    rows = []
    for key, (label, unit) in label_map.items():
        val = metrics.get(key)
        if val is not None:
            if unit == "":
                rows.append({"Metric": label, "Value": str(int(val))})
            elif unit == "B":
                rows.append({"Metric": label, "Value": f"${val:.2f}B"})
            elif unit == "%":
                rows.append({"Metric": label, "Value": f"{val:.1f}%"})
            else:
                rows.append({"Metric": label, "Value": f"{val:.2f}x"})
    if rows:
        with st.expander("📊 Extracted Metrics", expanded=False):
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )


def render_risk_badge(risk: dict) -> None:
    if not risk:
        return
    score = risk.get("score", "N/A")
    level = risk.get("level", "Unknown")
    css_class = {
        "Low": "risk-low",
        "Medium": "risk-medium",
        "High": "risk-high",
    }.get(level, "risk-medium")
    st.markdown(
        f'<div class="{css_class}">⚠️ Risk Score: {score}/10 — {level}</div>',
        unsafe_allow_html=True,
    )


def render_chart(chart: dict) -> None:
    try:
        import plotly.graph_objects as go

        labels = chart.get("labels", [])
        values = chart.get("values", [])
        if not labels or not values:
            return
        ticker = chart.get("ticker", "")
        metric = chart.get("metric", "").replace("_", " ").title()
        period = chart.get("period", "")
        is_price = chart.get("metric") == "price"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=values,
                mode="lines+markers",
                line=dict(color="#7C3AED", width=2.5),
                marker=dict(size=6, color="#7C3AED"),
                fill="tozeroy",
                fillcolor="rgba(124,58,237,0.08)",
            )
        )
        y_title = "Stock Price ($)" if is_price else f"{metric} ($ Billions)"
        fig.update_layout(
            title=dict(
                text=f"{ticker} — {metric} ({period.title()})",
                font=dict(size=14, color="#0A0A0A"),
            ),
            xaxis_title="Period",
            yaxis_title=y_title,
            height=320,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="white",
            plot_bgcolor="#F9F7FF",
            xaxis=dict(tickangle=-30),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Chart unavailable: {e}")


def render_assistant_entry(entry: dict) -> None:
    """Render a full assistant chat entry with all its components."""
    st.markdown(entry["content"])
    render_agent_chips(entry.get("agents", []))
    render_metrics_table(entry.get("metrics", {}))
    render_risk_badge(entry.get("risk", {}))
    for chart in entry.get("charts", []):
        render_chart(chart)
    render_tool_calls(entry.get("tools", []))


# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------
if not st.session_state.display_history:
    st.markdown(
        """
        <div class="welcome-card">
            <h3 style="color:#7C3AED;margin-top:0">Welcome! Ask a financial question.</h3>
            <p>Example questions:</p>
            <ul style="text-align:left;display:inline-block">
                <li>What was Apple's revenue and net income in 2023?</li>
                <li>Compare Walmart and Costco revenue for 2023</li>
                <li>Show me NVIDIA's quarterly revenue trend</li>
                <li>What are the biggest risks for Tesla?</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for entry in st.session_state.display_history:
        with st.chat_message(entry["role"]):
            if entry["role"] == "user":
                st.markdown(entry["content"])
            else:
                render_assistant_entry(entry)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
prefill = st.session_state.pop("prefill_query", "")
user_input = st.chat_input("Ask about any Fortune 500 company…") or prefill

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Accumulators for this turn ──
    turn_tool_calls: list = []
    streamed_text: str = ""
    final_answer: str = ""
    pending_metrics: dict = {}
    pending_charts: list = []
    pending_risk: dict = {}
    completed_agents: list = []

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        with st.status("Starting analysis…", expanded=False) as status_box:
            for event in controller.generate(
                user_input, st.session_state.messages
            ):
                etype = event.get("type")

                if etype == "agent_start":
                    icon, name = AGENT_META.get(event["agent"], ("🤖", event["agent"]))
                    status_box.update(label=f"{icon} {name} Agent running…")

                elif etype == "agent_done":
                    completed_agents.append(event["agent"])

                elif etype == "tool_call":
                    turn_tool_calls.append(
                        {"tool": event["tool"], "input": event["input"], "result": ""}
                    )

                elif etype == "tool_result":
                    for tc in reversed(turn_tool_calls):
                        if tc["tool"] == event["tool"] and not tc["result"]:
                            tc["result"] = event["result"]
                            break

                elif etype == "text_delta":
                    streamed_text += event["text"]
                    answer_placeholder.markdown(streamed_text + " ▌")

                elif etype == "chart_data":
                    pending_charts.append(event)

                elif etype == "metrics_json":
                    pending_metrics = event.get("data", {})

                elif etype == "final":
                    final_answer = event.get("text", "")
                    status_box.update(label="Done", state="complete", expanded=False)

                elif etype == "error":
                    final_answer = f"**Error:** {event.get('text', 'Unknown error')}"
                    status_box.update(label="Error", state="error")

        # ── Determine display text ──
        display_text = final_answer or streamed_text or "_No response generated._"
        answer_placeholder.markdown(display_text, unsafe_allow_html=True)

        # ── Extract risk from metrics for badge ──
        if pending_metrics:
            # Infer risk level from metrics for badge coloring
            score = None
            flags = []
            rev_growth = pending_metrics.get("revenue_growth_pct")
            net_income = pending_metrics.get("net_income_b")
            op_margin = pending_metrics.get("operating_margin_pct")
            fcf = pending_metrics.get("free_cash_flow_b")
            de = pending_metrics.get("debt_to_equity")
            _score = 0
            if rev_growth is not None and rev_growth < 0:
                _score += 2; flags.append("Revenue declining")
            if net_income is not None and net_income < 0:
                _score += 3; flags.append("Net income negative")
            if de is not None and de > 3:
                _score += 2; flags.append("High debt/equity")
            if fcf is not None and fcf < 0:
                _score += 2; flags.append("Negative FCF")
            if op_margin is not None and op_margin < 5:
                _score += 1; flags.append("Low operating margin")
            level = "Low" if _score <= 3 else ("Medium" if _score <= 6 else "High")
            pending_risk = {"score": _score, "level": level, "flags": flags}

        # ── Post-stream rendering ──
        render_agent_chips(completed_agents)
        render_metrics_table(pending_metrics)
        render_risk_badge(pending_risk)
        for chart in pending_charts:
            render_chart(chart)
        render_tool_calls(turn_tool_calls)

    # ── Update session state ──
    st.session_state.messages.append({"role": "user", "content": user_input})
    if display_text:
        st.session_state.messages.append(
            {"role": "assistant", "content": display_text}
        )

    st.session_state.display_history.append(
        {"role": "user", "content": user_input, "tools": [], "agents": [],
         "metrics": {}, "charts": [], "risk": {}}
    )
    st.session_state.display_history.append(
        {
            "role": "assistant",
            "content": display_text,
            "tools": turn_tool_calls,
            "agents": completed_agents,
            "metrics": pending_metrics,
            "charts": pending_charts,
            "risk": pending_risk,
        }
    )
