# Financial RAG Agents — Fortune 500 Financial Analyst

> **Honorable Mention, MSADS Hackathon 2024 · University of Chicago**

A multi-agent AI system for financial analysis of any Fortune 500 company. Ask a question, get a concise answer with exact cited figures, a structured metrics table, a risk score, and trend charts — all in one pipeline.

---

## Before & After: v1 → v3

### v1 — Hackathon Submission (October 2024)

Built in 48 hours for the MSADS Hackathon at UChicago. A single RAG pipeline that could read annual report PDFs from 4 companies (Amazon, Apple, Alphabet, Meta) and answer basic questions.

**What it could do:**
- Answer questions about 4 company PDFs
- Use ColBERT reranking on top of a Chroma vector store
- Return raw text responses via a basic Streamlit UI

**Limitations:**
- Covered only 4 companies, only the documents in the knowledge base
- No live data — if you asked about a company not in the PDFs, it returned nothing
- Single LLM call — no separation of retrieval, analysis, and risk
- ~30–60 second response times due to ColBERT model loading on every query
- Basic light-themed UI with no charts, no structured output

**Tech:** Qwen2 (local LLM), ColBERT, ChromaDB, Unstructured.io, LlamaParse, Phi3 Vision for image summarization, Streamlit

---

### v3 — Current (April 2026)

Rebuilt into a 4-agent system. Each agent has a specific job. The system combines a private RAG knowledge base with live market data, SEC filings, and web search — covering any publicly traded Fortune 500 company.

**What it can do:**
- Answer financial questions about **any Fortune 500 company** (not just 5)
- Pull from 4 data sources: private RAG knowledge base, yfinance live financials, SEC EDGAR 10-K/10-Q filings, web search
- Stream the analyst response live (token by token)
- Output structured metrics (revenue, margins, FCF, debt/equity) in a table
- Score financial risk 0–10 with specific red flags from the Risk agent
- Render Plotly time-series charts for trend questions
- Remember context across follow-up questions in the same session
- Stop generation mid-stream via sidebar button

**Improvements over v1:**
- Response time: ~30–60s → ~10–20s (removed ColBERT, added MMR retrieval + yfinance caching)
- Coverage: 4 companies → 5 deep (RAG) + any Fortune 500 via live data
- Output: raw text → streamed narrative + metrics table + risk badge + charts
- Agents: 1 pipeline → 4 specialized agents with shared memory
- Prompt caching on all system prompts reduces API cost ~70% on warm cache
- Dollar sign rendering bug fixed (Streamlit LaTeX interference)
- Risk score now comes from the actual Risk agent, not a heuristic calculation
- Dark-themed charts consistent with UI (was white-on-black before)

**Tech:** Claude Sonnet 4.6 (Anthropic), ChromaDB + MMR, sentence-transformers, yfinance, SEC EDGAR API, Streamlit, Plotly

---

## How It Works

A question flows through 4 agents in sequence, each building on the previous:

```
User Query
    │
    ▼
[1] Retrieval Agent
    • RAG knowledge base: Amazon, Apple, Alphabet, Meta, NVIDIA (2020–2024)
      1,970 vision-extracted table and chart summaries from annual reports
    • yfinance: income statement, balance sheet, cash flow for any ticker (TTL-cached)
    • SEC EDGAR: 10-K / 10-Q filing text for qualitative context
    • Web search: current prices, news, analyst ratings (optional)
    │
    ▼
[2] Metrics Agent
    • Extracts structured JSON: revenue, net income, margins, FCF, debt/equity
    • Uses a financial calculator tool for YoY growth, CAGR, margin computation
    │
    ▼
[3] Analyst Agent  ← streams live
    • Concise narrative: direct answer + bullet-point metrics + one Takeaway
    • Every figure cited with fiscal year. Never guesses missing data.
    • Receives full conversation history for multi-turn follow-up questions
    │
    ▼
[4] Risk Agent
    • Scores financial risk 0–10 (Low / Medium / High)
    • Specific red flags: declining revenue, negative FCF, high debt/equity, etc.
    • Uses web search to check for recent regulatory or legal concerns
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Claude Sonnet 4.6 with prompt caching |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB (MMR retrieval, k=8 from fetch_k=20) |
| Live financials | yfinance with 1-hour TTL cache |
| SEC filings | EDGAR full-text search API |
| Web search | Google Custom Search API (optional) |
| Frontend | Streamlit + Plotly (dark theme) |

---

## Setup

```bash
git clone https://github.com/ashmita23/FinancialRAG_Agents
cd FinancialRAG_Agents
pip install -r requirements.txt

# .env file
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key      # optional — enables web search
GOOGLE_CSE_ID=your_cse_id   # optional

streamlit run streamLIT.py
```

---

## Example Questions

**RAG knowledge base (deep — Amazon, Apple, Alphabet, Meta, NVIDIA):**
- What was NVIDIA's revenue and net income in FY2023?
- How did Apple's gross margin trend from 2021 to 2023?
- What drove Meta's revenue recovery in 2023 after the 2022 decline?

**Live data (any Fortune 500):**
- What was Microsoft's revenue and net income in 2023?
- Compare Walmart and Costco revenue for 2023
- What are the biggest financial risks for Tesla?

**Time series / charts:**
- Show me NVIDIA's quarterly revenue trend over the last 2 years
- What is Apple's net income history quarterly?

---

*Originally submitted at MSADS Hackathon 2024, University of Chicago — Honorable Mention.*
