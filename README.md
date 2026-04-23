# Financial RAG Agents — Fortune 500 Financial Analyst

> Honorable Mention, MSADS Hackathon 2024 · University of Chicago

## TL;DR

Started as a single RAG pipeline for 4 company PDFs (hackathon project). Now a **4-agent AI system** that answers financial questions about any Fortune 500 company — combining a private vector knowledge base, live market data, SEC filings, and streaming responses.

Ask *"How did Apple's gross margin trend from 2021–2023?"* and get: exact figures pulled from annual reports, structured metrics, a concise narrative with cited numbers, and a 0–10 risk score — all in ~15 seconds.

**v1 → v3 in short:** single LLM call → 4 specialized agents → prompt caching + stop button + multi-turn memory.

A multi-agent AI system that answers financial questions about any Fortune 500 company — combining a private RAG knowledge base, live market data, SEC filings, and a structured 4-agent reasoning pipeline.

---

## What Changed: v1 → v3

| | v1 (Hackathon, 2024) | v2 | v3 (Current) |
|---|---|---|---|
| **LLM** | Qwen2 (open-source, local) | Claude Sonnet | Claude Sonnet 4.6 |
| **Agents** | 1 RAG pipeline | 4 agents (sequential) | 4 agents + prompt caching |
| **Data sources** | 4 company PDFs only | RAG + yfinance | RAG + yfinance + SEC EDGAR + web search |
| **Coverage** | 4 companies | 5 companies (RAG) | 5 deep + any Fortune 500 via live data |
| **Retrieval** | ColBERT reranker | ColBERT reranker | MMR (faster, no model load) |
| **Response time** | ~30–60s | ~20–40s | ~10–20s |
| **UI** | Basic Streamlit | Dark theme | Dark theme + dark charts + streaming |
| **Output** | Raw text | Streamed narrative | Narrative + metrics table + risk score + charts |

---

## How It Works

A user question flows through 4 specialized agents in sequence:

```
User Query
    │
    ▼
[1] Retrieval Agent — fetches raw data
    • RAG knowledge base (Amazon, Apple, Alphabet, Meta, NVIDIA 2020–2024)
    • yfinance — income statement, balance sheet, cash flow for any ticker
    • SEC EDGAR — 10-K / 10-Q filings for qualitative context
    • Web search — current prices, news, analyst ratings
    │
    ▼
[2] Metrics Agent — extracts structured JSON
    • Revenue, net income, margins, FCF, debt/equity
    • Uses a financial calculator tool for YoY growth and CAGR
    │
    ▼
[3] Analyst Agent — streams narrative (live)
    • 5–7 paragraph institutional-grade analysis
    • Every claim cited with exact figures and fiscal year
    │
    ▼
[4] Risk Agent — scores financial risk 0–10
    • Low / Medium / High with specific red flags
    • Uses web search to check for regulatory/legal concerns
```

---

## Tech Stack

- **LLM**: Claude Sonnet 4.6 (Anthropic) with prompt caching
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: ChromaDB with MMR retrieval
- **Live data**: yfinance (Fortune 500 financials, TTL-cached)
- **Filings**: SEC EDGAR full-text search API
- **Frontend**: Streamlit with Plotly charts (dark theme)

---

## Setup

```bash
git clone https://github.com/ashmita23/FinancialRAG_Agents
cd FinancialRAG_Agents
pip install -r requirements.txt

# Add to .env
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key      # optional — enables web search
GOOGLE_CSE_ID=your_cse_id   # optional

streamlit run streamLIT.py
```

---

## Example Questions

**Deep knowledge base (RAG):**
- What was NVIDIA's revenue and net income in FY2023?
- How did Apple's gross margin trend from 2021 to 2023?
- What drove Meta's revenue recovery in 2023?

**Live data (any Fortune 500):**
- What was Microsoft's revenue and net income in 2023?
- Compare Walmart and Costco revenue for 2023
- What are the biggest financial risks for Tesla?

**Time series:**
- Show me NVIDIA's quarterly revenue trend over the last 2 years
- What is Apple's net income history quarterly?

---

*Originally submitted at MSADS Hackathon 2024, University of Chicago — Honorable Mention.*
