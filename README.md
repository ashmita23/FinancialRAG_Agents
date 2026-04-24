# Financial RAG Agents — Fortune 500 Financial Analyst

> **Honorable Mention, MSADS Hackathon 2024 · University of Chicago**

A 4-agent AI system for financial analysis of any Fortune 500 company. Ask a question, get a concise answer with cited figures, structured metrics, a risk score, and trend charts.

---

## Before & After

**v1 (Hackathon, 2024)** — Single RAG pipeline over 4 company PDFs. Local Qwen2 LLM, ColBERT reranking, basic Streamlit UI, raw text output, ~45s response time.

**v3 (Current)** — 4 specialized agents with shared memory. Covers any Fortune 500 company via a private RAG knowledge base (1,970 document summaries) + live yfinance data + SEC EDGAR filings. Streaming output, structured metrics table, 0–10 risk score, Plotly charts, multi-turn memory, stop button. ~15s response time.

| | v1 | v3 |
|---|---|---|
| LLM | Qwen2 (local) | Claude Sonnet 4.6 |
| Coverage | 4 companies (PDF only) | 5 deep + any Fortune 500 live |
| Response time | ~45s | ~15s |
| Output | Raw text | Narrative + metrics + risk + charts |
| Multi-turn | No | Yes |

---

## How It Works

```
[1] Retrieval Agent  →  RAG (Amazon/Apple/Alphabet/Meta/NVIDIA 2020–2024)
                        + yfinance (any ticker, TTL-cached)
                        + SEC EDGAR 10-K/10-Q
                        + web search (optional)
[2] Metrics Agent   →  Extracts structured JSON (revenue, margins, FCF, debt/equity)
[3] Analyst Agent   →  Streams concise narrative with cited figures
[4] Risk Agent      →  Scores 0–10 with specific red flags
```

---

## Setup

```bash
git clone https://github.com/ashmita23/FinancialRAG_Agents
cd FinancialRAG_Agents
pip install -r requirements.txt
# add ANTHROPIC_API_KEY to .env
streamlit run streamLIT.py
```

---

*MSADS Hackathon 2024, University of Chicago — Honorable Mention.*
