import os
import json
import re
import hashlib
import requests
from typing import Generator

import anthropic
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest
from bs4 import BeautifulSoup

load_dotenv()


# =============================================================================
# RAG Helpers — plain Python, no LangChain
# =============================================================================

def _chunk_text(text: str, size: int = 400, overlap: int = 80) -> list:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
        i += size - overlap
    return chunks


def _fetch_sec_filing_text(company: str, form_type: str = "10-K") -> str:
    """Fetch the most recent SEC filing text for a company via EDGAR."""
    headers = {"User-Agent": "FinancialRAG research@example.com"}
    try:
        # Step 1: search EDGAR full-text search for accession number
        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": f'"{company}"',
                "forms": form_type,
                "dateRange": "custom",
                "startdt": "2022-01-01",
                "enddt": "2024-12-31",
            },
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        if not hits:
            return ""
        accession = hits[0]["_source"].get("id", "")
        if not accession:
            return ""

        # Step 2: build filing index URL from accession number
        # accession format: "0000320193-24-000006"
        cik = str(int(accession.split("-")[0]))  # strip leading zeros
        accession_nodash = accession.replace("-", "")
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik}/{accession_nodash}/{accession}-index.htm"
        )

        # Step 3: parse index page to find main document link
        idx_resp = requests.get(index_url, headers=headers, timeout=15)
        idx_resp.raise_for_status()
        soup = BeautifulSoup(idx_resp.text, "html.parser")
        doc_link = None
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".htm") and "index" not in href.lower():
                doc_link = (
                    f"https://www.sec.gov{href}" if href.startswith("/") else href
                )
                break
        if not doc_link:
            return ""

        # Step 4: fetch main document and extract clean text
        doc_resp = requests.get(doc_link, headers=headers, timeout=30)
        doc_resp.raise_for_status()
        doc_soup = BeautifulSoup(doc_resp.text, "html.parser")
        for tag in doc_soup(["script", "style", "table"]):
            tag.decompose()
        text = doc_soup.get_text(separator=" ", strip=True)
        # Cap at 150k chars to keep embedding fast
        return text[:150_000]
    except Exception:
        return ""


# =============================================================================
# RetrievalStack — Chroma + sentence-transformers + flashrank (no LangChain)
# =============================================================================

class RetrievalStack:
    """In-memory vector store with live SEC EDGAR fetching and flashrank reranking."""

    def __init__(self):
        self._db = chromadb.Client()
        self._col = self._db.get_or_create_collection("filings")
        self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        self._fetched: set = set()

    def fetch_and_index(self, company: str, form_type: str = "10-K") -> str:
        """Fetch a live SEC filing and index it. No-op if already indexed."""
        key = f"{company}:{form_type}"
        if key in self._fetched:
            return f"Already indexed {company} {form_type}"
        text = _fetch_sec_filing_text(company, form_type)
        if not text:
            return f"Could not fetch {form_type} for {company} from SEC EDGAR"
        chunks = _chunk_text(text, size=400, overlap=80)
        if not chunks:
            return f"No content extracted for {company}"
        embeddings = self._encoder.encode(chunks).tolist()
        ids = [
            hashlib.md5(f"{key}{i}".encode()).hexdigest()
            for i in range(len(chunks))
        ]
        metas = [
            {"company": company, "form_type": form_type, "chunk": i}
            for i in range(len(chunks))
        ]
        self._col.upsert(
            documents=chunks, embeddings=embeddings, metadatas=metas, ids=ids
        )
        self._fetched.add(key)
        return f"Indexed {len(chunks)} chunks for {company} {form_type}"

    def search(self, company: str, query: str, k: int = 8) -> list:
        """Embed query → Chroma similarity → flashrank rerank → top-k passages."""
        qemb = self._encoder.encode([query]).tolist()
        where = {"company": company} if company else None
        n = min(k * 5, 50)
        results = self._col.query(
            query_embeddings=qemb, n_results=n, where=where
        )
        docs = results["documents"][0] if results["documents"] else []
        if not docs:
            return []
        passages = [{"id": i, "text": d} for i, d in enumerate(docs)]
        reranked = self._reranker.rerank(RerankRequest(query=query, passages=passages))
        return [docs[r.id] for r in reranked[:k]]


# =============================================================================
# Tool Schemas
# =============================================================================

_TOOL_RAG_SEARCH = {
    "name": "rag_search",
    "description": (
        "Fetch the most recent 10-K filing from SEC EDGAR for any publicly traded company, "
        "index it in a vector store, and return the most relevant passages. "
        "Use this FIRST for any qualitative financial research question about any company."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {
                "type": "string",
                "description": "Company name, e.g. 'Apple', 'Microsoft', 'Tesla', 'Walmart'",
            },
            "query": {
                "type": "string",
                "description": "Specific question about the filing, include year if relevant",
            },
        },
        "required": ["company", "query"],
    },
}

_TOOL_GET_FINANCIALS = {
    "name": "get_financial_statements",
    "description": (
        "Fetch income statement, balance sheet, or cash flow for any publicly traded "
        "Fortune 500 company by ticker symbol. Use for companies NOT in the RAG knowledge base."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Stock ticker e.g. 'WMT', 'JPM', 'XOM'"},
            "statement_type": {"type": "string", "enum": ["income", "balance", "cashflow"]},
            "annual": {
                "type": "boolean",
                "description": "True for annual data (default), False for quarterly",
            },
        },
        "required": ["ticker", "statement_type"],
    },
}

_TOOL_SEC = {
    "name": "search_sec_filings",
    "description": (
        "Search SEC EDGAR for 10-K/10-Q filing text. Use for qualitative context: "
        "MD&A, risk factors, business description."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company_name": {"type": "string"},
            "query": {"type": "string"},
            "form_type": {"type": "string", "enum": ["10-K", "10-Q"]},
        },
        "required": ["company_name", "query", "form_type"],
    },
}

_TOOL_WEB_SEARCH = {
    "name": "web_search",
    "description": (
        "Search the web for current stock prices, news, analyst ratings, "
        "or recent data not available in the knowledge base."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

_TOOL_CALCULATOR = {
    "name": "financial_calculator",
    "description": (
        "Compute YoY growth, CAGR, margins, or ratios. "
        "Always use this instead of computing numbers in your head."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["yoy_growth", "cagr", "ratio", "margin"],
            },
            "values": {
                "type": "object",
                "description": (
                    "yoy_growth:{current,previous}  "
                    "cagr:{start,end,years}  "
                    "ratio/margin:{numerator,denominator}"
                ),
            },
        },
        "required": ["operation", "values"],
    },
}

_TOOL_COMPARE = {
    "name": "compare_companies",
    "description": "Compare one financial metric across multiple tickers for a given fiscal year.",
    "input_schema": {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "description": "e.g. 'revenue', 'net income'"},
            "tickers": {"type": "array", "items": {"type": "string"}},
            "year": {"type": "integer"},
        },
        "required": ["metric", "tickers", "year"],
    },
}

_TOOL_TIME_SERIES = {
    "name": "get_time_series",
    "description": (
        "Fetch historical time series for a financial metric. "
        "Use whenever the query mentions trend, WoW, MoM, YoY, chart, history, or progression."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "metric": {
                "type": "string",
                "enum": ["revenue", "net_income", "gross_profit", "price", "eps"],
            },
            "period": {
                "type": "string",
                "enum": ["quarterly", "annual"],
                "description": "Data granularity",
            },
        },
        "required": ["ticker", "metric", "period"],
    },
}

RETRIEVAL_TOOLS = [_TOOL_RAG_SEARCH, _TOOL_GET_FINANCIALS, _TOOL_SEC, _TOOL_WEB_SEARCH, _TOOL_TIME_SERIES]
METRICS_TOOLS = [_TOOL_CALCULATOR]
RISK_TOOLS = [_TOOL_CALCULATOR, _TOOL_WEB_SEARCH]

# =============================================================================
# System Prompts
# =============================================================================

RETRIEVAL_SYSTEM = """You are a financial data retrieval specialist. Your ONLY job is to gather raw financial data.
Do NOT analyze, compute growth rates, or interpret — just retrieve.
Use rag_search to fetch live SEC 10-K filings for ANY company mentioned in the query.
Also use get_financial_statements for structured numerical data (income, balance, cashflow).
For trend/WoW/MoM/YoY/history/chart questions: also call get_time_series.
For current prices or news: use web_search.
Retrieve all relevant data needed to fully answer the query, then stop.
Today is April 2026."""

METRICS_SYSTEM = """You are a financial metrics extraction specialist.
You are given raw financial data. Extract key metrics and use financial_calculator for any growth/margin computation.
Output ONLY a single valid JSON object — no markdown fences, no explanatory text:
{
  "revenue_b": <float or null>,
  "revenue_growth_pct": <float or null>,
  "net_income_b": <float or null>,
  "net_income_growth_pct": <float or null>,
  "gross_margin_pct": <float or null>,
  "operating_margin_pct": <float or null>,
  "free_cash_flow_b": <float or null>,
  "debt_to_equity": <float or null>,
  "year": <int or null>
}
Dollar values must be in billions. Percentages as plain numbers (e.g. 45.2 for 45.2%). Use null if unavailable."""

ANALYST_SYSTEM = """You are a senior financial analyst writing for institutional investors.
You have access to retrieved financial data and structured metrics provided in context.
Write a clear, insightful narrative analysis covering:
- Revenue and growth performance (cite specific figures)
- Profitability and margin analysis
- Key business changes, trends, or narrative shifts vs prior year
- Competitive positioning (if comparison data is provided)
Be concise (3-5 paragraphs). Always cite exact numbers. Focus on what matters most to investors.
Use bullet points (-) for any lists — never numbered lists.
Today is April 2026."""

RISK_SYSTEM = """You are a financial risk analyst. Score the company's financial risk on a 0-10 scale:
- Revenue declining YoY: +2 points
- Net income negative or declining more than 20%: +3 points
- Debt-to-equity ratio above 3: +2 points
- Free cash flow negative: +2 points
- Operating margin below 5%: +1 point
- Recent regulatory or legal concerns (use web_search to verify): +1 point
Scoring interpretation: 0-3 = Low | 4-6 = Medium | 7-10 = High

Output ONLY a valid JSON object — no markdown, no text outside the JSON:
{
  "score": <int 0-10>,
  "level": "<Low|Medium|High>",
  "flags": ["<specific risk factor found>", ...],
  "explanation": "<2-3 sentence narrative explaining the key risks>"
}"""


# =============================================================================
# Helpers
# =============================================================================

def _parse_json(text: str) -> dict:
    """Extract the first valid JSON object from a Claude response."""
    if not text:
        return {}
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # JSON inside code fences
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Any {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


# =============================================================================
# Shared Memory
# =============================================================================

class SharedMemory:
    """State object passed between agents."""

    def __init__(self, query: str):
        self.query = query
        self.companies: list = []
        self.tickers: list = []
        self.period: str = "2023"
        self.requires_comparison: bool = False
        self.requires_chart: bool = False
        self.is_followup: bool = False
        self.retrieved_context: str = ""
        self.time_series_data: dict = {}   # key → {ticker, metric, period, labels, values}
        self.metrics: dict = {}
        self.analysis: str = ""
        self.risk: dict = {}


# =============================================================================
# Tool Implementations (shared across all agents)
# =============================================================================

class ToolsImpl:
    """All tool implementations in one place to avoid duplication."""

    def __init__(self, retrieval_stack: RetrievalStack, google_api_key: str, google_cse_id: str):
        self.retrieval_stack = retrieval_stack
        self._google_api_key = google_api_key
        self._google_cse_id = google_cse_id
        self.time_series_cache: dict = {}

    def execute(self, name: str, inputs: dict) -> str:
        dispatch = {
            "rag_search": self._rag_search,
            "get_financial_statements": self._get_financial_statements,
            "search_sec_filings": self._search_sec_filings,
            "web_search": self._web_search,
            "financial_calculator": self._financial_calculator,
            "compare_companies": self._compare_companies,
            "get_time_series": self._get_time_series,
        }
        fn = dispatch.get(name)
        return fn(**inputs) if fn else f"Unknown tool: {name}"

    def _rag_search(self, company: str, query: str) -> str:
        status = self.retrieval_stack.fetch_and_index(company, "10-K")
        docs = self.retrieval_stack.search(company, query, k=5)
        if not docs:
            return f"{status}\nNo matching passages found for: {query}"
        return f"{status}\n\n" + "\n\n---\n\n".join(docs)

    def _get_financial_statements(
        self, ticker: str, statement_type: str, annual: bool = True
    ) -> str:
        try:
            import yfinance as yf

            co = yf.Ticker(ticker)
            pairs = {
                "income": (co.income_stmt, co.quarterly_income_stmt),
                "balance": (co.balance_sheet, co.quarterly_balance_sheet),
                "cashflow": (co.cashflow, co.quarterly_cashflow),
            }
            if statement_type not in pairs:
                return f"Unknown statement_type: {statement_type}"
            df = pairs[statement_type][0 if annual else 1]
            if df is None or df.empty:
                return f"No {statement_type} data for {ticker}"
            return df.to_string()
        except Exception as e:
            return f"Error fetching {ticker}: {e}"

    def _search_sec_filings(
        self, company_name: str, query: str, form_type: str
    ) -> str:
        try:
            url = "https://efts.sec.gov/LATEST/search-index"
            params = {
                "q": f'"{company_name}" {query}',
                "forms": form_type,
                "dateRange": "custom",
                "startdt": "2020-01-01",
                "enddt": "2024-12-31",
            }
            headers = {"User-Agent": "FinancialRAG research@example.com"}
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])[:5]
            if not hits:
                return f"No {form_type} filings for '{company_name}'"
            return "\n".join(
                f"Filed: {h.get('_source',{}).get('file_date','N/A')} | "
                f"Period: {h.get('_source',{}).get('period_of_report','N/A')} | "
                f"Entity: {h.get('_source',{}).get('entity_name','N/A')}"
                for h in hits
            )
        except Exception as e:
            return f"SEC search error: {e}"

    def _web_search(self, query: str) -> str:
        if not self._google_api_key or not self._google_cse_id:
            return "Web search unavailable: GOOGLE_API_KEY or GOOGLE_CSE_ID not set."
        try:
            from googleapiclient.discovery import build

            service = build("customsearch", "v1", developerKey=self._google_api_key)
            result = (
                service.cse().list(q=query, cx=self._google_cse_id, num=5).execute()
            )
            items = result.get("items", [])
            if not items:
                return f"No results for: {query}"
            return "\n\n".join(
                f"{i['title']}\n{i.get('snippet', '')}" for i in items
            )
        except Exception as e:
            return f"Web search error: {e}"

    def _financial_calculator(self, operation: str, values: dict) -> str:
        try:
            if operation == "yoy_growth":
                cur, prev = float(values["current"]), float(values["previous"])
                if prev == 0:
                    return "Cannot compute YoY: previous is 0"
                return f"YoY Growth: {(cur - prev) / abs(prev) * 100:.2f}%"
            elif operation == "cagr":
                start, end, years = (
                    float(values["start"]),
                    float(values["end"]),
                    float(values["years"]),
                )
                if start <= 0 or years <= 0:
                    return "Cannot compute CAGR: start and years must be > 0"
                return f"CAGR: {((end / start) ** (1 / years) - 1) * 100:.2f}%"
            elif operation in ("ratio", "margin"):
                num = float(values["numerator"])
                den = float(values["denominator"])
                if den == 0:
                    return "Cannot compute: denominator is 0"
                result = num / den * (100 if operation == "margin" else 1)
                return f"{'Margin' if operation == 'margin' else 'Ratio'}: {result:.2f}{'%' if operation == 'margin' else 'x'}"
            return f"Unknown operation: {operation}"
        except KeyError as e:
            return f"Missing value: {e}"
        except Exception as e:
            return f"Calculator error: {e}"

    def _compare_companies(self, metric: str, tickers: list, year: int) -> str:
        results = []
        ml = metric.lower().replace(" ", "")
        income_map = {
            "revenue": ["Total Revenue"],
            "netincome": ["Net Income"],
            "grossprofit": ["Gross Profit"],
            "operatingincome": ["Operating Income", "EBIT"],
        }
        for ticker in tickers:
            try:
                import yfinance as yf, pandas as pd

                df = yf.Ticker(ticker).income_stmt
                if df is None or df.empty:
                    results.append(f"{ticker}: No data")
                    continue
                year_cols = [c for c in df.columns if str(year) in str(c)]
                if not year_cols:
                    results.append(f"{ticker}: No {year} data")
                    continue
                col = year_cols[0]
                found = False
                for row_name in income_map.get(ml, [metric]):
                    if row_name in df.index:
                        val = df.loc[row_name, col]
                        if pd.notna(val):
                            results.append(f"{ticker}: ${val/1e9:.2f}B")
                            found = True
                            break
                if not found:
                    for idx in df.index:
                        if ml in str(idx).lower().replace(" ", ""):
                            val = df.loc[idx, col]
                            if pd.notna(val):
                                results.append(f"{ticker}: ${val/1e9:.2f}B [{idx}]")
                                found = True
                                break
                if not found:
                    results.append(f"{ticker}: '{metric}' not found")
            except Exception as e:
                results.append(f"{ticker}: Error — {e}")
        return f"Comparison — {metric} ({year}):\n" + "\n".join(results)

    def _get_time_series(
        self, ticker: str, metric: str, period: str = "quarterly"
    ) -> str:
        try:
            import yfinance as yf

            co = yf.Ticker(ticker)
            if metric == "price":
                interval = "1wk" if period == "quarterly" else "1mo"
                hist = co.history(period="2y", interval=interval)
                if hist.empty:
                    return "No price history available"
                series = hist["Close"].dropna()
                labels = [str(idx)[:10] for idx in series.index]
                values = [round(float(v), 2) for v in series.values]
            else:
                df = (
                    co.quarterly_income_stmt
                    if period == "quarterly"
                    else co.income_stmt
                )
                row_map = {
                    "revenue": "Total Revenue",
                    "net_income": "Net Income",
                    "gross_profit": "Gross Profit",
                    "eps": "Basic EPS",
                }
                row = row_map.get(metric, "Total Revenue")
                if df is None or df.empty or row not in df.index:
                    return f"No {metric} series for {ticker}"
                s = df.loc[row].dropna().sort_index()
                labels = [str(idx)[:10] for idx in s.index]
                values = [round(float(v) / 1e9, 3) for v in s.values]

            key = f"{ticker}_{metric}_{period}"
            self.time_series_cache[key] = {
                "ticker": ticker,
                "metric": metric,
                "period": period,
                "labels": labels,
                "values": values,
            }
            # Return text summary for Claude
            pairs = list(zip(labels, values))[-8:]
            lines = [
                f"{l}: {v:.2f}{'B' if metric != 'price' else ''}"
                for l, v in pairs
            ]
            return f"{ticker} {metric} ({period}):\n" + "\n".join(lines)
        except Exception as e:
            return f"Time series error: {e}"


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent:
    """Common tool-use loop shared by all agents."""

    def __init__(self, client: anthropic.Anthropic, tools: ToolsImpl):
        self.client = client
        self.tools = tools
        self._last_text: str = ""

    def _tool_use_loop(
        self,
        system: str,
        tool_schemas: list,
        messages: list,
        max_iter: int = 8,
    ) -> Generator:
        """
        Agentic tool-use loop. Yields tool_call / tool_result events.
        Stores final text response in self._last_text.
        """
        self._last_text = ""
        for _ in range(max_iter):
            kwargs: dict = dict(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=system,
                messages=messages,
            )
            if tool_schemas:
                kwargs["tools"] = tool_schemas

            response = self.client.messages.create(**kwargs)

            if response.stop_reason == "end_turn":
                self._last_text = "".join(
                    b.text for b in response.content if hasattr(b, "text")
                )
                return

            elif response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        yield {
                            "type": "tool_call",
                            "tool": block.name,
                            "input": block.input,
                        }
                        result = self.tools.execute(block.name, block.input)
                        yield {
                            "type": "tool_result",
                            "tool": block.name,
                            "result": result[:800],
                        }
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            else:
                self._last_text = ""
                return


# =============================================================================
# Agent 1 — Retrieval Agent
# =============================================================================

class RetrievalAgent(BaseAgent):
    def run(self, memory: SharedMemory, chat_history: list = None) -> Generator:
        history_ctx = ""
        if chat_history:
            recent = [
                m for m in chat_history[-6:]
                if isinstance(m.get("content"), str)
            ]
            if recent:
                history_ctx = (
                    "Recent conversation context:\n"
                    + "\n".join(
                        f"{m['role'].upper()}: {m['content'][:200]}" for m in recent
                    )
                    + "\n\n"
                )

        messages = [
            {
                "role": "user",
                "content": (
                    f"{history_ctx}"
                    f"Retrieve all relevant financial data for this query:\n{memory.query}\n\n"
                    f"Companies of interest: {', '.join(memory.companies) or 'unknown'}\n"
                    f"Period: {memory.period}\n"
                    f"Requires trend chart: {memory.requires_chart}"
                ),
            }
        ]
        yield from self._tool_use_loop(RETRIEVAL_SYSTEM, RETRIEVAL_TOOLS, messages)
        memory.retrieved_context = self._last_text or "No data retrieved."
        memory.time_series_data.update(self.tools.time_series_cache)


# =============================================================================
# Agent 2 — Metrics Agent
# =============================================================================

class MetricsAgent(BaseAgent):
    def run(self, memory: SharedMemory) -> Generator:
        context = memory.retrieved_context[:4000]
        messages = [
            {
                "role": "user",
                "content": (
                    f"Extract financial metrics from the data below.\n"
                    f"Query context: {memory.query}\n\n"
                    f"Raw financial data:\n{context}\n\n"
                    "Output ONLY valid JSON, no other text."
                ),
            }
        ]
        yield from self._tool_use_loop(METRICS_SYSTEM, METRICS_TOOLS, messages)
        memory.metrics = _parse_json(self._last_text)

        # Emit chart data events (populated by RetrievalAgent's get_time_series calls)
        for ts in memory.time_series_data.values():
            yield {"type": "chart_data", **ts}


# =============================================================================
# Agent 3 — Analyst Agent  (streams text token by token)
# =============================================================================

class AnalystAgent(BaseAgent):
    def run(self, memory: SharedMemory) -> Generator:
        parts = [
            f"Query: {memory.query}",
            f"Fiscal period of interest: {memory.period}",
        ]
        if memory.companies:
            parts.append(f"Companies: {', '.join(memory.companies)}")
        if memory.retrieved_context:
            parts.append(
                f"Retrieved financial data:\n{memory.retrieved_context[:3000]}"
            )
        if memory.metrics:
            parts.append(
                f"Structured metrics:\n{json.dumps(memory.metrics, indent=2)}"
            )
        if memory.requires_comparison and memory.tickers:
            # Direct tool call (no agent loop needed — comparison is deterministic)
            comp = self.tools.execute(
                "compare_companies",
                {
                    "metric": "revenue",
                    "tickers": memory.tickers,
                    "year": int(memory.period),
                },
            )
            parts.append(f"Revenue comparison data:\n{comp}")
            yield {
                "type": "tool_call",
                "tool": "compare_companies",
                "input": {"metric": "revenue", "tickers": memory.tickers, "year": memory.period},
            }
            yield {"type": "tool_result", "tool": "compare_companies", "result": comp[:800]}

        messages = [{"role": "user", "content": "\n\n".join(parts)}]

        # Stream the narrative analysis
        accumulated = ""
        with self.client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=ANALYST_SYSTEM,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                accumulated += text
                yield {"type": "text_delta", "text": text}

        memory.analysis = accumulated


# =============================================================================
# Agent 4 — Risk Agent  ⭐
# =============================================================================

class RiskAgent(BaseAgent):
    def run(self, memory: SharedMemory) -> Generator:
        context = (
            f"Company: {', '.join(memory.companies) or 'Unknown'}\n"
            f"Period: {memory.period}\n"
            f"Metrics:\n{json.dumps(memory.metrics, indent=2)}\n\n"
            f"Analyst summary (first 500 chars):\n{memory.analysis[:500]}"
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    "Score the financial risk and output ONLY valid JSON."
                ),
            }
        ]
        yield from self._tool_use_loop(RISK_SYSTEM, RISK_TOOLS, messages)
        memory.risk = _parse_json(self._last_text)


# =============================================================================
# Financial Controller (public interface — replaces FinancialOrchestrator)
# =============================================================================

# Keep backward-compatible alias so the Streamlit import still works
FinancialOrchestrator = None  # set at end of module

_TICKER_MAP = {
    "apple": ("Apple", "AAPL"),
    "amazon": ("Amazon", "AMZN"),
    "alphabet": ("Alphabet", "GOOGL"),
    "google": ("Alphabet", "GOOGL"),
    "meta": ("Meta", "META"),
    "facebook": ("Meta", "META"),
    "nvidia": ("NVIDIA", "NVDA"),
    "microsoft": ("Microsoft", "MSFT"),
    "walmart": ("Walmart", "WMT"),
    "tesla": ("Tesla", "TSLA"),
    "jpmorgan": ("JPMorgan", "JPM"),
    "exxon": ("ExxonMobil", "XOM"),
    "berkshire": ("Berkshire Hathaway", "BRK-B"),
    "johnson": ("Johnson & Johnson", "JNJ"),
    "unitedhealth": ("UnitedHealth", "UNH"),
    "costco": ("Costco", "COST"),
    "chevron": ("Chevron", "CVX"),
    "home depot": ("Home Depot", "HD"),
    "eli lilly": ("Eli Lilly", "LLY"),
}

_CHART_KEYWORDS = {
    "trend", "over time", "wow", "mom", "yoy", "quarterly", "history",
    "chart", "graph", "growth over", "changed", "progression", "timeline",
    "week over week", "month over month", "year over year",
}


class FinancialController:
    """Orchestrates 4 specialized agents with shared memory."""

    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self._retrieval_stack = RetrievalStack()
        self.tools = ToolsImpl(
            retrieval_stack=self._retrieval_stack,
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            google_cse_id=os.environ.get("GOOGLE_CSE_ID", ""),
        )
        self.retrieval_agent = RetrievalAgent(self.client, self.tools)
        self.metrics_agent   = MetricsAgent(self.client, self.tools)
        self.analyst_agent   = AnalystAgent(self.client, self.tools)
        self.risk_agent      = RiskAgent(self.client, self.tools)

    # ------------------------------------------------------------------
    # Routing (pattern-based, fast — no extra API call)
    # ------------------------------------------------------------------
    def _route(self, query: str, memory: SharedMemory) -> None:
        q = query.lower()

        # Comparison detection
        if any(kw in q for kw in ["compare", " vs ", " versus ", "difference between"]):
            memory.requires_comparison = True

        # Chart/trend detection
        if any(kw in q for kw in _CHART_KEYWORDS):
            memory.requires_chart = True

        # Company / ticker extraction
        for keyword, (company, ticker) in _TICKER_MAP.items():
            if keyword in q and company not in memory.companies:
                memory.companies.append(company)
                memory.tickers.append(ticker)

        # Year extraction
        m = re.search(r"\b(20\d{2})\b", query)
        if m:
            memory.period = m.group(1)

    _FOLLOWUP_PATTERNS = {
        "what about", "can you explain", "tell me more", "elaborate",
        "why is that", "how so", "and what", "what does that mean",
        "what caused", "go deeper", "expand on", "more detail",
    }

    def _detect_followup(self, query: str, memory: SharedMemory, chat_history: list) -> None:
        """Mark as follow-up if short, no new company, references prior context."""
        if not chat_history:
            return
        q = query.lower()
        if (
            len(query.split()) < 12
            and not memory.companies
            and any(p in q for p in self._FOLLOWUP_PATTERNS)
        ):
            memory.is_followup = True

    def _route_from_history(
        self, memory: SharedMemory, chat_history: list
    ) -> None:
        """Fallback: extract company from recent chat history if query has none."""
        if memory.companies:
            return
        for msg in reversed(chat_history[-6:]):
            if isinstance(msg.get("content"), str):
                for keyword, (company, ticker) in _TICKER_MAP.items():
                    if keyword in msg["content"].lower() and company not in memory.companies:
                        memory.companies.append(company)
                        memory.tickers.append(ticker)
                if memory.companies:
                    break

    # ------------------------------------------------------------------
    # Main generator (called by Streamlit)
    # ------------------------------------------------------------------
    def generate(self, query: str, chat_history: list) -> Generator:
        """
        Yields event dicts:
          {"type": "agent_start",  "agent": str}
          {"type": "agent_done",   "agent": str}
          {"type": "tool_call",    "tool": str, "input": dict}
          {"type": "tool_result",  "tool": str, "result": str}
          {"type": "text_delta",   "text": str}        ← streaming analyst text
          {"type": "chart_data",   "ticker": str, "metric": str, "period": str,
                                   "labels": list, "values": list}
          {"type": "metrics_json", "data": dict}
          {"type": "final",        "text": str}
          {"type": "error",        "text": str}
        """
        memory = SharedMemory(query)
        self.tools.time_series_cache.clear()

        try:
            self._route(query, memory)
            self._route_from_history(memory, chat_history)
            self._detect_followup(query, memory, chat_history)

            # Lite mode: follow-up questions skip Retrieval/Metrics/Risk
            if memory.is_followup:
                pipeline = [("analyst", self.analyst_agent)]
            else:
                pipeline = [
                    ("retrieval", self.retrieval_agent),
                    ("metrics",   self.metrics_agent),
                    ("analyst",   self.analyst_agent),
                    ("risk",      self.risk_agent),
                ]

            for agent_name, agent in pipeline:
                yield {"type": "agent_start", "agent": agent_name}
                try:
                    if agent_name == "retrieval":
                        yield from agent.run(memory, chat_history)
                    else:
                        yield from agent.run(memory)
                except Exception as e:
                    yield {
                        "type": "error",
                        "text": f"{agent_name} agent error: {e}",
                    }
                yield {"type": "agent_done", "agent": agent_name}

            # Emit structured metrics for table display
            if memory.metrics:
                yield {"type": "metrics_json", "data": memory.metrics}

            # Final compiled answer (analysis + risk section)
            yield {"type": "final", "text": self._compile_output(memory)}

        except anthropic.APIError as e:
            yield {"type": "error", "text": f"Anthropic API error: {e}"}
        except Exception as e:
            yield {"type": "error", "text": f"Unexpected error: {e}"}

    # ------------------------------------------------------------------
    # Output compiler
    # ------------------------------------------------------------------
    def _compile_output(self, memory: SharedMemory) -> str:
        return memory.analysis or "Analysis complete. No additional detail available."


# Backward-compatible alias
FinancialOrchestrator = FinancialController
