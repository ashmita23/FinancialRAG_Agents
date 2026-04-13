import os
import json
import re
import joblib
import requests
from typing import Generator

import anthropic
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain.retrievers import ContextualCompressionRetriever
from ragatouille import RAGPretrainedModel

load_dotenv()


# =============================================================================
# Tool Schemas
# =============================================================================

_TOOL_RAG_SEARCH = {
    "name": "rag_search",
    "description": (
        "Search deep financial knowledge base for Amazon, Apple, Alphabet, Meta, or NVIDIA "
        "(2020-2024). Includes vision-extracted table and chart summaries. Use FIRST for these 5 companies."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {
                "type": "string",
                "enum": ["Amazon", "Apple", "Alphabet", "Meta", "NVIDIA"],
            },
            "query": {
                "type": "string",
                "description": "Specific financial question; include year if relevant",
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
For Amazon, Apple, Alphabet, Meta, or NVIDIA: use rag_search first.
For all other companies: use get_financial_statements and search_sec_filings.
For trend/WoW/MoM/YoY/history/chart questions: also call get_time_series.
For current prices or news: use web_search.
Retrieve all relevant data needed to fully answer the query, then stop.
Today is April 2026. Most recent complete fiscal year in knowledge base: 2023."""

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
1. Revenue and growth performance (cite specific figures)
2. Profitability and margin analysis
3. Key business changes, trends, or narrative shifts vs prior year
4. Competitive positioning (if comparison data is provided)
Be concise (3-5 paragraphs). Always cite exact numbers. Focus on what matters most to investors.
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

    def __init__(self, retriever, google_api_key: str, google_cse_id: str):
        self.retriever = retriever
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
        try:
            docs = self.retriever.invoke(f"{company} {query}")
            if not docs:
                return f"No results found for: {company} {query}"
            return "\n\n---\n\n".join(d.page_content for d in docs[:5])
        except Exception as e:
            return f"RAG search error: {e}"

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
        self._build_retrieval_stack()
        self.tools = ToolsImpl(
            retriever=self.retriever,
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            google_cse_id=os.environ.get("GOOGLE_CSE_ID", ""),
        )
        self.retrieval_agent = RetrievalAgent(self.client, self.tools)
        self.metrics_agent = MetricsAgent(self.client, self.tools)
        self.analyst_agent = AnalystAgent(self.client, self.tools)
        self.risk_agent = RiskAgent(self.client, self.tools)

    # ------------------------------------------------------------------
    # Retrieval stack (Chroma + ColBERT — unchanged from v1)
    # ------------------------------------------------------------------
    def _build_retrieval_stack(self) -> None:
        self.vectorstore = Chroma(
            collection_name="docsAndSums",
            embedding_function=HuggingFaceInstructEmbeddings(
                model_name="hkunlp/instructor-xl",
                query_instruction=(
                    "Embed these chunks and questions for retrieval from Financial Report "
                    "Documents. The current year is 2026."
                ),
            ),
            persist_directory="chromaDocs",
        )
        self.byte_store = InMemoryByteStore()
        self.base_retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            byte_store=self.byte_store,
            id_key="id",
            search_type="similarity",
            search_kwargs={"k": 70},
        )
        for pkl_path, attr in [
            ("allPDFDocs.pkl", "docs"),
            ("allPDFSums (1).pkl", "sums"),
        ]:
            try:
                data = joblib.load(pkl_path)
                setattr(self, attr, data if isinstance(data, list) and data else [])
            except Exception:
                setattr(self, attr, [])

        all_docs = self.docs + self.sums
        if all_docs:
            ids = [d.metadata["id"] for d in all_docs]
            self.base_retriever.docstore.mset(list(zip(ids, all_docs)))

        colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        self.retriever = ContextualCompressionRetriever(
            base_compressor=colbert.as_langchain_document_compressor(k=10),
            base_retriever=self.base_retriever,
        )

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

            pipeline = [
                ("retrieval", self.retrieval_agent),
                ("metrics", self.metrics_agent),
                ("analyst", self.analyst_agent),
                ("risk", self.risk_agent),
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
        parts = []
        if memory.analysis:
            parts.append(memory.analysis)

        if memory.risk:
            score = memory.risk.get("score", "N/A")
            level = memory.risk.get("level", "Unknown")
            flags = memory.risk.get("flags", [])
            explanation = memory.risk.get("explanation", "")

            risk_md = f"\n\n---\n\n### Risk Assessment: {level} ({score}/10)\n"
            if flags:
                risk_md += "**Key risk factors:** " + " · ".join(flags) + "\n\n"
            risk_md += explanation
            parts.append(risk_md)

        return "".join(parts) or "Analysis complete. No additional detail available."


# Backward-compatible alias
FinancialOrchestrator = FinancialController
