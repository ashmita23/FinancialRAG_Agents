import os
import json
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

# ---------------------------------------------------------------------------
# Tool schemas for Claude native tool use
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "rag_search",
        "description": (
            "Search the deep financial knowledge base for Amazon, Apple, Alphabet, Meta, "
            "or NVIDIA (2020-2024). Includes vision-extracted table and chart summaries. "
            "Use this FIRST when the question is about one of those 5 companies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company": {
                    "type": "string",
                    "enum": ["Amazon", "Apple", "Alphabet", "Meta", "NVIDIA"],
                    "description": "The company to search financial data for",
                },
                "query": {
                    "type": "string",
                    "description": "Specific financial question, include the year if relevant",
                },
            },
            "required": ["company", "query"],
        },
    },
    {
        "name": "get_financial_statements",
        "description": (
            "Fetch structured financial data (income statement, balance sheet, or cash flow) "
            "for any publicly traded Fortune 500 company via its ticker symbol. "
            "Covers 3-5 years of history. Use for companies NOT in the RAG knowledge base."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'WMT', 'JPM', 'XOM'",
                },
                "statement_type": {
                    "type": "string",
                    "enum": ["income", "balance", "cashflow"],
                    "description": "Which financial statement to retrieve",
                },
                "annual": {
                    "type": "boolean",
                    "description": "True for annual data, False for quarterly. Default: true",
                },
            },
            "required": ["ticker", "statement_type"],
        },
    },
    {
        "name": "search_sec_filings",
        "description": (
            "Search SEC EDGAR for 10-K annual or 10-Q quarterly filing text for any public "
            "company. Use for management discussion, risk factors, or qualitative context "
            "not available in financial tables."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "Full company name or ticker symbol",
                },
                "query": {
                    "type": "string",
                    "description": "What to search for in the filings",
                },
                "form_type": {
                    "type": "string",
                    "enum": ["10-K", "10-Q"],
                    "description": "Filing type to search",
                },
            },
            "required": ["company_name", "query", "form_type"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current stock prices, analyst ratings, recent news, "
            "or any data more recent than the knowledge base. "
            "Use after structured tools when live or very recent data is needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query including company name and specific metric",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "financial_calculator",
        "description": (
            "Perform financial calculations: year-over-year growth, CAGR, margins, or ratios. "
            "Always use this tool instead of computing numbers in your head."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["yoy_growth", "cagr", "ratio", "margin"],
                    "description": "The calculation to perform",
                },
                "values": {
                    "type": "object",
                    "description": (
                        "Named numeric inputs. "
                        "yoy_growth: {current, previous}. "
                        "cagr: {start, end, years}. "
                        "ratio or margin: {numerator, denominator}."
                    ),
                },
            },
            "required": ["operation", "values"],
        },
    },
    {
        "name": "compare_companies",
        "description": (
            "Compare a single financial metric across multiple companies for a given fiscal year. "
            "Fetches data for each company and returns a structured side-by-side comparison."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "e.g. 'revenue', 'net income', 'gross margin', 'operating income'",
                },
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock ticker symbols",
                },
                "year": {
                    "type": "integer",
                    "description": "Fiscal year to compare",
                },
            },
            "required": ["metric", "tickers", "year"],
        },
    },
]

SYSTEM_PROMPT = """You are an expert financial analyst with access to financial data for Fortune 500 companies.

Data tier priority:
- For Amazon (AMZN), Apple (AAPL), Alphabet (GOOGL), Meta (META), or NVIDIA (NVDA): use rag_search FIRST — it contains richer context including vision-extracted tables and charts from annual and quarterly reports (2020-2024).
- For all other Fortune 500 public companies: use get_financial_statements for numbers and search_sec_filings for qualitative context.
- For current prices, recent news, or data after 2024: use web_search.

Rules:
- Always use financial_calculator for any numeric derivation (growth, margins, ratios). Never compute in your head.
- Use compare_companies when the user asks to compare multiple companies on one metric.
- State only facts sourced from tool results. Never hallucinate figures.
- Format monetary values with $ and clearly state billions (B) or millions (M).
- Express percentages with the % symbol.
- Today's date is April 2026. The most recent complete fiscal year in the knowledge base is 2023."""


class FinancialOrchestrator:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._build_retrieval_stack()
        self._setup_google_search()

    # ------------------------------------------------------------------
    # Retrieval stack (Chroma + ColBERT — unchanged from original)
    # ------------------------------------------------------------------
    def _build_retrieval_stack(self):
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

        # Load .pkl files — gracefully handle placeholders or missing files
        for pkl_path, attr in [
            ("allPDFDocs.pkl", "docs"),
            ("allPDFSums (1).pkl", "sums"),
        ]:
            try:
                data = joblib.load(pkl_path)
                if isinstance(data, list) and len(data) > 0:
                    setattr(self, attr, data)
                else:
                    setattr(self, attr, [])
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

    def _setup_google_search(self):
        self._google_api_key = os.environ.get("GOOGLE_API_KEY", "")
        self._google_cse_id = os.environ.get("GOOGLE_CSE_ID", "")

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------
    def _tool_rag_search(self, company: str, query: str) -> str:
        full_query = f"{company} {query}"
        try:
            docs = self.retriever.invoke(full_query)
            if not docs:
                return f"No results found in knowledge base for: {full_query}"
            return "\n\n---\n\n".join(d.page_content for d in docs[:5])
        except Exception as e:
            return f"RAG search error: {e}"

    def _tool_get_financial_statements(
        self, ticker: str, statement_type: str, annual: bool = True
    ) -> str:
        try:
            import yfinance as yf

            co = yf.Ticker(ticker)
            if statement_type == "income":
                df = co.income_stmt if annual else co.quarterly_income_stmt
            elif statement_type == "balance":
                df = co.balance_sheet if annual else co.quarterly_balance_sheet
            elif statement_type == "cashflow":
                df = co.cashflow if annual else co.quarterly_cashflow
            else:
                return f"Unknown statement_type: {statement_type}"

            if df is None or df.empty:
                return f"No {statement_type} data available for {ticker}"
            return df.to_string()
        except Exception as e:
            return f"Error fetching financial statements for {ticker}: {e}"

    def _tool_search_sec_filings(
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
                return f"No {form_type} filings found for '{company_name}' matching '{query}'"
            results = []
            for h in hits:
                src = h.get("_source", {})
                results.append(
                    f"Filed: {src.get('file_date', 'N/A')} | "
                    f"Period: {src.get('period_of_report', 'N/A')} | "
                    f"Entity: {src.get('entity_name', 'N/A')} | "
                    f"Form: {src.get('form_type', form_type)}"
                )
            return "\n".join(results)
        except Exception as e:
            return f"SEC EDGAR search error: {e}"

    def _tool_web_search(self, query: str) -> str:
        if not self._google_api_key or not self._google_cse_id:
            return "Web search unavailable: GOOGLE_API_KEY or GOOGLE_CSE_ID not configured."
        try:
            from googleapiclient.discovery import build

            service = build("customsearch", "v1", developerKey=self._google_api_key)
            result = (
                service.cse()
                .list(q=query, cx=self._google_cse_id, num=5)
                .execute()
            )
            items = result.get("items", [])
            if not items:
                return f"No web results found for: {query}"
            return "\n\n".join(
                f"{item['title']}\n{item.get('snippet', '')}" for item in items
            )
        except Exception as e:
            return f"Web search error: {e}"

    def _tool_financial_calculator(self, operation: str, values: dict) -> str:
        try:
            if operation == "yoy_growth":
                current = float(values["current"])
                previous = float(values["previous"])
                if previous == 0:
                    return "Cannot compute YoY growth: previous value is 0"
                growth = (current - previous) / abs(previous) * 100
                return f"YoY Growth: {growth:.2f}%"

            elif operation == "cagr":
                start = float(values["start"])
                end = float(values["end"])
                years = float(values["years"])
                if start <= 0 or years <= 0:
                    return "Cannot compute CAGR: start and years must be positive"
                cagr = ((end / start) ** (1 / years) - 1) * 100
                return f"CAGR over {years:.0f} years: {cagr:.2f}%"

            elif operation == "ratio":
                numerator = float(values["numerator"])
                denominator = float(values["denominator"])
                if denominator == 0:
                    return "Cannot compute ratio: denominator is 0"
                return f"Ratio: {numerator / denominator:.4f}x"

            elif operation == "margin":
                numerator = float(values["numerator"])
                denominator = float(values["denominator"])
                if denominator == 0:
                    return "Cannot compute margin: denominator is 0"
                return f"Margin: {numerator / denominator * 100:.2f}%"

            else:
                return f"Unknown operation: {operation}"
        except KeyError as e:
            return f"Missing value for {operation}: {e}"
        except Exception as e:
            return f"Calculator error: {e}"

    def _tool_compare_companies(
        self, metric: str, tickers: list, year: int
    ) -> str:
        results = []
        metric_lower = metric.lower().replace(" ", "")

        # Map common metric names to yfinance income statement row names
        income_keywords = {
            "revenue": ["Total Revenue", "TotalRevenue"],
            "netincome": ["Net Income", "NetIncome"],
            "grossprofit": ["Gross Profit", "GrossProfit"],
            "operatingincome": ["Operating Income", "OperatingIncome", "EBIT"],
            "ebitda": ["EBITDA", "Normalized EBITDA"],
        }

        for ticker in tickers:
            try:
                import yfinance as yf
                import pandas as pd

                co = yf.Ticker(ticker)
                df = co.income_stmt

                if df is None or df.empty:
                    results.append(f"{ticker}: No data available")
                    continue

                # Filter columns to the requested year
                year_cols = [c for c in df.columns if str(year) in str(c)]
                if not year_cols:
                    results.append(f"{ticker}: No data for {year}")
                    continue

                col = year_cols[0]

                # Find matching row
                found = False
                candidate_rows = income_keywords.get(metric_lower, [metric])
                for row_name in candidate_rows:
                    if row_name in df.index:
                        val = df.loc[row_name, col]
                        if pd.notna(val):
                            val_b = val / 1e9
                            results.append(f"{ticker} ({year}): ${val_b:.2f}B")
                            found = True
                            break

                if not found:
                    # Fall back: search index case-insensitively
                    for idx in df.index:
                        if metric_lower in str(idx).lower().replace(" ", ""):
                            val = df.loc[idx, col]
                            if pd.notna(val):
                                val_b = val / 1e9
                                results.append(f"{ticker} ({year}): ${val_b:.2f}B [{idx}]")
                                found = True
                                break
                    if not found:
                        results.append(f"{ticker}: '{metric}' not found in income statement")

            except Exception as e:
                results.append(f"{ticker}: Error — {e}")

        if not results:
            return "No data retrieved for any ticker"
        return f"Comparison — {metric} ({year}):\n" + "\n".join(results)

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------
    def _execute_tool(self, name: str, inputs: dict) -> str:
        if name == "rag_search":
            return self._tool_rag_search(inputs["company"], inputs["query"])
        elif name == "get_financial_statements":
            return self._tool_get_financial_statements(
                inputs["ticker"],
                inputs["statement_type"],
                inputs.get("annual", True),
            )
        elif name == "search_sec_filings":
            return self._tool_search_sec_filings(
                inputs["company_name"], inputs["query"], inputs["form_type"]
            )
        elif name == "web_search":
            return self._tool_web_search(inputs["query"])
        elif name == "financial_calculator":
            return self._tool_financial_calculator(inputs["operation"], inputs["values"])
        elif name == "compare_companies":
            return self._tool_compare_companies(
                inputs["metric"], inputs["tickers"], inputs["year"]
            )
        else:
            return f"Unknown tool: {name}"

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------
    def generate(self, query: str, chat_history: list) -> Generator:
        """
        Yields dicts:
          {"type": "tool_call",   "tool": str, "input": dict}
          {"type": "tool_result", "tool": str, "result": str}
          {"type": "final",       "text": str}
          {"type": "error",       "text": str}
        """
        messages = list(chat_history) + [{"role": "user", "content": query}]
        max_iterations = 10

        try:
            for _ in range(max_iterations):
                response = self.client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    tools=TOOLS,
                    messages=messages,
                )

                if response.stop_reason == "end_turn":
                    for block in response.content:
                        if hasattr(block, "text"):
                            yield {"type": "final", "text": block.text}
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
                            result_str = self._execute_tool(block.name, block.input)
                            yield {
                                "type": "tool_result",
                                "tool": block.name,
                                "result": result_str[:800],
                            }
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result_str,
                                }
                            )

                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

                else:
                    yield {
                        "type": "error",
                        "text": f"Unexpected stop reason: {response.stop_reason}",
                    }
                    return

            yield {"type": "error", "text": "Max iterations reached without a final answer."}

        except anthropic.APIError as e:
            yield {"type": "error", "text": f"Anthropic API error: {e}"}
        except Exception as e:
            yield {"type": "error", "text": f"Unexpected error: {e}"}
