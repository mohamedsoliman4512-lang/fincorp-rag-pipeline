"""FinCorp Enterprise RAG Pipeline Orchestration (NexusFlow AI PoC).

Execution flow::

    {"query": ..., "intent": ...}
        -> RunnablePassthrough.assign(retrieved=<query> | RunnableParallel)
        -> RunnableLambda(route_query)  # picks ONE inference chain by intent
        -> prompt-variable mapper | PromptTemplate | ChatOpenAI | Parser
        -> FinancialSummary | MarketNews

Requires ``OPENAI_API_KEY`` in the environment for live LLM calls.
"""

import os
from functools import partial
from typing import Any, Dict, List, Mapping, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

try:  # Optional: lets VS Code runs read OPENAI_API_KEY from a .env file.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

MODEL_NAME = "gpt-4o-mini"
SUPPORTED_INTENTS = ("financial", "news")

# ------------------------------------------------------------------------
# 1. Pydantic Models (Strict Data Contracts)
# ------------------------------------------------------------------------


class FinancialSummary(BaseModel):
    """Strict contract for the ``financial_summary`` schema.

    Acts as a structural contract that removes noise from LLM outputs for
    the downstream reporting tools used by FinCorp.
    """

    company_name: str = Field(
        ...,
        description=(
            "The official ticker or registered name of the entity "
            "(e.g. 'FinCorp Inc.', 'AAPL')."
        ),
    )
    quarter: str = Field(
        ...,
        pattern=r"^Q[1-4] \d{4}$",
        description=(
            "The fiscal quarter and year referenced in the query, strictly "
            "in the format 'QX YYYY' (e.g. 'Q3 2023')."
        ),
    )
    total_revenue: float = Field(
        ...,
        description=(
            "Total consolidated revenue for the quarter as a numeric float "
            "in millions (e.g. 12450.50)."
        ),
    )
    net_income: float = Field(
        ...,
        description=(
            "Bottom-line net income for the quarter as a numeric float "
            "in millions (e.g. 3120.75)."
        ),
    )


class MarketNews(BaseModel):
    """Strict contract for the ``market_news`` schema.

    Ensures unstructured news is parsed into a reliable, consistent format.
    """

    company_name: str = Field(
        ...,
        description=(
            "The target company being analyzed in the news aggregator."
        ),
    )
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description=(
            "Sentiment derived from the aggregated news, a float between "
            "-1.0 (highly negative) and 1.0 (highly positive)."
        ),
    )
    top_headlines: List[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description=(
            "Exactly the top 3 most relevant news headlines impacting the "
            "company's market perception."
        ),
    )


# ------------------------------------------------------------------------
# 2. Mock Data Retrieval Functions
# ------------------------------------------------------------------------


def fetch_structured_financials(query: str) -> str:
    """Mock loader simulating highly structured financial database queries."""
    return (
        "Financial Data DB Record: Q3 EPS $1.25, Net Revenue $45B. "
        f"Associated query: '{query}'."
    )


def fetch_unstructured_news(query: str) -> str:
    """Mock loader simulating unstructured market news wire extraction."""
    return (
        "News Wire Extract: Analysts upgrade target stock amidst strong AI "
        f"integration rally. Associated query: '{query}'."
    )


# ------------------------------------------------------------------------
# 3. Parallel Data Aggregation
# ------------------------------------------------------------------------


def build_parallel_data_loader() -> RunnableParallel:
    """Build a concurrent loader over both mock data sources.

    The returned runnable expects a plain ``str`` query (the retrieval
    functions take a string, not the whole input dict) and returns
    ``{"financial_data": str, "news_data": str}``. Both branches run
    concurrently, because ``RunnableParallel`` executes its branches in a
    thread pool for sync ``invoke``.

    Returns:
        A ``RunnableParallel`` mapping output keys to the two fetchers.
    """
    return RunnableParallel(
        financial_data=RunnableLambda(fetch_structured_financials),
        news_data=RunnableLambda(fetch_unstructured_news),
    )


# ------------------------------------------------------------------------
# 4. Conditional Routing
# ------------------------------------------------------------------------


def route_query(
    inputs: Dict[str, Any], chains: Mapping[str, Runnable]
) -> Runnable:
    """Select the inference chain that matches ``inputs["intent"]``.

    The signature gained a ``chains`` parameter so the router can reach the
    chains built inside ``build_fincorp_pipeline`` without globals. The
    pipeline binds it with ``functools.partial``, so the wrapped callable is
    still a one-argument ``inputs -> Runnable`` function.

    Args:
        inputs: Pipeline state; must contain an ``"intent"`` key.
        chains: Mapping of intent name to its ``Prompt | LLM | Parser`` chain.

    Returns:
        The single chain to execute. ``RunnableLambda`` invokes a returned
        ``Runnable`` with the same input, so only this branch ever runs.

    Raises:
        ValueError: If the intent is missing or not one of ``chains``.
    """
    intent = inputs.get("intent")
    key = intent.strip().lower() if isinstance(intent, str) else None
    if key is None or key not in chains:
        supported = ", ".join(sorted(chains))
        raise ValueError(
            f"Unsupported or missing intent: {intent!r}. "
            f"Supported intents: {supported}."
        )
    return chains[key]


# ------------------------------------------------------------------------
# 5. Pipeline Assembly
# ------------------------------------------------------------------------

_FINANCIAL_TEMPLATE = """You are a financial extraction engine for FinCorp.
Using ONLY the retrieved record and the user query below, extract the data
into the required JSON contract. Report monetary values as floats in
millions (e.g. $45B -> 45000.0). The quarter must look like 'Q3 2023'.

User query: {query}

Retrieved financial record:
{financial_data}

{format_instructions}
"""

_NEWS_TEMPLATE = """You are a market news analysis engine for FinCorp.
Using ONLY the retrieved news and the user query below, produce a sentiment
overview in the required JSON contract. The sentiment score must be between
-1.0 and 1.0, and top_headlines must contain exactly 3 headlines.

User query: {query}

Retrieved news:
{news_data}

{format_instructions}
"""


def _financial_prompt_inputs(state: Dict[str, Any]) -> Dict[str, str]:
    """Flatten pipeline state into the financial prompt's variables."""
    return {
        "query": state["query"],
        "financial_data": state["retrieved"]["financial_data"],
    }


def _news_prompt_inputs(state: Dict[str, Any]) -> Dict[str, str]:
    """Flatten pipeline state into the news prompt's variables."""
    return {
        "query": state["query"],
        "news_data": state["retrieved"]["news_data"],
    }


def build_fincorp_pipeline(llm: Optional[BaseChatModel] = None) -> Runnable:
    """Assemble the final LCEL orchestration pipeline.

    Args:
        llm: Optional chat model override (used for offline tests). When
            omitted, a ``ChatOpenAI`` using ``gpt-4o-mini`` is created and
            reads ``OPENAI_API_KEY`` from the environment. For local
            development with another OpenAI-compatible provider, set the
            optional ``LLM_MODEL`` and ``LLM_BASE_URL`` env vars.

    Returns:
        A runnable accepting ``{"query": str, "intent": str}`` and returning
        a ``FinancialSummary`` or ``MarketNews`` instance.
    """
    # 1. Initialize the LLM (gpt-4o-mini); temperature 0 for stable extraction
    model: BaseChatModel = llm or ChatOpenAI(
        model=os.environ.get("LLM_MODEL", MODEL_NAME),
        temperature=0,
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )

    # 2. Initialize PydanticOutputParsers for both models
    financial_parser = PydanticOutputParser(pydantic_object=FinancialSummary)
    news_parser = PydanticOutputParser(pydantic_object=MarketNews)

    # 3. Define PromptTemplates for financial summaries and market news
    financial_prompt = PromptTemplate(
        template=_FINANCIAL_TEMPLATE,
        input_variables=["query", "financial_data"],
        partial_variables={
            "format_instructions": financial_parser.get_format_instructions()
        },
    )
    news_prompt = PromptTemplate(
        template=_NEWS_TEMPLATE,
        input_variables=["query", "news_data"],
        partial_variables={
            "format_instructions": news_parser.get_format_instructions()
        },
    )

    chains: Dict[str, Runnable] = {
        "financial": RunnableLambda(_financial_prompt_inputs)
        | financial_prompt
        | model
        | financial_parser,
        "news": RunnableLambda(_news_prompt_inputs)
        | news_prompt
        | model
        | news_parser,
    }

    # 4. Assemble the LCEL chain
    # assign() keeps "query" and "intent" and adds "retrieved"; the loader
    # receives only inputs["query"] (a str), never the whole dict.
    retrieval_step = RunnablePassthrough.assign(
        retrieved=RunnableLambda(lambda state: state["query"])
        | build_parallel_data_loader()
    )
    router = RunnableLambda(partial(route_query, chains=chains))

    return retrieval_step | router


if __name__ == "__main__":
    # OPENAI_API_KEY must be set in the environment or in a local .env file
    if load_dotenv is not None:
        load_dotenv()

    print("Starting NexusFlow AI - FinCorp RAG Architecture PoC...")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (env var or .env).")

    pipeline = build_fincorp_pipeline()

    test_input = {
        "query": (
            "Impact of the new generative AI orchestration "
            "on Q4 profit margins"
        ),
        "intent": "financial",  # This field is utilized by the router logic
    }

    try:
        result = pipeline.invoke(test_input)
        print("Pipeline Execution Successful!")
        print(result)
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")
