"""FinCorp Enterprise RAG Pipeline Orchestration (NexusFlow AI PoC).

Execution flow::

    {"query": ..., "intent": ...}
        -> RunnablePassthrough.assign(retrieved=<query> | RunnableParallel)
        -> RunnableLambda(route_query)  # picks ONE inference chain by intent
        -> prompt-variable mapper | PromptTemplate | ChatOpenAI | Parser
        -> grounding check (rejects placeholder or guessed values)
        -> FinancialSummary | MarketNews

Requires ``OPENAI_API_KEY`` in the environment for live LLM calls.
"""

import math
import os
import re
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

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


def build_parallel_data_loader(
    financial_fetcher: Optional[Callable[[str], str]] = None,
    news_fetcher: Optional[Callable[[str], str]] = None,
) -> RunnableParallel:
    """Build a concurrent loader over both data sources.

    By default the two mock fetchers are used; other fetchers can be injected
    (used to run the pipeline on a complete test record).

    The returned runnable expects a plain ``str`` query (the retrieval
    functions take a string, not the whole input dict) and returns
    ``{"financial_data": str, "news_data": str}``. Both branches run
    concurrently, because ``RunnableParallel`` executes its branches in a
    thread pool for sync ``invoke``.

    Args:
        financial_fetcher: Optional replacement for the financial fetcher.
        news_fetcher: Optional replacement for the news fetcher.

    Returns:
        A ``RunnableParallel`` mapping output keys to the two fetchers.
    """
    return RunnableParallel(
        financial_data=RunnableLambda(
            financial_fetcher or fetch_structured_financials
        ),
        news_data=RunnableLambda(news_fetcher or fetch_unstructured_news),
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
# 5. Grounding Check (rejects placeholder or guessed values)
# ------------------------------------------------------------------------
# A Pydantic contract guarantees the SHAPE of an answer, not its truth. When
# the source lacks a required value, a model may still fill the field with a
# placeholder (UNKNOWN, 0.0) that passes validation. This check compares the
# parsed answer with the retrieved source and rejects unsupported values.
# It is a heuristic: it matches names, quarters and amounts written with a
# "$" sign, and does not verify headlines or sentiment.

PLACEHOLDER_TEXT = frozenset(
    {
        "unknown",
        "n/a",
        "na",
        "none",
        "null",
        "tbd",
        "not available",
        "not provided",
        "not specified",
    }
)
_UNIT_IN_MILLIONS: Dict[str, float] = {
    "trillion": 1e6,
    "tn": 1e6,
    "t": 1e6,
    "billion": 1e3,
    "bn": 1e3,
    "b": 1e3,
    "million": 1.0,
    "mn": 1.0,
    "m": 1.0,
    "thousand": 1e-3,
    "k": 1e-3,
}
_AMOUNT = re.compile(
    r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(trillion|billion|million|thousand|tn|bn|mn|[tbmk])?(?![a-z0-9])",
    re.IGNORECASE,
)


class UngroundedValueError(ValueError):
    """Raised when a parsed answer contains values the source lacks."""

    def __init__(self, model_name: str, problems: List[str]) -> None:
        super().__init__(f"Rejected {model_name}: " + "; ".join(problems))
        self.problems = problems


def extract_amounts_in_millions(text: str) -> List[float]:
    """Return every ``$`` amount in ``text``, converted to millions.

    ``$45B`` -> 45000.0, ``$9.2 billion`` -> 9200.0. An amount without a
    unit is read as dollars (``$1.25`` -> 0.00000125).
    """
    amounts: List[float] = []
    for number, unit in _AMOUNT.findall(text):
        scale = _UNIT_IN_MILLIONS.get(unit.lower(), 1e-6) if unit else 1e-6
        amounts.append(float(number.replace(",", "")) * scale)
    return amounts


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, and pad with spaces for word matching."""
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def _mentioned(value: str, context: str) -> bool:
    """Return True if ``value`` appears in ``context`` as whole words."""
    needle = _normalize(value)
    return needle.strip() != "" and needle in _normalize(context)


def _quarter_in(quarter: str, context: str) -> bool:
    """Return True if both the quarter and the year appear in ``context``."""
    label, year = quarter.split()  # contract guarantees the 'QX YYYY' form
    normalized = _normalize(context)
    return f" {label.lower()} " in normalized and f" {year} " in normalized


def _amount_in(value: float, amounts: Sequence[float]) -> bool:
    """Return True if ``value`` matches a source amount (0.5% tolerance)."""
    return any(
        math.isclose(value, amount, rel_tol=0.005, abs_tol=1e-9)
        for amount in amounts
    )


def _company_problem(name: str, context: str) -> Optional[str]:
    """Describe why a company name is unsupported, or return None."""
    if name.strip().lower() in PLACEHOLDER_TEXT:
        return f"company_name={name!r} is a placeholder"
    if not _mentioned(name, context):
        return f"company_name={name!r} is not in the source or the query"
    return None


def find_ungrounded_values(
    result: BaseModel, record: str, query: str
) -> List[str]:
    """List the values of ``result`` that the source does not support.

    Args:
        result: The parsed ``FinancialSummary`` or ``MarketNews``.
        record: The retrieved text the answer must be based on.
        query: The user query (it may name the company or the quarter).

    Returns:
        One message per unsupported field; empty if everything is grounded.
    """
    context = f"{record} {query}"
    problems: List[str] = []
    if isinstance(result, FinancialSummary):
        problem = _company_problem(result.company_name, context)
        if problem:
            problems.append(problem)
        if not _quarter_in(result.quarter, context):
            problems.append(
                f"quarter={result.quarter!r}: the quarter and the year are "
                "not both in the source or the query"
            )
        amounts = extract_amounts_in_millions(record)
        for field in ("total_revenue", "net_income"):
            value = getattr(result, field)
            if not _amount_in(value, amounts):
                problems.append(
                    f"{field}={value} is not found in the source "
                    f"(amounts found, in millions: {amounts})"
                )
    elif isinstance(result, MarketNews):
        problem = _company_problem(result.company_name, context)
        if problem:
            problems.append(problem)
    return problems


def check_grounding(state: Dict[str, Any]) -> BaseModel:
    """Return the parsed answer, or raise if the source does not support it.

    Args:
        state: Pipeline state with ``query``, ``retrieved`` and ``result``.

    Raises:
        UngroundedValueError: If a placeholder or unsupported value is found.
    """
    result: BaseModel = state["result"]
    key = (
        "financial_data"
        if isinstance(result, FinancialSummary)
        else ("news_data")
    )
    problems = find_ungrounded_values(
        result, state["retrieved"][key], state["query"]
    )
    if problems:
        raise UngroundedValueError(type(result).__name__, problems)
    return result


# ------------------------------------------------------------------------
# 6. Pipeline Assembly
# ------------------------------------------------------------------------

_FINANCIAL_TEMPLATE = """You are a financial extraction engine for FinCorp.
Using ONLY the retrieved record and the user query below, extract the data
into the required JSON contract. Report monetary values as floats in
millions (e.g. $45B -> 45000.0). The quarter must look like 'Q3 2023'.
Use ONLY values stated in the record or the query. Never guess a missing
value and never use placeholders such as UNKNOWN or 0: answers containing
values that are not in the source are rejected by a downstream check.

User query: {query}

Retrieved financial record:
{financial_data}

{format_instructions}
"""

_NEWS_TEMPLATE = """You are a market news analysis engine for FinCorp.
Using ONLY the retrieved news and the user query below, produce a sentiment
overview in the required JSON contract. The sentiment score must be between
-1.0 and 1.0, and top_headlines must contain exactly 3 headlines.
Use ONLY facts stated in the news or the query. Never guess a missing value
(such as the company name) and never use placeholders such as UNKNOWN:
answers containing values that are not in the source are rejected by a
downstream check.

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


def build_fincorp_pipeline(
    llm: Optional[BaseChatModel] = None,
    financial_fetcher: Optional[Callable[[str], str]] = None,
    news_fetcher: Optional[Callable[[str], str]] = None,
    enforce_grounding: bool = True,
) -> Runnable:
    """Assemble the final LCEL orchestration pipeline.

    Args:
        llm: Optional chat model override (used for offline tests). When
            omitted, a ``ChatOpenAI`` using ``gpt-4o-mini`` is created and
            reads ``OPENAI_API_KEY`` from the environment. For local
            development with another OpenAI-compatible provider, set the
            optional ``LLM_MODEL`` and ``LLM_BASE_URL`` env vars.
        financial_fetcher: Optional replacement for the mock financial
            fetcher.
        news_fetcher: Optional replacement for the mock news fetcher.
        enforce_grounding: If True (default), reject answers whose values are
            placeholders or not supported by the retrieved source.

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
    if enforce_grounding:
        # Keep the pipeline state next to the parsed answer so the check can
        # compare the answer with the retrieved source.
        chains = {
            intent: RunnablePassthrough.assign(result=chain)
            | RunnableLambda(check_grounding)
            for intent, chain in chains.items()
        }

    # 4. Assemble the LCEL chain
    # assign() keeps "query" and "intent" and adds "retrieved"; the loader
    # receives only inputs["query"] (a str), never the whole dict.
    retrieval_step = RunnablePassthrough.assign(
        retrieved=RunnableLambda(lambda state: state["query"])
        | build_parallel_data_loader(financial_fetcher, news_fetcher)
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

    query = (
        "Impact of the new generative AI orchestration on Q4 profit margins"
    )
    for intent in SUPPORTED_INTENTS:
        test_input = {
            "query": query,
            "intent": intent,  # This field is utilized by the router logic
        }
        print(f"\n--- intent={intent!r} ---")
        try:
            result = pipeline.invoke(test_input)
            print("Pipeline Execution Successful!")
            print(result)
        except UngroundedValueError as e:
            print(f"Pipeline REJECTED the answer (grounding check): {e}")
        except Exception as e:
            print(f"Pipeline Execution Failed: {e}")
