"""Offline structural tests (no API key needed) plus an opt-in live test."""

import os
import threading
import time
from typing import Any, Dict, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from pydantic import Field, ValidationError

import workflow_architecture as wa
from workflow_architecture import (
    FinancialSummary,
    MarketNews,
    build_fincorp_pipeline,
    build_parallel_data_loader,
)

FINANCIAL_JSON = (
    '{"company_name": "FinCorp Inc.", "quarter": "Q3 2023", '
    '"total_revenue": 45000.0, "net_income": 9000.0}'
)
NEWS_JSON = (
    '{"company_name": "FinCorp Inc.", "sentiment_score": 0.7, '
    '"top_headlines": ["h1", "h2", "h3"]}'
)
TEST_QUERY = (
    "Impact of the new generative AI orchestration on Q4 profit margins"
)


class RecordingFakeLLM(FakeListChatModel):
    """Fake chat model that records every prompt it receives."""

    seen: List[str] = Field(default_factory=list)

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        self.seen.append(str(messages[-1].content))
        return super()._call(messages, stop, run_manager, **kwargs)


def make_pipeline(response: str):
    """Return (pipeline, fake_llm) wired to a recording fake model."""
    llm = RecordingFakeLLM(responses=[response])
    return build_fincorp_pipeline(llm=llm), llm


def test_financial_intent_returns_financial_model_and_one_llm_call() -> None:
    pipeline, llm = make_pipeline(FINANCIAL_JSON)
    result = pipeline.invoke({"query": TEST_QUERY, "intent": "financial"})
    assert isinstance(result, FinancialSummary)
    assert len(llm.seen) == 1  # only the selected chain executed
    prompt = llm.seen[0]
    assert TEST_QUERY in prompt
    assert "Financial Data DB Record" in prompt
    assert "News Wire Extract" not in prompt
    assert "'" + TEST_QUERY + "'" in prompt  # query string, not the dict


def test_news_intent_returns_news_model_and_one_llm_call() -> None:
    pipeline, llm = make_pipeline(NEWS_JSON)
    result = pipeline.invoke({"query": TEST_QUERY, "intent": "news"})
    assert isinstance(result, MarketNews)
    assert len(llm.seen) == 1
    assert "News Wire Extract" in llm.seen[0]
    assert "Financial Data DB Record" not in llm.seen[0]


def test_invalid_and_missing_intent_raise_clear_error() -> None:
    pipeline, llm = make_pipeline(FINANCIAL_JSON)
    with pytest.raises(ValueError, match="Unsupported or missing intent"):
        pipeline.invoke({"query": "x", "intent": "weather"})
    with pytest.raises(ValueError, match="Unsupported or missing intent"):
        pipeline.invoke({"query": "x"})
    assert llm.seen == []  # the LLM was never called


def test_parallel_loader_output_shape_and_query_extraction() -> None:
    out: Dict[str, str] = build_parallel_data_loader().invoke("hello")
    assert set(out) == {"financial_data", "news_data"}
    assert "'hello'" in out["financial_data"]
    assert "'hello'" in out["news_data"]


def test_parallel_loader_runs_branches_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threads: List[int] = []
    delay = 0.5

    def slow(name: str):
        def fetch(query: str) -> str:
            threads.append(threading.get_ident())
            time.sleep(delay)
            return f"{name}:{query}"

        return fetch

    monkeypatch.setattr(wa, "fetch_structured_financials", slow("fin"))
    monkeypatch.setattr(wa, "fetch_unstructured_news", slow("news"))
    loader = wa.build_parallel_data_loader()
    start = time.perf_counter()
    out = loader.invoke("q")
    elapsed = time.perf_counter() - start
    assert out == {"financial_data": "fin:q", "news_data": "news:q"}
    assert elapsed < delay * 1.8  # sequential would be >= 2 * delay
    assert len(set(threads)) == 2


def test_input_state_is_preserved_by_assign() -> None:
    pipeline, llm = make_pipeline(FINANCIAL_JSON)
    pipeline.invoke(
        {"query": TEST_QUERY, "intent": "FINANCIAL "}
    )  # normalized
    assert len(llm.seen) == 1


def test_schema_constraints() -> None:
    with pytest.raises(ValidationError):
        FinancialSummary(
            company_name="A",
            quarter="2023 Q3",
            total_revenue=1.0,
            net_income=1.0,
        )
    with pytest.raises(ValidationError):
        MarketNews(
            company_name="A",
            sentiment_score=1.5,
            top_headlines=["a", "b", "c"],
        )
    with pytest.raises(ValidationError):
        MarketNews(
            company_name="A", sentiment_score=0.1, top_headlines=["a", "b"]
        )
    with pytest.raises(ValidationError):
        MarketNews(
            company_name="A",
            sentiment_score=0.1,
            top_headlines=["a", "b", "c", "d"],
        )


def test_malformed_llm_output_fails_loudly() -> None:
    pipeline, _ = make_pipeline('{"company_name": "X"}')
    with pytest.raises(Exception):
        pipeline.invoke({"query": "x", "intent": "financial"})


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY"
)
@pytest.mark.parametrize(
    "intent,model", [("financial", FinancialSummary), ("news", MarketNews)]
)
def test_live_gpt_4o_mini(intent: str, model: type) -> None:
    result = build_fincorp_pipeline().invoke(
        {"query": TEST_QUERY, "intent": intent}
    )
    assert isinstance(result, model)
