"""Offline structural tests (no API key needed) plus an opt-in live test."""

import os
from pathlib import Path
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
    """Return (pipeline, fake_llm); grounding is off: wiring tests only."""
    llm = RecordingFakeLLM(responses=[response])
    return build_fincorp_pipeline(llm=llm, enforce_grounding=False), llm


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


# ---------------------------------------------------------------------------
# Grounding check: placeholder / guessed values must be rejected
# ---------------------------------------------------------------------------

PLACEHOLDER_JSON = (
    '{"company_name": "UNKNOWN", "quarter": "Q4 2023", '
    '"total_revenue": 45000.0, "net_income": 0.0}'
)
FINANCIAL_INPUT = {"query": TEST_QUERY, "intent": "financial"}


def guarded(response: str, **kwargs):
    """Pipeline with the grounding check ON (the default)."""
    llm = RecordingFakeLLM(responses=[response])
    return build_fincorp_pipeline(llm=llm, **kwargs), llm


def complete_financial(query: str) -> str:
    return (
        "Financial Data DB Record: FinCorp Inc., Q3 2023, Total Revenue "
        f"$45B, Net Income $9.2B. Associated query: '{query}'."
    )


def complete_news(query: str) -> str:
    return (
        "News Wire Extract for FinCorp Inc.: analysts upgrade the stock. "
        f"Associated query: '{query}'."
    )


def test_guard_off_lets_placeholders_through_the_schema() -> None:
    """Documents the finding: the contract alone accepts UNKNOWN / 0.0."""
    pipeline, _ = make_pipeline(PLACEHOLDER_JSON)
    result = pipeline.invoke(FINANCIAL_INPUT)
    assert result.company_name == "UNKNOWN" and result.net_income == 0.0


def test_guard_rejects_placeholders_on_the_incomplete_mock_record() -> None:
    from workflow_architecture import UngroundedValueError

    pipeline, _ = guarded(PLACEHOLDER_JSON)
    with pytest.raises(UngroundedValueError) as info:
        pipeline.invoke(FINANCIAL_INPUT)
    message = str(info.value)
    assert "company_name='UNKNOWN' is a placeholder" in message
    assert "net_income=0.0 is not found in the source" in message
    assert "quarter='Q4 2023'" in message  # the mock has no year
    assert "total_revenue" not in message  # $45B IS in the mock record


def test_guard_rejects_a_guessed_company_name() -> None:
    from workflow_architecture import UngroundedValueError

    guess = PLACEHOLDER_JSON.replace("UNKNOWN", "FinCorp Inc.")
    pipeline, _ = guarded(guess)
    with pytest.raises(UngroundedValueError, match="not in the source"):
        pipeline.invoke(FINANCIAL_INPUT)


def test_guard_accepts_values_supported_by_a_complete_record() -> None:
    grounded = (
        '{"company_name": "FinCorp Inc.", "quarter": "Q3 2023", '
        '"total_revenue": 45000.0, "net_income": 9200.0}'
    )
    pipeline, _ = guarded(grounded, financial_fetcher=complete_financial)
    result = pipeline.invoke(FINANCIAL_INPUT)
    assert isinstance(result, FinancialSummary)
    assert result.net_income == 9200.0


def test_guard_checks_the_news_branch_too() -> None:
    from workflow_architecture import UngroundedValueError

    news = (
        '{"company_name": "FinCorp Inc.", "sentiment_score": 0.5, '
        '"top_headlines": ["a", "b", "c"]}'
    )
    news_input = {"query": TEST_QUERY, "intent": "news"}
    pipeline, _ = guarded(news)  # mock news names no company
    with pytest.raises(UngroundedValueError, match="company_name"):
        pipeline.invoke(news_input)
    pipeline, _ = guarded(news, news_fetcher=complete_news)
    assert isinstance(pipeline.invoke(news_input), MarketNews)


def test_invalid_intent_is_still_a_clear_error_with_the_guard_on() -> None:
    pipeline, llm = guarded(PLACEHOLDER_JSON)
    with pytest.raises(ValueError, match="Unsupported or missing intent"):
        pipeline.invoke({"query": "x", "intent": "weather"})
    assert llm.seen == []


def test_amount_parser_and_quarter_matching() -> None:
    from workflow_architecture import (
        _quarter_in,
        extract_amounts_in_millions,
    )

    assert extract_amounts_in_millions("Net Revenue $45B") == [45000.0]
    assert extract_amounts_in_millions("Income $9.2 billion.") == [9200.0]
    assert extract_amounts_in_millions("cost $300M, $2,500K") == [300.0, 2.5]
    assert extract_amounts_in_millions("no amounts here") == []
    assert not _quarter_in("Q4 2023", "Impact on Q4 profit margins")
    assert _quarter_in("Q4 2023", "Q4 results for 2023")


def test_loader_accepts_injected_fetchers() -> None:
    out = build_parallel_data_loader(
        financial_fetcher=lambda q: f"F:{q}", news_fetcher=lambda q: f"N:{q}"
    ).invoke("hi")
    assert out == {"financial_data": "F:hi", "news_data": "N:hi"}


class SmartFakeLLM(FakeListChatModel):
    """Placeholders for mock records, grounded values for complete ones."""

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        prompt = "\n".join(str(m.content) for m in messages)
        if "financial extraction engine" in prompt:
            if "FinCorp Inc., Q3 2023" in prompt:
                return (
                    '{"company_name": "FinCorp Inc.", "quarter": "Q3 2023", '
                    '"total_revenue": 45000.0, "net_income": 9200.0}'
                )
            return PLACEHOLDER_JSON
        return (
            '{"company_name": "FinCorp Inc.", "sentiment_score": 0.6, '
            '"top_headlines": ["a", "b", "c"]}'
        )


def test_run_evidence_report_covers_all_cases_offline() -> None:
    import run_evidence

    text = run_evidence.run_all(SmartFakeLLM(responses=["x"]), baseline_runs=1)
    assert "Mode: OFFLINE TEST (fake model)" in text  # never looks live
    assert "API key is not recorded" in text
    summary = text.split("SUMMARY")[1]
    assert "ACCEPTED" in summary and "REJECTED (grounding check)" in summary
    assert "ERROR (ValueError)" in summary  # the invalid intent
    assert "company_name='UNKNOWN', net_income=0.0" in summary
    assert "Raw model output:" in text and "Prompt sent to the model:" in text


def make_submission_tree(root: Path, evidence: str, sender: str) -> None:
    import make_submission as ms

    for name in ms.FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder file\n", encoding="utf-8")
    (root / ms.EVIDENCE).write_text(evidence, encoding="utf-8")
    (root / ms.PROBE).write_text(
        "Mode: LIVE\nAPI metadata: {}\n", encoding="utf-8"
    )
    (root / "design_proposal.md").write_text(
        f"**From:** {sender}\nSee {ms.EVIDENCE} and {ms.PROBE}\n\n"
        f"Best regards,\n{sender}\n",
        encoding="utf-8",
    )


def test_submission_check_blocks_offline_evidence_and_wrong_sender(
    tmp_path: Path,
) -> None:
    import make_submission as ms

    make_submission_tree(
        tmp_path, "Mode: OFFLINE TEST (fake model)\nSUMMARY\n", "Real Name"
    )
    problems = ms.check(tmp_path, "Someone Else")
    assert any("not from a LIVE run" in p for p in problems)
    assert any("**From:** Someone Else" in p for p in problems)


def test_submission_check_blocks_keys_and_passes_a_clean_live_tree(
    tmp_path: Path,
) -> None:
    import zipfile

    import make_submission as ms

    live = (
        "Mode: LIVE\nModel: x\nAPI metadata: {}\n"
        "Outcome: ACCEPTED -> FinancialSummary\n"
        "Outcome: ACCEPTED -> MarketNews\nSUMMARY\n"
    )
    make_submission_tree(tmp_path, live, "Real Name")
    assert ms.check(tmp_path, "Real Name") == []
    ms.build(tmp_path, tmp_path / "out.zip")
    with zipfile.ZipFile(tmp_path / "out.zip") as archive:
        assert sorted(archive.namelist()) == sorted(ms.FILES)

    leaked = live + "key gsk_" + "A" * 30 + "\n"
    (tmp_path / ms.EVIDENCE).write_text(leaked, encoding="utf-8")
    assert any(
        "looks like a key" in p for p in ms.check(tmp_path, "Real Name")
    )


def test_submission_check_blocks_a_live_run_where_every_call_failed(
    tmp_path: Path,
) -> None:
    import make_submission as ms

    failed = (
        "Mode: LIVE\nOutcome: ERROR OpenAIAuthenticationError: "
        "Error code: 401 - Incorrect API key provided\nSUMMARY\n"
    )
    make_submission_tree(tmp_path, failed, "Real Name")
    problems = ms.check(tmp_path, "Real Name")
    assert any("no real reply" in p for p in problems)
    assert any("authentication errors" in p for p in problems)
    assert any("FinancialSummary" in p for p in problems)


def test_probe_counts_placeholders_and_marks_offline_runs() -> None:
    import probe_placeholders as pp

    responses = [
        '{"company_name": "UNKNOWN", "quarter": "Q3 2023", '
        '"total_revenue": 45000.0, "net_income": 0.0}',
        "I cannot provide that.",
        '{"company_name": "", "sentiment_score": 0.5, '
        '"top_headlines": ["a", "a", "a"]}',
    ]
    text, outcomes = pp.run_probe(FakeListChatModel(responses=responses), 1)
    assert "Mode: OFFLINE TEST (fake model)" in text
    assert "API key is not recorded" in text
    assert "Prompt sent to the model" in text
    by_intent = {o.intent: o for o in outcomes}
    assert by_intent["financial"].status == "ACCEPTED"
    assert by_intent["financial"].findings == [
        "placeholder_text",
        "zero_amount",
    ]
    assert by_intent["news"].status == "PARSE_ERROR"
    assert pp.placeholder_findings(
        wa.MarketNews(
            company_name="", sentiment_score=0.1, top_headlines=["a"] * 3
        )
    ) == ["empty_name", "repeated_headline"]
