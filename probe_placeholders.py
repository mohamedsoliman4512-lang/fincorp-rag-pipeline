"""Probe: does the model fill missing fields with placeholders?

Usage::

    python probe_placeholders.py   # writes evidence/placeholder_probe.txt
    python probe_placeholders.py --runs 5

The saved run in ``evidence/real_run_output.txt`` used prompts that already
forbid guessing and placeholders. This probe sends the SAME mock records with
prompts that do NOT contain those two sentences, several times per intent, and
counts what the Pydantic contract lets through: placeholder text such as
``UNKNOWN``, an amount of ``0.0``, an empty company name, a repeated headline,
and any value the source does not support. There is no grounding check here:
the point is to see the model's behaviour, not to filter it.

The old pre-rule prompt of the first development run was not saved, so this is
a reconstruction (the current prompt minus the rule), not the original text.
Whatever the counts are, including zero, they are written as they are. The key
is never written.
"""

import argparse
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import workflow_architecture as wa
from run_evidence import QUERY, package_version

try:  # Optional: read keys from a local .env file.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

DEFAULT_OUTPUT = Path("evidence") / "placeholder_probe.txt"
RULE = "=" * 72

# The current prompts minus the sentences that forbid guessing/placeholders.
_FINANCIAL_NO_RULE = """You are a financial extraction engine for FinCorp.
Using ONLY the retrieved record and the user query below, extract the data
into the required JSON contract. Report monetary values as floats in
millions (e.g. $45B -> 45000.0). The quarter must look like 'Q3 2023'.

User query: {query}

Retrieved financial record:
{financial_data}

{format_instructions}
"""

_NEWS_NO_RULE = """You are a market news analysis engine for FinCorp.
Using ONLY the retrieved news and the user query below, produce a sentiment
overview in the required JSON contract. The sentiment score must be between
-1.0 and 1.0, and top_headlines must contain exactly 3 headlines.

User query: {query}

Retrieved news:
{news_data}

{format_instructions}
"""


@dataclass
class Outcome:
    """What happened in one live call."""

    intent: str
    run: int
    status: str  # ACCEPTED, PARSE_ERROR or API_ERROR
    findings: List[str] = field(default_factory=list)
    ungrounded: List[str] = field(default_factory=list)


def prompt_parts(
    intent: str,
) -> Tuple[str, PydanticOutputParser, str, Type[BaseModel]]:
    """Return (prompt text, parser, source record, model) for an intent."""
    if intent == "financial":
        parser = PydanticOutputParser(pydantic_object=wa.FinancialSummary)
        record = wa.fetch_structured_financials(QUERY)
        template = PromptTemplate(
            template=_FINANCIAL_NO_RULE,
            input_variables=["query", "financial_data"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
        )
        text = template.format(query=QUERY, financial_data=record)
        return text, parser, record, wa.FinancialSummary
    parser = PydanticOutputParser(pydantic_object=wa.MarketNews)
    record = wa.fetch_unstructured_news(QUERY)
    template = PromptTemplate(
        template=_NEWS_NO_RULE,
        input_variables=["query", "news_data"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )
    text = template.format(query=QUERY, news_data=record)
    return text, parser, record, wa.MarketNews


def placeholder_findings(result: BaseModel) -> List[str]:
    """Return the placeholder-style values found in a parsed answer."""
    found: List[str] = []
    name = str(getattr(result, "company_name", ""))
    if name.strip() == "":
        found.append("empty_name")
    elif name.strip().lower() in wa.PLACEHOLDER_TEXT:
        found.append("placeholder_text")
    for amount in ("total_revenue", "net_income"):
        if getattr(result, amount, None) == 0.0:
            found.append("zero_amount")
    headlines: List[str] = list(getattr(result, "top_headlines", []) or [])
    if headlines and len(set(headlines)) < len(headlines):
        found.append("repeated_headline")
    if any(h.strip().lower() in wa.PLACEHOLDER_TEXT for h in headlines):
        found.append("placeholder_text")
    return sorted(set(found))


def api_metadata(message: Any) -> Dict[str, Any]:
    """Pick the useful, non-secret fields of the response metadata."""
    meta = dict(getattr(message, "response_metadata", None) or {})
    keys = ("model_name", "id", "finish_reason", "token_usage")
    return {k: meta[k] for k in keys if k in meta}


def run_probe(
    llm: Optional[BaseChatModel] = None, runs: int = 5
) -> Tuple[str, List[Outcome]]:
    """Run the probe and return (report text, outcomes).

    Args:
        llm: Optional model override, used only by the offline tests; the
            report is then marked as an offline test.
        runs: Calls per intent.
    """
    live = llm is None
    model: BaseChatModel = llm or ChatOpenAI(
        model=os.environ.get("LLM_MODEL", wa.MODEL_NAME),
        temperature=0,
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )
    host = urlparse(os.environ.get("LLM_BASE_URL") or "").netloc
    lines: List[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    model_name = os.environ.get("LLM_MODEL", wa.MODEL_NAME)
    endpoint = host or "api.openai.com (default)"
    emit(RULE)
    emit("Placeholder probe: prompts WITHOUT the anti-placeholder rule")
    emit(RULE)
    emit(f"Mode: {'LIVE' if live else 'OFFLINE TEST (fake model)'}")
    emit(f"Model: {model_name if live else 'fake'}")
    emit(f"Endpoint host: {endpoint if live else 'none'}")
    emit(f"Date (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}")
    emit(f"Python: {platform.python_version()}")
    emit(
        f"Packages: langchain-core {package_version('langchain-core')}, "
        f"langchain-openai {package_version('langchain-openai')}, "
        f"pydantic {package_version('pydantic')}"
    )
    emit("Runs per intent: " + str(runs))
    emit("The API key is not recorded in this file.")

    outcomes: List[Outcome] = []
    for intent in wa.SUPPORTED_INTENTS:
        text, parser, record, _ = prompt_parts(intent)
        emit(f"\n{RULE}\nIntent: {intent}\n{RULE}")
        emit("Source record (mock):\n" + record)
        emit("Prompt sent to the model (same in every run):\n" + text.rstrip())
        for run in range(1, runs + 1):
            emit(f"\n--- {intent} run {run} ---")
            started = time.perf_counter()
            try:
                message = model.invoke(text)
            except Exception as exc:  # noqa: BLE001 - record as-is
                emit(f"API ERROR {type(exc).__name__}: {exc}")
                outcomes.append(Outcome(intent, run, "API_ERROR"))
                continue
            raw = str(message.content).strip()
            emit(f"API metadata: {api_metadata(message)}")
            emit("Raw model output:\n" + raw)
            try:
                result = parser.parse(raw)
            except Exception as exc:  # noqa: BLE001 - record as-is
                emit(f"Parse result: PARSE_ERROR {type(exc).__name__}")
                outcomes.append(Outcome(intent, run, "PARSE_ERROR"))
            else:
                found = placeholder_findings(result)
                bad = wa.find_ungrounded_values(result, record, QUERY)
                emit("Parse result: ACCEPTED by the contract")
                emit(f"Placeholder-style values: {found or 'none'}")
                emit(f"Values not supported by the source: {bad or 'none'}")
                outcomes.append(Outcome(intent, run, "ACCEPTED", found, bad))
            emit(f"Elapsed: {time.perf_counter() - started:.1f}s")

    emit(f"\n{RULE}\nSUMMARY\n{RULE}")
    emit(
        f"{'Intent':<10} {'Calls':>5} {'Parse err':>9} {'API err':>7} "
        f"{'Accepted':>8} {'Placeholder':>11} {'Ungrounded':>10}"
    )
    for intent in wa.SUPPORTED_INTENTS:
        rows = [o for o in outcomes if o.intent == intent]
        accepted = [o for o in rows if o.status == "ACCEPTED"]
        flagged = [o for o in accepted if o.findings]
        ungrounded = [o for o in accepted if o.ungrounded]
        emit(
            f"{intent:<10} {len(rows):>5} "
            f"{sum(o.status == 'PARSE_ERROR' for o in rows):>9} "
            f"{sum(o.status == 'API_ERROR' for o in rows):>7} "
            f"{len(accepted):>8} {len(flagged):>11} {len(ungrounded):>10}"
        )
    emit(
        "Placeholder = accepted answers with placeholder text, an amount of "
        "0.0, an empty company name or a repeated headline."
    )
    return "\n".join(lines) + "\n", outcomes


def main(argv: Optional[List[str]] = None) -> None:
    """Run the live probe and save the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if load_dotenv is not None:
        load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (env var or .env).")

    text, outcomes = run_probe(runs=args.runs)
    if all(o.status == "API_ERROR" for o in outcomes):
        raise SystemExit(
            "Every call failed (see the errors above). Nothing was saved."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"\nSaved the probe to {args.output}")


if __name__ == "__main__":
    main()
