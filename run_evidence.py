"""Run the pipeline against a REAL model and save the evidence.

Usage::

    python run_evidence.py     # writes evidence/real_run_output.txt
    python run_evidence.py --baseline-runs 3

Needs a real model: ``OPENAI_API_KEY`` (and, for another OpenAI-compatible
provider, ``LLM_BASE_URL`` and ``LLM_MODEL``) in the environment or ``.env``.
The report records, for every case, the input, the source record, the exact
prompt, the raw model output, the API metadata and the final outcome. The key
itself is never written.

Cases:
    A. Baseline: grounding check OFF, mock records (shows what the Pydantic
       contract alone lets through).
    B. Grounding check ON, same mock records.
    C. Grounding check ON, complete test records (the success path).
    D. Invalid intent (no model call).
"""

import argparse
import os
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable
from pydantic import BaseModel

import workflow_architecture as wa

try:  # Optional: read keys from a local .env file.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

QUERY = "Impact of the new generative AI orchestration on Q4 profit margins"
DEFAULT_OUTPUT = Path("evidence") / "real_run_output.txt"
RULE = "=" * 72

Row = Tuple[str, str, str, str]  # case, intent, outcome, flags


def complete_financial_record(query: str) -> str:
    """Test record that contains every value the contract needs."""
    return (
        "Financial Data DB Record: FinCorp Inc., Q3 2023, Total Revenue "
        f"$45B, Net Income $9.2B, EPS $1.25. Associated query: '{query}'."
    )


def complete_news_record(query: str) -> str:
    """Test record that names the company and gives three headlines."""
    return (
        "News Wire Extract for FinCorp Inc.: Analysts upgrade FinCorp Inc. "
        "target stock amidst strong AI integration rally. Headlines: "
        "(1) Analysts upgrade FinCorp stock to Buy; (2) FinCorp generative "
        "AI orchestration lifts Q4 margin outlook; (3) AI integration "
        "rally extends across enterprise software. "
        f"Associated query: '{query}'."
    )


class Recorder(BaseCallbackHandler):
    """Capture the prompt sent to the model and what it returned."""

    def __init__(self) -> None:
        self.prompt = ""
        self.raw_output = ""
        self.api_metadata: Dict[str, Any] = {}

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Store the full prompt text."""
        self.prompt = "\n".join(
            str(message.content) for batch in messages for message in batch
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Store the raw text and the API metadata of the response."""
        generation = response.generations[0][0]
        self.raw_output = generation.text
        message = getattr(generation, "message", None)
        meta = dict(getattr(message, "response_metadata", None) or {})
        keys = ("model_name", "id", "finish_reason", "token_usage")
        self.api_metadata = {k: meta[k] for k in keys if k in meta}


def placeholder_flags(result: BaseModel) -> List[str]:
    """Return the placeholder-looking values in a parsed answer."""
    flags: List[str] = []
    name = getattr(result, "company_name", "")
    if str(name).strip().lower() in wa.PLACEHOLDER_TEXT:
        flags.append(f"company_name={name!r}")
    for field in ("total_revenue", "net_income"):
        if getattr(result, field, None) == 0.0:
            flags.append(f"{field}=0.0")
    return flags


class Report:
    """Print every line and keep it for the evidence file."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self.rows: List[Row] = []

    def emit(self, text: str = "") -> None:
        """Print a line and store it."""
        print(text)
        self.lines.append(text)

    def title(self, text: str) -> None:
        """Emit a section title."""
        self.emit(f"\n{RULE}\n{text}\n{RULE}")

    def case(
        self,
        label: str,
        pipeline: Runnable,
        intent: str,
        show_prompt: bool = True,
    ) -> None:
        """Run one case and record everything about it."""
        payload = {"query": QUERY, "intent": intent}
        recorder = Recorder()
        self.emit(f"\n--- {label} | intent={intent!r} ---")
        self.emit(f"Input: {payload}")
        started = time.perf_counter()
        outcome, flags = "", ""
        try:
            result = pipeline.invoke(payload, config={"callbacks": [recorder]})
            found = placeholder_flags(result)
            outcome = "ACCEPTED"
            flags = ", ".join(found) if found else "none"
            self._model_io(recorder, show_prompt)
            self.emit(f"Outcome: ACCEPTED -> {type(result).__name__}")
            self.emit(result.model_dump_json(indent=2))
            self.emit(f"Placeholder-looking values: {flags}")
        except wa.UngroundedValueError as exc:
            outcome, flags = "REJECTED (grounding check)", "n/a"
            self._model_io(recorder, show_prompt)
            self.emit("Outcome: REJECTED by the grounding check")
            for problem in exc.problems:
                self.emit(f"  - {problem}")
        except Exception as exc:  # noqa: BLE001 - record any failure as-is
            outcome, flags = f"ERROR ({type(exc).__name__})", "n/a"
            self._model_io(recorder, show_prompt)
            self.emit(f"Outcome: ERROR {type(exc).__name__}: {exc}")
        self.emit(f"Elapsed: {time.perf_counter() - started:.1f}s")
        self.rows.append((label, intent, outcome, flags))

    def _model_io(self, recorder: Recorder, show_prompt: bool) -> None:
        """Emit the prompt, raw model output and API metadata."""
        if recorder.api_metadata:
            self.emit(f"API metadata: {recorder.api_metadata}")
        if show_prompt and recorder.prompt:
            self.emit("Prompt sent to the model:")
            self.emit(recorder.prompt.rstrip())
        if recorder.raw_output:
            self.emit("Raw model output:")
            self.emit(recorder.raw_output.rstrip())

    def summary(self) -> None:
        """Emit the final table."""
        self.title("SUMMARY")
        self.emit(f"{'Case':<44} {'Intent':<10} {'Outcome':<28} Placeholders")
        for label, intent, outcome, flags in self.rows:
            self.emit(f"{label:<44} {intent:<10} {outcome:<28} {flags}")


def package_version(name: str) -> str:
    """Return an installed package version, or 'unknown'."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def run_all(
    llm: Optional[BaseChatModel] = None, baseline_runs: int = 2
) -> str:
    """Run every case and return the full report text.

    Args:
        llm: Optional model override. Only used by the offline tests; the
            report is then marked as an offline test, not a live run.
        baseline_runs: Repetitions of each intent in case A.

    Returns:
        The report text.
    """
    live = llm is None
    base_url = os.environ.get("LLM_BASE_URL") or ""
    host = urlparse(base_url).netloc or "api.openai.com (default)"
    model = os.environ.get("LLM_MODEL", wa.MODEL_NAME)

    report = Report()
    report.title("FinCorp RAG pipeline: run evidence")
    report.emit(f"Mode: {'LIVE' if live else 'OFFLINE TEST (fake model)'}")
    report.emit(f"Model: {model if live else 'fake'}")
    report.emit(f"Endpoint host: {host if live else 'none'}")
    report.emit(f"Date (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}")
    report.emit(f"Python: {platform.python_version()}")
    report.emit(
        "Packages: langchain-core "
        f"{package_version('langchain-core')}, langchain-openai "
        f"{package_version('langchain-openai')}, pydantic "
        f"{package_version('pydantic')}"
    )
    report.emit("The API key is not recorded in this file.")

    baseline = wa.build_fincorp_pipeline(llm, enforce_grounding=False)
    guarded = wa.build_fincorp_pipeline(llm)
    complete = wa.build_fincorp_pipeline(
        llm,
        financial_fetcher=complete_financial_record,
        news_fetcher=complete_news_record,
    )

    report.title(
        "A. Baseline: grounding check OFF, mock records "
        "(what the Pydantic contract alone lets through)"
    )
    for run in range(1, baseline_runs + 1):
        for intent in wa.SUPPORTED_INTENTS:
            report.case(f"A{run} baseline, mock records", baseline, intent)

    report.title("B. Grounding check ON, same mock records")
    for intent in wa.SUPPORTED_INTENTS:
        report.case("B guard ON, mock records", guarded, intent)

    report.title(
        "C. Grounding check ON, complete TEST records "
        "(written for this run; not FinCorp data)"
    )
    for intent in wa.SUPPORTED_INTENTS:
        report.case("C guard ON, complete test records", complete, intent)

    report.title("D. Invalid intent (no model call)")
    report.case("D invalid intent", guarded, "weather")

    report.summary()
    return "\n".join(report.lines) + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    """Run the live evidence collection and save the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-runs", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if load_dotenv is not None:
        load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (env var or .env).")

    text = run_all(baseline_runs=args.baseline_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"\nSaved the evidence to {args.output}")


if __name__ == "__main__":
    main()
