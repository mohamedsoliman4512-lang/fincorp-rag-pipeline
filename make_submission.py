"""Build the submission ZIP, refusing to include secrets or missing evidence.

Usage::

    python make_submission.py --sender "Your Name As On The Platform"

Checks before zipping:
  * every required file exists, including the saved live-run evidence;
  * the evidence file is from a LIVE run (not the offline test mode);
  * no API key (pattern or the value of OPENAI_API_KEY) is in any file;
  * design_proposal.md names the sender exactly, in the From line and in the
    sign-off, and points to the evidence file.
"""

import argparse
import os
import re
import zipfile
from pathlib import Path
from typing import List, Optional

try:  # Optional: read OPENAI_API_KEY from .env to scan for it.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

EVIDENCE = "evidence/real_run_output.txt"
PROBE = "evidence/placeholder_probe.txt"
FILES = [
    "workflow_architecture.py",
    "test_workflow_architecture.py",
    "run_evidence.py",
    "probe_placeholders.py",
    "requirements.txt",
    "README.md",
    "design_proposal.md",
    ".env.example",
    EVIDENCE,
    PROBE,
]
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9]{20,}")
OUTPUT = "fincorp_submission_final.zip"


def check_real_replies(evidence: str) -> List[str]:
    """Return problems showing the evidence holds no real model replies.

    ``Mode: LIVE`` only means a key was present. A run whose every call
    failed (for example a 401 from the wrong endpoint) must not pass, so
    the file needs API metadata and one accepted answer per intent.
    """
    problems: List[str] = []
    if "API metadata:" not in evidence:
        problems.append("The evidence has no API metadata: no real reply.")
    if re.search(r"AuthenticationError|Incorrect API key", evidence):
        problems.append("The evidence contains authentication errors.")
    for model_name in ("FinancialSummary", "MarketNews"):
        if f"Outcome: ACCEPTED -> {model_name}" not in evidence:
            problems.append(f"No accepted live answer for {model_name}.")
    return problems


def check(root: Path, sender: str) -> List[str]:
    """Return a list of problems that must be fixed before submitting."""
    problems: List[str] = []
    missing = [name for name in FILES if not (root / name).is_file()]
    if missing:
        return [f"Missing file: {name}" for name in missing]

    texts = {n: (root / n).read_text(encoding="utf-8") for n in FILES}

    evidence = texts[EVIDENCE]
    if "Mode: LIVE" not in evidence:
        problems.append(
            "The evidence file is not from a LIVE run. Run run_evidence.py "
            "with your real key."
        )
    if "SUMMARY" not in evidence:
        problems.append("The evidence file is incomplete (no SUMMARY).")
    problems.extend(check_real_replies(evidence))

    secret = os.environ.get("OPENAI_API_KEY", "")
    for name, text in texts.items():
        if KEY_PATTERN.search(text) or (len(secret) > 8 and secret in text):
            problems.append(f"{name} contains something that looks like a key")

    proposal = texts["design_proposal.md"]
    if f"**From:** {sender}" not in proposal:
        problems.append(f'design_proposal.md has no "**From:** {sender}"')
    if not proposal.rstrip().endswith(sender):
        problems.append(f'design_proposal.md must end with "{sender}"')
    for pointer in (EVIDENCE, PROBE):
        if pointer not in proposal:
            problems.append(f"design_proposal.md does not mention {pointer}")
    probe = texts[PROBE]
    if "Mode: LIVE" not in probe or "API metadata:" not in probe:
        problems.append("The probe file has no live model replies.")
    return problems


def build(root: Path, output: Path) -> None:
    """Write the ZIP with the required files only."""
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in FILES:
            archive.write(root / name, name)


def main(argv: Optional[List[str]] = None) -> None:
    """Check, then build the ZIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender", required=True)
    args = parser.parse_args(argv)
    if load_dotenv is not None:
        load_dotenv()

    root = Path(".")
    problems = check(root, args.sender)
    if problems:
        print("NOT built. Fix these first:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    build(root, Path(OUTPUT))
    print(f"Built {OUTPUT} with:")
    with zipfile.ZipFile(OUTPUT) as archive:
        for info in archive.infolist():
            print(f"  {info.filename} ({info.file_size} bytes)")


if __name__ == "__main__":
    main()
