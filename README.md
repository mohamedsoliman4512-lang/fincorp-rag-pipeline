# FinCorp Enterprise RAG Pipeline (NexusFlow AI PoC)

LCEL pipeline: parallel retrieval (`RunnableParallel`), intent routing
(`RunnableLambda`), Pydantic-validated outputs, and a grounding check that
rejects placeholder or guessed values. See `design_proposal.md` for the
design and `evidence/real_run_output.txt` for the saved output of a live run.

## Setup (VS Code terminal, Windows PowerShell)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    copy .env.example .env      # then put your key in .env; never commit it

## Run

    python workflow_architecture.py   # both intents, live model, needs a key
    python -m pytest -v               # offline tests (fake model)

## Real-run evidence

The offline tests use a fake model, so they only show that the code is wired
correctly. The evidence of a real model run is produced by:

    python run_evidence.py

It writes `evidence/real_run_output.txt` with, for every case, the input, the
exact prompt, the raw model output, the API metadata and the outcome:

- A. grounding check OFF, mock records: what the Pydantic contract alone lets
  through. The mock records have no company name, quarter year or net income;
- B. grounding check ON, same mock records;
- C. grounding check ON, complete test records (written for the run, not
  FinCorp data): the success path for both intents;
- D. invalid intent: a clear error, no model call.

The saved run used Groq (`api.groq.com`) with `openai/gpt-oss-120b`, not
`gpt-4o-mini`. Its results are summarised in section 5 of
`design_proposal.md`. In short: the model sometimes refused to produce JSON
(parser error), sometimes invented or left empty the company name (accepted by
the contract alone, and in one case rejected by the grounding check), and
answered correctly when the record was complete. In that run, whose prompts
forbid placeholders, `UNKNOWN` and `0.0` did not appear. Each case is one or
two calls, so these are observations, not statistics.

A second script, `probe_placeholders.py`, sends the same mock records with
prompts that lack that rule, 5 calls per intent, and saves
`evidence/placeholder_probe.txt`. There the model returned
`net_income = 0.0` in 4 of 5 financial calls and the company name `FinCorp`,
which the source does not contain, in all 9 accepted answers; `UNKNOWN` did
not appear. At temperature 0 the repeats are almost identical, so treat the
counts as one observed behaviour, not as statistics.

## Grounding check

A Pydantic contract guarantees the shape of an answer, not its truth.
`check_grounding` compares the parsed answer with the retrieved source and
raises `UngroundedValueError` for a placeholder company name, or a company,
quarter-and-year or `$` amount that the source does not contain. It is a
heuristic text match: it does not verify headlines or sentiment, and it does
not detect a repeated headline. The prompts also tell the model not to guess
missing values.

## Development note

The task targets `gpt-4o-mini` (the default in the code), which I have not
run. Development and the saved evidence used another OpenAI-compatible
provider through `LLM_BASE_URL` and `LLM_MODEL` in `.env` (see
`.env.example`); the header of the evidence file states the model and
endpoint used.
