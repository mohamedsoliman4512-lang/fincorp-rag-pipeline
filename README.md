# FinCorp Enterprise RAG Pipeline (NexusFlow AI PoC)

## Setup (VS Code terminal)

    python -m venv .venv
    # Windows PowerShell:  .venv\Scripts\Activate.ps1
    # Windows cmd:         .venv\Scripts\activate.bat
    # macOS/Linux:         source .venv/bin/activate
    pip install -r requirements.txt

Copy `.env.example` to `.env` and put your NEW OpenAI key in it.
`.env` is git-ignored; never commit it.

## Run

    python workflow_architecture.py   # live call, needs the key
    pytest -v                         # offline tests; live ones need the key

## Development note

The task targets `gpt-4o-mini` (the default). The live development runs were
made on Groq's OpenAI-compatible endpoint with `openai/gpt-oss-120b`, set via
`LLM_BASE_URL` and `LLM_MODEL` in `.env` (see `.env.example`). `gpt-4o-mini`
itself was not run live.

## Design proposal

See `design_proposal.md`: the technical email on router chains, parallel
execution and Pydantic contracts.
