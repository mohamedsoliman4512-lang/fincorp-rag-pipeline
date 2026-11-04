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
