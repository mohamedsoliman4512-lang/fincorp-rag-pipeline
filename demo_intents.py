"""Run the pipeline for each intent, including an invalid one."""

from dotenv import load_dotenv

from workflow_architecture import build_fincorp_pipeline

load_dotenv()

QUERY = "Impact of the new generative AI orchestration on Q4 profit margins"

pipeline = build_fincorp_pipeline()

for intent in ("financial", "news", "weather"):
    print(f"\n--- intent={intent!r} ---")
    try:
        print(pipeline.invoke({"query": QUERY, "intent": intent}))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")