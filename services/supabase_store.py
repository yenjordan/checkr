import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def save_analysis(
    filename: str,
    paper_text: str,
    page_count: int,
    result: dict,
    *,
    math_chunks: list | None = None,
) -> dict:
    structured = result.get("structured_response") or {}
    subagent = result.get("subagent_responses") or {}

    row = {
        "filename": filename,
        "paper_text": paper_text,
        "page_count": page_count,
        "verdict": structured.get("verdict", "unknown"),
        "summary": structured.get("summary", ""),
        "planner_steps": subagent.get("planner", {}).get("steps", []),
        "code_chunks": subagent.get("code_extractor", {}).get("chunks", []),
        "coding_review": subagent.get("coding", {}),
        "math_chunks": (
            math_chunks
            if math_chunks is not None
            else subagent.get("math_extractor", {}).get("chunks", [])
        ),
        "math_analysis": subagent.get("math", {}),
        "code_execution_results": subagent.get("replanner", {}).get("results", []),
    }

    client = _get_client()
    resp = client.table("paper_analyses").insert(row).execute()
    return resp.data