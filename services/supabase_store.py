import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def _serialize_for_db(obj):
    """Ensure value is JSON-serializable for DB insert (str keys, no custom types)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialize_for_db(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_db(x) for x in obj]
    return str(obj)

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
        "filename": str(filename),
        "paper_text": str(paper_text),
        "page_count": int(page_count),
        "verdict": str(structured.get("verdict", "unknown")),
        "summary": str(structured.get("summary", "")),
        "planner_steps": _serialize_for_db(subagent.get("planner", {}).get("steps", [])),
        "code_chunks": _serialize_for_db(subagent.get("code_extractor", {}).get("chunks", [])),
        "coding_review": _serialize_for_db(subagent.get("coding", {})),
        "math_chunks": _serialize_for_db(
            math_chunks
            if math_chunks is not None
            else subagent.get("math_extractor", {}).get("chunks", [])
        ),
        "math_analysis": _serialize_for_db(subagent.get("math", {})),
        "code_execution_results": _serialize_for_db(subagent.get("replanner", {}).get("results", [])),
    }

    client = _get_client()
    resp = client.table("paper_analyses").insert(row).execute()
    return resp.data