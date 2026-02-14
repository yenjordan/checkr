import json
from services.supabase_store import _get_client
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Use Super 49B for chat — Ultra 253B returns empty responses on long context
_llm = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",
    max_tokens=1024,
)


def fetch_paper_context(filename: str) -> dict | None:
    """Retrieve the most recent analysis for a given filename from Supabase."""
    client = _get_client()
    resp = (
        client.table("paper_analyses")
        .select("*")
        .eq("filename", filename)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if resp.data:
        return resp.data[0]
    return None


def _build_context(row: dict) -> str:
    """Turn a Supabase row into a rich text context block for the LLM."""
    sections = []

    sections.append(f"FILENAME: {row.get('filename', 'unknown')}")
    sections.append(f"PAGES: {row.get('page_count', '?')}")
    sections.append(f"VERDICT: {row.get('verdict', 'unknown')}")
    sections.append(f"SUMMARY: {row.get('summary', 'N/A')}")

    # Planner steps
    steps = row.get("planner_steps") or []
    if steps:
        sections.append("VERIFICATION PLAN:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps)))

    # Code chunks extracted
    code_chunks = row.get("code_chunks") or []
    if code_chunks:
        parts = []
        for i, c in enumerate(code_chunks):
            lang = c.get("language", "unknown")
            ctx = c.get("context", "")
            code = c.get("code", "")
            parts.append(f"  Chunk {i+1} ({lang}): {ctx}\n```{lang}\n{code}\n```")
        sections.append("EXTRACTED CODE:\n" + "\n".join(parts))

    # Coding review
    cr = row.get("coding_review") or {}
    if cr:
        correct = cr.get("is_conceptually_correct", "N/A")
        explanation = cr.get("explanation", "")
        issues = cr.get("issues") or []
        block = f"  Conceptually correct: {correct}\n  Explanation: {explanation}"
        if issues:
            block += "\n  Issues:\n" + "\n".join(f"    - {iss}" for iss in issues)
        sections.append("CONCEPTUAL CODE REVIEW:\n" + block)

    # Math chunks extracted
    math_chunks = row.get("math_chunks") or []
    if math_chunks:
        parts = []
        for i, m in enumerate(math_chunks):
            parts.append(f"  Equation {i+1} ({m.get('equation_type', '?')}): {m.get('latex', '')}\n    Context: {m.get('context', '')}")
        sections.append("EXTRACTED MATH:\n" + "\n".join(parts))

    # Math analysis
    ma = row.get("math_analysis") or {}
    if ma and ma.get("explanation"):
        block = (
            f"  Mathematically valid: {ma.get('is_mathematically_valid', 'N/A')}\n"
            f"  Consistent with claims: {ma.get('is_consistent_with_claims', 'N/A')}\n"
            f"  Explanation: {ma.get('explanation', '')}"
        )
        issues = ma.get("issues") or []
        if issues:
            block += "\n  Issues:\n" + "\n".join(f"    - {iss}" for iss in issues)
        vsteps = ma.get("verified_steps") or []
        if vsteps:
            block += "\n  Verified steps:\n" + "\n".join(f"    {i+1}. {s}" for i, s in enumerate(vsteps))
        sections.append("MATH VERIFICATION:\n" + block)

    # Code execution results
    exec_results = row.get("code_execution_results") or []
    if exec_results:
        parts = []
        for i, r in enumerate(exec_results):
            passed = "PASSED" if r.get("ran_successfully") else "FAILED"
            analysis = r.get("analysis", "")
            stderr = r.get("stderr", "")
            entry = f"  Chunk {i+1} ({r.get('language', '?')}): {passed}"
            if analysis:
                entry += f"\n    Analysis: {analysis}"
            if stderr and not r.get("ran_successfully"):
                entry += f"\n    Error: {stderr[:300]}"
            parts.append(entry)
        sections.append("CODE EXECUTION RESULTS:\n" + "\n".join(parts))

    # Paper text with approximate page markers (truncated to fit context window)
    paper_text = row.get("paper_text") or ""
    page_count = row.get("page_count") or 1
    if paper_text:
        truncated = paper_text[:12000]
        if len(paper_text) > 12000:
            truncated += f"\n... [truncated, {len(paper_text)} chars total]"
        # Insert approximate page markers so the LLM can cite pages
        if page_count > 1:
            chars_per_page = len(truncated) // page_count
            marked = ""
            for p in range(page_count):
                start = p * chars_per_page
                end = (p + 1) * chars_per_page if p < page_count - 1 else len(truncated)
                marked += f"\n[Page {p + 1}]\n" + truncated[start:end]
            sections.append("PAPER TEXT:" + marked)
        else:
            sections.append("PAPER TEXT:\n[Page 1]\n" + truncated)

    return "\n\n".join(sections)


SYSTEM_PROMPT = (
    "You are CHECKR, a concise research paper verification assistant.\n\n"
    "RULES:\n"
    "1. Be extremely concise. Answer in 2-4 sentences max. No long paragraphs.\n"
    "2. NEVER use markdown. No **, ##, *, bullet lists, or numbered lists.\n"
    "3. Write flowing prose only. Use <b>bold</b> for emphasis and <br> for line breaks.\n"
    "4. For math, use ONLY dollar-sign LaTeX: $x^2$ for inline, $$E=mc^2$$ for display. "
    "NEVER use backslash-paren \\( \\) or backslash-bracket \\[ \\] notation.\n"
    "5. Cite page numbers like (p. 3) or (pp. 5-7) when referencing specific claims. "
    "The paper text has [Page N] markers you can use.\n"
    "6. Reference verification findings (verdict, code results, math validity) when relevant.\n"
    "7. Answer the question directly. Do not summarize the whole paper unless asked."
)


async def chat(filename: str, question: str, history: list[dict] | None = None) -> str:
    """RAG chat: retrieve paper context from Supabase, answer with Nemotron Ultra."""
    row = fetch_paper_context(filename)
    if not row:
        return "I don't have any analysis data for this paper yet. Please run the verification first."

    context = _build_context(row)
    print(f"[RAG] Context length: {len(context)} chars for {filename}")

    # Build messages with context injected into the first user message
    # (some models handle user-role context better than huge system prompts)
    sys_msg = SYSTEM_PROMPT
    user_msg = (
        "Here is all the information about the paper and its verification:\n\n"
        + context
        + "\n\n---\n\nUser question: " + question
    )

    messages = [("system", sys_msg)]

    # Append conversation history
    if history:
        for msg in history:
            role = msg.get("role", "user")
            messages.append((role, msg["content"]))

    messages.append(("user", user_msg))

    try:
        response = await _llm.ainvoke(messages)
        content = response.content or ""
        print(f"[RAG] Response length: {len(content)} chars")
        if content.strip():
            return content
    except Exception as e:
        print(f"[RAG] LLM error: {e}")

    # Fallback: try again with a shorter context (just the analysis, no paper text)
    print("[RAG] Retrying with shorter context...")
    short_context = "\n\n".join(
        s for s in context.split("\n\n") if not s.startswith("PAPER TEXT:")
    )
    short_user_msg = (
        "Here is the verification analysis for a paper:\n\n"
        + short_context
        + "\n\n---\n\nUser question: " + question
    )

    try:
        response = await _llm.ainvoke([
            ("system", sys_msg),
            ("user", short_user_msg),
        ])
        content = response.content or ""
        print(f"[RAG] Retry response length: {len(content)} chars")
        if content.strip():
            return content
    except Exception as e:
        print(f"[RAG] Retry also failed: {e}")

    return "Sorry, I couldn't generate a response right now. Please try again."
