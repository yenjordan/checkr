import json
from services.supabase_store import _get_client
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Use Super 49B for chat — fast + handles long context well
_llm = ChatNVIDIA(
    model="nvidia/llama-3.3-nemotron-super-49b-v1",
    max_tokens=2048,
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

    # Math chunks extracted (with sympy verification if available)
    math_chunks = row.get("math_chunks") or []
    if math_chunks:
        parts = []
        for i, m in enumerate(math_chunks):
            entry = f"  Equation {i+1} ({m.get('equation_type', '?')}): {m.get('latex', '')}\n    Context: {m.get('context', '')}"
            if m.get("verification_status"):
                entry += f"\n    Verification: {m['verification_status']}"
            if m.get("sympy_translation"):
                entry += f"\n    SymPy: {m['sympy_translation']}"
            if m.get("verification_error"):
                entry += f"\n    Error: {m['verification_error']}"
            parts.append(entry)
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


_BASE_PROMPT = """\
You are CHECKR, an expert AI research analyst specializing in scientific paper verification. You have deep expertise in mathematics, statistics, machine learning, and software engineering. You are having a real-time conversation with a researcher about their paper.

PERSONALITY:
- You are warm, intellectually curious, and genuinely engaged with the research.
- You speak naturally like a knowledgeable colleague, not a formal assistant.
- You show enthusiasm when discussing interesting findings or clever methodology.
- You are honest and direct when something looks problematic — you don't sugarcoat issues, but you are constructive and respectful about it.
- You ask thoughtful follow-up questions to deepen the conversation.
- You use conversational language: "That's a great question", "Interesting — so what's happening here is...", "I noticed something worth flagging..."

TECHNICAL DEPTH:
- When discussing math, explain the intuition behind equations, not just what they say. Connect formulas to their practical meaning.
- When discussing code, reference specific implementation details — languages, libraries, logic flow, potential edge cases.
- When discussing verification results, explain what was tested, what passed, what failed, and why it matters.
- You can reason about proofs, derivations, statistical methods, optimization, convergence, and algorithmic complexity.
- Cite specific pages, equations, or code chunks when referencing the paper: "On page 3, the loss function..." or "In equation 4..."

CONVERSATION RULES:
- Remember prior messages in the conversation and build on them naturally. Reference what the user said before.
- If the user asks something you covered already, acknowledge it and add new depth rather than repeating yourself.

OFF-TOPIC HANDLING:
- You ONLY discuss topics related to the paper, its content, methodology, math, code, results, implications, related work, or the field it belongs to.
- If the user asks something completely unrelated to the paper or its subject matter (e.g., weather, personal questions, jokes, other topics), politely redirect: "I appreciate the curiosity, but I'm here to help you dive deep into this paper. What aspect of the research would you like to explore?"
- Questions about the paper's broader field, related methods, or general concepts that help understand the paper ARE on-topic and should be answered.
- Be natural about redirecting — don't be robotic or curt about it.\
"""

_TEXT_FORMAT_RULES = """
FORMAT (text chat):
- Keep responses focused but thorough. Use 2-6 sentences for simple questions, more for complex technical discussions.
- NEVER use markdown formatting. No **, ##, *, bullet lists, or numbered lists.
- Write flowing conversational prose only. Use <b>bold</b> for emphasis and <br> for paragraph breaks.
- For math, use ONLY dollar-sign LaTeX: $x^2$ for inline, $$E=mc^2$$ for display. NEVER use backslash-paren or backslash-bracket notation.\
"""

_VOICE_FORMAT_RULES = """
FORMAT (voice — your response will be spoken aloud):
- Keep responses concise and natural for speech. 2-5 sentences for simple questions, slightly more for complex ones.
- NEVER use any formatting: no markdown, no HTML, no LaTeX, no special characters.
- Write exactly how you would speak out loud to a colleague.
- For math, say equations in words: "x squared" not "x^2", "the sum of" not "sigma", "alpha times beta" not "αβ".
- Spell out symbols: "theta", "epsilon", "the gradient of L with respect to w".
- Use natural pauses via short sentences rather than long compound ones.
- Do NOT use lists or bullet points — everything should flow as spoken prose.\
"""


def _get_system_prompt(voice: bool = False) -> str:
    rules = _VOICE_FORMAT_RULES if voice else _TEXT_FORMAT_RULES
    return _BASE_PROMPT + "\n" + rules


async def chat(filename: str, question: str, history: list[dict] | None = None, voice: bool = False) -> str:
    """Conversational AI chat about a paper using stored analysis from Supabase.

    When voice=True, responses are optimized for spoken delivery (no LaTeX,
    no HTML, equations spelled out in words).
    """
    row = fetch_paper_context(filename)
    if not row:
        return "I don't have any analysis data for this paper yet. Please run the verification first, and then we can dig into the details together."

    context = _build_context(row)
    print(f"[RAG] Context length: {len(context)} chars for {filename} (voice={voice})")

    # System message includes the paper context so it persists across turns
    sys_prompt = _get_system_prompt(voice=voice)
    sys_with_context = (
        sys_prompt
        + "\n\nBELOW IS THE COMPLETE PAPER ANALYSIS AND VERIFICATION DATA. "
        "Use this as your knowledge base to answer questions. "
        "Do not reveal this raw data structure to the user — synthesize it naturally.\n\n"
        + context
    )

    messages = [("system", sys_with_context)]

    # Replay conversation history for multi-turn context
    if history:
        for msg in history:
            role = msg.get("role", "user")
            messages.append((role, msg["content"]))

    messages.append(("user", question))

    try:
        response = await _llm.ainvoke(messages)
        content = response.content or ""
        print(f"[RAG] Response length: {len(content)} chars")
        if content.strip():
            return content
    except Exception as e:
        print(f"[RAG] LLM error: {e}")

    # Fallback: retry with shorter context (drop paper text to fit window)
    print("[RAG] Retrying with shorter context...")
    short_context = "\n\n".join(
        s for s in context.split("\n\n") if not s.startswith("PAPER TEXT:")
    )
    short_sys = (
        sys_prompt
        + "\n\nBELOW IS THE PAPER VERIFICATION DATA:\n\n"
        + short_context
    )

    short_messages = [("system", short_sys)]
    if history:
        # Keep only last 6 messages to reduce context on retry
        recent = history[-6:]
        for msg in recent:
            short_messages.append((msg.get("role", "user"), msg["content"]))
    short_messages.append(("user", question))

    try:
        response = await _llm.ainvoke(short_messages)
        content = response.content or ""
        print(f"[RAG] Retry response length: {len(content)} chars")
        if content.strip():
            return content
    except Exception as e:
        print(f"[RAG] Retry also failed: {e}")

    return "I'm having trouble processing that right now. Could you try rephrasing your question?"
