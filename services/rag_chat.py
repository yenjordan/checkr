import os
import json
import tempfile

import vertexai
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Content, Part
from google.oauth2 import service_account

from services.supabase_store import _get_client


# ── Vertex AI init ──────────────────────────────────────────────────
def _init_vertex():
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    creds = None
    if sa_json:
        creds = service_account.Credentials.from_service_account_info(
            json.loads(sa_json),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    vertexai.init(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT_ID", ""),
        location=os.environ.get("VERTEX_AI_RAG_LOCATION", "europe-west1"),
        credentials=creds,
    )

_init_vertex()


# ── RAG Corpus management ──────────────────────────────────────────
_CORPUS_DISPLAY_NAME = "CHECKR Paper Analyses"
_corpus_name: str | None = None
# In-memory cache: filename → rag file resource name
_rag_file_cache: dict[str, str] = {}


def _get_or_create_corpus() -> str:
    """Lazily create (or find) the shared RAG corpus."""
    global _corpus_name
    if _corpus_name:
        return _corpus_name

    # Check existing corpora
    try:
        for corpus in rag.list_corpora():
            if corpus.display_name == _CORPUS_DISPLAY_NAME:
                _corpus_name = corpus.name
                print(f"[RAG] Reusing corpus: {_corpus_name}")
                return _corpus_name
    except Exception as e:
        print(f"[RAG] list_corpora error: {e}")

    # Create new corpus with text-embedding-005
    embedding_config = rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    )
    corpus = rag.create_corpus(
        display_name=_CORPUS_DISPLAY_NAME,
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=embedding_config,
        ),
    )
    _corpus_name = corpus.name
    print(f"[RAG] Created corpus: {_corpus_name}")
    return _corpus_name


def upload_paper_to_corpus(filename: str, context_text: str) -> str | None:
    """Upload (or replace) a paper's analysis text in the RAG corpus.

    Returns the RAG file resource name, or None on failure.
    """
    corpus_name = _get_or_create_corpus()

    # Delete any existing file for this paper so we get fresh data
    try:
        for f in rag.list_files(corpus_name=corpus_name):
            if f.display_name == filename:
                rag.delete_file(name=f.name)
                print(f"[RAG] Deleted old RAG file for: {filename}")
                break
    except Exception as e:
        print(f"[RAG] list_files warning: {e}")

    # Write context to temp .txt file and upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(context_text)
        temp_path = f.name

    try:
        rag_file = rag.upload_file(
            corpus_name=corpus_name,
            path=temp_path,
            display_name=filename,
            description=f"CHECKR analysis for {filename}",
        )
        _rag_file_cache[filename] = rag_file.name
        print(f"[RAG] Uploaded to corpus: {rag_file.name}")
        return rag_file.name
    except Exception as e:
        print(f"[RAG] Upload failed: {e}")
        return None
    finally:
        os.unlink(temp_path)


def _ensure_paper_in_corpus(filename: str, row: dict) -> str | None:
    """Make sure the paper's analysis is in the RAG corpus.

    Checks cache first, then corpus file list, and uploads if missing.
    Returns the RAG file resource name.
    """
    # 1. Check cache
    if filename in _rag_file_cache:
        return _rag_file_cache[filename]

    # 2. Check if already in corpus
    corpus_name = _get_or_create_corpus()
    try:
        for f in rag.list_files(corpus_name=corpus_name):
            if f.display_name == filename:
                _rag_file_cache[filename] = f.name
                return f.name
    except Exception:
        pass

    # 3. Upload on the fly
    context_text = _build_context(row, truncate_paper=False)
    return upload_paper_to_corpus(filename, context_text)


# ── Supabase helpers (unchanged — used by main.py) ─────────────────

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


def _build_context(row: dict, *, truncate_paper: bool = True) -> str:
    """Turn a Supabase row into a rich text context block.

    When truncate_paper=False (used for RAG corpus upload), the full paper
    text is included so the RAG Engine can chunk and embed it properly.
    """
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

    # Paper text
    paper_text = row.get("paper_text") or ""
    page_count = row.get("page_count") or 1
    if paper_text:
        if truncate_paper:
            # Old behavior for backward-compat (Hume context endpoint)
            text = paper_text[:12000]
            if len(paper_text) > 12000:
                text += f"\n... [truncated, {len(paper_text)} chars total]"
        else:
            # Full text for RAG corpus — let the engine chunk it
            text = paper_text

        if page_count > 1:
            chars_per_page = len(text) // page_count
            marked = ""
            for p in range(page_count):
                start = p * chars_per_page
                end = (p + 1) * chars_per_page if p < page_count - 1 else len(text)
                marked += f"\n[Page {p + 1}]\n" + text[start:end]
            sections.append("PAPER TEXT:" + marked)
        else:
            sections.append("PAPER TEXT:\n[Page 1]\n" + text)

    return "\n\n".join(sections)


# ── System prompts ─────────────────────────────────────────────────

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
- Be natural about redirecting — don't be robotic or curt about it.

GROUNDING:
- You have access to a retrieval tool that searches the paper's full analysis data. Use the retrieved context to ground your answers.
- Do not reveal raw data structures to the user — synthesize retrieved information naturally.\
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


# ── Chat with Vertex AI RAG Engine ─────────────────────────────────

def _retrieve_from_corpus(question: str, corpus_name: str, rag_file_name: str | None) -> str:
    """Run a retrieval query against the RAG corpus and return concatenated chunk text."""
    rag_resource_kwargs = {"rag_corpus": corpus_name}
    if rag_file_name:
        file_id = rag_file_name.rsplit("/", 1)[-1]
        rag_resource_kwargs["rag_file_ids"] = [file_id]

    try:
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(**rag_resource_kwargs)],
            text=question,
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=10,
                filter=rag.Filter(vector_distance_threshold=0.5),
            ),
        )
        chunks = []
        for ctx in response.contexts.contexts:
            if ctx.text:
                chunks.append(ctx.text)
        retrieved = "\n\n---\n\n".join(chunks)
        print(f"[RAG] Retrieved {len(chunks)} chunks ({len(retrieved)} chars)")
        return retrieved
    except Exception as e:
        print(f"[RAG] Retrieval query failed: {e}")
        return ""


async def chat(
    filename: str,
    question: str,
    history: list[dict] | None = None,
    voice: bool = False,
) -> str:
    """RAG chat about a paper using Vertex AI RAG Engine for grounded generation."""
    row = fetch_paper_context(filename)
    if not row:
        return (
            "I don't have any analysis data for this paper yet. "
            "Please run the verification first, and then we can dig into the details together."
        )

    # Ensure paper analysis is uploaded to the RAG corpus
    rag_file_name = _ensure_paper_in_corpus(filename, row)
    corpus_name = _get_or_create_corpus()

    print(f"[RAG] Chat for {filename}, rag_file={rag_file_name}")

    # Retrieve relevant chunks from the corpus
    retrieved_context = _retrieve_from_corpus(question, corpus_name, rag_file_name)

    # Build system prompt with retrieved context
    sys_prompt = _get_system_prompt(voice=voice)
    if retrieved_context:
        sys_prompt += (
            "\n\nBELOW IS RETRIEVED CONTEXT FROM THE PAPER ANALYSIS. "
            "Use this to ground your answers. Do not reveal raw data — synthesize naturally.\n\n"
            + retrieved_context
        )
    else:
        # Fallback: use full context from Supabase
        context = _build_context(row, truncate_paper=True)
        sys_prompt += (
            "\n\nBELOW IS THE PAPER VERIFICATION DATA:\n\n"
            + context
        )

    model = GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=sys_prompt,
    )

    # Convert history dicts → Vertex AI Content objects
    chat_history = []
    if history:
        for msg in history:
            role = "user" if msg.get("role") == "user" else "model"
            chat_history.append(
                Content(role=role, parts=[Part.from_text(msg["content"])])
            )

    chat_session = model.start_chat(history=chat_history)

    try:
        response = await chat_session.send_message_async(question)
        content = response.text or ""
        print(f"[RAG] Response length: {len(content)} chars")
        if content.strip():
            return content
    except Exception as e:
        print(f"[RAG] Generation error: {e}")

    # Last resort fallback with shorter context
    print("[RAG] Falling back to shorter context...")
    try:
        short_context = _build_context(row, truncate_paper=True)
        short_sections = [s for s in short_context.split("\n\n") if not s.startswith("PAPER TEXT:")]
        fallback_sys = (
            _get_system_prompt(voice=voice)
            + "\n\nBELOW IS THE PAPER VERIFICATION DATA:\n\n"
            + "\n\n".join(short_sections)
        )
        fallback_model = GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=fallback_sys,
        )
        fallback_history = []
        if history:
            for msg in history[-6:]:
                role = "user" if msg.get("role") == "user" else "model"
                fallback_history.append(
                    Content(role=role, parts=[Part.from_text(msg["content"])])
                )
        fallback_chat = fallback_model.start_chat(history=fallback_history)
        response = await fallback_chat.send_message_async(question)
        content = response.text or ""
        if content.strip():
            return content
    except Exception as e:
        print(f"[RAG] Fallback also failed: {e}")

    return "I'm having trouble processing that right now. Could you try rephrasing your question?"
