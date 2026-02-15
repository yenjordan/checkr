import sys
import os
import tempfile
import traceback
from pathlib import Path

# Add src/ to import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import ocrmypdf

from dotenv import load_dotenv
load_dotenv()

from graph import workflow, MAX_RETRIES
from services.code_extract import extract_pdf, parse_document
from services.supabase_store import save_analysis
from services.rag_chat import chat as rag_chat, fetch_paper_context, upload_paper_to_corpus, _build_context
from agents.chunk_locator import locate_chunks
from pydantic import BaseModel

app = FastAPI(title="CHECKR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile the LangGraph workflow
graph = workflow.compile()


def _make_json_serializable(obj):
    """Recursively ensure object is JSON-serializable (str keys, no custom types)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    return str(obj)


@app.post("/api/check")
async def check_paper(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Accept a PDF, run OCR, feed into the agent workflow, return results."""

    # Save uploaded file to temp location
    suffix = Path(file.filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 1: OCR via Document AI
        document = extract_pdf(tmp_path)
        parsed = parse_document(document)
        paper_text = parsed["full_text"]

        if not paper_text or not paper_text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Could not extract text from the PDF."},
            )

        # Step 2: Run the LangGraph agent workflow
        initial_state = {
            "messages": [],
            "query": paper_text,
            "subagent_responses": {},
            "entities": parsed.get("entities"),
            "structured_response": None,
            "remaining_tries": MAX_RETRIES,
        }

        result = await graph.ainvoke(initial_state)

        # Step 3: per-chunk SymPy verification 
        raw_math_chunks = result.get("subagent_responses", {}).get("math_extractor", {}).get("chunks", [])
        sympy_chunk_results = result.get("subagent_responses", {}).get("sympy_verify", {}).get("chunk_results", [])
        lean_chunk_results = result.get("subagent_responses", {}).get("lean_verify", {}).get("chunk_results", [])
        enriched_math_chunks = []
        for i, chunk in enumerate(raw_math_chunks):
            enriched = dict(chunk) if isinstance(chunk, dict) else {}
            if not isinstance(enriched, dict):
                enriched = {}
            if i < len(sympy_chunk_results):
                r = sympy_chunk_results[i]
                enriched["sympy_translation"] = r.get("sympy_translation", {})
                enriched["verification_status"] = r.get("status", "unknown")
                enriched["proof"] = r.get("proof", {})
                enriched["verification_error"] = r.get("error", None)
            else:
                enriched["sympy_translation"] = None
                enriched["verification_status"] = None
                enriched["proof"] = None
                enriched["verification_error"] = None
            # Add Lean verification
            if i < len(lean_chunk_results):
                l = lean_chunk_results[i]
                enriched["lean_code"] = l.get("lean_code", "")
                enriched["lean_success"] = l.get("ran_successfully", False)
                enriched["lean_error"] = l.get("stderr", "")
            else:
                enriched["lean_code"] = None
                enriched["lean_success"] = None
                enriched["lean_error"] = None
            enriched_math_chunks.append(enriched)

        # Step 4: Save to Supabase
        try:
            save_analysis(
                filename=file.filename,
                paper_text=paper_text,
                page_count=parsed["pages"],
                result=result,
                math_chunks=enriched_math_chunks,
            )
        except Exception as store_err:
            print(f"[Supabase] Failed to save analysis: {store_err}")

        # Step 4b: Upload to Vertex AI RAG corpus in background (non-blocking)
        def _bg_rag_upload(fname):
            try:
                r = fetch_paper_context(fname)
                if r:
                    ctx = _build_context(r, truncate_paper=False)
                    upload_paper_to_corpus(fname, ctx)
            except Exception as rag_err:
                print(f"[RAG] Failed to upload to corpus: {rag_err}")

        if background_tasks:
            background_tasks.add_task(_bg_rag_upload, file.filename)

        # Step 5: Build response
        structured = result.get("structured_response", {})
        subagent = result.get("subagent_responses", {})

        # Locate chunks in the Document AI layout for precise highlighting
        math_chunks = subagent.get("math_extractor", {}).get("chunks", [])
        code_results = subagent.get("replanner", {}).get("results", [])
        if parsed.get("page_layouts") and (math_chunks or code_results):
            try:
                await locate_chunks(
                    parsed["page_layouts"],
                    math_chunks,
                    code_results,
                )
            except Exception as loc_err:
                print(f"[ChunkLocator] Failed: {loc_err}")

        payload = {
            "verdict": structured.get("verdict", "unknown"),
            "summary": structured.get("summary", ""),
            "code_results": code_results,
            "math": subagent.get("math", {}),
            "math_chunks": enriched_math_chunks,
            "sympy_verify": subagent.get("sympy_verify", {}),
            "lean_verify": subagent.get("lean_verify", {}),
            "coding_review": subagent.get("coding", {}),
            "planner_steps": subagent.get("planner", {}).get("steps", []),
            "pages": parsed["pages"],
            "page_texts": parsed.get("page_texts", []),
            "page_layouts": parsed.get("page_layouts", []),
            "filename": file.filename,
        }
        return _make_json_serializable(payload)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        os.unlink(tmp_path)


class ChatRequest(BaseModel):
    filename: str
    question: str
    history: list[dict] = []


@app.get("/api/analysis")
async def get_analysis(filename: str):
    """Return the latest stored analysis for a filename. Use for loading results w/o rerunning check"""
    row = fetch_paper_context(filename)
    if not row:
        return JSONResponse(status_code=404, content={"error": "No analysis found for this filename."})
    return {
        "verdict": row.get("verdict", "unknown"),
        "summary": row.get("summary", ""),
        "code_results": row.get("code_execution_results", []),
        "math": row.get("math_analysis", {}),
        "math_chunks": row.get("math_chunks", []),
        "coding_review": row.get("coding_review", {}),
        "planner_steps": row.get("planner_steps", []),
        "pages": row.get("page_count"),
        "filename": row.get("filename", filename),
    }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """RAG chatbot: answer questions about a paper using stored analysis."""
    try:
        answer = await rag_chat(
            filename=req.filename,
            question=req.question,
            history=req.history,
        )
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


# ── Hume EVI Voice AI Integration ──────────────────────────────────
# Browser fetches an access token + config_id, then connects directly
# to Hume's EVI WebSocket with Gemini as the LLM.

_hume_config_id: str | None = None  # cached after first creation


async def _get_or_create_hume_config() -> str:
    """Create a Hume EVI config with Gemini as the LLM (cached after first call)."""
    import httpx

    global _hume_config_id
    if _hume_config_id:
        return _hume_config_id

    api_key = os.environ.get("HUME_API_KEY", "")

    async with httpx.AsyncClient() as client:
        # Check if we already have a CHECKR config
        list_resp = await client.get(
            "https://api.hume.ai/v0/evi/configs",
            headers={"X-Hume-Api-Key": api_key},
            params={"page_size": 50},
        )
        if list_resp.status_code == 200:
            configs = list_resp.json().get("configs_page", [])
            for cfg in configs:
                if cfg.get("name") == "CHECKR Voice":
                    _hume_config_id = cfg["id"]
                    print(f"[Hume] Reusing config: {_hume_config_id}")
                    return _hume_config_id

        # Create a new config with Gemini + EVI 3 voice
        create_resp = await client.post(
            "https://api.hume.ai/v0/evi/configs",
            headers={
                "X-Hume-Api-Key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "evi_version": "3",
                "name": "CHECKR Voice",
                "voice": {
                    "provider": "HUME_AI",
                    "id": "9e068547-5ba4-4c8e-8e03-69282a008f04",
                },
                "language_model": {
                    "model_provider": "GOOGLE",
                    "model_resource": "gemini-2.0-flash",
                },
            },
        )
        if create_resp.status_code in (200, 201):
            data = create_resp.json()
            _hume_config_id = data.get("id")
            print(f"[Hume] Created config: {_hume_config_id}")
            return _hume_config_id

        print(f"[Hume] Config creation failed: {create_resp.status_code} {create_resp.text}")
        return ""


@app.post("/api/hume/token")
async def hume_access_token():
    """Generate a short-lived Hume access token + config_id for the browser."""
    import httpx
    import base64

    api_key = os.environ.get("HUME_API_KEY", "")
    secret_key = os.environ.get("HUME_SECRET_KEY", "")
    if not api_key or not secret_key:
        return JSONResponse(status_code=500, content={"error": "HUME_API_KEY / HUME_SECRET_KEY not configured."})

    credentials = base64.b64encode(f"{api_key}:{secret_key}".encode()).decode()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.hume.ai/oauth2-cc/token",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data="grant_type=client_credentials",
        )
        if resp.status_code != 200:
            return JSONResponse(status_code=resp.status_code, content={"error": resp.text})

        token_data = resp.json()

    # Ensure the Gemini config exists
    config_id = await _get_or_create_hume_config()

    return {
        "access_token": token_data.get("access_token"),
        "config_id": config_id,
    }


@app.get("/api/hume/context")
async def hume_paper_context(filename: str):
    """Build a voice-optimized system prompt + context for Hume EVI."""
    from services.rag_chat import _build_context, _get_system_prompt

    row = fetch_paper_context(filename)
    if not row:
        return JSONResponse(status_code=404, content={"error": "No analysis found."})

    system_prompt = _get_system_prompt(voice=True)
    paper_context = _build_context(row)

    return {
        "system_prompt": system_prompt,
        "context": paper_context,
    }


@app.post("/api/ocr")
async def ocr_pdf(file: UploadFile = File(...)):
    """Run OCR on a PDF and return a searchable version with selectable text."""
    suffix = Path(file.filename).suffix or ".pdf"
    input_path = None
    output_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            input_path = tmp.name

        output_path = input_path + "_ocr.pdf"

        ocrmypdf.ocr(
            input_path,
            output_path,
            force_ocr=True,
            optimize=1,
            skip_big=50,
            jobs=2,
        )

        ocr_bytes = Path(output_path).read_bytes()
        return Response(content=ocr_bytes, media_type="application/pdf")

    except Exception as e:
        print(f"[OCR] Failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)


# Serve static files (assets/) and index.html
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/")
async def serve_index():
    return FileResponse("index.html")
