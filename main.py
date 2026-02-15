import sys
import os
import json
import re
import tempfile
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
from services.rag_chat import chat as rag_chat
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


@app.post("/api/check")
async def check_paper(file: UploadFile = File(...)):
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

        # Step 3: Save to Supabase for RAG pipeline
        try:
            save_analysis(
                filename=file.filename,
                paper_text=paper_text,
                page_count=parsed["pages"],
                result=result,
            )
        except Exception as store_err:
            print(f"[Supabase] Failed to save analysis: {store_err}")

        # Step 4: Build response
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

        return {
            "verdict": structured.get("verdict", "unknown"),
            "summary": structured.get("summary", ""),
            "code_results": code_results,
            "math": subagent.get("math", {}),
            "math_chunks": math_chunks,
            "coding_review": subagent.get("coding", {}),
            "planner_steps": subagent.get("planner", {}).get("steps", []),
            "pages": parsed["pages"],
            "page_texts": parsed.get("page_texts", []),
            "page_layouts": parsed.get("page_layouts", []),
            "filename": file.filename,
        }

    except Exception as e:
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
