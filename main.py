import sys
import os
import json
import tempfile
from pathlib import Path

# Add src/ to import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv()

from graph import workflow, MAX_RETRIES
from services.code_extract import extract_pdf, parse_document

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

        # Step 3: Build response
        structured = result.get("structured_response", {})
        subagent = result.get("subagent_responses", {})

        return {
            "verdict": structured.get("verdict", "unknown"),
            "summary": structured.get("summary", ""),
            "code_results": subagent.get("replanner", {}).get("results", []),
            "math": subagent.get("math", {}),
            "coding_review": subagent.get("coding", {}),
            "planner_steps": subagent.get("planner", {}).get("steps", []),
            "pages": parsed["pages"],
            "filename": file.filename,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )
    finally:
        os.unlink(tmp_path)


# Serve static files (assets/) and index.html
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/")
async def serve_index():
    return FileResponse("index.html")
