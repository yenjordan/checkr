import sys
import tempfile
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from graph import workflow
from code_extract import extract_pdf, parse_document

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ROOT_DIR = Path(__file__).resolve().parent.parent
app.mount("/assets", StaticFiles(directory=ROOT_DIR / "assets"), name="assets")


@app.get("/")
async def root():
    return FileResponse(ROOT_DIR / "index.html")


@app.post("/api/verify")
@app.post("/api/check")
async def verify_paper(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        paper_text = parse_document(extract_pdf(tmp_path))["full_text"]
        result = await workflow.compile().ainvoke({
            "messages": [],
            "query": paper_text,
            "subagent_responses": {},
            "entities": None,
            "structured_response": None,
            "remaining_tries": 3,
        })
        resp = result.get("subagent_responses", {})
        replanner = resp.get("replanner", {})
        sympy_verify = resp.get("sympy_verify", {})
        math_extractor_chunks = resp.get("math_extractor", {}).get("chunks", [])
        sympy_chunk_results = sympy_verify.get("chunk_results", [])

        math_chunks = []
        for i, mc in enumerate(math_extractor_chunks):
            merged = dict(mc)
            if i < len(sympy_chunk_results):
                sr = sympy_chunk_results[i]
                merged["verification_status"] = sr.get("status")
                merged["proof"] = sr.get("proof", {})
                merged["lean_success"] = sr.get("lean_success")
                merged["lean_code"] = sr.get("lean_code", "")
                merged["lean_error"] = sr.get("lean_error", "")
            math_chunks.append(merged)

        return {
            "status": resp.get("status"),
            "verdict": result.get("structured_response", {}).get("verdict"),
            "summary": result.get("structured_response", {}).get("summary"),
            "planner_steps": resp.get("planner", {}).get("steps", []),
            "coding": resp.get("coding"),
            "coding_review": resp.get("coding"),
            "code_results": replanner.get("results", []),
            "math": resp.get("math"),
            "math_chunks": math_chunks,
            "sympy_verify": sympy_verify,
        }
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
