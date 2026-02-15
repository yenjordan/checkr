import sys
import tempfile
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "services"))

from graph import workflow
from code_extract import extract_pdf, parse_document

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/api/verify")
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
            "remaining_tries": 3
        })
        return {
            "status": result.get("subagent_responses", {}).get("status"),
            "verdict": result.get("structured_response", {}).get("verdict"),
            "summary": result.get("structured_response", {}).get("summary"),
            "planner_steps": result.get("subagent_responses", {}).get("planner", {}).get("steps", []),
            "coding": result.get("subagent_responses", {}).get("coding"),
            "math": result.get("subagent_responses", {}).get("math"),
        }
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
