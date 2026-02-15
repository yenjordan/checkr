import re
import json

from schemas import AgentFState, CodeExtractorOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response

MIN_CODE_LEN = 50

async def CodeExtractorAgent(state: AgentFState) -> AgentFState:
    paper_text = state.get("query") or ""
    print("[CodeExtractor] input length:", len(paper_text), "chars", flush=True)

    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at extracting code from academic papers. "
            "Given the full text of a research paper, identify and extract ALL code chunks, "
            "including algorithms, code listings, function definitions, and inline code snippets. "
            "Extract code from Algorithm boxes, listings, and any block that looks like runnable code (Python, pseudocode that is close to Python, etc.). "
            "Include only chunks that are executable or nearly executable in a real language (python, javascript, typescript, bash, shell). "
            "Do not include pure pseudocode with no syntax—but DO include algorithm blocks that use Python-like syntax (def, for, if, =, etc.). "
            "When in doubt, include a chunk; it is better to extract a borderline block than to omit it.\n"
            "Include code even if it looks incomplete or references undefined variables.\n"
            "CRITICAL: NEVER abbreviate or omit code. Always include the COMPLETE, FULL code for every chunk. "
            "Do not write '... (omitted for brevity)', '... (rest of code)', '# ... rest omitted', or any similar placeholder—extract the entire code block only. "
            "If the full code is not in the paper, extract only what is actually written; do not invent placeholders.\n"
            "For each chunk use only the keys: code, language, context (no extra keys). Identify the programming language and provide a brief description.\n"
            "When you see a function definition followed by assert or test lines, extract only the function (in full).\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"chunks": [{{"code": "...", "language": "python", "context": "description"}}, ...]}}\n'
            "If the paper truly has no code at all (no algorithms, no listings, no snippets), return: {{}}"
        )),
        ("human", "Extract all code chunks from the following paper text:\n\n{paper_text}")
    ])

    chunks = []
    try:
        chain = extractor_prompt | llm
        response = await chain.ainvoke({"paper_text": paper_text})
        raw = (response.content or "").strip()
        if raw:
            try:
                result = parse_json_response(raw, CodeExtractorOutput)
                chunks = [
                    {"code": str(c.code), "language": str(c.language), "context": str(c.context)}
                    for c in (result.chunks or [])
                    if len(str(c.code).strip()) >= MIN_CODE_LEN
                ]
                print("[CodeExtractor] parsed", len(chunks), "chunks (excluding < %d chars)" % MIN_CODE_LEN, flush=True)
            except Exception as e:
                print("[CodeExtractor] parse failed:", e, flush=True)
                match = re.search(r'"chunks"\s*:\s*\[', raw)
                if match:
                    start = raw.find("[", match.start())
                    depth = 0
                    for i in range(start, len(raw)):
                        c = raw[i]
                        if c == "[":
                            depth += 1
                        elif c == "]":
                            depth -= 1
                            if depth == 0:
                                arr_str = raw[start : i + 1]
                                for parse_fn in (json.loads, None):
                                    try:
                                        if parse_fn is None:
                                            import json_repair
                                            arr = json_repair.loads(arr_str)
                                        else:
                                            arr = parse_fn(arr_str)
                                        for item in arr:
                                            if isinstance(item, dict) and item.get("code"):
                                                code = str(item.get("code", ""))
                                                if len(code.strip()) >= MIN_CODE_LEN:
                                                    chunks.append({
                                                        "code": code,
                                                        "language": str(item.get("language", "python")),
                                                        "context": str(item.get("context", "")),
                                                    })
                                        if chunks:
                                            print("[CodeExtractor] fallback parsed", len(chunks), "chunks", flush=True)
                                        break
                                    except Exception:
                                        if parse_fn is None:
                                            pass
                                        continue
                                break
                if not chunks and raw.strip().startswith("{"):
                    try:
                        import json_repair
                        data = json_repair.loads(raw)
                        arr = data.get("chunks") if isinstance(data, dict) else []
                        for item in arr or []:
                            if isinstance(item, dict) and item.get("code"):
                                code = str(item.get("code", ""))
                                if len(code.strip()) >= MIN_CODE_LEN:
                                    chunks.append({
                                        "code": code,
                                        "language": str(item.get("language", "python")),
                                        "context": str(item.get("context", "")),
                                    })
                        if chunks:
                            print("[CodeExtractor] json_repair fallback parsed", len(chunks), "chunks", flush=True)
                    except Exception:
                        pass
    except Exception as e:
        print("[CodeExtractor] failed:", e, flush=True)
        chunks = []

    return {
        "subagent_responses": {
            "code_extractor": {"chunks": chunks}
        }
    }
