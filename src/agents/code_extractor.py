from schemas import AgentFState, CodeExtractorOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response

async def CodeExtractorAgent(state: AgentFState) -> AgentFState:
    paper_text = state.get("query") or ""
    print("[CodeExtractor] input length:", len(paper_text), "chars")

    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at extracting code from academic papers. "
            "Given the full text of a research paper, identify and extract ALL code chunks, "
            "including algorithms, code listings, function definitions, "
            "and inline code snippets. Code in algorithm boxes that is written in Python or other runnable syntax should be extracted. "
            "Include only chunks that are executable in a real language (python, javascript, typescript, bash, shell). "
            "Do not include pseudocode, algorithm notation, or non-executable snippets—omit them entirely; only output runnable code.\n"
            "Include code even if it looks incomplete or references undefined variables; do not exclude snippets for that reason.\n"
            "For each chunk use only the keys: code, language, context (no extra keys). Identify the programming language and provide a brief description.\n"
            "When you see a function definition followed by assert or test lines (e.g. assert f(...) == ??), "
            "extract only the function; do not create a separate chunk for the assert/test line.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"chunks": [{{"code": "...", "language": "python", "context": "description"}}, ...]}}\n'
            "If no code is found, return: {{}}"
        )),
        ("human", "Extract all code chunks from the following paper text:\n\n{paper_text}")
    ])

    chunks = []
    try:
        chain = extractor_prompt | llm
        response = await chain.ainvoke({"paper_text": paper_text})
        raw = (response.content or "").strip()
        if raw:
            result = parse_json_response(raw, CodeExtractorOutput)
            chunks = [
                {"code": str(c.code), "language": str(c.language), "context": str(c.context)}
                for c in (result.chunks or [])
            ]
    except Exception as e:
        print("[CodeExtractor] failed:", e)
        chunks = []

    return {
        "subagent_responses": {
            "code_extractor": {"chunks": chunks}
        }
    }
