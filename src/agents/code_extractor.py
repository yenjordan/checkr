from schemas import AgentFState, CodeExtractorOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response

async def CodeExtractorAgent(state: AgentFState) -> AgentFState:
    paper_text = state["query"]

    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at extracting code from academic papers. "
            "Given the full text of a research paper, identify and extract ALL code chunks, "
            "including algorithms, pseudocode, code listings, function definitions, "
            "and inline code snippets. For each chunk, identify the programming language "
            "(use 'pseudocode' if it is not a real language) and provide a brief description "
            "of what the code does based on the surrounding context in the paper.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"chunks": [{{"code": "...", "language": "python", "context": "description"}}, ...]}}\n'
            "If no code is found, return: {{}}"
        )),
        ("human", "Extract all code chunks from the following paper text:\n\n{paper_text}")
    ])

    chain = extractor_prompt | llm
    response = await chain.ainvoke({"paper_text": paper_text})

    try:
        result = parse_json_response(response.content, CodeExtractorOutput)
        chunks = [
            {"code": chunk.code, "language": chunk.language, "context": chunk.context}
            for chunk in result.chunks
        ]
    except Exception:
        chunks = []

    return {
        "subagent_responses": {
            "code_extractor": {"chunks": chunks}
        }
    }
