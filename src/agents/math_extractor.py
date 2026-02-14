from schemas import AgentFState, MathExtractorOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response

async def MathExtractorAgent(state: AgentFState) -> AgentFState:
    paper_text = state["query"]

    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at extracting mathematical content from academic papers. "
            "Given the full text of a research paper, identify and extract ALL mathematical content: "
            "equations, formulas, theorems, lemmas, proofs, and derivations. "
            "For each chunk, identify the equation type (definition, theorem, loss function, etc.) "
            "and provide the surrounding context explaining what the math represents.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"chunks": [{{"latex": "E = mc^2", "context": "description", "equation_type": "definition"}}, ...]}}\n'
            "If no math is found, return: {{}}"
        )),
        ("human", "Extract all mathematical content from:\n\n{paper_text}")
    ])

    chain = extractor_prompt | llm
    response = await chain.ainvoke({"paper_text": paper_text})

    try:
        result = parse_json_response(response.content, MathExtractorOutput)
        chunks = [
            {"latex": chunk.latex, "context": chunk.context, "equation_type": chunk.equation_type}
            for chunk in result.chunks
        ]
    except Exception:
        chunks = []

    return {
        "subagent_responses": {
            "math_extractor": {"chunks": chunks}
        }
    }
