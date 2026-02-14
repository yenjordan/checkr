from schemas import AgentFState, MathExtractorOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm

async def MathExtractorAgent(state: AgentFState) -> AgentFState:
    paper_text = state["query"]

    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at extracting mathematical content from academic papers. "
            "Given the full text of a research paper, identify and extract ALL mathematical content: "
            "equations, formulas, theorems, lemmas, proofs, and derivations. "
            "For each chunk, identify the equation type (definition, theorem, loss function, etc.) "
            "and provide the surrounding context explaining what the math represents."
        )),
        ("human", "Extract all mathematical content from:\n\n{paper_text}")
    ])

    extractor = extractor_prompt | llm.with_structured_output(MathExtractorOutput)
    result = extractor.invoke({"paper_text": paper_text})

    new_state = dict(state)
    new_state["subagent_responses"] = dict(new_state.get("subagent_responses", {}))
    new_state["subagent_responses"]["math_extractor"] = {
        "chunks": [
            {"latex": chunk.latex, "context": chunk.context, "equation_type": chunk.equation_type}
            for chunk in result.chunks
        ]
    }

    return new_state
