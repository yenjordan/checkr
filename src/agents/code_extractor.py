from schemas import AgentFState, CodeExtractorOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm

async def CodeExtractorAgent(state: AgentFState) -> AgentFState:
    paper_text = state["query"]

    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at extracting code from academic papers. "
            "Given the full text of a research paper, identify and extract ALL code chunks, "
            "including algorithms, pseudocode, code listings, function definitions, "
            "and inline code snippets. For each chunk, identify the programming language "
            "(use 'pseudocode' if it is not a real language) and provide a brief description "
            "of what the code does based on the surrounding context in the paper."
        )),
        ("human", "Extract all code chunks from the following paper text:\n\n{paper_text}")
    ])

    extractor = extractor_prompt | llm.with_structured_output(CodeExtractorOutput)
    result = await extractor.ainvoke({"paper_text": paper_text})

    return {
        "subagent_responses": {
            "code_extractor": {
                "chunks": [
                    {"code": chunk.code, "language": chunk.language, "context": chunk.context}
                    for chunk in result.chunks
                ]
            }
        }
    }
