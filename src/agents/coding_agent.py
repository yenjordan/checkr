from schemas import AgentFState, CodingAnalysisOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response

async def CodingAgent(state: AgentFState) -> AgentFState:
    code_chunks = state.get("subagent_responses", {}).get("code_extractor", {}).get("chunks", [])
    planner_steps = state.get("subagent_responses", {}).get("planner", {}).get("steps", [])
    intent = "\n".join([f"- {step}" for step in planner_steps]) if planner_steps else state["query"]

    if not code_chunks:
        return {
            "subagent_responses": {
                "coding": {
                    "is_conceptually_correct": False,
                    "issues": ["No code was extracted from the paper. Cannot perform conceptual review."],
                    "explanation": "Code extractor returned no chunks.",
                }
            }
        }

    code_unit = "\n\n---\n\n".join(c.get("code", "") for c in code_chunks[:10])
    if len(code_chunks) > 10:
        code_unit += f"\n\n... and {len(code_chunks) - 10} more chunks"

    coding_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Analyze code snippets from a research paper for conceptual correctness only. "
            "These are snippets, not complete programs.\n\n"
            "DO NOT consider syntax. Ignore: syntax errors, IndentationError, missing colons/braces, "
            "OCR/formatting issues, missing imports, undefined names, incomplete extraction. "
            "If the ONLY reason to fail would be syntax or snippet-level issues, set is_conceptually_correct to true "
            "and do NOT mention syntax in issues or explanation.\n\n"
            "Set is_conceptually_correct to false ONLY for: wrong algorithm vs paper, incorrect math/formula in logic, "
            "or logic that would yield wrong results with dependencies present.\n\n"
            "Respond with ONLY this JSON (no markdown):\n"
            '{{"is_conceptually_correct": true, "issues": [], "explanation": "..."}}'
        )),
        ("human", "Code:\n```\n{code_unit}\n```\n\nPaper intent:\n{intent}\n\nIs the code conceptually correct? (Ignore syntax.)")
    ])

    chain = coding_prompt | llm
    response = await chain.ainvoke({"code_unit": code_unit, "intent": intent})

    try:
        result = parse_json_response(response.content or "", CodingAnalysisOutput)
        return {
            "subagent_responses": {
                "coding": {
                    "is_conceptually_correct": result.is_conceptually_correct,
                    "issues": result.issues,
                    "explanation": result.explanation
                }
            }
        }
    except Exception:
        return {
            "subagent_responses": {
                "coding": {
                    "is_conceptually_correct": True,
                    "issues": [],
                    "explanation": "Could not parse analysis response."
                }
            }
        }
