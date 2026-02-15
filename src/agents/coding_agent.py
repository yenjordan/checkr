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
            "You are an expert code reviewer who analyzes code from research papers for FUNDAMENTAL correctness. "
            "Your job is to determine whether the code is conceptually sound and aligned with the paper's claims.\n\n"
            "CRITICAL: These are code SNIPPETS extracted from a paper — they are NOT complete programs. "
            "Do NOT flag issues that are artifacts of incomplete extraction, such as:\n"
            "- Missing imports, undefined variables, or missing helper functions (these exist elsewhere in the paper/codebase)\n"
            "- Minor syntax issues from OCR or formatting (missing colons, indentation)\n"
            "- Missing context or setup code\n"
            "- Variables referenced but not defined in the snippet\n\n"
            "DO flag these HIGH-VALUE issues:\n"
            "- Algorithmic errors: the logic doesn't match what the paper claims it does\n"
            "- Incorrect math/formulas translated to code (wrong operations, off-by-one in critical computations)\n"
            "- Fundamental mismatches between the code's behavior and the paper's stated methodology\n"
            "- Hardcoded values that contradict the paper's described parameters\n"
            "- Logic that would produce incorrect results even with all dependencies present\n\n"
            "If the code snippets are fundamentally in the right direction for the paper's claims, "
            "mark is_conceptually_correct as true even if there are minor snippet-level issues.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"is_conceptually_correct": true, "issues": ["issue1", ...], "explanation": "..."}}'
        )),
        ("human", "Code snippets from the paper:\n```\n{code_unit}\n```\n\nPaper's claims/methodology:\n{intent}\n\nAnalyze whether this code is fundamentally correct and aligned with the paper's claims.")
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
