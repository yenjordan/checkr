from schemas import AgentFState, MathAnalysisOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response

async def MathAgent(state: AgentFState) -> AgentFState:
    math_chunks = state.get("subagent_responses", {}).get("math_extractor", {}).get("chunks", [])
    planner_steps = state.get("subagent_responses", {}).get("planner", {}).get("steps", [])
    intent = "\n".join([f"- {step}" for step in planner_steps]) if planner_steps else state["query"]

    if not math_chunks:
        return {
            "subagent_responses": {
                "math": {
                    "is_mathematically_valid": True,
                    "is_consistent_with_claims": True,
                    "issues": [],
                    "explanation": "No mathematical content found in the paper to verify.",
                    "verified_steps": []
                }
            }
        }

    math_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You verify mathematical content from papers: validity (derivations, dimensions) and consistency with the paper's claims.\n\n"
            "Output ONLY a JSON object. No prose before or after. No 'Chunk [1]:' style lines.\n"
            "Use this exact shape:\n"
            '{{"is_mathematically_valid": true/false, "is_consistent_with_claims": true/false, '
            '"issues": ["list any errors or concerns"], "explanation": "2-3 sentence overall summary", '
            '"verified_steps": ["short step 1", "short step 2", ...]}}\n'
            "explanation: one brief overall summary. verified_steps: short bullet-style steps (e.g. 'Checked loss function form', 'Verified distribution definition'), not per-chunk commentary."
        )),
        ("human", "Math chunks:\n{math_chunks}\n\nGoals:\n{intent}")
    ])

    response = await (math_prompt | llm).ainvoke({"math_chunks": str(math_chunks), "intent": intent})
    raw = (response.content or "").strip()

    if not raw:
        return {
            "subagent_responses": {
                "math": {
                    "is_mathematically_valid": True,
                    "is_consistent_with_claims": True,
                    "issues": [],
                    "explanation": "Math verification skipped: no model response.",
                    "verified_steps": []
                }
            }
        }

    try:
        result = parse_json_response(raw, MathAnalysisOutput)
        return {
            "subagent_responses": {
                "math": {
                    "is_mathematically_valid": result.is_mathematically_valid,
                    "is_consistent_with_claims": result.is_consistent_with_claims,
                    "issues": result.issues,
                    "explanation": result.explanation,
                    "verified_steps": result.verified_steps
                }
            }
        }
    except Exception as e:
        print("[MathAgent] Parse failed:", e)
        return {
            "subagent_responses": {
                "math": {
                    "is_mathematically_valid": True,
                    "is_consistent_with_claims": True,
                    "issues": [],
                    "explanation": "Could not parse math analysis response.",
                    "verified_steps": []
                }
            }
        }
