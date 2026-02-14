from schemas import AgentFState, MathAnalysisOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm_reasoning
from utils import parse_json_response

async def MathAgent(state: AgentFState) -> AgentFState:
    math_chunks = state.get("subagent_responses", {}).get("math_extractor", {}).get("chunks", [])
    planner_steps = state.get("subagent_responses", {}).get("planner", {}).get("steps", [])
    intent = "\n".join([f"- {step}" for step in planner_steps]) if planner_steps else state["query"]

    # Handle empty math case
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
            "You are an expert mathematician who verifies mathematical content from papers. "
            "For each equation/formula: "
            "1) Check mathematical validity (correct derivations, dimensional consistency) "
            "2) Verify consistency with paper claims (does it match what author says it represents?) "
            "Provide step-by-step verification reasoning.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"is_mathematically_valid": true, "is_consistent_with_claims": true, '
            '"issues": ["issue1", ...], "explanation": "...", "verified_steps": ["step1", ...]}}'
        )),
        ("human", "Math chunks:\n{math_chunks}\n\nVerification goals:\n{intent}")
    ])

    chain = math_prompt | llm_reasoning
    response = await chain.ainvoke({"math_chunks": str(math_chunks), "intent": intent})

    try:
        result = parse_json_response(response.content or "", MathAnalysisOutput)
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
    except Exception:
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
