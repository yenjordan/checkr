from schemas import AgentFState, MathAnalysisOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm_reasoning

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
            "Provide step-by-step verification reasoning."
        )),
        ("human", "Math chunks:\n{math_chunks}\n\nVerification goals:\n{intent}")
    ])

    analyzer = math_prompt | llm_reasoning.with_structured_output(MathAnalysisOutput)
    result = await analyzer.ainvoke({"math_chunks": math_chunks, "intent": intent})

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
