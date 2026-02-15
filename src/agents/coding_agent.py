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
                    "is_conceptually_correct": True,
                    "issues": [],
                    "explanation": "No code found in the paper. Skipping code review.",
                }
            }
        }

    code_unit = "\n\n---\n\n".join(c.get("code", "") for c in code_chunks[:10])
    if len(code_chunks) > 10:
        code_unit += f"\n\n... and {len(code_chunks) - 10} more chunks"

    coding_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert code reviewer who analyzes code for conceptual correctness. "
            "Compare the provided code unit with the stated intent/requirements and determine "
            "if the code conceptually matches what it's supposed to do. Look for logical errors, "
            "incorrect algorithms, or mismatches between intent and implementation.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"is_conceptually_correct": true, "issues": ["issue1", ...], "explanation": "..."}}'
        )),
        ("human", "Code unit:\n```\n{code_unit}\n```\n\nIntent/Requirements:\n{intent}\n\nAnalyze if this code is conceptually correct given the intent.")
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
