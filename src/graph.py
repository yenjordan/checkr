from langgraph.graph import StateGraph, END, START
from agents.planner import PlannerAgent
from agents.code_extractor import CodeExtractorAgent
from agents.math_extractor import MathExtractorAgent
from agents.coding_agent import CodingAgent
from agents.math_agent import MathAgent
from agents.replanner import ReplannerAgent
from schemas import AgentFState

MAX_RETRIES = 3


def check_replanner_result(state: AgentFState) -> str:
    results = (
        state.get("subagent_responses", {})
        .get("replanner", {})
        .get("results", [])
    )

    all_passed = all(r.get("ran_successfully", False) for r in results)

    if all_passed:
        return "success"

    if state.get("remaining_tries", 0) <= 0:
        return "max_attempts"

    return "retry"


def decrement_retries(state: AgentFState) -> AgentFState:
    return {
        "remaining_tries": state.get("remaining_tries", MAX_RETRIES) - 1,
        "subagent_responses": {
            "status": "retrying"
        }
    }


def mark_hard_to_verify(state: AgentFState) -> AgentFState:
    return {
        "subagent_responses": {
            "status": "hard_to_verify"
        },
        "structured_response": {
            "verdict": "hard_to_verify",
            "summary": state.get("subagent_responses", {}).get("replanner", {}).get("summary", ""),
        }
    }


def mark_success(state: AgentFState) -> AgentFState:
    return {
        "subagent_responses": {
            "status": "verified"
        },
        "structured_response": {
            "verdict": "verified",
            "summary": state.get("subagent_responses", {}).get("replanner", {}).get("summary", ""),
            "results": state.get("subagent_responses", {}).get("replanner", {}).get("results", []),
        }
    }


workflow = StateGraph(AgentFState, config={})

workflow.add_node("planner", PlannerAgent)
workflow.add_node("code_extractor", CodeExtractorAgent)
workflow.add_node("math_extractor", MathExtractorAgent)
workflow.add_node("coding", CodingAgent)
workflow.add_node("math", MathAgent)
workflow.add_node("replanner", ReplannerAgent)
workflow.add_node("decrement_retries", decrement_retries)
workflow.add_node("mark_hard_to_verify", mark_hard_to_verify)
workflow.add_node("mark_success", mark_success)

# Flow: planner → [extractors in parallel] → [analyzers in parallel] → replanner
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "code_extractor")
workflow.add_edge("planner", "math_extractor")
workflow.add_edge("code_extractor", "coding")
workflow.add_edge("math_extractor", "math")
workflow.add_edge("coding", "replanner")
workflow.add_edge("math", "replanner")

workflow.add_conditional_edges(
    "replanner",
    check_replanner_result,
    {
        "success": "mark_success",
        "retry": "decrement_retries",
        "max_attempts": "mark_hard_to_verify",
    },
)

workflow.add_edge("decrement_retries", "replanner")

workflow.add_edge("mark_success", END)
workflow.add_edge("mark_hard_to_verify", END)
