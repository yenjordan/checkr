from langgraph.graph import StateGraph, END, START
from agents.planner import PlannerAgent
from agents.coding_agent import CodingAgent
from agents.replanner import ReplannerAgent
from schemas import AgentFState


workflow = StateGraph(AgentFState, config = {})

workflow.add_node("planner", PlannerAgent)
workflow.add_node("coding", CodingAgent)
workflow.add_node("replanner", ReplannerAgent)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "coding")
workflow.add_edge("coding", "replanner")
workflow.add_edge("replanner", END)