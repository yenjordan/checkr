from schemas import AgentFState, CodingAnalysisOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm

async def CodingAgent(state: AgentFState) -> AgentFState:
    code_unit = state["query"]
    planner_steps = state.get("subagent_responses", {}).get("planner", {}).get("steps", [])
    intent = "\n".join([f"- {step}" for step in planner_steps]) if planner_steps else state["query"]
    
    coding_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert code reviewer who analyzes code for conceptual correctness. Compare the provided code unit with the stated intent/requirements and determine if the code conceptually matches what it's supposed to do. Look for logical errors, incorrect algorithms, or mismatches between intent and implementation."),
        ("human", "Code unit:\n```\n{code_unit}\n```\n\nIntent/Requirements:\n{intent}\n\nAnalyze if this code is conceptually correct given the intent.")
    ])
    
    coding_analyzer = coding_prompt | llm.with_structured_output(CodingAnalysisOutput)
    result = await coding_analyzer.ainvoke({"code_unit": code_unit, "intent": intent})
    
    return {
        "subagent_responses": {
            "coding": {
                "is_conceptually_correct": result.is_conceptually_correct,
                "issues": result.issues,
                "explanation": result.explanation
            }
        }
    }