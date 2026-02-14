from schemas import AgentFState, PlannerOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm

async def PlannerAgent(state: AgentFState) -> AgentFState:
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant who generates plans to achieve user goals. Depending on whether the user input is a code unit or mathematics, create a plan with step(s) to verify the validity of the code to ensure it is conceptually correct."),
        ("human", "{query}")
    ])
    
    planner = planner_prompt | llm.with_structured_output(PlannerOutput)
    
    # Invoke planner
    result = await planner.ainvoke({"query": state["query"]})
    
    return {
        "subagent_responses": {
            "planner": {"steps": result.steps}
        }
    }