from schemas import AgentFState, PlannerOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response

async def PlannerAgent(state: AgentFState) -> AgentFState:
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant who generates plans to achieve user goals. "
            "Depending on whether the user input is a code unit or mathematics, "
            "create a plan with step(s) to verify the validity of the code to ensure it is conceptually correct.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"steps": ["step 1", "step 2", ...]}}'
        )),
        ("human", "{query}")
    ])

    chain = planner_prompt | llm
    response = await chain.ainvoke({"query": state["query"]})
    result = parse_json_response(response.content, PlannerOutput)

    return {
        "subagent_responses": {
            "planner": {"steps": result.steps}
        }
    }
