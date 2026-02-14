from typing import Dict, Any, Optional, TypedDict, List
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel

class AgentFState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    subagent_responses: Dict[str, Any]
    entities: Optional[Dict]
    structured_response: Optional[Any]
    remaining_tries: int

class PlannerOutput(BaseModel):
    steps: List[str]

class CodingAnalysisOutput(BaseModel):
    is_conceptually_correct: bool
    issues: List[str]
    explanation: str

class CodeChunk(BaseModel):
    code: str
    language: str
    context: str

class CodeExtractorOutput(BaseModel):
    chunks: List[CodeChunk]