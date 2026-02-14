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

class CodeExecutionResult(BaseModel):
    code: str
    language: str
    ran_successfully: bool
    stdout: str
    stderr: str
    analysis: str

class ReplannerOutput(BaseModel):
    results: List[CodeExecutionResult]
    summary: str

class MathChunk(BaseModel):
    latex: str              # The equation (LaTeX or plain text)
    context: str            # Surrounding text explaining the equation
    equation_type: str      # "definition", "theorem", "loss_function", "derivation", etc.

class MathExtractorOutput(BaseModel):
    chunks: List[MathChunk]

class MathAnalysisOutput(BaseModel):
    is_mathematically_valid: bool
    is_consistent_with_claims: bool
    issues: List[str]
    explanation: str
    verified_steps: List[str]  # Step-by-step verification reasoning