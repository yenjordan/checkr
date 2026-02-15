from typing import Dict, Any, Optional, TypedDict, List
from typing_extensions import Annotated
from langgraph.graph.message import add_messages
from pydantic import BaseModel

def reduce_subagent_responses(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two subagent_responses dictionaries"""
    result = dict(left) if left else {}
    if right:
        result.update(right)
    return result

class AgentFState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    subagent_responses: Annotated[Dict[str, Any], reduce_subagent_responses]
    entities: Optional[Dict]
    structured_response: Optional[Any]
    remaining_tries: int

class PlannerOutput(BaseModel):
    steps: List[str]

class IssueWithSeverity(BaseModel):
    message: str
    severity: str  # "critical" or "warning"

class ChunkAnalysisLLMResponse(BaseModel):
    """Schema for LLM response when analyzing a single chunk (no chunk_index)"""
    is_correct: bool
    issues: List[IssueWithSeverity]
    explanation: str

class ChunkAnalysis(BaseModel):
    """Schema for internal storage with chunk_index added"""
    chunk_index: int
    is_correct: bool
    issues: List[IssueWithSeverity]
    explanation: str

class CodingAnalysisOutput(BaseModel):
    is_conceptually_correct: bool
    chunk_results: List[ChunkAnalysis]  # Detailed per-chunk results
    issues: List[str]  # Only critical issues aggregated
    explanation: str

class CodeChunk(BaseModel):
    code: str
    language: str
    context: str

class CodeExtractorOutput(BaseModel):
    chunks: List[CodeChunk] = []

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
    source_text: str = ""   # Exact raw text as it appears in the paper (for highlighting)

class MathExtractorOutput(BaseModel):
    chunks: List[MathChunk] = []

class MathAnalysisOutput(BaseModel):
    is_mathematically_valid: bool
    is_consistent_with_claims: bool
    issues: List[str]
    explanation: str
    verified_steps: List[str]  # Step-by-step verification reasoning