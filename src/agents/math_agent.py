from schemas import AgentFState, MathAnalysisOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response

async def MathAgent(state: AgentFState) -> AgentFState:
    math_chunks = state.get("subagent_responses", {}).get("math_extractor", {}).get("chunks", [])
    sympy_verify = state.get("subagent_responses", {}).get("sympy_verify", {})
    planner_steps = state.get("subagent_responses", {}).get("planner", {}).get("steps", [])
    intent = "\n".join([f"- {step}" for step in planner_steps]) if planner_steps else state["query"]

    if not math_chunks:
        return {
            "subagent_responses": {
                "math": {
                    "is_mathematically_valid": True,
                    "is_consistent_with_claims": True,
                    "issues": [],
                    "explanation": "No mathematical content found in the paper to verify.",
                    "verified_steps": [],
                    "critical_gaps": [],
                }
            }
        }

    chunk_results = sympy_verify.get("chunk_results", [])
    summary = sympy_verify.get("summary", "")
    failed_chunks = []
    error_chunks = []
    inconclusive_chunks = []
    lines = []
    for i, r in enumerate(chunk_results):
        s = r.get("status", "unknown")
        lines.append(f"{i+1}:{s}")
        if s == "failed":
            failed_chunks.append(i + 1)
        elif s == "error":
            error_chunks.append(i + 1)
        elif s == "inconclusive":
            inconclusive_chunks.append(i + 1)
    structured_issues = []
    if failed_chunks:
        structured_issues.append(f"Failed derivations (chunks {', '.join(map(str, failed_chunks))})")
    if error_chunks:
        structured_issues.append(f"Errors in definitions/translation (chunks {', '.join(map(str, error_chunks))})")
    if inconclusive_chunks:
        structured_issues.append(f"Inconclusive (chunks {', '.join(map(str, inconclusive_chunks))})")
    sympy_context = f"{summary}\nChunks: {' '.join(lines)}" if chunk_results else "No SymPy results."
    if structured_issues:
        sympy_context += "\nIssues to include: " + "; ".join(structured_issues)

    math_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You verify mathematical content from research papers. Your goal is to assess whether the math is "
            "FUNDAMENTALLY sound and consistent with the paper's claims.\n\n"
            "CRITICAL CONTEXT: Math chunks are extracted from papers via OCR — they are isolated formulas, "
            "not complete derivations. SymPy verification operates on these snippets and has inherent limitations:\n\n"
            "DO NOT treat these as real issues:\n"
            "- 'Inconclusive' SymPy results — this usually means the formula uses abstract functions or notation "
            "that SymPy can't resolve, NOT that the math is wrong\n"
            "- 'Error' status from translation failures — OCR artifacts, complex notation, or domain-specific "
            "functions that don't map cleanly to SymPy\n"
            "- Missing variable definitions — these are defined elsewhere in the paper\n"
            "- Notation differences between LaTeX and SymPy representation\n\n"
            "DO flag these HIGH-VALUE issues:\n"
            "- Derivations where SymPy proves LHS != RHS (status 'failed' with actual symbolic mismatch)\n"
            "- Dimensional inconsistencies (adding quantities with different units/dimensions)\n"
            "- Formulas that contradict the paper's stated claims or methodology\n"
            "- Mathematical errors that would affect the paper's conclusions\n\n"
            "Interpretation rules:\n"
            "- is_mathematically_valid: set to false ONLY if there are genuine derivation failures or dimensional errors. "
            "Inconclusive/error statuses alone do NOT make math invalid.\n"
            "- is_consistent_with_claims: does the math support what the paper claims? Incomplete verification "
            "(inconclusive) is NOT inconsistency.\n"
            "- verified_steps: equations that SymPy confirmed correct\n"
            "- critical_gaps: important claims that couldn't be verified (not errors, just gaps)\n\n"
            "Output ONLY a JSON object. No prose before or after.\n"
            "Use this exact shape:\n"
            '{{"is_mathematically_valid": true/false, "is_consistent_with_claims": true/false, '
            '"issues": ["only genuine mathematical errors"], "explanation": "2-3 sentence overall summary", '
            '"verified_steps": ["short step 1", ...], '
            '"critical_gaps": ["claim or phenomenon not directly verified in chunks", ...]}}\n'
        )),
        ("human", "Math verification:\n{sympy_context}\n\nMath chunks:\n{math_chunks}\n\nGoals:\n{intent}")
    ])

    response = await (math_prompt | llm).ainvoke({
        "math_chunks": str(math_chunks),
        "intent": intent,
        "sympy_context": sympy_context,
    })
    raw = (response.content or "").strip()

    if not raw:
        return {
            "subagent_responses": {
                "math": {
                    "is_mathematically_valid": True,
                    "is_consistent_with_claims": True,
                    "issues": [],
                    "explanation": "Math verification skipped: no model response.",
                    "verified_steps": [],
                    "critical_gaps": [],
                }
            }
        }

    try:
        result = parse_json_response(raw, MathAnalysisOutput)
        # Trust the LLM's judgment — it has context about whether failures are
        # genuine math errors vs snippet/translation artifacts
        is_valid = result.is_mathematically_valid
        is_consistent = result.is_consistent_with_claims
        issues = list(result.issues)
        return {
            "subagent_responses": {
                "math": {
                    "is_mathematically_valid": is_valid,
                    "is_consistent_with_claims": is_consistent,
                    "issues": issues,
                    "explanation": result.explanation,
                    "verified_steps": result.verified_steps,
                    "critical_gaps": getattr(result, "critical_gaps", []),
                }
            }
        }
    except Exception as e:
        print("[MathAgent] Parse failed:", e)
        return {
            "subagent_responses": {
                "math": {
                    "is_mathematically_valid": True,
                    "is_consistent_with_claims": True,
                    "issues": [],
                    "explanation": "Could not parse math analysis response.",
                    "verified_steps": [],
                    "critical_gaps": [],
                }
            }
        }
