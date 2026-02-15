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
            "You verify mathematical content from papers: validity (derivations, dimensions) and consistency with the paper's claims.\n\n"
            "Rules from SymPy verification:\n"
            "- If there are failed derivations (status 'failed'), errors (status 'error'), or many inconclusive steps (status 'inconclusive'), the provided chunks do NOT support mathematical validity. Set is_mathematically_valid to false and is_consistent_with_claims to false.\n"
            "- Include the structured issues from verification in your issues list (failed derivations, errors in definitions, inconclusive steps with chunk numbers).\n"
            "- verified_steps: only what was directly validated (e.g. equations that passed). If none verified, use an empty list or note what was attempted.\n"
            "- critical_gaps: main claims or phenomena not directly supported by the chunks (e.g. FLD alignment, implicit bias, grokking, experimental invariance).\n\n"
            "Output ONLY a JSON object. No prose before or after.\n"
            "Use this exact shape:\n"
            '{{"is_mathematically_valid": true/false, "is_consistent_with_claims": true/false, '
            '"issues": ["list including any structured issues from verification"], "explanation": "2-3 sentence overall summary", '
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
        has_failures = bool(failed_chunks or error_chunks)
        is_valid = result.is_mathematically_valid and not has_failures
        is_consistent = result.is_consistent_with_claims and not has_failures
        issues = list(result.issues)
        for s in structured_issues:
            if s not in issues:
                issues.append(s)
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
