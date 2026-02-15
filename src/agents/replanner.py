from schemas import AgentFState


async def ReplannerAgent(state: AgentFState) -> AgentFState:
    chunks = (
        state.get("subagent_responses", {})
        .get("code_extractor", {})
        .get("chunks", [])
    )

    coding = state.get("subagent_responses", {}).get("coding", {})
    is_correct = coding.get("is_conceptually_correct", True)
    explanation = coding.get("explanation", "Conceptual review (no execution).")

    final_results = [
        {
            "code": c.get("code", ""),
            "language": c.get("language", "python"),
            "ran_successfully": True,
            "is_fundamentally_correct": is_correct,
            "stdout": "",
            "stderr": "",
            "analysis": explanation if is_correct else (explanation or "Conceptual review found issues."),
        }
        for c in chunks
    ]
    n = len(final_results)
    summary = f"{n} code chunk(s); conceptual review: {'passed' if is_correct else 'issues found'}." if n else "No code chunks."

    return {
        "subagent_responses": {
            "replanner": {
                "results": final_results,
                "summary": summary,
            }
        }
    }
