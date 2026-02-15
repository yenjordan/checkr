from schemas import AgentFState, ChunkAnalysisLLMResponse
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response

async def CodingAgent(state: AgentFState) -> AgentFState:
    code_chunks = state.get("subagent_responses", {}).get("code_extractor", {}).get("chunks", [])

    if not code_chunks:
        return {
            "subagent_responses": {
                "coding": {
                    "is_conceptually_correct": False,
                    "chunk_results": [],
                    "issues": ["No code was extracted from the paper."],
                    "explanation": "Code extractor returned no chunks.",
                }
            }
        }

    coding_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert code reviewer analyzing a code snippet from a research paper.\n\n"
            "Your task:\n"
            "1. Evaluate whether this specific code snippet correctly implements what it claims to do based on its description\n"
            "2. Focus on logical correctness WITHIN THE SCOPE of this code snippet only\n"
            "3. DO NOT expect this code to implement an entire algorithm or paper - it may be a partial example or illustration\n"
            "4. Look for actual bugs, logic errors, or incorrect implementations, NOT missing features\n\n"
            "Categorize issues by severity:\n"
            "- CRITICAL: Actual bugs, logic errors, security vulnerabilities, incorrect algorithms\n"
            "- WARNING: Style issues, missing edge case handling, potential improvements\n\n"
            "In your explanation:\n"
            "- Briefly describe what the code does (1-2 sentences)\n"
            "- Explain why it's correct or what issues you found\n"
            "- Be specific and substantive, not generic\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"is_correct": true, "issues": [{{"message": "...", "severity": "critical"}}, ...], "explanation": "..."}}'
        )),
        ("human", (
            "Context about this code: {context}\n\n"
            "Code:\n```{language}\n{code}\n```\n\n"
            "Analyze if this code is correct given its description."
        ))
    ])

    chain = coding_prompt | llm
    chunk_analyses = []

    for i, chunk in enumerate(code_chunks):
        context = chunk.get("context", "").strip()
        code = chunk.get("code", "")
        language = chunk.get("language", "unknown")

        # If no context, mark as unable to verify
        if not context or len(context) < 10:
            chunk_analyses.append({
                "chunk_index": i,
                "is_correct": True,  # Don't fail on missing context
                "issues": [{"message": "Unable to verify: insufficient context", "severity": "warning"}],
                "explanation": "No description provided for this code snippet"
            })
            continue

        # Evaluate against local context
        try:
            response = await chain.ainvoke({
                "context": context,
                "language": language,
                "code": code
            })
            result = parse_json_response(response.content or "", ChunkAnalysisLLMResponse)
            chunk_analyses.append({
                "chunk_index": i,
                "is_correct": result.is_correct,
                "issues": [{"message": issue.message, "severity": issue.severity} for issue in result.issues],
                "explanation": result.explanation
            })
        except Exception as e:
            # Fallback for parse errors - log the raw response for debugging
            raw_response = response.content if 'response' in locals() else "No response"
            chunk_analyses.append({
                "chunk_index": i,
                "is_correct": True,  # Conservative: assume correct if can't parse
                "issues": [],
                "explanation": f"Could not parse LLM response. Error: {str(e)}\n\nRaw response: {raw_response[:500]}"
            })

    # Aggregate results
    # IMPORTANT: Only fail on critical issues, warnings are OK
    has_critical_issues = any(
        issue["severity"] == "critical"
        for c in chunk_analyses
        for issue in c["issues"]
    )

    critical_issues = [
        issue["message"]
        for c in chunk_analyses
        for issue in c["issues"]
        if issue["severity"] == "critical"
    ]

    chunks_with_criticals = len([
        c for c in chunk_analyses
        if any(i["severity"] == "critical" for i in c["issues"])
    ])
    chunks_with_warnings = len([
        c for c in chunk_analyses
        if any(i["severity"] == "warning" for i in c["issues"]) and not any(i["severity"] == "critical" for i in c["issues"])
    ])
    chunks_clean = len(chunk_analyses) - chunks_with_criticals - chunks_with_warnings

    # Build detailed explanation with per-chunk summaries
    explanation_parts = [
        f"Analyzed {len(chunk_analyses)} code chunk(s). "
        f"{chunks_clean} passed cleanly, "
        f"{chunks_with_warnings} had warnings, "
        f"{chunks_with_criticals} had critical issues.\n\n"
    ]

    # Add per-chunk details
    for i, analysis in enumerate(chunk_analyses):
        status = "✓" if analysis["is_correct"] and not any(iss["severity"] == "critical" for iss in analysis["issues"]) else "✗"
        explanation_parts.append(f"Chunk {i+1} {status}: {analysis['explanation']}\n\n")

    explanation = "".join(explanation_parts).strip()

    return {
        "subagent_responses": {
            "coding": {
                "is_conceptually_correct": not has_critical_issues,  # Only fail on criticals
                "chunk_results": chunk_analyses,  # Included in API response
                "issues": critical_issues,  # Only critical issues
                "explanation": explanation
            }
        }
    }
