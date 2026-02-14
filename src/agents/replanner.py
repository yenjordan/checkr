import subprocess
import tempfile
import os
from schemas import AgentFState, ReplannerOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm_reasoning
from utils import parse_json_response

LANG_CONFIG = {
    "python": {"ext": ".py", "cmd": ["python3"]},
    "javascript": {"ext": ".js", "cmd": ["node"]},
    "typescript": {"ext": ".ts", "cmd": ["npx", "ts-node"]},
    "bash": {"ext": ".sh", "cmd": ["bash"]},
    "shell": {"ext": ".sh", "cmd": ["bash"]},
}

def execute_code(code: str, language: str, timeout: int = 30) -> dict:
    lang = language.lower()
    config = LANG_CONFIG.get(lang)

    if not config:
        return {
            "ran_successfully": False,
            "stdout": "",
            "stderr": f"Unsupported language: {language}. Cannot execute.",
        }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=config["ext"], delete=False
    ) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            config["cmd"] + [tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ran_successfully": result.returncode == 0,
            "stdout": result.stdout[:5000],
            "stderr": result.stderr[:5000],
        }
    except subprocess.TimeoutExpired:
        return {
            "ran_successfully": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
        }
    finally:
        os.unlink(tmp_path)


async def ReplannerAgent(state: AgentFState) -> AgentFState:
    chunks = (
        state.get("subagent_responses", {})
        .get("code_extractor", {})
        .get("chunks", [])
    )

    execution_results = []

    for chunk in chunks:
        exec_result = execute_code(chunk["code"], chunk["language"])

        execution_results.append({
            "code": chunk["code"],
            "language": chunk["language"],
            "ran_successfully": exec_result["ran_successfully"],
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
        })

    # If no chunks to analyze, return early
    if not execution_results:
        return {
            "subagent_responses": {
                "replanner": {
                    "results": [],
                    "summary": "No code chunks found to execute.",
                }
            }
        }

    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at analyzing code execution results from research papers. "
            "Given the code chunks, their execution results (stdout/stderr), and whether "
            "they ran successfully, provide a detailed analysis of each chunk's validity "
            "and accuracy. For chunks that failed, explain likely causes. For chunks that "
            "ran, assess whether the output looks correct given the code's intended purpose. "
            "For non-executable languages like pseudocode, analyze the logic for correctness.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"results": [{{"code": "...", "language": "python", "ran_successfully": true, '
            '"stdout": "...", "stderr": "...", "analysis": "..."}}, ...], "summary": "..."}}'
        )),
        ("human", "Analyze these code execution results:\n\n{results}")
    ])

    results_text = ""
    for i, r in enumerate(execution_results):
        results_text += (
            f"--- Chunk {i+1} ({r['language']}) ---\n"
            f"Code:\n```\n{r['code']}\n```\n"
            f"Ran successfully: {r['ran_successfully']}\n"
            f"Stdout: {r['stdout'] or '(empty)'}\n"
            f"Stderr: {r['stderr'] or '(empty)'}\n\n"
        )

    chain = analysis_prompt | llm_reasoning
    response = await chain.ainvoke({"results": results_text})

    # Try to parse LLM analysis; fall back to execution results alone
    try:
        analysis = parse_json_response(response.content or "", ReplannerOutput)
        final_results = []
        for i, r in enumerate(execution_results):
            matching = analysis.results[i] if i < len(analysis.results) else None
            final_results.append({
                "code": r["code"],
                "language": r["language"],
                "ran_successfully": r["ran_successfully"],
                "stdout": r["stdout"],
                "stderr": r["stderr"],
                "analysis": matching.analysis if matching else "No analysis available",
            })
        summary = analysis.summary
    except Exception:
        final_results = []
        for r in execution_results:
            final_results.append({
                "code": r["code"],
                "language": r["language"],
                "ran_successfully": r["ran_successfully"],
                "stdout": r["stdout"],
                "stderr": r["stderr"],
                "analysis": "Passed" if r["ran_successfully"] else f"Failed: {r['stderr'][:200]}",
            })
        passed = sum(1 for r in execution_results if r["ran_successfully"])
        summary = f"{passed}/{len(execution_results)} code chunks executed successfully."

    return {
        "subagent_responses": {
            "replanner": {
                "results": final_results,
                "summary": summary,
            }
        }
    }
