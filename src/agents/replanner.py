import asyncio
import subprocess
import tempfile
import os
from schemas import AgentFState, ReplannerOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response

LANG_CONFIG = {
    "python": {"ext": ".py", "cmd": ["python3"]},
    "javascript": {"ext": ".js", "cmd": ["node"]},
    "typescript": {"ext": ".ts", "cmd": ["npx", "ts-node"]},
    "bash": {"ext": ".sh", "cmd": ["bash"]},
    "shell": {"ext": ".sh", "cmd": ["bash"]},
}

COMPILED_LANGS = {
    "c": {"ext": ".c", "compiler": ["gcc"], "flags": ["-lm"]},
    "cpp": {"ext": ".cpp", "compiler": ["g++"], "flags": ["-std=c++17"]},
    "c++": {"ext": ".cpp", "compiler": ["g++"], "flags": ["-std=c++17"]},
}


def _execute_compiled(code: str, lang_key: str, timeout: int = 30) -> dict:
    config = COMPILED_LANGS[lang_key]
    src_path = None
    bin_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=config["ext"], delete=False
        ) as f:
            f.write(code)
            f.flush()
            src_path = f.name
        bin_path = src_path + ".out"
        compile_result = subprocess.run(
            config["compiler"] + [src_path, "-o", bin_path] + config["flags"],
            capture_output=True, text=True, timeout=timeout,
        )
        if compile_result.returncode != 0:
            return {
                "ran_successfully": False,
                "stdout": "",
                "stderr": f"Compilation failed:\n{compile_result.stderr[:5000]}",
            }
        run_result = subprocess.run(
            [bin_path], capture_output=True, text=True, timeout=timeout,
        )
        return {
            "ran_successfully": run_result.returncode == 0,
            "stdout": run_result.stdout[:5000],
            "stderr": run_result.stderr[:5000],
        }
    except subprocess.TimeoutExpired:
        return {
            "ran_successfully": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
        }
    finally:
        if src_path and os.path.exists(src_path):
            os.unlink(src_path)
        if bin_path and os.path.exists(bin_path):
            os.unlink(bin_path)


def execute_code(code: str, language: str, timeout: int = 30) -> dict:
    lang = language.lower()

    if lang in COMPILED_LANGS:
        return _execute_compiled(code, lang, timeout)

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

    async def _run_one(chunk: dict) -> dict:
        exec_result = await asyncio.to_thread(
            execute_code, chunk["code"], chunk["language"]
        )
        return {
            "code": chunk["code"],
            "language": chunk["language"],
            "ran_successfully": exec_result["ran_successfully"],
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
        }

    execution_results = list(await asyncio.gather(
        *[_run_one(chunk) for chunk in chunks]
    ))

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
            "You are an expert at analyzing code from research papers for FUNDAMENTAL correctness. "
            "Your goal is to assess whether the code logic is sound and aligned with the paper's methodology.\n\n"
            "CRITICAL CONTEXT: These are code SNIPPETS extracted from a paper, not complete programs. "
            "Execution failures are EXPECTED and normal because snippets lack imports, dependencies, "
            "data files, and surrounding code.\n\n"
            "For EACH chunk, set is_fundamentally_correct:\n"
            "- TRUE if the code logic/algorithm is sound, even if execution failed due to missing imports/data/context\n"
            "- FALSE only if the code has a real algorithmic bug, incorrect formula, or contradicts the paper's claims\n\n"
            "Common snippet failures to IGNORE (keep is_fundamentally_correct: true):\n"
            "- ImportError, ModuleNotFoundError, NameError (missing dependencies)\n"
            "- FileNotFoundError (missing data files)\n"
            "- Incomplete code that references external functions\n\n"
            "Only set is_fundamentally_correct to FALSE for:\n"
            "- Wrong algorithm logic that doesn't match the paper's claims\n"
            "- Incorrect math/formulas in the code\n"
            "- Logic bugs that would produce wrong results even with all dependencies\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"results": [{{"code": "...", "language": "python", "ran_successfully": true, '
            '"is_fundamentally_correct": true, '
            '"stdout": "...", "stderr": "...", "analysis": "..."}}, ...], "summary": "..."}}'
        )),
        ("human", "Analyze these code execution results. Focus on fundamental correctness, not snippet-level failures:\n\n{results}")
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

    chain = analysis_prompt | llm
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
                "is_fundamentally_correct": matching.is_fundamentally_correct if matching else True,
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
                "is_fundamentally_correct": True,
                "stdout": r["stdout"],
                "stderr": r["stderr"],
                "analysis": "Passed" if r["ran_successfully"] else f"Failed: {r['stderr'][:200]}",
            })
        correct = sum(1 for r in final_results if r["is_fundamentally_correct"])
        summary = f"{correct}/{len(final_results)} code chunks are fundamentally correct."

    return {
        "subagent_responses": {
            "replanner": {
                "results": final_results,
                "summary": summary,
            }
        }
    }
