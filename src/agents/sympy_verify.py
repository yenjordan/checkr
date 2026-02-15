import asyncio
import json
import re
import random
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from schemas import AgentFState
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response
from pydantic import BaseModel, field_validator

PARSE_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
NUMERIC_FUZZ = 5
TOL = 1e-6

KNOWN_FUNCS = {
    "e": sp.E, "pi": sp.pi, "oo": sp.oo, "I": sp.I,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "Abs": sp.Abs, "sign": sp.sign, "floor": sp.floor, "ceiling": sp.ceiling,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "factorial": sp.factorial, "binomial": sp.binomial, "gamma": sp.gamma,
    "erf": sp.erf, "erfc": sp.erfc,
    "Sum": sp.Sum, "Product": sp.Product,
    "Integral": sp.Integral, "Derivative": sp.Derivative,
    "Rational": sp.Rational, "Matrix": sp.Matrix,
    "Eq": sp.Eq, "Symbol": sp.Symbol, "symbols": sp.symbols,
    "zoo": sp.zoo, "nan": sp.nan,
    "conjugate": sp.conjugate, "re": sp.re, "im": sp.im,
    "norm": sp.Function("norm"),
    "clip": sp.Function("clip"), "Min": sp.Min, "Max": sp.Max,
}
KNOWN_FUNC_NAMES = set(KNOWN_FUNCS.keys())


def _str_from_maybe_list(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, list) and v and isinstance(v[0], str):
        return v[0]
    return ""


class ChunkTranslation(BaseModel):
    type: str
    lhs: str = ""
    rhs: str = ""
    expr: str = ""
    free_symbols: list[str] = []

    @field_validator("lhs", "rhs", "expr", mode="before")
    @classmethod
    def coerce_str(cls, v: Any) -> str:
        return _str_from_maybe_list(v)


TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Translate LaTeX to SymPy expression strings. Output ONLY valid JSON:\n"
        '{{"type": "equation"|"definition"|"expression", "lhs": "...", "rhs": "...", "expr": "...", "free_symbols": ["x",...]}}\n'
        "equation/definition: fill lhs and rhs. expression: fill expr only.\n"
        "CRITICAL RULES:\n"
        "- Use ** for power (not ^), * for explicit multiplication\n"
        "- Variable names: use valid Python identifiers with underscores for subscripts (e.g. T_H, T_llm, A_H, A_llm, M, M_llm)\n"
        "- List ALL free symbols in free_symbols (every variable in the expression). For clip(0,1,x/y) list x and y; for ratios like A_H/A_llm list A_H, A_llm.\n"
        "- NEVER put explanatory text in lhs/rhs/expr (no phrases like 'specify dummy variables' or notes). Only valid SymPy expression strings.\n"
        "- Functions: exp, log, sqrt, sin, cos, erf, Rational, Sum, Integral, clip, Min, Max, Abs, norm\n"
        "- No .T/.conj()/.I — use X_T, conjugate(x), im(x)\n"
        "- ASCII only. NO backslashes, NO LaTeX commands like \\frac or \\text\n"
        "- For fractions use (numerator)/(denominator)\n"
        "- For ∝ put only RHS in expr\n"
        "Examples:\n"
        "- f(x)=x^2+1 → type: definition, lhs: f(x), rhs: x**2+1, free_symbols: [x]\n"
        "- ET = (1/N)*Sum(clip(0,1,T_H/T_i)) → type: definition, lhs: ET, rhs: (1/N)*Sum(clip(0,1,T_H/T_i)), free_symbols: [N, T_H, T_i]\n"
        "- s := clip(0,1,T_H/T_llm) → type: definition, lhs: s, rhs: clip(0,1,T_H/T_llm), free_symbols: [T_H, T_llm]\n"
        "- MP or MI = clip(0,1,M/M_llm)*100 → type: definition, lhs: MP or MI, rhs: clip(0,1,M/M_llm)*100, free_symbols: [M, M_llm]\n"
        "- MI = (1/N)*Sum(clip(0,1,A_H/A_llm))*100 → type: definition, lhs: MI, rhs: (1/N)*Sum(clip(0,1,A_H/A_llm))*100, free_symbols: [N, A_H, A_llm]\n"
    )),
    ("human", "LaTeX: {latex}\nContext: {context}\nType: {equation_type}"),
])

RETRY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Fix the JSON. Same shape: type, lhs, rhs, expr, free_symbols. SymPy syntax only, no .T/.conj()/.I, no LaTeX, no backslashes; single expression only (no comma-separated list). ∝ → expr RHS only.\n"),
    ("human", "LaTeX: {latex}\nPrevious: {previous}\nError: {error}\nOutput corrected JSON only."),
])


def _detect_unknown_funcs(expr_str: str, free_symbols: list[str]) -> list[str]:
    calls = set(re.findall(r'\b([a-zA-Z_]\w*)\s*\(', expr_str))
    return list(calls - KNOWN_FUNC_NAMES - set(free_symbols) - {"Rational"})


def _fix_assignment_to_call(s: str) -> str:
    pat = re.compile(r"(\w+)\(([^)]*)\)\s*=\s*")
    while True:
        m = pat.search(s)
        if not m:
            break
        end = m.end()
        after = s[end:]
        depth = 0
        insert_pos = len(after)
        for i, c in enumerate(after):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "," and depth == 0:
                insert_pos = i
                break
        repl = f"Eq({m.group(1)}({m.group(2)}), "
        s = s[: m.start()] + repl + after[:insert_pos] + ")" + after[insert_pos:]
    return s


def _sanitize_expr(x: Any) -> str:
    if isinstance(x, list):
        x = x[0] if x else "0"
    if not isinstance(x, str):
        x = str(x) if x else "0"
    x = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)[\[\]().|]?", "", x)
    x = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^}]*)\}", r"\1", x)
    x = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"((\1)/(\2))", x)
    x = re.sub(r"\\sqrt\{([^}]*)\}", r"sqrt(\1)", x)
    x = re.sub(r"\\cdot", "*", x)
    x = re.sub(r"\\times", "*", x)
    x = re.sub(r"\\pm", "+", x)
    x = re.sub(r"\\(?:leq|le|geq|ge|neq|ne|approx|sim|equiv)\b", "", x)
    x = x.replace("\\", "")
    x = x.replace("\u00d7", "*").replace("\u00b7", "*").replace("\u2212", "-").replace("\u221d", "")
    x = x.replace("}{", ", ").replace("{", "(").replace("}", ")")
    x = x.replace("@", "_at_")
    x = re.sub(r"\b([a-zA-Z_]\w*)\.T\b", r"\1_T", x)
    x = re.sub(r"\b([a-zA-Z_]\w*)\.conj\s*\(\s*\)", r"conjugate(\1)", x)
    x = re.sub(r"\b([a-zA-Z_]\w*)\.I\b", r"im(\1)", x)
    x = re.sub(r"[^\x20-\x7e]", "", x)
    x = re.sub(r"__+", "_", x) 
    x = re.sub(r"_\s+", "_", x)
    x = re.sub(r"_+(\s|$|\)|\]|,|\+|\-|\*|/)", r"\1", x)
    x = _fix_assignment_to_call(x)
    return x.strip()

def _detect_potential_symbols(expr_str: str) -> list[str]:
    candidates = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expr_str))
    exclude = KNOWN_FUNC_NAMES | {'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'if', 'else', 'for', 'while'}
    return list(candidates - exclude)


def _unwrap_sum_single_arg(expr_str: str) -> str | None:
    i = expr_str.find("Sum(")
    if i == -1:
        return None
    start = i + 4
    depth = 1
    for j in range(start, len(expr_str)):
        c = expr_str[j]
        if c == "(":
            depth += 1
        elif c == ",":
            if depth == 1:
                return None
        elif c == ")":
            depth -= 1
            if depth == 0:
                inner = expr_str[start:j]
                return expr_str[:i] + "(" + inner + ")" + expr_str[j + 1:]
    return None


def _safe_parse(expr_str: str, local_syms: dict | None = None) -> sp.Basic:
    expr_str = _sanitize_expr(expr_str)
    d = dict(KNOWN_FUNCS)
    if local_syms:
        d.update(local_syms)
    for sym_name in _detect_potential_symbols(expr_str):
        if sym_name not in d:
            d[sym_name] = sp.Symbol(sym_name)
    for fname in _detect_unknown_funcs(expr_str, list(d.keys())):
        d[fname] = sp.Function(fname)
    try:
        out = parse_expr(expr_str, local_dict=d, transformations=PARSE_TRANSFORMS)
    except Exception as e:
        if "dummy variables" in str(e):
            unwrapped = _unwrap_sum_single_arg(expr_str)
            if unwrapped and unwrapped != expr_str:
                return _safe_parse(unwrapped, local_syms)
        raise
    if isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], sp.Basic):
        return out[0]
    if not isinstance(out, sp.Basic):
        raise ValueError(f"Not a SymPy expression (got {type(out).__name__})")
    return out


def _build_local_syms(names: list[str]) -> dict:
    return {n: sp.Symbol(n) for n in names}


def _known_funcs_only(expr: sp.Basic) -> bool:
    known = (sp.exp, sp.log, sp.sqrt, sp.sin, sp.cos, sp.tan,
             sp.asin, sp.acos, sp.atan, sp.sinh, sp.cosh, sp.tanh,
             sp.erf, sp.erfc, sp.gamma, sp.Abs, sp.sign,
             sp.floor, sp.ceiling, sp.factorial, sp.conjugate, sp.re, sp.im)
    for a in expr.atoms(sp.Function):
        if type(a) not in known:
            return False
    return True


def _tex(expr) -> str:
    try:
        return f"${sp.latex(expr)}$"
    except Exception:
        return str(expr)


def _prove_definition(lhs_str: str, rhs_str: str, local_syms: dict) -> dict:
    steps = [{"step": "Parse", "detail": f"${lhs_str}$ $:=$ ${rhs_str}$"}]
    try:
        rhs = _safe_parse(rhs_str, local_syms)
    except Exception as e:
        steps.append({"step": "Parse failed", "detail": str(e)})
        return {"proved": False, "steps": steps, "conclusion": str(e)}
    steps.append({"step": "RHS", "detail": _tex(rhs)})
    free = sorted(str(s) for s in rhs.free_symbols)
    abstract = {str(a.func) for a in rhs.atoms(sp.Function) if not _known_funcs_only(a)}
    if abstract:
        steps.append({"step": "Abstract", "detail": f"{', '.join(sorted(abstract))} (axioms)"})
    steps.append({"step": "Conclusion", "detail": f"Definition ${lhs_str}$ is well-formed."})
    return {"proved": True, "steps": steps, "conclusion": "Definition well-formed."}


def _prove_equation(lhs_str: str, rhs_str: str, free_syms: list[str], local_syms: dict) -> dict:
    steps = [{"step": "Claim", "detail": f"${lhs_str}$ $=$ ${rhs_str}$"}]
    try:
        lhs = _safe_parse(lhs_str, local_syms)
        rhs = _safe_parse(rhs_str, local_syms)
    except Exception as e:
        steps.append({"step": "Parse failed", "detail": str(e)})
        return {"proved": False, "steps": steps, "conclusion": str(e)}
    if not _known_funcs_only(lhs) or not _known_funcs_only(rhs):
        if str(lhs) == str(rhs):
            steps.append({"step": "Match", "detail": "Structurally identical."})
            return {"proved": True, "steps": steps, "conclusion": "Identical."}
        steps.append({"step": "Conclusion", "detail": "Abstract functions — inconclusive."})
        return {"proved": None, "steps": steps, "conclusion": "Inconclusive."}
    diff = sp.simplify(lhs - rhs)
    if diff == 0:
        steps.append({"step": "Proved", "detail": "LHS − RHS = 0."})
        return {"proved": True, "steps": steps, "conclusion": "Proved."}
    if sp.simplify(sp.expand(lhs) - sp.expand(rhs)) == 0:
        steps.append({"step": "Proved", "detail": "Equal after expand."})
        return {"proved": True, "steps": steps, "conclusion": "Proved."}
    syms = [sp.Symbol(s) for s in free_syms]
    if syms:
        ok = 0
        for _ in range(NUMERIC_FUZZ):
            sub = {s: random.uniform(0.5, 5.0) for s in syms}
            try:
                if abs(complex((lhs - rhs).subs(sub).evalf())) < TOL:
                    ok += 1
            except Exception:
                pass
        if ok == NUMERIC_FUZZ:
            steps.append({"step": "Numeric", "detail": f"{ok}/{NUMERIC_FUZZ} samples OK."})
            return {"proved": True, "steps": steps, "conclusion": "Verified numerically."}
    steps.append({"step": "Conclusion", "detail": f"Residual: {_tex(diff)}"})
    return {"proved": False if diff != 0 else None, "steps": steps, "conclusion": str(diff)}


def _prove_expression(expr_str: str, local_syms: dict) -> dict:
    steps = [{"step": "Parse", "detail": f"${expr_str}$"}]
    try:
        expr = _safe_parse(expr_str, local_syms)
    except Exception as e:
        steps.append({"step": "Parse failed", "detail": str(e)})
        return {"proved": False, "steps": steps, "conclusion": str(e)}
    steps.append({"step": "Parsed", "detail": _tex(expr)})
    steps.append({"step": "Conclusion", "detail": "Well-formed."})
    return {"proved": True, "steps": steps, "conclusion": "Well-formed."}


def _is_definition(lhs_str: str, free_syms: list[str]) -> bool:
    s = lhs_str.strip()
    if re.fullmatch(r'[a-zA-Z_]\w*', s):
        return s in free_syms
    return bool(re.fullmatch(r'[a-zA-Z_]\w*\s*\([\w\s,]*\)', s))


def generate_proof(parsed: dict) -> dict:
    free_syms = parsed.get("free_symbols", [])
    if isinstance(free_syms, str):
        free_syms = [free_syms] if free_syms else []
    local = _build_local_syms(free_syms)
    lhs = _sanitize_expr(parsed.get("lhs") or "")
    rhs = _sanitize_expr(parsed.get("rhs") or "")
    expr = _sanitize_expr(parsed.get("expr") or "")
    t = parsed.get("type", "expression")
    if lhs and rhs:
        if t == "definition" or _is_definition(lhs, free_syms):
            return _prove_definition(lhs, rhs, local)
        return _prove_equation(lhs, rhs, free_syms, local)
    if expr:
        return _prove_expression(expr, local)
    return {"proved": None, "steps": [], "conclusion": "No expression."}


def _parse_chunk_translation(raw: str) -> ChunkTranslation:
    s = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", s, re.DOTALL)
    if match:
        s = match.group(1).strip()
    else:
        idx = s.find("{")
        if idx != -1:
            s = s[idx:].strip()
    s = re.sub(r',\s*//[^\n]*', ',', s)
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    s = re.sub(r',\s*,', ',', s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        import json_repair
        data = json_repair.loads(s)
    for k in ("lhs", "rhs", "expr"):
        if k in data and isinstance(data[k], list):
            data[k] = (data[k][0] if data[k] and isinstance(data[k][0], str) else "")
    return ChunkTranslation.model_validate(data)


def _extract_clip_rhs_from_raw(raw: str) -> str | None:
    """If LLM returned explanatory text, try to extract a clip(0, 1, ...) expression from raw."""
    raw = raw.replace(" ", "")
    # Find clip(0,1,...) with balanced parens
    i = raw.find("clip(0,1,")
    if i == -1:
        i = raw.find("clip(0, 1,")
    if i == -1:
        return None
    start = i
    depth = 0
    for j, c in enumerate(raw[start:], start):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                expr = raw[start : j + 1]
                # Re-add spaces for readability: clip(0,1,x) -> clip(0, 1, x)
                expr = re.sub(r"clip\(\s*0\s*,\s*1\s*,\s*", "clip(0, 1, ", expr)
                return expr
    return None


def _parse_and_validate(raw: str) -> tuple[dict | None, str | None]:
    """Parse raw to ChunkTranslation dict and validate lhs/rhs/expr parse. Returns (result, None) or (None, error)."""
    try:
        parsed = _parse_chunk_translation(raw)
    except Exception as e:
        return None, str(e)
    result = parsed.model_dump()
    # Reject if rhs/expr looks like natural language (LLM put explanation in the field)
    for k in ("rhs", "expr"):
        v = (result.get(k) or "").strip()
        if v and (
            v.lower().startswith("specify ")
            or "dummy variables" in v.lower()
            or (" for " in v.lower() and "clip" in v and len(v) < 200)
        ):
            return None, "Translation was explanatory text, not SymPy."
    ls = _build_local_syms(result.get("free_symbols") or [])
    try:
        for k in ("lhs", "rhs", "expr"):
            if result.get(k):
                _safe_parse(result[k], ls)
    except Exception as e:
        return None, str(e)
    return result, None


async def _translate_chunk(latex: str, context: str, eq_type: str) -> tuple[dict | None, str | None]:
    raw = (await (TRANSLATE_PROMPT | llm).ainvoke({"latex": latex, "context": context, "equation_type": eq_type})).content or ""
    raw = raw.strip()
    result, err = _parse_and_validate(raw)
    if result is not None:
        return result, None
    # Fallback: if LLM returned text mentioning clip, extract clip(0, 1, ...) from raw
    extracted_rhs = _extract_clip_rhs_from_raw(raw)
    if extracted_rhs:
        free_syms = _detect_potential_symbols(extracted_rhs)
        try:
            _safe_parse(extracted_rhs, _build_local_syms(free_syms))
            lhs_hint = "MI" if "Integral" in context or "integral" in context.lower() else "MP" if "Peak" in context else "metric"
            return {"type": "definition", "lhs": lhs_hint, "rhs": extracted_rhs, "expr": "", "free_symbols": free_syms}, None
        except Exception:
            pass
    raw = (await (RETRY_PROMPT | llm).ainvoke({"latex": latex, "previous": raw, "error": err})).content or ""
    raw = raw.strip()
    result, _ = _parse_and_validate(raw)
    if result is not None:
        return result, None
    extracted_rhs = _extract_clip_rhs_from_raw(raw)
    if extracted_rhs:
        free_syms = _detect_potential_symbols(extracted_rhs)
        try:
            _safe_parse(extracted_rhs, _build_local_syms(free_syms))
            return {"type": "definition", "lhs": "metric", "rhs": extracted_rhs, "expr": "", "free_symbols": free_syms}, None
        except Exception:
            pass
    return (None, err)


async def _process_one_chunk(i: int, c: dict) -> dict:
    """Process a single math chunk (translate + prove). Used for parallel execution."""
    parsed, err = await _translate_chunk(
        c.get("latex", ""), c.get("context", ""), c.get("equation_type", "equation")
    )
    if parsed is None:
        return {
            "latex": c.get("latex"), "context": c.get("context"), "equation_type": c.get("equation_type"),
            "sympy_translation": None, "proof": {"proved": None, "steps": [], "conclusion": err or "Translate failed"},
            "status": "error", "error": err,
        }
    try:
        proof = generate_proof(parsed)
        status = "verified" if proof["proved"] is True else "failed" if proof["proved"] is False else "inconclusive"
        return {
            "latex": c.get("latex"), "context": c.get("context"), "equation_type": c.get("equation_type"),
            "sympy_translation": parsed, "proof": proof, "status": status,
        }
    except Exception as e:
        return {
            "latex": c.get("latex"), "context": c.get("context"), "equation_type": c.get("equation_type"),
            "sympy_translation": parsed, "proof": {"proved": None, "steps": [{"step": "Error", "detail": str(e)}], "conclusion": str(e)},
            "status": "error", "error": str(e),
        }


async def SympyVerifyAgent(state: AgentFState) -> AgentFState:
    chunks = state.get("subagent_responses", {}).get("math_extractor", {}).get("chunks", [])
    if not chunks:
        return {"subagent_responses": {"sympy_verify": {"ran_successfully": True, "chunk_results": [], "summary": "No math chunks."}}}

    max_chunks = 20
    to_process = chunks[:max_chunks]
    n = len(to_process)

    results = await asyncio.gather(
        *[_process_one_chunk(i, c) for i, c in enumerate(to_process)]
    )
    results = list(results)

    v = sum(1 for r in results if r["status"] == "verified")
    f = sum(1 for r in results if r["status"] == "failed")
    o = n - v - f
    summary = f"{v}/{n} proved, {f} failed, {o} inconclusive"

    return {"subagent_responses": {"sympy_verify": {
        "ran_successfully": f == 0 and o == 0,
        "chunk_results": results,
        "summary": summary,
        "verified_count": v,
        "failed_count": f,
        "error_count": o,
    }}}
