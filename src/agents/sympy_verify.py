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
from pydantic import BaseModel

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
}
KNOWN_FUNC_NAMES = set(KNOWN_FUNCS.keys())


class ChunkTranslation(BaseModel):
    type: str
    lhs: str = ""
    rhs: str = ""
    expr: str = ""
    free_symbols: list[str] = []


TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "Translate LaTeX to SymPy expression strings. Output ONLY valid JSON:\n"
        '{{"type": "equation"|"definition"|"expression", "lhs": "...", "rhs": "...", "expr": "...", "free_symbols": ["x",...]}}\n'
        "equation/definition: fill lhs and rhs. expression: fill expr only.\n"
        "Rules: ** for power, * for multiplication. exp, log, sqrt, sin, cos, erf, Rational, Sum, Integral. Greek/subscripts as names (sigma_f, mu_0).\n"
        "No .T/.conj()/.I on variables — use X_T, conjugate(x), im(x). For ∝ put only RHS in expr (e.g. Sigma_inv*mu), no I. Norms: norm(x). ASCII only.\n"
        "Examples: f(x)=x^2+1 → type definition, lhs f(x), rhs x**2+1. S∝Σ^{{-1}}μ → type expression, expr Sigma_inv*mu.\n"
    )),
    ("human", "LaTeX: {latex}\nContext: {context}\nType: {equation_type}"),
])

RETRY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Fix the JSON. Same shape: type, lhs, rhs, expr, free_symbols. SymPy syntax only, no .T/.conj()/.I, no LaTeX. ∝ → expr RHS only.\n"),
    ("human", "LaTeX: {latex}\nPrevious: {previous}\nError: {error}\nOutput corrected JSON only."),
])


def _detect_unknown_funcs(expr_str: str, free_symbols: list[str]) -> list[str]:
    calls = set(re.findall(r'\b([a-zA-Z_]\w*)\s*\(', expr_str))
    return list(calls - KNOWN_FUNC_NAMES - set(free_symbols) - {"Rational"})


def _sanitize_expr(x: Any) -> str:
    if isinstance(x, list):
        x = x[0] if x else "0"
    if not isinstance(x, str):
        x = str(x) if x else "0"
    x = re.sub(r"\b([a-zA-Z_]\w*)\.T\b", r"\1_T", x)
    x = re.sub(r"\b([a-zA-Z_]\w*)\.conj\s*\(\s*\)", r"conjugate(\1)", x)
    x = re.sub(r"\b([a-zA-Z_]\w*)\.I\b", r"im(\1)", x)
    return x.strip()


def _safe_parse(expr_str: str, local_syms: dict | None = None) -> sp.Basic:
    expr_str = _sanitize_expr(expr_str)
    d = dict(KNOWN_FUNCS)
    if local_syms:
        d.update(local_syms)
    for fname in _detect_unknown_funcs(expr_str, list(local_syms.keys()) if local_syms else []):
        d[fname] = sp.Function(fname)
    out = parse_expr(expr_str, local_dict=d, transformations=PARSE_TRANSFORMS)
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


def _prove_definition(lhs_str: str, rhs_str: str, local_syms: dict) -> dict:
    steps = [{"step": "Parse", "detail": f"{lhs_str} := {rhs_str}"}]
    try:
        rhs = _safe_parse(rhs_str, local_syms)
    except Exception as e:
        steps.append({"step": "Parse failed", "detail": str(e)})
        return {"proved": False, "steps": steps, "conclusion": str(e)}
    steps.append({"step": "RHS", "detail": str(rhs)})
    free = sorted(str(s) for s in rhs.free_symbols)
    abstract = {str(a.func) for a in rhs.atoms(sp.Function) if not _known_funcs_only(a)}
    if abstract:
        steps.append({"step": "Abstract", "detail": f"{', '.join(sorted(abstract))} (axioms)"})
    steps.append({"step": "Conclusion", "detail": "Definition well-formed."})
    return {"proved": True, "steps": steps, "conclusion": "Definition well-formed."}


def _prove_equation(lhs_str: str, rhs_str: str, free_syms: list[str], local_syms: dict) -> dict:
    steps = [{"step": "Claim", "detail": f"{lhs_str} = {rhs_str}"}]
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
    steps.append({"step": "Conclusion", "detail": f"Residual: {diff}"})
    return {"proved": False if diff != 0 else None, "steps": steps, "conclusion": str(diff)}


def _prove_expression(expr_str: str, local_syms: dict) -> dict:
    steps = [{"step": "Parse", "detail": expr_str}]
    try:
        expr = _safe_parse(expr_str, local_syms)
    except Exception as e:
        steps.append({"step": "Parse failed", "detail": str(e)})
        return {"proved": False, "steps": steps, "conclusion": str(e)}
    steps.append({"step": "Parsed", "detail": str(expr)})
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


async def _translate_chunk(latex: str, context: str, eq_type: str) -> tuple[dict | None, str | None]:
    raw = (await (TRANSLATE_PROMPT | llm).ainvoke({"latex": latex, "context": context, "equation_type": eq_type})).content or ""
    raw = raw.strip()
    try:
        parsed = parse_json_response(raw, ChunkTranslation, llm=None)
        result = parsed.model_dump()
        ls = _build_local_syms(result.get("free_symbols") or [])
        for k in ("lhs", "rhs", "expr"):
            if result.get(k):
                _safe_parse(result[k], ls)
        return result, None
    except Exception as e:
        err = str(e)
    raw = (await (RETRY_PROMPT | llm).ainvoke({"latex": latex, "previous": raw, "error": err})).content or ""
    raw = raw.strip()
    try:
        parsed = parse_json_response(raw, ChunkTranslation, llm=None)
        result = parsed.model_dump()
        ls = _build_local_syms(result.get("free_symbols") or [])
        for k in ("lhs", "rhs", "expr"):
            if result.get(k):
                _safe_parse(result[k], ls)
        return result, None
    except Exception as e:
        return None, str(e)


async def SympyVerifyAgent(state: AgentFState) -> AgentFState:
    chunks = state.get("subagent_responses", {}).get("math_extractor", {}).get("chunks", [])
    if not chunks:
        return {"subagent_responses": {"sympy_verify": {"ran_successfully": True, "chunk_results": [], "summary": "No math chunks."}}}

    max_chunks = 20
    n = min(len(chunks), max_chunks)
    print(f"[SymPy verify] {n} chunk(s)...", flush=True)
    results = []

    for i, c in enumerate(chunks[:max_chunks]):
        parsed, err = await _translate_chunk(c.get("latex", ""), c.get("context", ""), c.get("equation_type", "equation"))
        if parsed is None:
            results.append({
                "latex": c.get("latex"), "context": c.get("context"), "equation_type": c.get("equation_type"),
                "sympy_translation": None, "proof": {"proved": None, "steps": [], "conclusion": err or "Translate failed"},
                "status": "error", "error": err,
            })
            print(f"  chunk {i+1}: ERROR", flush=True)
            continue
        try:
            proof = generate_proof(parsed)
            status = "verified" if proof["proved"] is True else "failed" if proof["proved"] is False else "inconclusive"
            results.append({
                "latex": c.get("latex"), "context": c.get("context"), "equation_type": c.get("equation_type"),
                "sympy_translation": parsed, "proof": proof, "status": status,
            })
            print(f"  chunk {i+1}: {status}", flush=True)
        except Exception as e:
            results.append({
                "latex": c.get("latex"), "context": c.get("context"), "equation_type": c.get("equation_type"),
                "sympy_translation": parsed, "proof": {"proved": None, "steps": [{"step": "Error", "detail": str(e)}], "conclusion": str(e)},
                "status": "error", "error": str(e),
            })
            print(f"  chunk {i+1}: ERROR", flush=True)

    v = sum(1 for r in results if r["status"] == "verified")
    f = sum(1 for r in results if r["status"] == "failed")
    o = n - v - f
    summary = f"{v}/{n} proved, {f} failed, {o} inconclusive"
    print(f"[SymPy verify] {summary}", flush=True)
    return {"subagent_responses": {"sympy_verify": {
        "ran_successfully": f == 0 and o == 0,
        "chunk_results": results,
        "summary": summary,
        "verified_count": v,
        "failed_count": f,
        "error_count": o,
    }}}
