import os
import subprocess
import tempfile

from schemas import AgentFState
from langchain_core.prompts import ChatPromptTemplate
from config import llm

LEAN_TIMEOUT = 90


def run_lean(lean_code: str) -> dict:
    """Write code to a temp .lean file and run `lean` to typecheck. Returns success, stdout, stderr."""
    if not lean_code or not lean_code.strip():
        return {"ran_successfully": False, "stdout": "", "stderr": "No Lean code generated."}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".lean", delete=False) as f:
        f.write(lean_code)
        f.flush()
        path = f.name
    try:
        result = subprocess.run(
            ["lean", path],
            capture_output=True,
            text=True,
            timeout=LEAN_TIMEOUT,
        )
        return {
            "ran_successfully": result.returncode == 0,
            "stdout": (result.stdout or "")[:4000],
            "stderr": (result.stderr or "")[:4000],
        }
    except subprocess.TimeoutExpired:
        return {"ran_successfully": False, "stdout": "", "stderr": f"Timed out after {LEAN_TIMEOUT}s."}
    except FileNotFoundError:
        return {"ran_successfully": False, "stdout": "", "stderr": "Lean not found on PATH."}
    except Exception as e:
        return {"ran_successfully": False, "stdout": "", "stderr": str(e)}
    finally:
        os.unlink(path)


LEAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert at formalizing mathematics in Lean 4. Given a list of mathematical statements "
        "(equations, definitions, theorems) from a research paper, produce a single Lean 4 file that "
        "formalizes as much as possible and typechecks.\n\n"
        "Rules:\n"
        "- The file is compiled with `lean file.lean` with NO project and NO external libraries (no Mathlib). "
        "Use ONLY Lean 4 core: no `import`. No `constant` (use `axiom` or `opaque`).\n"
        "- Use ONLY Nat and Int. Do NOT use Float or Real (they cause match/recursion issues in core Lean). "
        "Nat has no unary minus: use Int for any parameter that is negated (e.g. -y*z). Do not mix Nat and Int in one expression: if a def uses negation or Int-valued axioms, use Int for all parameters and return type and use Int literals (e.g. `def loss_function (z y a : Int) : Int := (1 - a) * log_like (1 + exp_like (-y * z)) + a * f z` with `axiom exp_like log_like f : Int → Int`). No Nat.succ or Nat.pred to convert; keep the whole def in one type.\n"
        "For loss/exp/log: use axiom (or opaque) for uninterpreted symbols, e.g. `axiom exp_like : Int → Int` and `axiom log_like : Int → Int`, not `def exp_like (x : Int) : Int := 9999`. The def for the formula should use only those axioms and Int arithmetic.\n"
        "- Do not use opaque/axiom with a Type and terms of that type in an equation using * or ^. "
        "For equations like E = m*c^2 use `def energy (m c : Nat) : Nat := m * c ^ 2`.\n"
        "- Syntax: Use `def` (not `definition`). No `match` on Float/Real or Int (only Nat has 0 and (n+1)); for Int use a non-recursive def or axiom. For recursive defs use Nat and `match n with | 0 => ... | (n+1) => ... f n ... end`. No LaTeX, no subscripts, no ∝; for proportionality use ∃ k, x = k * y. No pseudocode in def bodies (e.g. no 'summation from k'; write only valid Lean).\n"
        "- If a theorem mentions parameters (e.g. ρ μ σ_f), the def it refers to must take those parameters: e.g. `def A_gen (rho mu_f sigma_f sigma_perp : Int) : Int := ...` not `def A_gen : Int := ...`.\n"
        "- One axiom per line: write `axiom exp_like : Int → Int` and `axiom log_like : Int → Int` on separate lines, not `axiom exp_like log_like f : Int → Int` (invalid in Lean). Same for opaque.\n"
        "- axiom and opaque never take a value: write `axiom f : Int → Int`, not `axiom A : Int := 1` (use `def A : Int := 1` for constants) and not `def f (x : Int) : Int := axiom` (invalid).\n"
        "- Every theorem must have an explicit type and proof: `theorem name : PropType := sorry` (e.g. `theorem t : A = 1 := sorry` or `theorem t : ∃ k, p = k * q := sorry`). Never write `theorem name := sorry` (missing type). Use `:= sorry` only, not `by sorry`. No theorem returning Int. No typeclasses, no structures. For proportionality do not use ∝; use `∃ k, P_min = k * x` or `∃ k, P_min = k / x` and proof `:= sorry`.\n"
        "- Division and addition: write `(1 + erf x) / 2` with parentheses so the divisor applies to the sum, not `1 + erf x / 2`. Use only ASCII in names (no μ, ρ, σ); use mu, rho, sigma.\n"
        "- Declaration order: put every axiom and opaque first, then every def, then every theorem. (Defs and theorems must come after the axioms they use.)\n"
        "- No axiom or opaque inside a def body (axiom is top-level only). Use only ASCII in names (no ρ, σ_⊥, Uᵀ, ²); use rho, sigma_perp, U_T. No space in names (use r_erotisk not r erotisk). For the number 1 use the digit 1, not ⟨1,1⟩. There is no Int.sqrt in core Lean; use axiom sqrt_like : Int → Int if needed.\n"
        "- Keep each chunk minimal: a few axiom/opaque, one or two defs, one theorem := sorry. Do not declare the same name twice (no duplicate axiom/def lines).\n"
        "- Output ONLY the Lean 4 source code, no markdown fences, no trailing backticks, no explanation. The file must typecheck.\n"
    )),
    ("human", "Formalize as Lean 4 (Nat/Int only, no Mathlib):\n\n{math_chunks}"),
])


def _strip_fence(code: str) -> str:
    code = (code or "").strip()
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code


def _one_axiom_per_line(code: str) -> str:
    """Expand 'axiom a b c : T' into one line per name."""
    out = []
    for line in code.split("\n"):
        s = line.strip()
        if (s.startswith("axiom ") or s.startswith("opaque ")) and " : " in s and "(" not in s.split(" : ")[0]:
            prefix, _, rest = s.partition(" : ")
            parts = prefix.split()
            if len(parts) > 2:
                for name in parts[1:]:
                    out.append(parts[0] + " " + name + " : " + rest)
                continue
        out.append(line)
    return "\n".join(out)


async def LeanVerifyAgent(state: AgentFState) -> AgentFState:
    chunks = state.get("subagent_responses", {}).get("math_extractor", {}).get("chunks", [])
    if not chunks:
        return {"subagent_responses": {"lean_verify": {"ran_successfully": True, "chunk_results": []}}}

    max_chunks = 20
    chunk_results = []

    for i, c in enumerate(chunks[:max_chunks]):
        try:
            text = f"[{c.get('equation_type', 'equation')}] {c.get('latex', '')}  -- {c.get('context', '')}"
            response = await (LEAN_PROMPT | llm).ainvoke({"math_chunks": text})
            code = _one_axiom_per_line(_strip_fence(response.content or ""))
            res = run_lean(code)
            chunk_results.append({
                "ran_successfully": res["ran_successfully"],
                "lean_code": code[:8000],
                "stderr": res.get("stderr", ""),
            })
            ok = "OK" if res["ran_successfully"] else "FAIL"
            print(f"  Lean chunk {i + 1}: {ok}", flush=True)
        except Exception as e:
            print(f"  Lean chunk {i + 1}: ERROR - {e}", flush=True)
            chunk_results.append({
                "ran_successfully": False,
                "lean_code": "",
                "stderr": str(e),
            })

    return {
        "subagent_responses": {
            "lean_verify": {
                "ran_successfully": all(r["ran_successfully"] for r in chunk_results),
                "chunk_results": chunk_results,
            }
        }
    }
