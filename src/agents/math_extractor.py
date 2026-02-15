import asyncio
from schemas import AgentFState, MathExtractorOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
import re
from utils import parse_json_response, fix_json_with_llm

_CODE_MARKERS = (
    # Python-specific
    "self.", "np.", "def ", "return ", ".append(", "import ", "functools", "operator.",
    " in range(", "len(", ".size", ".ndim", "for a in", "for i in", "if any(", "if min(",
    "forain", "foriin", "__", "self[", "lambda ", "yield ", ".reshape", ".mean(", ".std(",

    # Common programming patterns (language-agnostic)
    "==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=",
    "function(", "const ", "let ", "var ", ".push(", ".pop(", ".map(", ".filter(",
    "array[", "list[", ".get(", ".set(", ".add(", "new ", "null", "None", "true", "false",
    ".length", "[i]", "[j]", "printf", "cout", "print(", "console.log",

    # Method/attribute access patterns
    ".item(", ".value", ".data[", ".shape[", ".dtype", "torch.", "tf.", "jax.",

    # Control flow
    "if (", "else:", "elif ", "while (", "while ", "for (", "switch(", "case ",

    # Common code variable names that aren't math
    "batch_size", "num_epochs", "learning_rate", "max_iter", "min_value", "config.",
)


def _is_code(chunk: dict) -> bool:
    """
    Detect if a chunk is actually code rather than a mathematical equation.
    Check latex, context, and equation_type fields for code indicators.
    """
    latex = chunk.get("latex") or ""
    latex_lower = latex.lower()
    context_lower = (chunk.get("context") or "").lower()
    eq_type = (chunk.get("equation_type") or "").lower()

    # Check equation_type - if it's labeled as "code" or related, it's code
    if any(keyword in eq_type for keyword in ["code", "assignment", "snippet", "implementation", "original", "optimized", "intermediate"]):
        return True

    # Check context for code-related keywords
    code_context_keywords = [
        "original code", "original source", "chatgpt", "attempt", "final adjustments",
        "using while", "using for", "loop", "tuple-based", "merged version",
        "pivotal moment", "incorrect code", "optimized code", "intermediate code",
        "source code", "implementation", "snippet"
    ]
    if any(keyword in context_lower for keyword in code_context_keywords):
        return True

    # Check for control flow keywords (these should NEVER be math)
    control_keywords = ["break", "continue", "return", "if:", "if ", "else:", "elif ", "while ", "for "]
    if any(keyword in latex_lower for keyword in control_keywords):
        return True

    # Check for explicit code markers
    if any(m in latex for m in _CODE_MARKERS):
        return True

    # AGGRESSIVE: Check for simple variable assignments (the most common false positive)
    # Pattern: "variable = number" or "var = constant" with NO mathematical operators or structure
    # Examples to catch: "n = 255", "x = 0", "i = 0", "a, b = 1, 2"
    if "=" in latex:
        # No LaTeX structure? Check if it's a simple assignment
        has_latex_structure = any(marker in latex for marker in ["\\", "^", "_"])

        if not has_latex_structure:
            # Remove spaces for easier pattern matching
            latex_no_space = latex.replace(" ", "").replace("\n", "")

            # Pattern 1: Simple number assignment: "x=123", "n=255"
            import re
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*=[0-9]+$', latex_no_space):
                return True

            # Pattern 2: Multiple variable assignment: "a,b=1,2" or "n,x=255,0"
            if "," in latex and re.match(r'^[a-zA-Z_,0-9]+=[0-9,]+$', latex_no_space):
                return True

            # Pattern 3: Variable assignment with simple arithmetic but no LaTeX
            # "x = 0", "res_min = i", etc. - if it's short and looks like assignment
            if len(latex) < 30 and re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[a-zA-Z0-9_]+', latex):
                # Check that it's NOT a mathematical formula
                # Math would have operators like +, -, *, /, or functions
                has_math_ops = any(op in latex for op in ["+", "-", "*", "/", "^", "\\frac", "\\sum", "\\int", "\\sqrt", "\\log", "\\exp"])
                if not has_math_ops:
                    # It's just "var = val" with no operators - that's code
                    return True

        # Count equals signs - multiple assignments suggest code
        equals_count = latex.count("=")
        if equals_count > 2:  # Multiple assignments suggest code
            return True

        # Look for variable assignment patterns without LaTeX structure
        if not has_latex_structure and ("(" in latex or "[" in latex):
            # Has parentheses or brackets but no LaTeX - likely code
            return True

    # Check for array/list indexing patterns
    bracket_patterns = ["][", "[i]", "[j]", "[k]", "[0]", "[1]", "[-1]"]
    if any(pattern in latex for pattern in bracket_patterns):
        return True

    # Check for common code idioms
    code_idioms = ["range(", "len(", ".split(", ".join(", ".strip(", "enumerate("]
    if any(idiom in latex_lower for idiom in code_idioms):
        return True

    return False


def _is_problematic_math(chunk: dict) -> bool:
    latex = chunk.get("latex") or ""
    context_lower = (chunk.get("context") or "").lower()

    if any(keyword in context_lower for keyword in ["let ", "define ", "denote ", "where "]):
        if ":=" in latex or ":=" in context_lower:
            print(f"[MathExtractor] Skipping definition statement: {latex[:60]}", flush=True)
            return True

    if "\\sum" in latex or "∑" in latex:
        # Check if summation has proper bounds
        has_bounds = ("_{" in latex and "^" in latex) or ("_" in latex and "^" in latex)
        if not has_bounds:
            if "=" not in latex:
                return True

    if latex.count("_") > 3 and "\\sum" not in latex and "\\prod" not in latex:
        if "," in latex and "_" in latex and "(" in latex:
            if "=" not in latex:
                return True

    return False


async def MathExtractorAgent(state: AgentFState) -> AgentFState:
    paper_text = state.get("query") or ""
    print("[MathExtractor] input length:", len(paper_text), "chars", flush=True)

    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at extracting mathematical content from academic papers. "
            "Extract ONLY predicates and formulas: real mathematical expressions, equations with structure, metric definitions, loss functions, etc.\n\n"
            "DO NOT extract:\n"
            "- A lone variable or symbol (e.g. just 'N', 'x', 'k' with no formula)\n"
            "- **SIMPLE VARIABLE ASSIGNMENTS** - These are CODE, not math equations:\n"
            "  * NEVER extract: n = 255, x = 0, i = 0, res_min = i, res_max = i\n"
            "  * NEVER extract: n, x = 255, 0 or a, b, c = 1, 2, 3\n"
            "  * NEVER extract: batch_size = 32, learning_rate = 0.001, k = 3\n"
            "  These are programming variable assignments, NOT mathematical equations!\n"
            "- **CODE SNIPPETS AND STATEMENTS**: Do NOT extract programming code:\n"
            "  * Control flow: break, continue, if val:, while i < n:\n"
            "  * Programming syntax: ==, !=, <=, >=, &&, ||, for loops, if/else\n"
            "  * Array/list indexing: array[i], list[0], values[j]\n"
            "  * Language keywords: def, return, import, let, const, var, lambda, print\n"
            "  * Method calls: .append(), .map(), .filter(), .reshape()\n"
            "  * Function calls: len(), range(), sum(), print()\n"
            "  * Code in context: If the context says 'original code', 'optimized code', 'ChatGPT attempt', etc. - it's CODE!\n"
            "- **INCOMPLETE OR PURE DEFINITION STATEMENTS**:\n"
            "  * Pure definitions with ':=' that don't have verifiable equations (e.g., 'Let x := the value of y')\n"
            "  * Incomplete summations without bounds or equation structure (e.g., just '∑ f(x)' with no = or bounds)\n"
            "  * Notation fragments that aren't complete equations (e.g., 't_{{s,r,n}}(f)' without equation context)\n\n"
            "DO extract: formulas and predicates, e.g. "
            "display equations, metric definitions (ET = ..., MP = ..., clip(0,1,x)), expressions with clip(0,1,...), Sum, fractions, ratios, "
            "definitions where the RHS is an expression (not just a number), loss functions, derivations. "
            "Preserve subscripts (T_H, M_i), function names (clip, Sum), and LaTeX structure. "
            "Skip standalone labels with no formula (e.g. 'Pass@1', 'BLEU'). "
            "When in doubt, include only if it is a real formula/predicate, not code or a simple variable/constant assignment.\n\n"
            "**DISTINGUISHING CODE FROM MATH**:\n"
            "- CODE (DO NOT EXTRACT): `n = 255`, `x = 0`, `res_min = i`, `break`, `if val:`, `a, b = 1, 2`\n"
            "- CODE (DO NOT EXTRACT): `loss = sum(errors) / len(errors)`, `x = a * b + c`, `for i in range(N): total += x[i]`\n"
            "- MATH (DO EXTRACT): `L = \\frac{{\\sum_{{i}} e_i}}{{N}}`, `x = a \\cdot b + c`, `S = \\sum_{{i=1}}^{{N}} x_i`\n"
            "Mathematical notation uses proper mathematical symbols, LaTeX formatting, and equation structure.\n"
            "Code uses programming language syntax, method calls, and procedural logic.\n"
            "**CRITICAL**: If the surrounding context mentions 'code', 'implementation', 'snippet', 'ChatGPT', or programming - SKIP IT!\n\n"
            "For each chunk provide: latex (the formula in LaTeX or clear ASCII), context (what it denotes), equation_type (e.g. definition, loss function, efficiency metric).\n\n"
            "CRITICAL LATEX FORMATTING RULES - FOLLOW EXACTLY:\n"
            "- ALL subscripts with multiple characters MUST use braces: T_{{H}}, T_{{llm}}, T_{{text}}\n"
            "- For nested subscripts, wrap the ENTIRE subscript: T_{{\\text{{llm}}_{{i}}}} NOT T_\\text{{llm}}_i\n"
            "- Text in subscripts: ALWAYS use \\text{{}}: P_{{\\text{{pass}}}}, E_{{\\text{{total}}}}, A_{{\\text{{llm}}}}\n"
            "- Fractions: ALWAYS \\frac{{numerator}}{{denominator}} with braces around both parts\n"
            "- Summations: \\sum_{{i=1}}^{{N}}, \\sum_{{k}} with bounds in braces\n"
            "- Multiplication: use \\cdot or * between variables: a \\cdot b, not ab (unless single letters)\n"
            "- Functions: \\log, \\exp, \\sin, \\cos (with backslash)\n"
            "- Greek letters: \\alpha, \\beta, \\theta, etc. (with backslash)\n"
            "- Ensure ALL braces are balanced - count opening {{ and closing }}\n"
            "- NO Unicode math symbols - use LaTeX commands only\n"
            "- Clean up OCR artifacts: remove stray spaces in subscripts, fix broken commands\n\n"
            "**EXTRACTION COMPLETENESS**: Your goal is to find and extract ALL mathematical equations, formulas, and predicates in the paper. "
            "Be thorough and comprehensive - scan the entire text carefully. Include:\n"
            "- Display equations and numbered equations\n"
            "- Inline mathematical expressions with substantive content\n"
            "- Algorithm pseudocode that contains mathematical formulas (but NOT the programming code)\n"
            "- Metric definitions, loss functions, objective functions\n"
            "- Theoretical results, lemmas, theorems with mathematical statements\n"
            "- All variations and derived forms of equations\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"chunks": [{{"latex": "<properly formatted LaTeX>", "context": "<short description>", "equation_type": "<type>", "source_text": "<optional exact snippet from paper>"}}, ...]}}\n'
            'Include every distinct equation or formula. Be comprehensive! Only return {{"chunks": []}} if the paper truly has no equations, no metrics, and no formulas at all.'
        )),
        ("human", "Extract all mathematical content from:\n\n{paper_text}")
    ])

    raw = (await (extractor_prompt | llm).ainvoke({"paper_text": paper_text})).content or ""
    raw = raw.strip()

    chunks = []
    try:
        result = parse_json_response(raw, MathExtractorOutput, llm=llm)
        chunks = [{"latex": c.latex, "context": c.context, "equation_type": c.equation_type, "source_text": getattr(c, "source_text", "") or ""} for c in (result.chunks or [])]
        print("[MathExtractor] parsed", len(chunks), "chunks", flush=True)
    except Exception as e:
        print("[MathExtractor] parse failed:", e, flush=True)
        try:
            result = await asyncio.to_thread(fix_json_with_llm, raw, MathExtractorOutput, llm)
            chunks = [{"latex": c.latex, "context": c.context, "equation_type": c.equation_type, "source_text": getattr(c, "source_text", "") or ""} for c in (result.chunks or [])]
            print("[MathExtractor] LLM-fixed JSON, parsed", len(chunks), "chunks", flush=True)
        except Exception as e2:
            print("[MathExtractor] LLM fix failed:", e2, flush=True)
            match = re.search(r'"chunks"\s*:\s*\[', raw)
            if match:
                start = raw.find("[", match.start())
                depth = 0
                for i in range(start, len(raw)):
                    c = raw[i]
                    if c == "[":
                        depth += 1
                    elif c == "]":
                        depth -= 1
                        if depth == 0:
                            arr_str = raw[start : i + 1]
                            try:
                                import json
                                arr = json.loads(arr_str)
                                for item in arr:
                                    if isinstance(item, dict) and item.get("latex"):
                                        st = item.get("source_text", "")
                                        if isinstance(st, list):
                                            st = " ".join(str(x) for x in st)
                                        else:
                                            st = str(st) if st else ""
                                        chunks.append({
                                            "latex": str(item.get("latex", "")),
                                            "context": str(item.get("context", "")),
                                            "equation_type": str(item.get("equation_type", "equation")),
                                            "source_text": st,
                                        })
                                if chunks:
                                    print("[MathExtractor] fallback parsed", len(chunks), "chunks", flush=True)
                            except Exception:
                                pass
                            break

    # Filter out code snippets and problematic math expressions
    original_count = len(chunks)
    filtered_chunks = []
    code_chunks = []
    problematic_chunks = []

    for c in chunks:
        if _is_code(c):
            code_chunks.append(c)
        elif _is_problematic_math(c):
            problematic_chunks.append(c)
        else:
            filtered_chunks.append(c)

    if code_chunks:
        print(f"[MathExtractor] Filtered out {len(code_chunks)} code snippets:", flush=True)
        for cc in code_chunks[:5]: 
            print(f"  - CODE: {cc.get('latex', '')[:80]}", flush=True)

    if problematic_chunks:
        print(f"[MathExtractor] Filtered out {len(problematic_chunks)} problematic expressions:", flush=True)
        for pc in problematic_chunks[:3]: 
            print(f"  - PROBLEMATIC: {pc.get('latex', '')[:80]}", flush=True)

    total_filtered = len(code_chunks) + len(problematic_chunks)
    print(f"[MathExtractor] Final: {len(filtered_chunks)}/{original_count} chunks (removed {total_filtered}: {len(code_chunks)} code, {len(problematic_chunks)} problematic)", flush=True)

    return {
        "subagent_responses": {
            "math_extractor": {"chunks": filtered_chunks}
        }
    }
