from schemas import AgentFState, MathExtractorOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
import re
from utils import parse_json_response

async def MathExtractorAgent(state: AgentFState) -> AgentFState:
    paper_text = state.get("query") or ""
    print("[MathExtractor] input length:", len(paper_text), "chars", flush=True)

    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at extracting mathematical content from academic papers. "
            "Given the full text (which may be OCR output with imperfect formatting), identify and extract ALL mathematical content: "
            "display equations, inline formulas, metrics (e.g. ET = ..., MP = ..., MI = ..., clip(0,1,x)), definitions, and derivations. "
            "ALWAYS extract: any formula containing clip(0,1,...), Sum, fractions, or explicit = with an expression; efficiency/metric definitions (MP, MI, ET, etc.). "
            "Extract even when notation is inline or slightly broken: preserve subscripts (e.g. T_H, M_i, A_H), function names (e.g. clip, Sum), and ratios. "
            "Only skip: standalone labels with no formula (e.g. 'Pass@1', 'BLEU' with no equation). "
            "When in doubt, include the chunk — it is better to include a borderline formula than to omit it.\n"
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
            "Respond with ONLY a JSON object in this exact format:\n"
            '{{"chunks": [{{"latex": "<properly formatted LaTeX>", "context": "<short description>", "equation_type": "<type>", "source_text": "<optional exact snippet from paper>"}}, ...]}}\n'
            "Include every distinct equation or formula. Only return {{}} if the paper truly has no equations, no metrics, and no formulas at all."
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
                                    chunks.append({
                                        "latex": str(item.get("latex", "")),
                                        "context": str(item.get("context", "")),
                                        "equation_type": str(item.get("equation_type", "equation")),
                                        "source_text": str(item.get("source_text", "")),
                                    })
                            if chunks:
                                print("[MathExtractor] fallback parsed", len(chunks), "chunks", flush=True)
                        except Exception:
                            pass
                        break

    return {
        "subagent_responses": {
            "math_extractor": {"chunks": chunks}
        }
    }
