import json
import re
from typing import Type, TypeVar, Optional, Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _schema_description_for_model(model: Type[BaseModel]) -> str:
    try:
        schema = model.model_json_schema()
        return json.dumps(schema, indent=0)[:1200]
    except Exception:
        return '{"chunks": [{"latex": "...", "context": "...", "equation_type": "...", "source_text": "..."}, ...]}'


def _extract_json_from_llm_output(text: str) -> str:
    """Strip markdown/code fences and return the JSON part for validation."""
    if not text or not text.strip():
        return ""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    if start != -1:
        return text[start:].strip()
    return text


def fix_json_with_llm(raw: str, model: Type[T], llm: Any) -> T:
    schema_desc = _schema_description_for_model(model)
    prompt = (
        "The following JSON may be truncated, malformed, or contain syntax errors. "
        "Fix or complete it so it is valid and matches the expected schema. "
        "Preserve all content; only fix structure (close brackets, escape quotes, fix commas). "
        "Output ONLY the corrected JSON: no explanation, no markdown code fences, no surrounding text.\n\n"
        "Expected schema (for reference):\n"
        f"{schema_desc}\n\n"
        "Broken JSON:\n\n"
        f"{raw}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    out = getattr(response, "content", None) or str(response)
    extracted = _extract_json_from_llm_output(out)
    if not extracted:
        raise ValueError("LLM returned no JSON")
    return model.model_validate_json(extracted)


def _sanitize_json_string(text: str) -> str:
    def fix_string_value(match):
        s = match.group(0)
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return s

    return re.sub(r'"(?:[^"\\]|\\.)*"', fix_string_value, text, flags=re.DOTALL)


def _repair_leading_brace(raw: str) -> str:
    """Fix JSON where LLM output has extra '{' at start (e.g. '{ { \"key\": ...')."""
    # Match leading { followed by optional whitespace and another {
    m = re.match(r"^\s*\{\s*\{", raw)
    if m:
        return raw[m.end() - 1 :]  # keep single { and rest
    return raw


def _escape_control_chars_in_strings(raw: str) -> str:
    result = []
    in_string = False
    escape_next = False
    i = 0
    while i < len(raw):
        c = raw[i]
        if escape_next:
            result.append(c)
            escape_next = False
            i += 1
            continue
        if c == "\\" and in_string:
            result.append(c)
            escape_next = True
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
            i += 1
            continue
        if in_string and ord(c) < 32:
            if c == "\n":
                result.append("\\n")
            elif c == "\r":
                result.append("\\r")
            elif c == "\t":
                result.append("\\t")
            else:
                result.append(f"\\u{ord(c):04x}")
            i += 1
            continue
        result.append(c)
        i += 1
    return "".join(result)


def parse_json_response(text: str, model: Type[T], llm: Optional[Any] = None) -> T:
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        raw = match.group(1)
    else:
        start = text.find("{")
        if start != -1:
            raw = text[start:]
        else:
            raw = text

    raw = raw.strip()

    raw = re.sub(r',\s*//[^\n]*', ',', raw)
    raw = re.sub(r'^\s*#.*$', '', raw, flags=re.MULTILINE)

    last_error = None
    raw_fixed_control = _escape_control_chars_in_strings(raw)
    candidates = [
        raw,
        raw_fixed_control,
        _repair_leading_brace(raw),
        _repair_leading_brace(raw_fixed_control),
        _sanitize_json_string(raw),
        _sanitize_json_string(raw_fixed_control),
    ]
    for s in candidates:
        if not s or not s.strip():
            continue
        try:
            return model.model_validate_json(s)
        except Exception as e:
            last_error = e

    structural = re.sub(r',(\s*[}\]])', r'\1', raw)
    structural = re.sub(r',\s*,', ',', structural)
    try:
        return model.model_validate_json(structural)
    except Exception as e:
        last_error = e

    if llm is not None:
        try:
            return fix_json_with_llm(raw, model, llm)
        except Exception as e:
            last_error = e

    if last_error is not None:
        raise last_error
    raise ValueError("Could not parse JSON response")
