import re
from typing import Type, TypeVar, Optional, Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _sanitize_json_string(text: str) -> str:
    def fix_string_value(match):
        s = match.group(0)
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return s

    return re.sub(r'"(?:[^"\\]|\\.)*"', fix_string_value, text, flags=re.DOTALL)


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
    for s in (raw, _sanitize_json_string(raw)):
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
            prompt = "Fix the following JSON so it is valid. Preserve all content. Output only the corrected JSON, no other text.\n\n" + raw
            fixed = llm.invoke([HumanMessage(content=prompt)])
            out = getattr(fixed, "content", None) or str(fixed)
            if out and out.strip():
                return model.model_validate_json(out.strip())
        except Exception as e:
            last_error = e

    if last_error is not None:
        raise last_error
    raise ValueError("Could not parse JSON response")
