import json
import re
from pydantic import BaseModel
from typing import Type, TypeVar

T = TypeVar("T", bound=BaseModel)


def _sanitize_json_string(text: str) -> str:
    def fix_string_value(match):
        s = match.group(0)
        s = s.replace('\n', '\\n')
        s = s.replace('\r', '\\r')
        s = s.replace('\t', '\\t')
        return s

    return re.sub(r'"(?:[^"\\]|\\.)*"', fix_string_value, text, flags=re.DOTALL)


def parse_json_response(text: str, model: Type[T]) -> T:
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

    try:
        return model.model_validate_json(raw)
    except Exception:
        pass

    # Truncated JSON (e.g. LLM hit token limit): try closing the structure
    if '"chunks"' in raw or '"code"' in raw:
        for suffix in [
            '", "context": ""}]}',   # inside "language": "..." -> close string, add context, close chunk/array
            '"}]}',                   # inside "language": " -> close string, close object, array, outer
            '"]}',                   # after last complete chunk
        ]:
            try:
                return model.model_validate_json(raw + suffix)
            except Exception:
                continue

    sanitized = _sanitize_json_string(raw)
    return model.model_validate_json(sanitized)