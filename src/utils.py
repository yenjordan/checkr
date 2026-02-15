import re
from typing import Type, TypeVar, Optional, Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _sanitize_json_string(text: str) -> str:
    def fix_string_value(match):
        s = match.group(0)
        if len(s) < 2:
            return s

        content = s[1:-1]  # Remove surrounding quotes

        # Process character by character to handle escapes properly
        result = []
        i = 0
        while i < len(content):
            if content[i] == '\\' and i + 1 < len(content):
                next_char = content[i + 1]
                # Check if this is a valid JSON escape sequence
                # Valid: \" \\ \/ \b \f \n \r \t \uXXXX
                if next_char in '"\\/:bfnrt':
                    # Valid escape, keep as is
                    result.append('\\')
                    result.append(next_char)
                    i += 2
                elif next_char == 'u':
                    # Unicode escape, keep as is
                    result.append('\\u')
                    i += 2
                else:
                    # Invalid escape (like \text in LaTeX), double the backslash
                    result.append('\\\\')
                    i += 1
            elif content[i] == '\n':
                # Actual newline character, escape it
                result.append('\\n')
                i += 1
            elif content[i] == '\r':
                # Actual carriage return, escape it
                result.append('\\r')
                i += 1
            elif content[i] == '\t':
                # Actual tab character, escape it
                result.append('\\t')
                i += 1
            else:
                result.append(content[i])
                i += 1

        return '"' + ''.join(result) + '"'

    return re.sub(r'"(?:[^"\\]|\\.)*"', fix_string_value, text, flags=re.DOTALL)


def _repair_leading_brace(raw: str) -> str:
    """Fix JSON where LLM output has extra '{' at start (e.g. '{ { \"key\": ...')."""
    # Match leading { followed by optional whitespace and another {
    m = re.match(r"^\s*\{\s*\{", raw)
    if m:
        return raw[m.end() - 1 :]  # keep single { and rest
    return raw


def _normalize_list_fields(raw: str) -> str:
    """Convert JSON fields with list values to joined strings.

    Handles: "source_text": ["item1", "item2"] → "source_text": "item1\\nitem2"
    """
    import json

    try:
        data = json.loads(raw)

        def normalize_obj(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, list) and key in ('source_text', 'context'):
                        obj[key] = '\n'.join(str(item) for item in value if item)
                    elif isinstance(value, (dict, list)):
                        normalize_obj(value)
            elif isinstance(obj, list):
                for item in obj:
                    normalize_obj(item)

        normalize_obj(data)
        return json.dumps(data)
    except Exception:
        return raw


def parse_json_response(text: str, model: Type[T], llm: Optional[Any] = None) -> T:
    # Try to extract JSON from markdown code fences
    # Look for ```json or just ``` followed by JSON
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        raw = match.group(1).strip()
    else:
        # If no code fence match, manually clean the text
        text_cleaned = text.strip()

        # Remove leading text before first ```
        fence_start = text_cleaned.find('```')
        if fence_start != -1:
            text_cleaned = text_cleaned[fence_start:]
            # Skip the fence line (```json or ```)
            first_newline = text_cleaned.find('\n')
            if first_newline != -1:
                text_cleaned = text_cleaned[first_newline + 1:]

        # Remove trailing ```
        fence_end = text_cleaned.rfind('```')
        if fence_end != -1:
            text_cleaned = text_cleaned[:fence_end]

        # Find the JSON object
        start = text_cleaned.find("{")
        if start != -1:
            raw = text_cleaned[start:].strip()
        else:
            raw = text_cleaned.strip()

    raw = raw.strip()

    raw = re.sub(r',\s*//[^\n]*', ',', raw)
    raw = re.sub(r'^\s*#.*$', '', raw, flags=re.MULTILINE)

    last_error = None
    candidates = [raw, _repair_leading_brace(raw), _sanitize_json_string(raw), _normalize_list_fields(raw)]
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
            prompt = (
                "Fix the following JSON so it is valid. "
                "For any fields containing arrays of strings that should be single strings, "
                "join them with newlines. Preserve all content. "
                "Output only the corrected JSON, no other text.\n\n" + raw
            )
            fixed = llm.invoke([HumanMessage(content=prompt)])
            out = getattr(fixed, "content", None) or str(fixed)
            if out and out.strip():
                return model.model_validate_json(out.strip())
        except Exception as e:
            last_error = e

    if last_error is not None:
        raise last_error
    raise ValueError("Could not parse JSON response")
