from schemas import AgentFState, CodeExtractorOutput
from langchain_core.prompts import ChatPromptTemplate
from config import llm
from utils import parse_json_response
import re

async def CodeExtractorAgent(state: AgentFState) -> AgentFState:
    paper_text = state.get("query") or ""
    print("[CodeExtractor] input length:", len(paper_text), "chars", flush=True)

    # Debug: Check if code signatures appear in the text
    code_indicators = ['class Solution:', 'def ', 'function ', 'import ', 'return ', 'if __name__']
    found_indicators = [ind for ind in code_indicators if ind in paper_text]
    if found_indicators:
        print(f"[CodeExtractor] Found code indicators: {found_indicators}", flush=True)

    # Show last 1000 chars to verify end of document is included
    print(f"[CodeExtractor] Last 200 chars of input: ...{paper_text[-200:]}", flush=True)

    extractor_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at extracting code from academic papers. "
            "Given the full text of a research paper, identify and extract ALL code chunks.\n\n"
            "WHAT TO EXTRACT:\n"
            "- Complete functions and classes (Python, JavaScript, TypeScript, Java, C++, etc.)\n"
            "- Code snippets with clear syntax (even if incomplete)\n"
            "- Algorithm implementations in actual programming languages\n"
            "- Shell/bash commands that are executable\n"
            "- Any text that looks like source code with proper indentation and syntax\n\n"
            "WHAT TO SKIP:\n"
            "- High-level pseudocode without specific syntax (e.g., 'for each item, do something')\n"
            "- Mathematical notation that isn't code\n"
            "- Plain English descriptions of algorithms\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Scan the ENTIRE document including the beginning, middle, AND END\n"
            "- Code often appears at the end of papers - don't miss it!\n"
            "- Include code even if it references undefined variables or seems incomplete\n"
            "- Look for function definitions (def, function, class, etc.)\n"
            "- Look for code blocks with indentation and programming syntax\n"
            "- When you find a function followed by test/assert lines, extract ONLY the function\n\n"
            "OUTPUT FORMAT:\n"
            "For each code chunk, provide: code (the actual code), language (python/javascript/etc), context (brief description)\n"
            "Respond with ONLY valid JSON in this exact format:\n"
            '{{"chunks": [{{"code": "...", "language": "python", "context": "description"}}, ...]}}\n'
            "If truly no code exists, return: {{\"chunks\": []}}"
        )),
        ("human", "Extract all code chunks from this paper. Scan the entire document carefully, including the end:\n\n{paper_text}")
    ])

    chunks = []
    try:
        # Use higher max_tokens to prevent truncation of long code extraction responses
        chain = extractor_prompt | llm.bind(max_tokens=8192)
        response = await chain.ainvoke({"paper_text": paper_text})
        raw = (response.content or "").strip()
        if raw:
            result = parse_json_response(raw, CodeExtractorOutput, llm=llm)
            chunks = [
                {"code": str(c.code), "language": str(c.language), "context": str(c.context)}
                for c in (result.chunks or [])
            ]
            print(f"[CodeExtractor] parsed {len(chunks)} chunks", flush=True)
            if len(chunks) == 0:
                print(f"[CodeExtractor] LLM returned empty chunks. Raw response length: {len(raw)} chars", flush=True)
                print(f"[CodeExtractor] First 500 chars of raw response: {raw[:500]}", flush=True)
    except Exception as e:
        print(f"[CodeExtractor] parse failed: {e}", flush=True)
        print(f"[CodeExtractor] First 300 chars of raw LLM response: {raw[:300]}", flush=True)

        # Fallback: bracket-balancing manual extraction
        if 'raw' in locals() and raw:
            # Strip markdown code fences if present (even with text before them)
            fallback_text = raw

            # Find first occurrence of ``` and remove everything before it
            fence_start = fallback_text.find('```')
            if fence_start != -1:
                # Skip past the fence line (```json or ```)
                fallback_text = fallback_text[fence_start:]
                newline_after_fence = fallback_text.find('\n')
                if newline_after_fence != -1:
                    fallback_text = fallback_text[newline_after_fence + 1:]

            # Find last occurrence of ``` and remove it and everything after
            fence_end = fallback_text.rfind('```')
            if fence_end != -1:
                fallback_text = fallback_text[:fence_end]

            print(f"[CodeExtractor] Attempting fallback on cleaned text (length: {len(fallback_text)})", flush=True)
            print(f"[CodeExtractor] First 500 chars of cleaned text: {fallback_text[:500]}", flush=True)

            # Try direct JSON parse first (simpler and more reliable than bracket balancing)
            try:
                import json
                data = json.loads(fallback_text)
                if isinstance(data, dict) and "chunks" in data:
                    arr = data["chunks"]
                    print(f"[CodeExtractor] Direct JSON parse succeeded! Found {len(arr)} items in chunks array", flush=True)
                    for item in arr:
                        if isinstance(item, dict) and item.get("code"):
                            chunks.append({
                                "code": str(item.get("code", "")),
                                "language": str(item.get("language", "python")),
                                "context": str(item.get("context", "")),
                            })
                    if chunks:
                        print(f"[CodeExtractor] Direct parse extracted {len(chunks)} code chunks!", flush=True)
            except Exception as parse_ex:
                print(f"[CodeExtractor] Direct JSON parse failed: {parse_ex}", flush=True)
                # Fall back to bracket balancing if direct parse fails
                pass

            # Only try bracket balancing if direct parse didn't work
            if len(chunks) == 0:
                match = re.search(r'"chunks"\s*:\s*\[', fallback_text)
            if match:
                print(f"[CodeExtractor] Found 'chunks' array in fallback text", flush=True)
                start = fallback_text.find("[", match.start())
                print(f"[CodeExtractor] Starting bracket balancing at position {start}, text length: {len(fallback_text)}", flush=True)
                depth = 0
                found_end = False
                for i in range(start, len(fallback_text)):
                    c = fallback_text[i]
                    if c == "[":
                        depth += 1
                    elif c == "]":
                        depth -= 1
                        if depth == 0:
                            found_end = True
                            arr_str = fallback_text[start : i + 1]
                            print(f"[CodeExtractor] Extracted array (length: {len(arr_str)}), first 300 chars: {arr_str[:300]}", flush=True)
                            try:
                                import json
                                arr = json.loads(arr_str)
                                print(f"[CodeExtractor] Successfully parsed array with {len(arr)} items", flush=True)
                                for item in arr:
                                    if isinstance(item, dict) and item.get("code"):
                                        chunks.append({
                                            "code": str(item.get("code", "")),
                                            "language": str(item.get("language", "python")),
                                            "context": str(item.get("context", "")),
                                        })
                                if chunks:
                                    print(f"[CodeExtractor] fallback parsed {len(chunks)} chunks", flush=True)
                                else:
                                    print(f"[CodeExtractor] Array had {len(arr)} items but no valid code chunks", flush=True)
                            except Exception as ex:
                                print(f"[CodeExtractor] Fallback JSON parse error: {ex}", flush=True)
                            break

                if not found_end:
                    print(f"[CodeExtractor] Never found closing ] for chunks array (final depth: {depth})", flush=True)
            else:
                print(f"[CodeExtractor] 'chunks' array NOT found in fallback text", flush=True)

    return {
        "subagent_responses": {
            "code_extractor": {"chunks": chunks}
        }
    }
