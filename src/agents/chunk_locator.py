"""Chunk Locator Agent

Given page layout lines (from Document AI) and extracted math/code chunks,
uses an LLM to find exactly which lines each chunk corresponds to.

This is fundamentally an LLM task because the math extractor returns LaTeX
(e.g. \\theta, \\frac{x}{y}) while Document AI text uses Unicode (θ, x/y).
Token-based matching can't bridge this gap reliably.
"""

import json
import re
from langchain_core.prompts import ChatPromptTemplate
from config import llm


async def locate_chunks(
    page_layouts: list[list[dict]],
    math_chunks: list[dict],
    code_chunks: list[dict],
) -> None:
    """Annotate each chunk with _loc = {page, lines} in-place."""

    # Build numbered line listing grouped by page
    # Each line gets a global ID for unambiguous LLM reference
    line_listing = []
    line_map: dict[int, tuple[int, int]] = {}  # global_id → (page_idx, line_idx)
    gid = 0
    for pi, page_lines in enumerate(page_layouts):
        line_listing.append(f"\n--- Page {pi + 1} ---")
        for li, line in enumerate(page_lines):
            text = line.get("t", "").strip()
            if text:
                line_listing.append(f"[{gid}] {text}")
                line_map[gid] = (pi, li)
                gid += 1

    if not line_listing or (not math_chunks and not code_chunks):
        return

    paper_lines = "\n".join(line_listing)

    # Build chunk descriptions
    chunk_descs = []
    chunk_keys = []

    for i, ch in enumerate(math_chunks):
        key = f"M{i}"
        chunk_keys.append(("math", i, key))
        latex = ch.get("latex", "")
        ctx = ch.get("context", "")
        chunk_descs.append(f'{key}: LaTeX: {latex}  |  Context: {ctx}')

    for i, ch in enumerate(code_chunks):
        key = f"C{i}"
        chunk_keys.append(("code", i, key))
        code = (ch.get("code", "") or "")[:200]
        lang = ch.get("language", "")
        chunk_descs.append(f'{key}: {lang} code starting with: {code}')

    if not chunk_descs:
        return

    chunks_text = "\n".join(chunk_descs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert at locating mathematical equations and code snippets "
            "within OCR-extracted text from academic papers.\n\n"
            "You will be given:\n"
            "1. Numbered lines from a paper (each prefixed with [ID])\n"
            "2. A list of extracted chunks (math equations in LaTeX, or code snippets)\n\n"
            "Your job: for each chunk, find which line ID(s) contain that equation or code.\n\n"
            "IMPORTANT NOTES:\n"
            "- Math in the paper text uses Unicode (θ, ∑, ∂, ∞, ≤) NOT LaTeX (\\theta, \\sum)\n"
            "- Fractions like \\frac{{a}}{{b}} may appear as a/b or on separate lines\n"
            "- An equation may span 1-3 lines. Return ALL relevant line IDs.\n"
            "- If you cannot find a chunk, return an empty array for it.\n"
            "- Return ONLY line IDs, not the text.\n\n"
            "Respond with ONLY a JSON object mapping chunk keys to arrays of line IDs:\n"
            '{{"M0": [45], "M1": [102, 103], "C0": [200, 201, 202], ...}}'
        )),
        ("human",
         "PAPER LINES:\n{paper_lines}\n\n"
         "CHUNKS TO LOCATE:\n{chunks_text}")
    ])

    try:
        raw = (await (prompt | llm).ainvoke({
            "paper_lines": paper_lines,
            "chunks_text": chunks_text,
        })).content or ""

        # Parse the JSON response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return
        result = json.loads(match.group(0))

        # Apply results to chunks
        for ctype, idx, key in chunk_keys:
            line_ids = result.get(key, [])
            if not line_ids or not isinstance(line_ids, list):
                continue

            # Convert global IDs to (page, line_idx) pairs
            pages = set()
            line_indices_by_page: dict[int, list[int]] = {}
            for lid in line_ids:
                if isinstance(lid, int) and lid in line_map:
                    pi, li = line_map[lid]
                    pages.add(pi)
                    line_indices_by_page.setdefault(pi, []).append(li)

            if not line_indices_by_page:
                continue

            # Pick the page with the most matched lines
            best_page = max(line_indices_by_page, key=lambda p: len(line_indices_by_page[p]))
            lines = sorted(line_indices_by_page[best_page])

            loc = {"page": best_page, "lines": lines}

            if ctype == "math" and idx < len(math_chunks):
                math_chunks[idx]["_loc"] = loc
            elif ctype == "code" and idx < len(code_chunks):
                code_chunks[idx]["_loc"] = loc

    except Exception as e:
        print(f"[ChunkLocator] Failed: {e}")
