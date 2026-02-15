# CHECKR Demo / Walkthrough Script

Use this for a live walkthrough or recorded demo. Adjust timing and depth based on audience (investors = short punchy; technical = show math/code details).

---

## 1. Landing (10–15 sec)

**[ACTION]** Open CHECKR in browser. Land on homepage.

**[SAY]**
- "This is CHECKR — we help you **verify every claim** in a research paper."
- "You upload a PDF; we extract the code and math, run the checks, and give you a clear verdict. No PhD required."

**[ACTION]** Briefly point at hero: "Verify every claim," tagline, Get Started.

---

## 2. Get Started → Upload (15–20 sec)

**[ACTION]** Click **Get Started**. Overlay opens with paper stack + upload area.

**[SAY]**
- "You hit Get Started, then either pick one of the sample papers or upload your own."
- "We support PDFs — anything from a preprint to a published paper."

**[ACTION]** Upload a paper (or use a sample if available). Show filename appearing and **Check Paper** enabling.

**[SAY]**
- "Once you’ve got your file, you click **Check Paper**. CHECKR then runs the full pipeline: it plans what to verify, pulls out code and equations, runs the code, checks the math with SymPy, and synthesizes everything."

**[ACTION]** Click **Check Paper**.

---

## 3. Analysis View — Loading (5–10 sec)

**[ACTION]** Analysis view opens: PDF on the left, "Analyzing your paper" on the right.

**[SAY]**
- "The analysis usually takes a minute or two. You see the PDF on the left so you can follow along; results show up on the right."

**[OPTIONAL]** Mention: "Under the hood we’re using Document AI for the PDF, then a LangGraph workflow with planner, code and math extractors, execution, and verification agents."

---

## 4. Results — Verdict & Summary (20–30 sec)

**[ACTION]** When results load, scroll so **Verdict** and **Summary** are visible.

**[SAY]**
- "First thing you get is the **verdict**: Verified, Hard to Verify, or Unknown — plus a short **summary** in plain English."
- "That’s the answer most people care about: did the claims hold up or not?"

**[ACTION]** Expand **Verification Plan** if you want to show the planner.

**[SAY]** (if showing plan)
- "You can expand the verification plan to see the steps we took — what we decided to check and in what order."

---

## 5. Code Execution (15–25 sec)

**[ACTION]** Scroll to **Code Execution** (if the paper has code).

**[SAY]**
- "For any code in the paper, we extract it, run it, and report **PASSED** or **FAILED** — and we show a snippet plus a short analysis."
- "There’s also a **Conceptual Review** that checks whether the code actually does what the paper says, not just that it runs."

**[ACTION]** Click a code result card that has the "Find" / locate icon.

**[SAY]**
- "You can click **Find** on any result to jump to that spot in the PDF and see exactly where the code or equation lives."

---

## 6. Math Verification (20–30 sec)

**[ACTION]** Scroll to **Math Verification**.

**[SAY]**
- "For math, we extract equations, run them through **SymPy** for symbolic verification, and show you the proof steps."
- "Each equation gets a status: Verified, Failed, or Inconclusive — and you can expand the SymPy proof steps to see how we got there."
- "Where we have it, we also show **Lean 4** formalization — that’s the gold standard for formal verification."

**[ACTION]** Expand one math chunk: show equation (KaTeX), conclusion, and maybe "SymPy proof steps" or "Lean 4 formalization."

**[SAY]**
- "So you get both a high-level verdict and the option to dig into the actual derivation or formalization."

---

## 7. Optional: Chat & Voice (15–20 sec)

**[ACTION]** Click the **spark/chat** button to open the chat panel.

**[SAY]**
- "After the run, you can **chat** with CHECKR about the same paper — ask for key findings, a math check, code results, or methodology concerns. The model has full context from the analysis."
- "There’s also a **voice** option: you can talk to CHECKR about the paper via Hume EVI, for a more conversational follow-up."

**[ACTION]** Optionally click a suggestion (e.g. "Key findings" or "Math check") to show one reply, then close chat.

---

## 8. Closing (10–15 sec)

**[ACTION]** Return to verdict/summary or a strong result card.

**[SAY]**
- "So in one flow: upload a paper, get a verdict, see which code and math passed or failed, and drill into proof steps or chat for more."
- "Built for researchers before publication, founders building on science, and VCs doing deep-tech due diligence — **CHECKR checks every claim**."

**[ACTION]** Close results or go back home. End on hero or "How it works" if you have time.

---

## Short version (60–90 sec)

1. **Landing:** "CHECKR verifies every claim in a research paper — code and math — and gives you a clear verdict."
2. **Get Started → Upload** a PDF, click **Check Paper**.
3. **Analysis:** "We analyze for about a minute; PDF on the left, results on the right."
4. **Verdict:** "You get Verified / Hard to Verify plus a summary."
5. **Code:** "Code is run and marked Pass/Fail with a conceptual review."
6. **Math:** "Equations are checked with SymPy and proof steps; we show Lean 4 when available."
7. **Close:** "You can click Find to locate any result in the PDF, or chat/voice for follow-up. That’s CHECKR."

---

## Tips

- **Have one strong paper ready** (with both code and math) so Code Execution and Math Verification both have content.
- **Pre-load the tab** and do a test run so you know the verdict and where the good cards are.
- If **API or backend is slow**, keep the script flexible: "Usually a minute or two" and skip or shorten optional sections (chat, voice, full SymPy/Learn expansion).
- For **technical audiences**, add one line on stack: "FastAPI backend, LangGraph for the agent graph, Vertex AI for the models, SymPy for math, Document AI for the PDF."
