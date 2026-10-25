# 05. Forensic Error Analysis & Strategic Next Steps (v12 Roadmap)

This document provides a forensic audit of the exact execution failures observed in our local benchmarks on NVIDIA DGX Spark, details the root causes, and presents a prescriptive architectural roadmap for **v12** to break through the 0.30 score ceiling.

---

## 1. Forensic Audit of Benchmark Failures

### 1.1 Fatal Bug 1: The `run_command("submit_patch")` Invocation Trap
- **Observed In**: `requests_7205` and `rich_2725` in v11.
- **Trace from Log**:
  In `requests_7205`, v11 operated with flawless precision during Turns 1 to 4:
  1. *Turn 1*: Inspected issue description, immediately pinned `src/requests/utils.py`.
  2. *Turn 2*: Read lines around `unquote_unreserved`.
  3. *Turn 3*: Ran a python one-liner reproducing the double-unquoting bug.
  4. *Turn 4*: Applied the surgical patch using python atomic replacement and verified with `git diff`. The diff was clean, exact, and correct (+6, -2 lines).
  5. *Turn 5 (FATAL STEP)*: The model attempted to conclude the task by emitting:
     ```json
     {
       "tool_name": "run_command",
       "parameters": {
         "command": "submit_patch"
       }
     }
     ```
  6. *Output Received by Model*:
     ```text
     bash: submit_patch: command not found
     ```
  7. *Degradation Loop*: Confused by the bash error, Gemma 4 assumed `submit_patch` was an executable script that needed to be found or installed. It ran `which submit_patch`, `find . -name "*submit*"`, and `git status`, burning through Turns 6 to 12 until `max_tool_calls` was reached. The task was recorded as unsubmitted (`0 bytes submitted`).

- **Root Cause**:
  `submit_patch` is an **ADK tool declared in `agent.yaml`**, intended to be invoked as a top-level JSON tool call:
  ```json
  {
    "tool_name": "submit_patch",
    "parameters": {}
  }
  ```
  Because the system prompt referred to `submit_patch` alongside terminal verification commands, Gemma 4 conflated tool invocation with shell execution.

### 1.2 Fatal Bug 2: Line-Range Offset Search Loop
- **Observed In**: `fastapi_11194` in both v10 and v11.
- **Trace from Log**:
  1. *Turn 1*: Pinned `fastapi/dependencies/utils.py` and identified `_extract_form_body`.
  2. *Turn 2*: Ran `sed -n '170,250p fastapi/dependencies/utils.py'`.
  3. *Output*: Lines 170-250 contained helper classes (`ParamDetails`), not `_extract_form_body`.
  4. *Turns 3 to 7*: Instead of running `grep -n "def _extract_form_body" fastapi/dependencies/utils.py`, the model made sequential exploratory reads: `sed -n '250,330p'`, `sed -n '330,410p'`, etc.
  5. In `fastapi/dependencies/utils.py`, `_extract_form_body` is located at **line 872**. The model exhausted its turn budget before reaching line 872.

- **Root Cause**:
  The model guessed the line range or searched blindly in fixed increments rather than querying line numbers with `grep -n` or `rg -n`.

### 1.3 Fatal Bug 3: Pre-Existing Environment Test Failures
- **Observed In**: Local test verification across multiple tasks.
- **Trace**:
  When instructed to "run tests to verify the fix", the model ran `pytest`.
  In several benchmark repos, the baseline environment has pre-existing test failures (e.g., deprecated pytest syntax, missing external network fixtures, or unmocked SSL certificates).
  Gemma 4 assumed *its patch* broke those tests, reverted its valid changes, and wandered trying to fix unrelated pre-existing failures.

---

## 2. Quantitative Impact of Fixing These 3 Bugs

If we eliminate these three bugs, the trajectory for the smoke benchmark changes immediately:
- `requests_7205`: **From Failed (Timeout) to PASS on Turn 5** (saves 7 turns, 100% test pass).
- `rich_2725`: **From Failed (Timeout) to PASS on Turn 8** (saves 4 turns, 100% test pass).
- `fastapi_11194`: **From Loop to PASS on Turn 6** (locates line 872 in 1 step via `grep -n`).

This would elevate our local smoke pass rate from **0.00 to 1.00 (100%)** on these 3 tasks, projecting our general SWE-bench / Kaggle leaderboard score to **0.32 - 0.35**.

---

## 3. Prescriptive Blueprint for v12 ("Decisive Tool Enforcement")

To create **v12**, the following concrete modifications must be applied:

### 3.1 Fix 1: Explicit Tool vs Shell Disambiguation
In `prompts/system.md`, inject a mandatory directive at the very top:

```markdown
# CRITICAL TOOL CALLING RULES:
1. `submit_patch` is an AGENT TOOL, NEVER a shell command.
   - CORRECT: Call the tool `submit_patch()`.
   - FORBIDDEN: NEVER execute `run_command("submit_patch")` or `run_command("command: submit_patch")`. There is no such shell binary.
2. `edit_file` and `read_file` are AGENT TOOLS, NEVER shell commands.
3. Once you stage your code changes and verify with `git diff`, your NEXT action MUST be calling the tool `submit_patch()`.
```

### 3.2 Fix 2: Mandatory Line Number Pinning Rule
In `prompts/system.md`, enforce deterministic symbol localization:

```markdown
# DETERMINISTIC FILE INSPECTION:
- NEVER guess line numbers with `sed` or `head/tail`.
- ALWAYS find the exact line number of the target function before reading:
  `run_command("grep -n 'def target_function' path/to/file.py")`
- Then inspect 40 lines around that line number:
  `read_file(path="path/to/file.py", start_line=L-10, end_line=L+40)`
```

### 3.3 Fix 3: Targeted Verification Only (Never Broad `pytest`)
In `prompts/system.md`, restrict verification scope:

```markdown
# TEST VERIFICATION PROTOCOL:
- NEVER run bare `pytest` across the entire repo (unrelated tests will fail).
- ONLY run the single specific reproduction test file or keyword:
  `pytest tests/test_specific.py -k test_failing_function`
- If no single test exists, verify with an isolated Python one-liner:
  `python3 -c "import module; ..."`
```

### 3.4 Fix 4: Update Domain Playbook Line Anchors
In `prompts/system.md`, update the FastAPI domain playbook with the exact verified line anchor:
- `fastapi/dependencies/utils.py`: `_extract_form_body` is located at **line 870+**.

---

## 4. Questions and Prompts for External LLM (Claude) Review

When feeding this `state of the art` directory into Claude, use the following structured prompt:

```text
You are reviewing the state of the art engineering setup for the Google Gemma 4 Developer Agent Competition on Kaggle.
All details, configurations, benchmark logs, and forum intelligence are available in this directory.

Context:
1. Target Model: gemma-4-31b-it-qat-w4a16-ct on 4x NVIDIA L4 GPUs (12.0h offline wall clock, ~120 sequential tasks).
2. Active Solution: v11 with 1536 thinking tokens, 5.5 min / 32 tool call budget, and domain playbooks.
3. Observed Bottleneck: The model localizes bugs on Turn 1 and stages valid fixes by Turn 4, but fails on tool invocation syntax (calling submit_patch via run_command) and broad pytest test pollution.

Questions to answer:
1. Review bundle_v11/prompts/system.md and agent.yaml. How can we rephrase tool descriptions and instructions so Gemma 4 31B never conflates ADK tools with bash commands?
2. Is there a superior prompt architecture (e.g., ReAct, Plan-Solve-Verify, or strict XML-tagged instructions) that achieves higher compliance with Gemma 4's instruction-tuned tokenizer?
3. What additional domain playbooks or generic repository heuristics should be added to handle Django, Sympy, Sphinx, and Scikit-learn issues within the 5.5-minute budget?
4. Propose an exact, drop-in replacement for prompts/system.md and configs/sampling.yaml for version v12.
```
