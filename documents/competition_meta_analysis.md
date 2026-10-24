# Kaggle Gemma 4 Developer Agent Competition: Comprehensive Forum & Notebook Meta-Analysis

> **Synthesized from Discussions across Pages 1–13** (Sorted by Votes)  
> **and Comprehensive Analysis of Top Performing Notebooks** (Ranked by Votes & Declared Public LB Scores)  
> *Including in-depth ablation data from Ayush Thakur, Ruijie Zhu, Dmitrii Gluzdov, flexonafft, jvnhonglei, samvalladares, and Kaggle Staff (Ryan Holbrook, Dustin).*

---

## Executive Summary

Across the 13 discussion pages and the entire spectrum of public notebooks, the competition has converged on concrete design principles that separate 0.13–0.17 baselines from 0.22–0.24+ leaders:

1. **The "Harness Ceiling" Dominates Raw Model Power**:
   - Ruijie Zhu’s benchmark on 124 public tasks showed **DeepSeek V4 Pro** (58/124) and **Qwen3.6-27B** (59/123) only solve 4–5 more tasks than **Gemma 4 31B** (54/124).
   - The ceiling is driven by the evaluation harness: **32k context limits with compaction**, sparse issue texts (e.g., `"Fixes #1234"` with no description), and synchronous execution timeouts—not underlying coding intelligence.
2. **Execution Speed is the Primary Competitive Lever**:
   - Kaggle’s hard **12.0-hour limit** across ~120 sequential tasks imposes an inflexible arithmetic: $\le 360\text{ seconds/task}$ before container overhead.
   - Notebooks that set generous per-task limits (`8 minutes`, `80 turns`) inevitably encounter **Notebook Timeout** because when the agent struggles on 60+ tasks, it consumes the full 8 minutes, breaching the 720-minute notebook cutoff.
3. **Complex Multi-Agent Systems & Graph Tools Harm Performance**:
   - Ayush Thakur’s ablations on 32 B40 GPUs proved that AST subgraph and semantic code search tools dropped task resolution from **42.6% to 38.8%**. Semantic searches consistently returned empty results, distracting the agent.
   - Multi-agent architectures (delegation, planner-analyzer-coder) spend 3–6 tool turns chatting between agents, burning time without producing code edits. Clean single-agent pipelines dominate the leaderboard.
4. **LoRA SFT Discrepancy & vLLM Parser Traps**:
   - Whisper Last and others uncovered that in the competition's patched **vLLM 0.19.1**, LoRA adapters can be silently ignored or wiped under specific memory configs.
   - SFT models trained on verbose conversational trajectories waste turns discussing plans instead of editing immediately.
   - **Critical vLLM Bug**: `edit_file` calls containing `"""` triple-quote docstrings are mangled by vLLM’s Gemma-4 parser, leading to repeated failed retries (up to 27x) that burn the turn budget.

---

## 1. Top Performing Notebooks: Architectural & Score Decomposition

A rigorous inspection of the top-voted and highest-scoring public notebooks reveals how top competitors achieved their declared leaderboard (LB) scores:

| Notebook Title | Primary Author | Votes / Medals | Declared LB Score | Core Innovation & Architecture | Key Failure Controls |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`gemma-4-baseline-lb-0-24`** | `flexonafft` / community | Top Voted | **0.24** | **Targeted Prompt Refinement on Single Agent**: Strict call pacing, parallel sync/async implementation rules, and protected test paths (`pytest.ini`, `conftest.py`). | Declares graph tools in YAML but explicitly forbids their use in `system.md` ("They return nothing useful"). |
| **`gemma-4-budget-aware-coding-agent-v2`** | `flexonafft` | Silver | **0.22** | **Budget-Aware Operating Profile**: Enforces turn-12 edit deadline, bounded independent searches, and `git status --short && git diff` verification. | Guarded unique-match replacement; fallback from `edit_file` failures. |
| **`gemma-4-superagent-reproduction`** | `jvnhonglei` | 57 votes / Silver | **0.20** | **4-Phase Strict Sequential Execution**: Calls 1–8 locate/read, Calls 9–12 edit, Calls 13–20 verify, Call 20+ submit. | Eliminates search loops; cuts off exploration after turn 12. |
| **`gemma-4-agent-budget-fit-single-agent`** | `samvalladares` | 49 votes / Bronze | **0.18** | **Measured Budget Calibration**: First to reduce tool calls from 40–100 down to 28 calls / 8 minutes based on 129 local run timings. | Hard stop at 28 calls to prevent running out of notebook time. |
| **`gemma-4-autonomous-developer-agent`** | `lavinwins` | 42 votes / Bronze | **0.18** | **Calibrated Single Agent**: Clean scratch file containment (strictly in `/tmp`) and exact symbol location. | Prevents untracked dirty files from corrupting the graded patch. |
| **`black-cat-swe-agent-pack-instinct`** | `black-cat` team | 38 votes / Bronze | **0.18** | **Symbol Graph & Ripgrep Hybrids**: Hybrid bounded rg and symbol graph lookup with fallback prompt. | Fallback configuration if graph tools return empty data. |
| **`gemma-4: public-agent-ideas-and-evidence`** | `dmitriigluzdov` | 30 votes / Bronze | **0.18** (31B) / Timeout (12B) | **Catalog of 187 Notebooks & 74 Mechanisms**: Proved prompt and tools stay within noise, while time per task is the binding constraint. | Attempted 12B swap under 8-min cap, demonstrating timeout trap. |
| **`gemma + gemini - the ultimate duo`** | `matterhorn3838` | 40 votes / Bronze | **0.15** | **Hybrid Agent Architecture**: Attempted external Gemini API integration. | External API disabled during evaluation, falling back to local model. |

---

## 2. In-Depth Analysis of Notebook Code & Techniques

### A. The 0.22 $\to$ 0.24 Progression (`flexonafft`)
The jump from 0.22 to 0.24 was achieved **purely through prompt engineering** without changing the model, tools, or sampling config. Three specific prompt additions drove the +0.02 gain:
1. **Parallel Implementation Law**:
   Many Python libraries implement twin architectures (e.g. `Client` vs `AsyncClient` in `httpx`, sync route handlers vs `async def` in `fastapi`, and dual export in `__init__.py` and `__all__`). The 0.24 prompt explicitly instructs:
   > *"Apply a change to every parallel implementation (sync/async, Client/AsyncClient, backends, overloads, `__init__`/`__all__` re-exports)."*
2. **Protecting Test Infrastructure**:
   Agents often modified `pytest.ini` or `conftest.py` to bypass failing tests. The 0.24 prompt made this an explicit instant-failure rule:
   > *"`pytest.ini` / `conftest.py` added to protected files; scratch files must go to `/tmp`. Any file left under `/workspace` becomes part of the graded patch."*
3. **The Untracked File Trap (`git status --short`)**:
   `git diff` alone completely hides newly created scratch files or untracked test scripts. However, `submit_patch` uses `git diff HEAD`, including untracked files in the patch! Enforcing `git status --short && git diff` prevents invalid scratch files from contaminating the final patch.

### B. The Deceptive Tool Trap (`agent.yaml` vs `system.md`)
In `gemma-4-baseline-lb-0-24.ipynb`, the YAML declares:
```yaml
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
```
However, in `prompts/system.md`, the author explicitly wrote:
> **"- Never use `search_similar_code`, `get_code_neighbors` or `get_code_subgraph`. They return nothing useful."**

Competitors who removed these graph tools or instructed the model to ignore them consistently scored higher than those who attempted to utilize them.

---

## 3. Analysis of Competitor Comments & Failure Modes

Examining discussions and comments across the top notebooks highlights 4 recurring failure modes reported by participants:

### 1. The 12-Hour "Notebook Timeout" Epidemic
- **Observation**: Dozens of participants commented that their runs died at the 12-hour mark with `Notebook Timeout`.
- **Root Cause**: Many participants copied the `8 minutes / 28 tool calls` configuration from `budget-fit` or `superagent`. While 8 minutes works if only a few tasks fail, across the hidden test set of ~120 tasks:
  $$120 \text{ tasks} \times 8 \text{ min} = 960 \text{ minutes (16 hours)} \gg 720 \text{ minutes (12 hours)}$$
- **Resolution**: A strict per-task cap of **4.5 minutes (270s)** and **16–18 tool calls** guarantees worst-case runtime under 9.5 hours, fully immunizing the notebook against timeouts.

### 2. The Zero-Patch Scourge (`patch_size == 0`)
- **Observation**: Scores plummeting from 0.20+ down to 0.10–0.13.
- **Root Cause**: When `edit_file` fails due to indentation or whitespace mismatches, naive agents get stuck in search loops and run out of turns without editing anything. An unedited repository scores exactly 0.0.
- **Resolution**: Enforce a mandatory Python heredoc fallback via `run_command` if `edit_file` fails once.

### 3. Context Compaction & Repetitive Tool Stalls
- **Observation**: In later turns (turns 18+), Gemma 4 starts repeating identical `git grep` or `pytest` commands.
- **Root Cause**: Running full test suites dumps hundreds of lines into the context, triggering ADK's 32k context compaction. Compaction strips earlier tool call history, causing the model to forget that it already ran the search.
- **Resolution**: Strictly limit test output to `tail -40`, name one test file with `-k`, and enforce a hard exploration cutoff by turn 12.

### 4. Pre-Existing Environment Failures
- **Observation**: Agents wasting 10+ turns trying to fix `ModuleNotFoundError`, missing optional fixtures (e.g. redis, uvicorn), or network socket errors.
- **Root Cause**: Real-world open-source repositories often have optional dependencies that are not installed in the competition sandbox.
- **Resolution**: Explicit prompt directive: *"Failures that come from missing fixtures, missing optional dependencies, network errors, or import errors unrelated to your change are pre-existing. Never attempt to repair tests or the environment."*

---

## 4. Quantitative Benchmark & Ablation Tables

### A. Sampling & Thinking Ablation (Ayush Thakur, CoreWeave 32× B40 Cluster)
*Evaluated on 129 scorable SWE-bench tasks with `gemma-4-31b-it-qat-w4a16-ct`.*

| Configuration | Variation | Resolved / 129 | Resolution Rate | Observations |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline Reference** | Temp 1.0, Top-p 0.95, Budget 4096 | **51 / 129** | **39.53%** | Reference standard |
| **Temperature** | Temp 0.4 | **51 / 129** | **39.53%** | Ties Temp 1.0; more deterministic tool syntax |
| | Temp 0.0 (Greedy) | 47 / 129 | 36.43% | Vulnerable to repetitive search loops |
| **Thinking Toggle** | **Thinking Disabled** | **23 / 129** | **17.83%** | **Catastrophic -55% collapse**. Thinking is mandatory. |
| **Thought Summaries** | Summaries Disabled | 35 / 129 | 27.13% | Model loses planning cohesion across multi-turn context |
| **Thinking Budget** | Budget 2,048 tokens | 50 / 129 | 38.76% | **Optimal speed/quality trade-off** (-40% decoding latency) |
| | Budget 6,144 tokens | 48 / 129 | 37.21% | Slight regression; model over-thinks simple syntax fixes |

---

### B. Teacher Comparison: Is Gemma 4 31B the Limiting Factor? (Ruijie Zhu)
*Evaluated on 124 public tasks using swegemma 0.2.7 (32k context, events compaction, 25 min cap).*

| Model | Solved / 124 | Median Minutes / Task | Pass-to-Pass Variance |
| :--- | :---: | :---: | :---: |
| **Gemma 4 31B (Run 1)** | 54 / 124 | 5.2 min | Baseline |
| **Gemma 4 31B (Run 2)** | 53 / 123 | 5.2 min | 15% task flip between runs (high noise) |
| **DeepSeek V4 Pro** | 58 / 124 | 3.5 min | +3.2% resolution, 33% faster |
| **Qwen3.6-27B** | 59 / 123 | 3.9 min | +4.0% resolution, 25% faster |

---

## 5. Synthesis: How Our Workspace Implements the Winning Formula

Our newly built **`v7 (31B Pro)`** bundle (`submission.zip` / `submission_v7_gemma4_31b_pro.zip`) synthesizes every successful mechanism from the top-scoring notebooks while eliminating their vulnerabilities:

| Mechanism / Feature | Public Leaderboard Precedent | Our `v7 (31B Pro)` Implementation |
| :--- | :--- | :--- |
| **Target LB Score** | 0.22 (v2) $\to$ 0.24 (v3) | **Targeting 0.22 – 0.25+** |
| **Sampling Config** | Ayush Thakur 129-task sweep | `temperature: 0.4`, `thinking_budget: 2048`, `max_output_tokens: 4096` |
| **Per-Task Time Limit** | 8 min (vulnerable to 12h timeout) | **4.5 minutes hard cap** (100% immune to 12h notebook timeout) |
| **Turn Budget** | 28 calls / 80 turns | **18 tool calls / 22 turns** (prevents late-turn compaction thrashing) |
| **vLLM Parser Protection** | Discovered by Ruijie Zhu | Explicit guardrail forbidding `"""` / `'''` inside `edit_file` arguments |
| **Edit Fallback** | flexonafft / Gluzdov | Guaranteed one-line Python replacement script via `run_command` |
| **Parallel Implementations** | flexonafft 0.24 baseline | Sync/async, client/asyncclient, and `__all__` re-export coverage |
| **Domain Playbooks** | Custom workspace addition | Pre-compiled idioms for **FastAPI**, **Rich**, **Requests**, **Scikit-learn**, **SymPy**, **Sphinx** |
| **Toolset Minimalist** | Gluzdov / jvnhonglei | Core 6 ADK tools only; zero noise from disabled graph tools |
