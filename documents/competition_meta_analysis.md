# Kaggle Gemma 4 Developer Agent Competition: Comprehensive Forum Meta-Analysis

> **Synthesized from Discussions across Pages 1–13** (Sorted by Votes)  
> *Including in-depth ablation data from Ayush Thakur, Ruijie Zhu, Dmitrii Gluzdov, Whisper Last, and the Kaggle Organizer updates (Ryan Holbrook, Dustin).*

---

## Executive Summary

Across the 13 pages of discussion and ~200 public notebook experiments, the community has converged on several critical truths that overturn initial assumptions:

1. **The "Harness Ceiling" Dominates Raw Model Power**:
   - Ruijie Zhu’s benchmark on 124 public tasks showed **DeepSeek V4 Pro** (58/124) and **Qwen3.6-27B** (59/123) only solve 4–5 more tasks than **Gemma 4 31B** (54/124).
   - The ceiling is driven by the evaluation harness: **32k context limits with compaction**, sparse issue texts (e.g., `"Fixes #1234"` with no description), and synchronous execution timeouts—not underlying coding intelligence.
2. **Execution Speed is the Primary Competitive Lever**:
   - Kaggle’s hard **12.0-hour limit** across ~120 sequential tasks imposes an inflexible arithmetic: $\le 360\text{ seconds/task}$ before container overhead.
   - Teacher models resolved tasks in 3.5–3.9 minutes median; Gemma 4 31B averaged 5.2 minutes. When an agent exceeds 5 minutes on failed tasks, it starves subsequent solvable tasks, resulting in **Notebook Timeouts** (or severely truncated evaluations).
3. **Complex Multi-Agent Systems & Graph Tools Harm Performance**:
   - Ayush Thakur’s ablations on 32 B40 GPUs proved that AST subgraph and semantic code search tools dropped task resolution from **42.6% to 38.8%**. Semantic searches consistently returned empty results, distracting the agent.
   - Multi-agent architectures (delegation, planner-analyzer-coder) spend 3–6 tool turns chatting between agents, burning time without producing code edits. Clean single-agent pipelines dominate the leaderboard.
4. **LoRA SFT Discrepancy & vLLM Parser Traps**:
   - Whisper Last and others uncovered that in the competition's patched **vLLM 0.19.1**, LoRA adapters can be silently ignored or wiped under specific memory configs.
   - In addition, SFT models trained on verbose conversational trajectories waste turns discussing plans instead of editing immediately.
   - **Critical vLLM Bug**: `edit_file` calls containing `"""` triple-quote docstrings are mangled by vLLM’s Gemma-4 parser, leading to repeated failed retries (up to 27x) that burn the turn budget.

---

## 1. Quantitative Benchmark & Ablation Tables

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

> **Key Takeaway**: Frontiers models only gain 4–5 tasks over Gemma 4 31B. Therefore, **prompt architecture, time management, and tool reliability matter more than model scale**.

---

### C. Active Time Limits per Task (Ayush Thakur)

| Per-Task Cap | Resolved / 129 | Timeouts | Task-Hours Consumed |
| :---: | :---: | :---: | :---: |
| **6 minutes** | 37 | 61 | Low (High failure rate on multi-file repos) |
| **10 minutes** | 46 | 26 | Moderate |
| **15 minutes** | **54 (41.86%)** | **3** | **Optimal local quality setting** |
| **20 minutes** | 49 | 2 | Diminishing returns |
| **60 minutes** | 51 | 0 | +10.2% token waste, 0 extra tasks |

*Note for Kaggle: While 15 minutes is optimal when evaluating single tasks in isolation, on Kaggle's sequential 120-task run under a 12-hour limit, the maximum sustainable average is 4.5–5.0 minutes.*

---

## 2. Deep Dive: The 5 Major Competitor Strategies

### 1. The Gluzdov Single-Agent "Budget-Fit" Paradigm
- **Public Baseline**: Developed by flexonafft and analyzed across 187 notebooks by Dmitrii Gluzdov.
- **Philosophy**: Avoid multi-agent handoffs. Use a single prompt that establishes rigid "Turn Budgets":
  - Turns 1–4: Search and locate (strictly limited to 2 grep/find calls).
  - Turns 5–8: Edit deadline. A dirty working tree is required by turn 8.
  - Turns 9–12: Single targeted test execution.
  - Turn 13+: Diff check and mandatory submission.
- **Impact**: Consistently achieves **0.22 – 0.24** on the public leaderboard.

### 2. The Tool-Set Minimalist Movement
- Early public notebooks added specialized tools: `ast_grep`, `python_repl`, `code_graph_search`, `reproduce_issue`.
- **Competitor Finding**: Every extra tool increases the action space and model confusion.
  - In Ayush Thakur's experiments, graph search tools resulted in **-3.8% lower resolution**.
  - Replacing specialized file search tools with plain `run_command` (`git grep`, `git ls-files`, `sed`) yielded higher tool call validity because Gemma 4's pre-training on bash and git is vastly stronger than custom JSON tool schemas.

### 3. The Reproducer Script Controversy
- Early high-scoring notebooks created `/tmp/reproduce_issue.py` before editing.
- **The Pitfall**: Multiple competitors reported that when issues lack an explicit snippet, the model spends 3–5 turns writing synthetic tests that fail due to missing mock fixtures.
- **The Modern Consensus**: Eliminate the standalone reproduction script. Instead, execute one existing unit test via `pytest tests/... -k keyword` or `python -c "..."` directly.

### 4. The vLLM Docstring Bug Workaround
- **Bug**: In vLLM’s Gemma 4 tool-calling parser, when `edit_file` receives an `old_string` containing `"""` docstrings, the parser escapes quotes improperly, emitting `<|"|>,filepath` or corrupted JSON arguments.
- **Result**: ADK throws an error: `mandatory input parameters are not present`. Gemma 4 repeatedly attempts the exact same edit up to 27 times until exhausting the turn limit.
- **Competitor Workaround**:
  - Add an explicit rule in `system.md`: *"When modifying docstrings or large multiline blocks, use `run_command` with an inline Python heredoc script (`python3 -c '...'`) rather than `edit_file`."*

### 5. Small Model Feasibility (12B, E4B, E2B) & The $10,000 Bonus Prize
- Google/Kaggle introduced a **$10,000 bonus prize** specifically for smaller models (12B, E4B, E2B).
- **Competitor Observations**:
  - 12B generates tokens ~2.5x faster than 31B, eliminating the risk of Kaggle 12-hour timeouts.
  - However, 12B struggles with complex multi-file codebase navigation and subtle type inference.
  - **Optimal 12B Niche**: Tasks with self-contained stack traces (e.g. single-file bugs in `requests` or `fastapi`).

---

## 3. Synthesis: Key Recommendations for Our Project

Based on the meta-analysis of all 13 discussion pages and competitor ablations:

| Area | Current Workspace Status | Competitor Evidence & Next Best Action |
| :--- | :--- | :--- |
| **Model Selection** | We have both 31B and 12B bundles ready | **31B remains the primary choice for top-tier leaderboard score** (0.22–0.24). **12B is ideal for the $10,000 Small Model Bonus Prize**. |
| **Sampling Config** | `temp: 0.2`, `top_p: 0.95`, `budget: 1024` | Ayush Thakur proved `budget: 2048` is optimal (preserving 98% of 4096 quality while cutting latency in half). We can safely use 1024–2048. |
| **Time Budgets** | Hardened to `4.5 min` / `16 tool calls` | Exactly aligns with Ruijie Zhu & Gluzdov’s 360s arithmetic, eliminating the 12-hour notebook timeout. |
| **Tool Calling** | Core 6 ADK tools | Keep tools minimal (`run_command`, `read_file`, `edit_file`, `write_file`, `get_status`, `submit_patch`). Do NOT add graph or complex retrieval tools. |
| **Docstring Protection** | Standard `edit_file` instructions | Include explicit heredoc fallback in `system.md` to protect against the vLLM `"""` parser bug. |
