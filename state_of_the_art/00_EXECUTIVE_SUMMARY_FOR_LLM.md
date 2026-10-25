# Executive Briefing: Kaggle Gemma 4 Developer Agent Competition

> **Purpose**: This document provides an exhaustive, end-to-end briefing on the Google Gemma 4 Developer Agent Competition on Kaggle, the architecture and evolution of our autonomous SWE-bench coding agents, empirical benchmark findings on an NVIDIA DGX Spark system (Grace Blackwell GB10), and the exact engineering challenges requiring LLM analysis.

---

## 1. Competition Overview & Constraints

* **Competition**: [Google - The Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)
* **Goal**: Build an autonomous AI software engineering agent powered by `gemma-4-31b-it-qat-w4a16-ct` (or smaller Gemma 4 variants) that resolves real-world Python GitHub issues across a hidden evaluation suite of **~120 tasks** (58 public, ~62 private).
* **The Global 12-Hour Constraint**: 
  Submissions execute sequentially on Kaggle using a 4× NVIDIA L4 GPU instance (or host evaluation container). 
  The global notebook runtime limit is **exactly 12.0 hours (720 minutes)**.
  $$\frac{720\text{ minutes}}{120\text{ tasks}} = 6.0\text{ minutes/task maximum (including container overhead)}$$
  Any submission with an uncalibrated budget (e.g., standard 8–10 min caps) encounters **Notebook Timeout** after task ~75, resulting in an automatic **0.00**.
* **Submission Format**: A deterministic `submission.zip` archive containing declarative YAML and markdown configuration interpreted by Google's Agent Development Kit (ADK / `adk-submission`):
  ```text
  submission.zip
  ├── agent.yaml                   # Model selection, instruction include, tool list
  ├── eval_config.yaml             # Per-task execution budget & turn limits
  ├── configs/
  │   └── sampling.yaml            # Temperature, top-p, thinking budget
  └── prompts/
      └── system.md                # System prompt guiding the agent's multi-turn actions
  ```

---

## 2. Historical Score Progression

| Version | Configuration | Public LB Score | Status | Key Characteristics |
| :--- | :--- | :---: | :---: | :--- |
| **Official Starter** | 1-min timeout, naive prompt | **0.00** | Archived | Instant timeout on every task. |
| **v1 (Initial)** | Standard budget, naive search | **0.13** | Archived | Exploratory drift; model reads past turn 12 and times out. |
| **v3 (Call-Paced)** | Turn-12 deadline, sync/async rules | **0.17** | Baseline | Proven stable baseline, 28 tool calls, 8m per-task cap. |
| **Public SOTA** | `flexonafft / honghanhhh` (0.24) | **0.24** | Public Best | Byte-identical harness to 0.20; 100% prompt-driven gains (parallel implementations, test protection, `git status --short`). |
| **v4 (SFT 31B)** | LoRA adapter ($r=16, \alpha=32$) | **0.15** | Evaluated | Fine-tuned adapter lost to base model due to conversational drift and vLLM KV cache collapse. |
| **v9 (Loop-Breaker)** | Anti-repetition circuit breaker | Local Benchmark | Evaluated | Cut `requests_7205` time by 3.2x (84s); eliminated repeated grep loops. |
| **v10 (Lean Thinking)** | Thinking budget 1536 tokens | Local Benchmark | Evaluated | **Fastest average runtime recorded (269s/task)**; halved per-turn inference latency. |
| **v11 (Targeted Playbooks)** | Domain playbooks & inline edit | Local Benchmark | Archived | **Turn-1 localization** on FastAPI and Rich; verified fix applied by Turn 7 on Rich. |
| **v12 (Native Discipline)** | Native tool switching & prefix fix | Local Benchmark | Evaluated | **All-time fastest (175s/task)**; broke `run_command` collapse, but exited on Turn 5 before edit. |
| **v13 (Non-Empty Patch)** | Strict non-empty patch prerequisite | Active Target | **Active Flagship** | Enforces `patch_size > 0` before `submit_patch`; mandates Calls 3–6 edit phase. |

---

## 3. The Core Dilemma: Base Model vs. Fine-Tuning

A major revelation across Kaggle forum audits (258 topics) and our empirical SFT runs (`train_31b.log`):
1. **LoRA Fine-Tuning Pitfalls**:
   * **KV Cache Collapse**: Loading LoRA adapters on Kaggle's 4× L4 GPUs reduces KV cache capacity from ~32k tokens down to **~7.6k tokens**. Multi-turn tasks overflow and hang the inference server.
   * **vLLM Double-Layer Bug**: In the competition's patched vLLM 0.19.1, Gemma 4 decoder layers are registered twice, causing LoRA weights to be silently wiped or corrupted.
   * **Behavioral Drift**: Conversational SFT data causes the agent to output polite conversational text ("I have analyzed the problem, would you like me to proceed?") rather than calling tools deterministically.
2. **Prompt-Only Dominance**:
   * Zero-shot `gemma-4-31b-it-qat-w4a16-ct` with disciplined prompt engineering and turn budgeting achieves **0.24 LB**, outperforming all public fine-tuned attempts.

---

## 4. Key Engineering Innovations in v10 & v11

1. **Lean Thinking Budget (`thinking_budget: 1536`)**:
   Standard setups allocate 4096 thinking tokens. On a 31B model, decoding 4096 thought tokens takes 35–45s per turn. Dropping this to 1536 tokens reduced per-turn latency by ~50% without degrading reasoning precision.
2. **Strict Time Banking (`max_time_minutes: 5.5`, `max_tool_calls: 32`)**:
   Guarantees worst-case 120-task execution finishes in $\le 11.0$ hours, providing a 1.0-hour safety buffer.
3. **The 5-Turn Edit Mandate & Call-8 Emergency Patch**:
   Empirical research showed agents reading past Turn 12 have a $\sim 0\%$ solve rate. v10/v11 enforce that the agent must edit by Turn 5 (hard emergency stop at Turn 8). An imperfect edit scores infinitely better than an unedited working tree (`patch_size == 0`).
4. **Primary Inline Python Replacement**:
   Bypasses `edit_file` indentation mismatches and vLLM triple-quote docstring mangling via one-line scripts:
   ```bash
   python3 -c "p='path'; c=open(p).read(); assert 'needle' in c; open(p,'w').write(c.replace('needle','new',1))"
   ```
5. **Targeted Repository Playbooks**:
   Maps common SWE-bench repositories (`psf/requests`, `fastapi/fastapi`, `Textualize/rich`, `scikit-learn`, `sympy`, `sphinx`) to their exact functional files and failure patterns.

---

## 5. What the LLM Needs to Solve

When analyzing this directory, focus on resolving these remaining bottlenecks:
1. **The Tool-Calling vs Shell-Command Distinction**:
   In the v11 benchmark, the model fixed `rich_2725` by Turn 7 and `requests_7205` by Turn 4, but attempted to submit via `run_command: "submit_patch"` instead of emitting the native tool call `submit_patch()`. How can the prompt enforce strictly correct tool selection at the submission stage?
2. **FastAPI Line-Range Localization**:
   v11 immediately targeted `_extract_form_body` in `fastapi/dependencies/utils.py` on Turn 1, but inspected lines 100–250 instead of line 872. How can prompt guidance steer the model to inspect the exact line returned by grep?
3. **Pushing Resolution from 0.24 to 0.30+**:
   What additional behavioral constraints, prompt micro-heuristics, or sampling calibrations will turn candidate fixes into verified passes across the 120-task evaluation suite?
