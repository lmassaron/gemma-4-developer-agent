# Competition Research History: Discussions & Notebooks Catalog

> **Repository Reference**: Log of all Kaggle discussion threads and public notebooks indexed, analyzed, and synthesized for the Google Gemma 4 Developer Agent Competition.  
> **Last Updated**: 2026-10-24  
> **Source Competition**: [Google - The Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)

---

## 1. Consulted Discussion Threads (Pages 1–13, Sorted by Votes)

| # | Topic Title | Author | URL | Votes / Comments | Core Takeaway / Finding |
|---|:---|:---|:---|:---:|:---|
| 1 | **Clarification on distillation from external LLMs** | `c-number` | [Discussion 742807](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807) | 49 votes / 12 comments | Official rules clarification regarding external LLM synthetic data generation and distillation constraints. |
| 2 | **Update on "Kaggle Error" Submissions & Queue Delays** | `Dustin` (Kaggle Staff) | [Discussion 743242](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743242) | 36 votes / 91 comments | Evaluation queue bottlenecks on 4× L4 GPUs, container spin-up overheads, and automatic rerun protocols. |
| 3 | **New Gemma 4 Models Available: 12B, E4B, and E2B** | `Ryan Holbrook` (Kaggle Staff) | [Discussion 744474](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744474) | 35 votes / 35 comments | Release of quantized smaller Gemma 4 model checkpoints (`gemma-4-12b-it-qat-w4a16-ct`, `e4b`, `e2b`) for competition use. |
| 4 | **Gemma 4 agent experiments: sampling, thinking, time limits, and tool ablations** | `Ayush Thakur` | [Discussion 746250](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746250) | 34 votes / 11 comments | 129-task ablation study on 32× B40 GPUs: $T=0.4/1.0$ ties at 39.53%; thinking budget 2048 is optimal; disabling thinking collapses score to 17.8%; AST graph tools degrade score to 38.8%. |
| 5 | **⭐Bonus Prize: Small Model Development ($10,000)** | `Elan Markowitz` (Google) | [Discussion 744477](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744477) | 32 votes / 18 comments | Announcement of $10,000 bonus track rewarding the highest-performing agent built on non-31B models (12B, E4B, E2B). |
| 6 | **Suggestion: Temporarily Pause Gemma 4 for 24 Days to Address the L4x4 Shortage** | `Tony Li` | [Discussion 747915](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747915) | 29 votes / 45 comments | Analysis of community GPU capacity constraints and submission queue congestion. |
| 7 | **Stronger models barely beat Gemma 4 31B in this harness — teacher comparison on 124 public tasks** | `Ruijie Zhu` | [Discussion 747168](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747168) | 25 votes / 4 comments | Teacher model benchmark (DeepSeek V4 Pro: 58/124, Qwen3.6-27B: 59/123, Gemma 4 31B: 54/124). Identified vLLM triple-quote docstring parser corruption bug and synchronous tool bottlenecks. |
| 8 | **Rerunning Failed Submissions** | `Ryan Holbrook` (Kaggle Staff) | [Discussion 743683](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743683) | 21 votes / 102 comments | Official handling of "Kaggle Error" states versus genuine participant "Submission Errors". |
| 9 | **submission is running for 15h. is this due to evaluation ?** | `RISHAV KUMAR` | [Discussion 746480](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746480) | 21 votes / 37 comments | Analysis of long queue times vs active execution; establishes 12.0-hour hard limit for active container execution. |
| 10 | **Submission errors after the Sep 30 wheelhouse update** | `Asim Ersoy` | [Discussion 744018](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744018) | 21 votes / 22 comments | Dependency breaking changes and wheel compatibility in the offline evaluation sandbox environment. |
| 11 | **LoRA adapters are silently wiped in the patched vLLM 0.19.1** | `Whisper Last` | [Discussion 747512](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747512) | 15 votes / 8 comments | Root-cause analysis of SFT underperformance: vLLM 0.19.1 memory allocator silently dropping LoRA adapter weights under high concurrency. |
| 12 | **Submission 57361964 got “Submission Scoring Error” — is it covered by the queue recovery?** | `Satish` | [Discussion 748102](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748102) | 8 votes / 14 comments | Clarification on patch syntax errors (`git apply` rejects) causing scoring errors vs platform timeouts. |
| 13 | **Evaluation Harness Timeouts and Sandbox Limits** | Community | Forum Pages 8–13 | Various | Mathematical proof that 8-minute/task caps across 120 hidden tasks exceed 12 hours ($120 \times 8 = 960\text{m} > 720\text{m}$), causing notebook timeout. |

---

## 2. Consulted Public Notebooks (Sorted by Votes & Declared Score)

| # | Notebook Name | Author | URL | Medals / Votes | Declared LB | Architectural Highlights & Code Mechanisms |
|---|:---|:---|:---|:---:|:---:|:---|
| 1 | **`gemma-4-baseline-lb-0-24`** | `flexonafft` / community | [Link](https://www.kaggle.com/code/flexonafft/gemma-4-baseline-lb-0-24) | Top Voted | **0.24** | Added parallel sync/async handling, client/asyncclient parity, and `git status --short && git diff`. Declares graph tools in YAML but forbids their use in prompt. |
| 2 | **`gemma-4-budget-aware-coding-agent-v2`** | `flexonafft` | [Link](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2) | Silver / 45+ | **0.22** | First budget-paced agent with turn-12 edit deadline, bounded searches, and guarded unique-match replacement. |
| 3 | **`gemma-4-superagent-reproduction`** | `jvnhonglei` | [Link](https://www.kaggle.com/code/jvnhonglei/gemma-4-superagent-reproduction) | Silver / 57 | **0.20** | Strict 4-phase sequential workflow (locate $\to$ edit $\to$ verify $\to$ submit). Hard cutoff on exploration at turn 12. |
| 4 | **`gemma-4-superagent`** | `matterhorn3838` | [Link](https://www.kaggle.com/code/matterhorn3838/gemma-4-superagent) | Bronze / 40+ | **0.18** | Foundational single-agent architecture with bounded turn pacing that served as the base for `flexonafft`. |
| 5 | **`gemma-4-agent-budget-fit-single-agent`** | `samvalladares` | [Link](https://www.kaggle.com/code/samvalladares/gemma-4-agent-budget-fit-single-agent) | Bronze / 49 | **0.18** | Calibrated per-task tool calls from 100 down to 28 based on 129 local test runs; demonstrated that exploring past 15 calls has near-zero success. |
| 6 | **`gemma-4-autonomous-developer-agent-calibrated-single-agent`** | `lavinwins` | [Link](https://www.kaggle.com/code/lavinwins/gemma-4-autonomous-developer-agent-calibrated-single-agent) | Bronze / 42 | **0.18** | Strict scratch file hygiene (`/tmp` only) to prevent dirty untracked files from polluting the evaluation patch. |
| 7 | **`black-cat-swe-agent-pack-instinct`** | `black-cat` team | [Link](https://www.kaggle.com/code/blackcat/black-cat-swe-agent-pack-instinct) | Bronze / 38 | **0.18** | Ripgrep and symbol graph hybrid with dual fallback configuration (`PRIMARY_FILES` and `FALLBACK_FILES`). |
| 8 | **`gemma-4: public-agent-ideas-and-evidence`** | `dmitriigluzdov` | [Link](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-public-agent-ideas-and-evidence) | Bronze / 30 | **0.18** | Empirical taxonomy of 187 notebooks and 74 mechanisms across 9 families; local screen of 31 tasks across 4 fresh repositories. Tested 12B model swap under 8-min cap. |
| 9 | **`gemma + gemini - the ultimate duo`** | `matterhorn3838` | [Link](https://www.kaggle.com/code/matterhorn3838/gemma-gemini-the-ultimate-duo) | Bronze / 40 | **0.15** | Hybrid orchestration attempting Gemini API calls; highlighted offline sandbox networking restrictions during scoring. |
| 10 | **`gemma-eda-baseline-for-a-start-lb-top-1`** | `tatamikenn` | [Link](https://www.kaggle.com/code/tatamikenn/gemma-eda-baseline-for-a-start-lb-top-1) | Bronze / 35 | **0.13** | Comprehensive exploratory data analysis of the SWE-bench task distribution, repo frequencies, and issue text lengths. |
| 11 | **`gemma-4-task-soundness-serving-and-evaluation`** | `dmitriigluzdov` | [Link](https://www.kaggle.com/code/dmitriigluzdov/gemma-4-task-soundness-serving-and-evaluation) | Bronze / 25 | Dev Tools | Complete local Docker serving and evaluation harness mimicking the official Kaggle test container. |
| 12 | **`📘Gemma-4 | The Complete Guide. Expanded`** | `leoprovorov` | [Link](https://www.kaggle.com/code/leoprovorov/gemma-4-the-complete-guide-expanded) | Bronze / 32 | Baseline | End-to-end walkthrough of Google ADK agent assembly, YAML schema validation, and deterministic zip generation. |
| 13 | **`Gemma 4 Patch Berserker: Evaluate, Then Submit`** | `hitarthjain0` | [Link](https://www.kaggle.com/code/hitarthjain0/gemma-4-patch-berserker-evaluate-then-submit) | Bronze / 29 | Baseline | Strict pre-submission evaluation rules to eliminate zero-patch (`patch_size == 0`) submissions. |

---

## 3. Consultation Coverage & Checkpoints

- **Discussion Forum Coverage**: Complete through **Page 13** (Sorted by Votes, capturing all topics down to $\le 0$ net votes).
- **Notebooks Catalog Coverage**: Complete through all top-voted notebooks and the full **187-notebook snapshot** analyzed by Dmitrii Gluzdov.
- **Key Knowledge Artifacts in Repository**:
  - [`documents/competition_meta_analysis.md`](file:///Users/lucamassaron/code/gemma-4-developer-agent-competition/documents/competition_meta_analysis.md): Detailed analytical synthesis of the findings, ablations, and architectural recommendations.
  - [`documents/dmitrii_gluzdov_meta_analysis.md`](file:///Users/lucamassaron/code/gemma-4-developer-agent-competition/documents/dmitrii_gluzdov_meta_analysis.md): Archived raw data of the 187-notebook mechanism catalogue.
  - [`documents/history.md`](file:///Users/lucamassaron/code/gemma-4-developer-agent-competition/documents/history.md): This master index of all titles, authors, and URLs.
