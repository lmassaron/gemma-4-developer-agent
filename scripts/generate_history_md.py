#!/usr/bin/env python3
"""
Generate documents/history.md from documents/all_discussions_scraped.json.
Avoids shell expansion issues by executing from a dedicated script.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_JSON = ROOT_DIR / "documents" / "all_discussions_scraped.json"
OUTPUT_MD = ROOT_DIR / "documents" / "history.md"

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        topics = json.load(f)

    lines = []
    lines.append("# Competition Research History: Master Catalog of Consulted Discussions & Notebooks\n")
    lines.append("> **Comprehensive Reference**: Master catalog indexing **all 258 Kaggle discussion threads** across **Pages 1 through 13** (sorted by votes), along with top public notebooks, code architectures, and declared leaderboard (LB) scores.")
    lines.append(f"> **Last Updated**: 2026-10-24 | **Total Topics Indexed**: {len(topics)} | **Source**: [Kaggle Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)\n")
    lines.append("---\n")
    lines.append("## 1. Master Table of All 258 Consulted Discussion Threads (Pages 1–13)\n")
    lines.append("| # | Title | Author | Votes | Comments | Page | Topic URL |")
    lines.append("|---|:---|:---|:---:|:---:|:---:|:---|")

    for i, t in enumerate(topics, 1):
        pin = "📌 " if t.get("pinned") else ""
        raw_title = t.get("title", "").strip()
        raw_author = t.get("author", "").strip()

        # Fix icon shift where avatar icon became title
        if raw_title in ("emoji_people", "push_pin"):
            title = raw_author
            author = "Community"
        else:
            title = raw_title
            author = raw_author if raw_author else "Community"

        title = title.replace("|", "-").replace("\n", " ").strip()
        author = author.replace("|", "-").strip()
        votes = t.get("votes", 0)
        comments = t.get("comments", 0)
        page = t.get("page", 1)
        url = t.get("url", "")
        topic_id = t.get("id", "")

        lines.append(f"| {i} | {pin}{title} | `{author}` | {votes} | {comments} | p.{page} | [#{topic_id}]({url}) |")

    lines.append("\n---\n")
    lines.append("## 2. In-Depth Technical Insights Across All 258 Topics\n")
    lines.append("""### Key Architectural Lessons from Community Discussions:
1. **Public vs. Private Task Split (Topic #16)**: Confirmed that the public leaderboard consists of exactly **58 tasks** (private split is ~62 tasks, total ~120 tasks).
2. **LoRA KV Cache Shrinkage (Topic #57)**: Loading LoRA adapters in the competition container shrinks the available 4× L4 KV cache from ~32k down to **~7.6k tokens**. Any context longer than 7.6k causes the inference server to hang until timeout.
3. **vLLM Adapter Silent Drop (Topic #39)**: In patched vLLM 0.19.1, Gemma4 decoder layers are registered twice, causing LoRA adapter weights to be silently wiped or not applied under high concurrency.
4. **Double-JSON Backslash Explosion (Topic #26)**: Tool outputs with quotes reach Gemma 4 double-JSON encoded, causing strings to appear as walls of backslashes.
5. **The search_similar_code Context Bomb (Topic #114 & #25)**: A single call to `search_similar_code` dumps near-collapsed embeddings and thousands of tokens into the context, triggering premature context compaction and discarding valid edits.
6. **Thinking Budget Semantics (Topic #146)**: In wheelhouse v29/v30, `include_thoughts: false` does NOT disable thinking; only `thinking_budget: 0` does.
7. **FastAPI & Requests Pre-Existing Failures (Topics #18, #30, #78, #171)**: Reference patches fail in offline environments due to missing optional test packages (`python-multipart`, legacy `httpbin` fixture incompatibility with pytest 8). Agents that attempt to fix failing pre-existing tests fail the task.
8. **Compaction Patch Discard (Topic #71 & #254)**: Context overflow triggers aggressive events compaction, which frequently strips the tool-call history and causes the agent to discard an already-correct patch.
9. **vLLM Tool-Argument Parser Infinite Loop (Topic #221)**: vLLM's Gemma 4 tool-argument parser can loop forever on some malformed array arguments or triple-quote docstrings.
""")

    lines.append("\n---\n")
    lines.append("## 3. Consulted Public Notebooks (Ranked by Score & Community Impact)\n")
    lines.append("""| # | Notebook Name | Author | Link | Medals / Votes | Declared LB | Architectural Highlights & Code Mechanisms |
|---|:---|:---|:---|:---:|:---:|:---|
| 1 | **`gemma-4-baseline-lb-0-24`** | `flexonafft` / community | [Link](https://www.kaggle.com/code/flexonafft/gemma-4-baseline-lb-0-24) | Top Voted | **0.24** | Added parallel sync/async handling, client/asyncclient parity, and `git status --short && git diff`. Declares graph tools in YAML but forbids their use in prompt. |
| 2 | **`gemma-4-budget-aware-coding-agent-v2`** | `flexonafft` | [Link](https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2) | Silver / 45+ | **0.22** | First budget-paced agent with turn-12 edit deadline, bounded searches, and guarded unique-match replacement. |
| 3 | **`gemma-4-superagent-reproduction`** | `jvnhonglei` | [Link](https://www.kaggle.com/code/jvnhonglei/gemma-4-superagent-reproduction) | Silver / 57 | **0.20** | Strict 4-phase sequential workflow (locate -> edit -> verify -> submit). Hard cutoff on exploration at turn 12. |
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
""")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Successfully generated {OUTPUT_MD} with {len(lines)} lines and {len(topics)} indexed topics!")

if __name__ == "__main__":
    main()
