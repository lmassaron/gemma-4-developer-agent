# Submissions Tracking Log

This document tracks all submission bundles built for the Kaggle Gemma 4 Developer Agent Competition, including their exact Git commits, SHA-256 hashes, features, and evaluation outcomes.

| Version | Archive File | Git Commit | SHA-256 | Features / Notes | LB Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v1 (Initial)** | `submission_v1.zip` | `a1b4551` | `5c7ef...` | Baseline prompt with standard turn budget. | 0.13 | Archived |
| **v2 (Fallback)** | [`submission_v2_fallback.zip`](file:///Users/lucamassaron/code/gemma-4-developer-agent-competition/submissions_archive/submission_v2_fallback.zip) | `dbef043` | `bb0e1769c074458fc578bcedc1ef022aa440965b` | Turn-12 test-and-repair deadline, heredoc fallback, sync/async guidance. | 0.22 - 0.24 (Public peer baseline) | **Ready (Fallback)** |
| **v3 (Current)** | [`submission_v3_upgraded.zip`](file:///Users/lucamassaron/code/gemma-4-developer-agent-competition/submissions_archive/submission_v3_upgraded.zip) / `submission.zip` | `0c376cb` | `152b6f70a367030253d3fc0ece397d23d24a75816e5102b2a437896e644039d1` | Time banking via early submit, isolated `/tmp` reproducer check, repo playbooks (FastAPI, Rich, Requests/HTTPX), tail -40 truncation guard, graph tools pruned from schema. | *Unmeasured (Scheduled for tomorrow)* | **Ready (Primary Prompt-Only)** |
| **v4 (SFT 31B)** | [`submission_v4_gemma4_31b_sft.zip`](file:///Users/lucamassaron/code/gemma-4-developer-agent-competition/submissions_archive/submission_v4_gemma4_31b_sft.zip) / [`submission_gemma4_31b_sft.zip`](file:///Users/lucamassaron/code/gemma-4-developer-agent-competition/submission_gemma4_31b_sft.zip) | `cce3bf8` | `774abc59e2b83a984b11bca531f1ef25b7c6257ff5195950c6ab0599ad42169f` | Fine-tuned 31B LoRA adapter (`gemma4_31b_swe_lora`, 60 layers, $r=16, \alpha=32$, 467 MB). Trained 50 steps on DGX Spark (NVIDIA GB10), loss converged from 3.15 to 0.56. Directly compatible with `gemma-4-31b-it-qat-w4a16-ct`. | *Trained LoRA Candidate* | **Ready (Primary SFT)** |

