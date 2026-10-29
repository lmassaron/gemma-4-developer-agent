# Gemma 4 Developer Agent — Local Benchmark Leaderboard

> Standardized evaluation of Gemma 4 developer agent candidates on verified SWE-bench tasks under official 360s/task competition constraints.

## Overall Leaderboard

| Rank | Model / Configuration | Suite | Resolution Rate | Resolved | Avg Time / Task | Timeouts (>360s) | Avg Turns | Last Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **v16_phased_edit_enforcement** | `smoke_3` | **66.7%** | 2/3 | 215.5s | 0 | 13.0 | 2026-10-28 19:01:54 |
| #2 | **v17_hill_climbed** | `climb_11` | **63.6%** | 7/11 | 158.7s | 1 | 8.4 | 2026-10-29 07:28:00 |
| #3 | **v16_phased_edit_enforcement** | `core_60` | **20.0%** | 12/60 | 350.9s | 33 | 12.6 | 2026-10-29 01:35:00 |
| #4 | **gemma-4-12b-sft-v6** | `smoke_3` | **0.0%** | 0/3 | 146.3s | 0 | 20.0 | 2026-10-19 08:47:43 |
| #5 | **v10_lean_thinking** | `smoke_3` | **0.0%** | 0/3 | 269.2s | 1 | 15.0 | 2026-10-25 11:08:51 |

## Benchmark Suites
- **`smoke_3`**: 3 representative tasks across core repositories (`psf/requests`, `fastapi/fastapi`, `Textualize/rich`) for rapid timeout verification and regression testing.
- **`core_12`**: 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich) providing a high-confidence local signal.

## Latest Evaluation Run Details
**Run ID**: `v17_hill_climbed_climb_11_2026-10-29 07:28:00`  
**Model**: `v17_hill_climbed` | **Suite**: `climb_11`  

| Instance ID | Repository | Status | Resolved | Duration (s) | Turns | Tool Calls | Patch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `requests_7205` | `psf/requests` | `RESOLVED` | ✅ PASS | 95.46s | 5 | 5 | 503 B |
| `rich_2725` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 189.23s | 6 | 6 | 1099 B |
| `fastapi_14786` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 79.25s | 5 | 5 | 406 B |
| `fastapi_14492` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 110.03s | 7 | 7 | 393 B |
| `fastapi_14794` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 106.22s | 6 | 6 | 487 B |
| `fastapi_14487` | `fastapi/fastapi` | `FAILED` | ❌ FAIL | 138.64s | 8 | 8 | 735 B |
| `fastapi_14349` | `fastapi/fastapi` | `FAILED` | ❌ FAIL | 241.64s | 13 | 13 | 514 B |
| `requests_7427` | `psf/requests` | `RESOLVED` | ✅ PASS | 88.35s | 5 | 5 | 716 B |
| `requests_7315` | `psf/requests` | `RESOLVED` | ✅ PASS | 76.71s | 5 | 5 | 522 B |
| `rich_4006` | `Textualize/rich` | `FAILED` | ❌ FAIL | 252.06s | 13 | 13 | 478 B |
| `rich_4077` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 367.96s | 19 | 19 | 0 B |
