# Gemma 4 Developer Agent — Local Benchmark Leaderboard

> Standardized evaluation of Gemma 4 developer agent candidates on verified SWE-bench tasks under official 360s/task competition constraints.

## Overall Leaderboard

| Rank | Model / Configuration | Suite | Resolution Rate | Resolved | Avg Time / Task | Timeouts (>360s) | Avg Turns | Last Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **v16_phased_edit_enforcement** | `smoke_3` | **66.7%** | 2/3 | 215.5s | 0 | 13.0 | 2026-10-28 19:01:54 |
| #2 | **v17_hill_climbed** | `climb_11` | **63.6%** | 7/11 | 158.7s | 1 | 8.4 | 2026-10-29 07:28:00 |
| #3 | **v18_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 197.2s | 2 | 9.1 | 2026-10-29 13:26:00 |
| #4 | **v17_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 225.4s | 3 | 10.3 | 2026-10-29 09:05:00 |
| #5 | **v16_phased_edit_enforcement** | `core_60` | **20.0%** | 12/60 | 350.9s | 33 | 12.6 | 2026-10-29 01:35:00 |
| #6 | **gemma-4-12b-sft-v6** | `smoke_3` | **0.0%** | 0/3 | 146.3s | 0 | 20.0 | 2026-10-19 08:47:43 |
| #7 | **v10_lean_thinking** | `smoke_3` | **0.0%** | 0/3 | 269.2s | 1 | 15.0 | 2026-10-25 11:08:51 |

## Benchmark Suites
- **`smoke_3`**: 3 representative tasks across core repositories (`psf/requests`, `fastapi/fastapi`, `Textualize/rich`) for rapid timeout verification and regression testing.
- **`core_12`**: 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich) providing a high-confidence local signal.

## Latest Evaluation Run Details
**Run ID**: `v18_hill_climbed_climb_12_expanded_2026-10-29 13:26:00`  
**Model**: `v18_hill_climbed` | **Suite**: `climb_12_expanded`  

| Instance ID | Repository | Status | Resolved | Duration (s) | Turns | Tool Calls | Patch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `requests_7205` | `psf/requests` | `RESOLVED` | ✅ PASS | 98.23s | 5 | 5 | 503 B |
| `requests_7315` | `psf/requests` | `RESOLVED` | ✅ PASS | 77.51s | 5 | 5 | 522 B |
| `fastapi_14786` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 80.32s | 5 | 5 | 406 B |
| `rich_3882` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 109.11s | 5 | 5 | 503 B |
| `rich_3486` | `Textualize/rich` | `FAILED` | ❌ FAIL | 343.22s | 12 | 12 | 1027 B |
| `fastapi_14372` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 99.11s | 7 | 7 | 518 B |
| `rich_3777` | `Textualize/rich` | `FAILED` | ❌ FAIL | 128.99s | 7 | 7 | 525 B |
| `rich_3934` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 414.1s | 16 | 16 | 0 B |
| `requests_7309` | `psf/requests` | `RESOLVED` | ✅ PASS | 117.73s | 5 | 5 | 692 B |
| `requests_6757` | `psf/requests` | `FAILED` | ❌ FAIL | 294.89s | 16 | 16 | 528 B |
| `fastapi_14459` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 369.0s | 15 | 15 | 0 B |
| `rich_3772` | `Textualize/rich` | `FAILED` | ❌ FAIL | 234.35s | 11 | 11 | 563 B |
