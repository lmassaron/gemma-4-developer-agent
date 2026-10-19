# Gemma 4 Developer Agent — Local Benchmark Leaderboard

> Standardized evaluation of Gemma 4 developer agent candidates on verified SWE-bench tasks under official 360s/task competition constraints.

## Overall Leaderboard

| Rank | Model / Configuration | Suite | Resolution Rate | Resolved | Avg Time / Task | Timeouts (>360s) | Avg Turns | Last Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **gemma-4-12b-sft-v6** | `smoke_3` | **0.0%** | 0/3 | 146.3s | 0 | 20.0 | 2026-10-19 08:47:43 |

## Benchmark Suites
- **`smoke_3`**: 3 representative tasks across core repositories (`psf/requests`, `fastapi/fastapi`, `Textualize/rich`) for rapid timeout verification and regression testing.
- **`core_12`**: 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich) providing a high-confidence local signal.

## Latest Evaluation Run Details
**Run ID**: `gemma-4-12b-sft-v6_smoke_3_2026-10-19 08:47:43`  
**Model**: `gemma-4-12b-sft-v6` | **Suite**: `smoke_3`  

| Instance ID | Repository | Status | Resolved | Duration (s) | Turns | Tool Calls | Patch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `requests_7205` | `psf/requests` | `NO_PATCH` | ❌ FAIL | 161.37s | 20 | 20 | 0 B |
| `fastapi_11194` | `fastapi/fastapi` | `NO_PATCH` | ❌ FAIL | 144.16s | 20 | 20 | 0 B |
| `rich_2725` | `Textualize/rich` | `NO_PATCH` | ❌ FAIL | 133.37s | 20 | 20 | 0 B |
