# Gemma 4 Developer Agent — Local Benchmark Leaderboard

> Standardized evaluation of Gemma 4 developer agent candidates on verified SWE-bench tasks under official 360s/task competition constraints.

## Overall Leaderboard

| Rank | Model / Configuration | Suite | Resolution Rate | Resolved | Avg Time / Task | Timeouts (>360s) | Avg Turns | Last Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **v19_hill_climbed** | `climb_12_cross` | **66.7%** | 8/12 | 237.9s | 4 | 9.0 | 2026-10-29 21:14:26 |
| #2 | **v16_phased_edit_enforcement** | `smoke_3` | **66.7%** | 2/3 | 215.5s | 0 | 13.0 | 2026-10-28 19:01:54 |
| #3 | **v40_budget_surgical_v2** | `core_12` | **66.7%** | 8/12 | 366.1s | 3 | 12.2 | 2026-11-14 03:13:21 |
| #4 | **v17_hill_climbed** | `climb_11` | **63.6%** | 7/11 | 158.7s | 1 | 8.4 | 2026-10-29 07:28:00 |
| #5 | **v18_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 197.2s | 2 | 9.1 | 2026-10-29 13:26:00 |
| #6 | **v17_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 225.4s | 3 | 10.3 | 2026-10-29 09:05:00 |
| #7 | **v39_budget_surgical** | `core_12` | **50.0%** | 6/12 | 417.1s | 4 | 13.8 | 2026-11-14 01:51:56 |
| #8 | **v35_compaction2048_lean6_lora** | `core_12` | **50.0%** | 6/12 | 438.6s | 6 | 12.6 | 2026-11-11 12:02:58 |
| #9 | **v37_v23_restored_plus_tailguard** | `core_12` | **50.0%** | 6/12 | 540.4s | 6 | 13.3 | 2026-11-14 00:21:00 |
| #10 | **v18_hill_climbed** | `external_hf_py312` | **41.7%** | 5/12 | 346.8s | 6 | 12.3 | 2026-10-29 15:46:00 |
| #11 | **v34_clean024_mandatory_verify_anchor** | `core_12` | **41.7%** | 5/12 | 416.6s | 6 | 12.4 | 2026-11-11 10:26:45 |
| #12 | **v38_patchsmith_v9_tailguard** | `core_12` | **25.0%** | 3/12 | 501.3s | 9 | 11.5 | 2026-11-13 22:25:40 |
| #13 | **v19_hill_climbed** | `core_60` | **23.3%** | 14/60 | 294.0s | 38 | 11.3 | 2026-10-30 01:50:22 |
| #14 | **v16_phased_edit_enforcement** | `core_60` | **20.0%** | 12/60 | 350.9s | 33 | 12.6 | 2026-10-29 01:35:00 |
| #15 | **gemma-4-12b-sft-v6** | `smoke_3` | **0.0%** | 0/3 | 146.3s | 0 | 20.0 | 2026-10-19 08:47:43 |
| #16 | **v10_lean_thinking** | `smoke_3` | **0.0%** | 0/3 | 269.2s | 1 | 15.0 | 2026-10-25 11:08:51 |

## Benchmark Suites
- **`smoke_3`**: 3 representative tasks across core repositories (`psf/requests`, `fastapi/fastapi`, `Textualize/rich`) for rapid timeout verification and regression testing.
- **`core_12`**: 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich) providing a high-confidence local signal.

## Latest Evaluation Run Details
**Run ID**: `v40_budget_surgical_v2_core_12_2026-11-14 03:13:21`  
**Model**: `v40_budget_surgical_v2` | **Suite**: `core_12`  

| Instance ID | Repository | Status | Resolved | Duration (s) | Turns | Tool Calls | Patch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `requests_7205` | `psf/requests` | `RESOLVED` | ✅ PASS | 120.51s | 5 | 5 | 503 B |
| `requests_7433` | `psf/requests` | `RESOLVED` | ✅ PASS | 86.87s | 5 | 5 | 584 B |
| `requests_7502` | `psf/requests` | `RESOLVED_ANYTIME` | ✅ PASS | 488.9s | 15 | 15 | 615 B |
| `requests_7505` | `psf/requests` | `RESOLVED` | ✅ PASS | 416.74s | 16 | 16 | 1049 B |
| `fastapi_11194` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 501.65s | 23 | 23 | 0 B |
| `fastapi_11355` | `fastapi/fastapi` | `NO_PATCH` | ❌ FAIL | 421.27s | 28 | 28 | 0 B |
| `fastapi_12942` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 482.72s | 12 | 12 | 0 B |
| `fastapi_13537` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 394.29s | 6 | 6 | 828 B |
| `rich_2725` | `Textualize/rich` | `RESOLVED_ANYTIME` | ✅ PASS | 604.62s | 8 | 8 | 1099 B |
| `rich_2943` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 136.31s | 8 | 8 | 400 B |
| `rich_3006` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 192.2s | 12 | 12 | 558 B |
| `rich_3052` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 546.83s | 8 | 8 | 1964 B |
