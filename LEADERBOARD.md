# Gemma 4 Developer Agent — Local Benchmark Leaderboard

> Standardized evaluation of Gemma 4 developer agent candidates on verified SWE-bench tasks under official 360s/task competition constraints.

## Overall Leaderboard

| Rank | Model / Configuration | Suite | Resolution Rate | Resolved | Avg Time / Task | Timeouts (>360s) | Avg Turns | Last Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **v40_heldout12** | `custom` | **91.7%** | 11/12 | 141.8s | 0 | 7.2 | 2026-11-14 22:31:16 |
| #2 | **v19_hill_climbed** | `climb_12_cross` | **66.7%** | 8/12 | 237.9s | 4 | 9.0 | 2026-10-29 21:14:26 |
| #3 | **v16_phased_edit_enforcement** | `smoke_3` | **66.7%** | 2/3 | 215.5s | 0 | 13.0 | 2026-10-28 19:01:54 |
| #4 | **v40_budget_surgical_v2** | `core_12` | **66.7%** | 8/12 | 366.1s | 3 | 12.2 | 2026-11-14 03:13:21 |
| #5 | **v17_hill_climbed** | `climb_11` | **63.6%** | 7/11 | 158.7s | 1 | 8.4 | 2026-10-29 07:28:00 |
| #6 | **v41_budget_surgical_v3** | `core_12` | **58.3%** | 7/12 | 412.3s | 5 | 12.8 | 2026-11-15 02:01:39 |
| #7 | **v18_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 197.2s | 2 | 9.1 | 2026-10-29 13:26:00 |
| #8 | **v17_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 225.4s | 3 | 10.3 | 2026-10-29 09:05:00 |
| #9 | **v39_budget_surgical** | `core_12` | **50.0%** | 6/12 | 417.1s | 4 | 13.8 | 2026-11-14 01:51:56 |
| #10 | **v35_compaction2048_lean6_lora** | `core_12` | **50.0%** | 6/12 | 438.6s | 6 | 12.6 | 2026-11-11 12:02:58 |
| #11 | **v37_v23_restored_plus_tailguard** | `core_12` | **50.0%** | 6/12 | 540.4s | 6 | 13.3 | 2026-11-14 00:21:00 |
| #12 | **v18_hill_climbed** | `external_hf_py312` | **41.7%** | 5/12 | 346.8s | 6 | 12.3 | 2026-10-29 15:46:00 |
| #13 | **v34_clean024_mandatory_verify_anchor** | `core_12` | **41.7%** | 5/12 | 416.6s | 6 | 12.4 | 2026-11-11 10:26:45 |
| #14 | **v40_nearmiss_hard12** | `custom` | **41.7%** | 5/12 | 448.4s | 5 | 12.7 | 2026-11-15 00:28:16 |
| #15 | **v41_nearmiss_hard12** | `custom` | **25.0%** | 3/12 | 382.6s | 4 | 13.3 | 2026-11-15 03:25:10 |
| #16 | **v38_patchsmith_v9_tailguard** | `core_12` | **25.0%** | 3/12 | 501.3s | 9 | 11.5 | 2026-11-13 22:25:40 |
| #17 | **v19_hill_climbed** | `core_60` | **23.3%** | 14/60 | 294.0s | 38 | 11.3 | 2026-10-30 01:50:22 |
| #18 | **v16_phased_edit_enforcement** | `core_60` | **20.0%** | 12/60 | 350.9s | 33 | 12.6 | 2026-10-29 01:35:00 |
| #19 | **gemma-4-12b-sft-v6** | `smoke_3` | **0.0%** | 0/3 | 146.3s | 0 | 20.0 | 2026-10-19 08:47:43 |
| #20 | **v10_lean_thinking** | `smoke_3` | **0.0%** | 0/3 | 269.2s | 1 | 15.0 | 2026-10-25 11:08:51 |

## Benchmark Suites
- **`smoke_3`**: 3 representative tasks across core repositories (`psf/requests`, `fastapi/fastapi`, `Textualize/rich`) for rapid timeout verification and regression testing.
- **`core_12`**: 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich) providing a high-confidence local signal.

## Latest Evaluation Run Details
**Run ID**: `v41_nearmiss_hard12_custom_2026-11-15 03:25:10`  
**Model**: `v41_nearmiss_hard12` | **Suite**: `custom`  

| Instance ID | Repository | Status | Resolved | Duration (s) | Turns | Tool Calls | Patch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `requests_7328` | `psf/requests` | `FAILED` | ❌ FAIL | 287.03s | 13 | 13 | 885 B |
| `fastapi_15588` | `fastapi/fastapi` | `FAILED` | ❌ FAIL | 458.33s | 10 | 10 | 1530 B |
| `fastapi_14964` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 496.97s | 13 | 13 | 1254 B |
| `fastapi_14873` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 491.3s | 14 | 14 | 1487 B |
| `fastapi_14794` | `fastapi/fastapi` | `RESOLVED_ANYTIME` | ✅ PASS | 492.55s | 15 | 15 | 793 B |
| `fastapi_14605` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 490.59s | 20 | 20 | 1784 B |
| `fastapi_14463` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 363.8s | 12 | 12 | 698 B |
| `fastapi_14258` | `fastapi/fastapi` | `FAILED` | ❌ FAIL | 423.8s | 12 | 12 | 506 B |
| `rich_4075` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 483.12s | 25 | 25 | 0 B |
| `rich_3944` | `Textualize/rich` | `FAILED` | ❌ FAIL | 227.83s | 10 | 10 | 1053 B |
| `rich_3882` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 123.94s | 6 | 6 | 503 B |
| `rich_3675` | `Textualize/rich` | `FAILED` | ❌ FAIL | 251.35s | 10 | 10 | 633 B |
