# Gemma 4 Developer Agent — Local Benchmark Leaderboard

> Standardized evaluation of Gemma 4 developer agent candidates on verified SWE-bench tasks under official 360s/task competition constraints.

## Overall Leaderboard

| Rank | Model / Configuration | Suite | Resolution Rate | Resolved | Avg Time / Task | Timeouts (>360s) | Avg Turns | Last Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **v40_heldout12** | `custom` | **91.7%** | 11/12 | 141.8s | 0 | 7.2 | 2026-11-14 22:31:16 |
| #2 | **v19_hill_climbed** | `climb_12_cross` | **66.7%** | 8/12 | 237.9s | 4 | 9.0 | 2026-10-29 21:14:26 |
| #3 | **v16_phased_edit_enforcement** | `smoke_3` | **66.7%** | 2/3 | 215.5s | 0 | 13.0 | 2026-10-28 19:01:54 |
| #4 | **v40_budget_surgical_v2** | `core_12` | **66.7%** | 8/12 | 366.1s | 3 | 12.2 | 2026-11-14 03:13:21 |
| #5 | **v44_v40_surgical_v3** | `core_12` | **66.7%** | 8/12 | 395.7s | 4 | 12.7 | 2026-11-15 18:26:00 |
| #6 | **v17_hill_climbed** | `climb_11` | **63.6%** | 7/11 | 158.7s | 1 | 8.4 | 2026-10-29 07:28:00 |
| #7 | **v41_budget_surgical_v3** | `core_12` | **58.3%** | 7/12 | 412.3s | 5 | 12.8 | 2026-11-15 02:01:39 |
| #8 | **v18_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 197.2s | 2 | 9.1 | 2026-10-29 13:26:00 |
| #9 | **v17_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 225.4s | 3 | 10.3 | 2026-10-29 09:05:00 |
| #10 | **v39_budget_surgical** | `core_12` | **50.0%** | 6/12 | 417.1s | 4 | 13.8 | 2026-11-14 01:51:56 |
| #11 | **v43_v40_kaggle_armor** | `core_12` | **50.0%** | 6/12 | 437.0s | 5 | 12.7 | 2026-11-15 16:58:34 |
| #12 | **v35_compaction2048_lean6_lora** | `core_12` | **50.0%** | 6/12 | 438.6s | 6 | 12.6 | 2026-11-11 12:02:58 |
| #13 | **v37_v23_restored_plus_tailguard** | `core_12` | **50.0%** | 6/12 | 540.4s | 6 | 13.3 | 2026-11-14 00:21:00 |
| #14 | **v18_hill_climbed** | `external_hf_py312` | **41.7%** | 5/12 | 346.8s | 6 | 12.3 | 2026-10-29 15:46:00 |
| #15 | **v34_clean024_mandatory_verify_anchor** | `core_12` | **41.7%** | 5/12 | 416.6s | 6 | 12.4 | 2026-11-11 10:26:45 |
| #16 | **v40_nearmiss_hard12** | `custom` | **41.7%** | 5/12 | 448.4s | 5 | 12.7 | 2026-11-15 00:28:16 |
| #17 | **v40_budget_surgical_v2_sympy12** | `custom` | **41.7%** | 5/12 | 502.3s | 7 | 14.1 | 2026-11-15 21:43:42 |
| #18 | **v44_v40_surgical_v3** | `custom` | **33.3%** | 4/12 | 403.4s | 4 | 13.1 | 2026-11-15 19:54:50 |
| #19 | **v42_patchsmith_surgical** | `core_12` | **33.3%** | 4/12 | 469.3s | 8 | 12.5 | 2026-11-15 15:22:34 |
| #20 | **v45_exp021_surgical_smoke3_thinking** | `smoke_3` | **33.3%** | 1/3 | 2309.5s | 2 | 12.0 | 2026-11-17 11:46:49 |
| #21 | **v41_nearmiss_hard12** | `custom` | **25.0%** | 3/12 | 382.6s | 4 | 13.3 | 2026-11-15 03:25:10 |
| #22 | **v38_patchsmith_v9_tailguard** | `core_12` | **25.0%** | 3/12 | 501.3s | 9 | 11.5 | 2026-11-13 22:25:40 |
| #23 | **v19_hill_climbed** | `core_60` | **23.3%** | 14/60 | 294.0s | 38 | 11.3 | 2026-10-30 01:50:22 |
| #24 | **v16_phased_edit_enforcement** | `core_60` | **20.0%** | 12/60 | 350.9s | 33 | 12.6 | 2026-10-29 01:35:00 |
| #25 | **v44_v40_surgical_v3_sympy12_thinking** | `custom` | **16.7%** | 2/12 | 723.4s | 10 | 6.3 | 2026-11-17 00:40:53 |
| #26 | **v42_patchsmith_surgical_thinking** | `core_12` | **8.3%** | 1/12 | 660.9s | 11 | 5.8 | 2026-11-16 22:08:13 |
| #27 | **gemma-4-12b-sft-v6** | `smoke_3` | **0.0%** | 0/3 | 146.3s | 0 | 20.0 | 2026-10-19 08:47:43 |
| #28 | **v10_lean_thinking** | `smoke_3` | **0.0%** | 0/3 | 269.2s | 1 | 15.0 | 2026-10-25 11:08:51 |

## Benchmark Suites
- **`smoke_3`**: 3 representative tasks across core repositories (`psf/requests`, `fastapi/fastapi`, `Textualize/rich`) for rapid timeout verification and regression testing.
- **`core_12`**: 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich) providing a high-confidence local signal.

## Latest Evaluation Run Details
**Run ID**: `v45_exp021_surgical_smoke3_thinking_smoke_3_2026-11-17 11:46:49`  
**Model**: `v45_exp021_surgical_smoke3_thinking` | **Suite**: `smoke_3`  

| Instance ID | Repository | Status | Resolved | Duration (s) | Turns | Tool Calls | Patch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `requests_7205` | `psf/requests` | `RESOLVED_ANYTIME` | ✅ PASS | 2062.92s | 10 | 10 | 492 B |
| `fastapi_11194` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 2292.1s | 13 | 13 | 0 B |
| `rich_2725` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 2573.59s | 13 | 13 | 0 B |
