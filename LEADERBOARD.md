# Gemma 4 Developer Agent — Local Benchmark Leaderboard

> Standardized evaluation of Gemma 4 developer agent candidates on verified SWE-bench tasks under official 360s/task competition constraints.

## Overall Leaderboard

| Rank | Model / Configuration | Suite | Resolution Rate | Resolved | Avg Time / Task | Timeouts (>360s) | Avg Turns | Last Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #1 | **v19_hill_climbed** | `climb_12_cross` | **66.7%** | 8/12 | 237.9s | 4 | 9.0 | 2026-10-29 21:14:26 |
| #2 | **v16_phased_edit_enforcement** | `smoke_3` | **66.7%** | 2/3 | 215.5s | 0 | 13.0 | 2026-10-28 19:01:54 |
| #3 | **v17_hill_climbed** | `climb_11` | **63.6%** | 7/11 | 158.7s | 1 | 8.4 | 2026-10-29 07:28:00 |
| #4 | **v18_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 197.2s | 2 | 9.1 | 2026-10-29 13:26:00 |
| #5 | **v17_hill_climbed** | `climb_12_expanded` | **50.0%** | 6/12 | 225.4s | 3 | 10.3 | 2026-10-29 09:05:00 |
| #6 | **v18_hill_climbed** | `external_hf_py312` | **41.7%** | 5/12 | 346.8s | 6 | 12.3 | 2026-10-29 15:46:00 |
| #7 | **v19_hill_climbed** | `core_60` | **23.3%** | 14/60 | 294.0s | 38 | 11.3 | 2026-10-30 01:50:22 |
| #8 | **v16_phased_edit_enforcement** | `core_60` | **20.0%** | 12/60 | 350.9s | 33 | 12.6 | 2026-10-29 01:35:00 |
| #9 | **gemma-4-12b-sft-v6** | `smoke_3` | **0.0%** | 0/3 | 146.3s | 0 | 20.0 | 2026-10-19 08:47:43 |
| #10 | **v10_lean_thinking** | `smoke_3` | **0.0%** | 0/3 | 269.2s | 1 | 15.0 | 2026-10-25 11:08:51 |

## Benchmark Suites
- **`smoke_3`**: 3 representative tasks across core repositories (`psf/requests`, `fastapi/fastapi`, `Textualize/rich`) for rapid timeout verification and regression testing.
- **`core_12`**: 12 balanced tasks (4 Requests, 4 FastAPI, 4 Rich) providing a high-confidence local signal.

## Latest Evaluation Run Details
**Run ID**: `v19_hill_climbed_core_60_2026-10-30 01:50:22`  
**Model**: `v19_hill_climbed` | **Suite**: `core_60`  

| Instance ID | Repository | Status | Resolved | Duration (s) | Turns | Tool Calls | Patch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `fastapi_15661` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 351.03s | 14 | 14 | 0 B |
| `fastapi_15588` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 333.07s | 11 | 11 | 711 B |
| `fastapi_15589` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 379.57s | 11 | 11 | 761 B |
| `fastapi_15030` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 339.18s | 18 | 18 | 0 B |
| `fastapi_14962` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 335.49s | 14 | 14 | 0 B |
| `fastapi_15280` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 334.19s | 15 | 15 | 0 B |
| `fastapi_14964` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 353.01s | 8 | 8 | 1254 B |
| `fastapi_15023` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 351.95s | 14 | 14 | 0 B |
| `fastapi_14986` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 384.16s | 10 | 10 | 1289 B |
| `fastapi_14978` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 392.09s | 16 | 16 | 725 B |
| `fastapi_14258` | `fastapi/fastapi` | `FAILED` | ❌ FAIL | 279.3s | 7 | 7 | 513 B |
| `fastapi_14851` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 366.85s | 14 | 14 | 0 B |
| `fastapi_14873` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 387.19s | 7 | 7 | 1905 B |
| `fastapi_13920` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 348.08s | 19 | 19 | 0 B |
| `fastapi_14794` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 199.18s | 9 | 9 | 487 B |
| `fastapi_14786` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 74.63s | 5 | 5 | 406 B |
| `fastapi_14492` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 128.63s | 7 | 7 | 393 B |
| `fastapi_14479` | `fastapi/fastapi` | `FAILED` | ❌ FAIL | 131.43s | 5 | 5 | 526 B |
| `fastapi_14463` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 169.65s | 7 | 7 | 698 B |
| `fastapi_12942` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 357.93s | 14 | 14 | 0 B |
| `fastapi_14099` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 360.16s | 13 | 13 | 0 B |
| `fastapi_14609` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 381.29s | 12 | 12 | 814 B |
| `fastapi_14372` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 113.68s | 5 | 5 | 518 B |
| `fastapi_14349` | `fastapi/fastapi` | `RESOLVED` | ✅ PASS | 291.83s | 10 | 10 | 1286 B |
| `fastapi_14512` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 335.29s | 11 | 11 | 0 B |
| `fastapi_14487` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 335.36s | 12 | 12 | 1155 B |
| `fastapi_11355` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 336.23s | 17 | 17 | 0 B |
| `fastapi_14485` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 361.61s | 16 | 16 | 0 B |
| `fastapi_14482` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 343.17s | 14 | 14 | 0 B |
| `fastapi_14306` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 377.65s | 12 | 12 | 975 B |
| `fastapi_14458` | `fastapi/fastapi` | `TIMEOUT` | ❌ FAIL | 340.72s | 9 | 9 | 1840 B |
| `requests_7505` | `psf/requests` | `RESOLVED` | ✅ PASS | 180.19s | 8 | 8 | 1049 B |
| `requests_7502` | `psf/requests` | `RESOLVED` | ✅ PASS | 214.74s | 10 | 10 | 615 B |
| `requests_7433` | `psf/requests` | `RESOLVED` | ✅ PASS | 94.68s | 5 | 5 | 584 B |
| `requests_7427` | `psf/requests` | `TIMEOUT` | ❌ FAIL | 379.67s | 12 | 12 | 0 B |
| `requests_7315` | `psf/requests` | `RESOLVED` | ✅ PASS | 86.08s | 5 | 5 | 522 B |
| `requests_7328` | `psf/requests` | `FAILED` | ❌ FAIL | 222.85s | 10 | 10 | 885 B |
| `rich_4070` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 364.98s | 14 | 14 | 0 B |
| `rich_4077` | `Textualize/rich` | `FAILED` | ❌ FAIL | 160.35s | 9 | 9 | 463 B |
| `rich_4079` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 346.94s | 14 | 14 | 0 B |
| `rich_4076` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 339.72s | 19 | 19 | 0 B |
| `rich_4075` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 350.89s | 15 | 15 | 0 B |
| `rich_4006` | `Textualize/rich` | `FAILED` | ❌ FAIL | 183.22s | 7 | 7 | 531 B |
| `rich_3718` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 274.44s | 11 | 11 | 753 B |
| `rich_3934` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 346.8s | 13 | 13 | 0 B |
| `rich_3953` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 355.13s | 16 | 16 | 0 B |
| `rich_3944` | `Textualize/rich` | `FAILED` | ❌ FAIL | 233.62s | 10 | 10 | 1061 B |
| `rich_3942` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 349.74s | 12 | 12 | 0 B |
| `rich_3938` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 332.55s | 14 | 14 | 0 B |
| `rich_3882` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 76.88s | 5 | 5 | 503 B |
| `rich_3894` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 400.3s | 12 | 12 | 1075 B |
| `rich_3935` | `Textualize/rich` | `FAILED` | ❌ FAIL | 232.83s | 10 | 10 | 615 B |
| `rich_3905` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 181.99s | 9 | 9 | 627 B |
| `rich_3930` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 353.17s | 13 | 13 | 0 B |
| `rich_3180` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 355.01s | 13 | 13 | 0 B |
| `rich_3486` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 411.69s | 12 | 12 | 302 B |
| `rich_3777` | `Textualize/rich` | `RESOLVED` | ✅ PASS | 335.33s | 13 | 13 | 1383 B |
| `rich_3468` | `Textualize/rich` | `TIMEOUT` | ❌ FAIL | 334.83s | 12 | 12 | 0 B |
| `rich_3782` | `Textualize/rich` | `FAILED` | ❌ FAIL | 185.54s | 9 | 9 | 524 B |
| `httpx_3672` | `encode/httpx` | `TIMEOUT` | ❌ FAIL | 380.68s | 13 | 13 | 1317 B |
