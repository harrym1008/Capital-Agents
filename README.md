# Capital-Agents

A multi-agent, locally run, portfolio management system using multiple roles which LLM agents will complete their analyses under.

Agents run in a "boardroom" (Single Equity Rating, Portfolio Creation, Portfolio Rebalancing, Agent Portfolio Simulation) with live tools for prices, fundamentals, SEC filings, news sentiment, macro/FX, sectors and short interest. There is also a User-Driven Portfolio Simulation (backtester/charts, no LLM required).

> **You do not need to edit any code to run this.** Follow the steps below: install Python deps → create `.env` → restore `data/` → start the UI → pick an LLM provider → run a boardroom.

---

## 1. System requirements

| Requirement | Details |
|---|---|
| OS | Windows 10/11 64-bit (primary target — `pywin32`, `llama-server.exe`, native file dialogs are Windows-oriented). Should also work on Linux with a custom `llama-server` binary path, but not officially tested. |
| Python | **Python 3.12.x** (developed on 3.12.6). Create a venv — do not install globally. |
| RAM / Disk | 16 GB+ RAM recommended. ~10–30 GB free for `data/` parquet + `finbert/models/` + GGUF models. |
| GPU (optional but strongly recommended) | Any NVIDIA CUDA-capable GPU. Both the LLM (via llama.cpp + CUDA) and FinBERT sentiment can use CUDA. **Everything also runs on CPU**, just slower (see §4). |
| CUDA stack (GPU only) | `requirements.txt` pins `torch==2.12.1+cu130`, `onnxruntime-gpu==1.28.0`, `tensorrt==11.2.1.2` / `tensorrt_cu13`, `nvidia-cudnn-cu13`, etc. Install the matching NVIDIA driver + CUDA 13.x runtime. No manual CUDA install is needed for PyTorch/ONNX in most cases — pip wheels bundle it — but TensorRT requires a working CUDA driver. |
| Network | Needed once for pip + Hugging Face model download (`tabularisai/ModernFinBERT`) + OpenRouter/cloud LLMs + live Finnhub/Alpaca/FRED calls. Boardroom backtests themselves run off local `data/` parquet. |

---

## 2. Install

```powershell
# 1. Clone / unzip the project, then from the project root:
python --version
# -> should be 3.12.x

python -m venv venv
.\venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> First install takes a while (torch + CUDA wheels, transformers, onnxruntime-gpu, TensorRT, yfinance, edgartools, alpaca-py, Flask, websockets, etc.).

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# e.g. 2.12.1+cu130 True  (GPU)  or  ... False (CPU-only — still fine)
```

---

## 3. API keys — `.env` setup

The app reads keys via `python-dotenv` (`load_dotenv()`). **Create a file named `.env` in the project root** (same folder as `main.py`). It is git-ignored — never commit it.

Copy-paste this template and fill in your own values (leave unused ones blank — see notes):

```ini
# --- LLM (cloud options, optional if you only use local llama.cpp) ---
OPENROUTER_API_KEY=

# --- Market data collectors + live boardroom tools ---
ALPACA_API_KEY=
ALPACA_API_SECRET=
ALPACA_API_KEY_2=
ALPACA_API_SECRET_2=

FRED_API_KEY=
FINNHUB_API_KEY=

MASSIVE_API_KEY=
ALPHAVANTAGE_API_KEY=
```

| Key | Where to get it (free tier works) | What it is used for | Required? |
|---|---|---|---|
| `OPENROUTER_API_KEY` | https://openrouter.ai/ → Keys | OpenRouter provider in the LLM picker (`llm/server_manager.py` reads `os.getenv("OPENROUTER_API_KEY")`). Powers cloud boardrooms. `:free` models are rate-limited to ~20 req/min in-app; paid models ~10/sec. | Only if you use OpenRouter. Not needed for local llama.cpp. |
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | https://alpaca.markets/ → paper keys | OHLCV + corporate-action downloads (`collectors/ohlcv_dl_client.py`) and ticker enrichment (`collectors/ticker_dl_client.py`). Also fallback for live news (`dataquery/news_provider.py`). | Needed for full local data + live tools. |
| `ALPACA_API_KEY_2` / `ALPACA_API_SECRET_2` | Same as above — create a **second** Alpaca key, or reuse the first | Dedicated news download key (`collectors/news_dl_client.py`). Lets you download OHLCV and news in parallel without sharing one rate-limit bucket. Falls back to `ALPACA_API_KEY` at runtime if `_2` is blank. | Recommended. Reuse key 1 if you only have one. |
| `FRED_API_KEY` | https://fred.stlouisfed.org/ → API Keys | Macro series (rates, CPI, unemployment, etc.) via `fredapi` (`collectors/macro_dl_client.py`, `dataquery/macro_provider.py`). If blank, FRED tools return empty and the app continues on yfinance macro tickers. | Needed for macro/FRED tools. |
| `FINNHUB_API_KEY` | https://finnhub.io/ → free key | Live point-in-time fundamentals (P/E, margins, ROE, EPS growth, etc.) via `dataquery/finnhub_provider.py` + ticker fallback in `dataquery/ticker_provider.py`. If blank, those tools return `None`/empty — boardroom still runs. | Needed for fundamentals tools. |
| `MASSIVE_API_KEY` | https://massive.com/ (formerly Polygon) | Full NYSE+NASDAQ ticker universe incl. delisted (`collectors/ticker_dl_client.py`). | Only needed if you re-run the mass ticker download. Not needed if you use the pre-downloaded `data.zip`. |
| `ALPHAVANTAGE_API_KEY` | https://www.alphavantage.co/ | Legacy ticker fallback (currently mostly commented out / optional). | Optional. |
| SEC EDGAR | No key — hardcoded as `CapitalAgents/1.0 (hpmm1@student.london.ac.uk)` in `collectors/constants.py` via `edgartools.set_identity()` | 10-Q/10-K filings. | Nothing to do. |
| yfinance | No key | Macro indices/commodities + price fallbacks. | Nothing to do. |

> The author will **not** provide keys. Each link above has a free tier sufficient for evaluation. You do **not** need to read or share anyone else's `.env` — just create your own.

---

## 4. Data setup (do this before running boardrooms)

Boardrooms are point-in-time backtests over local parquet in `data/` (2016-01-01 → 2026-09-19, see `collectors/constants.py`). Without `data/`, tickers/prices/news/macro lookups fail.

### Option A — recommended: restore the pre-downloaded zip

> **[Download pre-downloaded `data.zip` from Google Drive](https://drive.google.com/REPLACE_ME)** — maintainer: replace `REPLACE_ME` with your share link.

1. Download `CapitalAgents_data.zip`.
2. Extract it **into the project root** so you get:
   ```
   data/
     tickers.parquet  news.parquet  shortdata.parquet
     corpactions.parquet  tickerchanges.parquet  sectorleaders.parquet
     newsfiltered.parquet (optional)
     ohlcv_nyse/  ohlcv_nasdaq/  macro/  forex/  sectors/  newsbatches/
     models/  (HF cache for ModernFinBERT tokenizer)
   finbert/models/
     ModernFinBERT_fp32.onnx
     ModernFinBERT_fp8.engine
     hf/  (optional HF cache)
   ```
3. Done — skip to §5. No API keys or multi-hour downloads needed for backtests (keys are still needed for *live* Finnhub/FRED/Alpaca calls and cloud LLMs).

### Option B — download everything yourself (multiple hours)

```powershell
.\venv\Scripts\Activate.ps1
python exec/mass_downloader.py
```

You will be prompted per dataset (tickers → OHLCV → news → macro → forex → short → sector → sector-leaders). Tickers **must** run first. OHLCV/news/macro/forex/short/sector then run in parallel (8 threads each). Expect hours for full 2016→2026 history and respect rate limits. Requires all keys in §3.

### Expected layout after either option

Run from project root — paths are relative (`DATA_DIR = "data"`):

- `data/tickers.parquet`, `data/news.parquet`, `data/shortdata.parquet`, `data/sectorleaders.parquet`, `data/corpactions.parquet`, `data/tickerchanges.parquet`
- `data/ohlcv_nyse/*.parquet`, `data/ohlcv_nasdaq/*.parquet`, `data/macro/*`, `data/forex/*`, `data/sectors/*`

If a file is missing the corresponding tool returns empty rather than crashing, but results will be thin — prefer Option A.

---

## 5. Sentiment analysis setup (FinBERT / ModernFinBERT — CPU and CUDA both supported)

Sentiment (`llmtools/functions/sentiment_*.py`) uses `tabularisai/ModernFinBERT` (3 labels: `bearish`/`neutral`/`bullish`) through `finbert/finbert_engines.py`. **You do not need to configure anything** — on first boardroom start the app preloads the best engine available in a background thread (`ServerManager.startSentimentEnginePreload()`) and warms it up.

Engine auto-selection, fastest → slowest:

| # | Engine class | Device | File needed | Batch | When it loads |
|---|---|---|---|---|---|
| 1 | `TrtCudaInferenceEngine` | CUDA | `finbert/models/ModernFinBERT_fp8.engine` | 16 | NVIDIA GPU + TensorRT working + `.engine` present |
| 2 | `OnnxCudaInferenceEngine` | CUDA | `finbert/models/ModernFinBERT_fp32.onnx` | 8 | NVIDIA GPU + `onnxruntime-gpu` + `.onnx` present |
| 3 | `PytorchCudaInferenceEngine` | CUDA | HF `tabularisai/ModernFinBERT` (auto-download to `data/models`) | 16 | NVIDIA GPU + torch CUDA (no files needed) |
| 4 | `OnnxCpuInferenceEngine` | CPU | `finbert/models/ModernFinBERT_fp32.onnx` | 4 | Always works if `.onnx` present — no GPU needed |
| 5 | `PytorchCpuInferenceEngine` | CPU | HF `tabularisai/ModernFinBERT` (auto-download) | 4 | Last-resort fallback — pure CPU + torch, always works with internet on first run |

What this means for you:

- **CPU-only machine:** just run the app. It will use #4 if you restored `data.zip` (contains the `.onnx`), else #5 (downloads ~150 MB from Hugging Face to `data/models` on first run). No CUDA/TensorRT needed.
- **NVIDIA GPU machine:** install `requirements.txt` as-is (includes `torch+cu130`, `onnxruntime-gpu`, `tensorrt`). If you restored `data.zip` you immediately get #1 (FP8 TensorRT, lowest VRAM) or #2. Otherwise #3 still gives GPU acceleration with zero extra files.
- **Regenerating the quantised files (optional, GPU + TensorRT only):** `python finbert/finbert_quantise.py` — exports FP32 ONNX (`convertToFp32Onnx`) then FP8-quantises with `modelopt` + builds the TensorRT engine (`compileTensorRtEngine`, max batch 16, seq 768). Needs the FinancialPhraseBank calibration file + a CUDA+TensorRT host. You do **not** need to do this if you used the Drive zip.
- **VRAM behaviour:** the sentiment engine shares VRAM with your local LLM. Stopping the server (`Stop Server` button) calls `unloadSentimentEngine()` + `torch.cuda.empty_cache()`. Tick `Clear VRAM before Loading` on the Server page to run `rudimentaryVramClear()` before llama.cpp starts. See `exec/tests/test_finbert_vram.py` for per-engine VRAM benchmarks (`python exec/tests/test_finbert_vram.py` on a CUDA host).
- Tokenizer cache: `AutoTokenizer.from_pretrained("tabularisai/ModernFinBERT", cache_dir="data/models", model_max_length=768)`. Keep `data/models/` from the zip or let it re-download.

If sentiment fails to load, the boardroom still starts — sentiment tools just return errors. Check the llama/server logs in the UI for `[FinBERT Engine] ...` lines.

---

## 6. Local LLM usage (llama.cpp — fully offline)

Local inference is via `llama-server` (OpenAI-compatible HTTP on `127.0.0.1:9081`) driven by `llm/llamacpp/llamacpp_init.py` + `llm/llamacpp/llamacpp_args.py`.

### 6.1 Get `llama-server.exe`

- **Easiest (Windows):** download a prebuilt CUDA release from https://github.com/ggerganov/llama.cpp/releases (e.g. `llama-*-bin-win-cuda-*.zip`), extract `llama-server.exe`, and either:
  - put it on your `PATH`, or
  - paste its full path into the app later (§7.2 `Llama Server Executable` field — `Browse` opens a native file dialog).
- The app looks for `llama-server.exe` by default (`LLAMACPP_EXECUTABLE`, `LLAMACPP_PORT = 9081`). Any OpenAI-compatible `llama-server` build works; for GPU offload use the `-cu*` build matching your driver.
- Verify: `llama-server.exe --version` should print.

### 6.2 Get at least one GGUF model

- Download any instruction-tuned **GGUF** (e.g. from Hugging Face — search `Qwen3 GGUF`, `Gemma 3 GGUF`, `Llama 3.1 GGUF`, `DeepSeek-R1-Distill GGUF`). Quant examples: `Q4_K_M` (balanced), `Q6_K`/`Q8_0` (better quality, more VRAM/RAM).
- **Model requirements for this app (important):**
  - Must support **tool calling / function calling** (boardroom agents call ~20 tools via OpenAI `tools` schema — `llm/llm_client.py`). Most recent Qwen/Gemma/Llama/Mistral instruct models do; tiny or base (non-instruct) models do not.
  - **Context ≥ 32k, ideally ≥ 50k.** The UI warns if OpenRouter context is < 50k — same applies locally. Boardroom prompts + tool outputs are large; small-context models will truncate/fail.
  - **Thinking/reasoning models supported** (`thinkingBudget` maps to `thinking_budget_tokens`/`reasoning_effort` for llama.cpp, `reasoning.max_tokens` for OpenRouter). Both reasoning and non-reasoning models work; if a model errors on `tool_choice="required"` the client auto-retries with `"auto"`.
  - Tested class during development: 12B-scale instruct models (e.g. Gemma-3-12B, Qwen3-14B). Smaller 7–8B models run on less VRAM but give weaker financial reasoning; larger 24–32B need more VRAM/RAM and offload flags.
- Keep the `.gguf` anywhere (e.g. `D:\models\qwen3-14b-q4_k_m.gguf`). The app validates the `GGUF` magic header on add (`validateGGUFPath`).

### 6.3 Suggested llama.cpp flags (you set these in the UI, §7.2)

- GPU offload: `-ngl 999` (offload all layers) or `-ngl 0` for CPU-only.
- Context: `-c 65536` (or at least 32768) + `--context-shift` (app already forces `--context-shift`).
- Parallel: app handles `-np`/`--parallel` via the `Two Slots in Parallel` checkbox — leave it ticked for boardrooms (agents run tools in parallel threads).
- Sampling/personality is set per-model in the UI (`--temp`, `--top-p`, `--top-k`, `--min-p`, `--presence-penalty`, `--repeat-penalty`, `--frequency-penalty`). Boardroom calls themselves use `temperature=0.5` by default (configurable per run).

---

## 7. The LLM picker — Server Configuration + llama.cpp Setup pages

Start the app (§8), then open the **Server Configuration** page. This is the "LLM picker".

### 7.1 Provider dropdown (`/` → `Server Configuration`)

| Provider value | What happens on `Start Server` | Fields |
|---|---|---|
| `Llama.cpp` (default, fully local) | Spawns `llama-server.exe -m <your GGUF> --host 127.0.0.1 --port 9081 ...` via `LlamaCppProcessInitiator`, waits up to 120 s for `/health`, then sends a test chat (`Reply solely with 'OK'`). On success the landing-page boardroom buttons unlock. Metrics (`tokens/s`) poll `/metrics`. | `Selected Model` dropdown (from your llama.cpp config) + `Two Slots in Parallel` (checked = multi-agent parallelism) + `Clear VRAM before Loading`. `Configure Llama.cpp` button jumps to §7.2. |
| `OpenRouter` (cloud) | Creates `OpenRouterClient(base_url=https://openrouter.ai/api/v1, api_key=OPENROUTER_API_KEY)` and test-queries the chosen model ID. No local process. | Searchable model box (live catalog from `https://openrouter.ai/api/v1/models`, cached 1 h) + `Selected Provider` router (`Automatic` or pin e.g. `together`, `fireworks` — sent as `extra_body.provider.order` with fallbacks allowed) + live pricing per 1M (`Input/Output/Cache`) + context < 50k warning. Invalid IDs are rejected client-side. |
| `OpenAI Compatible` (cloud or another local server) | Creates `OpenAICompatibleClient(baseUrl, apiKey, model)` and test-queries it. Use for OpenAI, vLLM, LM Studio, Ollama (`http://localhost:11434/v1`), a second llama-server, etc. | `Base URL` (e.g. `http://127.0.0.1:8000/v1`), `API Key` (use `EMPTY`/blank for local servers), `Model Name` (e.g. `gpt-4o`, `qwen3:14b`). |

- `Start Server` → status panel (`Ready to Use` + `Choose Boardroom →`) + live logs (`llamaCppLog` events). `Stop Server` kills llama.cpp (if local), unloads FinBERT, resets cost tracker, and broadcasts `serverStatus` over WebSocket. Config persists to `llm/server_config.json` (git-ignored): last provider, last OpenRouter model + router, last OpenAI-Compatible URL/key/model.
- Cost tracker (`llm/token_cost_tracker.py`) records usage from stream `usage` chunks (OpenRouter `include_usage=True`).

### 7.2 `Configure Llama.cpp` page (`/llamacpp-setup`) — models + parameters

This edits the `llamacpp` section of `llm/server_config.json` (via `GET/POST /api/llamacpp/config`). Autosaves (~300 ms debounce) + explicit `Save Configuration`.

- **Left card — Global Arguments:**
  - `Llama Server Executable`: filename on `PATH` (`llama-server.exe`) or full path. `Browse` opens a native OS dialog (`POST /api/llamacpp/browse-executable`).
  - `Default Argument Flags`: key/value rows (e.g. `--ctx-size` / `65536`, `-ngl` / `999`, `--flash-attn` / `on`) with ✓/✗ enable toggle and `-` delete, `+` to add. Applied to **every** model unless that model overrides the same key.
- **Right card — Models:**
  - `Selected Model` dropdown + `Add Model` (native `.gguf` picker → `POST /api/llamacpp/browse-gguf`, validates header + pre-fills alias from filename) / `Delete` (confirm). Each entry stores `{id, alias, modelPath, executablePath, args}`. Display name = alias or filename without `.gguf`, sorted A–Z with last-used hoisted to top.
  - `Alias`: friendly name shown in the Server page dropdown (e.g. `Qwen3-14B-Q4`).
  - `Model File Path`: read-only full `.gguf` path.
  - `Per Model Arguments`: same key/value rows as global, but only for this model. Active per-model keys **override** globals (global row shows `O` = overridden).
  - `Sampling Params` (right column, dedicated inputs — equivalent to adding these flags manually): `Temperature (--temp)`, `Top P (--top-p)`, `Top K (--top-k)`, `Min P (--min-p)`, `Presence Penalty (--presence-penalty)`, `Repetition Penalty (--repeat-penalty)`, `Frequency Penalty (--frequency-penalty)`. Empty = flag omitted. Good defaults to try: `temp 0.5–0.7`, `top-p 0.9`, `top-k 40`, `repeat-penalty 1.1`.
- **Locked by the app (cannot be set):** `--host`/`-h` (forced `127.0.0.1`), `--port`/`-p` (forced `9081`), `-m`/`--model` (from your selected model), `--log-verbosity`/`-lv` (forced `4`). The UI blocks these with an alert; the backend strips them in `cleanUserArgs()` too.
- Final command preview is logged on start (`Command: ...` in the status log window, built by `buildLlamaCppCommandLine()`): locked flags → `-m <gguf>` → enabled globals → enabled per-model → `-np 1` only if parallel unchecked.

---

## 8. Run the project

```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

- Flask UI → http://127.0.0.1:9091 (landing page)
- Boardroom WebSocket → `127.0.0.1:9092` (automatic — no need to open manually)
- Local LLM HTTP (when llama.cpp is started) → http://127.0.0.1:9081 (`/health`, `/v1`, `/metrics`)

Workflow:

1. Open http://127.0.0.1:9091 → `Server Configuration`.
2. Pick a provider (§7.1). For first local run: `Configure Llama.cpp` → set executable → `Add Model` → pick your `.gguf` → set `-ngl`/`-c`/sampling → `Save Configuration` → back → select alias → `Start Server` → wait for `Ready to Use`.
3. `Choose Boardroom →` (or click header links). Boardroom pages require a loaded model and redirect to `/` otherwise — except `User-Driven Portfolio Simulation`, which works with no LLM.
4. Configure the boardroom (ticker/date/horizons/bias/limits/temperature/thinking budget — defaults are sane: `temperature 0.5`, `thinkingBudget 2048`, `maxIterations 10`) → `Start`. Watch agents, tool calls, sources, and Q&A. `Stop` halts via `boardroomManager.stopBoardroom()`.
5. Logs: `/llamacpp-logs` page (`GET /api/server/logs`) + terminal stdout.

Other entry points (optional):

```powershell
python exec/mass_downloader.py        # full re-download (see §4 Option B)
python exec/view_lru_cache.py         # inspect persistent LRU cache
python exec/tests/test_finbert.py     # FinBERT accuracy smoke test
python exec/tests/test_finbert_vram.py# FinBERT VRAM benchmark (CUDA host)
python newtooltest.py                 # ad-hoc tool scratchpad (git-ignored)
```

---

## 9. Boardroom parameters (what to tune)

Set per run in each boardroom page (stored in `BoardroomConfig` subclasses — `boardroom/boardroom_config.py`):

- `temperature` (default `0.5`): LLM sampling temperature for `runConversation()`. Lower = more deterministic financials; higher = more diverse ideas.
- `thinkingBudget` (default `2048`): reasoning tokens. Mapped to llama.cpp (`thinking_budget_tokens`/`reasoning_effort` low ≤256 / medium ≤2048 / high above) or OpenRouter (`reasoning.max_tokens` + `thinking.budget_tokens`). Set `0` to disable thinking. Summariser uses a fixed 256 budget.
- `maxIterations` (default `10`): max tool-call rounds per agent before a forced final answer (`tool_choice="none"`).
- `boardroomPace`: `one_shot` / `fast` / `complete` / `preset` — how many agent stages/debate rounds run.
- Time horizons, sector/stock caps, allocation bias (1=max growth … 6=max defensive), rebalance amount (1=very light … 6=full overhaul) — see `TIME_HORIZON_INFO`, `ALLOCATION_BIAS_INFO`, `REBALANCE_AMOUNT_INFO` for exact prompt wording.

Model guidance: prefer large-context, tool-capable instruct models. If the UI flags `< 50k context`, expect failures on portfolio/agent sims. If a cloud model rejects `tool_choice="required"`, the client auto-retries with `"auto"`.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Llama.cpp executable not found` | Install/download `llama-server.exe` (§6.1) and set its full path in `/llamacpp-setup` or put it on `PATH`. |
| `GGUF model file not found` / `Invalid GGUF header` | Re-pick the file via `Add Model`. Path must end `.gguf` and start with `GGUF` magic bytes. |
| `boardroom server failed to start on port 9081` | Another `llama-server` is holding the port — the app tries `killRemainingLlamaCppProcesses()`, else kill it manually / reboot. Check `/llamacpp-logs`. |
| CUDA not used / `torch.cuda.is_available() == False` | You installed CPU-only torch, or no NVIDIA driver. Reinstall `requirements.txt` inside the venv on an NVIDIA host. CPU path still works (FinBERT #4/#5). |
| TensorRT load failed → falls back to ONNX/PyTorch | Normal on non-TensorRT hosts. To silence, delete/ignore `finbert/models/ModernFinBERT_fp8.engine` — the loader auto-falls back with a log line. |
| First run downloads HF model slowly | Normal — `tabularisai/ModernFinBERT` + tokenizer to `data/models/`. Keep that folder or restore it from the Drive zip. |
| OpenRouter `429 Too Many Requests` | Free-tier models limited to 20/min in-app with backoff (up to 8 retries). Wait, switch router, or use a paid model / local LLM. |
| `FINNHUB_API_KEY not found` / empty fundamentals | Add the key to `.env` and restart `main.py`. App continues without it, just with thinner data. Same for `FRED_API_KEY`, Alpaca keys. |
| `No models configured!` on Server page | Go to `Configure Llama.cpp` → `Add Model` → `Save Configuration` first. |
| Port `9091` in use | Stop the old `python main.py` / Flask instance. UI + WS ports are fixed (`UI_PORT=9091`, `WS_PORT=9092` in `exec/ui_app.py`). |
| `llm/server_config.json` corrupted | Delete it — the app recreates defaults (`lastSelectedProvider: llamacpp`) on next start. |

---

## 11. Project structure (map)

```
main.py                      # entry — exec.ui_app.startServer() (Flask 9091 + WS 9092)
.env                         # YOUR keys (git-ignored, see §3 template)
data/                        # local parquet universe (git-ignored, restore from Drive zip)
finbert/
  finbert_engines.py         # TRT-FP8 → ONNX-CUDA → PyTorch-CUDA → ONNX-CPU → PyTorch-CPU
  finbert_quantise.py        # optional: rebuild .onnx/.engine (CUDA+TRT host only)
  models/                    # ModernFinBERT_fp32.onnx, ModernFinBERT_fp8.engine (git-ignored)
llm/
  server_manager.py          # provider switch: llamacpp / openrouter / openaicompatible
  server_config_store.py     # llm/server_config.json persistence
  llamacpp/                  # args builder, process initiator (port 9081), client
  cloud/                     # OpenRouter + OpenAI-Compatible clients
  agents/                    # FinancialAgent + prompts
  llm_client.py              # tool-call loop, streaming, thinking budgets, retries
boardroom/                   # engines, FSM, manager, BoardroomConfig (pace/horizons/bias)
llmtools/                    # tool registry + functions (prices, filings, sentiment, macro, sector…)
dataquery/                   # live+local providers (Alpaca, Finnhub, FRED, EDGAR, cache/locks)
collectors/                  # mass downloaders + constants (dates, DATA_DIR, EDGAR identity)
simulation/                  # user/agent portfolio sim + backtest/chart APIs
ui/templates/                # landing_page, server_config (LLM picker), llamacpp_setup, boardroom_*
ui/static/                   # css/js
exec/
  ui_app.py                  # Flask + WebSocket wiring, routes
  mass_downloader.py         # full dataset rebuild
output/  _cache/             # runs + persistent LRU (git-ignored)
```

---

## 12. Notes for markers / new users

- Start with **Option A (Drive zip)** + **local llama.cpp** for a fully offline-markable demo after first setup; cloud keys only extend live fundamentals/pricing.
- Suggested first run: `Single Equity Rating` on `NVDA` with defaults — exercises prices, filings, news sentiment, and tool calling end-to-end.
- Cost/speed: local llama.cpp = free after download, speed scales with GPU offload (`-ngl`) and context (`-c`); OpenRouter = pay-per-token (see in-UI pricing per 1M) but zero local VRAM.
- Do not commit `.env`, `data/`, `output/`, `finbert/models/*.{engine,onnx}`, or `llm/server_config.json` — all are already in `.gitignore`.
