# CapitalAgents
CapitalAgents is a local-first, multi-agent financial intelligence and portfolio management platform. It convenes specialised teams of AI financial analysts - such as Macro Analysts, Bullish Value Analysts, Bearish Risk Analysts, Senior Risk Analysts, Stock Hunters, and Impartial Portfolio Managers - inside a structured Boardroom deliberation system. The agents perform deep fundamental research, debate risk-reward trade-offs, cite real financial data, and formulate equity ratings, asset allocations, and rebalancing decisions without temporal look-ahead bias. The platform also provides both user-driven paper-trading and autonomous agent-driven multi-period historical backtesting simulations.

# Current Page: Llamacpp Setup Page

In this page, the user configures local llama.cpp execution settings and manages GGUF models. Always click Save Configuration after making adjustments.

### Global Arguments (Left Column)
- Llama Server Executable: Path to llama-server.exe (can be detected from system PATH or browsed directly).
- Recommended Command-Line Arguments:
  - --flash-attn : on (Enables flash attention, reducing memory and speeding up processing).
  - --cache-type-k / --cache-type-v : q8_0 (Quantised KV cache to conserve VRAM).
  - --jinja : EMPTY (Enables Jinja chat templates for model prompt formatting).
  - --load-mode : dio (Direct I/O, reducing RAM usage during model loading).
  - --metrics : EMPTY (Enables telemetry shown in the top header).
  - -np : 2 (Runs 2 parallel slots so multiple boardroom agents can analyse simultaneously).
  - --ctx-size : 131072 (Context window; set to number of slots multiplied by 65536 for long financial analyses).
  - --cache-ram : 4096 (Allocates 4GB system RAM for prompt checkpoint caching).
  - -b : 2048 and -ub : 512 (Prompt processing batch sizes).
  - --reasoning : on (Enables chain-of-thought reasoning mode).

### Models & Sampling Parameters (Right Column)
- Click Add Model to load .gguf files downloaded from Hugging Face.
- Set a friendly Model Alias for UI recognition.
- Sampling Parameters: Set temperature (recommended 0.3 - 0.7 for financial analysis), top_p, and top_k based on the model creator's model card.
- Per-Model Arguments: Supply custom CLI overrides for specific models (such as --fit or --n-gpu-layers).

### Hardware-Based Model Recommendations (Gemma 4 Series)
Quantisation-Aware Trained (QAT) models offer superior financial reasoning accuracy:
- Low VRAM or CPU-only: unsloth/gemma-4-E2B-it-qat-GGUF
- 8 GB VRAM: unsloth/gemma-4-E4B-it-qat-GGUF
- 12 - 16 GB VRAM: unsloth/gemma-4-12B-it-qat-GGUF
- >20 GB VRAM: unsloth/gemma-4-26B-A4B-it-qat-GGUF (Mixture of Experts; very fast on 24GB GPUs. Can run on 8-16GB with --fit and --fit-target 2500 to leave room for sentiment models).

### Troubleshooting
- If experiencing crashes or unexpected inference stalls, recommend testing an earlier stable release of llama.cpp (tested stable as of September 2026).