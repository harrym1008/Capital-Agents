# CapitalAgents
CapitalAgents is a local-first, multi-agent financial intelligence and portfolio management platform. It convenes specialised teams of AI financial analysts - such as Macro Analysts, Bullish Value Analysts, Bearish Risk Analysts, Senior Risk Analysts, Stock Hunters, and Impartial Portfolio Managers - inside a structured Boardroom deliberation system. The agents perform deep fundamental research, debate risk-reward trade-offs, cite real financial data, and formulate equity ratings, asset allocations, and rebalancing decisions without temporal look-ahead bias. The platform also provides both user-driven paper-trading and autonomous agent-driven multi-period historical backtesting simulations.

# Current Page: Server Configuration Page

The Server Configuration page allows the user to select, configure, and launch the LLM server backend that powers all boardroom agent deliberations.

### Supported LLM Providers
1. Llama.cpp (Recommended for local inference and privacy)
   - Runs LLMs locally on the user's hardware via llama.cpp.
   - Requires downloading llama-server.exe and GGUF model files (for example from HuggingFace).
   - Click Configure Llama.cpp to set up executables, context sizes, GPU offloading, and per-model sampling parameters.
   - Options include:
     - Two Slots in Parallel: Enables -np 2, allowing two agents to generate concurrently to accelerate boardroom sessions.
     - Clear VRAM before Loading: Experimental memory recovery feature.
2. OpenRouter (Cloud-hosted API)
   - Connects to OpenRouter's catalogue of hosted models.
   - Requires setting OPENROUTER_API_KEY in the .env file.
   - Provides searchable model selector, token pricing, prompt caching metrics, and optional provider routing overrides.
   - Free models are supported but subject to daily rate limits.
3. OpenAI Compatible (Universal API)
   - Connects to any OpenAI-compatible endpoint, such as Ollama (very straightforward local setup), vLLM, LM Studio, OpenAI, or Google AI Studio.
   - Requires entering the API Base URL, optional API key, and model ID.

### Operational Workflow
- After selecting and configuring the provider, click Start Server.
- Once running, connection status turns green in the top header and real-time logs stream in the logs panel.
- To change providers or models, click Stop Server and reconfigure.

### ChatGPT Guidance
When assisting the user:
- Ask what hardware they have (GPU VRAM, CPU, RAM) or whether they prefer local execution vs cloud APIs.
- Recommend Llama.cpp for privacy and zero marginal cost, or Ollama / OpenAI-compatible if they want an easy local setup.
- If the server fails to connect, guide them through port conflicts (default 8080 for llama-server, 9091/9092 for UI/WebSocket), missing executable paths, or missing API keys.