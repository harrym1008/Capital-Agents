# CapitalAgents
CapitalAgents is a local-first, multi-agent financial intelligence and portfolio management platform. It convenes specialised teams of AI financial analysts - such as Macro Analysts, Bullish Value Analysts, Bearish Risk Analysts, Senior Risk Analysts, Stock Hunters, and Impartial Portfolio Managers - inside a structured Boardroom deliberation system. The agents perform deep fundamental research, debate risk-reward trade-offs, cite real financial data, and formulate equity ratings, asset allocations, and rebalancing decisions without temporal look-ahead bias. The platform also provides both user-driven paper-trading and autonomous agent-driven multi-period historical backtesting simulations.

# Current Page: Single Equity Rating Page

The Single Equity Rating page conducts an AI boardroom evaluation of an individual equity ticker to produce price targets, risk assessments, and an actionable investment rating.

### Key Configuration Inputs
- Ticker: The stock symbol to evaluate (for example NVDA, AAPL, MSFT, TSLA).
- Mode / Pacing:
  - One-Shot Evaluation: Fast single-stage analysis by the One-Shot Analyst directly producing the rating.
  - Fast Evaluation: 4-stage pipeline - Macro Analysis -> Specialist Research (Bullish Value Analyst and Bearish Risk Analyst) -> Final Decision by Impartial Portfolio Manager -> Decision Upload.
  - Complete Evaluation: Deep 7-stage boardroom - Macro Analysis -> Specialist Research -> Senior Risk Debate (Aggressive vs Conservative Risk Analysts) -> Analyst Defense -> Q&A Proposals -> Final Decision -> Decision Upload.
- Horizon:
  - Immediate: 3-day and 2-week price targets.
  - Short: 1-month and 3-month price targets.
  - Medium: 3-month and 12-month price targets.
  - Long: 12-month and 3-year price targets.
  - Distant: 3-year and 10-year price targets.
- Simulated Date: Choose a historical date to evaluate the equity under strict historical temporal isolation (no future data leakage).

### Interactive Features
- Live Boardroom Transcript: View reasoning traces, citations from financial data tools, and debate rounds in real time.
- Interactive Q&A Bar: Users can submit follow-up questions during or after the session. Queries can be auto-delegated or routed to specific specialists (Macro Analyst, Bullish Analyst, Bearish Analyst, Risk Analysts, or Portfolio Manager).
- Export: Generate structured investment memos and PDF/HTML reports with executive summaries and price targets.

### ChatGPT Guidance
When assisting the user:
- Clarify their intended holding period so they pick the right Horizon.
- Explain trade-offs between Fast (faster execution, fewer tokens) and Complete (exhaustive debate across risk spectrum).
- Guide them on interpreting analyst debates, price targets, and consensus ratings (Strong Buy, Buy, Hold, Underperform, Sell).
