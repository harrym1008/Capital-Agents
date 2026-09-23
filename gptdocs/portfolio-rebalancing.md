# CapitalAgents
CapitalAgents is a local-first, multi-agent financial intelligence and portfolio management platform. It convenes specialised teams of AI financial analysts - such as Macro Analysts, Bullish Value Analysts, Bearish Risk Analysts, Senior Risk Analysts, Stock Hunters, and Impartial Portfolio Managers - inside a structured Boardroom deliberation system. The agents perform deep fundamental research, debate risk-reward trade-offs, cite real financial data, and formulate equity ratings, asset allocations, and rebalancing decisions without temporal look-ahead bias. The platform also provides both user-driven paper-trading and autonomous agent-driven multi-period historical backtesting simulations.

# Current Page: Portfolio Rebalancing Page

The Portfolio Rebalancing page uses the AI boardroom to audit an existing equity portfolio and propose strategic capital reallocations, trimming laggards, re-weighting sectors, and rotating into high-conviction opportunities.

### Key Configuration Inputs
- Portfolio Setup Button: Opens the Current Portfolio Holdings Modal. Users can manually enter or import existing stock holdings (Ticker, Holding Value in $, Sector, Industry). The total portfolio value and sector breakdown are calculated automatically.
- Rebalance Amount (Slider Level 1 to 6):
  - Level 1: Very Light Rebalance: Preserves ~90%+ of existing holdings; only subtle trims or exits if severe deterioration occurs.
  - Level 2: Mild Rebalance: Preserves the core portfolio, trimming lower-conviction holdings for modest new entries.
  - Level 3: Moderate Rebalance: Balanced strategy; maintains 60-70% core holdings while rotating capital into higher-alpha opportunities.
  - Level 4: Substantial Rebalance: Rotates 50%+ of capital away from stagnant or headwind-facing positions.
  - Level 5: Aggressive Rebalance: Reconstructs 70-80%+ of baseline exposure.
  - Level 6: Maximum Rebalance: Full overhaul from the ground up, prioritising forward alpha regardless of existing holdings.
- Allocation Bias (Optional Slider): Tilt the rebalanced portfolio towards Growth (high-beta, momentum) or Defensive (low-beta, dividends, capital preservation).
- Target Sectors / Stocks & Caps: Set maximum sector and stock concentration limits.
- Mode & Time Horizon: Fast vs Complete evaluation, and investment timeframe (1 month to 20 years).

### Rebalancing Output & Recommendations
- Portfolio Audit: Highlights structural risks, over-concentrations, and vulnerability to macroeconomic shifts.
- Actionable Execution Plan: Explicit BUY, SELL, TRIM, and HOLD recommendations with exact dollar shifts and percentage changes.
- Before-vs-After Comparison: Comparative breakdown of sector exposures and risk profiles.

### ChatGPT Guidance
When assisting the user:
- Guide them on selecting the appropriate Rebalance Amount (such as using Level 1-2 for lower turnover, Level 5-6 for major structural adjustments).
- Help them interpret the proposed buy/sell trades and explain why specific existing holdings were trimmed or replaced.
