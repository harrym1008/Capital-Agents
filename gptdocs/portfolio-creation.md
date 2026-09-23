# CapitalAgents
CapitalAgents is a local-first, multi-agent financial intelligence and portfolio management platform. It convenes specialised teams of AI financial analysts - such as Macro Analysts, Bullish Value Analysts, Bearish Risk Analysts, Senior Risk Analysts, Stock Hunters, and Impartial Portfolio Managers - inside a structured Boardroom deliberation system. The agents perform deep fundamental research, debate risk-reward trade-offs, cite real financial data, and formulate equity ratings, asset allocations, and rebalancing decisions without temporal look-ahead bias. The platform also provides both user-driven paper-trading and autonomous agent-driven multi-period historical backtesting simulations.

# Current Page: Portfolio Creation Page

The Portfolio Creation page convenes the AI boardroom to construct a fresh equity portfolio from the ground up, starting from cash and distributing capital across sectors and vetted equities.

### Key Configuration Inputs
- Initial Capital ($): Starting cash to deploy (for example $100,000).
- Target Sector Count: Number of distinct GICS sectors to include (1 to 11, or leave empty for dynamic AI selection).
- Max Sector %: Hard ceiling on allocation to any single sector (20% to 80%, default 40%).
- Target Stock Count: Total number of individual stock holdings (1 to 30, or leave empty for dynamic AI selection).
- Max Stock %: Hard ceiling on allocation to any single equity (10% to 60%, default 20%).
- Allocation Bias (Optional Slider):
  - Level 1 - 3 (Growth Bias): Skews sector and stock selection towards high-beta, innovative, revenue-expanding growth companies.
  - Level 4 - 6 (Defensive Bias): Skews selection towards low-beta, dividend-yielding, capital-preserving defensive equities.
  - Unchecked: Neutral, balanced risk-return strategy.
- Mode:
  - Fast Evaluation: Streamlined macro analysis and stock hunting.
  - Complete Evaluation: Full multi-agent debate and risk review.
  - Pre-set Sector Allocation: Opens the Sector Allocation Modal, enabling the user to manually set exact percentage weights across all 11 GICS sectors with interactive sliders and a real-time pie chart.
- Time Horizon: Investment duration (from 1 month up to 20 years).

### Boardroom Deliberation Workflow
1. Macro Regime Analysis: Macro Analyst evaluates the broader economic cycle, interest rate trajectory, and inflation trends.
2. Sector Weighting: Determines sector overweights/underweights (or applies user presets).
3. Stock Hunting: Growth Stock Hunter and Value/Defensive Stock Hunter scout candidate equities from the S&P 500 universe fitting the chosen sectors.
4. Portfolio Synthesis: Impartial Portfolio Manager calculates final dollar allocations, share counts, and risk weightings.

### ChatGPT Guidance
When assisting the user:
- Help them set sensible diversification constraints (such as max 30-40% in any single sector, 10-20% max per stock).
- Clarify how time horizon and allocation bias influence agent recommendations.
- Explain the output table containing ticker weights, share purchases, and macro justifications.
