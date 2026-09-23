# CapitalAgents
CapitalAgents is a local-first, multi-agent financial intelligence and portfolio management platform. It convenes specialised teams of AI financial analysts - such as Macro Analysts, Bullish Value Analysts, Bearish Risk Analysts, Senior Risk Analysts, Stock Hunters, and Impartial Portfolio Managers - inside a structured Boardroom deliberation system. The agents perform deep fundamental research, debate risk-reward trade-offs, cite real financial data, and formulate equity ratings, asset allocations, and rebalancing decisions without temporal look-ahead bias. The platform also provides both user-driven paper-trading and autonomous agent-driven multi-period historical backtesting simulations.

# Current Page: Agent-Driven Portfolio Simulation Page

The Agent-Driven Portfolio Simulation page runs an autonomous, multi-period historical backtest. At recurring calendar intervals, the AI boardroom convenes to re-evaluate market conditions and manage an investment portfolio against real historical market data.

### Key Configuration Inputs
- Start Date & End Date: Historical simulation timeframe (between 2016 and 2026).
- Initial Capital ($): Starting cash allocation (default $100,000).
- Timestep: Frequency of boardroom review and rebalancing (such as 1 week, 2 weeks, 3 weeks, 1 month, 2 months, 3 months).
- Rebalance Amount (Slider Level 1 to 6): Controls how assertively the agents adjust holdings at each recurring timestep.
- Allocation Bias (Optional Slider): Dictates whether the agent team tilts towards Growth or Defensive opportunities over the simulation run.
- Sectors & Stocks Limits: Target number of sectors/stocks and maximum allocation caps.

### How the Simulation Operates
1. Initial Step (Step 0): The AI boardroom executes the Portfolio Creation pipeline as of the start date, selecting initial sectors and equities.
2. Recurring Timestep Progression: Time advances by the selected timestep.
3. Periodic Rebalancing: At each timestep, agents receive updated historical prices, corporate news, earnings reports, and macro data, then convene a Portfolio Rebalancing boardroom to generate trades.
4. Historical Execution: Trades execute at actual historical market prices, tracking cash, equity valuation, returns, drawdowns, and turnover.
5. Interactive Timeline: Users can browse any historical step on the timeline to inspect past boardroom debate transcripts, agent reasoning, trade orders, and performance curves against the S&P 500 benchmark.

### ChatGPT Guidance
When assisting the user:
- Advise on balancing timeframe and timestep (longer simulations with short timesteps require more LLM generations).
- Help them analyse backtest results: CAGR, Sharpe/Sortino ratios, max drawdown, and alpha generation against the S&P 500 benchmark.
- Explain why specific trades occurred during past market volatility or macroeconomic shifts.
