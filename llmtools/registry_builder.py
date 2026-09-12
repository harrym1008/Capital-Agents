import pandas as pd
from collectors.constants import NEW_YORK, UTC

from llmtools.tool_registry import ToolRegistry, Tool
from llmtools.functions.macro import fetchMacroContext, fetchMacroNews
from llmtools.functions.company import (
    fetchCompanyProfile, 
    fetchCompanyRecentNews, 
    fetchStockPricePerformance, 
    calculateDistFromCurrPrice,
    fetchFinnhubCompanyFundamentals,
    fetchBatchFinnhubMetrics
)
from llmtools.functions.edgar import (
    fetchCompanyValuationMetrics, 
    fetchIncomeStatement, 
    fetchBalanceSheet, 
    fetchCashFlowStatement, 
    fetchStatementOfEquity, 
    fetchComprehensiveIncomeStatement
)
from llmtools.functions.sentiment_news import (
    fetchTickerSentimentHistory, 
    fetchSentimentDivergence, 
    fetchMacroSentimentHistory
)
from llmtools.functions.sentiment_10q import fetchLatest10QSentiment
from llmtools.functions.sector import (
    fetchSectorPerformance, 
    fetchAllSectorRankings, 
    fetchSectorProfile, 
    fetchAllSectorsPerformance,
    fetchAllSectorProfiles,
    fetchStocksInSector,
    DB_SECTOR_TO_TICKER
)
from llmtools.functions.other import (
    executePythonCalculation, 
    transferToAgent
)
from llmtools.functions.confirmation import (
    confirmBoardroomDecisionImmediateTerm,
    confirmBoardroomDecisionShortTerm, 
    confirmBoardroomDecisionMediumTerm, 
    confirmBoardroomDecisionLongTerm, 
    confirmBoardroomDecisionDistantTerm,
    confirmSectorAllocation,
    confirmPortfolioAllocation
)

from llmtools.lru_cacher import startPrecacheThread


ALL_SECTORS_STRING = ", ".join([k for k in DB_SECTOR_TO_TICKER.keys()])
    
SCHEMAS = {
    "empty": {"type": "object", "properties": {}, "required": []},

    "confirmSectorAllocation": {
        "type": "object",
        "properties": {
            "sectorAllocations": {
                "type": "object",
                "description": "Mapping of sector names (or ETF tickers) to percentage numbers summing to 100% (e.g. {'information_technology': 35.0, 'health_care': 25.0, 'financials': 20.0, 'cash': 20.0})."
            },
            "rationale": {
                "type": "string",
                "description": "Clear executive rationale explaining the macro, cyclical, and risk justification for this sector weighting."
            }
        },
        "required": ["sectorAllocations", "rationale"]
    },

    "confirmPortfolioAllocation": {
        "type": "object",
        "properties": {
            "positions": {
                "type": "array",
                "description": "List of individual stock allocations for the portfolio.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. 'NVDA', 'AAPL')."},
                        "sector": {"type": "string", "description": "GICS sector of the stock."},
                        "weightPct": {"type": "number", "description": "Percentage allocation weight in the portfolio (e.g. 12.5)."},
                        "investmentRole": {
                            "type": "string",
                            "enum": ["CORE_GROWTH", "HIGH_BETA_UPSIDE", "DEFENSIVE_VALUE", "DIVIDEND_STABILITY", "CASH_BUFFER"],
                            "description": "Strategic portfolio role."
                        },
                        "rationale": {"type": "string", "description": "Concise rationale for including this stock."}
                    },
                    "required": ["ticker", "sector", "weightPct", "investmentRole"]
                }
            },
            "portfolioRationale": {
                "type": "string",
                "description": "Clear executive rationale explaining portfolio construction, risk management, and sector alignment."
            },
            "cashWeightPct": {
                "type": "number",
                "description": "Unallocated percentage held in cash (e.g. 5.0). Defaults to 0.0.",
                "default": 0.0
            }
        },
        "required": ["positions", "portfolioRationale"]
    },

    "fetchStocksInSector": {
        "type": "object",
        "properties": {
            "sector": {
                "type": "string",
                "description": "The GICS sector name or ETF ticker (e.g. 'information_technology', 'XLK', 'health_care', 'XLV'), selectable from: " + ALL_SECTORS_STRING + "."
            },
            "style": {
                "type": "string",
                "enum": ["growth", "value", "defensive", "all"],
                "description": "The investment style bias for screening candidates (growth, value, defensive, or all). Defaults to all.",
                "default": "all"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of candidate stocks to return (4 to 20). Defaults to 12.",
                "default": 12
            }
        },
        "required": ["sector"]
    },

    "sectorQuery": {
        "type": "object",
        "properties": {
            "sectorOrTicker": {
                "type": "string",
                "description": "The GICS sector name, selectable from this list: " + ALL_SECTORS_STRING + "."
            }
        },
        "required": ["sectorOrTicker"]
    },


    "sectorRanking": {
        "type": "object",
        "properties": {
            "lookback": {
                "type": "string",
                "enum": ["5d", "1mo", "3mo", "6mo", "12mo"],
                "description": "The lookback period for ranking sectors (5d, 1mo, 3mo, 6mo, 12mo). Defaults to 1mo.",
                "default": "1mo"
            }
        },
        "required": []
    },

    "macroNews": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "The maximum number of news stories to fetch (1-18). Defaults to 12 (use 12 where possible).",
                "default": 12
            }
        },
        "required": []
    },

    "justTicker": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol."
            }
        },
        "required": ["ticker"]
    },

    "batchTickers": {
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of stock ticker symbols to query (e.g. ['AAPL', 'MSFT', 'NVDA']). Maximum 15 tickers."
            }
        },
        "required": ["tickers"]
    },

    "tickerNews": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol."
            },
            "limit": {
                "type": "integer",
                "description": "The maximum number of news stories to fetch (1-18). Defaults to 12 (use 12 where possible).",
                "default": 12
            }
        },
        "required": ["ticker"]
    },

    "finStatement": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol."
            },
            "periodType": {
                "type": "string",
                "enum": ["annual", "quarterly"],
                "description": "Select annual statements or quarterly statements. Defaults to annual."
            }
        },
        "required": ["ticker"]
    },

    "confirmDecisionImmediate": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol."},
            "rating": {
                "type": "string",
                "enum": ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"],
                "description": "The final stock rating made by the boardroom (STRONG BUY/BUY/HOLD/SELL/STRONG SELL)."
            },
            "weighting": {
                "type": "string",
                "enum": ["UNDERWEIGHT", "EQUAL-WEIGHT", "OVERWEIGHT"],
                "description": "The final weighting assigned to the stock (UNDERWEIGHT/EQUAL-WEIGHT/OVERWEIGHT)."
            },
            "threeDayTarget": {"type": "number", "description": "The final 3-day target price for the stock."},
            "twoWeekTarget": {"type": "number", "description": "The final 2-week target price for the stock."}
        },
        "required": ["ticker", "rating", "weighting", "threeDayTarget", "twoWeekTarget"]
    },

    "confirmDecisionShortTerm": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol."},
            "rating": {
                "type": "string",
                "enum": ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"],
                "description": "The final stock rating made by the boardroom (STRONG BUY/BUY/HOLD/SELL/STRONG SELL)."
            },
            "weighting": {
                "type": "string",
                "enum": ["UNDERWEIGHT", "EQUAL-WEIGHT", "OVERWEIGHT"],
                "description": "The final weighting assigned to the stock (UNDERWEIGHT/EQUAL-WEIGHT/OVERWEIGHT)."
            },
            "oneMonthTarget": {"type": "number", "description": "The final 1-month target price for the stock."},
            "threeMonthTarget": {"type": "number", "description": "The final 3-month target price for the stock."}
        },
        "required": ["ticker", "rating", "weighting", "oneMonthTarget", "threeMonthTarget"]
    },

    "confirmDecisionMediumTerm": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol."},
            "rating": {
                "type": "string",
                "enum": ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"],
                "description": "The final stock rating made by the boardroom (STRONG BUY/BUY/HOLD/SELL/STRONG SELL)."
            },
            "weighting": {
                "type": "string",
                "enum": ["UNDERWEIGHT", "EQUAL-WEIGHT", "OVERWEIGHT"],
                "description": "The final weighting assigned to the stock (UNDERWEIGHT/EQUAL-WEIGHT/OVERWEIGHT)."
            },
            "threeMonthTarget": {"type": "number", "description": "The final 3-month target price for the stock."},
            "twelveMonthTarget": {"type": "number", "description": "The final 12-month (1-year) target price for the stock."}
        },
        "required": ["ticker", "rating", "weighting", "threeMonthTarget", "twelveMonthTarget"]
    },

    "confirmDecisionLongTerm": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol."},
            "rating": {
                "type": "string",
                "enum": ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"],
                "description": "The final stock rating made by the boardroom (STRONG BUY/BUY/HOLD/SELL/STRONG SELL)."
            },
            "weighting": {
                "type": "string",
                "enum": ["UNDERWEIGHT", "EQUAL-WEIGHT", "OVERWEIGHT"],
                "description": "The final weighting assigned to the stock (UNDERWEIGHT/EQUAL-WEIGHT/OVERWEIGHT)."
            },
            "twelveMonthTarget": {"type": "number", "description": "The final 12-month (1-year) target price for the stock."},
            "threeYearTarget": {"type": "number", "description": "The final 3-year (36-month) target price for the stock."}
        },
        "required": ["ticker", "rating", "weighting", "twelveMonthTarget", "threeYearTarget"]
    },

    "confirmDecisionDistantTerm": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol."},
            "rating": {
                "type": "string",
                "enum": ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"],
                "description": "The final stock rating made by the boardroom (STRONG BUY/BUY/HOLD/SELL/STRONG SELL)."
            },
            "weighting": {
                "type": "string",
                "enum": ["UNDERWEIGHT", "EQUAL-WEIGHT", "OVERWEIGHT"],
                "description": "The final weighting assigned to the stock (UNDERWEIGHT/EQUAL-WEIGHT/OVERWEIGHT)."
            },
            "threeYearTarget": {"type": "number", "description": "The final 3-year (36-month) target price for the stock."},
            "tenYearTarget": {"type": "number", "description": "The final 10-year target price for the stock."}
        },
        "required": ["ticker", "rating", "weighting", "threeYearTarget", "tenYearTarget"]
    },

    "stockPriceChange": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol."
            },
            "targetPrice": {
                "type": "number",
                "description": "The target stock price to compare against the current, most recent price."
            }
        },
        "required": ["ticker", "targetPrice"]
    },

    "pythonCode": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The exact Python code or mathematical expression to execute "
                               "(e.g. '15000 * (1 + 0.055)**10' or multi-line assignments with local variables). "
            }
        },
        "required": ["code"]
    },

    "transferToAgent": {
        "type": "object",
        "properties": {
            "agentRole": {
                "type": "string",
                "enum": [
                    "Macro Analyst",
                    "Bullish Value Analyst",
                    "Bearish Risk Analyst",
                    "Aggressive Risk Analyst",
                    "Conservative Risk Analyst",
                    "Impartial Portfolio Manager",
                    "One-Shot Analyst"
                ],
                "description": "The exact name of the specialist boardroom agent to transfer to."
            },
            "transferMessage": {
                "type": "string",
                "description": "A clear, concise instruction or summary of the question for the specialist agent to answer."
            }
        },
        "required": ["agentRole", "transferMessage"]
    }
}




def buildToolRegistry(initMacroThread=False):
    toolReg = ToolRegistry()

    # macro.py
    toolReg.registerTool(Tool(
        toolFunction=fetchMacroContext,
        toolName="fetchMacroContext",
        toolDescription="Fetches most recent values/prices for market indices, commodities, forex and other economic indicators from the Federal Reserve Economic Data database.",
        parameterSchema=SCHEMAS["empty"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchMacroNews,
        toolName="fetchMacroNews",
        toolDescription="Fetches the latest geopolitical and macroeconomic headlines and stories via Benzinga.",
        parameterSchema=SCHEMAS["macroNews"],
        storeIntoSources=True
    ))


    # sector.py
    toolReg.registerTool(Tool(
        toolFunction=fetchSectorPerformance,
        toolName="fetchSectorPerformance",
        toolDescription="Fetches trailing performance metrics (1d, 5d, 1mo, 3mo, 6mo, 12mo), technical indicators (RSI, 50/200 SMA), 52-week range, volatility, and relative alpha vs S&P 500 for a specific GICS sector or sector ETF.",
        parameterSchema=SCHEMAS["sectorQuery"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchAllSectorRankings,
        toolName="fetchAllSectorRankings",
        toolDescription="Fetches a ranked leaderboard of all 11 GICS sector ETFs by trailing performance over a specified lookback (5d, 1mo, 3mo, 6mo, 12mo) to evaluate sector rotation and leadership.",
        parameterSchema=SCHEMAS["sectorRanking"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchSectorProfile,
        toolName="fetchSectorProfile",
        toolDescription="Fetches the descriptive profile and industry categorization for a given GICS sector or sector ETF.",
        parameterSchema=SCHEMAS["sectorQuery"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchAllSectorsPerformance,
        toolName="fetchAllSectorsPerformance",
        toolDescription="Fetches trailing performance metrics, technicals, volatility, and relative alpha for ALL 11 GICS sector ETFs in a single consolidated list.",
        parameterSchema=SCHEMAS["empty"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchAllSectorProfiles,
        toolName="fetchAllSectorProfiles",
        toolDescription="Fetches descriptive profiles, categories, and ETF details for ALL 11 GICS sectors in a single consolidated list.",
        parameterSchema=SCHEMAS["empty"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=confirmSectorAllocation,
        toolName="confirmSectorAllocation",
        toolDescription="Confirms and records the executive sector allocation decisions (%-wise) for the portfolio.",
        parameterSchema=SCHEMAS["confirmSectorAllocation"],
        storeIntoSources=False
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchStocksInSector,
        toolName="fetchStocksInSector",
        toolDescription="Screens and returns curated equity candidates (10-15 stocks) within a specified GICS sector or sector ETF, with key performance metrics, market cap, and style indicators (growth, value, defensive, all).",
        parameterSchema=SCHEMAS["fetchStocksInSector"],
        storeIntoSources=True
    ))


    # company.py
    toolReg.registerTool(Tool(
        toolFunction=fetchCompanyProfile,
        toolName="fetchCompanyProfile",
        toolDescription="Fetches the company profile for a given stock ticker, detailing their name, industry and an *outdated* company summary.",
        parameterSchema=SCHEMAS["justTicker"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchCompanyRecentNews,
        toolName="fetchCompanyRecentNews",
        toolDescription="Fetches the latest Benzinga news articles for a given stock ticker.",
        parameterSchema=SCHEMAS["tickerNews"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchStockPricePerformance,
        toolName="fetchStockPricePerformance",
        toolDescription="Fetches stock price history for a company, and calculates various metrics including volatility, Sharpe ratio, RSI and others.",
        parameterSchema=SCHEMAS["justTicker"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=calculateDistFromCurrPrice,
        toolName="calculateDistFromCurrPrice",
        toolDescription="Calculates the percentage distance of a target price from the current stock price.",
        parameterSchema=SCHEMAS["stockPriceChange"],
        storeIntoSources=False
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchFinnhubCompanyFundamentals,
        toolName="fetchFinnhubCompanyFundamentals",
        toolDescription="Fast point-in-time financial metrics for an individual stock (P/E, P/B, margins, ROE, debt-to-equity, EPS, EBITDA) via Finnhub with local caching. Ideal for stock scouting without EDGAR filing overhead.",
        parameterSchema=SCHEMAS["justTicker"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchBatchFinnhubMetrics,
        toolName="fetchBatchFinnhubMetrics",
        toolDescription="Fast point-in-time valuation and profitability metrics for multiple candidate stocks in a single call (up to 15 tickers). Returns P/E, P/B, margins, ROE, and leverage for quick cross-stock comparison.",
        parameterSchema=SCHEMAS["batchTickers"],
        storeIntoSources=True
    ))


    # edgar.py
    toolReg.registerTool(Tool(
        toolFunction=fetchCompanyValuationMetrics,
        toolName="fetchCompanyValuationMetrics",
        toolDescription="Fetches market value, price ratios, margins and more financial metrics for a given ticker.",
        parameterSchema=SCHEMAS["justTicker"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchIncomeStatement,
        toolName="fetchIncomeStatement",
        toolDescription="Fetches the most recent income statement (annual or quarterly) for a given stock ticker.",
        parameterSchema=SCHEMAS["finStatement"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchBalanceSheet,
        toolName="fetchBalanceSheet",
        toolDescription="Fetches the most recent balance sheet (annual or quarterly) for a given stock ticker.",
        parameterSchema=SCHEMAS["finStatement"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchCashFlowStatement,
        toolName="fetchCashFlowStatement",
        toolDescription="Fetches the most recent cash flow statement (annual or quarterly) for a given stock ticker.",
        parameterSchema=SCHEMAS["finStatement"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchStatementOfEquity,
        toolName="fetchStatementOfEquity",
        toolDescription="Fetches the most recent statement of equity (annual or quarterly, if it exists) for a given stock ticker.",
        parameterSchema=SCHEMAS["finStatement"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchComprehensiveIncomeStatement,
        toolName="fetchComprehensiveIncomeStatement",
        toolDescription="Fetches the most recent comprehensive income statement (annual or quarterly, if it exists) for a given stock ticker.",
        parameterSchema=SCHEMAS["finStatement"],
        storeIntoSources=True
    ))


    # sentimentnews.py
    toolReg.registerTool(Tool(
        toolFunction=fetchTickerSentimentHistory,
        toolName="fetchTickerSentimentHistory",
        toolDescription="Fetches the history of the (estimated) news sentiment for a given stock ticker.",
        parameterSchema=SCHEMAS["justTicker"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchSentimentDivergence,
        toolName="fetchSentimentDivergence",
        toolDescription="Analyses the divergence between stock price performance and (estimated) news sentiment over a 3-month period.",
        parameterSchema=SCHEMAS["justTicker"],
        storeIntoSources=True
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchMacroSentimentHistory,
        toolName="fetchMacroSentimentHistory",
        toolDescription="Fetches the history of the (estimated) news sentiment for major macroeconomic and geopolitical topics.",
        parameterSchema=SCHEMAS["empty"],
        storeIntoSources=True
    ))


    # sentiment10q.py
    toolReg.registerTool(Tool(
        toolFunction=fetchLatest10QSentiment,
        toolName="fetchLatest10QSentiment",
        toolDescription="Fetches the most recent 10-Q filing for a given stock ticker, extracts the MD&A and Risk Factors sections, and estimates the operational sentiment score based on the text.",
        parameterSchema=SCHEMAS["justTicker"],
        storeIntoSources=True
    ))


    # other.py
    toolReg.registerTool(Tool(
        toolFunction=executePythonCalculation,
        toolName="executePythonCalculation",
        toolDescription="Executes concise mathematical formulas or short variable assignments (1-5 lines max) "
                        "in a secure Python sandbox with math and numpy enabled. "
                        "Do NOT pass complex scripts, functions, or loops. "
                        "Returns stdout and assigned variable values. "
                        "If an error occurs, returns the error message and hint.",
                        
        parameterSchema=SCHEMAS["pythonCode"],
        storeIntoSources=False
    ))

    toolReg.registerTool(Tool(
        toolFunction=confirmBoardroomDecisionImmediateTerm,
        toolName="confirmBoardroomDecisionImmediateTerm",
        toolDescription="Confirms the final stock rating, weighting, and 3-day & 2-week target prices for a stock "
                        "after the boardroom has produced its final consensus for an immediate-term horizon.",
        parameterSchema=SCHEMAS["confirmDecisionImmediate"],
        storeIntoSources=False
    ))

    toolReg.registerTool(Tool(
        toolFunction=confirmBoardroomDecisionShortTerm,
        toolName="confirmBoardroomDecisionShortTerm",
        toolDescription="Confirms the final stock rating, weighting, and 1-month & 3-month target prices for a stock "
                        "after the boardroom has produced its final consensus for a short-term horizon.",
        parameterSchema=SCHEMAS["confirmDecisionShortTerm"],
        storeIntoSources=False
    ))

    toolReg.registerTool(Tool(
        toolFunction=confirmBoardroomDecisionMediumTerm,
        toolName="confirmBoardroomDecisionMediumTerm",
        toolDescription="Confirms the final stock rating, weighting, and 3-month & 12-month target prices for a stock "
                        "after the boardroom has produced its final consensus for a medium-term horizon.",
        parameterSchema=SCHEMAS["confirmDecisionMediumTerm"],
        storeIntoSources=False
    ))

    toolReg.registerTool(Tool(
        toolFunction=confirmBoardroomDecisionLongTerm,
        toolName="confirmBoardroomDecisionLongTerm",
        toolDescription="Confirms the final stock rating, weighting, and 12-month & 3-year target prices for a stock "
                        "after the boardroom has produced its final consensus for a long-term horizon.",
        parameterSchema=SCHEMAS["confirmDecisionLongTerm"],
        storeIntoSources=False
    ))

    toolReg.registerTool(Tool(
        toolFunction=confirmBoardroomDecisionDistantTerm,
        toolName="confirmBoardroomDecisionDistantTerm",
        toolDescription="Confirms the final stock rating, weighting, and 3-year & 10-year target prices for a stock "
                        "after the boardroom has produced its final consensus for a distant-term horizon.",
        parameterSchema=SCHEMAS["confirmDecisionDistantTerm"],
        storeIntoSources=False
    ))

    toolReg.registerTool(Tool(
        toolFunction=confirmPortfolioAllocation,
        toolName="confirmPortfolioAllocation",
        toolDescription="Confirms and records the executive portfolio creation verdict with individual stock positions, % weightings, dollar amounts, and cash buffer.",
        parameterSchema=SCHEMAS["confirmPortfolioAllocation"],
        storeIntoSources=False
    ))

    toolReg.registerTool(Tool(
        toolFunction=transferToAgent,
        toolName="transferToAgent",
        toolDescription="Transfers the conversation to a specialist boardroom agent with a summary of the question for them to answer directly.",
        parameterSchema=SCHEMAS["transferToAgent"],
        storeIntoSources=False
    ))

    if initMacroThread:
        timestamp = (pd.Timestamp.today().normalize() + pd.Timedelta(hours=9)).tz_localize(NEW_YORK).tz_convert(UTC)
        startPrecacheThread(toolReg, timestamp, macroTools=True)
    
    return toolReg



