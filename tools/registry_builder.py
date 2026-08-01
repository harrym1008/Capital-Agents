from tools.tool_registry import ToolRegistry, Tool
from tools.functions.macro import fetchMacroContext, fetchMacroNews
from tools.functions.company import fetchCompanyProfile, fetchCompanyRecentNews, fetchStockPricePerformance, calculateDistFromCurrPrice
from tools.functions.edgar import fetchCompanyValuationMetrics, fetchIncomeStatement, fetchBalanceSheet, \
                                      fetchCashFlowStatement, fetchStatementOfEquity, fetchComprehensiveIncomeStatement 
from tools.functions.sentiment import fetchTickerSentimentHistory, fetchSentimentDivergence, fetchMacroSentimentHistory
from tools.functions.other import executePythonCalculation, confirmBoardroomDecision


SCHEMAS = {
    "empty": {"type": "object", "properties": {}, "required": []},

    "macroNews": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "The maximum number of news stories to fetch (1-18). Defaults to 12.",
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

    "tickerNews": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol."
            },
            "limit": {
                "type": "integer",
                "description": "The maximum number of news stories to fetch (1-18). Defaults to 12.",
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

    "confirmDecision": {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol."
            },
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
            "twelveMonthTarget": {
                "type": "number",
                "description": "The final 12-month (1-year) target price for the stock."
            },
            "threeYearTarget": {
                "type": "number",
                "description": "The final 3-year (36-month) target price for the stock."
            }
        },
        "required": ["ticker", "rating", "weighting", "twelveMonthTarget", "threeYearTarget"]
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
    }    
}




def buildToolRegistry():
    toolReg = ToolRegistry()

    # macro.py
    toolReg.registerTool(Tool(
        toolFunction=fetchMacroContext,
        toolName="fetchMacroContext",
        toolDescription="Fetches most recent values/prices for market indices, commodities, forex and other economic indicators from the Federal Reserve Economic Data database.",
        parameterSchema=SCHEMAS["empty"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchMacroNews,
        toolName="fetchMacroNews",
        toolDescription="Fetches the latest geopolitical and macroeconomic headlines and stories via Benzinga.",
        parameterSchema=SCHEMAS["macroNews"]
    ))


    # company.py
    toolReg.registerTool(Tool(
        toolFunction=fetchCompanyProfile,
        toolName="fetchCompanyProfile",
        toolDescription="Fetches the company profile for a given stock ticker, detailing their name, industry and an *outdated* company summary.",
        parameterSchema=SCHEMAS["justTicker"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchCompanyRecentNews,
        toolName="fetchCompanyRecentNews",
        toolDescription="Fetches the latest Benzinga news articles for a given stock ticker.",
        parameterSchema=SCHEMAS["tickerNews"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchStockPricePerformance,
        toolName="fetchStockPricePerformance",
        toolDescription="Fetches stock price history for a company, and calculates various metrics including volatility, Sharpe ratio, RSI and others.",
        parameterSchema=SCHEMAS["justTicker"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=calculateDistFromCurrPrice,
        toolName="calculateDistFromCurrPrice",
        toolDescription="Calculates the percentage distance of a target price from the current stock price.",
        parameterSchema=SCHEMAS["stockPriceChange"]
    ))


    # edgar.py
    toolReg.registerTool(Tool(
        toolFunction=fetchCompanyValuationMetrics,
        toolName="fetchCompanyValuationMetrics",
        toolDescription="Fetches market value, price ratios, margins and more financial metrics for a given ticker.",
        parameterSchema=SCHEMAS["justTicker"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchIncomeStatement,
        toolName="fetchIncomeStatement",
        toolDescription="Fetches the most recent income statement (annual or quarterly) for a given stock ticker.",
        parameterSchema=SCHEMAS["finStatement"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchBalanceSheet,
        toolName="fetchBalanceSheet",
        toolDescription="Fetches the most recent balance sheet (annual or quarterly) for a given stock ticker.",
        parameterSchema=SCHEMAS["finStatement"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchCashFlowStatement,
        toolName="fetchCashFlowStatement",
        toolDescription="Fetches the most recent cash flow statement (annual or quarterly) for a given stock ticker.",
        parameterSchema=SCHEMAS["finStatement"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchStatementOfEquity,
        toolName="fetchStatementOfEquity",
        toolDescription="Fetches the most recent statement of equity (annual or quarterly, if it exists) for a given stock ticker.",
        parameterSchema=SCHEMAS["finStatement"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchComprehensiveIncomeStatement,
        toolName="fetchComprehensiveIncomeStatement",
        toolDescription="Fetches the most recent comprehensive income statement (annual or quarterly, if it exists) for a given stock ticker.",
        parameterSchema=SCHEMAS["finStatement"]
    ))


    # sentiment.py
    toolReg.registerTool(Tool(
        toolFunction=fetchTickerSentimentHistory,
        toolName="fetchTickerSentimentHistory",
        toolDescription="Fetches the history of the news sentiment for a given stock ticker.",
        parameterSchema=SCHEMAS["justTicker"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchSentimentDivergence,
        toolName="fetchSentimentDivergence",
        toolDescription="Analyzes the divergence between stock price performance and news sentiment over a 3-month period.",
        parameterSchema=SCHEMAS["justTicker"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=fetchMacroSentimentHistory,
        toolName="fetchMacroSentimentHistory",
        toolDescription="Fetches the history of the news sentiment for major macroeconomic and geopolitical topics.",
        parameterSchema=SCHEMAS["empty"]
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
                        
        parameterSchema=SCHEMAS["pythonCode"]
    ))

    toolReg.registerTool(Tool(
        toolFunction=confirmBoardroomDecision,
        toolName="confirmBoardroomDecision",
        toolDescription="Confirms the final stock rating, weighting, and target prices for a given stock ticker "
                        "after the boardroom has produced its final consensus.",
        parameterSchema=SCHEMAS["confirmDecision"]
    ))

    return toolReg



