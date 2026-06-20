import yfinance as yf
import pandas as pd
import numpy as np
import sys
import io
import math
from typing import Any, Dict, List, Callable


class Tool:
    def __init__(self, toolFunction: Callable, toolName: str, toolDescription: str, parameterSchema: Dict[str, Any]):
        self.toolFunction = toolFunction
        self.toolName = toolName
        self.toolDescription = toolDescription
        self.parameterSchema = parameterSchema

    def getToolSchema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.toolName,
                "description": self.toolDescription,
                "parameters": self.parameterSchema
            }
        }
    
    def executeTool(self, **kwargs):
        return self.toolFunction(**kwargs)



def cleanKey(key):
    if hasattr(key, "strftime"):
        return key.strftime("%Y-%m-%d")
    return str(key).strip()


def cleanData(value):
    if isinstance(value, dict):
        return {cleanKey(k): cleanData(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [cleanData(item) for item in value]
    elif hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    elif isinstance(value, (int, np.integer, float, np.floating)):
        if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
            return None
        if hasattr(value, "item"):
            return value.item()
        return value 
    elif isinstance(value, pd.Series):
        return {cleanKey(k): cleanData(v) for k, v in value.items()}
    elif isinstance(value, pd.DataFrame):
        outputDict = {}
        for colName in value.columns:
            cleanedColName = str(colName)
            outputDict[cleanedColName] = {cleanKey(idx): cleanData(val) for idx, val in value[colName].items()}
        return outputDict
    return value


def fetchMacroContext() -> Dict[str, Any]:
    try:
        macroTickers = {
            "S&P 500": "^GSPC",
            "Nasdaq 100": "^NDX",
            "Gold": "GC=F",
            "WTI Crude Oil": "CL=F",
            "10-Yr US Treasury Yield": "^TNX",
            "CBOE Volatility Index": "^VIX"
        }
        macroSummary = {}
        for metricName, ticker in macroTickers.items():
            tickerObj = yf.Ticker(ticker)
            historyFrame = tickerObj.history(period="1mo").dropna(subset=["Open", "High", "Low", "Close"])

            if not historyFrame.empty:
                lastPrice = historyFrame["Close"].iloc[-1]
                firstPrice = historyFrame["Close"].iloc[0]
                priceChangePct = ((lastPrice - firstPrice) / firstPrice) * 100
                highPrice = historyFrame["High"].max()
                lowPrice = historyFrame["Low"].min()
                
                macroSummary[metricName] = {
                    "ticker": ticker,
                    "currentValue": round(lastPrice, 2),
                    "oneMonthChangePct": f"{round(priceChangePct, 2)}%",
                    "oneMonthHigh": round(highPrice, 2),
                    "oneMonthLow": round(lowPrice, 2)
                }
            else:
                macroSummary[metricName] = {"error": "Could not fetch data for this ticker."}

        return cleanData(macroSummary)
    except Exception as e:
        return {"error": f"An error occurred while fetching macro context: {str(e)}"}
    

def fetchCompanyProfile(ticker: str) -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        tickerInfo = tickerObj.info
        
        profileData = {
            "ticker": ticker.upper(),
            "exchange": tickerInfo.get("exchange"),
            "currency": tickerInfo.get("currency"),
            "shortName": tickerInfo.get("shortName"),
            "longName": tickerInfo.get("longName"),
            "sector": tickerInfo.get("sector"),
            "industry": tickerInfo.get("industry"),
            "country": tickerInfo.get("country"),
            "fullTimeEmployees": tickerInfo.get("fullTimeEmployees"),
            "website": tickerInfo.get("website"),
            "longBusinessSummary": tickerInfo.get("longBusinessSummary")
        }
        return cleanData(profileData)
    except Exception as e:
        return {"error": f"Failed to fetch company profile for {ticker}: {str(e)}"}
    

def fetchStockKeyFundamentals(ticker: str) -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        tickerInfo = tickerObj.info
        
        fundamentalData = {
            "marketCap": tickerInfo.get("marketCap"),
            "enterpriseValue": tickerInfo.get("enterpriseValue"),
            "trailingPE": tickerInfo.get("trailingPE"),
            "forwardPE": tickerInfo.get("forwardPE"),
            "pegRatio": tickerInfo.get("pegRatio"),
            "priceToBook": tickerInfo.get("priceToBook"),
            "beta": tickerInfo.get("beta"),
            "dividendYield": tickerInfo.get("dividendYield"),
            "profitMargins": tickerInfo.get("profitMargins"),
            "ebitdaMargins": tickerInfo.get("ebitdaMargins"),
            "operatingMargins": tickerInfo.get("operatingMargins"),
            "returnOnEquity": tickerInfo.get("returnOnEquity"),
            "shortRatio": tickerInfo.get("shortRatio")
        }
        return cleanData(fundamentalData)
    except Exception as e:
        return {"error": f"Failed to fetch key fundamentals for {ticker}: {str(e)}"}
    

def fetchIncomeStatement(ticker: str, periodType: str = "annual") -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        incomeStatement = tickerObj.quarterly_income_stmt if periodType.lower() == "quarterly" else tickerObj.income_stmt

        if incomeStatement is None or incomeStatement.empty:
            return {"error": "No statement records found."}
        
        targetLabels = ["Total Revenue", "Cost Of Revenue", "Gross Profit", "Operating Income", "Net Income", "EBITDA", "Diluted EPS"]
        filteredStatement = incomeStatement[incomeStatement.index.isin(targetLabels)]

        if filteredStatement.empty:
            return {"error": "No relevant statement records found."}
        
        return cleanData(filteredStatement)
    except Exception as e:
        return {"error": f"Failed to fetch income statement for {ticker}: {str(e)}"}
    

def fetchBalanceSheet(ticker: str, periodType: str = "annual") -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        balanceSheet = tickerObj.quarterly_balance_sheet if periodType.lower() == "quarterly" else tickerObj.balance_sheet
        
        if balanceSheet is None or balanceSheet.empty:
            return {"error": "No statement records found."}
        
        targetLabels = ["Cash And Cash Equivalents", "Total Assets", "Total Liabilities Net Min Interest", 
                        "Stockholders Equity", "Total Debt", "Net Debt"]
        filteredStatement = balanceSheet[balanceSheet.index.isin(targetLabels)]
        
        if filteredStatement.empty:
            return {"error": "No relevant statement records found."}
        
        return cleanData(filteredStatement)
    except Exception as e:
        return {"error": f"Failed to fetch balance sheet for {ticker}: {str(e)}"}


def fetchCashFlowStatement(ticker: str, periodType: str = "annual") -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        cashFlowStatement = tickerObj.quarterly_cashflow if periodType.lower() == "quarterly" else tickerObj.cashflow

        if cashFlowStatement is None or cashFlowStatement.empty:
            return {"error": "No statement records found."}
            
        targetLabels = ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow", 
                        "Investing Cash Flow", "Financing Cash Flow"]
        filteredStatement = cashFlowStatement[cashFlowStatement.index.isin(targetLabels)]

        if filteredStatement.empty:
            return {"error": "No relevant statement records found."}
        
        return cleanData(filteredStatement)
    except Exception as e:
        return {"error": f"Failed to fetch cash flow statement for {ticker}: {str(e)}"}


def fetchStockPricePerformance(ticker: str, period: str = "6mo") -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        historyFrame = tickerObj.history(period=period)
        
        if historyFrame.empty:
            return {"error": f"No historical prices found for {ticker} using period {period}."}
            
        lastClose = historyFrame["Close"].iloc[-1]
        startClose = historyFrame["Close"].iloc[0]
        periodReturnPct = ((lastClose - startClose) / startClose) * 100
        
        fiftyDaySma = historyFrame["Close"].rolling(window=min(50, len(historyFrame))).mean().iloc[-1]
        twoHundredDaySma = historyFrame["Close"].rolling(window=min(200, len(historyFrame))).mean().iloc[-1]
        
        highPrice52W = historyFrame["High"].max()
        lowPrice52W = historyFrame["Low"].min()
        
        performanceData = {
            "ticker": ticker.upper(),
            "period": period,
            "currentClose": round(lastClose, 2),
            "periodReturnPct": f"{round(periodReturnPct, 2)}%",
            "fiftyDaySMA": round(fiftyDaySma, 2) if not pd.isna(fiftyDaySma) else None,
            "twoHundredDaySMA": round(twoHundredDaySma, 2) if not pd.isna(twoHundredDaySma) else None,
            "high52Week": round(highPrice52W, 2),
            "low52Week": round(lowPrice52W, 2)
        }
        return cleanData(performanceData)
    except Exception as e:
        return {"error": f"Failed to evaluate price performance for {ticker}: {str(e)}"}



def fetchAnalystConsensus(ticker: str) -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        tickerInfo = tickerObj.info
        
        priceTargets = {
            "targetHighPrice": tickerInfo.get("targetHighPrice"),
            "targetLowPrice": tickerInfo.get("targetLowPrice"),
            "targetMeanPrice": tickerInfo.get("targetMeanPrice"),
            "targetMedianPrice": tickerInfo.get("targetMedianPrice"),
            "recommendationMean": tickerInfo.get("recommendationMean"),
            "recommendationKey": tickerInfo.get("recommendationKey"),
            "numberOfAnalystOpinions": tickerInfo.get("numberOfAnalystOpinions")
        }
        
        # Get historical recommendations breakdown table if available
        recommendationsBreakdown = {}
        try:
            recFrame = tickerObj.recommendations
            if recFrame is not None and not recFrame.empty:
                # Extract recent period count (usually row index 0 is modern estimates)
                firstRow = recFrame.iloc[0]
                recommendationsBreakdown = {
                    "period": firstRow.get("period", "Current"),
                    "strongBuy": firstRow.get("strongBuy", 0),
                    "buy": firstRow.get("buy", 0),
                    "hold": firstRow.get("hold", 0),
                    "sell": firstRow.get("sell", 0),
                    "strongSell": firstRow.get("strongSell", 0)
                }
        except Exception:
            pass # Breakdown not crucial if key targets exist
            
        consensusOutput = {
            "ticker": ticker.upper(),
            "priceTargets": priceTargets,
            "recommendationBreakdown": recommendationsBreakdown
        }
        return cleanData(consensusOutput)
    except Exception as e:
        return {"error": f"Failed to fetch analyst expectations for {ticker}: {str(e)}"}
    

def fetchCompanyRecentNews(ticker: str) -> List[Dict[str, Any]]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        rawNews = tickerObj.news
        
        if not rawNews:
            return []
            
        cleanedStories = []
        for index, item in enumerate(rawNews[:6]):
            # yfinance news schema can be flat or occasionally nested inside a content payload
            title = item.get("title") or item.get("content", {}).get("title", "No Headline")
            publisher = item.get("publisher") or item.get("content", {}).get("provider", {}).get("fn", "Unknown")
            link = item.get("link") or item.get("content", {}).get("clickThroughUrl", {}).get("url")
            publishTime = item.get("providerPublishTime") or item.get("content", {}).get("pubDate")
            
            cleanedStories.append({
                "storyIndex": index,
                "headline": title,
                "publisher": publisher,
                "sourceLink": link,
                "publishTimestamp": publishTime
            })
            
        return cleanData(cleanedStories)
    except Exception as e:
        return [{"error": f"Failed to scrape stock news for {ticker}: {str(e)}"}]
    


def executePythonCalculation(code: str) -> Any:
    oldStdout = sys.stdout
    redirectedOutput = io.StringIO()
    sys.stdout = redirectedOutput

    safeGlobals = {
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "chr": chr,
            "dict": dict,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "hash": hash,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
        },
        "math": math,
    }

    try:
        import numpy as np
        safeGlobals["numpy"] = np
        safeGlobals["np"] = np
    except ImportError:
        pass

    try:
        strippedCode = code.strip()
        try:
            resultValue = eval(strippedCode, safeGlobals)
            capturedStdout = redirectedOutput.getvalue()
            return {
                "success": True,
                "result": cleanData(resultValue),
                "stdout": capturedStdout
            }
        except SyntaxError:     # The code contains statements, it is not just a pure expression
            localScope = {}
            exec(strippedCode, safeGlobals, localScope)
            capturedStdout = redirectedOutput.getvalue()

            cleanedLocalScope = {k: cleanData(v) for k, v in localScope.items() if not k.startswith("_")}
            return {
                "success": True,
                "variables": cleanedLocalScope,
                "stdout": capturedStdout
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"{e.__class__.__name__}: {str(e)}"
        }
    finally:
        sys.stdout = oldStdout


# Schema Generation

tickerSchema = {
    "type": "object",
    "properties": {
        "ticker": {
            "type": "string",
            "description": "The target stock ticker symbol"
        },
    },
    "required": ["ticker"]
}

statementSchema = {
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
}

stockPriceSchema = {
    "type": "object",
    "properties": {
        "ticker": {
            "type": "string",
            "description": "The stock ticker symbol."
        },
        "period": {
            "type": "string",
            "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            "description": "The lookback period. Defaults to 6mo."
        }
    },
    "required": ["ticker"]
}


def buildToolsRegistry() -> List[Tool]:
    return [
        Tool(
            toolFunction=fetchMacroContext,
            toolName="fetchMacroContext",
            toolDescription="Fetch real-time indicators for market indices (S&P 500, Nasdaq, Gold, 10-Yr Bond Yield, VIX) to identify macro market conditions.",
            parameterSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            toolFunction=fetchCompanyProfile,
            toolName="fetchCompanyProfile",
            toolDescription="Get high-level qualitative metadata, industry classifications, business summary, and employee records.",
            parameterSchema=tickerSchema
        ),
        Tool(
            toolFunction=fetchStockKeyFundamentals,
            toolName="fetchStockKeyFundamentals",
            toolDescription="Fetch valuations, margins, PE multiples, PEG ratios, Enterprise Value, and short interest parameters.",
            parameterSchema=tickerSchema
        ),
        Tool(
            toolFunction=fetchIncomeStatement,
            toolName="fetchIncomeStatement",
            toolDescription="Fetch data from the latest income statement for a given stock ticker, including revenue, gross profit, operating income, net income, and EPS.",
            parameterSchema=statementSchema
        ),
        Tool(
            toolFunction=fetchBalanceSheet,
            toolName="fetchBalanceSheet",
            toolDescription="Fetch data from the latest balance sheet for a given stock ticker, including total assets, liabilities, equity, and cash positions.",
            parameterSchema=statementSchema
        ),
        Tool(
            toolFunction=fetchCashFlowStatement,
            toolName="fetchCashFlowStatement",
            toolDescription="Fetch data from the latest cash flow statement for a given stock ticker, including operating cash flow, capital expenditures, and free cash flow.",
            parameterSchema=statementSchema
        ),
        Tool(
            toolFunction=fetchStockPricePerformance,
            toolName="fetchStockPricePerformance",
            toolDescription="Fetch historical stock price performance for a given ticker over a specified period, including returns, moving averages, and 52-week high/low.",
            parameterSchema=stockPriceSchema
        ),
        Tool(
            toolFunction=fetchAnalystConsensus,
            toolName="fetchAnalystConsensus",
            toolDescription="Fetch average target projections and current consensus buy/hold/sell rankings from analysts.",
            parameterSchema=tickerSchema
        ),
        Tool(
            toolFunction=fetchCompanyRecentNews,
            toolName="fetchCompanyRecentNews",
            toolDescription="Fetch the latest news headlines and publisher information for a given stock ticker.",
            parameterSchema=tickerSchema
        ),
        Tool(
            toolFunction=executePythonCalculation,
            toolName="executePythonCalculation",
            toolDescription="Executes standard mathematical formulas, statistics, multi-line assignments, or algorithms in a secure Python sandbox with math and numpy enabled.",
            parameterSchema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The exact Python code or mathematical expression to execute (e.g. '15000 * (1 + 0.055)**10' or multi-line assignments with local variables)."
                    }
                },
                "required": ["code"]
            }
        )
    ]