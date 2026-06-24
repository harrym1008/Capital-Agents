import os
import sys
import io
import math
import json
import requests
from typing import Any, Dict, List, Callable
from enum import Enum

import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from fredapi import Fred

from cli.ansi import ANSI

from dotenv import load_dotenv
load_dotenv()


class Tool:
    def __init__(self, toolFunction: Callable, toolName: str, toolDescription: str, parameterSchema: Dict[str, Any]):
        self.toolFunction = toolFunction
        self.toolName = toolName
        self.toolDescription = toolDescription
        self.parameterSchema = parameterSchema
        self.toolLog = []

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
        return self.toolFunction(self, **kwargs)



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


class NumberType(Enum):
    DOLLARS = 1
    DOLLARS_CHANGE = 2
    DECIMAL = 3
    DECIMAL_CHANGE = 4
    LARGE_DOLLARS = 5
    LARGE_DOLLARS_CHANGE = 6
    SMALL_DOLLARS = 7
    SMALL_DOLLARS_CHANGE = 8
    PERCENTAGE = 9
    PERCENTAGE_CHANGE = 10
    STOCK_PRICE = 11
    STOCK_PRICE_CHANGE = 12


def cleanNumber(value, numType: NumberType):
    if pd.isna(value):
        return "null"
    
    def stringifyNumber(value, sf=4, minDp=1, plusSign=False, leading="", trailing=""):
        if value == 0:
            return ("0" + "0" * (minDp - 1)) if minDp > 1 else "0"
        
        sign = "-" if value < 0 else ("+" if plusSign and value > 0 else "")
        value = abs(value)
        
        magnitude = math.floor(math.log10(value))
        decimals = max(sf - magnitude - 1, minDp)
        formatted = f"{value:.{decimals}f}"

        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return sign + leading + formatted + trailing
        
    def formatLargeDollars(value):
        absValue = abs(value)
        if absValue >= 1_000_000_000_000:
            return stringifyNumber(value / 1_000_000_000_000, sf=4, minDp=1, leading="$") + "tn"
        elif absValue >= 1_000_000_000:
            return stringifyNumber(value / 1_000_000_000, sf=4, minDp=1, leading="$") + "bn"
        elif absValue >= 1_000_000:
            return stringifyNumber(value / 1_000_000, sf=4, minDp=1, leading="$") + "mn"
        elif absValue >= 1_000:
            return stringifyNumber(value / 1_000, sf=4, minDp=1, leading="$") + "k"
        else:
            return stringifyNumber(value, sf=4, minDp=1, leading="$")


    match numType:
        case NumberType.DOLLARS:
            return stringifyNumber(value, sf=4, minDp=1, leading="$")
        case NumberType.DOLLARS_CHANGE:
            return stringifyNumber(value, sf=4, minDp=1, plusSign=True, leading="$")
        case NumberType.DECIMAL:
            return stringifyNumber(value, sf=3, minDp=1)
        case NumberType.DECIMAL_CHANGE:
            return stringifyNumber(value, sf=3, minDp=1, plusSign=True)
        case NumberType.SMALL_DOLLARS:
            return stringifyNumber(value, sf=4, minDp=2, leading="$")
        case NumberType.SMALL_DOLLARS_CHANGE:
            return stringifyNumber(value, sf=4, minDp=2, plusSign=True, leading="$")
        case NumberType.PERCENTAGE: 
            return stringifyNumber(value, sf=3, minDp=1, trailing="%")
        case NumberType.PERCENTAGE_CHANGE:
            return stringifyNumber(value, sf=3, minDp=1, plusSign=True, trailing="%")
        case NumberType.LARGE_DOLLARS:
            return formatLargeDollars(value)
        case NumberType.LARGE_DOLLARS_CHANGE:
            return formatLargeDollars(value) if value < 0 else "+" + formatLargeDollars(value)
        case NumberType.STOCK_PRICE:
            numStr = stringifyNumber(value, sf=4, minDp=2, leading="$")
            if "." not in numStr:
                numStr += ".00"
            elif len(numStr.split(".")[1]) == 1:
                numStr += "0"
            return numStr
            
        case NumberType.STOCK_PRICE_CHANGE:
            numStr = stringifyNumber(value, sf=4, minDp=2, plusSign=True, leading="$")
            if "." not in numStr:
                numStr += ".00"
            elif len(numStr.split(".")[1]) == 1:
                numStr += "0"
            return numStr
        case _:
            return str(value)


def fetchMacroIndicators(tool: Tool) -> Dict[str, Any]:
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
                    "currentValue":         cleanNumber(lastPrice, NumberType.STOCK_PRICE),
                    "oneMonthChangePct":    cleanNumber(priceChangePct, NumberType.PERCENTAGE_CHANGE),
                    "oneMonthHigh":         cleanNumber(highPrice, NumberType.STOCK_PRICE),
                    "oneMonthLow":          cleanNumber(lowPrice, NumberType.STOCK_PRICE)
                }
            else:
                macroSummary[metricName] = {"error": "Could not fetch data for this ticker."}

        return cleanData(macroSummary)
    except Exception as e:
        return {"error": f"An error occurred while fetching macro context: {str(e)}"}
    

def fetchMacroNews(tool: Tool, limit: int = 8) -> List[Dict[str, Any]]:
    try:
        if limit < 1 or limit > 25:
            return [{"error": "Limit must be between 1 and 25."}]
        
        apiKeyId = os.environ.get("ALPACA_API_KEY")
        apiKeySecret = os.environ.get("ALPACA_API_SECRET")

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": apiKeyId,
            "APCA-API-SECRET-KEY": apiKeySecret
        }

        macroSymbols = "SPY,QQQ,DIA,GLD,USO,TLT"  # ETFs for S&P 500, Nasdaq 100, Dow Jones, gold, oil, and 20Y+ treasury bonds
        url = (
            f"https://data.alpaca.markets/v1beta1/news"
            f"?sort=desc&symbols={macroSymbols}&limit={limit}"
            f"&include_content=true&exclude_contentless=true"
        )

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return [{"error": f"News API returned status code {response.status_code}: {response.text}"}]
        
        data = response.json()
        rawNews = data.get("news", [])

        nowUtc = pd.Timestamp.now(tz="UTC")

        def cleanHtml(htmlContent: str) -> str:
            if not htmlContent:
                return ""
            soup = BeautifulSoup(htmlContent, "html.parser")
            rawText = soup.get_text(separator=" ")
            return " ".join(rawText.split())

        cleanedStories = []
        for index, item in enumerate(rawNews):
            rawHeadline = item.get("headline", "")
            cleanHeadline = cleanHtml(rawHeadline)

            rawHtml = item.get("content", "")
            cleanContent = cleanHtml(rawHtml)

            publishTimestamp = item.get("created_at")
            storyAge = "unknown"

            if publishTimestamp:
                try:
                    pubDt = pd.Timestamp(publishTimestamp).tz_convert("UTC")
                    dayDiff = (nowUtc.date() - pubDt.date()).days
                    if dayDiff == 0:
                        storyAge = "today"
                    elif dayDiff == 1:
                        storyAge = "yesterday"
                    else:
                        storyAge = f"{dayDiff} days old"
                except Exception:
                    pass

            cleanedStories.append({
                "storyIndex": index,
                "headline": cleanHeadline,
                "publisher": item.get("source"),
                "storyAge": storyAge,
                "content": cleanContent
            })

        return cleanData(cleanedStories)
    except Exception as e:
        return [{"error": f"An error occurred while fetching macro news: {str(e)}"}]       



FRED_SERIES_MAP = {
    "CPI": {
        "id": "CPIAUCSL",  
        "desc": "Consumer Price Index", 
        "unit": "Index (1982-1984=100)",
        "numType": NumberType.DECIMAL
    },
    "CORECPI": {
        "id": "CPILFESL",
        "desc": "Core CPI (excl. food & energy)",
        "unit": "Index (1982-1984=100)",
        "numType": NumberType.DECIMAL
    },
    "UNEMPLOYMENT": {
        "id": "UNRATE",
        "desc": "Unemployment Rate",
        "unit": "%",
        "numType": NumberType.PERCENTAGE
    },
    "FEDFUNDS": {
        "id": "FEDFUNDS",
        "desc": "Federal Funds Rate",
        "unit": "%",
        "numType": NumberType.PERCENTAGE
    },
    "GDP": {
        "id": "GDP",
        "desc": "Gross Domestic Product",
        "unit": "$",
        "numType": NumberType.LARGE_DOLLARS
    },
    "TREAS_10Y": {
        "id": "DGS10",
        "desc": "10-Year Treasury Yield",
        "unit": "%",
        "numType": NumberType.PERCENTAGE
    },
    "TREAS_2Y": {
        "id": "DGS2",
        "desc": "2-Year Treasury Yield",
        "unit": "%",
        "numType": NumberType.PERCENTAGE
    },
    "TREAS_3MO": {
        "id": "DGS3MO",
        "desc": "3-Month Treasury Yield",
        "unit": "%",
        "numType": NumberType.PERCENTAGE
    }
}


def fetchMacroFredSeries(tool: Tool, period: str = "1y") -> Dict[str, Any]:
    try:
        fredClient = Fred(api_key=os.environ.get("FRED_API_KEY"))

        months = 12
        try:
            if "y" in period:
                months = int(period.replace("y", "")) * 12
            elif "mo" in period:
                months = int(period.replace("mo", ""))
            else:
                raise ValueError("Invalid period format.")            
        except Exception as e:
            return {"error": f"Could not parse period parameter: {period}. Only use provided formats."}

        if months < 1 or months > 120:
            return {"error": "Period must be between 1 month and 10 years (120 months)."}
        
        startDate = (pd.Timestamp.now(tz="America/New_York") - pd.DateOffset(months=months)).strftime("%Y-%m-%d")

        results = []
        for name, info in FRED_SERIES_MAP.items():
            output = {"name": name, "description": info["desc"], "unit": info["unit"], "records": []}
            try:
                series = fredClient.get_series(info["id"], observation_start=startDate)
                df = series.reset_index()
                df.columns = ["date", "value"]
                df["value"]  = df["value"].bfill().ffill()
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")

                if df.empty:
                    records = []
                else:
                    if name != "GDP":
                        dfMonthly = df.set_index("date").resample("MS").first().reset_index()
                    else:
                        dfMonthly = df.set_index("date").resample("QS").first().reset_index()       # GDP data is quarterly
                        dfMonthly["value"] *= 1_000_000_000   # Convert from billions to actual dollars
                    records = [
                        {"date": d.strftime("%Y-%m-%d"), "value": cleanNumber(v, info["numType"]) if pd.notna(v) else "unknown"}
                        for d, v in zip(dfMonthly["date"], dfMonthly["value"])
                    ]
                output["records"] = records

            except Exception as e:
                output["records"] = [{"error": f"Failed to fetch FRED series {name}: {str(e)}"}]
            results.append(output)

        return cleanData({
            "period": period,
            "startDate": startDate,
            "series": results
        })
    except Exception as e:
        return {"error": f"An error occurred while fetching FRED series: {str(e)}"}


def fetchCompanyProfile(tool: Tool, ticker: str) -> Dict[str, Any]:
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
            "ipoDate": tickerInfo.get("ipoDate"),
            "fullTimeEmployees": tickerInfo.get("fullTimeEmployees"),
            "website": tickerInfo.get("website"),
            "longBusinessSummary": tickerInfo.get("longBusinessSummary")
        }
        return cleanData(profileData)
    except Exception as e:
        return {"error": f"Failed to fetch company profile for {ticker}: {str(e)}"}
    

def fetchCompanyValuationMetrics(tool: Tool, ticker: str) -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        tickerInfo = tickerObj.info
        
        fundamentalData = {
            "marketCap":        cleanNumber(tickerInfo.get("marketCap"), NumberType.LARGE_DOLLARS),
            "enterpriseValue":  cleanNumber(tickerInfo.get("enterpriseValue"), NumberType.LARGE_DOLLARS),
            "trailingPE":       cleanNumber(tickerInfo.get("trailingPE"), NumberType.DECIMAL),
            "forwardPE":        cleanNumber(tickerInfo.get("forwardPE"), NumberType.DECIMAL),
            "pegRatio":         cleanNumber(tickerInfo.get("pegRatio"), NumberType.DECIMAL),
            "priceToBook":      cleanNumber(tickerInfo.get("priceToBook"), NumberType.DECIMAL),
            "beta":             cleanNumber(tickerInfo.get("beta"), NumberType.DECIMAL),
            "dividendYield":    cleanNumber(tickerInfo.get("dividendYield"), NumberType.PERCENTAGE),
            "profitMargins":    cleanNumber(tickerInfo.get("profitMargins") * 100, NumberType.PERCENTAGE),
            "ebitdaMargins":    cleanNumber(tickerInfo.get("ebitdaMargins") * 100, NumberType.PERCENTAGE),
            "operatingMargins": cleanNumber(tickerInfo.get("operatingMargins") * 100, NumberType.PERCENTAGE),
            "returnOnEquity":   cleanNumber(tickerInfo.get("returnOnEquity") * 100, NumberType.PERCENTAGE),
            "shortRatio":       cleanNumber(tickerInfo.get("shortRatio"), NumberType.DECIMAL)
        }
        return cleanData(fundamentalData)
    except Exception as e:
        return {"error": f"Failed to fetch key fundamentals for {ticker}: {str(e)}"}
    

def fetchIncomeStatement(tool: Tool, ticker: str, periodType: str = "annual") -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        incomeStatement = tickerObj.quarterly_income_stmt if periodType.lower() == "quarterly" else tickerObj.income_stmt

        if incomeStatement is None or incomeStatement.empty:
            return {"error": "No statement records found."}
        
        targetLabelsTypes = {
            "Total Revenue": NumberType.LARGE_DOLLARS,
            "Cost Of Revenue": NumberType.LARGE_DOLLARS,
            "Gross Profit": NumberType.LARGE_DOLLARS,
            "Operating Income": NumberType.LARGE_DOLLARS,
            "Net Income": NumberType.LARGE_DOLLARS,
            "EBITDA": NumberType.LARGE_DOLLARS,
            "Diluted EPS": NumberType.DECIMAL
        }
        filteredStatement = incomeStatement[incomeStatement.index.isin(targetLabelsTypes.keys())]        

        if filteredStatement.empty:
            return {"error": "No relevant statement records found."}
        
        formattedStatement = filteredStatement.astype(object)
        for label, numType in targetLabelsTypes.items():
            if label in filteredStatement.index:
                formattedStatement.loc[label] = formattedStatement.loc[label].apply(lambda x: cleanNumber(float(x), numType))

        return cleanData(formattedStatement)
    except Exception as e:
        return {"error": f"Failed to fetch income statement for {ticker}: {str(e)}"}
    

def fetchBalanceSheet(tool: Tool, ticker: str, periodType: str = "annual") -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        balanceSheet = tickerObj.quarterly_balance_sheet if periodType.lower() == "quarterly" else tickerObj.balance_sheet
        
        if balanceSheet is None or balanceSheet.empty:
            return {"error": "No statement records found."}
        
        targetLabelsTypes = {
            "Cash And Cash Equivalents": NumberType.LARGE_DOLLARS,
            "Total Assets": NumberType.LARGE_DOLLARS,
            "Total Liabilities Net Min Interest": NumberType.LARGE_DOLLARS,
            "Stockholders Equity": NumberType.LARGE_DOLLARS,
            "Total Debt": NumberType.LARGE_DOLLARS,
            "Net Debt": NumberType.LARGE_DOLLARS
        }
        filteredStatement = balanceSheet[balanceSheet.index.isin(targetLabelsTypes.keys())]

        if filteredStatement.empty:
            return {"error": "No relevant statement records found."}
        
        formattedStatement = filteredStatement.astype(object)
        for label, numType in targetLabelsTypes.items():
            if label in filteredStatement.index:
                formattedStatement.loc[label] = formattedStatement.loc[label].apply(lambda x: cleanNumber(float(x), numType))
        
        return cleanData(formattedStatement)
    except Exception as e:
        return {"error": f"Failed to fetch balance sheet for {ticker}: {str(e)}"}


def fetchCashFlowStatement(tool: Tool, ticker: str, periodType: str = "annual") -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        cashFlowStatement = tickerObj.quarterly_cashflow if periodType.lower() == "quarterly" else tickerObj.cashflow

        if cashFlowStatement is None or cashFlowStatement.empty:
            return {"error": "No statement records found."}
            
        targetLabelsTypes = {
            "Operating Cash Flow": NumberType.LARGE_DOLLARS,
            "Capital Expenditure": NumberType.LARGE_DOLLARS,
            "Free Cash Flow": NumberType.LARGE_DOLLARS,
            "Investing Cash Flow": NumberType.LARGE_DOLLARS,
            "Financing Cash Flow": NumberType.LARGE_DOLLARS
        }
        filteredStatement = cashFlowStatement[cashFlowStatement.index.isin(targetLabelsTypes.keys())]

        if filteredStatement.empty:
            return {"error": "No relevant statement records found."}
        
        formattedStatement = filteredStatement.astype(object)
        for label, numType in targetLabelsTypes.items():
            if label in filteredStatement.index:
                formattedStatement.loc[label] = formattedStatement.loc[label].apply(lambda x: cleanNumber(float(x), numType))

        return cleanData(formattedStatement)
    except Exception as e:
        return {"error": f"Failed to fetch cash flow statement for {ticker}: {str(e)}"}


def fetchStockPricePerformance(tool: Tool, ticker: str, period: str = "6mo") -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())

        period = period.lower()
        shortPeriods = ["5d", "1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y"]
        fetchPeriod = "1y" if period in shortPeriods else period

        historyFrame = tickerObj.history(period=fetchPeriod)
        
        if historyFrame.empty:
            return {"error": f"No historical prices found for {ticker} using period {period}."}
            
        lastClose = historyFrame["Close"].iloc[-1]
        endDate = historyFrame.index[-1]

        try:
            if period == "ytd":
                startDate = pd.Timestamp(year=endDate.year, month=1, day=1, tz=endDate.tz)
            elif "y" in period:
                years = int(period.replace("y", ""))
                startDate = endDate - pd.DateOffset(years=years)
            elif "mo" in period:
                months = int(period.replace("mo", ""))
                startDate = endDate - pd.DateOffset(months=months)
            elif "d" in period:
                days = int(period.replace("d", ""))
                startDate = endDate - pd.DateOffset(days=days)
            else:
                raise ValueError("Invalid period format.")            
        except Exception as e:
            return {"error": f"Could not parse period parameter: {period}. Only use provided formats."}
        
        ipoNotice = False
        firstAvailableDate = historyFrame.index[0]
        if startDate < firstAvailableDate:
            startDate = firstAvailableDate
            ipoNotice = True
            

        fiftyDaySma = historyFrame["Close"].rolling(window=min(50, len(historyFrame))).mean().iloc[-1]
        distanceFrom50DaySMA = lastClose - fiftyDaySma if not pd.isna(fiftyDaySma) else None
        twoHundredDaySma = historyFrame["Close"].rolling(window=min(200, len(historyFrame))).mean().iloc[-1]
        distanceFrom200DaySMA = lastClose - twoHundredDaySma if not pd.isna(twoHundredDaySma) else None

        endDate = historyFrame.index[-1]
        oneYearAgoDate = endDate - pd.DateOffset(years=1)
        frame52Week = historyFrame[oneYearAgoDate:]

        highPrice52W = frame52Week["High"].max()
        lowPrice52W = frame52Week["Low"].min()

        periodFrame = historyFrame.loc[startDate:]
        if periodFrame.empty:
            periodFrame = historyFrame

        startClose = periodFrame["Close"].iloc[0]
        if startClose == 0:
            startClose = 1e-10   # Avoid division by zero in case of erroneous data
        periodReturnPct = ((lastClose - startClose) / startClose) * 100
        
        
        performanceData = {
            "ticker": ticker.upper(),
            "period": period,
            "periodStartPrice":     cleanNumber(startClose, NumberType.STOCK_PRICE),
            "mostRecentPrice":      cleanNumber(lastClose, NumberType.STOCK_PRICE),
            "periodReturnPct":      cleanNumber(periodReturnPct, NumberType.PERCENTAGE_CHANGE),
            "50DaySMA":             cleanNumber(fiftyDaySma, NumberType.STOCK_PRICE) ,
            "distFrom50DaySMA":     cleanNumber(distanceFrom50DaySMA, NumberType.STOCK_PRICE_CHANGE),
            "200DaySMA":            cleanNumber(twoHundredDaySma, NumberType.STOCK_PRICE),
            "distFrom200DaySMA":    cleanNumber(distanceFrom200DaySMA, NumberType.STOCK_PRICE_CHANGE),
            "high52Week":           cleanNumber(highPrice52W, NumberType.STOCK_PRICE),
            "low52Week":            cleanNumber(lowPrice52W, NumberType.STOCK_PRICE)
        }
        if ipoNotice:
            performanceData["notice"] = (
                f"Requested period {period} exceeds available historical data. "
                f"Possible the company IPO'd during the requested period. "
                f"Using earliest available date {firstAvailableDate.date()} as start."
            )
            
        return cleanData(performanceData)
    
    except Exception as e:
        return {"error": f"Failed to evaluate price performance for {ticker}: {str(e)}"}



def fetchAnalystConsensus(tool: Tool, ticker: str) -> Dict[str, Any]:
    try:
        tickerObj = yf.Ticker(ticker.upper())
        tickerInfo = tickerObj.info
        
        priceTargets = {
            "targetHighPrice":  cleanNumber(tickerInfo.get("targetHighPrice"), NumberType.STOCK_PRICE),
            "targetLowPrice": cleanNumber(tickerInfo.get("targetLowPrice"), NumberType.STOCK_PRICE),
            "targetMeanPrice": cleanNumber(tickerInfo.get("targetMeanPrice"), NumberType.STOCK_PRICE),
            "targetMedianPrice": cleanNumber(tickerInfo.get("targetMedianPrice"), NumberType.STOCK_PRICE),
            "numberOfAnalystOpinions": tickerInfo.get("numberOfAnalystOpinions")
        }
        
        # Get historical recommendations breakdown table if available
        recommendationsBreakdown = {}
        numberOfAnalystRecommendations = 0
        try:
            recFrame = tickerObj.recommendations
            if recFrame is not None and not recFrame.empty:
                # Extract recent period count (usually row index 0 is modern estimates)
                firstRow = recFrame.iloc[0]
                recommendationsBreakdown = {
                    "strongBuy": firstRow.get("strongBuy", 0),
                    "buy": firstRow.get("buy", 0),
                    "hold": firstRow.get("hold", 0),
                    "sell": firstRow.get("sell", 0),
                    "strongSell": firstRow.get("strongSell", 0)
                }
                numberOfAnalystRecommendations = sum(recommendationsBreakdown.values())
                recommendationsBreakdown["totalRecommendations"] = numberOfAnalystRecommendations

        except Exception:
            pass
            
        consensusOutput = {
            "ticker": ticker.upper(),
            "priceTargets": priceTargets,
            "recommendations": recommendationsBreakdown
        }
        return cleanData(consensusOutput)
    except Exception as e:
        return {"error": f"Failed to fetch analyst expectations for {ticker}: {str(e)}"}
    

def fetchCompanyRecentNews(tool: Tool, ticker: str, limit: int = 6) -> List[Dict[str, Any]]:
    try:
        if limit < 1 or limit > 16:
            return [{"error": "Limit must be between 1 and 16."}] 
        
        apiKeyId = os.environ.get("ALPACA_API_KEY")
        apiKeySecret = os.environ.get("ALPACA_API_SECRET")

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": apiKeyId,
            "APCA-API-SECRET-KEY": apiKeySecret
        }

        tickerUpper = ticker.upper()
        url = (
            f"https://data.alpaca.markets/v1beta1/news"
            f"?sort=desc&symbols={tickerUpper}&limit={limit}"
            f"&include_content=true&exclude_contentless=false"
        )
        
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return [{"error": f"News API returned status code {response.status_code}: {response.text}"}]
        
        data = response.json()
        rawNews = data.get("news", [])

        nowUtc = pd.Timestamp.now(tz="UTC")

        def cleanHtml(htmlContent: str) -> str:
            if not htmlContent:
                return ""
            soup = BeautifulSoup(htmlContent, "html.parser")
            rawText = soup.get_text(separator=" ")
            return " ".join(rawText.split())

        cleanedStories = []
        for index, item in enumerate(rawNews):
            rawHeadline = item.get("headline", "")
            cleanHeadline = cleanHtml(rawHeadline)

            rawHtml = item.get("content", "")
            cleanContent = cleanHtml(rawHtml)

            publishTimestamp = item.get("created_at")
            storyAge = "unknown"

            if publishTimestamp:
                try:
                    pubDt = pd.Timestamp(publishTimestamp).tz_convert("UTC")
                    dayDiff = (nowUtc.date() - pubDt.date()).days
                    if dayDiff == 0:
                        storyAge = "today"
                    elif dayDiff == 1:
                        storyAge = "yesterday"
                    else:
                        storyAge = f"{dayDiff} days old"
                except Exception:
                    pass

            cleanedStories.append({
                "storyIndex": index,
                "headline": cleanHeadline,
                "publisher": item.get("source"),
                "storyAge": storyAge,
                "content": cleanContent
            })

        return cleanData(cleanedStories)
    except Exception as e:
        return [{"error": f"An error occurred while fetching company news for {ticker}: {str(e)}"}]
    


def executePythonCalculation(tool: Tool, code: str) -> Any:
    oldStdout = sys.stdout
    redirectedOutput = io.StringIO()
    sys.stdout = redirectedOutput

    # NOTE: __import__ is included so that LLM-generated code can use
    # standard "import math" / "import numpy as np" statements inside
    # the sandbox. Without it, any import statement raises ImportError.
    safeGlobals = {
        "__builtins__": {
            "__import__": __import__,
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
        "numpy": np,
        "np": np,
        "random": __import__("random"),
        "datetime": __import__("datetime"),
    }

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

            preloadedModules = {"math", "numpy", "np", "random", "datetime"}
            cleanedLocalScope = {
                k: cleanData(v)
                for k, v in localScope.items()
                if not k.startswith("_") and k not in preloadedModules
            }
            output = {
                "success": True,
                "variables": cleanedLocalScope,
                "stdout": capturedStdout
            }
            tool.toolLog.append(output)
            return output
        
    except Exception as e:
        return {
            "success": False,
            "error": f"{e.__class__.__name__}: {str(e)}"
        }
    finally:
        sys.stdout = oldStdout



def confirmBoardroomDecision(tool: Tool, ticker: str, rating: str, weighting: str, twelveMonthTarget: float, threeYearTarget: float) -> Dict[str, Any]:
    try:
        rating = rating.upper()
        weighting = weighting.upper()

        if rating not in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
            return {"error": f"Invalid rating value: {rating}. Must be one of STRONG BUY, BUY, HOLD, SELL, STRONG SELL."}
        if weighting not in ["UNDERWEIGHT", "EQUAL-WEIGHT", "OVERWEIGHT"]:
            return {"error": f"Invalid weighting value: {weighting}. Must be one of UNDERWEIGHT, EQUAL-WEIGHT, OVERWEIGHT."}

        finalDecisionStr = f"""{ANSI.BOLD}┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Completed boardroom decision confirmation of {(ticker+':'):<12}   ┃
┃                                                              ┃
┃        Rating:  {ANSI.ITALIC}{rating:<12}{ANSI.RESET}{ANSI.BOLD}                                 ┃
┃     Weighting:  {ANSI.ITALIC}{weighting:<12}{ANSI.RESET}{ANSI.BOLD}                                 ┃
┃  12-Mo Target:  {cleanNumber(twelveMonthTarget, NumberType.STOCK_PRICE):<12}                                 ┃
┃   3-Yr Target:  {cleanNumber(threeYearTarget, NumberType.STOCK_PRICE):<12}                                 ┃ 
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n{ANSI.RESET}"""
        
        result = {
            "ticker": ticker,
            "rating": rating,
            "weighting": weighting,
            "twelveMonthTarget": cleanNumber(twelveMonthTarget, NumberType.STOCK_PRICE),
            "threeYearTarget": cleanNumber(threeYearTarget, NumberType.STOCK_PRICE)
        }
        tool.toolLog.append(finalDecisionStr)

        return cleanData(result)
    except Exception as e:
        return {"error": f"An error occurred while confirming boardroom decision: {str(e)}"}



# Schema Generation

newsLimitSchema = {
    "type": "object",
    "properties": {
        "limit": {
            "type": "integer",
            "description": "The maximum number of news stories to fetch (1-25). Defaults to 8."
        }
    },
    "required": []
}

fredSeriesSchema = {
    "type": "object",
    "properties": {
        "period": {
            "type": "string",
            "description": "Lookback period, can be any number of months 'mo' or years 'y'. Examples '3mo', '1y', '5y'. Defaults to '1y'. Maximum of '10y' or '120mo' allowed.."
        }
    },
    "required": []
}

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

tickerNewsSchema = {
    "type": "object",
    "properties": {
        "ticker": {
            "type": "string",
            "description": "The target stock ticker symbol"
        },
        "limit": {
            "type": "integer",
            "description": "The maximum number of news stories to fetch (1-16). Defaults to 6."
        }
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
            "enum": ["5d", "1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y"],
            "description": "The lookback period."
        }
    },
    "required": ["ticker"]
}


confirmSchema = {
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
}



def buildToolsRegistry() -> List[Tool]:
    return [
        Tool(
            toolFunction=fetchMacroIndicators,
            toolName="fetchMacroIndicators",
            toolDescription="Fetch real-time values for market indicators (S&P 500, Nasdaq, Gold, 10-Yr Bond Yield, VIX) to identify macro market conditions.",
            parameterSchema={"type": "object", "properties": {}, "required": []}
        ),
        Tool(
            toolFunction=fetchMacroNews,
            toolName="fetchMacroNews",
            toolDescription="Fetch the latest geopolitical and macroeconomic headlines and summaries via Benzinga.",
            parameterSchema=newsLimitSchema
        ),
        Tool(
            toolFunction=fetchMacroFredSeries,
            toolName="fetchMacroFredSeries",
            toolDescription="Fetch historical, monthly macroeconomic data series (CPI, Unemployment, GDP, etc.) from the Federal Reserve Economic Data (FRED) database.",
            parameterSchema=fredSeriesSchema
        ),
        Tool(
            toolFunction=fetchCompanyProfile,
            toolName="fetchCompanyProfile",
            toolDescription="Get high-level qualitative metadata, industry classifications, business summary, and employee records.",
            parameterSchema=tickerSchema
        ),
        Tool(
            toolFunction=fetchCompanyValuationMetrics,
            toolName="fetchCompanyValuationMetrics",
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
            toolDescription="Fetch the latest news headlines and article content (if available) for a given stock ticker.",
            parameterSchema=tickerNewsSchema
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
        ),
        Tool(
            toolFunction=confirmBoardroomDecision,
            toolName="confirmBoardroomDecision",
            toolDescription="Confirm the final executive decision made by the boardroom after evaluating all available information and producing the final consensus.",
            parameterSchema=confirmSchema
        )
    ]



def jsonPrettyPrint(data: Any) -> str:
    try:
        print(json.dumps(data, indent=4, sort_keys=True))
    except Exception as e:
        print(f"Error formatting JSON: {str(e)}")


if __name__ == "__main__":
    jsonPrettyPrint(fetchCompanyRecentNews("SPCX"))