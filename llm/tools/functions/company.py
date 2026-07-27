import os
import requests

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred

from dotenv import load_dotenv
load_dotenv()

from dataquery.ticker_provider import CompanyProfile
from dataquery.macro_provider import MacroSeries
from collectors.constants import IPO_BEFORE_START_DATE

from llm.tools.tool_registry import DataProviders, Tool
from llm.tools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, isLocalDataAvailable, NumberType 


def fetchCompanyProfile(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str):
    useLocal = isLocalDataAvailable(timestamp)

    tickerProfile: CompanyProfile = data.tickers.getTickerProfile(ticker)
    if tickerProfile is None:
        return {"error": f"No profile found for ticker {ticker}"}

    if not useLocal:
        key = f"company|profile_{ticker}_{timestamp.strftime('%Y-%m-%dH%H')}"
        cached = data.cache.get(key)
        if cached is not None:
            yfInfo = cached
        else:
            # Fetch profile from yfinance
            data.rateLimiters.yFinanceLimiter.wait()
            yfTicker = yf.Ticker(ticker)
            yfInfo = yfTicker.info
            data.cache.put(key, yfInfo)
            
        tickerProfile.summary = yfInfo.get("longBusinessSummary") or tickerProfile.summary

    profileDict = {
        "ticker": tickerProfile.ticker,
        "exchange": "NASDAQ" if tickerProfile.exchange == "XNAS" else "NYSE",
        "name": tickerProfile.name,
        "currency": "USD",
        "ipoDate": cleanKey(tickerProfile.ipoDate),
        "sector": tickerProfile.sector,
        "industry": tickerProfile.industry,
        "website": tickerProfile.website,
        "summary": tickerProfile.summary,
        "summaryNote": "Company profile could be highly outdated! Do not use for investment decisions."
    }

    if tickerProfile.ipoDate <= IPO_BEFORE_START_DATE:
        profileDict["ipoDate"] = "pre-2016"

    if tickerProfile.delistDate is None or tickerProfile.delistDate > timestamp:
        pass
    else:
        if tickerProfile.delistDate < timestamp:
            profileDict["isDelisted"] = True 

    return cleanData(profileDict)
    

def fetchCompanyRecentNews(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, limit: int = 12):
    useLocal = isLocalDataAvailable(timestamp)
    limit = min(max(limit, 1), 50)

    jsonResult = []

    if useLocal:
        newsWithContent = data.news.getRecentNewsForTicker(ticker, before=timestamp, limit=limit, mustHaveContent=True)

        idx = 0
        if not newsWithContent.empty:
            for _, row in newsWithContent.iterrows():
                headline = row.get("headline", "").strip()
                if not headline:
                    continue
                else:
                    headline = cleanHtmlContent(headline)

                content = row.get("content", "").strip()
                if not content:
                    content = "Article has no content available."
                else:
                    content = cleanHtmlContent(content)

                author = row.get("author", "").strip()

                articleTimestamp = row["updated_at"].replace(tzinfo=timestamp.tzinfo)
                age = timestamp - articleTimestamp
                if age < pd.Timedelta(hours=1):
                    ageStr = f"{age.components.minutes}m old"
                if age < pd.Timedelta(days=1):
                    ageStr = f"{age.components.hours}h {age.components.minutes}m old"
                else:
                    ageStr = f"{age.components.days}d old"

                articleId = str(row.get("id", ""))
                year = articleTimestamp.year % 100
                month = articleTimestamp.month
                url = f"https://www.benzinga.com/news/{year:02d}/{month:02d}/{articleId}"

                jsonResult.append({
                    "index": idx,
                    "headline": headline,
                    "content": content,
                    "author": author,
                    "articleAge": ageStr,
                    "url": url
                })
                idx += 1
        else:
            jsonResult.append({"error": "No local news data available."})

    else:
        # Retrieve news from the Alpaca API
        apiKeyId = os.getenv("ALPACA_API_KEY")
        apiKeySecret = os.getenv("ALPACA_API_SECRET")

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": apiKeyId,
            "APCA-API-SECRET-KEY": apiKeySecret
        }

        url = "https://data.alpaca.markets/v1beta1/news"

        params = {
            "start": (timestamp - pd.Timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%S-04:00"),
            "end": timestamp.strftime("%Y-%m-%dT%H:%M:%S-04:00"),
            "sort": "desc",
            "symbols": ticker,
            "limit": limit,
            "include_content": "true",
            "exclude_contentless": "true",
        }

        try:
            # Check if it is in the cache
            cacheKey = f"alpaca|news_{ticker}_{timestamp.strftime('%Y-%m-%dH%H')}_{limit}"
            cached = data.cache.get(cacheKey)
            if cached is not None:
                rawNews = cached
            else:
                data.rateLimiters.alpacaLimiter.wait()
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                rawNews = response.json().get("news", [])
                data.cache.put(cacheKey, rawNews)

        except requests.exceptions.RequestException as e:
            jsonResult.append({"error": f"Error fetching news from Alpaca API: {str(e)}"})
            return cleanData(jsonResult)
        
        for idx, article in enumerate(rawNews):
            headline = article.get("headline", "").strip()
            if not headline:
                continue
            else:
                headline = cleanHtmlContent(headline)

            content = article.get("content", "").strip()
            if not content:
                content = "Article has no content available."       # Should not occur due to exclude_contentless=true
            else:
                content = cleanHtmlContent(content)

            author = article.get("author", "").strip()

            articleTimestamp = pd.Timestamp(article.get("updated_at")).tz_convert(timestamp.tzinfo)
            age = timestamp - articleTimestamp
            if age < pd.Timedelta(hours=1):
                ageStr = f"{age.components.minutes}m old"
            elif age < pd.Timedelta(days=1):
                ageStr = f"{age.components.hours}h {age.components.minutes}m old"
            else:
                ageStr = f"{age.components.days}d old"

            url = article.get("url", "")

            jsonResult.append({
                "index": idx,
                "headline": headline,
                "content": content,
                "author": author,
                "articleAge": ageStr,
                "url": url
            }) 

    if len(jsonResult) < limit:
        jsonResult.append({"info": f"Only {len(jsonResult)} articles could be retrieved."})

    return {"date": timestamp.strftime("%Y-%m-%d"), "news": cleanData(jsonResult)}


def calculateRsi(priceData: pd.DataFrame, period: int = 14) -> float:
    closes = priceData["close"]

    deltas = closes.diff()
    gains = deltas.clip(lower=0)
    losses = -1 * deltas.clip(upper=0)

    # Wilders smoothing
    avgGain = gains.ewm(alpha=1/period, adjust=False).mean()
    avgLoss = losses.ewm(alpha=1/period, adjust=False).mean()

    rs = avgGain / avgLoss
    rsiSeries = 100 - (100 / (1 + rs))

    rsiSeries = rsiSeries.replace(np.inf, 100)
    return float(rsiSeries.iloc[-1])


def calculateMacd(priceData: pd.DataFrame, fast: int, slow: int, signal: int):
    close = priceData["close"]

    fastEma = close.ewm(span=fast, adjust=False).mean()
    slowEma = close.ewm(span=slow, adjust=False).mean()

    macd = fastEma - slowEma
    signalLine = macd.ewm(span=signal, adjust=False).mean()

    histogram = macd - signalLine
    return {
        "macd": macd.iloc[-1],
        "signal": signalLine.iloc[-1],
        "histogram": histogram.iloc[-1]
    }


def calculateSharpeRatio(priceData: pd.DataFrame, treasuryData: pd.DataFrame):
    priceDf = priceData.copy()
    treasuryDf = treasuryData.copy()

    if priceDf["date"].dt.tz is not None:
        priceDf["date"] = priceDf["date"].dt.tz_localize(None)

    if treasuryDf["date"].dt.tz is not None:
        treasuryDf["date"] = treasuryDf["date"].dt.tz_localize(None)

    treasuryDf = treasuryDf.rename(columns={"value": "treasuryRate"})
    mergedData = pd.merge(priceDf, treasuryDf, on="date", how="inner")

    returns = mergedData["close"].pct_change()
    dailyRiskFree = (1 + mergedData["treasuryRate"] / 100) ** (1 / 252) - 1
    excessReturns = returns - dailyRiskFree

    sharpeRatio = excessReturns.mean() / excessReturns.std() * np.sqrt(252)
    return sharpeRatio


def downloadTreasuryData(data: DataProviders, startDate: pd.Timestamp, endDate: pd.Timestamp):
    fredClient = Fred(api_key=os.getenv("FRED_API_KEY"))

    data.rateLimiters.fredLimiter.wait()
    df = fredClient.get_series("DGS2", observation_start=startDate.strftime("%Y-%m-%d"), observation_end=endDate.strftime("%Y-%m-%d"))
    df = df.reset_index()
    df.columns = ["date", "value"]
    df.dropna(subset=["value"], inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")

    return df


def findSmaCrossovers(priceData: pd.DataFrame, maxLookbackDays: int = 503, maxCrossovers: int = 8):     # 503 trading days is 2 years +/- 1 day
    df = priceData.copy()

    df["sma50"] = df["close"].rolling(window=50).mean()
    df["sma200"] = df["close"].rolling(window=200).mean()

    # 50 SMA was <= 200 SMA yesterday, but > today
    goldenCondition = (df["sma50"] > df["sma200"]) & (df["sma50"].shift(1) <= df["sma200"].shift(1))

    # 50 SMA was >= 200 SMA yesterday, but < today
    deathCondition = (df["sma50"] < df["sma200"]) & (df["sma50"].shift(1) >= df["sma200"].shift(1))

    df["crossType"] = None
    df.loc[goldenCondition, "crossType"] = "goldenCross"
    df.loc[deathCondition, "crossType"] = "deathCross"

    crossoversDf = df[df["crossType"].notna()].copy().tail(maxLookbackDays)
    crossovers = []

    for _, row in crossoversDf.iterrows():
        crossovers.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "type": row["crossType"],
            "price": cleanNumber(row["close"], NumberType.STOCK_PRICE)
        })
    return crossovers[-maxCrossovers:] 


def fetchStockPricePerformance(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str):
    useLocal = isLocalDataAvailable(timestamp)
    startDate = timestamp - pd.DateOffset(years=5, weeks=1)

    # Check if it is in the cache
    cacheKey = f"stockPerf|{ticker}_{timestamp.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    priceData: pd.DataFrame = None
    
    if useLocal:
        priceData = data.ohlcv.getPeriodDailyTickerData(ticker, startDate=startDate, endDate=timestamp)

        # Adjust for splits        
        finalSplitFactor = priceData["splitFactor"].iloc[-1]
        for col in ["open", "high", "low", "close", "vwap"]:
            priceData[col] = priceData[col] * (priceData["splitFactor"] / finalSplitFactor)
        priceData["splitFactor"] = finalSplitFactor

        priceData.drop(columns=["volume", "vwap", "splitFactor", "corpActionToday", "outstandingShares", "marketCap"], inplace=True)
        priceData = priceData.dropna().reset_index(drop=True)

        if priceData.empty:
            return {"error": "No local price data available."}
    else:
        # Get via yfinance
        data.rateLimiters.yFinanceLimiter.wait()
        yfTicker = yf.Ticker(ticker)
        priceData = yfTicker.history(
            start=startDate, 
            end=timestamp, 
            interval="1d",
            auto_adjust=True,
            actions=False
        )

        tickerInfo = yfTicker.info
        mostRecentPrice = float(
            tickerInfo.get("regularMarketOpen")
            or tickerInfo.get("regularMarketPrice")
        )       # Sometimes the most recent day of trading is all NaN, update the most recent day

        priceData.drop(columns=["Volume"], inplace=True)
        priceData = priceData.dropna().reset_index()
        priceData.columns = priceData.columns.str.lower()
        priceData["date"] = priceData["date"].dt.tz_convert("UTC")
        priceData = priceData[["date", "open", "high", "low", "close"]]
        priceData.loc[priceData.index[-1], "close"] = mostRecentPrice

        if priceData.empty:
            return {"error": "No price data available from yfinance."}

    # Both are in the exact same format now
    mostRecentPrice = priceData["close"].iloc[-1]
    earliestDate = priceData["date"].iloc[0]    

    # Calculate simple moving averages
    sma10Day = priceData["close"].rolling(window=min(10, len(priceData))).mean().iloc[-1]
    sma20Day = priceData["close"].rolling(window=min(20, len(priceData))).mean().iloc[-1]
    sma50Day = priceData["close"].rolling(window=min(50, len(priceData))).mean().iloc[-1]
    sma200Day = priceData["close"].rolling(window=min(200, len(priceData))).mean().iloc[-1]

    # 52W data
    fiftyTwoWeekAgo = timestamp - pd.DateOffset(weeks=52)
    priceData52W = priceData[priceData["date"] >= fiftyTwoWeekAgo]
    highPrice52W = priceData52W["high"].max()
    lowPrice52W = priceData52W["low"].min()

    # Calculate RSI
    rsi7Day = calculateRsi(priceData, period=7)
    rsi14Day = calculateRsi(priceData, period=14)
    rsi30Day = calculateRsi(priceData, period=30)

    # Calculate 30d volatility
    pctReturns = priceData["close"].pct_change(fill_method=None).dropna()
    volatility30d = pctReturns.tail(30).std() * np.sqrt(252)

    # Calculate max drawdown
    runningMax = priceData["close"].cummax()
    maxDrawdown = (priceData["close"] / runningMax - 1).min()

    # Calculate MACD
    macdData = calculateMacd(priceData, fast=12, slow=26, signal=9)

    # Calculate Sharpe ratio
    if useLocal:
        treasuryData = data.macro.getSeries(MacroSeries.TREAS_2Y, startDate=startDate, endDate=timestamp)
    else:
        treasuryData = downloadTreasuryData(data, startDate, timestamp)
    sharpeRatio = calculateSharpeRatio(priceData, treasuryData)

    # Calculate periodic returns
    periods = {
        "5d": timestamp - pd.DateOffset(days=5),
        "1m": timestamp - pd.DateOffset(months=1),
        "3m": timestamp - pd.DateOffset(months=3),
        "6m": timestamp - pd.DateOffset(months=6),
        "1y": timestamp - pd.DateOffset(years=1),
        "3y": timestamp - pd.DateOffset(years=3),
        "5y": timestamp - pd.DateOffset(years=5)
    }
    pctReturns = {}
    for periodName, periodStart in periods.items():
        if periodStart < earliestDate:
            continue
        periodData = priceData[priceData["date"] >= periodStart]
        if not periodData.empty:
            startPrice = periodData["close"].iloc[0]
            endPrice = periodData["close"].iloc[-1]
            pctReturns[periodName] = cleanNumber(100 * (endPrice - startPrice) / startPrice, NumberType.PERCENTAGE_CHANGE)
        else:
            pctReturns[periodName] = "N/A"

    # Find SMA crossovers
    crossovers = findSmaCrossovers(priceData, maxLookbackDays=503, maxCrossovers=8)

    # Compile results
    stockPricePerformance = {
        "ticker": ticker.upper(),
        "mostRecentPrice": cleanNumber(mostRecentPrice, NumberType.STOCK_PRICE),

        "sma": {
            "10Day": {
                "value": cleanNumber(sma10Day, NumberType.STOCK_PRICE),
                "distance": cleanNumber(priceData["close"].iloc[-1] - sma10Day, NumberType.STOCK_PRICE_CHANGE)
            },
            "20Day": {
                "value": cleanNumber(sma20Day, NumberType.STOCK_PRICE),
                "distance": cleanNumber(priceData["close"].iloc[-1] - sma20Day, NumberType.STOCK_PRICE_CHANGE)
            },
            "50Day": {
                "value": cleanNumber(sma50Day, NumberType.STOCK_PRICE),
                "distance": cleanNumber(priceData["close"].iloc[-1] - sma50Day, NumberType.STOCK_PRICE_CHANGE)
            },
            "200Day": {
                "value": cleanNumber(sma200Day, NumberType.STOCK_PRICE),
                "distance": cleanNumber(priceData["close"].iloc[-1] - sma200Day, NumberType.STOCK_PRICE_CHANGE)
            }
        },

        "52Week": {
            "high": {
                "value": cleanNumber(highPrice52W, NumberType.STOCK_PRICE),
                "distance": cleanNumber(priceData["close"].iloc[-1] - highPrice52W, NumberType.STOCK_PRICE_CHANGE)
            },
            "low": {
                "value": cleanNumber(lowPrice52W, NumberType.STOCK_PRICE),
                "distance": cleanNumber(priceData["close"].iloc[-1] - lowPrice52W, NumberType.STOCK_PRICE_CHANGE)
            }
        },

        "rsi": {
            "7Day": cleanNumber(rsi7Day, NumberType.DECIMAL),
            "14Day": cleanNumber(rsi14Day, NumberType.DECIMAL),
            "30Day": cleanNumber(rsi30Day, NumberType.DECIMAL)
        },

        "macd_12/26/9": {
            "macd": cleanNumber(macdData["macd"], NumberType.DECIMAL),
            "signal": cleanNumber(macdData["signal"], NumberType.DECIMAL),
            "histogram": cleanNumber(macdData["histogram"], NumberType.DECIMAL)
        },

        "volatility30Day": cleanNumber(volatility30d, NumberType.DECIMAL),
        "maxDrawdown": cleanNumber(maxDrawdown, NumberType.DECIMAL),
        "sharpeRatio": cleanNumber(sharpeRatio, NumberType.DECIMAL),

        "returns": pctReturns,
        "smaCrossovers": crossovers
    }

    data.cache.put(cacheKey, stockPricePerformance)
    return stockPricePerformance



def calculateDistFromCurrPrice(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, targetPrice: float):    
    useLocal = isLocalDataAvailable(timestamp)
    mostRecentPrice: float = None
    
    if useLocal:
        priceData = data.ohlcv.getSingleDayTickerData(ticker, timestamp)
        mostRecentPrice = float(priceData["close"])
    else:
        data.rateLimiters.yFinanceLimiter.wait()
        tickerObj = yf.Ticker(ticker.upper())
        tickerInfo = tickerObj.info
        mostRecentPrice = float(
            tickerInfo.get("regularMarketOpen")
            or tickerInfo.get("regularMarketPrice")
        )

    percentChange = 100 * (targetPrice - mostRecentPrice) / mostRecentPrice
    result = {
        "ticker": ticker.upper(),
        "mostRecentPrice": cleanNumber(mostRecentPrice, NumberType.STOCK_PRICE),
        "targetPrice": cleanNumber(targetPrice, NumberType.STOCK_PRICE),
        "currentToTargetPctChange": cleanNumber(percentChange, NumberType.PERCENTAGE_CHANGE)
    }
    return result
