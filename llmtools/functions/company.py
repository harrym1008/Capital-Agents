import os
import numpy as np
import pandas as pd

from dataquery.ticker_provider import CompanyProfile
from dataquery.macro_provider import MacroSeries
from collectors.constants import IPO_BEFORE_START_DATE

from llmtools.tool_registry import DataProviders, Tool
from llmtools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, formatArticleAge, NumberType


def fetchCompanyProfile(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str):
    tickerProfile: CompanyProfile = data.tickers.getTickerProfile(ticker)
    if tickerProfile is None:
        return f"No profile found for ticker {ticker}"

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


def fetchFinnhubCompanyFundamentals(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str):
    cleanTicker = ticker.strip().upper()
    metrics = data.finnhub.getPointInTimeMetrics(cleanTicker, timestamp)
    if not metrics:
        return {"error": f"No Finnhub financial metrics available for ticker {cleanTicker}."}

    # Fetch latest price & mkt cap from local store if available
    todayRow = data.ohlcv.getSingleDayTickerData(cleanTicker, timestamp)
    latestPrice = None
    marketCap = None
    if todayRow is not None and not todayRow.empty:
        latestPrice = todayRow.get("close")
        shs = todayRow.get("outstandingShares")
        if latestPrice is not None and shs is not None:
            marketCap = latestPrice * shs

    profile = data.tickers.getTickerProfile(cleanTicker)
    companyName = profile.name if profile else cleanTicker
    sectorName = profile.sector if profile else "Unknown"
    industryName = profile.industry if profile else "Unknown"

    return cleanData({
        "ticker": cleanTicker,
        "companyName": companyName,
        "sector": sectorName,
        "industry": industryName,
        "asOfDate": metrics.get("asOfDate", timestamp.strftime("%Y-%m-%d")),
        "latestPrice": cleanNumber(latestPrice, NumberType.STOCK_PRICE) if latestPrice else None,
        "marketCap": cleanNumber(marketCap, NumberType.LARGE_DOLLARS) if marketCap else None,
        "valuation": {
            "peTTM": cleanNumber(metrics.get("peTTM"), NumberType.DECIMAL),
            "pb": cleanNumber(metrics.get("pb"), NumberType.DECIMAL),
            "psTTM": cleanNumber(metrics.get("psTTM"), NumberType.DECIMAL),
            "dividendYield": cleanNumber(metrics.get("dividendYield"), NumberType.UNSCALED_PERCENTAGE),
            "beta": cleanNumber(metrics.get("beta"), NumberType.DECIMAL)
        },
        "profitability": {
            "grossMargin": cleanNumber(metrics.get("grossMargin"), NumberType.UNSCALED_PERCENTAGE),
            "operatingMargin": cleanNumber(metrics.get("operatingMargin"), NumberType.UNSCALED_PERCENTAGE),
            "netMargin": cleanNumber(metrics.get("netMargin"), NumberType.UNSCALED_PERCENTAGE),
            "roeTTM": cleanNumber(metrics.get("roeTTM"), NumberType.UNSCALED_PERCENTAGE),
            "roaTTM": cleanNumber(metrics.get("roaTTM"), NumberType.UNSCALED_PERCENTAGE)
        },
        "financialHealth": {
            "currentRatio": cleanNumber(metrics.get("currentRatio"), NumberType.DECIMAL),
            "quickRatio": cleanNumber(metrics.get("quickRatio"), NumberType.DECIMAL),
            "debtToEquity": cleanNumber(metrics.get("debtToEquity"), NumberType.DECIMAL),
            "eps": cleanNumber(metrics.get("eps"), NumberType.DECIMAL),
            "ebitda": cleanNumber(metrics.get("ebitda"), NumberType.LARGE_DOLLARS)
        },
        "note": "Point-in-time financial metrics sourced via Finnhub, temporally isolated to evaluation date."
    })


def fetchBatchFinnhubMetrics(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, tickers: list):
    if not isinstance(tickers, list) or not tickers:
        return {"error": "tickers must be a non-empty list of ticker symbols."}

    results = []
    cleanTickers = [str(t).strip().upper() for t in tickers[:15]]  # Cap at 15 to avoid context bloat

    for t in cleanTickers:
        m = data.finnhub.getPointInTimeMetrics(t, timestamp)
        if not m:
            continue

        todayRow = data.ohlcv.getSingleDayTickerData(t, timestamp)
        price = todayRow.get("close") if (todayRow is not None and not todayRow.empty) else None

        results.append({
            "ticker": t,
            "price": cleanNumber(price, NumberType.STOCK_PRICE) if price else None,
            "peTTM": cleanNumber(m.get("peTTM"), NumberType.DECIMAL),
            "pb": cleanNumber(m.get("pb"), NumberType.DECIMAL),
            "grossMargin": cleanNumber(m.get("grossMargin"), NumberType.UNSCALED_PERCENTAGE),
            "netMargin": cleanNumber(m.get("netMargin"), NumberType.UNSCALED_PERCENTAGE),
            "roeTTM": cleanNumber(m.get("roeTTM"), NumberType.UNSCALED_PERCENTAGE),
            "debtToEquity": cleanNumber(m.get("debtToEquity"), NumberType.DECIMAL),
            "dividendYield": cleanNumber(m.get("dividendYield"), NumberType.UNSCALED_PERCENTAGE),
            "beta": cleanNumber(m.get("beta"), NumberType.DECIMAL)
        })

    return cleanData({
        "asOfDate": timestamp.strftime("%Y-%m-%d"),
        "stockCount": len(results),
        "stocks": results
    })


def fetchCompanyRecentNews(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, limit: int = 12):
    limit = min(max(limit, 1), 18)
    jsonResult = []

    newsWithContent = data.news.getRecentNewsForTicker(ticker, before=timestamp, limit=limit, mustHaveContent=True)

    idx = 0
    if newsWithContent is not None and not newsWithContent.empty:
        for _, row in newsWithContent.head(limit).iterrows():
            headline = row.get("headline", "").strip()
            if not headline:
                continue
            headline = cleanHtmlContent(headline)

            content = row.get("content", "").strip()
            if not content:
                content = "Article has no content available."
            else:
                content = cleanHtmlContent(content)
                if len(content) > 2500:
                    content = content[:2500] + "... [content truncated]"

            author = row.get("author", "").strip()
            rawDate = row.get("date")
            ageStr = formatArticleAge(rawDate, timestamp)
            articleId = str(row.get("id", ""))
            rawTs = pd.to_datetime(rawDate)
            if not pd.isna(rawTs):
                year = rawTs.year % 100
                month = rawTs.month
                url = f"https://www.benzinga.com/news/{year:02d}/{month:02d}/{articleId}"
            else:
                url = f"https://www.benzinga.com/news/01/16/{articleId}"

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
        jsonResult.append({"error": "No news data available."})

    if len(jsonResult) < limit and "error" not in jsonResult[0]:
        jsonResult.append({"info": f"Only {len(jsonResult)} articles could be retrieved."})

    return {"date": timestamp.strftime("%Y-%m-%d"), "news": cleanData(jsonResult)}


def calculateRsi(priceData: pd.DataFrame, period: int = 14) -> float:
    closes = priceData["close"]
    deltas = closes.diff()
    gains = deltas.clip(lower=0)
    losses = -1 * deltas.clip(upper=0)

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
    if priceData.empty or treasuryData.empty:
        return 0.0

    priceDf = priceData.copy()
    treasuryDf = treasuryData.copy()

    if priceDf["date"].dt.tz is not None:
        priceDf["date"] = priceDf["date"].dt.tz_localize(None)

    if treasuryDf["date"].dt.tz is not None:
        treasuryDf["date"] = treasuryDf["date"].dt.tz_localize(None)

    treasuryDf = treasuryDf.rename(columns={"value": "treasuryRate"})
    mergedData = pd.merge(priceDf, treasuryDf, on="date", how="inner")

    if mergedData.empty:
        return 0.0

    returns = mergedData["close"].pct_change()
    dailyRiskFree = (1 + mergedData["treasuryRate"] / 100) ** (1 / 252) - 1
    excessReturns = returns - dailyRiskFree

    stdDev = excessReturns.std()
    if stdDev == 0 or pd.isna(stdDev):
        return 0.0

    sharpeRatio = excessReturns.mean() / stdDev * np.sqrt(252)
    return sharpeRatio


def findSmaCrossovers(priceData: pd.DataFrame, maxLookbackDays: int = 503, maxCrossovers: int = 8):
    df = priceData.copy()
    df["sma50"] = df["close"].rolling(window=50).mean()
    df["sma200"] = df["close"].rolling(window=200).mean()

    goldenCondition = (df["sma50"] > df["sma200"]) & (df["sma50"].shift(1) <= df["sma200"].shift(1))
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


def calculate30DayAverageVolume(priceData: pd.DataFrame, timestamp: pd.Timestamp = None) -> float:
    if priceData.empty or "volume" not in priceData.columns:
        return 0.0
    if timestamp is not None and "date" in priceData.columns:
        thirtyDaysAgo = timestamp - pd.DateOffset(days=30)
        recentData = priceData[priceData["date"] >= thirtyDaysAgo]
        if not recentData.empty:
            return float(recentData["volume"].mean())
    return float(priceData["volume"].tail(30).mean())


def fetchStockPricePerformance(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str):
    startDate = timestamp - pd.DateOffset(years=5, weeks=1)
    cacheKey = f"stockPerf|{ticker}_{timestamp.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    priceData = data.ohlcv.getPeriodDailyTickerData(ticker, startDate=startDate, endDate=timestamp)
    if priceData.empty:
        return f"No price data available for ticker {ticker} before {timestamp.strftime('%Y-%m-%d')}"

    if "splitFactor" in priceData.columns and not priceData["splitFactor"].empty:
        finalSplitFactor = priceData["splitFactor"].iloc[-1]
        for col in ["open", "high", "low", "close", "vwap"]:
            if col in priceData.columns:
                priceData[col] = priceData[col] * (priceData["splitFactor"] / finalSplitFactor)

    avgVolume30Day = calculate30DayAverageVolume(priceData, timestamp)

    dropCols = [c for c in ["volume", "vwap", "splitFactor", "corpActionToday", "outstandingShares", "marketCap"] if c in priceData.columns]
    priceData = priceData.drop(columns=dropCols).dropna().reset_index(drop=True)

    if priceData.empty:
        return f"No price data available for ticker {ticker} before {timestamp.strftime('%Y-%m-%d')}"

    mostRecentPrice = priceData["close"].iloc[-1]
    earliestDate = priceData["date"].iloc[0]

    sma10Day = priceData["close"].rolling(window=min(10, len(priceData))).mean().iloc[-1]
    sma20Day = priceData["close"].rolling(window=min(20, len(priceData))).mean().iloc[-1]
    sma50Day = priceData["close"].rolling(window=min(50, len(priceData))).mean().iloc[-1]
    sma200Day = priceData["close"].rolling(window=min(200, len(priceData))).mean().iloc[-1]

    fiftyTwoWeekAgo = timestamp - pd.DateOffset(weeks=52)
    priceData52W = priceData[priceData["date"] >= fiftyTwoWeekAgo]
    highPrice52W = priceData52W["high"].max() if not priceData52W.empty else mostRecentPrice
    lowPrice52W = priceData52W["low"].min() if not priceData52W.empty else mostRecentPrice

    rsi7Day = calculateRsi(priceData, period=7)
    rsi14Day = calculateRsi(priceData, period=14)
    rsi30Day = calculateRsi(priceData, period=30)

    pctReturns = priceData["close"].pct_change(fill_method=None).dropna()
    volatility30d = pctReturns.tail(30).std() * np.sqrt(252)

    runningMax = priceData["close"].cummax()
    maxDrawdown = (priceData["close"] / runningMax - 1).min()

    macdData = calculateMacd(priceData, fast=12, slow=26, signal=9)
    treasuryData = data.macro.getSeries(MacroSeries.TREAS_2Y, startDate=startDate, endDate=timestamp)
    sharpeRatio = calculateSharpeRatio(priceData, treasuryData)

    periods = {
        "5d": timestamp - pd.DateOffset(days=5),
        "1mo": timestamp - pd.DateOffset(months=1),
        "3mo": timestamp - pd.DateOffset(months=3),
        "6mo": timestamp - pd.DateOffset(months=6),
        "12mo": timestamp - pd.DateOffset(years=1),
        "3y": timestamp - pd.DateOffset(years=3),
        "5y": timestamp - pd.DateOffset(years=5)
    }
    pctReturnsDict = {}
    for periodName, periodStart in periods.items():
        if periodStart < earliestDate:
            continue
        periodData = priceData[priceData["date"] >= periodStart]
        if not periodData.empty:
            startPrice = periodData["close"].iloc[0]
            endPrice = periodData["close"].iloc[-1]
            pctReturnsDict[periodName] = cleanNumber(100 * (endPrice - startPrice) / startPrice, NumberType.PERCENTAGE_CHANGE)
        else:
            pctReturnsDict[periodName] = "N/A"

    crossovers = findSmaCrossovers(priceData, maxLookbackDays=503, maxCrossovers=8)

    stockPricePerformance = {
        "ticker": ticker.upper(),
        "mostRecentPrice": cleanNumber(mostRecentPrice, NumberType.STOCK_PRICE),

        "sma": {
            "10Day": {
                "value": cleanNumber(sma10Day, NumberType.STOCK_PRICE),
                "distance": cleanNumber(mostRecentPrice - sma10Day, NumberType.STOCK_PRICE_CHANGE)
            },
            "20Day": {
                "value": cleanNumber(sma20Day, NumberType.STOCK_PRICE),
                "distance": cleanNumber(mostRecentPrice - sma20Day, NumberType.STOCK_PRICE_CHANGE)
            },
            "50Day": {
                "value": cleanNumber(sma50Day, NumberType.STOCK_PRICE),
                "distance": cleanNumber(mostRecentPrice - sma50Day, NumberType.STOCK_PRICE_CHANGE)
            },
            "200Day": {
                "value": cleanNumber(sma200Day, NumberType.STOCK_PRICE),
                "distance": cleanNumber(mostRecentPrice - sma200Day, NumberType.STOCK_PRICE_CHANGE)
            }
        },

        "52Week": {
            "high": {
                "value": cleanNumber(highPrice52W, NumberType.STOCK_PRICE),
                "distance": cleanNumber(mostRecentPrice - highPrice52W, NumberType.STOCK_PRICE_CHANGE)
            },
            "low": {
                "value": cleanNumber(lowPrice52W, NumberType.STOCK_PRICE),
                "distance": cleanNumber(mostRecentPrice - lowPrice52W, NumberType.STOCK_PRICE_CHANGE)
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
        "averageVolume30Day": cleanNumber(avgVolume30Day, NumberType.LARGE_NUMBER),

        "returns": pctReturnsDict,
        "smaCrossovers": crossovers
    }

    data.cache.put(cacheKey, stockPricePerformance)
    return stockPricePerformance


def calculateDistFromCurrPrice(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, targetPrice: float):
    priceData = data.ohlcv.getSingleDayTickerData(ticker, timestamp)
    if priceData is None or "close" not in priceData:
        return f"Could not retrieve price for ticker {ticker}"

    mostRecentPrice = float(priceData["close"])
    percentChange = 100 * (targetPrice - mostRecentPrice) / mostRecentPrice
    result = {
        "ticker": ticker.upper(),
        "mostRecentPrice": cleanNumber(mostRecentPrice, NumberType.STOCK_PRICE),
        "targetPrice": cleanNumber(targetPrice, NumberType.STOCK_PRICE),
        "currentToTargetPctChange": cleanNumber(percentChange, NumberType.PERCENTAGE_CHANGE)
    }
    return result
