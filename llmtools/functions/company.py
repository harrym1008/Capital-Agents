import os
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from dataquery.ticker_provider import CompanyProfile
from dataquery.macro_provider import MacroSeries
from collectors.constants import IPO_BEFORE_START_DATE

from llmtools.tool_registry import DataProviders, Tool
from llmtools.functions.helpers import cleanKey, cleanData, cleanNumber, cleanHtmlContent, formatArticleAge, NumberType
from llmtools.functions.stock_search import parseMarketCapValue
from llmtools.functions.sentiment_main import scoreTextsWithCache, aggregateSentiment


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


def fetchBatchStockOverviews(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, tickers: list):
    if not isinstance(tickers, list) or not tickers:
        return {"error": "tickers must be a non-empty list of ticker symbols."}

    cleanTickers = [str(t).strip().upper() for t in tickers[:30]]
    totalTickers = len(cleanTickers)

    # Normalise timestamp for date comparisons
    tsNorm = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp

    # Pre-fetch SPY data once for beta computation (shared across all stocks)
    oneYearAgo = timestamp - pd.DateOffset(years=1, weeks=1)
    spyReturns = None
    try:
        spyDf = data.ohlcv.getPeriodDailyTickerData("SPY", startDate=oneYearAgo, endDate=timestamp)
        if spyDf is not None and not spyDf.empty and len(spyDf) > 60:
            if "date" in spyDf.columns:
                spyDf["date"] = pd.to_datetime(spyDf["date"], utc=True).dt.tz_localize(None)
            if "splitFactor" in spyDf.columns and not spyDf["splitFactor"].empty:
                finalSplit = spyDf["splitFactor"].iloc[-1]
                spyDf["close"] = spyDf["close"] * (spyDf["splitFactor"] / finalSplit)
            spyReturns = spyDf.set_index("date")["close"].pct_change().dropna()
    except Exception:
        pass

    results = []
    failures = []
    progressLock = threading.Lock()
    completedCount = [0]

    def processStock(ticker):
        stockResult = {"ticker": ticker}

        # Company profile info & summary (truncated up to 50 words)
        try:
            tickerProfile = data.tickers.getTickerProfile(ticker)
            if tickerProfile:
                stockResult["companyName"] = tickerProfile.name
                stockResult["industry"] = tickerProfile.industry
                summaryWords = tickerProfile.summary.strip().split()
                if len(summaryWords) > 50:
                    stockResult["companySummary"] = " ".join(summaryWords[:50]) + "..."
                else:
                    stockResult["companySummary"] = tickerProfile.summary.strip()
                    
        except Exception:
            pass

        # ===== STAGE 1: Fundamentals + Price =====
        # OHLCV current day data
        todayRow = data.ohlcv.getSingleDayTickerData(ticker, timestamp)
        price = todayRow.get("close") if (todayRow is not None and not todayRow.empty) else None

        if not price:
            return None  # No price data, skip this stock

        stockResult["price"] = cleanNumber(price, NumberType.STOCK_PRICE)

        # Market cap from OHLCV
        marketCapNum = None
        marketCapRaw = todayRow.get("marketCap") if (todayRow is not None and not todayRow.empty) else None
        if marketCapRaw:
            marketCapNum = parseMarketCapValue(marketCapRaw)
            if marketCapNum > 0:
                stockResult["marketCap"] = cleanNumber(marketCapNum, NumberType.LARGE_NUMBER)

        # Point-in-time fundamentals using today's price and market cap for dynamic valuation ratios
        m = data.finnhub.getPointInTimeMetrics(ticker, timestamp, currentPrice=float(price), currentMarketCap=marketCapNum)

        # 1-year price data for returns, beta, volatility
        priceData = data.ohlcv.getPeriodDailyTickerData(ticker, startDate=oneYearAgo, endDate=timestamp)

        if priceData is not None and not priceData.empty and len(priceData) > 20:
            if "date" in priceData.columns:
                priceData["date"] = pd.to_datetime(priceData["date"], utc=True).dt.tz_localize(None)
            if "splitFactor" in priceData.columns and not priceData["splitFactor"].empty:
                finalSplit = priceData["splitFactor"].iloc[-1]
                for col in ["open", "high", "low", "close", "vwap"]:
                    if col in priceData.columns:
                        priceData[col] = priceData[col] * (priceData["splitFactor"] / finalSplit)

            closes = priceData["close"]
            latestClose = float(closes.iloc[-1])
            earliestDate = priceData["date"].iloc[0]

            # Returns (1mo, 3mo, 6mo, 1y)
            returnsDict = {}
            periods = {
                "1mo": timestamp - pd.DateOffset(months=1),
                "3mo": timestamp - pd.DateOffset(months=3),
                "6mo": timestamp - pd.DateOffset(months=6),
                "1y": timestamp - pd.DateOffset(years=1),
            }
            for periodName, periodStart in periods.items():
                startNorm = periodStart.tz_localize(None) if periodStart.tzinfo else periodStart
                if startNorm < earliestDate:
                    continue
                pData = priceData[priceData["date"] >= startNorm]
                if not pData.empty:
                    startP = float(pData["close"].iloc[0])
                    if startP > 0:
                        pctReturn = 100 * (latestClose - startP) / startP
                        returnsDict[periodName] = cleanNumber(pctReturn, NumberType.PERCENTAGE_CHANGE)
            if returnsDict:
                stockResult["returns"] = returnsDict

            # Volatility (30-day annualised)
            dailyReturns = closes.pct_change(fill_method=None).dropna()
            if len(dailyReturns) >= 20:
                vol30d = float(dailyReturns.tail(30).std() * np.sqrt(252) * 100)
                stockResult["volatility30d"] = cleanNumber(vol30d, NumberType.PERCENTAGE)

            # Average volume (30-day)
            if "volume" in priceData.columns:
                avgVol = float(priceData["volume"].tail(30).mean())
                stockResult["avgVolume30d"] = cleanNumber(avgVol, NumberType.LARGE_NUMBER)

            # Beta (vs SPY from OHLCV - fully point-in-time)
            if spyReturns is not None and not spyReturns.empty:
                try:
                    stockReturnsSeries = priceData.set_index("date")["close"].pct_change().dropna()
                    merged = pd.DataFrame({"stock": stockReturnsSeries, "spy": spyReturns}).dropna()
                    if len(merged) > 30:
                        covMatrix = np.cov(merged["stock"], merged["spy"])
                        spyVar = covMatrix[1][1]
                        if spyVar > 0:
                            beta = float(covMatrix[0][1] / spyVar)
                            stockResult["beta"] = cleanNumber(beta, NumberType.DECIMAL)
                except Exception:
                    pass

        # Finnhub series metrics (all point-in-time, no live fallbacks)
        if m:
            # Valuation
            if m.get("peTTM") is not None:
                stockResult["peTTM"] = cleanNumber(m["peTTM"], NumberType.DECIMAL)
            if m.get("pb") is not None:
                stockResult["pb"] = cleanNumber(m["pb"], NumberType.DECIMAL)
            if m.get("psTTM") is not None:
                stockResult["psTTM"] = cleanNumber(m["psTTM"], NumberType.DECIMAL)
            if m.get("evEbitdaTTM") is not None:
                stockResult["evEbitdaTTM"] = cleanNumber(m["evEbitdaTTM"], NumberType.DECIMAL)

            # Profitability
            if m.get("grossMargin") is not None:
                stockResult["grossMargin"] = cleanNumber(m["grossMargin"], NumberType.UNSCALED_PERCENTAGE)
            if m.get("operatingMargin") is not None:
                stockResult["operatingMargin"] = cleanNumber(m["operatingMargin"], NumberType.UNSCALED_PERCENTAGE)
            if m.get("netMargin") is not None:
                stockResult["netMargin"] = cleanNumber(m["netMargin"], NumberType.UNSCALED_PERCENTAGE)
            if m.get("fcfMargin") is not None:
                stockResult["fcfMargin"] = cleanNumber(m["fcfMargin"], NumberType.UNSCALED_PERCENTAGE)

            # Returns & Efficiency
            if m.get("roeTTM") is not None:
                stockResult["roeTTM"] = cleanNumber(m["roeTTM"], NumberType.UNSCALED_PERCENTAGE)
            if m.get("roicTTM") is not None:
                stockResult["roicTTM"] = cleanNumber(m["roicTTM"], NumberType.UNSCALED_PERCENTAGE)

            # Growth
            if m.get("epsGrowthQoQ") is not None:
                stockResult["epsGrowthQoQ"] = cleanNumber(m["epsGrowthQoQ"], NumberType.PERCENTAGE_CHANGE)
            if m.get("epsGrowthYoY") is not None:
                stockResult["epsGrowthYoY"] = cleanNumber(m["epsGrowthYoY"], NumberType.PERCENTAGE_CHANGE)
            if m.get("revenueGrowthQoQ") is not None:
                stockResult["revenueGrowthQoQ"] = cleanNumber(m["revenueGrowthQoQ"], NumberType.PERCENTAGE_CHANGE)
            if m.get("revenueGrowthYoY") is not None:
                stockResult["revenueGrowthYoY"] = cleanNumber(m["revenueGrowthYoY"], NumberType.PERCENTAGE_CHANGE)

            # Leverage
            if m.get("debtToEquity") is not None:
                stockResult["debtToEquity"] = cleanNumber(m["debtToEquity"], NumberType.DECIMAL)
            if m.get("currentRatio") is not None:
                stockResult["currentRatio"] = cleanNumber(m["currentRatio"], NumberType.DECIMAL)
            if m.get("quickRatio") is not None:
                stockResult["quickRatio"] = cleanNumber(m["quickRatio"], NumberType.DECIMAL)

            # Dividend yield (computed from payoutRatio × EPS TTM / price - all point-in-time)
            payoutRatio = m.get("payoutRatioTTM")
            epsVal = m.get("epsTTM") or m.get("eps")
            if payoutRatio is not None and epsVal is not None and epsVal > 0 and price and price > 0:
                divPerShare = epsVal * (payoutRatio / 100.0)
                if divPerShare > 0:
                    dividendYield = divPerShare / float(price)
                    stockResult["dividendYield"] = cleanNumber(dividendYield, NumberType.UNSCALED_PERCENTAGE)

        # ===== STAGE 2: News + Sentiment =====
        try:
            newsDf = data.news.getRecentNewsForTicker(
                ticker, before=timestamp, limit=50,
                mustHaveContent=False, maxReferencedTickers=5, summaryMaxChars=500
            )

            if newsDf is not None and not newsDf.empty:
                headlines = []
                headlineDates = []
                for _, row in newsDf.iterrows():
                    hl = row.get("headline", "")
                    if isinstance(hl, str) and hl.strip():
                        headlines.append(cleanHtmlContent(hl.strip()))
                        headlineDates.append(row.get("date"))

                # Display last 10 headlines (most recent)
                displayHeadlines = []
                for i, (hl, dt) in enumerate(zip(headlines[:10], headlineDates[:10])):
                    age = formatArticleAge(dt, timestamp)
                    displayHeadlines.append({"headline": hl, "age": age})
                if displayHeadlines:
                    stockResult["recentHeadlines"] = displayHeadlines

                # Score up to 50 headlines with the decay-weighted aggregation
                if headlines:
                    sentimentScores = scoreTextsWithCache(headlines[:50], data)
                    if sentimentScores:
                        finalSentiment, rating = aggregateSentiment(
                            sentimentScores, headlineDates[:50], tsNorm
                        )
                        if finalSentiment is not None:
                            stockResult["newsSentimentScore"] = cleanNumber(finalSentiment, NumberType.DECIMAL)
                            stockResult["newsSentimentRating"] = rating
        except Exception:
            pass

        # ===== STAGE 3: Short Interest =====
        try:
            shortInfo = data.short.getLatestShortInterestForTicker(ticker, before=timestamp)
            if shortInfo is not None:
                stockResult["shortInterest"] = cleanData({
                    "currentPositions": cleanNumber(shortInfo.currentShortPositions, NumberType.LARGE_NUMBER),
                    "changePercent": cleanNumber(shortInfo.changePercent, NumberType.PERCENTAGE_CHANGE),
                    "daysToCover": cleanNumber(shortInfo.daysToCover, NumberType.DECIMAL)
                })
        except Exception:
            pass

        # Progress update
        with progressLock:
            completedCount[0] += 1
            pct = completedCount[0] / totalTickers * 100
            tool.updateProgress(pct)

        return stockResult

    # Process all stocks concurrently
    with ThreadPoolExecutor(max_workers=min(totalTickers, 4)) as executor:
        futureToTicker = {executor.submit(processStock, t): t for t in cleanTickers}
        for future in as_completed(futureToTicker):
            ticker = futureToTicker[future]
            try:
                result = future.result()
                if result and len(result) > 1:
                    results.append(result)
                else:
                    failures.append(ticker)
            except Exception:
                failures.append(ticker)
            tool.updateProgress((len(results) + len(failures)) / totalTickers * 100)
            

    output = {
        "asOfDate": timestamp.strftime("%Y-%m-%d"),
        "stockCount": len(results),
        "stocks": results
    }
    if failures:
        output["failedTickers"] = failures

    return cleanData(output)


def fetchCompanyRecentNews(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, limit: int = 12):
    limit = min(max(limit, 1), 18)
    jsonResult = []

    newsWithContent = data.news.getRecentNewsForTicker(ticker, before=timestamp, limit=limit, mustHaveContent=True)

    idx = 1
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
                "newsCitationNumber": idx,
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

    priceDate = pd.to_datetime(priceDf["date"], utc=True)
    treasuryDate = pd.to_datetime(treasuryDf["date"], utc=True)

    priceDf["date"] = priceDate.dt.tz_localize(None).dt.normalize()
    treasuryDf["date"] = treasuryDate.dt.tz_localize(None).dt.normalize()

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

    if "date" in priceData.columns:
        if timestamp.tzinfo is not None:
            priceData["date"] = pd.to_datetime(priceData["date"], utc=True).dt.tz_convert(timestamp.tz)
        else:
            priceData["date"] = pd.to_datetime(priceData["date"], utc=True).dt.tz_localize(None)

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
    sharpeComparison = data.macro.getSeries(MacroSeries.TREAS_3MO, startDate=startDate, endDate=timestamp)
    sharpeRatio = calculateSharpeRatio(priceData, sharpeComparison)

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


def fetchShortInterestHistory(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, ticker: str, months: int = 6):
    ticker = str(ticker).strip().upper()
    months = max(1, min(months, 24))

    startDate = timestamp - pd.DateOffset(months=months)
    df = data.short.getShortInterestForTicker(ticker, startDate=startDate, endDate=timestamp)

    if df is None or df.empty:
        return cleanData({
            "ticker": ticker,
            "status": "no_data",
            "message": f"No short interest data found for {ticker} in the last {months} months."
        })

    entries = []
    for _, row in df.iterrows():
        entries.append({
            "date": row.get("date"),
            "currentPositions": cleanNumber(row.get("currentShortPositions"), NumberType.LARGE_NUMBER),
            "previousPositions": cleanNumber(row.get("previousShortPositions"), NumberType.LARGE_NUMBER),
            "changePercent": cleanNumber(row.get("changePercent"), NumberType.PERCENTAGE_CHANGE),
            "avgDailyVolume": cleanNumber(row.get("avgDailyVolume"), NumberType.LARGE_NUMBER),
            "daysToCover": cleanNumber(row.get("daysToCover"), NumberType.DECIMAL),
        })

    # Compute trend direction
    if len(entries) >= 2:
        latestPositions = df.iloc[0].get("currentShortPositions", 0)
        oldestPositions = df.iloc[-1].get("currentShortPositions", 0)
        if oldestPositions and oldestPositions > 0:
            trendPct = ((latestPositions - oldestPositions) / oldestPositions) * 100
            trend = "increasing" if trendPct > 5 else ("decreasing" if trendPct < -5 else "stable")
        else:
            trendPct = None
            trend = "unknown"
    else:
        trendPct = None
        trend = "unknown"

    result = {
        "ticker": ticker,
        "periodMonths": months,
        "asOfDate": timestamp.strftime("%Y-%m-%d"),
        "reportingPeriods": len(entries),
        "trend": trend,
        "trendChangePct": cleanNumber(trendPct, NumberType.PERCENTAGE_CHANGE) if trendPct is not None else None,
        "history": entries
    }

    return cleanData(result)

