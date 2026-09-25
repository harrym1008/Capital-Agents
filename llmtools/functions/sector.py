import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from collectors.sector_dl_client import GICS_SECTORS, DB_SECTOR_TO_TICKER
from collectors.constants import UTC, NEW_YORK, END_DATE
from dataquery.macro_provider import MacroSeries
from llmtools.tool_registry import DataProviders, Tool
from llmtools.functions.helpers import (
    cleanData, 
    formatArticleAge, 
    cleanHtmlContent, 
    normaliseSectorInput, 
    attachSectorWarning, 
    FINANCIAL_SERVICES_WARNING
)
from llmtools.functions.stock_search import parseMarketCapValue
from llmtools.functions.sentiment_main import scoreTextsWithCache, deriveSentimentRating, getSentimentEngine, aggregateSentiment
from finbert.finbert_engines import TrtCudaInferenceEngine, OnnxCudaInferenceEngine, PytorchCudaInferenceEngine


# Compute Relative Strength Index over 14-day default window
def calculateRsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0

    deltas = series.diff()
    gains = deltas.clip(lower=0)
    losses = -1 * deltas.clip(upper=0)

    avgGain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avgLoss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avgGain / avgLoss
    rsiSeries = 100 - (100 / (1 + rs))
    rsiSeries = rsiSeries.replace(np.inf, 100)
    return float(rsiSeries.iloc[-1])


# Calculate S&P 500 benchmark trailing returns across specified lookback intervals
def calculateBenchmarkReturns(data: DataProviders, timestamp: pd.Timestamp, lookbacks: Dict[str, pd.Timestamp]) -> Dict[str, float]:
    benchReturns = {}
    if data.macro is not None:
        try:
            sp500Df = data.macro.loadSeries(MacroSeries.SP500)
            if not sp500Df.empty:
                tsNorm = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
                validDf = sp500Df[sp500Df["date"] <= tsNorm].sort_values("date")
                if not validDf.empty:
                    latestClose = validDf["close"].iloc[-1]
                    for period, pastDate in lookbacks.items():
                        pastNorm = pastDate.tz_localize(None) if pastDate.tzinfo is not None else pastDate
                        pastRows = validDf[validDf["date"] <= pastNorm]
                        if not pastRows.empty:
                            pastClose = pastRows["close"].iloc[-1]
                            benchReturns[period] = float(((latestClose - pastClose) / pastClose) * 100)
        except Exception:
            pass
    return benchReturns


# Calculate moving averages, RSI, and technical trend categorization for sector ETF
def calculateSectorTechnicals(df: pd.DataFrame, latestClose: float) -> Dict[str, Any]:
    closes = df["close"]
    sma50 = float(closes.rolling(window=min(50, len(df))).mean().iloc[-1])
    sma200 = float(closes.rolling(window=min(200, len(df))).mean().iloc[-1])
    rsi14 = round(calculateRsi(closes, period=14), 1)

    distFromSma50 = round(((latestClose - sma50) / sma50) * 100, 2)
    distFromSma200 = round(((latestClose - sma200) / sma200) * 100, 2)

    if latestClose > sma50 and sma50 > sma200:
        trend = "Strong Uptrend (Above 50 & 200 SMA)"
    elif latestClose > sma50:
        trend = "Moderate Uptrend (Above 50 SMA)"
    elif latestClose < sma50 and sma50 < sma200:
        trend = "Strong Downtrend (Below 50 & 200 SMA)"
    else:
        trend = "Consolidating / Mixed"

    return {
        "rsi14": rsi14,
        "sma50": round(sma50, 2),
        "distFromSma50Pct": distFromSma50,
        "sma200": round(sma200, 2),
        "distFromSma200Pct": distFromSma200,
        "trend": trend
    }


# Calculate 52-week price range and drawdown percentage from peak
def calculateSectorRange52Week(df: pd.DataFrame, timestamp: pd.Timestamp, latestClose: float) -> Dict[str, Any]:
    tsNorm = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    oneYearAgo = tsNorm - pd.DateOffset(years=1)
    pastYearDf = df[df["date"] >= oneYearAgo]

    if not pastYearDf.empty:
        fiftyTwoWeekHigh = float(pastYearDf["high"].max() if "high" in pastYearDf.columns else pastYearDf["close"].max())
        fiftyTwoWeekLow = float(pastYearDf["low"].min() if "low" in pastYearDf.columns else pastYearDf["close"].min())
        drawdownFromHigh = round(((latestClose - fiftyTwoWeekHigh) / fiftyTwoWeekHigh) * 100, 2)
    else:
        fiftyTwoWeekHigh = latestClose
        fiftyTwoWeekLow = latestClose
        drawdownFromHigh = 0.0

    return {
        "52wHigh": round(fiftyTwoWeekHigh, 2),
        "52wLow": round(fiftyTwoWeekLow, 2),
        "drawdownFromHighPct": drawdownFromHigh
    }


# Calculate 90-day annualised price volatility percentage
def calculateAnnualisedVolatility(df: pd.DataFrame, window: int = 90) -> float:
    recentCloses = df["close"].tail(min(window, len(df)))
    dailyReturns = recentCloses.pct_change().dropna()
    if len(dailyReturns) > 1:
        return round(float(dailyReturns.std() * np.sqrt(252) * 100), 2)
    return 0.0


# Retrieve trailing returns, relative alpha, technicals, and volatility for single GICS sector ETF
def fetchSectorPerformance(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, sectorOrTicker: str) -> Dict[str, Any]:
    sectorOrTicker, wasUpdated = normaliseSectorInput(sectorOrTicker)
    tsNy = timestamp.tz_convert(NEW_YORK) if timestamp.tzinfo is not None else timestamp.tz_localize(NEW_YORK)
    effectiveTs = min(tsNy, END_DATE)
    cacheKey = f"sector|perf_{sectorOrTicker}_{effectiveTs.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return attachSectorWarning(cached, wasUpdated)

    ticker, resolvedName, note = data.sectors.resolveSector(sectorOrTicker)
    if resolvedName == "Unknown":
        return attachSectorWarning(cleanData({
            "status": "unknown_sector",
            "query": sectorOrTicker,
            "message": note or "Sector is listed as 'unknown' in the dataset. No GICS sector ETF applies."
        }), wasUpdated)
    if not ticker:
        return attachSectorWarning(cleanData({"error": note or f"Sector or ticker '{sectorOrTicker}' not recognised among the 11 GICS sectors."}), wasUpdated)

    info = GICS_SECTORS.get(ticker)
    sectorName = info.name if info else ticker
    sectorCategory = info.category if info else "Unknown"

    startDate = effectiveTs - pd.DateOffset(years=5, weeks=2)
    df = data.sectors.getSectorData(ticker, startDate=startDate, endDate=effectiveTs)

    if df.empty:
        if ticker == "XLC" and effectiveTs < pd.Timestamp("2018-06-19", tz=NEW_YORK):
            return cleanData({
                "ticker": ticker,
                "sector": sectorName,
                "notice": "Historical data for this sector (XLC) is not available before 2018-06-19."
            })
        return cleanData({"error": f"No historical data available for sector ETF {ticker} before {effectiveTs.strftime('%Y-%m-%d')}."})

    latestRow = df.iloc[-1]
    latestClose = float(latestRow["close"])
    latestDateStr = latestRow["date"].strftime("%Y-%m-%d")

    # Run through the follow list of lookback periods to calculate trailing returns and relative alpha vs S&P 500
    pastPeriods = {
        "1d": effectiveTs - pd.DateOffset(days=1),
        "5d": effectiveTs - pd.DateOffset(weeks=1),
        "1mo": effectiveTs - pd.DateOffset(months=1),
        "3mo": effectiveTs - pd.DateOffset(months=3),
        "6mo": effectiveTs - pd.DateOffset(months=6),
        "12mo": effectiveTs - pd.DateOffset(years=1),
        "3y": effectiveTs - pd.DateOffset(years=3),
        "5y": effectiveTs - pd.DateOffset(years=5)
    }

    returns = {}
    for period, pastDate in pastPeriods.items():
        pastNorm = pastDate.tz_localize(None) if pastDate.tzinfo is not None else pastDate
        pastRows = df[df["date"] <= pastNorm]
        if not pastRows.empty:
            pastClose = float(pastRows["close"].iloc[-1])
            pctChange = ((latestClose - pastClose) / pastClose) * 100
            returns[period] = round(pctChange, 2)
        else:
            returns[period] = None

    benchReturns = calculateBenchmarkReturns(data, effectiveTs, pastPeriods)
    relativeAlpha = {}
    for period in ["5d", "1mo", "3mo", "6mo", "12mo"]:
        secRet = returns.get(period)
        spyRet = benchReturns.get(period)
        if secRet is not None and spyRet is not None:
            relativeAlpha[period] = round(secRet - spyRet, 2)

    technicals = calculateSectorTechnicals(df, latestClose)
    range52Week = calculateSectorRange52Week(df, effectiveTs, latestClose)
    annualisedVol = calculateAnnualisedVolatility(df, window=90)

    output = {
        "ticker": ticker,
        "sector": sectorName,
        "category": sectorCategory,
        "asOfDate": latestDateStr,
        "latestPrice": round(latestClose, 2),
        "trailingReturns": returns,
        "relativeAlphaVsSP500": relativeAlpha,
        "technicals": technicals,
        "range52Week": range52Week,
        "annualisedVolatilityPct": annualisedVol
    }
    if note:
        output["resolutionNote"] = note

    cleanedResult = cleanData(output)
    data.cache.put(cacheKey, cleanedResult)
    return attachSectorWarning(cleanedResult, wasUpdated)


# Rank all 11 GICS sector ETFs by trailing performance over specified lookback window
def fetchAllSectorRankings(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, lookback: str = "1mo") -> Dict[str, Any]:
    validLookbacks = ["5d", "1mo", "3mo", "6mo", "12mo"]
    if lookback not in validLookbacks:
        lookback = "1mo"

    tsNy = timestamp.tz_convert(NEW_YORK) if timestamp.tzinfo is not None else timestamp.tz_localize(NEW_YORK)
    effectiveTs = min(tsNy, END_DATE)
    cacheKey = f"sector|leaderboard_{lookback}_{effectiveTs.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    with data.sectors.keyedLocks.lockKey(cacheKey):
        cached = data.cache.get(cacheKey)
        if cached is not None:
            return cached

        lookbackDeltas = {
            "5d": effectiveTs - pd.DateOffset(weeks=1),
            "1mo": effectiveTs - pd.DateOffset(months=1),
            "3mo": effectiveTs - pd.DateOffset(months=3),
            "6mo": effectiveTs - pd.DateOffset(months=6),
            "12mo": effectiveTs - pd.DateOffset(years=1)
        }
        pastDate = lookbackDeltas.get(lookback, effectiveTs - pd.DateOffset(months=1))
        benchReturns = calculateBenchmarkReturns(data, effectiveTs, {lookback: pastDate})
        spyReturn = benchReturns.get(lookback)

        rankings = []
        startDate = effectiveTs - pd.DateOffset(years=2)
        pastNorm = pastDate.tz_localize(None) if pastDate.tzinfo is not None else pastDate

        def processSector(ticker, info):
            df = data.sectors.getSectorData(ticker, startDate=startDate, endDate=timestamp)
            if df.empty:
                return None

            latestClose = float(df["close"].iloc[-1])
            pastRows = df[df["date"] <= pastNorm]

            if pastRows.empty:
                return None

            pastClose = float(pastRows["close"].iloc[-1])
            secReturn = round(((latestClose - pastClose) / pastClose) * 100, 2)
            alpha = round(secReturn - spyReturn, 2) if spyReturn is not None else None

            technicals = calculateSectorTechnicals(df, latestClose)

            return {
                "ticker": ticker,
                "sector": info.name,
                "category": info.category,
                "latestPrice": round(latestClose, 2),
                f"return_{lookback}": secReturn,
                "relativeAlphaVsSP500": alpha,
                "trend": technicals.get("trend"),
                "rsi14": technicals.get("rsi14")
            }

        totalSectors = len(GICS_SECTORS)
        completedSectors = 0
        with ThreadPoolExecutor(max_workers=min(totalSectors, 8)) as executor:
            futures = [executor.submit(processSector, ticker, info) for ticker, info in GICS_SECTORS.items()]
            for future in as_completed(futures):
                completedSectors += 1
                tool.updateProgress((completedSectors / totalSectors) * 100.0)
                try:
                    item = future.result()
                    if item is not None:
                        rankings.append(item)
                except Exception:
                    pass

        rankings.sort(key=lambda item: item[f"return_{lookback}"], reverse=True)

        for i, item in enumerate(rankings):
            item["rank"] = i + 1

        topSector = rankings[0] if rankings else None
        bottomSector = rankings[-1] if rankings else None

        result = {
            "asOfDate": effectiveTs.strftime("%Y-%m-%d"),
            "lookback": lookback,
            "benchmarkSP500Return": round(spyReturn, 2) if spyReturn is not None else None,
            "rankings": rankings,
            "summary": {
                "leadingSector": f"{topSector['sector']} ({topSector['ticker']}) at {topSector[f'return_{lookback}']}%" if topSector else "N/A",
                "laggingSector": f"{bottomSector['sector']} ({bottomSector['ticker']}) at {bottomSector[f'return_{lookback}']}%" if bottomSector else "N/A"
            }
        }
        cleanedResult = cleanData(result)
        data.cache.put(cacheKey, cleanedResult)
        return cleanedResult


# Return descriptive profile and category for single sector ETF
def fetchSectorProfile(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, sectorOrTicker: str) -> Dict[str, Any]:
    sectorOrTicker, wasUpdated = normaliseSectorInput(sectorOrTicker)
    ticker, resolvedName, note = data.sectors.resolveSector(sectorOrTicker)
    if resolvedName == "Unknown":
        return attachSectorWarning(cleanData({
            "status": "unknown_sector",
            "query": sectorOrTicker,
            "message": note or "Sector is marked as 'unknown' in the dataset. No GICS sector ETF applies."
        }), wasUpdated)
    if not ticker or ticker not in GICS_SECTORS:
        return attachSectorWarning(cleanData({"error": note or f"Sector or ticker '{sectorOrTicker}' not recognised among the 11 GICS sectors."}), wasUpdated)

    info = GICS_SECTORS[ticker]
    result = {
        "ticker": info.ticker,
        "name": info.name,
        "category": info.category,
        "description": info.description
    }
    if note:
        result["resolutionNote"] = note
    return attachSectorWarning(cleanData(result), wasUpdated)


# Retrieve performance metrics and technicals across all 11 GICS sectors in parallel
def fetchAllSectorsPerformance(tool: Tool, data: DataProviders, timestamp: pd.Timestamp) -> Dict[str, Any]:
    tsNy = timestamp.tz_convert(NEW_YORK) if timestamp.tzinfo is not None else timestamp.tz_localize(NEW_YORK)
    effectiveTs = min(tsNy, END_DATE)
    cacheKey = f"sector|all_perf_{effectiveTs.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    with data.sectors.keyedLocks.lockKey(cacheKey):
        cached = data.cache.get(cacheKey)
        if cached is not None:
            return cached

        results = []
        totalSectors = len(GICS_SECTORS)
        completedSectors = 0

        with ThreadPoolExecutor(max_workers=min(totalSectors, 8)) as executor:
            futures = {executor.submit(fetchSectorPerformance, tool, data, effectiveTs, ticker): ticker for ticker in GICS_SECTORS}
            for future in as_completed(futures):
                completedSectors += 1
                tool.updateProgress((completedSectors / totalSectors) * 100.0)
                try:
                    res = future.result()
                    if res and not res.get("error"):
                        results.append(res)
                except Exception:
                    pass

        results.sort(key=lambda x: x.get("ticker", ""))
        cleanedResult = cleanData(results)
        data.cache.put(cacheKey, cleanedResult)
        return cleanedResult


# Return metadata and descriptions for all 11 GICS sectors
def fetchAllSectorProfiles(tool: Tool, data: DataProviders, timestamp: pd.Timestamp) -> Dict[str, Any]:
    profiles = []
    for ticker, info in GICS_SECTORS.items():
        profiles.append({
            "ticker": info.ticker,
            "name": info.name,
            "category": info.category,
            "description": info.description
        })
    profiles.sort(key=lambda x: x.get("ticker", ""))
    return cleanData(profiles)


# Calculate historical price ratio channel percentiles relative to SPY
def calculateChannelPercentiles(secClosesDf: pd.DataFrame, spyClosesDf: pd.DataFrame) -> Dict[str, Optional[float]]:
    if secClosesDf.empty or spyClosesDf.empty:
        return {"channel1YearPercentile": None, "channel3YearPercentile": None}

    try:
        secCopy = secClosesDf.copy()
        spyCopy = spyClosesDf.copy()
        secCopy["date"] = pd.to_datetime(secCopy["date"], utc=True).dt.tz_localize(None).dt.normalize()
        spyCopy["date"] = pd.to_datetime(spyCopy["date"], utc=True).dt.tz_localize(None).dt.normalize()

        secCopy = secCopy.drop_duplicates(subset=["date"]).sort_values("date")
        spyCopy = spyCopy.drop_duplicates(subset=["date"]).sort_values("date")

        merged = pd.merge(
            secCopy[["date", "close"]].rename(columns={"close": "secClose"}),
            spyCopy[["date", "close"]].rename(columns={"close": "spyClose"}),
            on="date"
        ).sort_values("date").reset_index(drop=True)

        if len(merged) < 20:
            return {"channel1YearPercentile": None, "channel3YearPercentile": None}

        merged["ratio"] = merged["secClose"] / merged["spyClose"]
        currentRatio = float(merged["ratio"].iloc[-1])

        # 1-Year trailing channel percentile
        slice1y = merged["ratio"].tail(min(252, len(merged)))
        min1y = float(slice1y.min())
        max1y = float(slice1y.max())
        pct1y = round(((currentRatio - min1y) / (max1y - min1y) * 100), 1) if max1y > min1y else 50.0

        # 3-Year trailing channel percentile
        slice3y = merged["ratio"].tail(min(756, len(merged)))
        min3y = float(slice3y.min())
        max3y = float(slice3y.max())
        pct3y = round(((currentRatio - min3y) / (max3y - min3y) * 100), 1) if max3y > min3y else 50.0

        return {
            "channel1YearPercentile": pct1y,
            "channel3YearPercentile": pct3y
        }
    except Exception:
        return {"channel1YearPercentile": None, "channel3YearPercentile": None}


# Calculate interest rate sensitivity beta against 10-Year Treasury yield changes
def calculateTreasuryBeta(secClosesDf: pd.DataFrame, treasDf: pd.DataFrame) -> Dict[str, Any]:
    if secClosesDf.empty or treasDf.empty:
        return {"treasury10yBeta": 0.0, "sensitivity": "Insufficient rate history"}

    try:
        secCopy = secClosesDf.copy()
        treasCopy = treasDf.copy()

        secCopy["date"] = pd.to_datetime(secCopy["date"], utc=True).dt.tz_localize(None).dt.normalize()
        treasCopy["date"] = pd.to_datetime(treasCopy["date"], utc=True).dt.tz_localize(None).dt.normalize()

        secCopy = secCopy.drop_duplicates(subset=["date"]).sort_values("date")
        treasCopy = treasCopy.drop_duplicates(subset=["date"]).sort_values("date")

        secRet = secCopy.set_index("date")["close"].pct_change() * 100.0
        treasDiff = treasCopy.set_index("date")["value"].diff()

        merged = pd.concat([secRet.rename("ret"), treasDiff.rename("diff")], axis=1).dropna().tail(90)
        if len(merged) < 20:
            return {"treasury10yBeta": 0.0, "sensitivity": "Insufficient rate history"}

        varDiff = float(merged["diff"].var())
        covar = float(merged["ret"].cov(merged["diff"]))
        beta = round(covar / varDiff, 2) if varDiff > 0 else 0.0

        if beta > 0.15:
            label = "Positive rate sensitivity (benefits from rising yields)"
        elif beta < -0.15:
            label = "Negative rate sensitivity (vulnerable to rising yields)"
        else:
            label = "Yield neutral / low rate sensitivity"

        return {
            "treasury10yBeta": beta,
            "sensitivity": label
        }
    except Exception:
        return {"treasury10yBeta": 0.0, "sensitivity": "Insufficient rate history"}


# Subsample sector articles based on active hardware inference engine
def subsampleSectorArticles(newsDf: pd.DataFrame) -> pd.DataFrame:
    if newsDf is None or newsDf.empty:
        return newsDf

    try:
        engine = getSentimentEngine()
        engineType = type(engine)

        if engineType is TrtCudaInferenceEngine:
            return newsDf
        elif engineType is OnnxCudaInferenceEngine:
            return newsDf.iloc[::2].reset_index(drop=True)
        elif engineType is PytorchCudaInferenceEngine:
            keepIndices = [i for i in range(len(newsDf)) if i % 5 in (0, 2)]
            return newsDf.iloc[keepIndices].reset_index(drop=True)
        else:
            return newsDf.iloc[::8].reset_index(drop=True)
    except Exception:
        return newsDf.iloc[::2].reset_index(drop=True)


# Execute comprehensive quantitative, fundamental, and FinBERT sentiment analysis across all 11 sectors
def fetchAllSectorsAnalysis(tool: Tool, data: DataProviders, timestamp: pd.Timestamp) -> Dict[str, Any]:
    tsNy = timestamp.tz_convert(NEW_YORK) if timestamp.tzinfo is not None else timestamp.tz_localize(NEW_YORK)
    effectiveTs = min(tsNy, END_DATE)
    effectiveNorm = effectiveTs.tz_localize(None) if effectiveTs.tzinfo is not None else effectiveTs
    cacheKey = f"sector|all_analysis_{effectiveTs.strftime('%Y-%m-%dH%H')}"

    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    with data.sectors.keyedLocks.lockKey(cacheKey):
        cached = data.cache.get(cacheKey)
        if cached is not None:
            return cached

        # Pre-load S&P 500 and 10-Year Treasury benchmark historical series
        spyClosesDf = pd.DataFrame()
        try:
            sp500Raw = data.macro.loadSeries(MacroSeries.SP500)
            if sp500Raw is not None and not sp500Raw.empty:
                validSpy = sp500Raw.copy()
                validSpy["date"] = pd.to_datetime(validSpy["date"], utc=True).dt.tz_localize(None)
                validSpy = validSpy[validSpy["date"] <= effectiveNorm].sort_values("date").reset_index(drop=True)
                spyClosesDf = validSpy[["date", "close"]]
        except Exception:
            pass

        treasDf = pd.DataFrame()
        try:
            treasRaw = data.macro.loadSeries(MacroSeries.TREAS_10Y)
            if treasRaw is not None and not treasRaw.empty:
                validTreas = treasRaw.copy()
                validTreas["date"] = pd.to_datetime(validTreas["date"], utc=True).dt.tz_localize(None)
                treasDf = validTreas[validTreas["date"] <= effectiveNorm].sort_values("date").reset_index(drop=True)
        except Exception:
            pass

        benchReturns = {}
        if not spyClosesDf.empty:
            latestSpyClose = float(spyClosesDf["close"].iloc[-1])
            spyPastDates = {
                "1mo": effectiveNorm - pd.DateOffset(months=1),
                "3mo": effectiveNorm - pd.DateOffset(months=3),
                "6mo": effectiveNorm - pd.DateOffset(months=6),
                "12mo": effectiveNorm - pd.DateOffset(years=1)
            }
            for period, pastDate in spyPastDates.items():
                pastRows = spyClosesDf[spyClosesDf["date"] <= pastDate]
                if not pastRows.empty:
                    pastSpyClose = float(pastRows["close"].iloc[-1])
                    benchReturns[period] = round(((latestSpyClose - pastSpyClose) / pastSpyClose) * 100.0, 2)

        oneYearAgo = effectiveNorm - pd.DateOffset(years=1, weeks=1)
        threeMoAgo = effectiveNorm - pd.DateOffset(months=3)

        sectorHoldingsData = {}
        sectorNewsDfs = {}
        sectorMarketCapSums = {}
        allUniqueHeadlines = set()

        totalSectors = len(DB_SECTOR_TO_TICKER)
        completedInit = 0

        leadersMap = data.sectorLeaders.getAllSectorLeaders(effectiveTs, limit=25)

        for dbKey, etfTicker in sorted(DB_SECTOR_TO_TICKER.items()):
            candidateProfiles = [
                p for p in data.tickers.tickerIndex.values()
                if p.sector == dbKey
                and p.industry and p.industry.lower() != "unknown"
                and data.tickers.isTickerListed(p.ticker, effectiveTs)
                and p.ticker in data.ohlcv.tickersPaths
            ]

            cleanIndustries = sorted(list(set(
                p.industry for p in candidateProfiles if p.industry and p.industry.lower() != "unknown"
            )))

            top25Tickers = leadersMap.get(etfTicker, [])
            top25Caps = []
            for t in top25Tickers:
                p = data.tickers.getTickerProfile(t)
                if p is not None:
                    row = data.ohlcv.getSingleDayTickerData(p.ticker, effectiveTs)
                    capStr = str(row["marketCap"]) if row is not None and "marketCap" in row else "N/A"
                    capNum = parseMarketCapValue(capStr)
                    top25Caps.append((p, capNum, capStr))

            sectorCapSum = sum(h[1] for h in top25Caps if h[1] is not None and h[1] > 0)
            sectorMarketCapSums[etfTicker] = sectorCapSum

            # Evaluate 1-year OHLCV for top 25 holdings for breadth and constituent divergence
            def evalTopHolding(item):
                p, capNum, capStr = item
                df = data.ohlcv.getPeriodDailyTickerData(p.ticker, startDate=oneYearAgo, endDate=effectiveTs)
                ret3mo = 0.0
                aboveSma50 = False
                aboveSma200 = False
                if df is not None and len(df) >= 50:
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
                    c = df["close"]
                    latest = float(c.iloc[-1])
                    s50 = float(c.rolling(50).mean().iloc[-1])
                    s200 = float(c.rolling(min(200, len(c))).mean().iloc[-1])
                    aboveSma50 = latest > s50
                    aboveSma200 = latest > s200

                    valid3mo = df[df["date"] <= threeMoAgo]["close"]
                    if not valid3mo.empty:
                        past3mo = float(valid3mo.iloc[-1])
                        ret3mo = round(((latest - past3mo) / past3mo) * 100.0, 2)

                return (p.ticker, p.name, p.industry, capNum, capStr, ret3mo, aboveSma50, aboveSma200)

            with ThreadPoolExecutor(max_workers=16) as executor:
                evaluatedTopHoldings = [r for r in executor.map(evalTopHolding, top25Caps) if r is not None]

            totalEval = len(evaluatedTopHoldings)
            pctAbove50 = round((sum(1 for h in evaluatedTopHoldings if h[6]) / totalEval) * 100.0, 1) if totalEval else 0.0
            pctAbove200 = round((sum(1 for h in evaluatedTopHoldings if h[7]) / totalEval) * 100.0, 1) if totalEval else 0.0
            rets3mo = [h[5] for h in evaluatedTopHoldings]
            medRet3mo = round(float(np.median(rets3mo)), 2) if rets3mo else 0.0
            top5 = evaluatedTopHoldings[:5]

            # Fetch recent sector headlines for qualitative agent review
            recentNewsDf = data.news.getRecentSectorNews(etfTicker, effectiveTs, limit=20, maxOtherSectorTickers=2)
            agentHeadlines = []
            if not recentNewsDf.empty and "headline" in recentNewsDf.columns:
                for _, r in recentNewsDf.iterrows():
                    hl = cleanHtmlContent(str(r.get("headline", "")).strip())
                    dtVal = r.get("date")
                    age = formatArticleAge(dtVal, effectiveTs)
                    if hl:
                        agentHeadlines.append(f"{hl} ({age})")

            # Collect constituent news for FinBERT neural sentiment scoring
            candidateTickers = [etfTicker] + top25Tickers
            rawNewsDf = data.news.getSectorConstituentsNews(candidateTickers, effectiveTs, limit=400, maxReferencedTickers=10)
            newsDf = subsampleSectorArticles(rawNewsDf)
            sectorNewsDfs[etfTicker] = newsDf

            if not newsDf.empty and "headline" in newsDf.columns:
                hls = newsDf["headline"].dropna().tolist()
                allUniqueHeadlines.update(hls)

            sectorHoldingsData[etfTicker] = {
                "dbKey": dbKey,
                "cleanIndustries": cleanIndustries,
                "recentHeadlines": agentHeadlines,
                "top5": [
                    {
                        "ticker": h[0],
                        "name": h[1],
                        "industry": h[2],
                        "marketCap": h[4],
                        "return3mo": h[5]
                    } for h in top5
                ],
                "pctAboveSma50": pctAbove50,
                "pctAboveSma200": pctAbove200,
                "medianConstituentReturn3mo": medRet3mo
            }

            completedInit += 1
            tool.updateProgress((completedInit / totalSectors) * 50.0)

        # Batch-score unique headlines across sectors with FinBERT
        tool.updateProgress(50.0)

        uniqueHeadlinesList = list(allUniqueHeadlines)
        totalHeadlinesToScore = len(uniqueHeadlinesList)
        scoredCount = [0]

        def onFinbertProgress(increment):
            if tool is not None and totalHeadlinesToScore > 0:
                scoredCount[0] += increment
                prog = 50.0 + (scoredCount[0] / totalHeadlinesToScore) * 50.0
                tool.updateProgress(prog)

        if uniqueHeadlinesList:
            scoreTextsWithCache(uniqueHeadlinesList, data=data, onProgressCallback=onFinbertProgress)

        tool.updateProgress(92.0)

        totalMarketCapAll = sum(sectorMarketCapSums.values()) or 1.0

        # Compile consolidated 11-sector analytics report
        sectorsResults = []
        for dbKey, etfTicker in sorted(DB_SECTOR_TO_TICKER.items()):
            info = GICS_SECTORS.get(etfTicker)
            sectorName = info.name if info else etfTicker
            category = info.category if info else "Unknown"

            holdingsMeta = sectorHoldingsData.get(etfTicker, {})
            cleanIndustries = holdingsMeta.get("cleanIndustries", [])
            top5Holdings = holdingsMeta.get("top5", [])
            pctAbove50 = holdingsMeta.get("pctAboveSma50", 0.0)
            pctAbove200 = holdingsMeta.get("pctAboveSma200", 0.0)
            medRet3mo = holdingsMeta.get("medianConstituentReturn3mo", 0.0)

            capWeightPct = round((sectorMarketCapSums.get(etfTicker, 0.0) / totalMarketCapAll) * 100.0, 1)

            startDate = effectiveNorm - pd.DateOffset(years=4)
            secDf = data.sectors.getSectorData(etfTicker, startDate=startDate, endDate=effectiveTs)
            if not secDf.empty and "date" in secDf.columns:
                secDf["date"] = pd.to_datetime(secDf["date"], utc=True).dt.tz_localize(None)

            trailingReturns = {}
            relativeAlpha = {}
            technicals = {}
            channelPercentiles = {"channel1YearPercentile": 50.0, "channel3YearPercentile": 50.0}
            treasurySensitivity = {"treasury10yBeta": 0.0, "sensitivity": "Neutral"}
            etfReturn3mo = 0.0

            if not secDf.empty:
                latestClose = float(secDf["close"].iloc[-1])
                pastDates = {
                    "1mo": effectiveNorm - pd.DateOffset(months=1),
                    "3mo": effectiveNorm - pd.DateOffset(months=3),
                    "6mo": effectiveNorm - pd.DateOffset(months=6),
                    "12mo": effectiveNorm - pd.DateOffset(years=1)
                }
                for period, pastDate in pastDates.items():
                    pastRows = secDf[secDf["date"] <= pastDate]
                    if not pastRows.empty:
                        pastClose = float(pastRows["close"].iloc[-1])
                        retVal = round(((latestClose - pastClose) / pastClose) * 100.0, 2)
                        trailingReturns[period] = retVal
                        if period == "3mo":
                            etfReturn3mo = retVal

                        spyRet = benchReturns.get(period)
                        if spyRet is not None:
                            relativeAlpha[period] = round(retVal - spyRet, 2)

                rawTechnicals = calculateSectorTechnicals(secDf, latestClose)
                range52 = calculateSectorRange52Week(secDf, effectiveTs, latestClose)
                technicals = {
                    "rsi14": rawTechnicals.get("rsi14"),
                    "distFromSma50Pct": rawTechnicals.get("distFromSma50Pct"),
                    "distFromSma200Pct": rawTechnicals.get("distFromSma200Pct"),
                    "drawdownFromHighPct": range52.get("drawdownFromHighPct"),
                    "trend": rawTechnicals.get("trend")
                }

                if not spyClosesDf.empty:
                    channelPercentiles = calculateChannelPercentiles(secDf, spyClosesDf)
                if not treasDf.empty:
                    treasurySensitivity = calculateTreasuryBeta(secDf, treasDf)

            divergencePct = round(etfReturn3mo - medRet3mo, 2)
            if divergencePct > 7.0 and pctAbove50 < 50.0:
                rallyChar = "Narrow mega-cap rally (low breadth, caution on broad exposure)"
            elif divergencePct < -5.0:
                rallyChar = "Broad equal-weight outperformance (improving underlying breadth)"
            elif pctAbove50 >= 60.0:
                rallyChar = "Broad healthy participation across constituents"
            else:
                rallyChar = "Moderate mixed constituent participation"

            newsDf = sectorNewsDfs.get(etfTicker, pd.DataFrame())
            sentimentPayload = {
                "newsSentimentScore": 0.0,
                "newsSentimentRating": "Stable neutral sentiment",
                "articlesAnalysed": 0,
                "dateSpan": "None"
            }

            if not newsDf.empty and "headline" in newsDf.columns:
                validRows = newsDf.dropna(subset=["headline"])
                hls = validRows["headline"].tolist()
                hldates = validRows["date"].tolist() if "date" in validRows.columns else [None] * len(hls)
                preds = scoreTextsWithCache(hls, data=data)
                if preds:
                    aggScore, aggRating = aggregateSentiment(preds, hldates, effectiveTs)
                    if aggScore is None:
                        aggScore, aggRating = 0.0, deriveSentimentRating(0.0)
                    avgScore = round(float(aggScore), 4)
                    rating = aggRating

                    dMin = str(newsDf["date"].min())[:10]
                    dMax = str(newsDf["date"].max())[:10]

                    sentimentPayload = {
                        "newsSentimentScore": avgScore,
                        "newsSentimentRating": rating,
                        "articlesAnalysed": len(hls),
                        "dateSpan": f"{dMin} to {dMax}"
                    }

            sectorsResults.append({
                "sectorKey": dbKey,
                "ticker": etfTicker,
                "sectorName": sectorName,
                "category": category,
                "benchmarkWeightPct": capWeightPct,
                "industries": cleanIndustries,
                "trailingReturns": trailingReturns,
                "relativeAlphaVsSP500": relativeAlpha,
                "channelPercentiles": channelPercentiles,
                "technicals": technicals,
                "breadth": {
                    "pctAboveSma50": pctAbove50,
                    "pctAboveSma200": pctAbove200
                },
                "constituentDivergence": {
                    "etfReturn3mo": etfReturn3mo,
                    "medianConstituentReturn3mo": medRet3mo,
                    "divergencePct": divergencePct,
                    "rallyCharacter": rallyChar
                },
                "macroSensitivity": treasurySensitivity,
                "topHoldings": top5Holdings,
                "recentHeadlines": holdingsMeta.get("recentHeadlines", []),
                "newsSentiment": sentimentPayload
            })

        sectorsResults.sort(key=lambda s: s.get("benchmarkWeightPct", 0.0), reverse=True)

        finalOutput = {
            "asOfDate": effectiveTs.strftime("%Y-%m-%d"),
            "sectorsCount": len(sectorsResults),
            "benchmarkSP500Returns": benchReturns,
            "sectors": sectorsResults
        }

        cleanedFinal = cleanData(finalOutput)
        data.cache.put(cacheKey, cleanedFinal)

        tool.updateProgress(100.0)
        return cleanedFinal
