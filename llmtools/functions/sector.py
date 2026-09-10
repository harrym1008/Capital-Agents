import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from collectors.sector_dl_client import GICS_SECTORS, DB_SECTOR_TO_TICKER
from collectors.constants import UTC
from dataquery.macro_provider import MacroSeries
from llmtools.tool_registry import DataProviders, Tool
from llmtools.functions.helpers import cleanData


def calculateRsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0  # Default neutral when insufficient data

    deltas = series.diff()
    gains = deltas.clip(lower=0)
    losses = -1 * deltas.clip(upper=0)

    avgGain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avgLoss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avgGain / avgLoss
    rsiSeries = 100 - (100 / (1 + rs))
    rsiSeries = rsiSeries.replace(np.inf, 100)
    return float(rsiSeries.iloc[-1])


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


def calculateAnnualisedVolatility(df: pd.DataFrame, window: int = 90) -> float:
    recentCloses = df["close"].tail(min(window, len(df)))
    dailyReturns = recentCloses.pct_change().dropna()
    if len(dailyReturns) > 1:
        return round(float(dailyReturns.std() * np.sqrt(252) * 100), 2)
    return 0.0


def fetchSectorPerformance(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, sectorOrTicker: str) -> Dict[str, Any]:
    cacheKey = f"sector|perf_{sectorOrTicker}_{timestamp.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    ticker, resolvedName, note = data.sectors.resolveSector(sectorOrTicker)
    if resolvedName == "Unknown":
        return cleanData({
            "status": "unknown_sector",
            "query": sectorOrTicker,
            "message": note or "Sector is listed as 'unknown' in the dataset. No GICS sector ETF applies."
        })
    if not ticker:
        return cleanData({"error": note or f"Sector or ticker '{sectorOrTicker}' not recognised among the 11 GICS sectors."})

    info = GICS_SECTORS.get(ticker)
    sectorName = info.name if info else ticker
    sectorCategory = info.category if info else "Unknown"

    startDate = timestamp - pd.DateOffset(years=5, weeks=2)
    df = data.sectors.getSectorData(ticker, startDate=startDate, endDate=timestamp)

    if df.empty:
        if ticker == "XLC" and timestamp < pd.Timestamp("2018-06-19"):
            return cleanData({
                "ticker": ticker,
                "sector": sectorName,
                "notice": "XLC (Communication Services) was launched in June 2018. Historical data is not available before 2018-06-19."
            })
        return cleanData({"error": f"No historical data available for sector ETF {ticker} before {timestamp.strftime('%Y-%m-%d')}."})

    latestRow = df.iloc[-1]
    latestClose = float(latestRow["close"])
    latestDateStr = latestRow["date"].strftime("%Y-%m-%d")

    pastPeriods = {
        "1d": timestamp - pd.DateOffset(days=1),
        "5d": timestamp - pd.DateOffset(weeks=1),
        "1mo": timestamp - pd.DateOffset(months=1),
        "3mo": timestamp - pd.DateOffset(months=3),
        "6mo": timestamp - pd.DateOffset(months=6),
        "12mo": timestamp - pd.DateOffset(years=1),
        "3y": timestamp - pd.DateOffset(years=3),
        "5y": timestamp - pd.DateOffset(years=5)
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

    benchReturns = calculateBenchmarkReturns(data, timestamp, pastPeriods)
    relativeAlpha = {}
    for period in ["5d", "1mo", "3mo", "6mo", "12mo"]:
        secRet = returns.get(period)
        spyRet = benchReturns.get(period)
        if secRet is not None and spyRet is not None:
            relativeAlpha[period] = round(secRet - spyRet, 2)

    technicals = calculateSectorTechnicals(df, latestClose)
    range52Week = calculateSectorRange52Week(df, timestamp, latestClose)
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
    return cleanedResult


def fetchAllSectorRankings(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, lookback: str = "1mo") -> Dict[str, Any]:
    validLookbacks = ["5d", "1mo", "3mo", "6mo", "12mo"]
    if lookback not in validLookbacks:
        lookback = "1mo"

    cacheKey = f"sector|leaderboard_{lookback}_{timestamp.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    with data.sectors.keyedLocks.lockKey(cacheKey):
        cached = data.cache.get(cacheKey)
        if cached is not None:
            return cached

        lookbackDeltas = {
            "5d": timestamp - pd.DateOffset(weeks=1),
            "1mo": timestamp - pd.DateOffset(months=1),
            "3mo": timestamp - pd.DateOffset(months=3),
            "6mo": timestamp - pd.DateOffset(months=6),
            "12mo": timestamp - pd.DateOffset(years=1)
        }
        pastDate = lookbackDeltas.get(lookback, timestamp - pd.DateOffset(months=1))
        benchReturns = calculateBenchmarkReturns(data, timestamp, {lookback: pastDate})
        spyReturn = benchReturns.get(lookback)

        rankings = []
        startDate = timestamp - pd.DateOffset(years=2)
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

        with ThreadPoolExecutor(max_workers=min(len(GICS_SECTORS), 8)) as executor:
            futures = [executor.submit(processSector, ticker, info) for ticker, info in GICS_SECTORS.items()]
            for future in as_completed(futures):
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
            "asOfDate": timestamp.strftime("%Y-%m-%d"),
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


def fetchSectorProfile(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, sectorOrTicker: str) -> Dict[str, Any]:
    ticker, resolvedName, note = data.sectors.resolveSector(sectorOrTicker)
    if resolvedName == "Unknown":
        return cleanData({
            "status": "unknown_sector",
            "query": sectorOrTicker,
            "message": note or "Sector is marked as 'unknown' in the dataset. No GICS sector ETF applies."
        })
    if not ticker or ticker not in GICS_SECTORS:
        return cleanData({"error": note or f"Sector or ticker '{sectorOrTicker}' not recognised among the 11 GICS sectors."})

    info = GICS_SECTORS[ticker]
    result = {
        "ticker": info.ticker,
        "name": info.name,
        "category": info.category,
        "description": info.description,
        "etfIssuer": "State Street Global Advisors (Select Sector SPDR)"
    }
    if note:
        result["resolutionNote"] = note
    return cleanData(result)


def confirmSectorAllocation(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, sectorAllocations: Dict[str, float], rationale: str) -> Dict[str, Any]:
    if not isinstance(sectorAllocations, dict) or not sectorAllocations:
        return {"error": "sectorAllocations must be a non-empty dictionary mapping sector names to percentage numbers."}

    cleanedAllocations = {}
    totalAllocated = 0.0

    for rawSector, rawPct in sectorAllocations.items():
        try:
            pctVal = round(float(rawPct), 2)
        except (ValueError, TypeError):
            return {"error": f"Allocation value for '{rawSector}' must be a valid number, got '{rawPct}'."}

        if pctVal <= 0:
            continue

        rawLower = str(rawSector).strip().lower()
        if rawLower in ["cash", "usd"]:
            cleanedAllocations["CASH"] = {
                "sector": "Cash",
                "ticker": "CASH",
                "allocationPct": pctVal
            }
            totalAllocated += pctVal
            continue

        ticker, resolvedName, note = data.sectors.resolveSector(rawSector)
        if resolvedName == "Unknown" or not ticker:
            return {"error": f"Sector '{rawSector}' could not be resolved to a valid GICS sector."}

        cleanedAllocations[ticker] = {
            "sector": resolvedName,
            "ticker": ticker,
            "allocationPct": pctVal
        }
        totalAllocated += pctVal

    # Check if total sums to ~100%
    totalAllocated = round(totalAllocated, 2)
    if totalAllocated < 95.0 or totalAllocated > 105.0:
        return {
            "error": f"Total sector allocations must sum to approximately 100.0%. Current sum: {totalAllocated}%. Please rebalance and retry.",
            "currentSum": totalAllocated,
            "currentAllocations": cleanedAllocations
        }

    decisionRecord = {
        "sectorAllocations": cleanedAllocations,
        "totalAllocatedPct": totalAllocated,
        "sectorCount": len([k for k in cleanedAllocations if k != "CASH"]),
        "rationale": str(rationale).strip()
    }

    tool.toolLog.append(decisionRecord)
    return cleanData({
        "status": "success",
        "message": f"Sector allocation confirmed with {len(cleanedAllocations)} sectors totaling {totalAllocated}%.",
        "confirmedAllocation": decisionRecord
    })

