import os
import re
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


def fetchAllSectorsPerformance(tool: Tool, data: DataProviders, timestamp: pd.Timestamp) -> Dict[str, Any]:
    cacheKey = f"sector|all_perf_{timestamp.strftime('%Y-%m-%dH%H')}"
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
            futures = {executor.submit(fetchSectorPerformance, tool, data, timestamp, ticker): ticker for ticker in GICS_SECTORS}
            for future in as_completed(futures):
                completedSectors += 1
                tool.updateProgress((completedSectors / totalSectors) * 100.0)
                try:
                    res = future.result()
                    if res and not res.get("error"):
                        results.append(res)
                except Exception:
                    pass

        # Sort predictably by ticker
        results.sort(key=lambda x: x.get("ticker", ""))
        cleanedResult = cleanData(results)
        data.cache.put(cacheKey, cleanedResult)
        return cleanedResult


def fetchAllSectorProfiles(tool: Tool, data: DataProviders, timestamp: pd.Timestamp) -> Dict[str, Any]:
    profiles = []
    for ticker, info in GICS_SECTORS.items():
        profiles.append({
            "ticker": info.ticker,
            "name": info.name,
            "category": info.category,
            "description": info.description,
            "etfIssuer": "State Street Global Advisors (Select Sector SPDR)"
        })
    profiles.sort(key=lambda x: x.get("ticker", ""))
    return cleanData(profiles)


from llmtools.functions.confirmation import confirmSectorAllocation


def parseMarketCapValue(val: Any) -> float:
    if not val or pd.isna(val):
        return 0.0
    s = str(val).strip().lower()
    m = re.match(r"([0-9.]+)\s*([a-z]+)?", s)
    if not m:
        return 0.0
    num = float(m.group(1))
    unit = m.group(2) or ""
    if "t" in unit:
        return num * 1e12
    if "b" in unit:
        return num * 1e9
    if "m" in unit:
        return num * 1e6
    if "k" in unit:
        return num * 1e3
    return num


def fetchStocksInSector(
    tool: Tool,
    data: DataProviders,
    timestamp: pd.Timestamp,
    sector: str,
    style: str = "all",
    limit: int = 12
) -> Dict[str, Any]:
    limit = max(4, min(int(limit) if limit else 12, 20))
    style = str(style).strip().lower() if style else "all"
    if style not in ["growth", "value", "defensive", "all"]:
        style = "all"

    ticker, resolvedName, note = data.sectors.resolveSector(sector)
    if resolvedName == "Unknown" or not ticker or ticker not in GICS_SECTORS:
        return cleanData({"error": note or f"Sector '{sector}' not recognised among the 11 GICS sectors."})

    dbKey = next((k for k, v in DB_SECTOR_TO_TICKER.items() if v == ticker), None)
    if not dbKey:
        return cleanData({"error": f"Could not map sector ETF '{ticker}' to internal database category."})

    cacheKey = f"sector|stocks_{ticker}_{style}_{limit}_{timestamp.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    candidateProfiles = [
        p for p in data.tickers.tickerIndex.values()
        if p.sector == dbKey and not p.isAdrc and data.tickers.isTickerListed(p.ticker, timestamp)
    ]

    if not candidateProfiles:
        return cleanData({
            "status": "no_candidates",
            "sector": resolvedName,
            "sectorEtf": ticker,
            "message": f"No listed common stock equities found for sector '{resolvedName}' as of {timestamp.strftime('%Y-%m-%d')}."
        })

    validProfiles = [p for p in candidateProfiles if p.ticker in data.ohlcv.tickersPaths]
    if not validProfiles:
        validProfiles = candidateProfiles[:30]

    normTs = timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    oneYearAgo = normTs - pd.DateOffset(years=1)
    threeMonthsAgo = normTs - pd.DateOffset(months=3)

    evaluatedList = []

    def processStock(profile):
        t = profile.ticker
        path = data.ohlcv.tickersPaths.get(t)
        if not path or not os.path.exists(path):
            return None
        try:
            df = pd.read_parquet(path)
            if df.empty or "close" not in df.columns or "date" not in df.columns:
                return None

            dateCol = pd.to_datetime(df["date"])
            if dateCol.dt.tz is not None:
                dateCol = dateCol.dt.tz_convert(UTC).dt.tz_localize(None)
            df["normDate"] = dateCol

            validDf = df[df["normDate"] <= normTs]
            if len(validDf) < 20:
                return None

            lastRow = validDf.iloc[-1]
            latestClose = float(lastRow["close"])
            if latestClose <= 0:
                return None

            capStr = str(lastRow.get("marketCap", "N/A"))
            capNum = parseMarketCapValue(capStr)

            # 3-Month Return
            df3m = validDf[validDf["normDate"] <= threeMonthsAgo]
            if not df3m.empty:
                c3m = float(df3m["close"].iloc[-1])
                return3m = round(((latestClose - c3m) / c3m) * 100, 1)
            else:
                return3m = 0.0

            # 12-Month Return
            df12m = validDf[validDf["normDate"] <= oneYearAgo]
            if not df12m.empty:
                c12m = float(df12m["close"].iloc[-1])
                return12m = round(((latestClose - c12m) / c12m) * 100, 1)
            else:
                return12m = return3m

            # 52-Week High and Drawdown
            pastYearDf = validDf[validDf["normDate"] >= oneYearAgo]
            high52 = float(pastYearDf["high"].max() if "high" in pastYearDf.columns else pastYearDf["close"].max()) if not pastYearDf.empty else latestClose
            drawdown52w = round(((latestClose - high52) / high52) * 100, 1) if high52 > 0 else 0.0

            # RSI 14
            rsi14 = round(calculateRsi(validDf["close"].tail(50), period=14), 1)

            # Annualized 90-day volatility
            recentCloses = validDf["close"].tail(min(90, len(validDf)))
            dailyReturns = recentCloses.pct_change().dropna()
            vol90d = round(float(dailyReturns.std() * np.sqrt(252) * 100), 1) if len(dailyReturns) > 1 else 25.0

            tags = []
            if capNum >= 1e12:
                tags.append("Mega-Cap Anchor")
            elif capNum >= 5e10:
                tags.append("Large-Cap Core")
            elif capNum >= 1e10:
                tags.append("Mid/Large-Cap")
            else:
                tags.append("Small/Mid-Cap")

            if return3m >= 15.0 and rsi14 >= 55.0:
                tags.append("High Momentum Growth")
            elif return3m <= -10.0 and rsi14 <= 40.0:
                tags.append("Oversold Rebound Potential")

            if vol90d <= 22.0 and drawdown52w >= -12.0:
                tags.append("Low Volatility Defensive")

            return {
                "ticker": t,
                "name": profile.name,
                "industry": profile.industry,
                "marketCap": capStr,
                "marketCapNum": capNum,
                "latestPrice": round(latestClose, 2),
                "trailingReturn3mo": f"{return3m:+0.1f}%",
                "trailingReturn12mo": f"{return12m:+0.1f}%",
                "rsi14": rsi14,
                "distFrom52wHigh": f"{drawdown52w:+0.1f}%",
                "volatility90d": f"{vol90d:.1f}%",
                "styleTags": tags,
                "_rawRet3m": return3m,
                "_rawRet12m": return12m,
                "_rawVol": vol90d,
                "_rawDrawdown": drawdown52w
            }
        except Exception:
            return None

    industryGroups: Dict[str, List[Any]] = {}
    for p in validProfiles:
        ind = p.industry or "general"
        industryGroups.setdefault(ind, []).append(p)

    def getProfileFileSize(p):
        path = data.ohlcv.tickersPaths.get(p.ticker)
        return os.path.getsize(path) if path and os.path.exists(path) else 0

    samplePool = []
    for ind, group in industryGroups.items():
        sortedGroup = sorted(group, key=getProfileFileSize, reverse=True)
        samplePool.extend(sortedGroup[:7])

    if len(samplePool) < 25:
        remaining = [p for p in validProfiles if p not in samplePool]
        remainingSorted = sorted(remaining, key=getProfileFileSize, reverse=True)
        samplePool.extend(remainingSorted[:max(0, 35 - len(samplePool))])

    with ThreadPoolExecutor(max_workers=min(len(samplePool), 12)) as executor:
        futures = [executor.submit(processStock, p) for p in samplePool]
        for f in as_completed(futures):
            res = f.result()
            if res:
                evaluatedList.append(res)

    if not evaluatedList:
        return cleanData({"error": f"Unable to retrieve validated price metrics for sector '{resolvedName}'."})

    if style == "growth":
        evaluatedList.sort(key=lambda x: (x["_rawRet3m"] * 0.7 + x["_rawRet12m"] * 0.3), reverse=True)
    elif style in ["value", "defensive"]:
        evaluatedList.sort(key=lambda x: (x["_rawVol"] - x["_rawDrawdown"]))
    else:
        evaluatedList.sort(key=lambda x: x["marketCapNum"], reverse=True)

    finalCandidates = evaluatedList[:limit]
    for c in finalCandidates:
        c.pop("_rawRet3m", None)
        c.pop("_rawRet12m", None)
        c.pop("_rawVol", None)
        c.pop("_rawDrawdown", None)
        c.pop("marketCapNum", None)

    result = {
        "sector": resolvedName,
        "sectorEtf": ticker,
        "asOfDate": timestamp.strftime("%Y-%m-%d"),
        "styleFilter": style,
        "candidateCount": len(finalCandidates),
        "candidates": finalCandidates,
        "scoutingGuidance": f"Found {len(finalCandidates)} qualified {style.upper()} candidates in {resolvedName}. Select top 2-3 for in-depth analysis."
    }

    cleaned = cleanData(result)
    data.cache.put(cacheKey, cleaned)
    return cleaned



