import os
import re
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

from collectors.constants import UTC, NEW_YORK, END_DATE
from collectors.sector_dl_client import GICS_SECTORS, DB_SECTOR_TO_TICKER
from llmtools.tool_registry import Tool, DataProviders
from llmtools.functions.helpers import cleanData


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


def calculateRsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avgGain = gain.ewm(com=period - 1, min_periods=period).mean().iloc[-1]
    avgLoss = loss.ewm(com=period - 1, min_periods=period).mean().iloc[-1]

    if avgLoss == 0:
        return 100.0
    rs = avgGain / avgLoss
    return float(100 - (100 / (1 + rs)))


def calculateGrowthScore(ret1y: float, ret3m: float, vol1y: float, isTrendAligned: bool, rsi14: float) -> float:
    # 1. 1-Year Sharpe proxy (reward return relative to volatility)
    # 2. Trend alignment bonus (Stage-2: Close > SMA50 > SMA200)
    # 3. Dual-horizon expansion bonus (both 12m and 3m positive)
    # 4. Penalise extremely overbought conditions where RSI > 75

    sharpeProxy = (ret1y / vol1y) if vol1y > 0 else 0.0    
    trendBonus = 1.0 if isTrendAligned else 0.0    
    consistencyBonus = 0.5 if (ret1y > 10.0 and ret3m > 0.0) else 0.0    
    overboughtPenalty = 0.75 if rsi14 > 75.0 else 0.0

    # Final weighted score: 40% of Sharpe proxy, 30% of 1-year return, 20% of 3-month return, plus bonuses minus penalties
    # Higher score is better for growth
    return (sharpeProxy * 0.4) + (ret1y / 100.0 * 0.3) + (ret3m / 100.0 * 0.2) + trendBonus + consistencyBonus - overboughtPenalty


def calculateDefensiveScore(
    downsideVol: float, 
    maxDd1y: float, 
    vol1y: float, 
    drawdown52w: float, 
    isAboveSma200: bool, 
    worstDayLoss: float
) -> float:
    # 1. Reject dormant tickers where volatility is suspiciously low (< 8.0%)
    if vol1y < 8.0:
        return 999.0

    # 2. Core Downside Risk: 50% downside semi-deviation + 30% 1-year max drawdown
    # 3. Penalise stocks currently is deep ongoing drawdowns (> 15% below 52w high) 
    # 4. Penalise stocks prone to severe single-day shocks (worst 1-day drop worse than -10.0%)
    # 5. Reward stocks holding above their 200-day SMA (avoiding falling knives)

    baseScore = (downsideVol * 0.5) + (abs(maxDd1y) * 0.3)
    drawdownPenalty = (abs(drawdown52w) - 15.0) * 0.2 if abs(drawdown52w) > 15.0 else 0.0
    severeDropPenalty = (abs(worstDayLoss) - 10.0) * 0.3 if worstDayLoss < -10.0 else 0.0
    trendBonus = 1.0 if isAboveSma200 else 0.0

    # Lower score is better for defensive, so invert the signs of bonuses and penalties
    return baseScore + drawdownPenalty + severeDropPenalty - trendBonus


def calculateValueScore(
    drawdown52w: float, 
    rsi14: float, 
    marketCapNum: float, 
    vol1y: float, 
    latestClose: float, 
    sma200: float
) -> float:
    # 1. Reject dormant tickers where volatility is suspiciously low (< 8.0%)
    if vol1y < 8.0:
        return -999.0

    # 2. Discount from 52W high (avoid falling knives, but reward value opportunities)
    # 3. Oversold accumulation (RSI between 32 and 48 is prime value)
    # 4. Bonus for larger, institutional-scale balance sheet
    # 5. Bonus for closing price within support distance of the 200-day SMA (strong mean-reversion value)

    absDd = abs(drawdown52w)
    if 8.0 <= absDd <= 28.0:
        discountScore = 2.0
    elif absDd < 8.0:
        discountScore = 1.0  # Mild consolidation
    elif absDd <= 38.0:
        discountScore = 0.5  # Elevated risk of value trap
    else:
        discountScore = -1.5 # Severe crash / falling knife (> 38% down)

    if 32.0 <= rsi14 <= 48.0:
        rsiScore = 1.5
    elif rsi14 < 32.0:
        rsiScore = 1.0  # Extremely oversold
    else:
        rsiScore = max(0.0, 1.0 - ((rsi14 - 50.0) / 25.0))

    scaleBonus = min(1.5, max(0.0, (np.log10(marketCapNum) - 10.0) * 0.75)) if marketCapNum > 0 else 0.0

    smaRatio = (latestClose / sma200) if sma200 > 0 else 1.0
    reversionBonus = 1.0 if 0.85 <= smaRatio <= 1.08 else 0.0

    # Higher score is better for value
    return discountScore + rsiScore + scaleBonus + reversionBonus


def fetchStocksInSector(tool: Tool, data: DataProviders, timestamp: pd.Timestamp, sector: str, 
                       style: str = "all", limit: int = 25) -> Dict[str, Any]:
    limit = max(4, min(int(limit) if limit else 25, 40))

    style = str(style).strip().lower() if style else "all"
    if style not in ["value", "defensive", "all"]:
        style = "all"

    ticker, resolvedName, note = data.sectors.resolveSector(sector)
    if resolvedName == "Unknown" or not ticker or ticker not in GICS_SECTORS:
        return cleanData({"error": note or f"Sector '{sector}' not recognised among the 11 GICS sectors."})

    dbKey = next((k for k, v in DB_SECTOR_TO_TICKER.items() if v == ticker), None)
    if not dbKey:
        return cleanData({"error": f"Could not map sector ETF '{ticker}' to internal database category."})

    tsNy = timestamp.tz_convert(NEW_YORK) if timestamp.tzinfo else timestamp.tz_localize(NEW_YORK)
    effectiveTs = min(tsNy, END_DATE)

    cacheKey = f"stocksearch|sector_{ticker}_{style}_{limit}_{effectiveTs.strftime('%Y-%m-%dH%H')}"
    cached = data.cache.get(cacheKey)
    if cached is not None:
        return cached

    with data.sectors.keyedLocks.lockKey(cacheKey):
        cached = data.cache.get(cacheKey)
        if cached is not None:
            return cached

        # 1. Find all valid candidate tickers
        candidateProfiles = [
            p for p in data.tickers.tickerIndex.values()
            if p.sector == dbKey                    # In correct sector
            and p.industry 
            and p.industry.lower() != "unknown"     # Filter out unknown industries
            and data.tickers.isTickerListed(p.ticker, effectiveTs)    # Ticker is listed as of effectiveTs
            and p.ticker in data.ohlcv.tickersPaths    # Has local OHLCV data available
        ]

        if not candidateProfiles:
            return cleanData({
                "status": "no_candidates",
                "sector": resolvedName,
                "message": f"No listed equities with known industries found for sector '{resolvedName}' as of {effectiveTs.strftime('%Y-%m-%d')}."
            })

        # 2. Filter candidates by market cap threshold (>= 20bn)... fallback to >= 10bn if too few
        minCapThreshold = 20e9

        def filterQualifying(threshold: float) -> List[tuple]:
            qualifying = []
            def checkStock(p):
                row = data.ohlcv.getSingleDayTickerData(p.ticker, effectiveTs)
                if row is None or "marketCap" not in row:
                    return None
                capStr = str(row["marketCap"])
                capNum = parseMarketCapValue(capStr)
                if threshold <= capNum:
                    return (p, capStr, capNum)
                return None

            # Run concurrent checks for market cap to speed up filtering
            totalCandidates = len(candidateProfiles)
            completedCandidates = 0
            with ThreadPoolExecutor(max_workers=min(totalCandidates, 16)) as executor:
                futures = [executor.submit(checkStock, p) for p in candidateProfiles]
                for f in as_completed(futures):
                    completedCandidates += 1
                    if tool is not None and totalCandidates > 0:
                        pct = (completedCandidates / totalCandidates) * 100.0
                        tool.updateProgress(f"Stage 1/2: {pct:.1f}%")
                    res = f.result()
                    if res:
                        qualifying.append(res)
            return qualifying

        qualifyingProfiles = filterQualifying(minCapThreshold)
        if len(qualifyingProfiles) < 8:
            minCapThreshold = 10e9
            qualifyingProfiles = filterQualifying(minCapThreshold)

            if len(qualifyingProfiles) < 8:
                minCapThreshold = 5e9
                qualifyingProfiles = filterQualifying(minCapThreshold)

        if not qualifyingProfiles:
            return cleanData({
                "status": "no_large_cap_candidates",
                "sector": resolvedName,
                "message": f"No equities (>5bn mkt cap) found in sector '{resolvedName}'."
            })

        normTs = effectiveTs.tz_localize(None) if effectiveTs.tzinfo is not None else effectiveTs
        tsUtc = normTs.tz_localize(UTC)
        threeMonthsAgoUtc = tsUtc - pd.DateOffset(months=3)
        oneYearAgoUtc = tsUtc - pd.DateOffset(years=1)
        startDate = oneYearAgoUtc - pd.DateOffset(days=40)

        # 3. Concurrent point-in-time metrics calculation via DataProviders
        def processStock(item):
            profile, massCapStr, massCapNum = item
            t = profile.ticker
            try:
                validDf = data.ohlcv.getPeriodDailyTickerData(t, startDate, normTs, referenceDate=normTs)
                if validDf.empty or len(validDf) < 60:
                    return None

                closes = validDf["close"].values
                latestClose = float(closes[-1])
                if latestClose <= 1.0:
                    return None

                lastRow = validDf.iloc[-1]
                mktCapStr = str(lastRow.get("marketCap", massCapStr))
                mktCapNum = parseMarketCapValue(mktCapStr)  # Dont bother checking, market cap was already validated

                # 3-Month and 12-Month Returns
                df3m = validDf[validDf["date"] <= threeMonthsAgoUtc]
                return3m = round(((latestClose - float(df3m["close"].iloc[-1])) / float(df3m["close"].iloc[-1])) * 100, 1) if not df3m.empty else 0.0

                df12m = validDf[validDf["date"] <= oneYearAgoUtc]
                return12m = round(((latestClose - float(df12m["close"].iloc[-1])) / float(df12m["close"].iloc[-1])) * 100, 1) if not df12m.empty else return3m

                # 52-Week High & Drawdown
                pastYearDf = validDf[validDf["date"] >= oneYearAgoUtc]
                high52 = float(pastYearDf["high"].max() if "high" in pastYearDf.columns else pastYearDf["close"].max()) if not pastYearDf.empty else latestClose
                drawdown52w = round(((latestClose - high52) / high52) * 100, 1) if high52 > 0 else 0.0

                # 1-Year Max Drawdown (Peak to Trough)
                closes1y = closes[-min(252, len(closes)):]
                runningMax = np.maximum.accumulate(closes1y)
                ddSeries = (closes1y - runningMax) / runningMax
                maxDd1y = round(float(np.min(ddSeries) * 100), 1)

                # Volatility (Annualised) & Downside Semi-Deviation
                dailyRets = np.diff(closes1y) / closes1y[:-1]
                vol1y = round(float(np.std(dailyRets) * np.sqrt(252) * 100), 1) if len(dailyRets) > 1 else 25.0

                negRets = dailyRets[dailyRets < 0]
                downsideVol = round(float(np.std(negRets) * np.sqrt(252) * 100), 1) if len(negRets) > 1 else vol1y

                worstDayLoss = round(float(np.min(dailyRets) * 100), 1) if len(dailyRets) > 0 else 0.0

                # Moving Averages & RSI
                sma50 = float(np.mean(closes[-min(50, len(closes)):]))
                sma200 = float(np.mean(closes[-min(200, len(closes)):]))
                rsi14 = round(calculateRsi(validDf["close"], period=14), 1)

                # Stage 2 Trend Alignment: Close > SMA50 > SMA200
                isTrendAligned = bool(latestClose > sma50 > sma200)
                isAboveSma200 = bool(latestClose > sma200)

                # Heuristic Scores
                growthScore = calculateGrowthScore(return12m, return3m, vol1y, isTrendAligned, rsi14)
                defensiveScore = calculateDefensiveScore(downsideVol, maxDd1y, vol1y, drawdown52w, isAboveSma200, worstDayLoss)
                valueScore = calculateValueScore(drawdown52w, rsi14, mktCapNum, vol1y, latestClose, sma200)

                tags = []
                if defensiveScore < 20.0 and isAboveSma200:
                    tags.append("defensive")
                if valueScore > 1.5:
                    tags.append("value")

                return {
                    "ticker": t,
                    "companyName": profile.name,
                    "industry": profile.industry,
                    "marketCap": mktCapStr,
                    "latestPrice": round(latestClose, 2),
                    "return12m": f"{return12m:+0.1f}%",
                    "return3m": f"{return3m:+0.1f}%",
                    "annualisedVol1y": f"{vol1y:0.1f}%",
                    "downsideVol": f"{downsideVol:0.1f}%",
                    "maxDrawdown1y": f"{maxDd1y:+0.1f}%",
                    "distFrom52wHigh": f"{drawdown52w:+0.1f}%",
                    "rsi14": rsi14,
                    "stage2Trend": isTrendAligned,
                    "styleTags": tags,

                    "_marketCapNum": mktCapNum,
                    "_growthScore": growthScore,
                    "_defensiveScore": defensiveScore,
                    "_valueScore": valueScore,
                    "_rawRet3m": return3m,
                    "_rawRet12m": return12m
                }
            except Exception:
                return None

        evaluatedList = []
        totalQualifying = len(qualifyingProfiles)
        completedQualifying = 0
        with ThreadPoolExecutor(max_workers=min(totalQualifying, 16)) as executor:
            futures = [executor.submit(processStock, item) for item in qualifyingProfiles]
            for f in as_completed(futures):
                completedQualifying += 1
                if tool is not None and totalQualifying > 0:
                    pct = (completedQualifying / totalQualifying) * 100.0
                    tool.updateProgress(f"Stage 2/2: {pct:.1f}%")
                res = f.result()
                if res:
                    evaluatedList.append(res)

        if not evaluatedList:
            return cleanData({"error": f"Unable to retrieve validated price metrics for sector '{resolvedName}'."})

        # 4. Group by sub-industry and select top 4 per industry
        industryBuckets: Dict[str, List[Dict[str, Any]]] = {}
        for item in evaluatedList:
            ind = item["industry"] or "general"
            industryBuckets.setdefault(ind, []).append(item)

        selectedPool: List[Dict[str, Any]] = []
        for ind, group in industryBuckets.items():
            if style == "value":
                group.sort(key=lambda x: x["_valueScore"], reverse=True)
            elif style == "defensive":
                group.sort(key=lambda x: x["_defensiveScore"])
            else:
                group.sort(key=lambda x: x["_marketCapNum"], reverse=True)
            # Take up to 4 top stocks per sub-industry
            selectedPool.extend(group[:4])

        # 5. Final Sort across the pooled candidates
        if style == "value":
            selectedPool.sort(key=lambda x: x["_valueScore"], reverse=True)
        elif style == "defensive":
            selectedPool.sort(key=lambda x: x["_defensiveScore"])
        else:
            selectedPool.sort(key=lambda x: x["_marketCapNum"], reverse=True)

        finalCandidates = selectedPool[:limit]

        # Clean temporary private scoring fields
        for c in finalCandidates:
            c.pop("_growthScore", None)
            c.pop("_defensiveScore", None)
            c.pop("_valueScore", None)
            c.pop("_rawRet3m", None)
            c.pop("_rawRet12m", None)
            c.pop("_marketCapNum", None)

        subIndustriesRepresented = sorted(list(set(c["industry"] for c in finalCandidates)))

        result = {
            "sector": resolvedName,
            "asOfDate": effectiveTs.strftime("%Y-%m-%d"),
            "styleFilter": style,
            "candidateCount": len(finalCandidates),
            "subIndustriesCovered": subIndustriesRepresented,
            "candidates": finalCandidates,
        }

        cleaned = cleanData(result)
        data.cache.put(cacheKey, cleaned)
        return cleaned
