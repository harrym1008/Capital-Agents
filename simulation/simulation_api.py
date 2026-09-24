import os
import time
import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import request, session

from concurrent.futures import ThreadPoolExecutor
from ui.ws_api import registerWsAction, sendWsResponse

from collectors.constants import END_DATE_STR, ALL_TICKERS_FILE, NYSE_DIRECTORY, NASDAQ_DIRECTORY, NEW_YORK, START_DATE
from collectors.rate_limiter import GlobalRateLimiters

from dataquery import (LRUCache, DailyPriceProvider, TickerDataProvider, MacroDataProvider, 
                       MacroSeries, NewsDataProvider, EdgarDataProvider)

from llmtools.functions.helpers import cleanNumber, NumberType
from llmtools.functions.edgar import FormType, CompanyRef, calculateHistoricalBeta, calculateDividendYield
from llmtools.functions.company import calculate30DayAverageVolume, calculateSharpeRatio

from simulation.market_sim import MarketSimulation
from simulation.orders import MarketOrder, LimitOrder, StopOrder, StopLimitOrder, OrderSide, OrderStatus


# Initialise isolated simulation data providers and rate limiters
class SimulationDataProviders:
    def __init__(self):
        # In-memory cache allocated to 256 MB
        self.cache = LRUCache(256 * 1024 ** 2, main=False)
        self.rateLimiters = GlobalRateLimiters()

        self.tickers = TickerDataProvider()
        self.macro = MacroDataProvider(self.cache, self.rateLimiters)
        self.ohlcv = DailyPriceProvider(self.tickers, self.cache, self.rateLimiters, allowOnlineDownloads=False)
        self.news = NewsDataProvider(self.cache, self.rateLimiters)
        self.edgar = EdgarDataProvider(self.tickers, self.cache, self.rateLimiters.edgarLimiter)


# Sim manager which keeps track of market simulation instances, backtests, and websocket state
class SimulationManager:
    def __init__(self):
        self.dataProviders = SimulationDataProviders()
        self.ohlcvChartCache = {}
        self.userSimulations = {}
        self.userHistory = {}
        self.inspectorExecutor = ThreadPoolExecutor(max_workers=4)
        self.activeInspectorJobs = {}

        # Build available tickers index from parquet store
        self.availableTickers = []
        self.allTickersCount = 0
        try:
            if os.path.exists(ALL_TICKERS_FILE):
                allTickersDf = pd.read_parquet(ALL_TICKERS_FILE)
                self.allTickersCount = len(allTickersDf)
                for row in allTickersDf.itertuples():
                    ticker = row.ticker
                    exchange = row.exchange
                    if exchange == "XNYS":
                        path = os.path.join(NYSE_DIRECTORY, f"{ticker}.parquet")
                    else:
                        path = os.path.join(NASDAQ_DIRECTORY, f"{ticker}.parquet")
                    if os.path.exists(path):
                        self.availableTickers.append(ticker)
        except Exception as e:
            raise RuntimeError(f"Failed to load available tickers: {str(e)}")


    # Compute rolling centered mean for smoothed series
    def getRollingMean(self, series, window=3):
        s = pd.Series(series)
        return s.rolling(window=window, center=True, min_periods=1).mean().values


    # Generate Catmull-Rom spline points through target coordinates
    def computeBezierThroughPoints(self, pointsX, pointsY, numSamples=300):
        n = len(pointsX)
        if n < 2:
            return np.array(pointsX), np.array(pointsY)
        
        xPts = np.array(pointsX, dtype=np.float64)
        yPts = np.array(pointsY, dtype=np.float64)
        
        def catmullRomPoint(t, p0, p1, p2, p3):
            t2 = t * t
            t3 = t2 * t
            return 0.5 * (
                2 * p1 +
                (-p0 + p2) * t +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
        
        samplesPerSegment = max(numSamples // n, 50)
        xAll = []
        yAll = []
        
        for i in range(n - 1):
            p0 = xPts[max(i - 1, 0)]
            p1 = xPts[i]
            p2 = xPts[min(i + 1, n - 1)]
            p3 = xPts[min(i + 2, n - 1)]
            
            y0 = yPts[max(i - 1, 0)]
            y1 = yPts[i]
            y2 = yPts[min(i + 1, n - 1)]
            y3 = yPts[min(i + 2, n - 1)]
            
            for j in range(samplesPerSegment):
                t = j / samplesPerSegment
                x = catmullRomPoint(t, p0, p1, p2, p3)
                y = catmullRomPoint(t, y0, y1, y2, y3)
                xAll.append(x)
                yAll.append(y)
        
        xAll.append(xPts[-1])
        yAll.append(yPts[-1])
        return np.array(xAll), np.array(yAll)


    # Segment future performance into coloured delta chunks
    def getFuturePerformanceChunks(self, futureDates, futureCloses, splineVals):
        numPoints = len(futureCloses)
        if numPoints == 0:
            return []
            
        if splineVals is None or len(splineVals) != numPoints:
            pts = [{"x": pd.Timestamp(d).strftime("%Y-%m-%d"), "y": round(float(c), 2)} for d, c in zip(futureDates, futureCloses)]
            return [{"color": "#f43f5e", "points": pts}]
            
        diffVals = futureCloses - splineVals
        refinedDates = []
        refinedCloses = []
        refinedDiffs = []
        
        for i in range(numPoints - 1):
            d1 = futureDates[i]
            c1 = futureCloses[i]
            diff1 = diffVals[i]
            
            refinedDates.append(d1)
            refinedCloses.append(c1)
            refinedDiffs.append(diff1)
            
            diff2 = diffVals[i + 1]
            if (diff1 > 0 and diff2 < 0) or (diff1 < 0 and diff2 > 0):
                d2 = futureDates[i + 1]
                c2 = futureCloses[i + 1]
                
                t = diff1 / (diff1 - diff2)
                t1 = d1.timestamp()
                t2 = d2.timestamp()
                tCross = t1 + t * (t2 - t1)
                if getattr(d1, 'tzinfo', None) is not None:
                    dCross = pd.Timestamp(tCross, unit='s', tz=d1.tzinfo).to_pydatetime()
                else:
                    dCross = pd.Timestamp(tCross, unit='s').to_pydatetime()
                    
                cCross = c1 + t * (c2 - c1)
                diffCross = 0.0
                
                refinedDates.append(dCross)
                refinedCloses.append(cCross)
                refinedDiffs.append(diffCross)
                
        refinedDates.append(futureDates[-1])
        refinedCloses.append(futureCloses[-1])
        refinedDiffs.append(diffVals[-1])
        
        refinedDatesArr = np.array(refinedDates)
        refinedClosesArr = np.array(refinedCloses)
        refinedDiffsArr = np.array(refinedDiffs)
        
        chunks = []
        
        def extractChunksForMask(mask, color):
            n = len(mask)
            inChunk = False
            chunkStart = 0
            for idx in range(n):
                if mask[idx]:
                    if not inChunk:
                        inChunk = True
                        chunkStart = idx
                else:
                    if inChunk:
                        cDates = refinedDatesArr[chunkStart:idx]
                        cCloses = refinedClosesArr[chunkStart:idx]
                        if len(cDates) >= 2:
                            pts = [{"x": pd.Timestamp(d).strftime("%Y-%m-%d"), "y": round(float(c), 2)} for d, c in zip(cDates, cCloses)]
                            chunks.append({"color": color, "points": pts})
                        inChunk = False
            if inChunk:
                cDates = refinedDatesArr[chunkStart:n]
                cCloses = refinedClosesArr[chunkStart:n]
                if len(cDates) >= 2:
                    pts = [{"x": pd.Timestamp(d).strftime("%Y-%m-%d"), "y": round(float(c), 2)} for d, c in zip(cDates, cCloses)]
                    chunks.append({"color": color, "points": pts})

        extractChunksForMask(refinedDiffsArr >= 0, '#10b981')
        extractChunksForMask(refinedDiffsArr <= 0, '#f43f5e')
        return chunks


    # Construct OHLCV chart payload and target projection curves
    def generateOhlcvChartData(self, ticker, simDateTs, targets=None, horizon="long"):
        todayTs = pd.Timestamp.now(tz=NEW_YORK).normalize()
        if targets is None:
            targets = []

        simDateTs = pd.Timestamp(simDateTs)
        if simDateTs.tzinfo is None:
            simDateTs = simDateTs.tz_localize(NEW_YORK)
        else:
            simDateTs = simDateTs.tz_convert(NEW_YORK)

        horizonStr = (horizon or "long").lower()
        simDateStr = simDateTs.strftime("%Y-%m-%d")
        cacheKey = f"{ticker}_{simDateStr}_{horizonStr}"

        if not targets and cacheKey in self.ohlcvChartCache:
            return self.ohlcvChartCache[cacheKey]
        
        if horizonStr in ["immediate", "1m"]:
            startDateTs = simDateTs - pd.DateOffset(months=1)
            endDateTs = simDateTs + pd.DateOffset(months=1)
        elif horizonStr in ["short", "3m"]:
            startDateTs = simDateTs - pd.DateOffset(months=3)
            endDateTs = simDateTs + pd.DateOffset(months=3)
        elif horizonStr in ["medium", "1y", "12m"]:
            startDateTs = simDateTs - pd.DateOffset(years=1)
            endDateTs = simDateTs + pd.DateOffset(years=1)
        elif horizonStr in ["3y", "36m"]:
            startDateTs = simDateTs - pd.DateOffset(years=3)
            endDateTs = simDateTs + pd.DateOffset(years=3)
        elif horizonStr in ["all", "max"]:
            startDateTs = START_DATE if isinstance(START_DATE, pd.Timestamp) else pd.Timestamp(START_DATE, tz=NEW_YORK)
            endDateTs = simDateTs + pd.DateOffset(years=1)
        elif horizonStr == "long":
            startDateTs = simDateTs - pd.DateOffset(years=1)
            endDateTs = simDateTs + pd.DateOffset(years=3)
        else:    # elif horizonStr == "distant":
            startDateTs = simDateTs - pd.DateOffset(years=2)
            endDateTs = simDateTs + pd.DateOffset(years=10)
        
        if targets:
            maxTargetMonths = max([t[0] for t in targets])
            targetEndTs = simDateTs + pd.DateOffset(months=int(maxTargetMonths))
            if targetEndTs > endDateTs:
                endDateTs = targetEndTs
        
        profile = self.dataProviders.tickers.getTickerProfile(ticker)
        if profile and profile.ipoDate is not None:
            ipoTs = pd.Timestamp(profile.ipoDate)
            if ipoTs.tzinfo is None:
                ipoTs = ipoTs.tz_localize(NEW_YORK)
            else:
                ipoTs = ipoTs.tz_convert(NEW_YORK)
            if ipoTs > startDateTs:
                startDateTs = ipoTs

        # Retrieve historical split-adjusted OHLCV price series
        dfHistorical = self.dataProviders.ohlcv.getPeriodDailyTickerData(ticker, startDateTs, simDateTs, referenceDate=simDateTs)
        
        dfFuture = pd.DataFrame()
        if targets and simDateTs < todayTs:
            futureEndTs = min(endDateTs, todayTs)
            dfFuture = self.dataProviders.ohlcv.getPeriodDailyTickerData(ticker, simDateTs, futureEndTs, referenceDate=simDateTs)

        lastHistoricalClose = None
        allPrices = []
        historicalPoints = []
        if not dfHistorical.empty and "close" in dfHistorical.columns:
            dfCleanHist = dfHistorical.dropna(subset=["close"])
            if not dfCleanHist.empty:
                dateCol = "dateNy" if "dateNy" in dfCleanHist.columns else "date"
                dateStrs = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dfCleanHist[dateCol]]
                closesRaw = dfCleanHist["close"].to_numpy(dtype=float)
                histClosesSmooth = self.getRollingMean(closesRaw, window=3)
                lastHistoricalClose = histClosesSmooth[-1]
                allPrices.extend(histClosesSmooth.tolist())
                historicalPoints = [{"x": d, "y": round(float(c), 2)} for d, c in zip(dateStrs, histClosesSmooth)]
                
        futureDates = []
        futureClosesSmooth = []
        if not dfFuture.empty and "close" in dfFuture.columns:
            dfCleanFuture = dfFuture.dropna(subset=["close"])
            if not dfCleanFuture.empty:
                dateColFut = "dateNy" if "dateNy" in dfCleanFuture.columns else "date"
                futureDates = [pd.Timestamp(d).to_pydatetime() for d in dfCleanFuture[dateColFut]]
                closesRaw = dfCleanFuture["close"].to_numpy(dtype=float)
                futureClosesSmooth = self.getRollingMean(closesRaw, window=3)
                allPrices.extend(futureClosesSmooth.tolist())

        # Compute target projection points and spline curve through them
        startTime = startDateTs.timestamp()
        simDateDaysFromStart = (simDateTs - startDateTs).total_seconds() / 86400.0
        targetX = [simDateDaysFromStart]
        targetY = [lastHistoricalClose if lastHistoricalClose is not None else (allPrices[0] if allPrices else 100.0)]
        
        simDateStr = simDateTs.strftime("%Y-%m-%d")
        simAnchorPoint = {"x": simDateStr, "y": round(float(targetY[0]), 2)}

        targetPoints = []
        for monthsOffset, price in targets:
            targetDateTs = simDateTs + pd.DateOffset(months=int(monthsOffset))
            daysFromStart = (targetDateTs - startDateTs).total_seconds() / 86400.0
            targetX.append(daysFromStart)
            targetY.append(price)
            allPrices.append(price)
            targetPoints.append({
                "x": targetDateTs.strftime("%Y-%m-%d"),
                "y": round(float(price), 2),
                "months": int(monthsOffset)
            })
            
        minPrice = min(allPrices) if allPrices else 0.0
        maxPrice = max(allPrices) if allPrices else 100.0
        priceRange = maxPrice - minPrice
        if priceRange == 0:
            priceRange = maxPrice * 0.2 if maxPrice > 0 else 10.0
            
        leeway = max(priceRange * 0.08, minPrice * 0.05)
        yMinCandidate = minPrice - leeway
        
        if yMinCandidate <= 0 or minPrice < 1.0:
            yMin = 0.0
        else:
            yMin = max(0.0, yMinCandidate)
            
        yMax = maxPrice + max(priceRange * 0.08, maxPrice * 0.05)

        # Compute Bezier spline curve through target points and segment future performance into coloured chunks
        curvePoints = []
        splineVals = None
        curveX, curveY = None, None
        if len(targets) > 0:
            curveX, curveY = self.computeBezierThroughPoints(targetX, targetY, numSamples=400)
            for x, y in zip(curveX, curveY):
                dt = pd.Timestamp(round(startTime + x * 86400.0), unit='s', tz=NEW_YORK)
                curvePoints.append({"x": dt.strftime("%Y-%m-%d"), "y": round(float(y), 2)})

        futureChunks = []
        if len(futureDates) > 0:
            if curveX is not None and len(curveX) > 0:
                futureDays = np.array([(pd.Timestamp(d) - startDateTs).total_seconds() / 86400.0 for d in futureDates])
                splineVals = np.interp(futureDays, curveX, curveY)
            futureChunks = self.getFuturePerformanceChunks(futureDates, futureClosesSmooth, splineVals)

        res = {
            "ticker": ticker,
            "simDate": simDateStr,
            "startDate": startDateTs.strftime("%Y-%m-%d"),
            "endDate": endDateTs.strftime("%Y-%m-%d"),
            "yMin": round(float(yMin), 2),
            "yMax": round(float(yMax), 2),
            "historical": historicalPoints,
            "simAnchor": simAnchorPoint,
            "targetPoints": targetPoints,
            "targetCurve": curvePoints,
            "futureChunks": futureChunks
        }
        if not targets:
            if len(self.ohlcvChartCache) > 1000:
                self.ohlcvChartCache.clear()
            self.ohlcvChartCache[cacheKey] = res

        return res

    # Calculate simulation horizon end timestamp
    def computeHorizonEndDate(self, simDateTs: pd.Timestamp, timeHorizon: str) -> pd.Timestamp:
        if not timeHorizon:
            return simDateTs
        h = str(timeHorizon).strip().lower()
        parts = h.split()
        if not parts:
            return simDateTs
        try:
            num = int(parts[0])
        except (ValueError, TypeError):
            num = 1
        if "month" in h:
            return simDateTs + pd.DateOffset(months=num)
        elif "year" in h:
            return simDateTs + pd.DateOffset(years=num)
        return simDateTs

    # Compute portfolio backtest performance metrics and series
    def _calculateBacktestSlice(self, simDateTs: pd.Timestamp, maxDateTs: pd.Timestamp, sp500Df: pd.DataFrame, stockDfs: dict, shares: dict, capital: float, cashDollar: float, cashWeightPct: float, originalShares: dict = None) -> dict:
        normMax = maxDateTs.tz_localize(None) if maxDateTs.tzinfo is not None else maxDateTs
        spSlice = sp500Df[sp500Df["normDate"] <= normMax].copy()
        if spSlice.empty or len(spSlice) < 2:
            return None

        sp0 = float(spSlice.iloc[0]["close"])
        tradingDates = sorted(list(spSlice["normDate"].unique()))

        portfolioPoints = []
        originalPoints = []
        sp500Points = []
        dailyValues = []
        origDailyValues = []
        allPrices = []

        lastPrices = {}
        for ticker, df in stockDfs.items():
            dfSlice = df[df.index <= normMax]
            if not dfSlice.empty:
                lastPrices[ticker] = float(dfSlice.iloc[0]["close"])

        spDict = dict(zip(spSlice["normDate"], spSlice["close"]))

        hasOriginal = bool(originalShares and len(originalShares) > 0)

        for d in tradingDates:
            dateStr = d.strftime("%Y-%m-%d")
            equityVal = 0.0
            for ticker, sh in shares.items():
                df = stockDfs.get(ticker)
                if df is not None and d in df.index:
                    lastPrices[ticker] = float(df.loc[d, "close"])
                equityVal += sh * lastPrices.get(ticker, 0.0)

            portVal = equityVal + cashDollar
            dailyValues.append(portVal)
            portfolioPoints.append({"x": dateStr, "y": round(float(portVal), 2)})
            allPrices.append(portVal)

            if hasOriginal:
                origVal = 0.0
                for oTicker, oSh in originalShares.items():
                    df = stockDfs.get(oTicker)
                    if df is not None and d in df.index:
                        lastPrices[oTicker] = float(df.loc[d, "close"])
                    origVal += oSh * lastPrices.get(oTicker, 0.0)
                origDailyValues.append(origVal)
                originalPoints.append({"x": dateStr, "y": round(float(origVal), 2)})
                allPrices.append(origVal)

            spClose = spDict.get(d, sp0)
            spVal = capital * (float(spClose) / sp0) if sp0 > 0 else capital
            sp500Points.append({"x": dateStr, "y": round(float(spVal), 2)})
            allPrices.append(spVal)

        if not dailyValues:
            return None

        portSeries = pd.Series(dailyValues, index=tradingDates)
        portFinal = float(portSeries.iloc[-1])
        spFinal = float(sp500Points[-1]["y"])

        portReturnPct = ((portFinal - capital) / capital) * 100.0
        portReturnDollar = portFinal - capital
        spReturnPct = ((spFinal - capital) / capital) * 100.0
        spReturnDollar = spFinal - capital
        alphaPct = portReturnPct - spReturnPct

        runningMax = portSeries.cummax()
        drawdown = (portSeries / runningMax) - 1.0
        maxDrawdownPct = float(drawdown.min()) * 100.0

        spSeries = pd.Series([p["y"] for p in sp500Points], index=tradingDates)
        spRunningMax = spSeries.cummax()
        spDrawdown = (spSeries / spRunningMax) - 1.0
        spMaxDrawdownPct = float(spDrawdown.min()) * 100.0

        # Compute original portfolio analytics if comparative baseline exists
        origReturnPct = None
        origReturnDollar = None
        origAlphaPct = None
        origSharpe = None
        origMaxDrawdownPct = None
        if hasOriginal and origDailyValues:
            origSeries = pd.Series(origDailyValues, index=tradingDates)
            origFinal = float(origSeries.iloc[-1])
            origReturnPct = ((origFinal - capital) / capital) * 100.0
            origReturnDollar = origFinal - capital
            origAlphaPct = portReturnPct - origReturnPct

            origRunningMax = origSeries.cummax()
            origDrawdown = (origSeries / origRunningMax) - 1.0
            origMaxDrawdownPct = float(origDrawdown.min()) * 100.0

            origDailyRet = origSeries.pct_change().dropna()
            if len(origDailyRet) >= 5 and origDailyRet.std() > 0:
                origSharpe = float((origDailyRet.mean() / origDailyRet.std()) * np.sqrt(252))
            else:
                origSharpe = 0.0

        # Compute Sharpe ratio using 2-year Treasury yield as risk-free rate
        portDfForSharpe = pd.DataFrame({"date": tradingDates, "close": dailyValues})
        treasDf = self.dataProviders.macro.loadSeries(MacroSeries.TREAS_2Y)
        sharpe = 0.0
        if treasDf is not None and not treasDf.empty:
            try:
                sharpe = float(calculateSharpeRatio(portDfForSharpe, treasDf))
            except Exception:
                sharpe = 0.0

        if sharpe == 0.0 or pd.isna(sharpe):
            dailyReturns = portSeries.pct_change().dropna()
            stdDev = dailyReturns.std()
            if stdDev > 0 and not pd.isna(stdDev):
                sharpe = float((dailyReturns.mean() / stdDev) * np.sqrt(252))

        minPrice = min(allPrices) if allPrices else capital
        maxPrice = max(allPrices) if allPrices else capital
        priceRange = maxPrice - minPrice
        if priceRange == 0:
            priceRange = maxPrice * 0.2 if maxPrice > 0 else 1000.0

        holdingReturns = {}
        for ticker, df in stockDfs.items():
            dfSlice = df[df.index <= normMax]
            if not dfSlice.empty:
                p0 = float(dfSlice.iloc[0]["close"])
                pEnd = float(dfSlice.iloc[-1]["close"])
                retPct = ((pEnd - p0) / p0) * 100.0 if p0 > 0 else 0.0
                sh = shares.get(ticker, originalShares.get(ticker, 0.0) if originalShares else 0.0)
                retDollar = (pEnd - p0) * sh
                holdingReturns[ticker] = {
                    "returnPct": round(float(retPct), 2),
                    "returnDollar": round(float(retDollar), 2),
                    "startPrice": round(float(p0), 2),
                    "endPrice": round(float(pEnd), 2)
                }

        if cashDollar > 0 or cashWeightPct > 0:
            holdingReturns["CASH"] = {
                "returnPct": 0.0,
                "returnDollar": 0.0,
                "startPrice": 1.0,
                "endPrice": 1.0
            }

        leeway = max(priceRange * 0.08, minPrice * 0.05)
        yMin = max(0.0, minPrice - leeway)
        yMax = maxPrice + leeway

        metricsDict = {
            "portfolioReturnPct": round(portReturnPct, 2),
            "portfolioReturnDollar": round(portReturnDollar, 2),
            "sp500ReturnPct": round(spReturnPct, 2),
            "sp500ReturnDollar": round(spReturnDollar, 2),
            "alphaPct": round(alphaPct, 2),
            "sharpeRatio": round(sharpe, 2) if not pd.isna(sharpe) else 0.0,
            "maxDrawdownPct": round(abs(maxDrawdownPct), 2),
            "sp500MaxDrawdownPct": round(abs(spMaxDrawdownPct), 2)
        }

        if hasOriginal and origReturnPct is not None:
            metricsDict["originalReturnPct"] = round(origReturnPct, 2)
            metricsDict["originalReturnDollar"] = round(origReturnDollar, 2)
            metricsDict["originalAlphaPct"] = round(origAlphaPct, 2)
            metricsDict["originalSharpeRatio"] = round(origSharpe, 2) if origSharpe is not None and not pd.isna(origSharpe) else 0.0
            metricsDict["originalMaxDrawdownPct"] = round(abs(origMaxDrawdownPct), 2)

        return {
            "startDate": tradingDates[0].strftime("%Y-%m-%d"),
            "endDate": tradingDates[-1].strftime("%Y-%m-%d"),
            "portfolioPoints": portfolioPoints,
            "originalPoints": originalPoints if hasOriginal else None,
            "sp500Points": sp500Points,
            "holdingReturns": holdingReturns,
            "yMin": round(float(yMin), 2),
            "yMax": round(float(yMax), 2),
            "metrics": metricsDict
        }

    # Generate portfolio backtest analytics against S&P 500 benchmark
    def generatePortfolioBacktestData(self, simDate, initialCapital: float = 100_000.0, positions: list = None, cashPosition: dict = None, timeHorizon: str = None, originalPositions: list = None) -> dict:
        todayTs = pd.Timestamp.now(tz=NEW_YORK).normalize()
        simDateTs = pd.Timestamp(simDate)
        if simDateTs.tzinfo is None:
            simDateTs = simDateTs.tz_localize(NEW_YORK)
        else:
            simDateTs = simDateTs.tz_convert(NEW_YORK)
        simDateTs = simDateTs.normalize()

        if positions is None:
            positions = []
        if cashPosition is None:
            cashPosition = {}

        capital = float(initialCapital) if initialCapital and float(initialCapital) > 0 else 100_000.0

        cashWeightPct = float(cashPosition.get("weightPct", 0.0) or 0.0)
        cashDollar = float(cashPosition.get("dollarAllocation", 0.0) or 0.0)
        if cashDollar <= 0.0 and cashWeightPct > 0.0:
            cashDollar = capital * (cashWeightPct / 100.0)

        # Fetch S&P 500 benchmark series up to current date
        sp500Df = self.dataProviders.macro.getSeries(MacroSeries.SP500, simDateTs, todayTs)
        if sp500Df.empty or "close" not in sp500Df.columns:
            sp500Df = self.dataProviders.macro.loadSeries(MacroSeries.SP500)
            if not sp500Df.empty and "date" in sp500Df.columns:
                spDateCol = pd.to_datetime(sp500Df["date"]).dt.tz_localize(None).dt.normalize()
                simNorm = simDateTs.tz_localize(None)
                sp500Df = sp500Df[spDateCol >= simNorm].reset_index(drop=True)

        if sp500Df.empty or "close" not in sp500Df.columns:
            return {
                "ok": False,
                "error": "Failed to retrieve S&P 500 benchmark series."
            }

        sp500Df = sp500Df.sort_values("date").reset_index(drop=True)
        sp500Df["normDate"] = pd.to_datetime(sp500Df["date"]).dt.tz_localize(None).dt.normalize()
        sp500Df = sp500Df.drop_duplicates(subset=["normDate"]).sort_values("normDate").reset_index(drop=True)

        # Fetch stock data for each holding concurrently
        stockDfs = {}
        shares = {}
        originalShares = {}
        totalStockAlloc = 0.0

        allPositionsToFetch = list(positions)
        seenTickers = {str(p.get("ticker", "")).strip().upper() for p in positions if p.get("ticker")}
        if originalPositions:
            for op in originalPositions:
                otick = str(op.get("ticker", "")).strip().upper()
                if otick and otick not in seenTickers:
                    allPositionsToFetch.append(op)
                    seenTickers.add(otick)

        def fetchStockHolding(pos):
            ticker = str(pos.get("ticker", "")).strip().upper()
            if not ticker:
                return None

            df = self.dataProviders.ohlcv.getPeriodDailyTickerData(ticker, simDateTs, todayTs, referenceDate=simDateTs)
            if df.empty or "close" not in df.columns:
                return None

            df = df.copy()
            dateCol = "dateNy" if "dateNy" in df.columns else "date"
            df["normDate"] = pd.to_datetime(df[dateCol]).dt.tz_localize(None).dt.normalize()
            df = df.dropna(subset=["close"]).drop_duplicates(subset=["normDate"]).sort_values("normDate").set_index("normDate")
            if df.empty:
                return None

            return (ticker, df)

        numWorkers = min(16, max(len(allPositionsToFetch), 1))
        with ThreadPoolExecutor(max_workers=numWorkers) as executor:
            results = list(executor.map(fetchStockHolding, allPositionsToFetch))

        for res in results:
            if res is not None:
                ticker, df = res
                stockDfs[ticker] = df

        for pos in positions:
            ticker = str(pos.get("ticker", "")).strip().upper()
            if not ticker or ticker not in stockDfs:
                continue
            df = stockDfs[ticker]
            dollarAlloc = float(pos.get("dollarAllocation", 0.0) or 0.0)
            weightPct = float(pos.get("weightPct", 0.0) or 0.0)
            if dollarAlloc <= 0.0 and weightPct > 0.0:
                dollarAlloc = capital * (weightPct / 100.0)
            totalStockAlloc += dollarAlloc
            p0 = float(df.iloc[0]["close"])
            shares[ticker] = dollarAlloc / p0 if p0 > 0 else 0.0

        if originalPositions:
            for op in originalPositions:
                otick = str(op.get("ticker", "")).strip().upper()
                if not otick or otick not in stockDfs:
                    continue
                df = stockDfs[otick]
                oDollar = float(op.get("dollarAmount", 0.0) or op.get("dollarAllocation", 0.0) or 0.0)
                oWeight = float(op.get("weightPct", 0.0) or 0.0)
                if oDollar <= 0.0 and oWeight > 0.0:
                    oDollar = capital * (oWeight / 100.0)
                p0 = float(df.iloc[0]["close"])
                originalShares[otick] = oDollar / p0 if p0 > 0 else 0.0

        # Allocate remaining unallocated capital to cash
        if cashDollar <= 0.0 and capital > totalStockAlloc:
            cashDollar = capital - totalStockAlloc

        # Calculate full backtest slice
        fullSlice = self._calculateBacktestSlice(simDateTs, todayTs, sp500Df, stockDfs, shares, capital, cashDollar, cashWeightPct, originalShares=originalShares)
        if not fullSlice:
            return {
                "ok": False,
                "error": "No trading dates found for the specified period."
            }

        # Calculate target horizon slice if applicable
        canToggle = False
        horizonSlice = None
        if timeHorizon:
            horizonEndTs = self.computeHorizonEndDate(simDateTs, timeHorizon)
            normHorizon = horizonEndTs.tz_localize(None) if horizonEndTs.tzinfo is not None else horizonEndTs
            normToday = todayTs.tz_localize(None) if todayTs.tzinfo is not None else todayTs
            if normHorizon < normToday:
                horizonSlice = self._calculateBacktestSlice(simDateTs, horizonEndTs, sp500Df, stockDfs, shares, capital, cashDollar, cashWeightPct, originalShares=originalShares)
                if horizonSlice and horizonSlice.get("endDate") != fullSlice.get("endDate"):
                    canToggle = True

        primarySlice = horizonSlice if (canToggle and horizonSlice) else fullSlice

        return {
            "ok": True,
            "simDate": simDateTs.strftime("%Y-%m-%d"),
            "canToggle": canToggle,
            "timeHorizon": timeHorizon,
            "horizon": horizonSlice,
            "full": fullSlice,
            **primarySlice
        }

    # Sanitise NaN and infinite values in nested data structures
    def cleanNans(self, obj):
        if isinstance(obj, dict):
            return {k: self.cleanNans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.cleanNans(x) for x in obj]
        elif isinstance(obj, (float, int)) or hasattr(obj, "dtype"):
            try:
                val = float(obj)
                if math.isnan(val) or math.isinf(val):
                    return None
                return val
            except (ValueError, TypeError):
                return str(obj)
        return obj

    # Resolve active session identifier from request headers or session store
    def getSessionId(self):
        reqSessionId = request.headers.get("X-Session-ID") or request.args.get("sessionId")
        if reqSessionId:
            return reqSessionId
        if "sim_session_id" not in session:
            session["sim_session_id"] = f"session_{time.perf_counter_ns()}"
        return session["sim_session_id"]

    # Instantiate and configure new market simulation session
    def createSimulation(self, sessionId, portfolioName, startDate, initialCash):
        activeSim = MarketSimulation(startDate, END_DATE_STR, 
                                     tickerDataProvider=self.dataProviders.tickers, 
                                     dailyPriceProvider=self.dataProviders.ohlcv)
        activeSim.initialiseUser(portfolioName, initialCash=initialCash)

        self.userSimulations[sessionId] = {
            "sim": activeSim,
            "portfolioName": portfolioName,
            "inspectedTicker": "NVDA",      # Inspect Nvidia stock by default at 3mo timeframe
            "inspectedTimeframe": "3M"
        }
        
        # Pre-populate preceding days at starting balance for baseline rendering
        startTs = pd.Timestamp(startDate)
        dayMinus2 = (startTs - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        dayMinus1 = (startTs - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        self.userHistory[sessionId] = [
            {"date": dayMinus2, "totalValue": float(initialCash)},
            {"date": dayMinus1, "totalValue": float(initialCash)}
        ]
        self.updateSimulationHistory(sessionId)
        return self.getSimulationState(sessionId)

    # Record portfolio valuation snapshot for current simulation date
    def updateSimulationHistory(self, sessionId):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            return
        activeSim = simData["sim"]
        portfolioName = simData["portfolioName"]

        if sessionId not in self.userHistory:
            self.userHistory[sessionId] = []

        currentDateStr = activeSim.currentDate.strftime("%Y-%m-%d")
        pfDict = activeSim.getPortfolioValueAtCurrentDate(portfolioName)
        if pfDict:
            totalValue = pfDict["cash"]
            for ticker, pos in pfDict["positions"].items():
                curPrice = pos.get("currentPrice")
                if curPrice is None or math.isnan(curPrice) or math.isinf(curPrice) or curPrice <= 0:
                    curPrice = pos.get("averagePrice", 0.0)
                if curPrice is not None and not math.isnan(curPrice) and not math.isinf(curPrice):
                    totalValue += pos["quantity"] * curPrice

            snapshot = {
                "date": currentDateStr,
                "totalValue": float(totalValue)
            }

            if self.userHistory[sessionId] and self.userHistory[sessionId][-1]["date"] == currentDateStr:
                self.userHistory[sessionId][-1] = snapshot
            else:
                self.userHistory[sessionId].append(snapshot)

    # Compile current simulation portfolio state, orders, and valuation
    def getSimulationState(self, sessionId, skipInspector=False):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            return {"active": False}

        activeSim = simData["sim"]
        portfolioName = simData["portfolioName"]

        d = activeSim.currentDate
        currentDateStr = d.strftime("%Y-%m-%d")
        currentDateFriendly = f"{d.strftime('%a')} {d.day} {d.strftime('%b %Y')}"

        pfDict = activeSim.getPortfolioValueAtCurrentDate(portfolioName)
        if pfDict is None:
            return {"active": False}

        totalValue = pfDict["cash"]
        stockValue = 0.0
        for ticker, pos in pfDict["positions"].items():
            curPrice = pos["currentPrice"]
            if curPrice is not None and not math.isnan(curPrice) and not math.isinf(curPrice):
                posVal = pos["quantity"] * curPrice
                totalValue += posVal
                stockValue += posVal

        pfDict["totalValue"] = totalValue
        pfDict["stockValue"] = stockValue
        pfDict["absReturn"] = totalValue - pfDict["startingCash"]
        if pfDict["startingCash"] != 0:
            pfDict["pctReturn"] = (totalValue / pfDict["startingCash"] - 1) * 100
        else:
            pfDict["pctReturn"] = 0.0

        portfolioObj = activeSim.userPortfolios.get(portfolioName)
        if portfolioObj and portfolioObj.mainLog:
            pfDict["log"] = [entry.toDict() if hasattr(entry, 'toDict') else entry for entry in portfolioObj.mainLog]
        else:
            pfDict["log"] = []

        # Compile list of pending orders with relevant details
        pendingOrdersList = []
        for idx, userOrder in enumerate(activeSim.pendingOrders):
            o = userOrder.order
            orderTypeStr = "Market"
            if isinstance(o, LimitOrder):
                orderTypeStr = "Limit"
            elif isinstance(o, StopOrder):
                orderTypeStr = "Stop"
            elif isinstance(o, StopLimitOrder):
                orderTypeStr = "Stop Limit"

            pendingOrdersList.append({
                "index": idx,
                "ticker": o.ticker,
                "side": o.side.value,
                "type": orderTypeStr,
                "isQuantityBased": o.isQuantityBased,
                "quantity": o.quantity,
                "cashValue": o.cashValue,
                "limitPrice": getattr(o, "limitPrice", None),
                "stopPrice": getattr(o, "stopPrice", None),
                "status": o.status.value
            })

        isEnded = activeSim.currentDate >= activeSim.endDate
        historyList = self.userHistory.get(sessionId, [])

        # Include top-half ticker snapshot for frontend live updates
        inspectedTicker = simData.get("inspectedTicker")
        inspectedTimeframe = simData.get("inspectedTimeframe")
        inspectedSnapshot = self.getTickerSnapshot(sessionId, inspectedTicker, inspectedTimeframe)

        rawState = {
            "active": True,
            "portfolioName": portfolioName,
            "currentDate": currentDateStr,
            "currentDateFriendly": currentDateFriendly,
            "isEnded": isEnded,
            "portfolio": pfDict,
            "pendingOrders": pendingOrdersList,
        }
        if inspectedSnapshot is not None:
            rawState["inspectedSnapshot"] = inspectedSnapshot
        cleanedState = self.cleanNans(rawState)
        cleanedState["history"] = historyList
        return cleanedState

    # Validate and queue user order into active simulation
    def submitOrder(self, sessionId, ticker, side, orderType, amountType, amountValue, limitPrice=None, stopPrice=None):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            raise ValueError("No active simulation session found. Please start a simulation.")

        activeSim = simData["sim"]
        portfolioName = simData["portfolioName"]

        if ticker not in activeSim.dailyPriceProvider.tickersPaths or activeSim.dailyPriceProvider.tickersPaths[ticker] is None:
            raise ValueError(f"Ticker '{ticker}' is not available in the price database.")

        portfolioObj = activeSim.userPortfolios.get(portfolioName)
        if not portfolioObj:
            raise ValueError("Portfolio not found.")

        normalizedAmountType = str(amountType).lower().strip()
        if normalizedAmountType in ["shares", "quantity", "qty"]:
            amountType = "quantity"
        elif normalizedAmountType in ["usd", "cash", "dollars"]:
            amountType = "cash"

        if side == "SELL":
            if ticker not in portfolioObj.positions:
                raise ValueError(f"Cannot sell '{ticker}' because you do not hold any active positions in it.")
            heldQty = portfolioObj.positions[ticker].quantity
            if amountType == "quantity":
                if amountValue > heldQty:
                    if amountValue - heldQty <= 0.0001:
                        amountValue = heldQty
                    else:
                        raise ValueError(f"Insufficient shares to sell. You hold {heldQty:.4f} shares of '{ticker}'.")
                elif heldQty - amountValue <= 0.0001:
                    amountValue = heldQty
        elif side == "BUY":
            if amountType == "cash":
                if amountValue > portfolioObj.cash:
                    raise ValueError(f"Insufficient cash balance. Available: ${portfolioObj.cash:,.2f}.")

        sideEnum = OrderSide.BUY if side == "BUY" else OrderSide.SELL

        if orderType == "limit":
            limitVal = float(limitPrice) if limitPrice else 0.0
            order = LimitOrder(ticker, sideEnum, limitPrice=limitVal, quantity=amountValue) if amountType == "quantity" else LimitOrder(ticker, sideEnum, limitPrice=limitVal, cashValue=amountValue)
        elif orderType == "stop":
            stopVal = float(stopPrice) if stopPrice else 0.0
            order = StopOrder(ticker, sideEnum, stopPrice=stopVal, quantity=amountValue) if amountType == "quantity" else StopOrder(ticker, sideEnum, stopPrice=stopVal, cashValue=amountValue)
        elif orderType == "stop_limit":
            stopVal = float(stopPrice) if stopPrice else 0.0
            limitVal = float(limitPrice) if limitPrice else 0.0
            order = StopLimitOrder(ticker, sideEnum, stopPrice=stopVal, limitPrice=limitVal, quantity=amountValue) if amountType == "quantity" else StopLimitOrder(ticker, sideEnum, stopPrice=stopVal, limitPrice=limitVal, cashValue=amountValue)
        else:
            order = MarketOrder(ticker, sideEnum, quantity=amountValue) if amountType == "quantity" else MarketOrder(ticker, sideEnum, cashValue=amountValue)

        activeSim.addOrder(order, portfolioName)
        return self.getSimulationState(sessionId)

    # Cancel pending simulation order by index
    def cancelOrder(self, sessionId, orderIndex):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            raise ValueError("No active simulation session.")

        activeSim = simData["sim"]
        idx = int(orderIndex)
        if 0 <= idx < len(activeSim.pendingOrders):
            activeSim.pendingOrders.pop(idx)
            return self.getSimulationState(sessionId)
        else:
            raise ValueError("Order index out of range.")

    # Filter available tickers listed at simulation timestamp
    def getCurrentlyListedTickers(self, sessionId=None):
        simData = self.userSimulations.get(sessionId) if sessionId else None
        if simData and simData["sim"]:
            simDate = simData["sim"].currentDate
        else:
            simDate = pd.Timestamp.now(tz=NEW_YORK).normalize()

        listedTickers = [
            t for t in self.availableTickers
            if self.dataProviders.tickers.isTickerListed(t, simDate)
        ]
        return listedTickers if len(listedTickers) > 0 else self.availableTickers

    # Select random ticker listed at simulation date
    def getRandomListedTicker(self, sessionId=None):
        listed = self.getCurrentlyListedTickers(sessionId)
        return random.choice(listed)

    # Advance simulation clock by specified days or target date
    def advanceSimulation(self, sessionId, days=None, targetDate=None, targetDateCutoff=None):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            raise ValueError("No active simulation session.")

        activeSim = simData["sim"]
        portfolioName = simData["portfolioName"]

        # Skip inspector metrics during rapid frontend stepping
        isRapidStep = targetDateCutoff is not None

        if targetDateCutoff:
            cutoffTs = pd.Timestamp(targetDateCutoff).tz_localize(NEW_YORK)
            if activeSim.currentDate >= cutoffTs:
                state = self.getSimulationState(sessionId, skipInspector=isRapidStep)
                return 0, state

            nextTradingTs = activeSim.currentDate + timedelta(days=1)
            while not activeSim.isTradingDay(nextTradingTs) and nextTradingTs <= activeSim.endDate:
                nextTradingTs += timedelta(days=1)

            if nextTradingTs > cutoffTs:
                state = self.getSimulationState(sessionId, skipInspector=isRapidStep)
                return 0, state

        untilEnd = False
        parsedTargetDate = None
        daysToAdvance = 0

        if targetDate:
            parsedTargetDate = pd.Timestamp(targetDate).tz_localize(NEW_YORK)
            if activeSim.currentDate >= parsedTargetDate:
                raise ValueError(f"Target date '{targetDate}' must be after current simulated date '{activeSim.currentDate.strftime('%Y-%m-%d')}'.")
            daysToAdvance = (parsedTargetDate - activeSim.currentDate).days
        elif days == "untilEnd":
            untilEnd = True
            daysToAdvance = 5000
        else:
            daysToAdvance = int(days)
            if daysToAdvance <= 0:
                raise ValueError("Days to advance must be positive.")

        # Continue advancing simulation until target date or specified days are reached
        i = 0
        executedDays = 0
        while True:
            if parsedTargetDate is not None:
                if activeSim.currentDate >= parsedTargetDate:
                    break
            elif not untilEnd and i >= daysToAdvance:
                break

            try:
                success = activeSim.runNextDay()
                if not success:
                    break

                i += 1
                executedDays += 1
                self.updateSimulationHistory(sessionId)

            except ValueError as e:
                currentDate = activeSim.currentDate
                portfolioObj = activeSim.userPortfolios.get(portfolioName)
                if activeSim.pendingOrders:
                    failedUserOrder = activeSim.pendingOrders.pop(0)
                    if portfolioObj:
                        portfolioObj.addToLog(currentDate, failedUserOrder.ticker, "Order Failed", f"{str(e)}")
                    failedUserOrder.order.setOrderStatus(OrderStatus.FAILED)
                    activeSim.ordersArchive.append(failedUserOrder)
                break

            except Exception:
                break

        state = self.getSimulationState(sessionId, skipInspector=isRapidStep)
        return executedDays, state

    # Clear simulation session state and valuation history
    def resetSimulation(self, sessionId):
        if sessionId in self.userSimulations:
            del self.userSimulations[sessionId]
        if sessionId in self.userHistory:
            del self.userHistory[sessionId]
        return True

    # Fetch full ticker information, chart data, and company profile
    def getTickerInfo(self, sessionId, ticker, timeframe=None, targetDate=None):
        ticker = ticker.strip().upper()
        simData = self.userSimulations.get(sessionId)

        if simData:
            simData["inspectedTicker"] = ticker
            if timeframe:
                simData["inspectedTimeframe"] = str(timeframe).strip().upper()

        inspectedTimeframe = simData.get("inspectedTimeframe", "3M") if simData else (timeframe or "3M").upper()
        timeframeStr = inspectedTimeframe.lower()

        # Generate chart aligned to target advance date
        if targetDate is not None:
            simDateTs = pd.Timestamp(targetDate).tz_localize(NEW_YORK)
            activeSim = simData["sim"] if (simData and simData["sim"]) else None
        elif simData and simData["sim"]:
            simDateTs = simData["sim"].currentDate
            activeSim = simData["sim"]
        else:
            simDateTs = pd.Timestamp.now(tz=NEW_YORK).normalize()
            activeSim = None

        tickerExists = False
        lastPrice = None
        if activeSim and hasattr(activeSim, "dailyPriceProvider"):
            tickerExists = (ticker in activeSim.dailyPriceProvider.tickersPaths and activeSim.dailyPriceProvider.tickersPaths[ticker] is not None)
            if tickerExists:
                row = activeSim.dailyPriceProvider.getSingleDayTickerData(ticker, simDateTs)
                if row is not None and "close" in row and pd.notna(row["close"]):
                    lastPrice = float(row["close"])

        chartData = self.generateOhlcvChartData(ticker, simDateTs, targets=[], horizon=timeframeStr)
        if lastPrice is None and chartData and chartData.get("simAnchor"):
            lastPrice = chartData["simAnchor"].get("y")

        profile = self.dataProviders.tickers.getTickerProfile(ticker)
        profileData = {}
        if profile:
            profileData = {
                "ticker": profile.ticker,
                "name": profile.name,
                "exchange": profile.exchange,
                "sector": profile.sector,
                "industry": profile.industry,
                "website": profile.website,
                "summary": profile.summary
            }
            tickerExists = True

        return {
            "ok": True,
            "exists": tickerExists,
            "ticker": ticker,
            "lastPrice": lastPrice,
            "chart": chartData,
            "profile": profileData
        }

    # Extract lightweight ticker snapshot and price performance
    def getTickerSnapshot(self, sessionId, ticker, timeframe="3M"):
        ticker = ticker.strip().upper()
        simData = self.userSimulations.get(sessionId)

        if simData and simData["sim"]:
            simDateTs = simData["sim"].currentDate
        else:
            simDateTs = pd.Timestamp.now(tz=NEW_YORK).normalize()

        profile = self.dataProviders.tickers.getTickerProfile(ticker)
        companyName = profile.name if profile and profile.name else ticker

        logoUrl = None
        try:
            logoUrl = self.dataProviders.tickers.getCompanyLogoFromFinnhub(ticker)
        except Exception:
            pass

        timeframeStr = (timeframe or "3M").lower()
        chartData = self.generateOhlcvChartData(ticker, simDateTs, targets=[], horizon=timeframeStr)

        currentPrice = None
        historical = []
        simAnchor = None
        if chartData:
            historical = chartData.get("historical", [])
            simAnchor = chartData.get("simAnchor")
            if simAnchor:
                currentPrice = simAnchor.get("y")

        # Extract raw closing price matching positions table
        if simData and simData["sim"]:
            activeSim = simData["sim"]
            if hasattr(activeSim, "dailyPriceProvider"):
                row = activeSim.dailyPriceProvider.getSingleDayTickerData(ticker, simDateTs)
                if row is not None and "close" in row and pd.notna(row["close"]):
                    currentPrice = float(row["close"])

        priceChange = None
        if len(historical) >= 2 and currentPrice is not None:
            startPrice = historical[0].get("y")
            if startPrice:
                priceChange = ((currentPrice / startPrice) - 1) * 100

        return {
            "companyName": companyName,
            "logoUrl": logoUrl,
            "currentPrice": currentPrice,
            "priceChange": priceChange,
            "chart": {
                "historical": historical,
                "simAnchor": simAnchor
            }
        }

    # Retrieve single-day closing price for order entry
    def getTickerLastPrice(self, sessionId, ticker):
        ticker = ticker.strip().upper()
        simData = self.userSimulations.get(sessionId)

        if simData and simData["sim"]:
            activeSim = simData["sim"]
            simDateTs = activeSim.currentDate
        else:
            activeSim = None
            simDateTs = pd.Timestamp.now(tz=NEW_YORK).normalize()

        tickerExists = False
        lastPrice = None
        if activeSim and hasattr(activeSim, "dailyPriceProvider"):
            tickerExists = (ticker in activeSim.dailyPriceProvider.tickersPaths and activeSim.dailyPriceProvider.tickersPaths[ticker] is not None)
            if tickerExists:
                row = activeSim.dailyPriceProvider.getSingleDayTickerData(ticker, simDateTs)
                if row is not None and "close" in row and pd.notna(row["close"]):
                    lastPrice = float(row["close"])

        # Fall back to chart anchor if close price is absent
        if lastPrice is None or lastPrice == 0:
            try:
                chartData = self.generateOhlcvChartData(ticker, simDateTs, targets=[], horizon="3m")
                if chartData and chartData.get("simAnchor"):
                    anchorPrice = chartData["simAnchor"].get("y")
                    if anchorPrice:
                        lastPrice = float(anchorPrice)
            except Exception:
                pass

        return {"ok": True, "exists": tickerExists, "ticker": ticker, "lastPrice": lastPrice}

    # Dispatch asynchronous ticker inspection job to thread pool
    def startAsyncInspectorJob(self, sessionId, ticker, targetDateStr, timeframeStr="3M", jobId=None, websocket=None, eventLoop=None):
        # Synchronise inspector job identifier with frontend client
        if jobId is None:
            jobId = self.activeInspectorJobs.get(sessionId, 0) + 1
        self.activeInspectorJobs[sessionId] = jobId

        def worker():
            if self.activeInspectorJobs.get(sessionId) != jobId:
                return

            tickerClean = ticker.strip().upper()
            try:
                info = self.getTickerInfo(sessionId, tickerClean, timeframe=timeframeStr, targetDate=targetDateStr)

                if self.activeInspectorJobs.get(sessionId) != jobId:
                    return

                fastData = self.getFastInspectorData(tickerClean, targetDateStr, timeframeStr)

                if self.activeInspectorJobs.get(sessionId) != jobId:
                    return

                result = {
                    "action": "sim_ticker_info_push",
                    "sessionId": sessionId,
                    "jobId": jobId,
                    "targetDate": targetDateStr,
                    "ticker": tickerClean,
                    "ok": True,
                    "exists": info["exists"],
                    "lastPrice": info["lastPrice"],
                    "chart": info["chart"],
                    "profile": info["profile"],
                    "inspector": fastData
                }
            except Exception as e:
                print(f"[INSPECTOR THREAD ERROR] job {jobId}: {e}")
                # Dispatch error payload to release frontend loading state
                result = {
                    "action": "sim_ticker_info_push",
                    "sessionId": sessionId,
                    "jobId": jobId,
                    "targetDate": targetDateStr,
                    "ticker": tickerClean,
                    "ok": False,
                    "error": str(e)
                }

            if websocket:
                sendWsResponse(websocket, result, eventLoop)
            return result

        self.inspectorExecutor.submit(worker)
        return jobId

    # Return listing counts and available tickers
    def getAvailableTickers(self):
        return {
            "tickers": self.availableTickers,
            "totalCount": self.allTickersCount,
            "availableCount": len(self.availableTickers)
        }

    # Format sector or industry string into title case
    def formatSectorOrIndustry(self, rawStr):
        if not rawStr or pd.isna(rawStr):
            return "N/A"
        cleanStr = str(rawStr).replace("_", " ").strip()
        return cleanStr.title().replace("And", "and") if cleanStr else "N/A"

    # Fetch technical indicators, valuation metrics, filings, and news for the inspector
    def fetchFastInspectorMetrics(self, ticker, targetTs, timeframeStr="3M"):
        profile = self.dataProviders.tickers.getTickerProfile(ticker)
        rawExchange = profile.exchange if profile and profile.exchange else "NASDAQ"
        rawSector = profile.sector if profile and profile.sector else "N/A"
        rawIndustry = profile.industry if profile and profile.industry else "N/A"
        companyName = profile.name if profile and profile.name else ticker

        startDate = targetTs - pd.Timedelta(days=365)
        df = self.dataProviders.ohlcv.getPeriodDailyTickerData(ticker, startDate, targetTs)

        lastPrice = None
        openPrice = None
        highPrice = None
        lowPrice = None
        volume = None
        prevClose = None
        fiftyTwoHigh = None
        fiftyTwoLow = None

        if df is not None and not df.empty:
            try:
                lastRow = df.iloc[-1]
                lastPrice = float(lastRow["close"])
                openPrice = float(lastRow["open"])
                highPrice = float(lastRow["high"])
                lowPrice = float(lastRow["low"])
                volume = float(lastRow["volume"])

                if len(df) >= 2:
                    prevRow = df.iloc[-2]
                    prevClose = float(prevRow["close"])

                highSeries = df["high"].dropna()
                if not highSeries.empty:
                    fiftyTwoHigh = float(highSeries.max())
                    
                lowSeries = df["low"].dropna()
                if not lowSeries.empty:
                    fiftyTwoLow = float(lowSeries.min())

            except Exception:
                pass

        valRes = {}

        # Extract shares outstanding and market capitalisation
        if df is not None and not df.empty:
            lastShares = df["outstandingShares"].dropna()
            lastMarketCap = df["marketCap"].dropna()
            if not lastShares.empty:
                valRes["sharesOutstanding"] = float(lastShares.iloc[-1])
                valRes["marketCap"] = "$" + str(lastMarketCap.iloc[-1])

        companyRef = CompanyRef(ticker)
        valRes["beta"] = calculateHistoricalBeta(companyRef, targetTs, self.dataProviders.ohlcv, self.dataProviders.macro)
        valRes["dividendYield"] = calculateDividendYield(companyRef, targetTs, self.dataProviders.ohlcv)

        # Retrieve latest 10-K and 10-Q filing references
        tenKUrl = None
        tenKDate = None
        tenQUrl = None
        tenQDate = None
        try:
            tenKFiling = self.dataProviders.edgar.getLatestFilingRef(companyRef, FormType.FORM_10K, before=targetTs)
            if tenKFiling:
                tenKUrl = getattr(tenKFiling, "filing_url", None) or getattr(tenKFiling, "url", None)
                rawFDate = getattr(tenKFiling, "filing_date", None)
                if rawFDate:
                    try:
                        tenKDate = pd.to_datetime(rawFDate).strftime("%d %b %Y")
                    except Exception:
                        pass

            tenQFiling = self.dataProviders.edgar.getLatestFilingRef(companyRef, FormType.FORM_10Q, before=targetTs)
            if tenQFiling:
                tenQUrl = getattr(tenQFiling, "filing_url", None) or getattr(tenQFiling, "url", None)
                rawQDate = getattr(tenQFiling, "filing_date", None)
                if rawQDate:
                    try:
                        tenQDate = pd.to_datetime(rawQDate).strftime("%d %b %Y")
                    except Exception:
                        pass
        except Exception:
            pass

        # Fetch recent news headlines for ticker
        newsList = []
        try:
            rawNews = self.dataProviders.news.getRecentNewsForTicker(ticker, before=targetTs, limit=50)
            if rawNews is not None and isinstance(rawNews, pd.DataFrame) and not rawNews.empty:
                for _, row in rawNews.iterrows():
                    headline = str(row.get("headline", ""))
                    author = str(row.get("author", "Benzinga"))
                    rawDate = row.get("date")
                    articleDate = ""
                    try:
                        articleDate = pd.to_datetime(rawDate).strftime("%d %b %Y")
                    except Exception:
                        pass

                    articleUrl = str(row.get("url", "#")) if "url" in row and pd.notna(row.get("url")) else "#"
                    if articleUrl == "#" or not articleUrl or articleUrl == "nan":
                        if "id" in row and pd.notna(row.get("id")):
                            articleUrl = f"https://www.benzinga.com/news/01/16/{row.get('id')}"
                        else:
                            articleUrl = "#"

                    newsList.append({
                        "headline": headline,
                        "author": author if author and author != "nan" else "Benzinga",
                        "articleDate": articleDate,
                        "url": articleUrl
                    })
        except Exception:
            pass

        # Calculate 30-day average trading volume
        avgVolume = None
        try:
            df = self.dataProviders.ohlcv.getPeriodDailyTickerData(ticker, targetTs - pd.Timedelta(days=30), targetTs)
            avgVolume = calculate30DayAverageVolume(df, targetTs)
            if not avgVolume:
                avgVolume = None
        except Exception:
            pass

        rawWebsite = profile.website if profile and profile.website else None

        return {
            "companyName": companyName,
            "rawExchange": rawExchange,
            "rawSector": rawSector,
            "rawIndustry": rawIndustry,
            "rawWebsite": rawWebsite,
            "lastPrice": lastPrice,
            "openPrice": openPrice,
            "highPrice": highPrice,
            "lowPrice": lowPrice,
            "prevClose": prevClose,
            "volume": volume,
            "avgVolume": avgVolume,
            "fiftyTwoHigh": fiftyTwoHigh,
            "fiftyTwoLow": fiftyTwoLow,
            "valRes": valRes,
            "newsList": newsList,
            "tenKUrl": tenKUrl,
            "tenKDate": tenKDate,
            "tenQUrl": tenQUrl,
            "tenQDate": tenQDate
        }

    # Compile full inspector payload for target ticker and timestamp
    def getFastInspectorData(self, ticker, targetDateStr, timeframeStr="3M"):
        ticker = ticker.strip().upper()
        try:
            targetTs = pd.Timestamp(targetDateStr).tz_localize(NEW_YORK)
        except Exception:
            targetTs = pd.Timestamp.now(tz=NEW_YORK).normalize()

        fastData = self.fetchFastInspectorMetrics(ticker, targetTs, timeframeStr)
        chartData = self.generateOhlcvChartData(ticker, targetTs, targets=[], horizon=(timeframeStr or "3m").lower())

        def getCompareClass(curr, compareVal):
            if curr is None or compareVal is None:
                return ""
            if curr >= compareVal:
                return "val-up"
            else:
                return "val-down"

        valRes = fastData.get("valRes", {})
        lastPrice = fastData["lastPrice"]
        exchange = "NASDAQ" if fastData["rawExchange"].upper() == "XNAS" else "NYSE"

        metricsDict = {
            "exchange": {"label": "Exchange", "value": exchange},
            "sector": {"label": "Sector", "value": self.formatSectorOrIndustry(fastData["rawSector"])},
            "industry": {"label": "Industry", "value": self.formatSectorOrIndustry(fastData["rawIndustry"])},
            "website": {
                "label": "Website",
                "value": fastData.get("rawWebsite") or "N/A",
                "url": fastData.get("rawWebsite")
            },
            "openToday": {
                "label": "Today Open",
                "value": cleanNumber(fastData.get("openPrice"), NumberType.STOCK_PRICE),
                "colorClass": getCompareClass(lastPrice, fastData.get("openPrice"))
            },
            "highToday": {
                "label": "Today High",
                "value": cleanNumber(fastData.get("highPrice"), NumberType.STOCK_PRICE),
                "colorClass": getCompareClass(lastPrice, fastData.get("highPrice"))
            },
            "lowToday": {
                "label": "Today Low",
                "value": cleanNumber(fastData.get("lowPrice"), NumberType.STOCK_PRICE),
                "colorClass": getCompareClass(lastPrice, fastData.get("lowPrice"))
            },
            "prevClose": {
                "label": "Prev Close",
                "value": cleanNumber(fastData.get("prevClose"), NumberType.STOCK_PRICE),
                "colorClass": getCompareClass(lastPrice, fastData.get("prevClose"))
            },
            "volume": {
                "label": "Volume",
                "value": cleanNumber(fastData.get("volume"), NumberType.LARGE_NUMBER)
            },
            "fiftyTwoWeekHigh": {
                "label": "52W High",
                "value": cleanNumber(fastData.get("fiftyTwoHigh"), NumberType.STOCK_PRICE)
            },
            "fiftyTwoWeekLow": {
                "label": "52W Low",
                "value": cleanNumber(fastData.get("fiftyTwoLow"), NumberType.STOCK_PRICE)
            },
            "marketCap": {
                "label": "Market Cap",
                "value": valRes.get("marketCap")
            },
            "sharesOutstanding": {
                "label": "Shares Outstanding",
                "value": cleanNumber(valRes.get("sharesOutstanding"), NumberType.LARGE_NUMBER)
            },
            "beta": {
                "label": "Beta",
                "value": cleanNumber(valRes.get("beta"), NumberType.DECIMAL)
            },
            "avgVolume": {
                "label": "Avg Volume",
                "value": cleanNumber(fastData.get("avgVolume"), NumberType.LARGE_NUMBER)
            },
            "dividendYield": {
                "label": "Dividend Yield",
                "value": cleanNumber(valRes.get("dividendYield"), NumberType.UNSCALED_PERCENTAGE)
            },
            "latestTenK": {
                "label": "Latest 10-K",
                "value": f"10-K ({fastData['tenKDate']})" if fastData.get("tenKDate") else ("View 10-K" if fastData.get("tenKUrl") else f"None found"),
                "url": fastData.get("tenKUrl")
            },
            "latestTenQ": {
                "label": "Latest 10-Q",
                "value": f"10-Q ({fastData['tenQDate']})" if fastData.get("tenQDate") else ("View 10-Q" if fastData.get("tenQUrl") else f"None found"),
                "url": fastData.get("tenQUrl"),
            }
        }

        logoUrl = None
        try:
            logoUrl = self.dataProviders.tickers.getCompanyLogoFromFinnhub(ticker)
        except Exception:
            pass

        return {
            "ok": True,
            "ticker": ticker,
            "targetDate": targetTs.strftime("%Y-%m-%d"),
            "companyName": fastData["companyName"],
            "logoUrl": logoUrl,
            "lastPrice": lastPrice,
            "chart": chartData,
            "metrics": metricsDict,
            "news": fastData["newsList"]
        }

    
# Global SimulationManager instance
simulationManager = SimulationManager()


# Resolve session identifier from websocket payload
def getWsSessionId(data):
    if not isinstance(data, dict):
        return "default_session"
    reqSessionId = data.get("sessionId") or data.get("headers", {}).get("X-Session-ID")
    return reqSessionId if reqSessionId else "default_session"


# Register simulation websocket action handlers
def registerSimulationWsRoutes():
    def handleSimStart(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        portfolioName = data.get("portfolioName", "My Growth Portfolio").strip() or "My Growth Portfolio"
        startDate = data.get("startDate", "2016-06-01").strip()
        dollarAmountStr = str(data.get("dollarAmount", "1000000"))

        try:
            dollarAmount = float(dollarAmountStr)
            if dollarAmount <= 0:
                return {"ok": False, "error": "Starting dollar amount must be positive."}
        except ValueError:
            return {"ok": False, "error": "Invalid starting dollar balance."}

        try:
            state = simulationManager.createSimulation(sessionId, portfolioName, startDate, dollarAmount)
            fastData = simulationManager.getFastInspectorData("NVDA", startDate, "3M")      # Inspect Nvidia stock by default at 3mo timeframe
            return {"ok": True, "state": state, "inspector": fastData, "sessionId": sessionId}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def handleSimState(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        state = simulationManager.getSimulationState(sessionId)
        return {"ok": True, "state": state}

    def handleSimOrder(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        ticker = data.get("ticker", "").upper().strip()
        side = data.get("side", "BUY").upper().strip()
        orderType = data.get("type", "market").lower().strip()
        amountType = data.get("amountType", "cash").lower().strip()
        amountValueStr = str(data.get("amountValue", "0"))
        limitPriceStr = data.get("limitPrice")
        stopPriceStr = data.get("stopPrice")

        if not ticker:
            return {"ok": False, "error": "Ticker symbol cannot be empty."}

        try:
            amountValue = float(amountValueStr)
            if amountValue <= 0:
                return {"ok": False, "error": "Amount must be greater than zero."}
        except ValueError:
            return {"ok": False, "error": "Amount must be a valid number."}

        try:
            state = simulationManager.submitOrder(
                sessionId=sessionId,
                ticker=ticker,
                side=side,
                orderType=orderType,
                amountType=amountType,
                amountValue=amountValue,
                limitPrice=limitPriceStr,
                stopPrice=stopPriceStr
            )
            return {"ok": True, "state": state}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def handleSimCancelOrder(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        orderIndex = data.get("index")
        if orderIndex is None:
            return {"ok": False, "error": "Missing order index."}

        try:
            state = simulationManager.cancelOrder(sessionId, orderIndex)
            return {"ok": True, "state": state}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def handleSimAdvance(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        daysToAdvanceStr = data.get("days")
        targetDateStr = data.get("targetDate")
        targetDateCutoffStr = data.get("targetDateCutoff")

        try:
            executedDays, state = simulationManager.advanceSimulation(
                sessionId=sessionId,
                days=daysToAdvanceStr,
                targetDate=targetDateStr,
                targetDateCutoff=targetDateCutoffStr
            )
            return {"ok": True, "executedDays": executedDays, "state": state}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def handleSimReset(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        simulationManager.resetSimulation(sessionId)
        return {"ok": True}

    def handleSimTickerInfo(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        ticker = data.get("ticker").strip().upper()
        timeframeVal = data.get("timeframe")
        timeframeStr = timeframeVal.strip() if timeframeVal else "3M"
        # Pass through client job identifier for response correlation
        jobId = data.get("jobId")
        
        simData = simulationManager.userSimulations.get(sessionId)
        defaultDateStr = simData["sim"].currentDate.strftime("%Y-%m-%d") if simData and "sim" in simData and simData["sim"] else "2016-06-01"
        targetDateStr = data.get("targetDate", defaultDateStr)

        try:
            jobId = simulationManager.startAsyncInspectorJob(sessionId, ticker, targetDateStr, timeframeStr, jobId, websocket, eventLoop)
            return {"ok": True, "status": "processing", "jobId": jobId, "targetDate": targetDateStr}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def handleSimTickerPrice(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        ticker = data.get("ticker").strip().upper()
        try:
            return simulationManager.getTickerLastPrice(sessionId, ticker)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def handleSimTickerChart(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        ticker = data.get("ticker").strip().upper()
        timeframeVal = data.get("timeframe")
        timeframeStr = timeframeVal.strip() if timeframeVal else "3M"

        simData = simulationManager.userSimulations.get(sessionId)
        if simData and simData["sim"]:
            simDateTs = simData["sim"].currentDate
        else:
            simDateTs = pd.Timestamp.now(tz=NEW_YORK).normalize()

        try:
            chartData = simulationManager.generateOhlcvChartData(ticker, simDateTs, targets=[], horizon=timeframeStr.lower())
            priceChange = None
            historical = (chartData or {}).get("historical") or []
            currentPrice = (chartData or {}).get("simAnchor", {}).get("y")
            if len(historical) >= 2 and currentPrice is not None:
                startPrice = historical[0].get("y")
                if startPrice:
                    priceChange = ((currentPrice / startPrice) - 1) * 100
            return {"ok": True, "ticker": ticker, "chart": chartData, "priceChange": priceChange}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def handleSimAvailableTickers(data, websocket, eventLoop):
        return {"ok": True, **simulationManager.getAvailableTickers()}

    def handleSimRandomTicker(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        ticker = simulationManager.getRandomListedTicker(sessionId)
        return {"ok": True, "ticker": ticker}

    registerWsAction("sim_start", handleSimStart)
    registerWsAction("sim_state", handleSimState)
    registerWsAction("sim_order", handleSimOrder)
    registerWsAction("sim_cancel_order", handleSimCancelOrder)
    registerWsAction("sim_advance", handleSimAdvance)
    registerWsAction("sim_reset", handleSimReset)
    registerWsAction("sim_ticker_info", handleSimTickerInfo)
    registerWsAction("sim_ticker_price", handleSimTickerPrice)
    registerWsAction("sim_ticker_chart", handleSimTickerChart)
    registerWsAction("sim_available_tickers", handleSimAvailableTickers)
    registerWsAction("sim_random_ticker", handleSimRandomTicker)
