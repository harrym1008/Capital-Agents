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

from dataquery.lru_cache import LRUCache
from dataquery.price_provider import DailyPriceProvider
from dataquery.ticker_provider import TickerDataProvider

from llmtools.registry_builder import buildToolRegistry
from llmtools.tool_registry import DataProviders

from llmtools.functions.helpers import cleanNumber, NumberType
from llmtools.functions.edgar import FormType, CompanyRef, calculateHistoricalBeta, calculateDividendYield
from llmtools.functions.company import calculate30DayAverageVolume, calculateSharpeRatio

from simulation.market_sim import MarketSimulation
from simulation.orders import MarketOrder, LimitOrder, StopOrder, StopLimitOrder, OrderSide, OrderStatus


class SimulationManager:
    def __init__(self, cacheSizeBytes=256 * 1024 ** 2):
        self.simCache = LRUCache(cacheSizeBytes)
        self.tickerProvider = TickerDataProvider()
        self.rateLimiters = GlobalRateLimiters()
        self.priceProvider = DailyPriceProvider(self.tickerProvider, self.simCache, self.rateLimiters)
        self.ohlcvChartCache = {}

        self.userSimulations = {}
        self.userHistory = {}
        self.inspectorExecutor = ThreadPoolExecutor(max_workers=4)
        self.activeInspectorJobs = {}

        # Build available tickers index
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

    def getRollingMean(self, series, window=3):
        s = pd.Series(series)
        return s.rolling(window=window, center=True, min_periods=1).mean().values

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
        else:   # elif horizonStr == "distant":
            startDateTs = simDateTs - pd.DateOffset(years=2)
            endDateTs = simDateTs + pd.DateOffset(years=10)
        
        if targets:
            maxTargetMonths = max([t[0] for t in targets])
            targetEndTs = simDateTs + pd.DateOffset(months=int(maxTargetMonths))
            if targetEndTs > endDateTs:
                endDateTs = targetEndTs
        
        profile = self.tickerProvider.getTickerProfile(ticker)
        if profile and profile.ipoDate is not None:
            ipoTs = pd.Timestamp(profile.ipoDate)
            if ipoTs.tzinfo is None:
                ipoTs = ipoTs.tz_localize(NEW_YORK)
            else:
                ipoTs = ipoTs.tz_convert(NEW_YORK)
            if ipoTs > startDateTs:
                startDateTs = ipoTs

        # Uses self.priceProvider (which shares self.simCache and priceProvider.adjustPriceDataSplits)
        dfHistorical = self.priceProvider.getPeriodDailyTickerData(ticker, startDateTs, simDateTs, referenceDate=simDateTs)
        
        dfFuture = pd.DataFrame()
        if targets and simDateTs < todayTs:
            futureEndTs = min(endDateTs, todayTs)
            dfFuture = self.priceProvider.getPeriodDailyTickerData(ticker, simDateTs, futureEndTs, referenceDate=simDateTs)
            
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

    def getSessionId(self):
        reqSessionId = request.headers.get("X-Session-ID") or request.args.get("sessionId")
        if reqSessionId:
            return reqSessionId
        if "sim_session_id" not in session:
            session["sim_session_id"] = f"session_{time.perf_counter_ns()}"
        return session["sim_session_id"]

    def createSimulation(self, sessionId, portfolioName, startDate, initialCash):
        activeSim = MarketSimulation(startDate, END_DATE_STR, 
                                     tickerDataProvider=self.tickerProvider, dailyPriceProvider=self.priceProvider)
        activeSim.initialiseUser(portfolioName, initialCash=initialCash)

        self.userSimulations[sessionId] = {
            "sim": activeSim,
            "portfolioName": portfolioName,
            "inspectedTicker": "NVDA",
            "inspectedTimeframe": "3M"
        }
        
        # Pre-populate 2 preceding days at starting balance so Chart.js always draws a line at start
        startTs = pd.Timestamp(startDate)
        dayMinus2 = (startTs - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        dayMinus1 = (startTs - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        self.userHistory[sessionId] = [
            {"date": dayMinus2, "totalValue": float(initialCash)},
            {"date": dayMinus1, "totalValue": float(initialCash)}
        ]
        self.updateSimulationHistory(sessionId)
        return self.getSimulationState(sessionId)

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

        # Always include the lightweight top-half snapshot (name, price, price change,
        # chart) for the inspected ticker - even during rapid stepping - so the frontend
        # can update the chart/price live on every advance step.
        inspectedTicker = simData.get("inspectedTicker", "NVDA")
        inspectedTimeframe = simData.get("inspectedTimeframe", "3M")
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
        else:  # market
            order = MarketOrder(ticker, sideEnum, quantity=amountValue) if amountType == "quantity" else MarketOrder(ticker, sideEnum, cashValue=amountValue)

        activeSim.addOrder(order, portfolioName)
        return self.getSimulationState(sessionId)

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

    def getCurrentlyListedTickers(self, sessionId=None):
        simData = self.userSimulations.get(sessionId) if sessionId else None
        if simData and simData["sim"]:
            simDate = simData["sim"].currentDate
        else:
            simDate = pd.Timestamp.now(tz=NEW_YORK).normalize()

        listedTickers = [
            t for t in self.availableTickers
            if self.tickerProvider.isTickerListed(t, simDate)
        ]
        return listedTickers if len(listedTickers) > 0 else self.availableTickers

    def getRandomListedTicker(self, sessionId=None):
        listed = self.getCurrentlyListedTickers(sessionId)
        return random.choice(listed)

    def advanceSimulation(self, sessionId, days=None, targetDate=None, targetDateCutoff=None):
        simData = self.userSimulations.get(sessionId)
        if not simData:
            raise ValueError("No active simulation session.")

        activeSim = simData["sim"]
        portfolioName = simData["portfolioName"]

        # When targetDateCutoff is used, we're in the rapid-stepping loop from the frontend.
        # Skip expensive inspector data on intermediate steps.
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

        state = self.getSimulationState(sessionId, skipInspector=isRapidStep)
        return executedDays, state

    def resetSimulation(self, sessionId):
        if sessionId in self.userSimulations:
            del self.userSimulations[sessionId]
        if sessionId in self.userHistory:
            del self.userHistory[sessionId]
        return True

    def getTickerInfo(self, sessionId, ticker, timeframe=None, targetDate=None):
        ticker = ticker.strip().upper()
        simData = self.userSimulations.get(sessionId)

        if simData:
            simData["inspectedTicker"] = ticker
            if timeframe:
                simData["inspectedTimeframe"] = str(timeframe).strip().upper()

        inspectedTimeframe = simData.get("inspectedTimeframe", "3M") if simData else (timeframe or "3M").upper()
        timeframeStr = inspectedTimeframe.lower()

        # When a targetDate is supplied (e.g. the async inspector job for an advance),
        # generate the chart/price for that date rather than the sim's current date, so
        # the pushed chart matches the date the simulation advanced to.
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

        profile = self.tickerProvider.getTickerProfile(ticker)
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

    def getTickerSnapshot(self, sessionId, ticker, timeframe="3M"):
        # Lightweight top-half data for the ticker inspector: name, current price,
        # price change over the timeframe, and the chart. No EDGAR, no news - fast.
        # This is included in every simulation state so the top half updates live.
        ticker = ticker.strip().upper()
        simData = self.userSimulations.get(sessionId)

        if simData and simData["sim"]:
            simDateTs = simData["sim"].currentDate
        else:
            simDateTs = pd.Timestamp.now(tz=NEW_YORK).normalize()

        profile = self.tickerProvider.getTickerProfile(ticker)
        companyName = profile.name if profile and profile.name else ticker

        logoUrl = None
        try:
            logoUrl = self.tickerProvider.getCompanyLogoFromFinnhub(ticker)
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

        # Use the raw close at the sim date (matching the positions table & order form
        # last price) rather than the split-adjusted chart anchor, so they all agree.
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

    def getTickerLastPrice(self, sessionId, ticker):
        # Lightweight synchronous lookup used by the order form. Avoids the heavy
        # chart/profile computation that getTickerInfo performs.
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

        # Fall back to the chart anchor if the single-day close is missing or zero, so the
        # order form never shows $0.00. generateOhlcvChartData is cached, so this is cheap.
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

    def startAsyncInspectorJob(self, sessionId, ticker, targetDateStr, timeframeStr="3M", jobId=None, websocket=None, eventLoop=None):
        # Use the jobId supplied by the frontend so both sides stay in sync.
        # This avoids the previous desync where the server kept its own independent counter.
        if jobId is None:
            jobId = self.activeInspectorJobs.get(sessionId, 0) + 1
        self.activeInspectorJobs[sessionId] = jobId

        def worker():
            if self.activeInspectorJobs.get(sessionId) != jobId:
                return

            tickerClean = (ticker or "NVDA").strip().upper()
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
                # Always send a push (even on failure) so the frontend can unblur
                # instead of waiting forever for data that will never arrive.
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

    def getAvailableTickers(self):
        return {
            "tickers": self.availableTickers,
            "totalCount": self.allTickersCount,
            "availableCount": len(self.availableTickers)
        }


    def formatSectorOrIndustry(self, rawStr):
        if not rawStr or pd.isna(rawStr):
            return "N/A"
        cleanStr = str(rawStr).replace("_", " ").strip()
        return cleanStr.title().replace("And", "and") if cleanStr else "N/A"

    def fetchFastInspectorMetrics(self, dataProviders: DataProviders, ticker, targetTs, timeframeStr="3M"):
        profile = dataProviders.tickers.getTickerProfile(ticker)
        rawExchange = profile.exchange if profile and profile.exchange else "NASDAQ"
        rawSector = profile.sector if profile and profile.sector else "N/A"
        rawIndustry = profile.industry if profile and profile.industry else "N/A"
        companyName = profile.name if profile and profile.name else ticker

        startDate = targetTs - pd.Timedelta(days=365)
        df = dataProviders.ohlcv.getPeriodDailyTickerData(ticker, startDate, targetTs)

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

                # Calculate sharpe ratio
                closeSeries = df["close"].dropna()
                if len(closeSeries) >= 5:
                    pctChange = closeSeries.pct_change().dropna()
                    if len(pctChange) >= 5:
                        tail30 = pctChange.tail(30)
                        if len(tail30) > 1 and tail30.std() > 0:
                            volatility30d = float(tail30.std() * (252 ** 0.5))
                            sharpeRatio = float((pctChange.mean() / pctChange.std()) * (252 ** 0.5))

                    if len(closeSeries) >= 15:
                        delta = closeSeries.diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        if not loss.empty and loss.iloc[-1] != 0:
                            rs = gain.iloc[-1] / loss.iloc[-1]
                            rsi14 = float(100 - (100 / (1 + rs)))
                        elif not gain.empty and gain.iloc[-1] > 0:
                            rsi14 = 100.0
            except Exception:
                pass

        valRes = {}

        # Get market cap, shares outstanding
        if df is not None and not df.empty:
            lastShares = df["outstandingShares"].dropna()
            lastMarketCap = df["marketCap"].dropna()
            if not lastShares.empty:
                valRes["sharesOutstanding"] = float(lastShares.iloc[-1])
                valRes["marketCap"] = "$" + str(lastMarketCap.iloc[-1])

        companyRef = CompanyRef(ticker)
        valRes["beta"] = calculateHistoricalBeta(companyRef, targetTs, dataProviders.ohlcv, dataProviders.macro)
        valRes["dividendYield"] = calculateDividendYield(companyRef, targetTs, dataProviders.ohlcv)

        # Find the latest 10-K and 10-Q filings before the target date
        tenKUrl = None
        tenKDate = None
        tenQUrl = None
        tenQDate = None
        try:
            tenKFiling = dataProviders.edgar.getLatestFilingRef(companyRef, FormType.FORM_10K, before=targetTs)
            if tenKFiling:
                tenKUrl = getattr(tenKFiling, "filing_url", None) or getattr(tenKFiling, "url", None)
                rawFDate = getattr(tenKFiling, "filing_date", None)
                if rawFDate:
                    try:
                        tenKDate = pd.to_datetime(rawFDate).strftime("%d %b %Y")
                    except Exception:
                        pass

            tenQFiling = dataProviders.edgar.getLatestFilingRef(companyRef, FormType.FORM_10Q, before=targetTs)
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

        # Get news headlines
        newsList = []
        try:
            rawNews = dataProviders.news.getRecentNewsForTicker(ticker, before=targetTs, limit=50)
            if rawNews is not None and isinstance(rawNews, pd.DataFrame) and not rawNews.empty:
                for _, row in rawNews.iterrows():
                    headline = str(row.get("headline", ""))
                    author = str(row.get("author", "Benzinga"))
                    rawDate = row.get("date")
                    articleDate = ""
                    try:
                        articleDate = pd.to_datetime(rawDate).strftime("%d %b %Y")  # 23 Jul 2026 for example
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

        # 30-day average volume (from the fetchStockPricePerformance helper)
        avgVolume = None
        try:
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


    def getFastInspectorData(self, ticker, targetDateStr, timeframeStr="3M"):
        ticker = (ticker or "NVDA").strip().upper()
        try:
            targetTs = pd.Timestamp(targetDateStr).tz_localize(NEW_YORK)
        except Exception:
            targetTs = pd.Timestamp.now(tz=NEW_YORK).normalize()

        reg = buildToolRegistry()
        del reg.dataProviders
        reg.dataProviders = DataProviders(allowOnlineDownloads=False)    # Bad hack but it works
        dataProviders = reg.dataProviders

        fastData = self.fetchFastInspectorMetrics(dataProviders, ticker, targetTs, timeframeStr)

        chartData = self.generateOhlcvChartData(ticker, targetTs, targets=[], horizon=(timeframeStr or "3m").lower())
        # if fastData["lastPrice"] is None and chartData and chartData.get("simAnchor"):
        #     fastData["lastPrice"] = chartData["simAnchor"].get("y")


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
            logoUrl = self.tickerProvider.getCompanyLogoFromFinnhub(ticker)
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


def getWsSessionId(data):
    if not isinstance(data, dict):
        return "default_session"
    reqSessionId = data.get("sessionId") or data.get("headers", {}).get("X-Session-ID")
    return reqSessionId if reqSessionId else "default_session"


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
            fastData = simulationManager.getFastInspectorData("NVDA", startDate, "3M")
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
        ticker = data.get("ticker", "NVDA").strip().upper()
        timeframeVal = data.get("timeframe")
        timeframeStr = timeframeVal.strip() if timeframeVal else "3M"
        # Pass through the frontend-generated jobId so the push can be matched reliably.
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
        ticker = data.get("ticker", "NVDA").strip().upper()
        try:
            return simulationManager.getTickerLastPrice(sessionId, ticker)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def handleSimTickerChart(data, websocket, eventLoop):
        sessionId = getWsSessionId(data)
        ticker = data.get("ticker", "NVDA").strip().upper()
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
