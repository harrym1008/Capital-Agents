import json
import time
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from flask import request, jsonify

from collectors.constants import NEW_YORK
from collectors.rate_limiter import GlobalRateLimiters
from dataquery import LRUCache, DailyPriceProvider, TickerDataProvider
from llm.llamacpp.llamacpp_args import LlamaCppModel, LLAMACPP_PORT
from llm.server_manager import serverManager

globalCache = None
globalTickerProvider = None
globalPriceProvider = None
globalRateLimiters = None

def getOhlcvProviders():
    global globalCache, globalTickerProvider, globalPriceProvider, globalRateLimiters
    if globalPriceProvider is None:
        globalCache = LRUCache(256 * 1024 ** 2)
        globalTickerProvider = TickerDataProvider()
        globalRateLimiters = GlobalRateLimiters()
        globalPriceProvider = DailyPriceProvider(globalTickerProvider, globalCache, globalRateLimiters)
    return globalTickerProvider, globalPriceProvider

def adjustPriceDataSplits(priceData, referenceDate=None, referenceSplitFactor=None):
    if priceData.empty or "splitFactor" not in priceData.columns:
        return priceData
    
    priceData = priceData.copy()
    if referenceSplitFactor is not None:
        finalSplitFactor = referenceSplitFactor
    elif referenceDate is not None and "dateNy" in priceData.columns:
        refNy = pd.Timestamp(referenceDate).normalize()
        matches = priceData[priceData["dateNy"] == refNy]
        if not matches.empty:
            finalSplitFactor = matches["splitFactor"].iloc[0]
        else:
            finalSplitFactor = priceData["splitFactor"].iloc[-1]
    else:
        finalSplitFactor = priceData["splitFactor"].iloc[-1]

    if pd.notna(finalSplitFactor) and finalSplitFactor != 0:
        for col in ["open", "high", "low", "close", "vwap"]:
            if col in priceData.columns:
                priceData[col] = priceData[col] * (priceData["splitFactor"] / finalSplitFactor)
        priceData["splitFactor"] = finalSplitFactor
    return priceData


def getRollingMean(series, window=3):
    s = pd.Series(series)
    return s.rolling(window=window, center=True, min_periods=1).mean().values


def computeBezierThroughPoints(pointsX, pointsY, numSamples=300):
    n = len(pointsX)
    if n < 2:
        return np.array(pointsX), np.array(pointsY)
    
    # Convert to numpy for computation
    xPts = np.array(pointsX, dtype=np.float64)
    yPts = np.array(pointsY, dtype=np.float64)
    
    # Catmull-Rom spline interpolation
    
    def catmullRomPoint(t, p0, p1, p2, p3):
        t2 = t * t
        t3 = t2 * t
        return 0.5 * (
            2 * p1 +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
            (-p0 + 3 * p1 - 3 * p2 + p3) * t3
        )
    
    # Build the full curve by interpolating between each pair of consecutive points
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
    
    # Always include the last point exactly
    xAll.append(xPts[-1])
    yAll.append(yPts[-1])
    
    return np.array(xAll), np.array(yAll)


def getFuturePerformanceChunks(futureDates, futureCloses, splineVals):
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


def generateOhlcvChartData(ticker, simDateTs, targets=None, horizon="long"):
    todayTs = pd.Timestamp.now(tz=NEW_YORK).normalize()
    if targets is None:
        targets = []

    horizonStr = (horizon or "long").lower()
    
    # Auto-infer horizon from targets if default 'long' was passed but targets indicate another horizon
    if targets and horizonStr == "long":
        maxMonths = max([t[0] for t in targets])
        if maxMonths <= 3:
            horizonStr = "short"
        elif maxMonths <= 12:
            horizonStr = "medium"
        elif maxMonths > 36:
            horizonStr = "distant"

    if horizonStr in ["1m", "month"]:
        startDateTs = simDateTs - pd.DateOffset(months=1)
        endDateTs = simDateTs + pd.DateOffset(months=1)
    elif horizonStr in ["3m", "short"]:
        startDateTs = simDateTs - pd.DateOffset(months=3)
        endDateTs = simDateTs + pd.DateOffset(months=3)
    elif horizonStr in ["medium", "med", "1y"]:
        startDateTs = simDateTs - pd.DateOffset(years=1)
        endDateTs = simDateTs + pd.DateOffset(years=1)
    elif horizonStr in ["distant", "3y", "3year"]:
        startDateTs = simDateTs - pd.DateOffset(years=3)
        endDateTs = simDateTs + pd.DateOffset(years=3)
    elif horizonStr in ["all", "max"]:
        startDateTs = pd.Timestamp("2000-01-01").tz_localize(NEW_YORK)
        endDateTs = simDateTs + pd.DateOffset(years=1)
    else:  # default 3m
        startDateTs = simDateTs - pd.DateOffset(months=3)
        endDateTs = simDateTs + pd.DateOffset(months=3)
    
    tickerProvider, priceProvider = getOhlcvProviders()
    profile = tickerProvider.getTickerProfile(ticker)
    
    if profile and profile.ipoDate is not None:
        if profile.ipoDate > startDateTs:
            startDateTs = profile.ipoDate
            
    dfHistorical = priceProvider.getPeriodDailyTickerData(ticker, startDateTs, simDateTs, referenceDate=simDateTs)
    dfHistorical = adjustPriceDataSplits(dfHistorical, referenceDate=simDateTs)
    
    dfFuture = pd.DataFrame()
    if simDateTs < todayTs:
        futureEndTs = min(endDateTs, todayTs)
        dfFuture = priceProvider.getPeriodDailyTickerData(ticker, simDateTs, futureEndTs, referenceDate=simDateTs)
        dfFuture = adjustPriceDataSplits(dfFuture, referenceDate=simDateTs)
        
    lastHistoricalClose = None
    allPrices = []
    
    historicalPoints = []
    if not dfHistorical.empty and "close" in dfHistorical.columns:
        dfCleanHist = dfHistorical.dropna(subset=["close"])
        if not dfCleanHist.empty:
            histDates = [pd.Timestamp(r.get("dateNy", r.get("date"))).to_pydatetime() for _, r in dfCleanHist.iterrows()]
            closesRaw = dfCleanHist["close"].values
            histClosesSmooth = getRollingMean(closesRaw, window=3)
            lastHistoricalClose = histClosesSmooth[-1]
            allPrices.extend(histClosesSmooth.tolist())
            for d, c in zip(histDates, histClosesSmooth):
                historicalPoints.append({"x": d.strftime("%Y-%m-%d"), "y": round(float(c), 2)})
            
    futureDates = []
    futureClosesSmooth = []
    if not dfFuture.empty and "close" in dfFuture.columns:
        dfCleanFuture = dfFuture.dropna(subset=["close"])
        if not dfCleanFuture.empty:
            futureDates = [pd.Timestamp(r.get("dateNy", r.get("date"))).to_pydatetime() for _, r in dfCleanFuture.iterrows()]
            closesRaw = dfCleanFuture["close"].values
            futureClosesSmooth = getRollingMean(closesRaw, window=3)
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
        curveX, curveY = computeBezierThroughPoints(targetX, targetY, numSamples=400)
        for x, y in zip(curveX, curveY):
            dt = pd.Timestamp(round(startTime + x * 86400.0), unit='s', tz=NEW_YORK)
            curvePoints.append({"x": dt.strftime("%Y-%m-%d"), "y": round(float(y), 2)})

    futureChunks = []
    if len(futureDates) > 0:
        if curveX is not None and len(curveX) > 0:
            futureDays = np.array([(pd.Timestamp(d) - startDateTs).total_seconds() / 86400.0 for d in futureDates])
            splineVals = np.interp(futureDays, curveX, curveY)
        futureChunks = getFuturePerformanceChunks(futureDates, futureClosesSmooth, splineVals)

    return {
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



openRouterModelsCache = None
openRouterCacheTime = 0


def registerApiRoutes(app):
    @app.route("/api/llamacpp-models")
    def getLlamaCppModels():
        models = [
            member.name
            for member in LlamaCppModel
            if member.name != "SUMMARY_MODEL"
        ]
        return jsonify({"models": models})

    @app.route("/api/openrouter-models")
    def getOpenRouterModels():
        global openRouterModelsCache, openRouterCacheTime
        currentTime = time.time()
        if openRouterModelsCache is not None and (currentTime - openRouterCacheTime) < 3600:
            return jsonify({"models": openRouterModelsCache})

        try:
            url = "https://openrouter.ai/api/v1/models"
            req = urllib.request.Request(url, headers={"User-Agent": "CapitalAgents"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                modelsData = data.get("data", [])
                openRouterModelsCache = modelsData
                openRouterCacheTime = currentTime
                return jsonify({"models": modelsData})
        except Exception as e:
            if openRouterModelsCache is not None:
                return jsonify({"models": openRouterModelsCache})
            return jsonify({"models": [], "error": str(e)}), 500

    @app.route("/api/openrouter-endpoints")
    def getOpenRouterEndpoints():
        modelId = request.args.get("model", "").strip()
        if not modelId:
            return jsonify({"providers": [], "endpoints": []})

        try:
            url = f"https://openrouter.ai/api/v1/models/{modelId}/endpoints"
            req = urllib.request.Request(url, headers={"User-Agent": "CapitalAgents"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                endpointsData = data.get("data", {}).get("endpoints", [])
                providers = []
                seen = set()
                for ep in endpointsData:
                    pName = ep.get("provider_name")
                    if pName and pName not in seen:
                        seen.add(pName)
                        providers.append(pName)
                return jsonify({"providers": providers, "endpoints": endpointsData})
        except Exception as e:
            return jsonify({"providers": [], "endpoints": [], "error": str(e)})


    @app.route("/api/server/start", methods=["POST"])
    @app.route("/api/llamacpp/start", methods=["POST"])
    @app.route("/api/openrouter/start", methods=["POST"])
    def apiStartServer():
        data = request.get_json(silent=True) or {}
        provider = data.get("provider")
        if not provider:
            provider = "openrouter" if request.path.endswith("/openrouter/start") else "llamacpp"

        modelName = data.get("model", "GEMMA_4_12B").strip()
        providerRouter = data.get("providerRouter") or data.get("router")
        allowParallel = data.get("allowParallel", True)
        wantSummaryServer = data.get("wantSummaryServer", False)

        success, message = serverManager.startServer(
            provider=provider,
            modelName=modelName,
            providerRouter=providerRouter,
            allowParallel=allowParallel,
            wantSummaryServer=wantSummaryServer
        )
        return jsonify({"ok": success, "message": message})


    @app.route("/api/server/stop", methods=["POST"])
    @app.route("/api/llamacpp/stop", methods=["POST"])
    @app.route("/api/openrouter/stop", methods=["POST"])
    @app.route("/api/llamacpp/summary/stop", methods=["POST"])
    def apiStopServer():
        message = serverManager.stopServer()
        return jsonify({"ok": True, "message": message})

    @app.route("/api/server/status")
    def apiServerStatus():
        return jsonify(serverManager.getStatus())

    @app.route("/api/ohlcv")
    def getOhlcvChart():
        ticker = request.args.get("ticker", "NVDA").strip()
        simDateStr = request.args.get("simDate")
        horizon = request.args.get("horizon", "long").strip().lower()

        if not simDateStr:
            simDateTs = pd.Timestamp.now(tz=NEW_YORK).normalize()
        else:
            try:
                simDateTs = pd.Timestamp(simDateStr, tz=NEW_YORK)
            except Exception:
                simDateTs = pd.Timestamp.now(tz=NEW_YORK).normalize()

        targetsRaw = request.args.get("targets", "")
        targetsList = []
        if targetsRaw:
            for item in targetsRaw.split(","):
                if ":" in item:
                    try:
                        monthsOffset, price = item.split(":")
                        targetsList.append((float(monthsOffset), float(price)))
                    except ValueError:
                        pass

        try:
            chartData = generateOhlcvChartData(ticker, simDateTs, targets=targetsList, horizon=horizon)
            return jsonify(chartData)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

