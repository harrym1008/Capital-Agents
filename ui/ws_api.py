import io
import base64
import urllib.request
import urllib.error
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from flask import request, jsonify

from collectors.constants import NEW_YORK
from dataquery import LRUCache, DailyPriceProvider, TickerDataProvider
from llm.llamacpp.llamacpp_args import LlamaCppModel, LLAMACPP_PORT
from llm.server_manager import serverManager

globalCache = None
globalTickerProvider = None
globalPriceProvider = None

def getOhlcvProviders():
    global globalCache, globalTickerProvider, globalPriceProvider
    if globalPriceProvider is None:
        globalCache = LRUCache(256 * 1024 ** 2)
        globalTickerProvider = TickerDataProvider()
        globalPriceProvider = DailyPriceProvider(globalTickerProvider, globalCache)
    return globalTickerProvider, globalPriceProvider

def adjustPriceDataSplits(priceData):
    if priceData.empty or "splitFactor" not in priceData.columns:
        return priceData
    
    priceData = priceData.copy()
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


def splitAndPlotPerformance(ax, futureDates, futureCloses, splineVals, yMin):
    numPoints = len(futureCloses)
    if numPoints == 0:
        return
        
    if splineVals is None or len(splineVals) != numPoints:
        ax.plot(futureDates, futureCloses, color='#f43f5e', linewidth=1.8, label='Actual Performance', zorder=3)
        ax.fill_between(futureDates, futureCloses, yMin, color='#f43f5e', alpha=0.08, zorder=2)
        return
        
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
    
    isGreenPoint = refinedDiffsArr >= 0
    isRedPoint = refinedDiffsArr <= 0
    
    def plotContiguousChunks(mask, color):
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
                        ax.plot(cDates, cCloses, color=color, linewidth=1.8, zorder=3)
                        ax.fill_between(cDates, cCloses, yMin, color=color, alpha=0.08, zorder=2)
                    inChunk = False
        if inChunk:
            cDates = refinedDatesArr[chunkStart:n]
            cCloses = refinedClosesArr[chunkStart:n]
            if len(cDates) >= 2:
                ax.plot(cDates, cCloses, color=color, linewidth=1.8, zorder=3)
                ax.fill_between(cDates, cCloses, yMin, color=color, alpha=0.08, zorder=2)

    plotContiguousChunks(isGreenPoint, '#10b981')
    plotContiguousChunks(isRedPoint, '#f43f5e')


def generateOhlcvChartImage(ticker, simDateTs, targets=None):
    todayTs = pd.Timestamp.now(tz=NEW_YORK).normalize()
    startDateTs = simDateTs - pd.DateOffset(years=3)
    endDateTs = simDateTs + pd.DateOffset(years=3)
    
    tickerProvider, priceProvider = getOhlcvProviders()
    profile = tickerProvider.getTickerProfile(ticker)
    
    if profile and profile.ipoDate is not None:
        if profile.ipoDate > startDateTs:
            startDateTs = profile.ipoDate
            
    dfHistorical = priceProvider.getPeriodDailyTickerData(ticker, startDateTs, simDateTs)
    dfHistorical = adjustPriceDataSplits(dfHistorical)
    
    dfFuture = pd.DataFrame()
    if simDateTs < todayTs:
        futureEndTs = min(endDateTs, todayTs)
        dfFuture = priceProvider.getPeriodDailyTickerData(ticker, simDateTs, futureEndTs)
        dfFuture = adjustPriceDataSplits(dfFuture)
        
    lastHistoricalClose = None
    allPrices = []
    
    histDates = []
    histClosesSmooth = []
    if not dfHistorical.empty and "close" in dfHistorical.columns:
        dfCleanHist = dfHistorical.dropna(subset=["close"])
        if not dfCleanHist.empty:
            histDates = [pd.Timestamp(r.get("dateNy", r.get("date"))).to_pydatetime() for _, r in dfCleanHist.iterrows()]
            closesRaw = dfCleanHist["close"].values
            histClosesSmooth = getRollingMean(closesRaw, window=3)
            lastHistoricalClose = histClosesSmooth[-1]
            allPrices.extend(histClosesSmooth.tolist())
            
    futureDates = []
    futureClosesSmooth = []
    if not dfFuture.empty and "close" in dfFuture.columns:
        dfCleanFuture = dfFuture.dropna(subset=["close"])
        if not dfCleanFuture.empty:
            futureDates = [pd.Timestamp(r.get("dateNy", r.get("date"))).to_pydatetime() for _, r in dfCleanFuture.iterrows()]
            closesRaw = dfCleanFuture["close"].values
            futureClosesSmooth = getRollingMean(closesRaw, window=3)
            allPrices.extend(futureClosesSmooth.tolist())
            
    if targets is None:
        targets = []
    
    startTime = startDateTs.timestamp()
    simDateDaysFromStart = (simDateTs - startDateTs).total_seconds() / 86400.0
    targetX = [simDateDaysFromStart]
    targetY = [lastHistoricalClose if lastHistoricalClose is not None else (allPrices[0] if allPrices else 100.0)]
    
    for monthsOffset, price in targets:
        targetDateTs = simDateTs + pd.DateOffset(months=int(monthsOffset))
        daysFromStart = (targetDateTs - startDateTs).total_seconds() / 86400.0
        targetX.append(daysFromStart)
        targetY.append(price)
        allPrices.append(price)
        
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
    
    fig, ax = plt.subplots(figsize=(5.6, 2.7), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    if len(histDates) > 0:
        ax.plot(histDates, histClosesSmooth, color='#2563eb', linewidth=1.8, label='Historical (-3Y)')
        ax.fill_between(histDates, histClosesSmooth, yMin, color='#2563eb', alpha=0.08)
        
    curveX, curveY = None, None
    if len(targets) > 0:
        curveX, curveY = computeBezierThroughPoints(targetX, targetY, numSamples=400)
        curveDates = [pd.Timestamp(startTime + x * 86400.0, unit='s', tz=NEW_YORK).to_pydatetime() for x in curveX]
        ax.plot(curveDates, curveY, color='#06b6d4', linewidth=2.8, solid_capstyle='round', zorder=5)
        
        for i, (monthsOffset, price) in enumerate(targets):
            targetDateTs = simDateTs + pd.DateOffset(months=int(monthsOffset))
            ax.plot(targetDateTs.to_pydatetime(), price, marker='o',
                    markersize=8, color='#06b6d4', zorder=6,
                    markeredgecolor='#ffffff', markeredgewidth=1.5)
                    
        ax.plot(simDateTs.to_pydatetime(), targetY[0], marker='o',
                markersize=5, color='#64748b', zorder=6,
                markeredgecolor='#ffffff', markeredgewidth=1)
                
    if len(futureDates) > 0:
        splineVals = None
        if curveX is not None and len(curveX) > 0:
            futureDays = np.array([(pd.Timestamp(d) - startDateTs).total_seconds() / 86400.0 for d in futureDates])
            splineVals = np.interp(futureDays, curveX, curveY)
            
        splitAndPlotPerformance(ax, futureDates, futureClosesSmooth, splineVals, yMin)
        
    ax.axvline(x=simDateTs.to_pydatetime(), color='#1e293b', linestyle='--', linewidth=1.5, zorder=4)
    
    xEnd = endDateTs.to_pydatetime()
    if len(targets) > 0:
        xRangeSeconds = (endDateTs - startDateTs).total_seconds()
        xEnd = endDateTs + pd.Timedelta(seconds=xRangeSeconds * 0.02)
        
    ax.set_xlim(startDateTs.to_pydatetime(), xEnd)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax.set_ylim(bottom=yMin, top=yMax)
    
    ax.grid(True, linestyle=':', alpha=0.4, color='#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#334155')
    ax.spines['bottom'].set_linewidth(1.2)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    ax.tick_params(colors='#334155', labelsize=8.5, length=3)
    for t in ax.get_xticklabels():
        t.set_fontweight('bold')
    for t in ax.get_yticklabels():
        t.set_fontweight('bold')
        
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.13)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor='#ffffff', edgecolor='none', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    
    imgB64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{imgB64}"

def registerApiRoutes(app):
    @app.route("/api/llamacpp-models")
    def getLlamaCppModels():
        models = [
            member.name
            for member in LlamaCppModel
            if member.name != "SUMMARY_MODEL"
        ]
        return jsonify({"models": models})

    @app.route("/api/metrics")
    def getBoardroomMetrics():
        metricsUrl = f"http://127.0.0.1:{LLAMACPP_PORT}/metrics"
        try:
            req = urllib.request.Request(metricsUrl)
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                content = resp.read().decode('utf-8')
                return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        except Exception as e:
            return f"# Error reaching llama-server metrics: {e}", 503, {'Content-Type': 'text/plain; charset=utf-8'}

    @app.route("/api/ohlcv")
    def getOhlcvData():
        tickerParam = request.args.get("ticker", "NVDA").strip().upper()
        simDateParam = request.args.get("simDate", None)
        
        # Parse dynamic targets: comma-separated "months:price" pairs (e.g. "12:75,36:100")
        targetsRaw = request.args.get("targets", "")
        targets = []
        if targetsRaw and targetsRaw != "--":
            for pair in targetsRaw.split(","):
                pair = pair.strip()
                if not pair or ":" not in pair:
                    continue
                parts = pair.split(":")
                try:
                    monthsOffset = int(parts[0].strip())
                    price = float(parts[1].replace("$", "").strip())
                    targets.append((monthsOffset, price))
                except (ValueError, IndexError):
                    pass
        
        todayTs = pd.Timestamp.now(tz=NEW_YORK).normalize()
        
        if not simDateParam:
            simDateTs = todayTs
        else:
            try:
                simDateTs = pd.Timestamp(simDateParam)
                if simDateTs.tzinfo is None:
                    simDateTs = simDateTs.tz_localize(NEW_YORK)
                else:
                    simDateTs = simDateTs.tz_convert(NEW_YORK)
                simDateTs = simDateTs.normalize()
            except Exception:
                simDateTs = todayTs
                
        simDateStr = simDateTs.strftime("%Y-%m-%d")
        
        chartImgStr = generateOhlcvChartImage(tickerParam, simDateTs, targets=targets)
        
        return jsonify({
            "ticker": tickerParam,
            "simDate": simDateStr,
            "chartImage": chartImgStr
        })

    @app.route("/api/server/start", methods=["POST"])
    @app.route("/api/llamacpp/start", methods=["POST"])
    @app.route("/api/openrouter/start", methods=["POST"])
    def apiStartServer():
        data = request.get_json(silent=True) or {}
        provider = data.get("provider")
        if not provider:
            provider = "openrouter" if request.path.endswith("/openrouter/start") else "llamacpp"

        modelName = data.get("model", "GEMMA_4_12B").strip()
        allowParallel = data.get("allowParallel", True)
        wantSummaryServer = data.get("wantSummaryServer", False)

        success, message = serverManager.startServer(
            provider=provider,
            modelName=modelName,
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
