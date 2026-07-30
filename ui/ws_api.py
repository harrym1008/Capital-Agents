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


def generateOhlcvChartImage(ticker, simDateTs, targets=None):
    todayTs = pd.Timestamp.now(tz=NEW_YORK).normalize()
    startDateTs = simDateTs - pd.DateOffset(years=3)
    endDateTs = simDateTs + pd.DateOffset(years=3)
    
    tickerProvider, priceProvider = getOhlcvProviders()
    profile = tickerProvider.getTickerProfile(ticker)
    
    if profile and profile.ipoDate is not None:
        if profile.ipoDate > startDateTs:
            startDateTs = profile.ipoDate
            
    # Fetch historical data (simDate - 3 years to simDate)
    dfHistorical = priceProvider.getPeriodDailyTickerData(ticker, startDateTs, simDateTs)
    dfHistorical = adjustPriceDataSplits(dfHistorical)
    
    # Fetch future real data if simDate < todayTs
    dfFuture = pd.DataFrame()
    if simDateTs < todayTs:
        futureEndTs = min(endDateTs, todayTs)
        dfFuture = priceProvider.getPeriodDailyTickerData(ticker, simDateTs, futureEndTs)
        dfFuture = adjustPriceDataSplits(dfFuture)
        
    fig, ax = plt.subplots(figsize=(5.6, 2.7), dpi=180)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    lastHistoricalClose = None
    allPrices = []
    
    # --- Historical data with 3-day moving average ---
    if not dfHistorical.empty and "close" in dfHistorical.columns:
        dfCleanHist = dfHistorical.dropna(subset=["close"])
        if not dfCleanHist.empty:
            dates = [pd.Timestamp(r.get("dateNy", r.get("date"))).to_pydatetime() for _, r in dfCleanHist.iterrows()]
            closesRaw = dfCleanHist["close"].values
            # Apply 3-day centered moving average to smooth spiky raw data
            closesSmooth = getRollingMean(closesRaw, window=3)
            lastHistoricalClose = closesSmooth[-1]
            allPrices.extend(closesSmooth.tolist())
            ax.plot(dates, closesSmooth, color='#2563eb', linewidth=1.8, label='Historical (-3Y)')
            ax.fill_between(dates, closesSmooth, 0, color='#2563eb', alpha=0.08)
            
    # --- Future actual data with 3-day moving average ---
    if not dfFuture.empty and "close" in dfFuture.columns:
        dfCleanFuture = dfFuture.dropna(subset=["close"])
        if not dfCleanFuture.empty:
            dates = [pd.Timestamp(r.get("dateNy", r.get("date"))).to_pydatetime() for _, r in dfCleanFuture.iterrows()]
            closesRaw = dfCleanFuture["close"].values
            # Apply 3-day centered moving average to smooth spiky raw data
            closesSmooth = getRollingMean(closesRaw, window=3)
            allPrices.extend(closesSmooth.tolist())
            ax.plot(dates, closesSmooth, color='#f43f5e', linewidth=1.8, label='Actual Performance')
            ax.fill_between(dates, closesSmooth, 0, color='#f43f5e', alpha=0.08)
            
    # --- Build target points list ---
    if targets is None:
        targets = []
    
    # Compute target dates and prices as numpy-compatible values (days from startDateTs)
    startTime = startDateTs.timestamp()
    simDateDaysFromStart = (simDateTs - startDateTs).total_seconds() / 86400.0
    targetX = [simDateDaysFromStart]  # first point is at simDate, not at graph edge
    targetY = [lastHistoricalClose if lastHistoricalClose is not None else (allPrices[0] if allPrices else 100.0)]
    
    for monthsOffset, price in targets:
        targetDateTs = simDateTs + pd.DateOffset(months=int(monthsOffset))
        daysFromStart = (targetDateTs - startDateTs).total_seconds() / 86400.0
        targetX.append(daysFromStart)
        targetY.append(price)
        allPrices.append(price)
    
    maxPrice = max(allPrices) if allPrices else 100.0
    
    # --- Plot bezier-curved target projection ---
    if len(targets) > 0:
        # Compute smooth Catmull-Rom spline through target points
        curveX, curveY = computeBezierThroughPoints(targetX, targetY, numSamples=400)
        
        # Convert days back to datetime for plotting
        curveDates = [pd.Timestamp(startTime + x * 86400.0, unit='s', tz=NEW_YORK).to_pydatetime() for x in curveX]
        
        ax.plot(curveDates, curveY, color='#06b6d4', linewidth=2.8, solid_capstyle='round', zorder=5)
        
        # Plot larger markers at each actual target point
        for i, (monthsOffset, price) in enumerate(targets):
            targetDateTs = simDateTs + pd.DateOffset(months=int(monthsOffset))
            ax.plot(targetDateTs.to_pydatetime(), price, marker='o',
                    markersize=8, color='#06b6d4', zorder=6,
                    markeredgecolor='#ffffff', markeredgewidth=1.5)
        
        # Also plot the start connection point (simDate) with a small marker
        ax.plot(simDateTs.to_pydatetime(), targetY[0], marker='o',
                markersize=5, color='#64748b', zorder=6,
                markeredgecolor='#ffffff', markeredgewidth=1)
        
    # Vertical dashed line at simDate
    ax.axvline(x=simDateTs.to_pydatetime(), color='#1e293b', linestyle='--', linewidth=1.5, zorder=4)
    
    # Configure X axis timeline — add right-side padding so the last marker isn't clipped
    x_end = endDateTs.to_pydatetime()
    if len(targets) > 0:
        # Add ~2% extra padding on the right for marker visibility
        x_range_seconds = (endDateTs - startDateTs).total_seconds()
        x_end = endDateTs + pd.Timedelta(seconds=x_range_seconds * 0.02)
    
    ax.set_xlim(startDateTs.to_pydatetime(), x_end)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Set Y axis minimum EXACTLY at 0 with 0 gap!
    ax.set_ylim(bottom=0, top=maxPrice * 1.08)
    
    # Grid and styling
    ax.grid(True, linestyle=':', alpha=0.4, color='#cbd5e1')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#334155')
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Typography & padding
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
