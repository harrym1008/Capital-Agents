import io
import base64
import urllib.request
import urllib.error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from flask import request, jsonify

from collectors.constants import NEW_YORK
from dataquery import LRUCache, DailyPriceProvider, TickerDataProvider
from llm.llamacpp.llamacpp_args import LlamaCppModel, LLAMACPP_PORT

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

def generateOhlcvChartImage(ticker, simDateTs, target12=None, target36=None):
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
    
    if not dfHistorical.empty and "close" in dfHistorical.columns:
        dfCleanHist = dfHistorical.dropna(subset=["close"])
        if not dfCleanHist.empty:
            allPrices.extend(dfCleanHist["close"].tolist())
            
    if not dfFuture.empty and "close" in dfFuture.columns:
        dfCleanFuture = dfFuture.dropna(subset=["close"])
        if not dfCleanFuture.empty:
            allPrices.extend(dfCleanFuture["close"].tolist())
            
    if target12 is not None:
        allPrices.append(target12)
    if target36 is not None:
        allPrices.append(target36)
        
    maxPrice = max(allPrices) if allPrices else 100.0
    
    # Plot historical data (Past 3Y) in Royal Indigo
    if not dfHistorical.empty and "close" in dfHistorical.columns:
        dfCleanHist = dfHistorical.dropna(subset=["close"])
        if not dfCleanHist.empty:
            dates = [pd.Timestamp(r.get("dateNy", r.get("date"))).to_pydatetime() for _, r in dfCleanHist.iterrows()]
            closes = dfCleanHist["close"].values
            lastHistoricalClose = closes[-1]
            ax.plot(dates, closes, color='#2563eb', linewidth=1.8, label='Historical (-3Y)')
            ax.fill_between(dates, closes, 0, color='#2563eb', alpha=0.08)
            
    # Plot actual future data (Post simDate) in Warm Rose
    if not dfFuture.empty and "close" in dfFuture.columns:
        dfCleanFuture = dfFuture.dropna(subset=["close"])
        if not dfCleanFuture.empty:
            dates = [pd.Timestamp(r.get("dateNy", r.get("date"))).to_pydatetime() for _, r in dfCleanFuture.iterrows()]
            closes = dfCleanFuture["close"].values
            ax.plot(dates, closes, color='#f43f5e', linewidth=1.8, label='Actual Performance')
            ax.fill_between(dates, closes, 0, color='#f43f5e', alpha=0.08)
            
    # Plot Target Projection Line in Electric Cyan (clean rounded cap, no arrow artifact)
    if target36 is not None or target12 is not None:
        startPrice = lastHistoricalClose if lastHistoricalClose is not None else (allPrices[0] if allPrices else 100.0)
        targetDates = [simDateTs.to_pydatetime()]
        targetPrices = [startPrice]
        
        if target12 is not None:
            t12Date = (simDateTs + pd.DateOffset(years=1)).to_pydatetime()
            targetDates.append(t12Date)
            targetPrices.append(target12)
            
        if target36 is not None:
            t36Date = (simDateTs + pd.DateOffset(years=3)).to_pydatetime()
            targetDates.append(t36Date)
            targetPrices.append(target36)
            
        ax.plot(targetDates, targetPrices, color='#06b6d4', linewidth=2.8, linestyle='-', solid_capstyle='round', zorder=5)
        ax.plot(targetDates[-1], targetPrices[-1], marker='o', markersize=4.5, color='#06b6d4', zorder=6)
        
    # Vertical dashed line at simDate
    ax.axvline(x=simDateTs.to_pydatetime(), color='#1e293b', linestyle='--', linewidth=1.5, zorder=4)
    
    # Configure X axis timeline
    ax.set_xlim(startDateTs.to_pydatetime(), endDateTs.to_pydatetime())
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
        target12Param = request.args.get("target12", None)
        target36Param = request.args.get("target36", None)
        
        target12Val = None
        if target12Param and target12Param != "--":
            try:
                target12Val = float(target12Param.replace("$", "").strip())
            except Exception:
                target12Val = None

        target36Val = None
        if target36Param and target36Param != "--":
            try:
                target36Val = float(target36Param.replace("$", "").strip())
            except Exception:
                target36Val = None

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
        
        chartImgStr = generateOhlcvChartImage(tickerParam, simDateTs, target12Val, target36Val)
        
        return jsonify({
            "ticker": tickerParam,
            "simDate": simDateStr,
            "chartImage": chartImgStr
        })
