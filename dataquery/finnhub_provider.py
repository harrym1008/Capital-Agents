import os
import json
import requests
import pandas as pd
from typing import Optional, Dict, Any

from collectors.rate_limiter import RateLimiter
from dataquery.lru_cache import LRUCache
from dataquery.keyed_lock import KeyedLockManager


class FinnhubDataProvider:
    def __init__(self, cache: LRUCache, rateLimiter: RateLimiter):
        self.cache = cache
        self.rateLimiter = rateLimiter
        self.keyedLocks = KeyedLockManager()


    def fetchRawMetrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        cleanTicker = ticker.strip().upper()
        now = pd.Timestamp.now(tz="UTC")
        cacheKey = f"finnhub|rawMetrics_{cleanTicker}_{now.strftime('%Y-%m-%dH%H')}"

        # 1. Check in-memory LRU cache
        cachedMem = self.cache.get(cacheKey)
        if cachedMem is not None:
            return cachedMem

        with self.keyedLocks.lockKey(cleanTicker):
            # Double check in-memory cache inside lock
            cachedMem = self.cache.get(cacheKey)
            if cachedMem is not None:
                return cachedMem

            # 2. Download from Finnhub API
            apiKey = os.getenv("FINNHUB_API_KEY")
            if not apiKey:
                print("[FinnhubDataProvider] Warning: FINNHUB_API_KEY not found in environment.")
                return None

            url = f"https://finnhub.io/api/v1/stock/metric?symbol={cleanTicker}&metric=all&token={apiKey}"
            
            try:
                self.rateLimiter.wait()
                response = requests.get(url, timeout=15)
                
                if response.status_code == 429:
                    self.rateLimiter.got429(1)
                    response = requests.get(url, timeout=15)

                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, dict) and data.get("metric"):
                        self.cache.put(cacheKey, data)
                        return data
                else:
                    print(f"[FinnhubDataProvider] API returned status {response.status_code} for {cleanTicker}.")
            except Exception as e:
                print(f"[FinnhubDataProvider] Error downloading metrics for {cleanTicker}: {e}")

            return None


    def getPointInTimeMetrics(self, ticker: str, asOfDate: pd.Timestamp) -> Dict[str, Any]:
        raw = self.fetchRawMetrics(ticker)
        if not raw:
            return {}

        asOfNorm = asOfDate.tz_localize(None) if asOfDate.tzinfo is not None else asOfDate
        series = raw.get("series", {})
        quarterly = series.get("quarterly", {})
        annual = series.get("annual", {})
        liveMetrics = raw.get("metric", {})

        def extractLatest(metricKey: str, preferQuarterly: bool = True) -> Optional[float]:
            primary = quarterly if preferQuarterly else annual
            secondary = annual if preferQuarterly else quarterly
            seriesList = primary.get(metricKey) or secondary.get(metricKey)
            if not seriesList:
                return None

            validItems = []
            for item in seriesList:
                pStr = item.get("period")
                if not pStr:
                    continue
                try:
                    pDate = pd.to_datetime(pStr)
                    if pDate.tzinfo is not None:
                        pDate = pDate.tz_localize(None)
                    if pDate <= asOfNorm:
                        validItems.append((pDate, item.get("v")))
                except Exception:
                    continue

            if not validItems:
                return None

            validItems.sort(key=lambda x: x[0])
            return validItems[-1][1]

        # Extract point-in-time metrics
        peVal = extractLatest("peTTM") or extractLatest("pe", preferQuarterly=False)
        pbVal = extractLatest("pb") or extractLatest("pbQuarterly")
        psVal = extractLatest("psTTM") or extractLatest("ps", preferQuarterly=False)
        grossMarginVal = extractLatest("grossMargin")
        netMarginVal = extractLatest("netMargin")
        operatingMarginVal = extractLatest("operatingMargin")
        roeVal = extractLatest("roeTTM") or extractLatest("roe", preferQuarterly=False)
        roaVal = extractLatest("roaTTM") or extractLatest("roa", preferQuarterly=False)
        currentRatioVal = extractLatest("currentRatio")
        quickRatioVal = extractLatest("quickRatio")
        debtToEquityVal = extractLatest("totalDebtToEquity")
        epsVal = extractLatest("eps") or extractLatest("epsTTM")
        ebitdaVal = extractLatest("ebitda")

        # Fallback to static snapshot metrics if backtest series is empty or before start of series
        isLiveFallback = False
        if peVal is None and liveMetrics.get("peTTM") is not None:
            peVal = liveMetrics.get("peTTM")
            isLiveFallback = True
        if pbVal is None and liveMetrics.get("pbQuarterly") is not None:
            pbVal = liveMetrics.get("pbQuarterly")
        if grossMarginVal is None and liveMetrics.get("grossMarginTTM") is not None:
            grossMarginVal = liveMetrics.get("grossMarginTTM") / 100.0 if liveMetrics.get("grossMarginTTM") > 1.0 else liveMetrics.get("grossMarginTTM")
        if netMarginVal is None and liveMetrics.get("netProfitMarginTTM") is not None:
            netMarginVal = liveMetrics.get("netProfitMarginTTM") / 100.0 if liveMetrics.get("netProfitMarginTTM") > 1.0 else liveMetrics.get("netProfitMarginTTM")
        if roeVal is None and liveMetrics.get("roeTTM") is not None:
            roeVal = liveMetrics.get("roeTTM") / 100.0 if liveMetrics.get("roeTTM") > 1.0 else liveMetrics.get("roeTTM")
        if betaVal := liveMetrics.get("beta"):
            pass
        else:
            betaVal = None

        divYieldVal = liveMetrics.get("currentDividendYieldTTM")
        if divYieldVal is not None and divYieldVal > 0:
            divYieldVal = divYieldVal / 100.0 if divYieldVal > 0.5 else divYieldVal

        return {
            "ticker": ticker.upper(),
            "asOfDate": asOfNorm.strftime("%Y-%m-%d"),
            "peTTM": peVal,
            "pb": pbVal,
            "psTTM": psVal,
            "grossMargin": grossMarginVal,
            "operatingMargin": operatingMarginVal,
            "netMargin": netMarginVal,
            "roeTTM": roeVal,
            "roaTTM": roaVal,
            "currentRatio": currentRatioVal,
            "quickRatio": quickRatioVal,
            "debtToEquity": debtToEquityVal,
            "eps": epsVal,
            "ebitda": ebitdaVal,
            "beta": betaVal,
            "dividendYield": divYieldVal,
            "isLiveFallback": isLiveFallback
        }
