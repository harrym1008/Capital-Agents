import os
import json
import requests
import pandas as pd
from typing import Optional, Dict, Any

from collectors.rate_limiter import RateLimiter
from dataquery.lru_cache import LRUCache
from dataquery.keyed_lock import KeyedLockManager


# Provider for company fundamental metrics and point-in-time financial ratios from Finnhub
class FinnhubDataProvider:
    def __init__(self, cache: LRUCache, rateLimiter: RateLimiter):
        self.cache = cache
        self.rateLimiter = rateLimiter
        self.keyedLocks = KeyedLockManager()


    def fetchRawLookAheadMetrics(self, ticker: str) -> Optional[Dict[str, Any]]:
        # Fetch comprehensive raw metric series from Finnhub API or cache
        cleanTicker = ticker.strip().upper()
        now = pd.Timestamp.now(tz="UTC")
        cacheKey = f"finnhub|rawMetrics_{cleanTicker}_{now.strftime('%Y-%m-%d')}"   # Invalidate cache daily to prevent stale metrics

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
        # Extract point-in-time financial metrics strictly prior to asOfDate to prevent lookahead
        raw = self.fetchRawLookAheadMetrics(ticker)
        if not raw:
            return {}

        asOfNorm = asOfDate.tz_localize(None) if asOfDate.tzinfo is not None else asOfDate
        series = raw.get("series", {})
        quarterly = series.get("quarterly", {})
        annual = series.get("annual", {})

        def getValidItems(metricKey: str, preferQuarterly: bool = True):
            # Retrieve all valid (date, value) pairs strictly before asOfDate
            primary = quarterly if preferQuarterly else annual
            secondary = annual if preferQuarterly else quarterly
            seriesList = primary.get(metricKey) or secondary.get(metricKey)
            if not seriesList:
                return []

            validItems = []
            for item in seriesList:
                pStr = item.get("period")
                if not pStr:
                    continue
                try:
                    pDate = pd.to_datetime(pStr)
                    if pDate.tzinfo is not None:
                        pDate = pDate.tz_localize(None)
                    if pDate < asOfNorm:     # MUST be strictly prior to asOfDate to avoid lookahead bias
                        validItems.append((pDate, item.get("v")))
                except Exception:
                    continue

            validItems.sort(key=lambda x: x[0])
            return validItems

        def extractLatest(metricKey: str, preferQuarterly: bool = True) -> Optional[float]:
            # Extract the most recent metric value strictly prior to cut-off
            items = getValidItems(metricKey, preferQuarterly)
            if not items:
                return None
            return items[-1][1]

        def extractGrowthQoQ(metricKey: str, preferQuarterly: bool = True) -> Optional[float]:
            # Compute quarter-over-quarter percentage growth from the two latest readings
            items = getValidItems(metricKey, preferQuarterly)
            if len(items) < 2:
                return None
            latest = items[-1][1]
            prior = items[-2][1]
            if prior is None or latest is None or prior == 0:
                return None
            return ((latest - prior) / abs(prior)) * 100

        def extractGrowthYoY(metricKey: str, preferQuarterly: bool = True) -> Optional[float]:
            # Compute year-over-year percentage growth against entry ~4 quarters back
            items = getValidItems(metricKey, preferQuarterly)
            if len(items) < 2:
                return None
            latestDate = items[-1][0]
            latestVal = items[-1][1]

            # Locate observation closest to 1 year prior (at least 10 months back)
            targetDate = latestDate - pd.DateOffset(months=10)
            priorVal = None
            for date, val in reversed(items[:-1]):
                if date <= targetDate:
                    priorVal = val
                    break

            if priorVal is None or latestVal is None or priorVal == 0:
                return None
            return ((latestVal - priorVal) / abs(priorVal)) * 100

        # Extract point-in-time metrics from series only
        # Price to earnings, price to book, price to sales, EV/EBITDA
        peVal = extractLatest("peTTM") or extractLatest("pe", preferQuarterly=False)
        pbVal = extractLatest("pb") or extractLatest("pbQuarterly")
        psTtmVal = extractLatest("psTTM") or extractLatest("ps", preferQuarterly=False)
        evEbitdaTtmVal = extractLatest("evEbitdaTTM") or extractLatest("evEbitda", preferQuarterly=False)

        # Profitability metrics: gross margin, operating margin, net margin, free cash flow per share, free cash flow margin
        grossMarginVal = extractLatest("grossMargin")
        operatingMarginVal = extractLatest("operatingMargin")
        netMarginVal = extractLatest("netMargin")
        fcfPerShareTtmVal = extractLatest("fcfPerShareTTM")
        fcfMarginVal = extractLatest("fcfMargin")

        # Return on equity, return on assets, return on invested capital
        roeVal = extractLatest("roeTTM") or extractLatest("roe", preferQuarterly=False)
        roaVal = extractLatest("roaTTM") or extractLatest("roa", preferQuarterly=False)
        roicTtmVal = extractLatest("roicTTM") or extractLatest("roic", preferQuarterly=False)

        # Leverage and liquidity ratios: current ratio, quick ratio, debt to equity
        currentRatioVal = extractLatest("currentRatio")
        quickRatioVal = extractLatest("quickRatio")
        debtToEquityVal = extractLatest("totalDebtToEquity")

        # Per-share metrics: EPS, EBITDA, payout ratio
        epsVal = extractLatest("eps") or extractLatest("epsTTM")
        ebitdaVal = extractLatest("ebitda")
        payoutRatioTtmVal = extractLatest("payoutRatioTTM")

        # EPS and revenue growth metrics: quarter-over-quarter and year-over-year
        epsGrowthQoQ = extractGrowthQoQ("eps")
        epsGrowthYoY = extractGrowthYoY("eps")
        revenueGrowthQoQ = extractGrowthQoQ("salesPerShare")
        revenueGrowthYoY = extractGrowthYoY("salesPerShare")

        return {
            "ticker": ticker.upper(),
            "asOfDate": asOfNorm.strftime("%Y-%m-%d"),
            
            "peTTM": peVal,
            "pb": pbVal,
            "psTTM": psTtmVal,
            "evEbitdaTTM": evEbitdaTtmVal,
            
            "grossMargin": grossMarginVal,
            "operatingMargin": operatingMarginVal,
            "netMargin": netMarginVal,
            "fcfPerShareTTM": fcfPerShareTtmVal,
            "fcfMargin": fcfMarginVal,
            
            "roeTTM": roeVal,
            "roaTTM": roaVal,
            "roicTTM": roicTtmVal,
            
            "debtToEquity": debtToEquityVal,
            "currentRatio": currentRatioVal,
            "quickRatio": quickRatioVal,
            
            "eps": epsVal,
            "ebitda": ebitdaVal,
            "payoutRatioTTM": payoutRatioTtmVal,
            
            "epsGrowthQoQ": epsGrowthQoQ,
            "epsGrowthYoY": epsGrowthYoY,
            "revenueGrowthQoQ": revenueGrowthQoQ,
            "revenueGrowthYoY": revenueGrowthYoY,
        }
