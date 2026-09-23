import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from enum import Enum
from dotenv import load_dotenv
import yfinance as yf
from fredapi import Fred

from collectors.constants import MACRO_DIRECTORY, UTC, END_DATE
from collectors.macro_dl_client import YFINANCE_MACRO_TICKERS
from collectors.rate_limiter import GlobalRateLimiters
from dataquery.lru_cache import LRUCache
from dataquery.keyed_lock import KeyedLockManager

load_dotenv()

# Mapping of macro indicators to FRED series IDs
FRED_SERIES_MAP = {
    "CPI": "CPIAUCSL",
    "CORECPI": "CPILFESL",
    "UNEMPLOYMENT": "UNRATE",
    "FEDFUNDS": "FEDFUNDS",
    "GDP": "GDP",
    "TREAS_30Y": "DGS30",
    "TREAS_10Y": "DGS10",
    "TREAS_2Y": "DGS2",
    "TREAS_3MO": "DGS3MO",
}


# Supported macroeconomic indicators and market benchmarks
class MacroSeries(Enum):
    SP500 = ("SP500", "yfinance", "close")
    NDQ100 = ("NDQ100", "yfinance", "close")
    DJIA = ("DJIA", "yfinance", "close")
    RUS2000 = ("RUS2000", "yfinance", "close")
    VIX = ("VIX", "yfinance", "close")
    OIL_WTI = ("OIL_WTI", "yfinance", "close")
    OIL_BRENT = ("OIL_BRENT", "yfinance", "close")
    GOLD = ("GOLD", "yfinance", "close")
    SILVER = ("SILVER", "yfinance", "close")
    NATGAS = ("NATGAS", "yfinance", "close")
    BTCUSD = ("BTCUSD", "yfinance", "close")
    GBPUSD = ("GBPUSD", "yfinance", "close")
    EURUSD = ("EURUSD", "yfinance", "close")
    USDJPY = ("USDJPY", "yfinance", "close")

    CPI = ("CPI", "fred", "value")
    CORECPI = ("CORECPI", "fred", "value")
    UNEMPLOYMENT = ("UNEMPLOYMENT", "fred", "value")
    FEDFUNDS = ("FEDFUNDS", "fred", "value")
    GDP = ("GDP", "fred", "value")
    TREAS_30Y = ("TREAS_30Y", "fred", "value")
    TREAS_10Y = ("TREAS_10Y", "fred", "value")
    TREAS_2Y = ("TREAS_2Y", "fred", "value")
    TREAS_3MO = ("TREAS_3MO", "fred", "value")

    def __init__(self, parquetName: str, source: str, valueCol: str):
        self.parquetName = parquetName
        self.source = source
        self.valueCol = valueCol


# Provider for querying macroeconomic metrics and index benchmark time-series
class MacroDataProvider:
    def __init__(self, cache: LRUCache, rateLimiters: GlobalRateLimiters):
        self.cache = cache
        self.rateLimiters = rateLimiters
        self.macroDir = MACRO_DIRECTORY
        self.availableSeries = self.buildMacroIndex()
        self.keyedLocks = KeyedLockManager()
        self.downloadLock = threading.RLock()
        self.fredClient = Fred(api_key=os.getenv("FRED_API_KEY")) if os.getenv("FRED_API_KEY") else None
        self.lastAttemptTime = {}


    def normaliseTimestamp(self, ts: pd.Timestamp) -> pd.Timestamp:
        # Convert timestamp to UTC naive representation
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)

    def buildMacroIndex(self):
        # Index local parquet file paths for macro series
        available = {}
        for series in MacroSeries:
            path = os.path.join(self.macroDir, f"{series.parquetName}.parquet")
            if os.path.exists(path):
                available[series] = path
            else:
                available[series] = None
        return available

    def downloadBatchYfinance(self, seriesList: list, startDate: pd.Timestamp, endDate: pd.Timestamp, onProgressCallback=None) -> dict:
        # Batch download market index time-series via Yahoo Finance
        results = {}
        if not seriesList:
            return results

        self.rateLimiters.yFinanceLimiter.wait()

        tickerToSeries = {}
        yfTickers = []
        for series in seriesList:
            yfMeta = YFINANCE_MACRO_TICKERS.get(series.parquetName, {})
            yfTicker = yfMeta.get("yfticker")
            if yfTicker:
                yfTickers.append(yfTicker)
                tickerToSeries[yfTicker] = series

        if not yfTickers:
            return results

        try:
            # One bulk download for all tickers, to reduce API calls
            rawDf = yf.download(
                tickers=yfTickers,
                start=startDate.strftime("%Y-%m-%d"),
                end=(endDate + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True
            )
            if onProgressCallback:
                onProgressCallback(len(yfTickers))

            if rawDf.empty:
                return results

            for yfTicker, series in tickerToSeries.items():
                try:
                    if len(yfTickers) == 1:
                        seriesDf = rawDf.copy()
                    else:
                        if isinstance(rawDf.columns, pd.MultiIndex):
                            # Extract the relevant series data from multi-index columns
                            seriesDf = rawDf.xs(yfTicker, axis=1, level=1).copy()
                        else:
                            continue

                    seriesDf = seriesDf.reset_index()
                    seriesDf.columns = seriesDf.columns.str.lower()

                    if "date" not in seriesDf.columns:
                        continue

                    dateCol = seriesDf["date"]
                    if dateCol.dt.tz is None:
                        dateCol = dateCol.dt.tz_localize(UTC)
                    else:
                        dateCol = dateCol.dt.tz_convert(UTC)
                    seriesDf["date"] = dateCol.dt.tz_localize(None)

                    cols = ["date", "open", "high", "low", "close"]
                    validCols = [c for c in cols if c in seriesDf.columns]
                    results[series] = seriesDf[validCols].dropna(subset=["date", "close"])
                except Exception:
                    continue
        except Exception:
            pass

        return results

    def downloadBatchFred(self, seriesList: list, startDate: pd.Timestamp, endDate: pd.Timestamp, onProgressCallback=None) -> dict:
        # Download multiple FRED series concurrently using thread pool
        results = {}
        if not seriesList or not self.fredClient:
            return results

        # Helper function to fetch a single FRED series 
        def fetchSingleFred(series: MacroSeries):
            fredId = FRED_SERIES_MAP.get(series.parquetName)
            if not fredId:
                return series, pd.DataFrame()

            self.rateLimiters.fredLimiter.wait()
            try:
                rawSeries = self.fredClient.get_series(
                    fredId,
                    observation_start=startDate.strftime("%Y-%m-%d"),
                    observation_end=endDate.strftime("%Y-%m-%d")
                ).dropna()

                if rawSeries.empty:
                    return series, pd.DataFrame()

                dfFred = rawSeries.reset_index()
                dfFred.columns = ["date", "value"]
                dateCol = pd.to_datetime(dfFred["date"])
                if dateCol.dt.tz is None:
                    dateCol = dateCol.dt.tz_localize(UTC)
                else:
                    dateCol = dateCol.dt.tz_convert(UTC)
                dfFred["date"] = dateCol.dt.tz_localize(None)
                return series, dfFred
            except Exception:
                return series, pd.DataFrame()

        # Use thread pool to fetch multiple FRED series concurrently
        with ThreadPoolExecutor(max_workers=min(len(seriesList), 8)) as executor:
            futureToSeries = {executor.submit(fetchSingleFred, s): s for s in seriesList}
            for future in as_completed(futureToSeries):
                try:
                    s, df = future.result()
                    if onProgressCallback:
                        onProgressCallback(1)
                    if not df.empty:
                        results[s] = df
                except Exception:
                    pass

        return results

    def downloadNonLocalMacro(self, name: MacroSeries, startDate: pd.Timestamp, endDate: pd.Timestamp) -> pd.DataFrame:
        # Route download requests to appropriate data provider
        if name.source == "yfinance":
            res = self.downloadBatchYfinance([name], startDate, endDate)
            return res.get(name, pd.DataFrame())
        elif name.source == "fred":
            res = self.downloadBatchFred([name], startDate, endDate)
            return res.get(name, pd.DataFrame())
        return pd.DataFrame()


    def loadSeries(self, name: MacroSeries) -> pd.DataFrame:
        # Load time-series dataframe for requested macro indicator from cache or disk
        key = f"macro|full_{name.parquetName}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        with self.keyedLocks.lockKey(key):
            cached = self.cache.get(key)
            if cached is not None:
                return cached

            path = self.availableSeries.get(name)
            df = pd.DataFrame()
            if path and os.path.exists(path):
                try:
                    df = pd.read_parquet(path, engine="pyarrow")
                    if "date" in df.columns:
                        dateCol = df["date"]
                        if dateCol.dt.tz is None:
                            dateCol = dateCol.dt.tz_localize(UTC)
                        else:
                            dateCol = dateCol.dt.tz_convert(UTC)
                        df["date"] = dateCol.dt.tz_localize(None)
                        df = df.sort_values("date").reset_index(drop=True)
                except Exception:
                    df = pd.DataFrame()

            self.cache.put(key, df)
            return df

    def ensureBulkCoverage(self, names: list, targetDate: pd.Timestamp, onProgressCallback=None):
        # Verify coverage for multiple series and trigger batched downloads for stale series
        targetNorm = self.normaliseTimestamp(targetDate)
        now = time.time()
        cooldownSeconds = 3600  # 1 hour cooldown per series/target date

        missingYf = []
        missingFred = []
        minMaxDates = {}

        for name in names:
            df = self.loadSeries(name)
            if df.empty:
                maxDate = pd.Timestamp(END_DATE).tz_localize(None) - pd.DateOffset(years=5)
            else:
                maxDate = df["date"].max()

            # Calculate frequency-aware staleness threshold in days
            daysDiff = (targetNorm - maxDate).days

            if name.parquetName == "GDP":
                staleThreshold = 90  # Quarterly series
            elif name.parquetName in ["CPI", "CORECPI", "UNEMPLOYMENT", "FEDFUNDS"]:
                staleThreshold = 32  # Monthly series
            elif targetNorm.weekday() in [5, 6]:
                staleThreshold = 3   # Weekend gap for daily series
            else:
                staleThreshold = 1   # Daily series

            if daysDiff > staleThreshold:
                checkKey = (name.parquetName, targetNorm.strftime("%Y-%m-%d"))
                lastAttempt = self.lastAttemptTime.get(checkKey, 0)
                if (now - lastAttempt) > cooldownSeconds:
                    minMaxDates[name] = maxDate
                    self.lastAttemptTime[checkKey] = now
                    if name.source == "yfinance":
                        missingYf.append(name)
                    elif name.source == "fred":
                        missingFred.append(name)

        if not missingYf and not missingFred:
            return

        with self.downloadLock:
            if missingYf:
                earliestStart = min([minMaxDates[s] for s in missingYf])
                yfResults = self.downloadBatchYfinance(missingYf, earliestStart, targetNorm, onProgressCallback=onProgressCallback)
                for s, dfInc in yfResults.items():
                    if not dfInc.empty:
                        fullDf = self.loadSeries(s)
                        origLen = len(fullDf)
                        fullDf = pd.concat([fullDf, dfInc], ignore_index=True)
                        fullDf = fullDf.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                        key = f"macro|full_{s.parquetName}"
                        self.cache.put(key, fullDf)
                        if len(fullDf) > origLen:
                            path = os.path.join(self.macroDir, f"{s.parquetName}.parquet")
                            try:
                                fullDf.to_parquet(path, index=False)
                                self.availableSeries[s] = path
                            except Exception:
                                pass

            if missingFred:
                earliestStart = min([minMaxDates[s] for s in missingFred])
                fredResults = self.downloadBatchFred(missingFred, earliestStart, targetNorm, onProgressCallback=onProgressCallback)
                for s, dfInc in fredResults.items():
                    if not dfInc.empty:
                        fullDf = self.loadSeries(s)
                        origLen = len(fullDf)
                        fullDf = pd.concat([fullDf, dfInc], ignore_index=True)
                        fullDf = fullDf.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                        key = f"macro|full_{s.parquetName}"
                        self.cache.put(key, fullDf)
                        if len(fullDf) > origLen:
                            path = os.path.join(self.macroDir, f"{s.parquetName}.parquet")
                            try:
                                fullDf.to_parquet(path, index=False)
                                self.availableSeries[s] = path
                            except Exception:
                                pass


    def ensureCoverage(self, name: MacroSeries, targetDate: pd.Timestamp) -> pd.DataFrame:
        # Ensure single series coverage up to target date
        self.ensureBulkCoverage([name], targetDate)
        return self.loadSeries(name)

    def getAllSeries(self) -> list:
        # Return list of all available MacroSeries enum items
        return list(MacroSeries)

    def getSeries(self, name: MacroSeries, startDate: pd.Timestamp, endDate: pd.Timestamp) -> pd.DataFrame:
        # Retrieve date-bounded time-series for macro indicator
        df = self.ensureCoverage(name, endDate)

        if df.empty:
            return pd.DataFrame()

        if startDate is not None:
            startNorm = self.normaliseTimestamp(startDate)
            df = df[df["date"] >= startNorm]
        if endDate is not None:
            endNorm = self.normaliseTimestamp(endDate)
            df = df[df["date"] <= endNorm]

        return df.reset_index(drop=True)

    def getLatestValue(self, name: MacroSeries, before: pd.Timestamp):
        # Fetch single latest reading strictly on or prior to cut-off date
        df = self.ensureCoverage(name, before)
        if df.empty:
            return None

        beforeNorm = self.normaliseTimestamp(before)
        prior = df[df["date"] <= beforeNorm]
        if prior.empty:
            return None

        return prior.iloc[-1][name.valueCol]

    def getSnapshot(self, names: list, before: pd.Timestamp, onProgressCallback=None) -> pd.DataFrame:
        # Retrieve snapshot of latest values across multiple macro series
        self.ensureBulkCoverage(names, before, onProgressCallback=onProgressCallback)
        beforeNorm = self.normaliseTimestamp(before)

        rows = []
        for name in names:
            df = self.loadSeries(name)
            if onProgressCallback:
                onProgressCallback(1)

            if df.empty:
                continue

            prior = df[df["date"] <= beforeNorm]
            if prior.empty:
                continue

            last = prior.iloc[-1]
            rows.append({
                "series": name,
                "date": last["date"],
                "value": last[name.valueCol]
            })

        if not rows:
            return pd.DataFrame(columns=["series", "date", "value"])

        return pd.DataFrame(rows)


    def getLowest(self, name: MacroSeries, startDate: pd.Timestamp, endDate: pd.Timestamp):
        # Find minimum value and corresponding timestamp in date range
        df = self.getSeries(name, startDate=startDate, endDate=endDate)
        if df.empty:
            return None, None

        valueCol = "low" if name.source == "yfinance" else name.valueCol
        if valueCol not in df.columns:
            return None, None

        idx = df[valueCol].idxmin()
        row = df.loc[idx]
        return row["date"], row[valueCol]

    def getHighest(self, name: MacroSeries, startDate: pd.Timestamp, endDate: pd.Timestamp):
        # Find maximum value and corresponding timestamp in date range
        df = self.getSeries(name, startDate=startDate, endDate=endDate)
        if df.empty:
            return None, None

        valueCol = "high" if name.source == "yfinance" else name.valueCol
        if valueCol not in df.columns:
            return None, None

        idx = df[valueCol].idxmax()
        row = df.loc[idx]
        return row["date"], row[valueCol]
