import os
import threading
import pandas as pd
from enum import Enum
from dotenv import load_dotenv
import yfinance as yf
from fredapi import Fred

from collectors.constants import MACRO_DIRECTORY, UTC, END_DATE
from collectors.macro_dl_client import YFINANCE_MACRO_TICKERS
from collectors.rate_limiter import GlobalRateLimiters
from dataquery.lru_cache import LRUCache

load_dotenv()

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


class MacroDataProvider:
    def __init__(self, cache: LRUCache, rateLimiters: GlobalRateLimiters):
        self.cache = cache
        self.rateLimiters = rateLimiters
        self.macroDir = MACRO_DIRECTORY
        self.availableSeries = self.buildMacroIndex()
        self.lock = threading.RLock()

    def normaliseTimestamp(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)

    def buildMacroIndex(self):
        available = {}
        for series in MacroSeries:
            path = os.path.join(self.macroDir, f"{series.parquetName}.parquet")
            if os.path.exists(path):
                available[series] = path
            else:
                available[series] = None
        return available

    def downloadNonLocalMacro(self, name: MacroSeries, startDate: pd.Timestamp, endDate: pd.Timestamp) -> pd.DataFrame:
        if name.source == "yfinance":
            self.rateLimiters.yFinanceLimiter.wait()

            yfMeta = YFINANCE_MACRO_TICKERS.get(name.parquetName, {})
            yfTicker = yfMeta.get("yfticker")
            if not yfTicker:
                return pd.DataFrame()

            try:
                rawDf = yf.download(
                    tickers=yfTicker,
                    start=startDate.strftime("%Y-%m-%d"),
                    end=(endDate + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
                    interval="1d",
                    auto_adjust=True,
                    progress=False
                )
                if rawDf.empty:
                    return pd.DataFrame()

                if isinstance(rawDf.columns, pd.MultiIndex):
                    rawDf = rawDf.xs(yfTicker, axis=1, level=1)

                rawDf = rawDf.reset_index()
                rawDf.columns = rawDf.columns.str.lower()

                dateCol = rawDf["date"]
                if dateCol.dt.tz is None:
                    dateCol = dateCol.dt.tz_localize(UTC)
                else:
                    dateCol = dateCol.dt.tz_convert(UTC)
                rawDf["date"] = dateCol.dt.tz_localize(None)

                cols = ["date", "open", "high", "low", "close"]
                validCols = [c for c in cols if c in rawDf.columns]
                return rawDf[validCols]
            except Exception:
                return pd.DataFrame()

        elif name.source == "fred":
            self.rateLimiters.fredLimiter.wait()

            fredId = FRED_SERIES_MAP.get(name.parquetName)
            if not fredId:
                return pd.DataFrame()

            try:
                fredClient = Fred(api_key=os.getenv("FRED_API_KEY"))
                rawSeries = fredClient.get_series(
                    fredId,
                    observation_start=startDate.strftime("%Y-%m-%d"),
                    observation_end=endDate.strftime("%Y-%m-%d")
                ).dropna()

                if rawSeries.empty:
                    return pd.DataFrame()

                dfFred = rawSeries.reset_index()
                dfFred.columns = ["date", "value"]
                dateCol = pd.to_datetime(dfFred["date"])
                if dateCol.dt.tz is None:
                    dateCol = dateCol.dt.tz_localize(UTC)
                else:
                    dateCol = dateCol.dt.tz_convert(UTC)
                dfFred["date"] = dateCol.dt.tz_localize(None)
                return dfFred
            except Exception:
                return pd.DataFrame()

        return pd.DataFrame()

    def loadSeries(self, name: MacroSeries) -> pd.DataFrame:
        key = f"macro|full_{name.parquetName}"
        with self.lock:
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

    def ensureCoverage(self, name: MacroSeries, targetDate: pd.Timestamp) -> pd.DataFrame:
        targetNorm = self.normaliseTimestamp(targetDate)
        with self.lock:
            df = self.loadSeries(name)

            if df.empty:
                maxDate = pd.Timestamp(END_DATE).tz_localize(None) - pd.DateOffset(years=5)
            else:
                maxDate = df["date"].max()

            if targetNorm > maxDate:
                key = f"macro|full_{name.parquetName}"
                dfInc = self.downloadNonLocalMacro(name, maxDate, targetNorm)
                if not dfInc.empty:
                    df = pd.concat([df, dfInc], ignore_index=True)
                    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                    self.cache.put(key, df)

            return df

    def getAllSeries(self) -> list:
        return list(MacroSeries)

    def getSeries(self, 
                  name: MacroSeries, 
                  startDate: pd.Timestamp,
                  endDate: pd.Timestamp) -> pd.DataFrame:
        with self.lock:
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

    def getLatestValue(self, 
                       name: MacroSeries, 
                       before: pd.Timestamp):
        df = self.ensureCoverage(name, before)
        if df.empty:
            return None

        beforeNorm = self.normaliseTimestamp(before)
        prior = df[df["date"] <= beforeNorm]
        if prior.empty:
            return None

        return prior.iloc[-1][name.valueCol]

    def getSnapshot(self, names: list, before: pd.Timestamp) -> pd.DataFrame:
        beforeNorm = self.normaliseTimestamp(before)

        rows = []
        for name in names:
            df = self.ensureCoverage(name, beforeNorm)
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
        df = self.getSeries(name, startDate=startDate, endDate=endDate)
        if df.empty:
            return None, None

        valueCol = "high" if name.source == "yfinance" else name.valueCol
        if valueCol not in df.columns:
            return None, None

        idx = df[valueCol].idxmax()
        row = df.loc[idx]
        return row["date"], row[valueCol]
