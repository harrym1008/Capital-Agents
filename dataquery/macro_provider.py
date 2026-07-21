import os
import pandas as pd
from enum import Enum

from collectors.constants import MACRO_DIRECTORY, UTC
from dataquery.lru_cache import LRUCache


class MacroSeries(Enum):
    # yfinance OHLCV series
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

    # FRED macro series
    CPI = ("CPI", "fred", "value")
    CORECPI = ("CORECPI", "fred", "value")
    UNEMPLOYMENT = ("UNEMPLOYMENT", "fred", "value")
    FEDFUNDS = ("FEDFUNDS", "fred", "value")
    GDP = ("GDP", "fred", "value")
    TREAS_10Y = ("TREAS_10Y", "fred", "value")
    TREAS_2Y = ("TREAS_2Y", "fred", "value")
    TREAS_3MO = ("TREAS_3MO", "fred", "value")

    def __init__(self, parquetName: str, source: str, valueCol: str):
        self.parquetName = parquetName
        self.source = source
        self.valueCol = valueCol


class MacroDataProvider:
    def __init__(self, cache: LRUCache):
        self.cache = cache
        self.macroDir = MACRO_DIRECTORY
        self.availableSeries = self.buildMacroIndex()


    def normaliseTimestamp(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)


    def buildMacroIndex(self):
        available = {}
        if not os.path.isdir(self.macroDir):
            return available

        for series in MacroSeries:
            path = os.path.join(self.macroDir, f"{series.parquetName}.parquet")
            if os.path.exists(path):
                available[series] = path

        return available


    def loadSeries(self, name: MacroSeries) -> pd.DataFrame:
        key = f"macro|full_{name.parquetName}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        path = self.availableSeries.get(name)
        if path is None:
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path, engine="pyarrow")
        except Exception:
            return pd.DataFrame()

        if "date" not in df.columns:
            return pd.DataFrame()

        # Normalise the date column to UTC-naive for consistent comparisons
        dateCol = df["date"]
        if dateCol.dt.tz is None:
            dateCol = dateCol.dt.tz_localize(UTC)
        else:
            dateCol = dateCol.dt.tz_convert(UTC)
        df["date"] = dateCol.dt.tz_localize(None)

        df = df.sort_values("date").reset_index(drop=True)

        self.cache.put(key, df)
        return df


    def getAllSeries(self) -> list:
        return list(self.availableSeries.keys())


    def getSeries(self, 
                  name: MacroSeries, 
                  startDate: pd.Timestamp = None,
                  endDate: pd.Timestamp = None) -> pd.DataFrame:
        df = self.loadSeries(name)
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
        df = self.loadSeries(name)
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
            df = self.loadSeries(name)
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
            return None

        valueCol = "low" if name.source == "yfinance" else name.valueCol

        if valueCol not in df.columns:
            return None
        
        idx = df[valueCol].idxmin()
        row = df.loc[idx]

        return row["date"], row[valueCol]
    
    def getHighest(self, name: MacroSeries, startDate: pd.Timestamp, endDate: pd.Timestamp):
        df = self.getSeries(name, startDate=startDate, endDate=endDate)
        if df.empty:
            return None

        valueCol = "high" if name.source == "yfinance" else name.valueCol

        if valueCol not in df.columns:
            return None
        
        idx = df[valueCol].idxmax()
        row = df.loc[idx]

        return row["date"], row[valueCol]
