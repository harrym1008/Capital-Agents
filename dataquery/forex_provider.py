import os
from typing import Optional
import pandas as pd

from collectors.constants import FOREX_DIRECTORY, UTC
from dataquery.lru_cache import LRUCache
from collectors.forex_dl_client import Currency, CURRENCY_MAP


class ForexDataProvider:
    def __init__(self, startDate: pd.Timestamp, endDate: pd.Timestamp, cache: LRUCache):
        self.startDate = self.normaliseTimestamp(startDate)
        self.endDate = self.normaliseTimestamp(endDate)
        self.cache = cache

        self.availableCurrencies = self.buildForexIndex()
        self.availableCurrencies["USD"] = None


    def normaliseTimestamp(self, ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)


    def buildForexIndex(self):
        available = {}
        if not os.path.isdir(FOREX_DIRECTORY):
            return available

        for currency in CURRENCY_MAP.values():
            parquetPath = os.path.join(FOREX_DIRECTORY, f"{currency.code}_USD.parquet")
            if os.path.isfile(parquetPath):
                available[currency.code] = parquetPath

        return available


    def loadCurrToUsdData(self, currencyCode: str):
        if currencyCode not in self.availableCurrencies:
            return pd.DataFrame()

        if currencyCode == "USD":
            # Make a fake Dataframe for USD to USD conversion
            df = pd.DataFrame({
                "date": pd.date_range(start=self.startDate, end=self.endDate, freq="D"),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
            })
            return df

        key = f"forex|full_{currencyCode}_USD"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        path = self.availableCurrencies[currencyCode]
        df = pd.read_parquet(path, engine="pyarrow")

        # Ensure the timestamp is in UTC and naive
        dateCol = df["date"]
        if dateCol.dt.tz is None:
            dateCol = dateCol.dt.tz_localize(UTC)
        else:
            dateCol = dateCol.dt.tz_convert(UTC)
        df["date"] = dateCol.dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)

        self.cache.put(key, df)
        return df


    def getCurrToUsdSeries(self, currencyCode: str, startDate: pd.Timestamp, endDate: pd.Timestamp) -> Optional[pd.DataFrame]:
        df = self.loadCurrToUsdData(currencyCode)
        if df.empty:
            return None

        if startDate is not None:
            startNorm = self.normaliseTimestamp(startDate)
            df = df[df["date"] >= startNorm]
        if endDate is not None:
            endNorm = self.normaliseTimestamp(endDate)
            df = df[df["date"] <= endNorm]

        return df.reset_index(drop=True)


    def getLatestCurrToUsd(self, currencyCode: str, before: pd.Timestamp) -> Optional[float]:
        beforeNorm = self.normaliseTimestamp(before)
        if beforeNorm < self.startDate:
            return None
        
        df = self.loadCurrToUsdData(currencyCode)
        if df.empty:
            return None

        prior = df[df["date"] <= beforeNorm]
        if prior.empty:
            return None

        return float(prior.iloc[-1]["close"])