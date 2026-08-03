import os
import threading
from typing import Optional
import pandas as pd
import yfinance as yf

from collectors.constants import FOREX_DIRECTORY, UTC, END_DATE
from collectors.rate_limiter import GlobalRateLimiters
from dataquery.lru_cache import LRUCache
from collectors.forex_dl_client import Currency, CURRENCY_MAP


class ForexDataProvider:
    def __init__(self, startDate: pd.Timestamp, endDate: pd.Timestamp, cache: LRUCache, rateLimiters: GlobalRateLimiters):
        self.startDate = self.normaliseTimestamp(startDate)
        self.endDate = self.normaliseTimestamp(endDate)
        self.cache = cache
        self.rateLimiters = rateLimiters
        self.lock = threading.RLock()

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

    def downloadNonLocalForex(self, currencyCode: str, startDate: pd.Timestamp, endDate: pd.Timestamp) -> pd.DataFrame:
        currency = CURRENCY_MAP.get(currencyCode)
        if not currency or not currency.yfinanceTicker:
            return pd.DataFrame()

        self.rateLimiters.yFinanceLimiter.wait()

        try:
            df = yf.download(
                currency.yfinanceTicker,
                start=startDate.strftime("%Y-%m-%d"),
                end=(endDate + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False
            )
            if df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            df = df.drop(columns=["adj_close", "volume"], errors="ignore")
            df.columns = ["date", "open", "high", "low", "close"]

            dateCol = pd.to_datetime(df["date"])
            if dateCol.dt.tz is None:
                dateCol = dateCol.dt.tz_localize(UTC)
            else:
                dateCol = dateCol.dt.tz_convert(UTC)
            df["date"] = dateCol.dt.tz_localize(None)
            return df.sort_values("date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def loadCurrToUsdData(self, currencyCode: str) -> pd.DataFrame:
        if currencyCode not in self.availableCurrencies:
            return pd.DataFrame()

        if currencyCode == "USD":
            df = pd.DataFrame({
                "date": pd.date_range(start=self.startDate, end=self.endDate, freq="D"),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
            })
            return df

        key = f"forex|full_{currencyCode}_USD"
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

            path = self.availableCurrencies[currencyCode]
            df = pd.DataFrame()
            if path and os.path.exists(path):
                try:
                    df = pd.read_parquet(path, engine="pyarrow")
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

    def ensureCoverage(self, currencyCode: str, targetDate: pd.Timestamp) -> pd.DataFrame:
        targetNorm = self.normaliseTimestamp(targetDate)
        with self.lock:
            df = self.loadCurrToUsdData(currencyCode)
            if currencyCode == "USD" or df.empty:
                maxDate = pd.Timestamp(END_DATE).tz_localize(None)
            else:
                maxDate = df["date"].max()

            if targetNorm > maxDate:
                key = f"forex|full_{currencyCode}_USD"
                dfInc = self.downloadNonLocalForex(currencyCode, maxDate, targetNorm)
                if not dfInc.empty:
                    df = pd.concat([df, dfInc], ignore_index=True)
                    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
                    self.cache.put(key, df)

            return df

    def getCurrToUsdSeries(self, currencyCode: str, startDate: pd.Timestamp, endDate: pd.Timestamp) -> Optional[pd.DataFrame]:
        with self.lock:
            if endDate is not None:
                df = self.ensureCoverage(currencyCode, endDate)
            else:
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
        
        with self.lock:
            df = self.ensureCoverage(currencyCode, beforeNorm)
            if df.empty:
                return None

            prior = df[df["date"] <= beforeNorm]
            if prior.empty:
                return None

            return float(prior.iloc[-1]["close"])