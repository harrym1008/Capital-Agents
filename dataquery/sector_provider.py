import os
import threading
import time
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf

from collectors.constants import SECTOR_DIRECTORY, UTC
from collectors.rate_limiter import GlobalRateLimiters
from collectors.sector_dl_client import GICS_SECTORS, SECTOR_NAME_TO_TICKER, DB_SECTOR_TO_TICKER
from dataquery.lru_cache import LRUCache
from dataquery.keyed_lock import KeyedLockManager
from dataquery.macro_provider import MacroSeries, MacroDataProvider
from dataquery.ticker_provider import TickerDataProvider


# Provider for sector resolution and sector ETF OHLCV price series
class SectorDataProvider:
    def __init__(self, cache: LRUCache, rateLimiters: GlobalRateLimiters, 
                 macroProvider: MacroDataProvider, tickerProvider: TickerDataProvider):
        self.cache = cache
        self.rateLimiters = rateLimiters
        self.macroProvider = macroProvider
        self.tickerProvider = tickerProvider
        self.sectorDir = SECTOR_DIRECTORY
        self.keyedLocks = KeyedLockManager()
        self.downloadLock = threading.RLock()
        self.availableSectors = self.buildSectorIndex()
        self.lastAttemptTime: Dict[str, float] = {}


    def normaliseTimestamp(self, ts: pd.Timestamp) -> pd.Timestamp:
        # Convert timestamp to UTC naive representation
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)


    def resolveSector(self, sectorOrTicker: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        # Resolve sector identifier or company ticker to GICS sector ETF symbol and canonical name
        if sectorOrTicker is None:
            return None, "Unknown", "Sector not specified"

        cleaned = str(sectorOrTicker).strip()
        lower = cleaned.lower()
        upper = cleaned.upper()

        # Catch explicit unknown tokens
        if lower == "unknown":
            pass    # Reject unknown token and return list of permitted sectors

        else:
            # Direct GICS ETF ticker match
            if upper in GICS_SECTORS:
                return upper, GICS_SECTORS[upper].name, None

            # Direct database underscored key match
            if lower in DB_SECTOR_TO_TICKER:
                etfTicker = DB_SECTOR_TO_TICKER[lower]
                return etfTicker, GICS_SECTORS[etfTicker].name, None

            # Space-separated lower match (e.g. 'health care')
            spaced = lower.replace("_", " ")
            if spaced in SECTOR_NAME_TO_TICKER:
                etfTicker = SECTOR_NAME_TO_TICKER[spaced]
                return etfTicker, GICS_SECTORS[etfTicker].name, None

            # Check if query matches a company ticker (e.g. NVDA, AAPL)
            compProfile = self.tickerProvider.getTickerProfile(upper)
            if compProfile is not None:
                compSector = (compProfile.sector or "").strip().lower()
                if compSector == "unknown":
                    return None, "Unknown", f"{upper}'s ticker is unknown."
                if compSector in DB_SECTOR_TO_TICKER:
                    etfTicker = DB_SECTOR_TO_TICKER[compSector]
                    return etfTicker, GICS_SECTORS[etfTicker].name, f"Resolved from company ticker '{upper}' (sector: '{compSector}')."

        # Fallback error with valid sector list
        allowedList = "'" + "', '".join(sorted(DB_SECTOR_TO_TICKER.keys())) + "'."
        return None, None, f"Sector or ticker '{sectorOrTicker}' not recognised. Choose from this list of sectors:\n{allowedList}"

    def resolveTicker(self, sectorOrTicker: str) -> Optional[str]:
        # Return ETF symbol for sector or company ticker
        ticker, _, _ = self.resolveSector(sectorOrTicker)
        return ticker


    def buildSectorIndex(self) -> Dict[str, Optional[str]]:
        # Index local parquet files for GICS sector ETFs
        available = {}
        for ticker in GICS_SECTORS:
            path = os.path.join(self.sectorDir, f"{ticker}.parquet")
            if os.path.exists(path):
                available[ticker] = path
            else:
                available[ticker] = None
        return available

    def downloadNonLocalSector(self, ticker: str, startDate: pd.Timestamp, endDate: pd.Timestamp) -> pd.DataFrame:
        # Download missing sector ETF OHLCV data from Yahoo Finance
        self.rateLimiters.yFinanceLimiter.wait()
        try:
            rawDf = yf.download(
                ticker,
                start=startDate.strftime("%Y-%m-%d"),
                end=(endDate + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=False,
                progress=False
            )
            if rawDf.empty:
                return pd.DataFrame()

            if isinstance(rawDf.columns, pd.MultiIndex):
                rawDf.columns = rawDf.columns.get_level_values(0)

            rawDf.reset_index(inplace=True)
            rawDf.columns = rawDf.columns.str.lower()

            dateCol = pd.to_datetime(rawDf["date"])
            if dateCol.dt.tz is not None:
                dateCol = dateCol.dt.tz_convert(UTC).dt.tz_localize(None)
            rawDf["date"] = dateCol

            targetCols = ["date", "open", "high", "low", "close", "volume"]
            validCols = [c for c in targetCols if c in rawDf.columns]
            df = rawDf[validCols].dropna(subset=["date", "close"])
            return df.sort_values("date").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()


    def loadSector(self, ticker: str) -> pd.DataFrame:
        # Load full sector ETF history from parquet or cache
        now = pd.Timestamp.now(tz="UTC")
        key = f"sector|full_{ticker}_{now.strftime('%Y-%m-%dH%H')}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        with self.keyedLocks.lockKey(key):
            cached = self.cache.get(key)
            if cached is not None:
                return cached

            path = self.availableSectors.get(ticker)
            df = pd.DataFrame()
            if path and os.path.exists(path):
                try:
                    df = pd.read_parquet(path, engine="pyarrow")
                    if "date" in df.columns:
                        dateCol = pd.to_datetime(df["date"])
                        if dateCol.dt.tz is not None:
                            dateCol = dateCol.dt.tz_convert(UTC).dt.tz_localize(None)
                        df["date"] = dateCol
                        df = df.sort_values("date").reset_index(drop=True)
                except Exception:
                    df = pd.DataFrame()

            self.cache.put(key, df)
            return df

    def ensureCoverage(self, ticker: str, targetDate: pd.Timestamp):
        # Ensure sector ETF coverage up to target date with rate-limited cooldown
        targetNorm = self.normaliseTimestamp(targetDate)
        currentTime = time.time()
        cooldownSeconds = 1800  # 30 min cooldown per ticker

        df = self.loadSector(ticker)
        needsDownload = False

        if df.empty:
            needsDownload = True
            dlStart = targetNorm - pd.DateOffset(years=10)
        else:
            maxDate = df["date"].max()
            if maxDate < targetNorm:
                needsDownload = True
                dlStart = maxDate + pd.Timedelta(days=1)

        if needsDownload:
            with self.downloadLock:
                lastAttempt = self.lastAttemptTime.get(ticker, 0.0)
                if (currentTime - lastAttempt) > cooldownSeconds:
                    self.lastAttemptTime[ticker] = currentTime
                    newDf = self.downloadNonLocalSector(ticker, dlStart, targetNorm)
                    if not newDf.empty:
                        combinedDf = pd.concat([df, newDf], ignore_index=True)
                        combinedDf = combinedDf.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
                        
                        nowTs = pd.Timestamp.now(tz="UTC")
                        key = f"sector|full_{ticker}_{nowTs.strftime('%Y-%m-%dH%H')}"
                        self.cache.put(key, combinedDf)


    def getSectorData(self, sectorOrTicker: str, startDate: Optional[pd.Timestamp] = None, 
                      endDate: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        # Retrieve date-filtered sector ETF prices
        ticker = self.resolveTicker(sectorOrTicker)
        if not ticker:
            return pd.DataFrame()

        if endDate is not None:
            self.ensureCoverage(ticker, endDate)

        df = self.loadSector(ticker)
        if df.empty:
            return pd.DataFrame()

        filteredDf = df.copy()
        if startDate is not None:
            startNorm = self.normaliseTimestamp(startDate)
            filteredDf = filteredDf[filteredDf["date"] >= startNorm]
        if endDate is not None:
            endNorm = self.normaliseTimestamp(endDate)
            filteredDf = filteredDf[filteredDf["date"] <= endNorm]

        return filteredDf.sort_values("date").reset_index(drop=True)
