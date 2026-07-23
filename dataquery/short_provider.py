import os
import pandas as pd
from dataclasses import dataclass, field

import duckdb

from collectors.constants import SHORT_PARQUET_PATH, UTC
from dataquery.lru_cache import LRUCache


@dataclass
class ShortInterest:
    date: pd.Timestamp
    ticker: str
    currentShortPositions: int
    previousShortPositions: int
    changePercent: float
    avgDailyVolume: int
    daysToCover: float


class ShortDataProvider:
    def __init__(self, cache: LRUCache):
        self.cache = cache
        self.shortDataPath = SHORT_PARQUET_PATH
        self.con = duckdb.connect(database=":memory:")


    def normaliseTimestamp(self, before: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(before)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)
    

    def getShortInterestForTicker(
        self,
        ticker: str,
        startDate: pd.Timestamp = None,
        endDate: pd.Timestamp = None
    ) -> pd.DataFrame:
        if startDate is None and endDate is None:
            raise ValueError("At least one of startDate or endDate must be provided.")
        if not os.path.exists(self.shortDataPath):
            return pd.DataFrame()
        
        key = f"short|single_{ticker}_{startDate.strftime('%Y-%m-%d') if startDate else 'none'}_{endDate.strftime('%Y-%m-%d') if endDate else 'none'}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        ticker = ticker.upper().strip()
        normStart = self.normaliseTimestamp(startDate)
        normEnd = self.normaliseTimestamp(endDate)

        # Convert to DuckDB-compatible datetime literals for parameterised queries
        startParam = normStart.to_pydatetime() if normStart else None
        endParam = normEnd.to_pydatetime() if normEnd else None

        sql = f"""
            SELECT *
            FROM read_parquet('{self.shortDataPath}')
            WHERE ticker = ?
            {"AND date >= ?" if not pd.isna(startParam) else ""}
            {"AND date <= ?" if not pd.isna(endParam) else ""}
            ORDER BY date DESC, changePercent ASC
        """

        params = [ticker]
        if not pd.isna(startParam):
            params.append(startParam)
        if not pd.isna(endParam):
            params.append(endParam)

        df = self.con.execute(sql, params).df()
        if df.empty:
            self.cache.put(key, df)
            return df
            
        dateCol = df["date"]
        if dateCol.dt.tz is None:
            dateCol = dateCol.dt.tz_localize("UTC")
        else:
            dateCol = dateCol.dt.tz_convert("UTC")
        df["date"] = dateCol.dt.tz_localize(None)

        self.cache.put(key, df)
        return df
    
    def getShortInterestForTickers(
        self,
        tickers: list[str],
        startDate: pd.Timestamp = None,
        endDate: pd.Timestamp = None
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        if startDate is None and endDate is None:
            raise ValueError("At least one of startDate or endDate must be provided.")

        output = {}
        for ticker in tickers:
            output[ticker] = self.getShortInterestForTicker(ticker, startDate, endDate)

        return output 


    def getLatestShortInterestForTicker(self, ticker: str, before: pd.Timestamp) -> ShortInterest | None:
        df = self.getShortInterestForTicker(ticker, endDate=before)
        if df.empty:
            return None
        latestRow = df.iloc[0]
        return ShortInterest(
            date=latestRow["date"],
            ticker=latestRow["ticker"],
            currentShortPositions=latestRow["currentShortPositions"],
            previousShortPositions=latestRow["previousShortPositions"],
            changePercent=latestRow["changePercent"],
            avgDailyVolume=latestRow["avgDailyVolume"],
            daysToCover=latestRow["daysToCover"]
        )