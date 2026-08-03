import os
import threading
import requests
from io import StringIO
from datetime import date, timedelta
import pandas as pd
from dataclasses import dataclass

import duckdb

from collectors.constants import SHORT_PARQUET_PATH, UTC, END_DATE
from collectors.rate_limiter import GlobalRateLimiters
from dataquery.lru_cache import LRUCache

CSV_URL_FEED = "https://cdn.finra.org/equity/otcmarket/biweekly/shrt{dateStr}.csv"


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
    def __init__(self, cache: LRUCache, rateLimiters: GlobalRateLimiters):
        self.cache = cache
        self.rateLimiters = rateLimiters
        self.shortDataPath = SHORT_PARQUET_PATH
        self.con = duckdb.connect(database=":memory:")
        self.lock = threading.RLock()
        self.maxLocalDate = self.getMaxLocalDate()

    def normaliseTimestamp(self, before: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(before)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)

    def getMaxLocalDate(self) -> pd.Timestamp:
        if os.path.exists(self.shortDataPath):
            try:
                res = self.con.execute(f"SELECT MAX(date) FROM read_parquet('{self.shortDataPath}')").fetchone()
                if res and res[0] is not None:
                    dt = pd.to_datetime(res[0])
                    self.maxLocalDate = dt.tz_localize(None) if dt.tzinfo else dt
                    return self.maxLocalDate
            except Exception:
                pass
        self.maxLocalDate = pd.Timestamp(END_DATE).tz_localize(None)
        return self.maxLocalDate

    def downloadNonLocalShortInterest(self, startDate: pd.Timestamp, endDate: pd.Timestamp):
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (CapitalAgents)"})

        cur = startDate.date() if hasattr(startDate, "date") else startDate
        endD = endDate.date() if hasattr(endDate, "date") else endDate
        fetched = []

        while cur <= endD:
            for day in [15, 28, 30, 31]:
                try:
                    target = date(cur.year, cur.month, min(day, 28 if cur.month == 2 else (30 if cur.month in [4, 6, 9, 11] else 31)))
                except ValueError:
                    continue
                if target < cur or target > endD:
                    continue

                dStr = target.strftime("%Y%m%d")
                url = CSV_URL_FEED.format(dateStr=dStr)

                self.rateLimiters.finraLimiter.wait()

                try:
                    resp = session.get(url, timeout=10)
                    if resp.status_code == 200 and resp.text.strip():
                        df = pd.read_csv(StringIO(resp.text), sep="|", engine="python")
                        if not df.empty and "settlementDate" in df.columns:
                            colsToKeep = ["settlementDate", "symbolCode", "marketClassCode", 
                                          "currentShortPositionQuantity", "previousShortPositionQuantity", 
                                          "changePercent", "averageDailyVolumeQuantity", "daysToCoverQuantity"]
                            validCols = [c for c in colsToKeep if c in df.columns]
                            df = df[validCols]
                            df = df.rename(columns={
                                "settlementDate": "date",
                                "symbolCode": "ticker",
                                "marketClassCode": "exchange",
                                "currentShortPositionQuantity": "currentShortPositions",
                                "previousShortPositionQuantity": "previousShortPositions",
                                "averageDailyVolumeQuantity": "avgDailyVolume",
                                "daysToCoverQuantity": "daysToCover"
                            })
                            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
                            df["ticker"] = df["ticker"].str.strip().str.upper()
                            fetched.append(df)
                except Exception:
                    pass

            if cur.month == 12:
                cur = date(cur.year + 1, 1, 1)
            else:
                cur = date(cur.year, cur.month + 1, 1)

        if fetched:
            dfInc = pd.concat(fetched, ignore_index=True)
            self.con.register("temp_inc", dfInc)
            self.con.execute("CREATE TABLE IF NOT EXISTS inc_short_table AS SELECT * FROM temp_inc WHERE 1=0")
            self.con.execute("INSERT INTO inc_short_table SELECT * FROM temp_inc")
            self.con.unregister("temp_inc")

    def ensureCoverage(self, targetTimestamp: pd.Timestamp):
        maxLocal = self.getMaxLocalDate()
        if targetTimestamp > maxLocal:
            with self.lock:
                self.downloadNonLocalShortInterest(maxLocal, targetTimestamp)
                self.maxLocalDate = targetTimestamp

    def getQueryRelationSql(self) -> str:
        hasInc = False
        try:
            res = self.con.execute("SELECT COUNT(*) FROM inc_short_table").fetchone()
            if res and res[0] > 0:
                hasInc = True
        except Exception:
            hasInc = False

        if hasInc:
            if os.path.exists(self.shortDataPath):
                return f"(SELECT * FROM read_parquet('{self.shortDataPath}') UNION ALL SELECT * FROM inc_short_table)"
            else:
                return "inc_short_table"
        else:
            if os.path.exists(self.shortDataPath):
                return f"read_parquet('{self.shortDataPath}')"
            else:
                return None

    def getShortInterestForTicker(
        self,
        ticker: str,
        startDate: pd.Timestamp = None,
        endDate: pd.Timestamp = None
    ) -> pd.DataFrame:
        if startDate is None and endDate is None:
            raise ValueError("At least one of startDate or endDate must be provided.")

        ticker = ticker.upper().strip()
        normStart = self.normaliseTimestamp(startDate) if startDate else None
        normEnd = self.normaliseTimestamp(endDate) if endDate else None

        key = f"short|single_{ticker}_{normStart.strftime('%Y-%m-%d') if normStart else 'none'}_{normEnd.strftime('%Y-%m-%d') if normEnd else 'none'}"

        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

            targetDate = normEnd or normStart
            if targetDate:
                self.ensureCoverage(targetDate)

            relationSql = self.getQueryRelationSql()
            if relationSql is None:
                return pd.DataFrame()

            startParam = normStart.to_pydatetime() if normStart else None
            endParam = normEnd.to_pydatetime() if normEnd else None

            sql = f"""
                SELECT *
                FROM {relationSql}
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

            try:
                df = self.con.execute(sql, params).df()
            except Exception:
                df = pd.DataFrame()

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