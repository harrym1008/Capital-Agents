import os
import threading
import duckdb
import pandas as pd

from collectors.constants import NEWS_PARQUET_PATH, UTC
from dataquery.lru_cache import LRUCache


class NewsDataProvider:
    def __init__(self, cache: LRUCache):
        self.cache = cache
        self.newsPath = NEWS_PARQUET_PATH
        self.con = duckdb.connect(database=":memory:")
        self.lock = threading.Lock()


    def normaliseTimestamp(self, before: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(before)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)


    def getRecentNewsForTicker(self, ticker: str, before: pd.Timestamp, 
                               limit: int = 12, 
                               mustHaveContent: bool = False, 
                               maxReferencedTickers: int = 5) -> pd.DataFrame:
        if limit < 1:
            return pd.DataFrame()

        ticker = ticker.upper()
        beforeNorm = self.normaliseTimestamp(before)

        key = f"news|single_{ticker}_{beforeNorm.strftime('%Y-%m-%dH%H')}_{limit}"
        cached = self.cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return cached

        if not os.path.exists(self.newsPath):
            return pd.DataFrame()

        sql = f"""
            SELECT *
            FROM read_parquet('{self.newsPath}')
            WHERE date < ?
                AND list_contains(tickers, ?)
                {"AND LENGTH(content) > 0" if mustHaveContent else ""}
                {"AND array_length(tickers) <= ?" if maxReferencedTickers is not None else ""}
            ORDER BY date DESC, id
            LIMIT ?
        """
        queryParams = [beforeNorm.to_pydatetime(), ticker]
        if maxReferencedTickers is not None:
            queryParams.append(int(maxReferencedTickers))
        queryParams.append(int(limit))
        try:
            with self.lock:
                df = self.con.execute(sql, queryParams).df()
        except Exception:
            df = pd.DataFrame()

        self.cache.put(key, df)
        return df


    def getRecentNewsForTickers(self, tickers: list[str], before: pd.Timestamp, 
                               limit: int = 12, 
                               mustHaveContent: bool = False, 
                               maxReferencedTickers: int = 5) -> pd.DataFrame:
        if limit < 1:
            return pd.DataFrame()

        # De-duplicate, uppercase, and sort so the cache key is stable regardless of input ordering or duplicates
        tickerSet = sorted({t.upper() for t in tickers})
        if not tickerSet:
            return pd.DataFrame()

        beforeNorm = self.normaliseTimestamp(before)

        key = f"news|multi_{','.join(tickerSet)}_{beforeNorm.strftime('%Y-%m-%dH%H')}_{limit}"
        cached = self.cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return cached

        if not os.path.exists(self.newsPath):
            return pd.DataFrame()

        sql = f"""
            SELECT *
            FROM read_parquet('{self.newsPath}')
            WHERE date < ?
                AND list_has_any(tickers, ?)
                {"AND LENGTH(content) > 0" if mustHaveContent else ""}
                {"AND array_length(tickers) <= ?" if maxReferencedTickers is not None else ""}
            ORDER BY date DESC, id
            LIMIT ?
        """
        queryParams = [beforeNorm.to_pydatetime(), tickerSet]
        if maxReferencedTickers is not None:
            queryParams.append(int(maxReferencedTickers))
        queryParams.append(int(limit))
        try:
            with self.lock:
                df = self.con.execute(sql, queryParams).df()
        except Exception:
            df = pd.DataFrame()

        self.cache.put(key, df)
        return df


    def getNewsForTickerBetweenTimes(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp, 
                                     mustHaveContent: bool = False, 
                                     maxReferencedTickers: int = 5) -> pd.DataFrame:
        ticker = ticker.upper()
        startNorm = self.normaliseTimestamp(start)
        endNorm = self.normaliseTimestamp(end)

        key = f"news|single_{ticker}_{startNorm.strftime('%Y-%m-%dH%H')}_{endNorm.strftime('%Y-%m-%dH%H')}"
        cached = self.cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return cached

        if not os.path.exists(self.newsPath):
            return pd.DataFrame()

        sql = f"""
            SELECT *
            FROM read_parquet('{self.newsPath}')
            WHERE date >= ? AND date <= ?
                AND list_contains(tickers, ?)
                {"AND LENGTH(content) > 0" if mustHaveContent else ""}
                {"AND array_length(tickers) <= ?" if maxReferencedTickers is not None else ""}
            ORDER BY date DESC, id
        """
        queryParams = [startNorm.to_pydatetime(), endNorm.to_pydatetime(), ticker]
        if maxReferencedTickers is not None:
            queryParams.append(int(maxReferencedTickers))
        try:
            with self.lock:
                df = self.con.execute(sql, queryParams).df()
        except Exception:
            df = pd.DataFrame()

        self.cache.put(key, df)
        return df


    def getNewsForTickersBetweenTimes(self, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp, 
                                      mustHaveContent: bool = False, 
                                      maxReferencedTickers: int = 5) -> pd.DataFrame:
        # De-duplicate, uppercase, and sort so the cache key is stable regardless of input ordering or duplicates
        tickerSet = sorted({t.upper() for t in tickers})
        if not tickerSet:
            return pd.DataFrame()

        startNorm = self.normaliseTimestamp(start)
        endNorm = self.normaliseTimestamp(end)

        key = f"news|single_{','.join(tickerSet)}_{startNorm.strftime('%Y-%m-%dH%H')}_{endNorm.strftime('%Y-%m-%dH%H')}"
        cached = self.cache.get(key)
        if isinstance(cached, pd.DataFrame):
            return cached

        if not os.path.exists(self.newsPath):
            return pd.DataFrame()

        sql = f"""
            SELECT *
            FROM read_parquet('{self.newsPath}')
            WHERE date >= ? AND date <= ?
                AND list_has_any(tickers, ?)
                {"AND LENGTH(content) > 0" if mustHaveContent else ""}
                {"AND array_length(tickers) <= ?" if maxReferencedTickers is not None else ""}
            ORDER BY date DESC, id
        """
        queryParams = [startNorm.to_pydatetime(), endNorm.to_pydatetime(), tickerSet]
        if maxReferencedTickers is not None:
            queryParams.append(int(maxReferencedTickers))
        try:
            with self.lock:
                df = self.con.execute(sql, queryParams).df()
        except Exception:
            df = pd.DataFrame()

        self.cache.put(key, df)
        return df
