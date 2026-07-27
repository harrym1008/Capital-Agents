import os
import duckdb
import pandas as pd

from collectors.constants import NEWS_PARQUET_PATH, UTC
from dataquery.lru_cache import LRUCache


class NewsDataProvider:
    def __init__(self, cache: LRUCache):
        self.cache = cache
        self.newsPath = NEWS_PARQUET_PATH
        self.con = duckdb.connect(database=":memory:")


    def normaliseTimestamp(self, before: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(before)
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        else:
            ts = ts.tz_convert(UTC)
        return ts.tz_localize(None)


    def getRecentNewsForTicker(self, ticker: str, before: pd.Timestamp, 
                               limit: int = 12, mustHaveContent: bool = False) -> pd.DataFrame:
        if limit < 1:
            return pd.DataFrame()

        ticker = ticker.upper()
        beforeNorm = self.normaliseTimestamp(before)

        key = f"news|single_{ticker}_{beforeNorm.isoformat()}_{limit}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if not os.path.exists(self.newsPath):
            return pd.DataFrame()

        sql = f"""
            SELECT *
            FROM read_parquet('{self.newsPath}')
            WHERE updated_at < ?
              AND list_contains(symbols, ?)
              {"AND LENGTH(content) > 0" if mustHaveContent else ""}
            ORDER BY updated_at DESC, id
            LIMIT ?
        """
        df = self.con.execute(
            sql,
            [beforeNorm.to_pydatetime(), ticker, int(limit)]
        ).df()

        self.cache.put(key, df)
        return df


    def getRecentNewsForTickers(self, tickers: list[str], before: pd.Timestamp, 
                                limit: int = 12, mustHaveContent: bool = False) -> pd.DataFrame:
        if limit < 1:
            return pd.DataFrame()

        # De-duplicate, uppercase, and sort so the cache key is stable regardless of input ordering or duplicates
        tickerSet = sorted({t.upper() for t in tickers})
        if not tickerSet:
            return pd.DataFrame()

        beforeNorm = self.normaliseTimestamp(before)

        key = f"news|multi_{','.join(tickerSet)}_{beforeNorm.isoformat()}_{limit}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if not os.path.exists(self.newsPath):
            return pd.DataFrame()

        sql = f"""
            SELECT *
            FROM read_parquet('{self.newsPath}')
            WHERE updated_at < ?
              AND list_has_any(symbols, ?)
              {"AND LENGTH(content) > 0" if mustHaveContent else ""}
            ORDER BY updated_at DESC, id
            LIMIT ?
        """
        df = self.con.execute(
            sql,
            [beforeNorm.to_pydatetime(), tickerSet, int(limit)]
        ).df()

        self.cache.put(key, df)
        return df
