import os
import duckdb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from collectors.constants import SECTOR_LEADERS_PARQUET_PATH, NEW_YORK, START_DATE, END_DATE
from collectors.sector_dl_client import DB_SECTOR_TO_TICKER, GICS_SECTORS
from dataquery.lru_cache import LRUCache
from dataquery.keyed_lock import KeyedLockManager


class SectorLeadersProvider:
    def __init__(self, cache: Optional[LRUCache] = None, parquetPath: str = SECTOR_LEADERS_PARQUET_PATH):
        self.cache = cache
        self.parquetPath = parquetPath
        self.con = duckdb.connect(database=":memory:")
        self.keyedLocks = KeyedLockManager()

    def normaliseTimestamp(self, date: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(date)
        if ts.tzinfo is None:
            ts = ts.tz_localize(NEW_YORK)
        else:
            ts = ts.tz_convert(NEW_YORK)
        return ts.normalize().tz_localize(None)

    def isDataAvailable(self) -> bool:
        return os.path.exists(self.parquetPath)

    def getSectorLeaders(self, sector: str, asOfDate: pd.Timestamp, limit: int = 25) -> List[str]:
        allLeaders = self.getAllSectorLeaders(asOfDate, limit=limit)
        cleanSec = sector.upper()
        if cleanSec in allLeaders:
            return allLeaders[cleanSec]
        secLower = sector.lower().replace(" ", "_")
        return allLeaders.get(secLower, [])

    def getAllSectorLeaders(self, asOfDate: pd.Timestamp, limit: int = 25) -> Dict[str, List[str]]:
        if not self.isDataAvailable():
            return {}

        normDate = self.normaliseTimestamp(asOfDate)
        dateStr = normDate.strftime("%Y-%m-%d")
        cacheKey = f"sectorleaders|all_{dateStr}_{limit}"

        if self.cache is not None:
            cached = self.cache.get(cacheKey)
            if cached is not None:
                return cached

        with self.keyedLocks.lockKey(cacheKey):
            if self.cache is not None:
                cached = self.cache.get(cacheKey)
                if cached is not None:
                    return cached

            cursor = self.con.cursor()
            escapedPath = self.parquetPath.replace("\\", "/")
            query = f"""
                SELECT *
                FROM read_parquet('{escapedPath}')
                WHERE date = (
                    SELECT MAX(date)
                    FROM read_parquet('{escapedPath}')
                    WHERE date <= ?
                )
            """
            try:
                df = cursor.execute(query, [normDate]).df()
            except Exception:
                df = pd.DataFrame()

            output = {}
            if not df.empty:
                row = df.iloc[0]
                for dbKey, etfTicker in DB_SECTOR_TO_TICKER.items():
                    tickersList = row.get(dbKey)
                    if tickersList is not None and isinstance(tickersList, (list, tuple, np.ndarray)):
                        tickers = list(tickersList)
                    elif hasattr(tickersList, "tolist"):
                        tickers = tickersList.tolist()
                    else:
                        tickers = []

                    if limit and len(tickers) > limit:
                        tickers = tickers[:limit]

                    output[etfTicker] = tickers
                    output[dbKey] = tickers

            if self.cache is not None and output:
                self.cache.put(cacheKey, output)

            return output
