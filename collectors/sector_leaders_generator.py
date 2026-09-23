import os
import concurrent.futures
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import Optional, Dict, List

from collectors.constants import ALL_TICKERS_FILE, NYSE_DIRECTORY, NASDAQ_DIRECTORY, SECTOR_LEADERS_PARQUET_PATH, NEW_YORK
from collectors.sector_dl_client import DB_SECTOR_TO_TICKER


# Generates a parquet file containing the top 25 tickers by market cap for each sector on each trading day
class SectorLeadersGenerator:
    def __init__(self, outputParquetPath: str = SECTOR_LEADERS_PARQUET_PATH):
        self.outputParquetPath = outputParquetPath
        self.tickerPaths = self.buildTickerPathsMap()

    def buildTickerPathsMap(self) -> Dict[str, str]:
        # Build a mapping of tickers to their corresponding parquet file paths in the NYSE and NASDAQ directories
        paths = {}
        if os.path.exists(NYSE_DIRECTORY):
            for fname in os.listdir(NYSE_DIRECTORY):
                if fname.endswith(".parquet"):
                    ticker = fname[:-8]
                    paths[ticker] = os.path.join(NYSE_DIRECTORY, fname)
        if os.path.exists(NASDAQ_DIRECTORY):
            for fname in os.listdir(NASDAQ_DIRECTORY):
                if fname.endswith(".parquet"):
                    ticker = fname[:-8]
                    paths[ticker] = os.path.join(NASDAQ_DIRECTORY, fname)
        return paths

    def readTickerMarketCaps(self, ticker: str, filePath: str) -> Optional[pd.DataFrame]:
        try:
            table = pq.read_table(filePath, columns=["date", "close", "outstandingShares"])
            df = table.to_pandas()
            if df.empty or "date" not in df.columns or "close" not in df.columns:
                return None

            df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(NEW_YORK).dt.normalize()
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["outstandingShares"] = pd.to_numeric(df["outstandingShares"], errors="coerce")

            df["marketCap"] = df["close"] * df["outstandingShares"]
            df = df.dropna(subset=["date", "marketCap"])
            df = df[(df["marketCap"] > 0) & (df["marketCap"] <= 10e12)]
            if df.empty:
                return None

            df["ticker"] = ticker
            return df[["date", "ticker", "marketCap"]]
        except Exception:
            return None

    def processSingleSector(self, dbKey: str, sectorTickers: List[str]) -> Dict[pd.Timestamp, List[str]]:
        sectorDfs = []
        validPaths = [(t, self.tickerPaths[t]) for t in sectorTickers if t in self.tickerPaths]

        if not validPaths:
            return {}

        # Read market cap data for each ticker in parallel using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futureToTicker = {executor.submit(self.readTickerMarketCaps, t, path): t for t, path in validPaths}
            for future in concurrent.futures.as_completed(futureToTicker):
                res = future.result()
                if res is not None and not res.empty:
                    sectorDfs.append(res)

        if not sectorDfs:
            return {}

        combined = pd.concat(sectorDfs, ignore_index=True)
        combined.sort_values(["date", "marketCap"], ascending=[True, False], inplace=True)

        sectorDateMap = {}
        for dateVal, group in combined.groupby("date", sort=True):
            topGroup = group.head(25)       # Get the top 25 tickers by market cap for this date
            topTickers = topGroup["ticker"].tolist()
            sectorDateMap[dateVal] = topTickers

        return sectorDateMap


    def generateSectorLeaders(self, pbar: Optional[tqdm] = None) -> str:
        if not os.path.exists(ALL_TICKERS_FILE):
            raise FileNotFoundError(f"Tickers file not found at {ALL_TICKERS_FILE}")

        tickersDf = pd.read_parquet(ALL_TICKERS_FILE)
        sectorColumns = sorted(list(DB_SECTOR_TO_TICKER.keys()))
        sectorResults = {}
        allDates = set()

        totalSectors = len(sectorColumns)
        if pbar is not None:
            pbar.total = totalSectors
            pbar.reset()

        for dbKey in sectorColumns:
            subset = tickersDf[
                (tickersDf["sector"] == dbKey) &
                (tickersDf["industry"].notna()) &
                (tickersDf["industry"].str.lower() != "unknown")
            ]   # Only consider tickers with a valid industry for this sector
            sectorTickers = subset["ticker"].dropna().unique().tolist()
            dateMap = self.processSingleSector(dbKey, sectorTickers)
            sectorResults[dbKey] = dateMap
            allDates.update(dateMap.keys())

            if pbar is not None:
                pbar.update(1)

        if not allDates:
            raise ValueError("No sector leaders data could be generated.")

        sortedDates = sorted(list(allDates))
        rows = []
        for d in sortedDates:
            row = {"date": d}
            for dbKey in sectorColumns:
                row[dbKey] = sectorResults.get(dbKey, {}).get(d, [])
            rows.append(row)

        resultDf = pd.DataFrame(rows)
        resultDf.sort_values("date", inplace=True)
        resultDf.reset_index(drop=True, inplace=True)

        os.makedirs(os.path.dirname(self.outputParquetPath), exist_ok=True)
        resultDf.to_parquet(self.outputParquetPath, engine="pyarrow", index=False)
        return self.outputParquetPath
