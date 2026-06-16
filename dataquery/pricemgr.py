import os
import pandas as pd

from collectors.constants import NYSE_DIRECTORY, NASDAQ_DIRECTORY, ALL_TICKERS_FILE, CORP_ACTIONS_OUTPUT, NEW_YORK, UTC
from dataquery.lrucache import LRUCache



class DailyPricesClient:
    def __init__(self, startDate: pd.Timestamp, endDate: pd.Timestamp, cache=None, cacheSize=1024**3):
        self.startDate = startDate
        self.endDate = endDate
        self.cache = LRUCache(cacheSize) if cache is None else cache
        self.tickersPaths = self.buildTickerIndex()


    def timestampToNyDay(self, ts):
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(NEW_YORK)
        else:
            ts = ts.tz_convert(NEW_YORK)
        return ts.normalize()


    def getSingleDayTickerData(self, ticker, date):
        # Account for clocks changing (+/- 1 hour)
        dfYear = self.getYear(ticker, date.year)
        dateNy = self.timestampToNyDay(date)
        
        if dfYear.empty or dateNy not in dfYear.index:
            return None
        
        row = dfYear.loc[dateNy]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row
        


    def getPeriodDailyTickerData(self, ticker, startDate: pd.Timestamp, endDate: pd.Timestamp):
        # Account for clocks changing (+/- 1 hour)
        startNy = self.timestampToNyDay(startDate)
        endNy = self.timestampToNyDay(endDate)
        if startNy > endNy:
            startNy, endNy = endNy, startNy

        years = range(startNy.year, endNy.year + 1)

        parts = []
        for year in years:
            dfYear = self.getYear(ticker, year)
            if not dfYear.empty:
                parts.append(dfYear)

        if not parts:
            return pd.DataFrame()
        
        df = pd.concat(parts).sort_index()
        df = df.loc[startNy:endNy]
        return df.reset_index(drop=True)
        

    def getYear(self, ticker, year):
        key = f"{ticker}_{year}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        path = self.tickersPaths.get(ticker)
        if not path:
            return pd.DataFrame()
        
        yearStartNY = pd.Timestamp(f"{year}-01-01", tz=NEW_YORK)
        yearEndNY = pd.Timestamp(f"{year+1}-01-01", tz=NEW_YORK)
        if yearEndNY > self.endDate:
            yearEndNY = self.endDate + pd.Timedelta(days=1)

        # Parquet files store dates in UTC; convert filter timestamps to match.
        df = pd.read_parquet(path, filters=[
            ("date", ">=", yearStartNY.tz_convert("UTC")), 
            ("date", "<", yearEndNY.tz_convert("UTC"))
        ])

        dateCol = df["date"]
        if dateCol.dt.tz is None:
            dateCol = dateCol.dt.tz_localize(UTC)
        else:
            dateCol = dateCol.dt.tz_convert(UTC)

        df["date"] = dateCol
        dateNy = dateCol.dt.tz_convert(NEW_YORK).dt.normalize()
        df = df.assign(dateNy=dateNy).set_index("dateNy").sort_index()
    
        self.cache.put(key, df)
        return df


    def getSingleDayCorpActions(self, date):
        actions = self.getYearCorporateActions(date.year)
        dateNy = self.timestampToNyDay(date)
        return actions[actions["date"].dt.normalize() == dateNy]



    def getYearCorporateActions(self, year):
        key = f"corpActions_{year}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        actionsDf = pd.read_parquet(CORP_ACTIONS_OUTPUT)
        actionsDf["date"] = pd.to_datetime(actionsDf["date"]).dt.tz_localize(NEW_YORK)

        result = actionsDf[actionsDf["date"].dt.year == year].reset_index(drop=True)

        self.cache.put(key, result)
        return result



    def buildTickerIndex(self):
        allTickersDf = pd.read_parquet(ALL_TICKERS_FILE)

        tickerPaths = {}
        for row in allTickersDf.itertuples():
            ticker = row.ticker
            exchange = row.exchange

            if exchange == "XNYS":
                path = os.path.join(NYSE_DIRECTORY, f"{ticker}.parquet")
            else:
                path = os.path.join(NASDAQ_DIRECTORY, f"{ticker}.parquet")

            if os.path.exists(path):
                tickerPaths[ticker] = path
            else:
                tickerPaths[ticker] = None
                # print(f"Warning: No price data found for {ticker} at expected path {path}")
        
        return tickerPaths

        
    