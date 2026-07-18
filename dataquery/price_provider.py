import os
import pandas as pd
import pyarrow.parquet as pq

from collectors.constants import NYSE_DIRECTORY, NASDAQ_DIRECTORY, ALL_TICKERS_FILE, CORP_ACTIONS_OUTPUT, NEW_YORK, UTC
from dataquery.lru_cache import LRUCache
from dataquery.ticker_provider import TickerDataProvider, CompanyProfile


class DailyPriceProvider:
    def __init__(self, startDate: pd.Timestamp, endDate: pd.Timestamp, tickerDataProvider: TickerDataProvider, cache: LRUCache):
        self.cache = cache

        self.startDate = startDate
        self.endDate = endDate

        self.tickerDataProvider = tickerDataProvider
        self.tickersPaths = self.buildTickerPathIndex()


    def timestampToNyDay(self, ts):
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(NEW_YORK)
        else:
            ts = ts.tz_convert(NEW_YORK)
        return ts.normalize()


    def getSingleDayTickerData(self, ticker, date):
        # Check the stock is currently listed
        if not self.tickerDataProvider.isTickerListed(ticker, date):
            return None

        # Account for clocks changing (+/- 1 hour)
        dateNy = self.timestampToNyDay(date)
        
        dfYear = self.getYear(ticker, date.year)
        if dateNy.dayofyear <= 7 and dateNy.year >= 2016:
            dfPrevYear = self.getYear(ticker, dateNy.year - 1)
            dfYear = pd.concat([dfPrevYear, dfYear]).sort_index()        
        
        if dfYear.empty:
            return None
        
        if dateNy in dfYear.index:
            row = dfYear.loc[dateNy]
            return row.iloc[0] if isinstance(row, pd.DataFrame) else row
        
        # Look back up to 7 days to find the last available price data
        validDates = dfYear.index[dfYear.index < dateNy]
        if validDates.empty:
            return None
        
        nearestDate = validDates.max()
        if (dateNy - nearestDate).days > 7:
            return None
        
        row = dfYear.loc[nearestDate]
        return row.iloc[0] if isinstance(row, pd.DataFrame) else row
        


    def getPeriodDailyTickerData(self, ticker, startDate: pd.Timestamp, endDate: pd.Timestamp):
        # Check the stock is currently listed
        if not self.tickerDataProvider.isTickerListed(ticker, startDate) and  \
           not self.tickerDataProvider.isTickerListed(ticker, endDate):
            return pd.DataFrame()
        
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
        key = f"ohlcv|{ticker}_{year}"
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
        try:
            # Check the date column exists in the parquet file and filter by date range
            if "date" not in pq.read_schema(path).names:
                return pd.DataFrame()       # This fixes an issue with "FFFZ.parquet", who pulled out of IPO after submitting for it, and has no data

            df = pd.read_parquet(path, engine="pyarrow", filters=[
                ("date", ">=", yearStartNY.tz_convert("UTC")), 
                ("date", "<", yearEndNY.tz_convert("UTC"))
            ])
        except:
            return pd.DataFrame()

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
        key = f"corpActions|{year}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        
        actionsDf = pd.read_parquet(CORP_ACTIONS_OUTPUT)
        actionsDf["date"] = pd.to_datetime(actionsDf["date"]).dt.tz_localize(NEW_YORK)

        result = actionsDf[actionsDf["date"].dt.year == year].reset_index(drop=True)

        self.cache.put(key, result)
        return result



    def buildTickerPathIndex(self):
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

        
    