import os
import threading
import pandas as pd
import pyarrow.parquet as pq
import yfinance as yf

from collectors.constants import NYSE_DIRECTORY, NASDAQ_DIRECTORY, ALL_TICKERS_FILE,  \
                                 CORP_ACTIONS_OUTPUT, NEW_YORK, UTC, START_DATE, END_DATE
from collectors.rate_limiter import GlobalRateLimiters
from dataquery.lru_cache import LRUCache
from dataquery.ticker_provider import TickerDataProvider


# Provider for daily stock OHLCV prices, corporate actions, and split adjustments
class DailyPriceProvider:
    def __init__(self, tickerDataProvider: TickerDataProvider, cache: LRUCache, rateLimiters: GlobalRateLimiters, allowOnlineDownloads: bool = True):
        self.cache = cache
        self.rateLimiters = rateLimiters
        self.startDate = START_DATE
        self.endDate = END_DATE
        self.tickerDataProvider = tickerDataProvider
        self.tickersPaths = self.buildTickerPathIndex()
        self.lock = threading.RLock()
        self.allowOnlineDownloads = allowOnlineDownloads

    def timestampToNyDay(self, ts):
        # Convert timestamp to normalised NY trading calendar day
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(NEW_YORK)
        else:
            ts = ts.tz_convert(NEW_YORK)
        return ts.normalize()

    def downloadNonLocalOhlcv(self, ticker: str, startDate: pd.Timestamp, endDate: pd.Timestamp) -> pd.DataFrame:
        # Download recent daily OHLCV bars from Yahoo Finance when exceeding local date range
        if not self.allowOnlineDownloads:
            return pd.DataFrame()

        self.rateLimiters.yFinanceLimiter.wait()

        try:
            yfTicker = yf.Ticker(ticker)
            dfOnline = yfTicker.history(
                start=startDate,
                end=endDate + pd.Timedelta(days=1),
                interval="1d",
                auto_adjust=True,
                actions=False
            )
            if dfOnline.empty:
                return pd.DataFrame()

            dfOnline = dfOnline.reset_index()
            dfOnline.columns = dfOnline.columns.str.lower()

            dateCol = dfOnline["date"]
            if dateCol.dt.tz is None:
                dateCol = dateCol.dt.tz_localize(UTC)
            else:
                dateCol = dateCol.dt.tz_convert(UTC)

            dfOnline["date"] = dateCol

            # Populate fallback defaults for missing columns
            if "vwap" not in dfOnline.columns:
                dfOnline["vwap"] = dfOnline["close"]
            if "volume" not in dfOnline.columns:
                dfOnline["volume"] = 0

            dfOnline["splitFactor"] = 1.0
            dfOnline["corpActionToday"] = False

            try:
                fallbackShares = getattr(yfTicker, "fast_info", {}).get("shares_outstanding")
                if not fallbackShares:
                    fallbackShares = 0
                dfOnline["outstandingShares"] = float(fallbackShares)
            except Exception:
                dfOnline["outstandingShares"] = 0.0

            def formatMarketCap(marketCap):
                # Format market cap number into human-readable compact string
                def clean(number):
                    if number >= 100:
                        return f"{number:.0f}"
                    if number >= 10:
                        return f"{number:.1f}"
                    return f"{number:.2f}"
                
                if pd.isna(marketCap) or marketCap == 0:
                    return "N/A"
                elif marketCap >= 1e12:
                    return f"{clean(marketCap / 1_000_000_000_000)}tn"
                elif marketCap >= 1e9:
                    return f"{clean(marketCap / 1_000_000_000)}bn"
                elif marketCap >= 1e6:
                    return f"{clean(marketCap / 1_000_000)}mn"
                elif marketCap >= 1e3:
                    return f"{clean(marketCap / 1_000)}k"
                else:
                    return f"{clean(marketCap)}"

            dfOnline["marketCapNumber"] = dfOnline["close"] * dfOnline["outstandingShares"]
            dfOnline["marketCap"] = dfOnline["marketCapNumber"].map(formatMarketCap)
            dfOnline.drop(columns=["marketCapNumber"], inplace=True)

            dateNy = dfOnline["date"].dt.tz_convert(NEW_YORK).dt.normalize()
            dfOnline = dfOnline.assign(dateNy=dateNy).set_index("dateNy").sort_index()

            cols = ["date", "open", "high", "low", "close", "volume", "vwap", "splitFactor", "corpActionToday", "outstandingShares", "marketCap"]
            validCols = [c for c in cols if c in dfOnline.columns]
            return dfOnline[validCols]
        except Exception:
            return pd.DataFrame()

    def adjustPriceDataSplits(self, df: pd.DataFrame, ticker: str, referenceDate: pd.Timestamp = None) -> pd.DataFrame:
        # Rebase historical price series to reference date split factor
        if df.empty or "splitFactor" not in df.columns:
            return df

        df = df.copy()
        refSplitFactor = None
        if referenceDate is not None:
            refNy = self.timestampToNyDay(referenceDate)
            if "dateNy" in df.columns and (df["dateNy"] == refNy).any():
                refSplitFactor = df.loc[df["dateNy"] == refNy, "splitFactor"].iloc[0]
            elif df.index.name == "dateNy" and refNy in df.index:
                row = df.loc[refNy]
                refSplitFactor = row["splitFactor"].iloc[0] if isinstance(row, pd.DataFrame) else row["splitFactor"]
            else:
                refRow = self.getSingleDayTickerData(ticker, referenceDate)
                if refRow is not None and "splitFactor" in refRow:
                    refSplitFactor = refRow["splitFactor"]

        if refSplitFactor is None or pd.isna(refSplitFactor):
            refSplitFactor = df["splitFactor"].iloc[-1]

        # Apply multiplier across price columns
        if pd.notna(refSplitFactor) and refSplitFactor != 0:
            for col in ["open", "high", "low", "close", "vwap"]:
                if col in df.columns:
                    df[col] = df[col] * (df["splitFactor"] / refSplitFactor)
            df["splitFactor"] = refSplitFactor
        return df

    def getSingleDayTickerData(self, ticker: str, date: pd.Timestamp, referenceDate: pd.Timestamp = None):
        # Fetch single trading day OHLCV record with split adjustment
        if not self.tickerDataProvider.isTickerListed(ticker, date):
            return None

        dateNy = self.timestampToNyDay(date)
        key = f"ohlcv|single_{ticker}_{dateNy.strftime('%Y-%m-%dH%H')}"

        with self.lock:
            cached = self.cache.get(key)
            result = None
            if cached is not None:
                result = cached
            elif self.allowOnlineDownloads and dateNy > self.endDate:
                # We need to download the data from Yahoo Finance for dates beyond local coverage
                dfOnline = self.downloadNonLocalOhlcv(ticker, dateNy - pd.Timedelta(days=7), dateNy)
                if not dfOnline.empty and dateNy in dfOnline.index:
                    row = dfOnline.loc[dateNy]
                    result = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                    self.cache.put(key, result)
                elif not dfOnline.empty:
                    validDates = dfOnline.index[dfOnline.index <= dateNy]
                    if not validDates.empty:
                        nearestDate = validDates.max()
                        row = dfOnline.loc[nearestDate]
                        result = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                        self.cache.put(key, result)

            if result is None:
                dfYear = self.getYear(ticker, date.year)
                # Load previous year data for first week boundary lookup
                if dateNy.dayofyear <= 7 and dateNy.year >= 2016:
                    dfPrevYear = self.getYear(ticker, dateNy.year - 1)
                    dfYear = pd.concat([dfPrevYear, dfYear]).sort_index()        
                
                if not dfYear.empty:
                    if dateNy in dfYear.index:
                        row = dfYear.loc[dateNy]
                        result = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                        self.cache.put(key, result)     # Cache the result for future lookups
                    else:
                        validDates = dfYear.index[dfYear.index < dateNy]
                        if not validDates.empty:
                            nearestDate = validDates.max()
                            if (dateNy - nearestDate).days <= 7:
                                row = dfYear.loc[nearestDate]
                                result = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                                self.cache.put(key, result)

            # Apply split adjustment relative to reference date if provided
            if referenceDate is not None and result is not None and "splitFactor" in result:
                refRow = self.getSingleDayTickerData(ticker, referenceDate)
                if refRow is not None and "splitFactor" in refRow:
                    refSplitFactor = refRow["splitFactor"]
                    if pd.notna(refSplitFactor) and refSplitFactor != 0:
                        result = result.copy()
                        for col in ["open", "high", "low", "close", "vwap"]:
                            if col in result:
                                result[col] = result[col] * (result["splitFactor"] / refSplitFactor)
                        result["splitFactor"] = refSplitFactor

            return result

    def getPeriodDailyTickerData(self, ticker: str, startDate: pd.Timestamp, endDate: pd.Timestamp, referenceDate: pd.Timestamp = None):
        # Fetch daily OHLCV dataframe over date range stitching annual parquet slices
        if not self.tickerDataProvider.isTickerListed(ticker, startDate) and  \
           not self.tickerDataProvider.isTickerListed(ticker, endDate):
            return pd.DataFrame()

        startNy = self.timestampToNyDay(startDate)
        endNy = self.timestampToNyDay(endDate)
        if startNy > endNy:
            startNy, endNy = endNy, startNy

        key = f"ohlcv|period_{ticker}_{startNy.strftime('%Y-%m-%d%H')}_{endNy.strftime('%Y-%m-%d%H')}"
        with self.lock:
            cached = self.cache.get(key)
            if isinstance(cached, pd.DataFrame):
                df = cached
            else:
                parts = []
                # Load parquet slices for years covered within local dataset range
                if startNy <= self.endDate:
                    localEndNy = min(endNy, self.endDate)
                    years = range(startNy.year, localEndNy.year + 1)
                    for year in years:
                        dfYear = self.getYear(ticker, year)
                        if not dfYear.empty:
                            parts.append(dfYear)

                # Fetch online updates for period past local end date
                if self.allowOnlineDownloads and endNy > self.endDate:
                    gapStart = max(startNy, self.endDate)
                    dfOnline = self.downloadNonLocalOhlcv(ticker, gapStart, endNy)
                    if not dfOnline.empty:
                        parts.append(dfOnline)

                if not parts:
                    return pd.DataFrame()

                df = pd.concat(parts).sort_index()
                df = df[~df.index.duplicated(keep="last")]
                df = df.loc[startNy:endNy].reset_index(drop=True)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(UTC)
                self.cache.put(key, df)

            return self.adjustPriceDataSplits(df, ticker, referenceDate=referenceDate)

    def getYear(self, ticker: str, year: int):
        # Load annual parquet partition for ticker
        key = f"ohlcv|{ticker}_{year}"
        now = pd.Timestamp.now(tz="UTC")
        if year == now.year:
            key += f"_{now.strftime('%Y-%m-%dH%H')}"

        with self.lock:
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

            try:
                if "date" not in pq.read_schema(path).names:
                    return pd.DataFrame()

                df = pd.read_parquet(path, engine="pyarrow", filters=[
                    ("date", ">=", yearStartNY.tz_convert("UTC")), 
                    ("date", "<", yearEndNY.tz_convert("UTC"))
                ])
            except Exception:
                return pd.DataFrame()

            dateCol = df["date"]
            if dateCol.dt.tz is None:
                dateCol = dateCol.dt.tz_localize(UTC)
            else:
                dateCol = dateCol.dt.tz_convert(UTC)

            df["date"] = dateCol
            dateNy = dateCol.dt.tz_convert(NEW_YORK).dt.normalize()
            df = df.assign(dateNy=dateNy).set_index("dateNy").sort_index()

            # Put the whole year into cache for future lookups
            self.cache.put(key, df)
            return df

    def getSingleDayCorpActions(self, date):
        # Fetch corporate actions taking place on target date
        actions = self.getYearCorporateActions(date.year)
        dateNy = self.timestampToNyDay(date)
        return actions[actions["date"].dt.normalize() == dateNy]

    def getYearCorporateActions(self, year: int):
        # Load corporate action records for given calendar year
        key = f"corpActions|{year}"
        now = pd.Timestamp.now(tz="UTC")
        if year == now.year:
            key += f"_{now.strftime('%Y-%m-%dH%H')}"
            
        with self.lock:
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            
            actionsDf = pd.read_parquet(CORP_ACTIONS_OUTPUT)
            actionsDf["date"] = pd.to_datetime(actionsDf["date"]).dt.tz_localize(NEW_YORK)

            result = actionsDf[actionsDf["date"].dt.year == year].reset_index(drop=True)

            self.cache.put(key, result)
            return result

    def buildTickerPathIndex(self):
        # Index local parquet file paths across exchanges
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
        
        return tickerPaths