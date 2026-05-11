
import pandas as pd

from collectors.ratelimiter import RateLimiter

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.requests import StockBarsRequest, CorporateActionsRequest
from alpaca.data.enums import Adjustment, CorporateActionsType
from alpaca.data.timeframe import TimeFrame


class AlpacaStockPricesClient:
    def __init__(self, apiKey, apiSecret, startDate, endDate):
        self.dailyPricesClient = StockHistoricalDataClient(apiKey, apiSecret)
        self.corporateActionsClient = CorporateActionsClient(apiKey, apiSecret)

        self.rateLimiter = RateLimiter("alpaca", 200, 60)   # Alpaca free allows 200 requests per minute

        self.startDate = startDate
        self.endDate = endDate

    
    def downloadOhlcAndSplitsBatch(self, tickers, nanOnSplits=True):
        if len(tickers) > 50:       # Not a hard cap by Alpaca, but I will only allow 50 tickers per batch
            raise ValueError("Too many tickers in one batch, reduce list size to 50 or less")
        
        barsRequest = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=self.startDate,
            end=self.endDate,
            adjustment=Adjustment.SPLIT
        )
        self.rateLimiter.wait()
        bars = self.dailyPricesClient.get_stock_bars(barsRequest)
        df = bars.df
        if df is None or len(df) == 0:
            return pd.DataFrame(), {}
        
        df = df.reset_index()

        # Rename columns from Alpaca's naming to standard naming convention
        df.rename(columns={
            "timestamp": "date",
            "symbol": "ticker",
        }, inplace=True)

        requiredColumns = ["date", "open", "high", "low", "close", "volume", "vwap", "ticker"]
        df["date"] = pd.to_datetime(df["date"])
        df = df[requiredColumns]

        # Now get the stock splits
        splitsRequest = CorporateActionsRequest(
            symbols=tickers,
            start=self.startDate,
            end=self.endDate,
            types=[CorporateActionsType.REVERSE_SPLIT, CorporateActionsType.FORWARD_SPLIT]
        )
        self.rateLimiter.wait()
        splitsResponse = self.corporateActionsClient.get_corporate_actions(splitsRequest)
        
        splitsData = splitsResponse.data
        allSplits = splitsData.get("forward_splits", []) + splitsData.get("reverse_splits", [])

        splitDatesByTicker = {}
        for splitInfo in allSplits:
            ticker = splitInfo.symbol
            splitDate = splitInfo.ex_date
            if ticker not in splitDatesByTicker:
                splitDatesByTicker[ticker] = set()
            splitDatesByTicker[ticker].add(splitDate)


        # NaN out the price data on split dates if requested
        # This is because Alpaca's split adjusted prices are very inaccurate around splits
        # Example: NVDA's 10-for-1 split on 10 June 2024, High value was $195.95  (data from Alpaca)
        # Yahoo Finance shows it should be $123.10 ... this is a huge discrepancy
        # As a result, the model is not allowed to trade on split dates
        # Other days, split adjustment is performed by Alpaca and is accurate and joins earlier prices with later prices correctly
        if nanOnSplits:
            ohlcColumns = ["open", "high", "low", "close", "vwap"]
            for ticker, dates in splitDatesByTicker.items():
                dateList = list(dates)
                mask = (df["ticker"] == ticker) & (df["date"].isin(dateList))
                for col in ohlcColumns:
                    df.loc[mask, col] = None

        return df, splitDatesByTicker

