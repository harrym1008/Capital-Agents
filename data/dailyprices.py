import pytz
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from data.ratelimiter import RateLimiter

NY_TZ = pytz.timezone("America/New_York")


class DailyPriceClient:
    def __init__(self, startDate="2020-01-01"):
        self.startDate = startDate
        self.yfLimiter = RateLimiter("dailyPrices", 30, 60)  # 30 calls per min
        # Yahoo finance has no specific rate limit, hopefully 30 calls per min is safe.
        
        self.nyReferenceTime = datetime.now(NY_TZ)
        self.marketCloseTime = self.nyReferenceTime.replace(hour=16, minute=2, second=0, microsecond=0)
                            # Add 2 minutes buffer to ensure market data is settled

    def getDailyPrices(self, ticker):
        iters = 0
        self.yfLimiter.wait()

        while iters < 10:
            try:
                tickerObj = yf.Ticker(ticker)
                priceDf = tickerObj.history(interval="1d", start=self.startDate)

                # Make sure to ignore the most recent price if it's before market close time, it is not the close price
                latestCompletedDate = self.nyReferenceTime.date()
                if self.nyReferenceTime < self.marketCloseTime:
                    latestCompletedDate -= timedelta(days=1)
                closedPrices = priceDf[priceDf.index.date <= latestCompletedDate]

                if not closedPrices.empty:
                    latestClosePrice = closedPrices["Close"].iloc[-1]
                    closedPrices["ticker"] = ticker
                    return closedPrices, latestClosePrice
                    
            except Exception as e:
                iters += 1
                if "429" in str(e):
                    self.yfLimiter.got429(iters)

        print(f"FAILED to fetch daily prices for {ticker} after 10 attempts!")
        return pd.DataFrame(), 0.0

