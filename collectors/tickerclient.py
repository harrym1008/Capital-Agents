import re
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

import pandas as pd
import financedatabase as fd
    
from collectors.mktcalendar import MarketCalendar
from collectors.ratelimiter import GlobalRateLimiters
from collectors.constants import IPO_BEFORE_START_DATE, NEW_YORK


def makeSectorOrIndustryKey(text):
    if text is None or pd.isna(text):
        return "unknown"
    text = str(text).strip()
    if text == "":
        return "unknown"
    
    text = re.sub(r"\([^)]*\)", "", text)           # Remove brackets and their contents
    text = text.lower()         
    text = text.replace("&", " and ")               
    text = re.sub(r"[^a-z0-9]+", "_", text)        # Replace non-alphanumeric characters with underscore
    text = re.sub(r"_+", "_", text)                # Replace multiple underscores with single underscore
    text = text.strip("_")                         # Remove leading/trailing underscores

    if text == "":
        return "unknown"
    return text


def getPreviousTradingDay(delistedUtcTs, calendar):
    current = datetime.strptime(delistedUtcTs.strftime("%Y-%m-%d"), "%Y-%m-%d")
    while True:
        current -= timedelta(days=1)        # Always at least 1 day before delist date... delist date is the first day the stock isnt trading
        dayStr = current.strftime("%Y-%m-%d")
        if dayStr in calendar.openDays:
            return dayStr
        


class TickerDataClient:
    def __init__(self, startDate, endDate, rateLimiterDatabase: GlobalRateLimiters = None):
        load_dotenv()

        self.alphaVantageApiKey = os.getenv("ALPHAVANTAGE_API_KEY")
        self.alpacaApiKey = os.getenv("ALPACA_API_KEY")
        self.alpacaApiSecret = os.getenv("ALPACA_API_SECRET")
        self.massiveApiKey = os.getenv("MASSIVE_API_KEY")

        self.startDate = startDate
        self.endDate = endDate
        self.limiters = rateLimiterDatabase if rateLimiterDatabase else GlobalRateLimiters()

    
    def massTickerDownloadWithData(self):
        if os.path.exists("all_tickers.parquet"):
            allTickersDf = pd.read_parquet("all_tickers.parquet")
            print(f"Loaded all_tickers.parquet with {len(allTickersDf):,} tickers.")

        else:
            # Download all tickers (active and delisted) from Massive API
            historicalTickers = {"XNAS": [], "XNYS": []}

            print("=" * 70)
            print("Fetching historical tickers from Massive API...")

            for exchange in ["XNAS", "XNYS"]:
                for status in [True, False]:
                    lenBefore = len(historicalTickers[exchange])
                    baseUrl = "https://api.massive.com/v3/reference/tickers"
                    params = {
                        "market": "stocks",
                        "type": "CS",       # "Common Stock" only
                        "exchange": exchange,
                        "active": str(status).lower(),
                        "sort": "ticker",
                        "order": "asc",
                        "limit": 1000,
                        "apiKey": self.massiveApiKey
                    }

                    while True:
                        self.limiters.massiveLimiter.wait()
                        response = requests.get(baseUrl, params=params).json()
                        if "results" in response:
                            historicalTickers[exchange].extend(response["results"])

                        if "next_url" in response:
                            params = {"apiKey": self.massiveApiKey}
                            baseUrl = response["next_url"]

                            print(f"\rFor {'listed' if status else 'delisted'} stocks on {'NASDAQ' if exchange == 'XNAS' else 'NYSE' \
                                        }, fetched {len(historicalTickers[exchange]) - lenBefore} tickers so far...  ", end="")
                        else:
                            print(f"\rCompleted fetching {'listed' if status else 'delisted'} stocks on {'NASDAQ' if exchange == 'XNAS' else 'NYSE' \
                                        }... got {len(historicalTickers[exchange]) - lenBefore:,} equities.        ")
                            break
            
            print(f"Fetched:")
            print(f"    {len(historicalTickers['XNAS']):,} historical NASDAQ tickers")
            print(f"    {len(historicalTickers['XNYS']):,} historical NYSE tickers")


            # Do not have IPO dates yet
            # But tickers that were delisted before startDate are not relevant and can be removed

            for exchange in ["XNAS", "XNYS"]:
                for i in range(len(historicalTickers[exchange]) - 1, -1, -1):
                    if historicalTickers[exchange][i]["active"]:        # Skip active tickers
                        continue

                    deleteTicker = False

                    delistDateStr = historicalTickers[exchange][i].get("delisted_utc", pd.NA)
                    if pd.isna(delistDateStr):    
                        # Active is False, but there is no delist date.... default to treating it as delisted
                        # print(f"Warning: Ticker {historicalTickers[exchange][i]['ticker']} is marked as inactive but has no delist date.") 
                        deleteTicker = True
                    else:
                        delistTs = pd.Timestamp(delistDateStr, tz="UTC").tz_convert(NEW_YORK)
                        startTs = pd.Timestamp(self.startDate)
                        deleteTicker = delistTs < startTs
                    if deleteTicker:
                        historicalTickers[exchange].pop(i)


            # Get metadata from Financedatabase for all tickers we got from Massive API

            allTickersData = [ticker for exchange in historicalTickers for ticker in historicalTickers[exchange]]
            allTickersDf = pd.DataFrame(allTickersData)

            allTickersDf.rename(columns={"primary_exchange": "exchange"}, inplace=True)
            # allTickersDf.rename(columns={"currency_name": "currency"}, inplace=True)
            allTickersDf.to_parquet("all_tickers.parquet", index=False)

        calendar = MarketCalendar(self.startDate.strftime("%Y-%m-%d"), self.endDate.strftime("%Y-%m-%d"))

        equities = fd.Equities()
        fdbDf = equities.select().reset_index().rename(columns={"index": "symbol"})
        fdbDf = fdbDf[fdbDf["exchange"].isin(["NMS", "NYQ"])]
        fdbDf = fdbDf.drop(columns=["exchange"])        # Prevent conflict with exchange column from Massive API

        # Merge names, prefer fdb names
        finalDf = pd.merge(allTickersDf, fdbDf, left_on="ticker", right_on="symbol", how="left")

        finalDf["name"] = finalDf["name_y"].fillna(finalDf["name_x"])
        finalDf.drop(columns=["name_x", "name_y", "symbol"], inplace=True, errors="ignore")

        # Active --> listed
        finalDf["listed"] = finalDf["active"].fillna(False)
        # Default IPO date for all ticker... updated when OHLCV is downloaded
        finalDf["ipoDate"] = IPO_BEFORE_START_DATE.strftime("%Y-%m-%d")    
        
        # Format delist date to the trading day before Massive says 
        # Massive returns the first day the stock is not trading, I want the last day it trades
        def formatDelistDate(row):
            if pd.isna(row.get("delisted_utc")):
                return pd.NA
            delistDate = pd.Timestamp(row["delisted_utc"], tz="UTC").tz_convert(NEW_YORK)
            return getPreviousTradingDay(delistDate, calendar)
        finalDf["delistDate"] = finalDf.apply(formatDelistDate, axis=1)

        # Pretty sector and industry keys for easier grouping and analysis later
        finalDf["sector"] = finalDf["sector"].apply(makeSectorOrIndustryKey)
        finalDf["industry"] = finalDf["industry"].apply(makeSectorOrIndustryKey)

        # Final other textual elements                               Regex = start string, 0 or more whitepace, end of string
        finalDf[["website", "summary"]] = finalDf[["website", "summary"]].replace(r"^\s*$", pd.NA, regex=True).fillna("unknown")
        finalDf["cik"] = finalDf["cik"].fillna("unknown")
        finalDf["isin"] = finalDf["isin"].fillna("unknown")

        finalDf = finalDf[[
            "ticker",
            "name",
            "exchange",
            "listed",
            "ipoDate",
            "delistDate",
            "sector",
            "industry",
            "website",
            "summary",
            "cik",
            "isin"
        ]]
        finalDf = finalDf.sort_values(["exchange", "ticker"]).reset_index(drop=True)
        finalDf.to_parquet("data/tickers.parquet", index=False)
        print(f"Saved final ticker metadata to data/tickers.parquet with {len(finalDf):,} tickers.")

        if os.path.exists("all_tickers.parquet"):
            os.remove("all_tickers.parquet")
