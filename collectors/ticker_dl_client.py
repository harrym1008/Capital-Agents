import re
import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

import pandas as pd
import financedatabase as fd
    
from collectors.market_calendar import MarketCalendar
from collectors.rate_limiter import GlobalRateLimiters
from collectors.constants import GOOD_SECURITY_TERMS, BAD_SECURITY_TERMS, IPO_BEFORE_START_DATE, NEW_YORK, \
                                 ALL_TICKERS_FILE, TEMP_TICKERS_FILE           



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
        

def getSecurityScore(row):
    name = str(row["name"]).lower()
    ticker = str(row["ticker"]).lower()
    
    # Prefer common stock
    if any(term in name for term in GOOD_SECURITY_TERMS):
        return 1

    # Strongly punish obvious non common stock securities
    if any(term in name for term in BAD_SECURITY_TERMS):
        return -len(ticker)     # In the case of multiple bad terms, prefer the shorter ticker
    
    return 0     


class TickerDataClient:
    def __init__(self, startDate, endDate, rateLimiterDatabase: GlobalRateLimiters = None):
        load_dotenv()

        self.massiveApiKey = os.getenv("MASSIVE_API_KEY")

        self.startDate = startDate
        self.endDate = endDate
        self.limiters = rateLimiterDatabase if rateLimiterDatabase else GlobalRateLimiters()

    
    def massTickerDownloadWithData(self):
        if os.path.exists(TEMP_TICKERS_FILE):
            allTickersDf = pd.read_parquet(TEMP_TICKERS_FILE)
            print(f"Loaded {TEMP_TICKERS_FILE} with {len(allTickersDf):,} tickers.")

        else:
            # Download all tickers (active and delisted) from Massive API
            historicalTickers = {"XNAS": {"CS": [], "ADRC": []}, "XNYS": {"CS": [], "ADRC": []}}

            print("=" * 70)
            print("Fetching historical tickers from Massive API...")

            for tickerType in ["CS", "ADRC"]:    # "Common Stock" and "American Depositary Receipt Common" only
                for exchange in ["XNAS", "XNYS"]:
                    for status in [True, False]:
                        lenBefore = len(historicalTickers[exchange][tickerType])
                        baseUrl = "https://api.massive.com/v3/reference/tickers"
                        params = {
                            "market": "stocks",
                            "type": tickerType,       
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
                                historicalTickers[exchange][tickerType].extend(response["results"])

                            if "next_url" in response:
                                params = {"apiKey": self.massiveApiKey}
                                baseUrl = response["next_url"]

                                print(f"\rFor {'listed' if status else 'delisted'} '{tickerType}' stocks on {'NASDAQ' if exchange == 'XNAS' else 'NYSE' \
                                            }, fetched {len(historicalTickers[exchange][tickerType]) - lenBefore} tickers so far...  ", end="")
                            else:
                                print(f"\rCompleted fetching {'listed' if status else 'delisted'} '{tickerType}' stocks on {'NASDAQ' if exchange == 'XNAS' else 'NYSE' \
                                            }... got {len(historicalTickers[exchange][tickerType]) - lenBefore:,} equities.        ")
                                break
            
            print(f"Fetched:")
            print(f"    {len(historicalTickers['XNAS']['CS']) + len(historicalTickers['XNAS']['ADRC']):,} historical NASDAQ tickers")
            print(f"    {len(historicalTickers['XNYS']['CS']) + len(historicalTickers['XNYS']['ADRC']):,} historical NYSE tickers")


            # Do not have IPO dates yet
            # But tickers that were delisted before startDate are not relevant and can be removed

            for exchange in ["XNAS", "XNYS"]:
                for tickerType in ["CS", "ADRC"]:
                    for i in range(len(historicalTickers[exchange][tickerType]) - 1, -1, -1):
                        tickerData = historicalTickers[exchange][tickerType][i]
                        tickerData["isAdrc"] = (tickerType == "ADRC")                           

                        if tickerData["active"]:        # Skip active tickers
                            continue

                        deleteTicker = False

                        delistDateStr = tickerData.get("delisted_utc", pd.NA)
                        if pd.isna(delistDateStr):    
                            # Active is False, but there is no delist date.... default to treating it as delisted
                            # print(f"Warning: Ticker {tickerData['ticker']} is marked as inactive but has no delist date.") 
                            deleteTicker = True
                        else:
                            delistTs = pd.Timestamp(delistDateStr, tz="UTC").tz_convert(NEW_YORK)
                            startTs = pd.Timestamp(self.startDate)
                            deleteTicker = delistTs < startTs

                        if tickerData["currency_name"] != "usd":        # Only concerned with USD stocks
                            deleteTicker = True

                        if deleteTicker:
                            historicalTickers[exchange][tickerType].pop(i)


            # Get metadata from Financedatabase for all tickers we got from Massive API

            allTickersData = [ticker for exchange in historicalTickers for tickerType in historicalTickers[exchange] for ticker in historicalTickers[exchange][tickerType]]
            allTickersDf = pd.DataFrame(allTickersData)

            allTickersDf.rename(columns={"primary_exchange": "exchange"}, inplace=True)
            # allTickersDf.rename(columns={"currency_name": "currency"}, inplace=True)
            allTickersDf.to_parquet(TEMP_TICKERS_FILE, index=False)

        todayMinus20Years = datetime.now() - timedelta(days=365*20)  # Earliest permitted date is 20 years ago (discrepancy with leap years gives safe leeway)
        todayPlusYear = datetime.now() + timedelta(days=364)         # Latest permitted date is 1 year from now (only add 364 days for safety)
        calendar = MarketCalendar(todayMinus20Years.strftime("%Y-%m-%d"), todayPlusYear.strftime("%Y-%m-%d")) 

        equities = fd.Equities()
        fdbDf = equities.select().reset_index().rename(columns={"index": "symbol"})
        fdbDf = fdbDf[fdbDf["exchange"].isin(["NMS", "NYQ"])]
        fdbDf = fdbDf.drop(columns=["exchange"])        # Prevent conflict with exchange column from Massive API

        # Merge names, prefer fdb names
        df = pd.merge(allTickersDf, fdbDf, left_on="ticker", right_on="symbol", how="left")

        df["name"] = df["name_y"].fillna(df["name_x"])
        df.drop(columns=["name_x", "name_y", "symbol"], inplace=True, errors="ignore")

        # Active --> listed
        df["listed"] = df["active"].fillna(False)
        # Default IPO date for all ticker... updated when OHLCV is downloaded
        df["ipoDate"] = IPO_BEFORE_START_DATE.strftime("%Y-%m-%d")    
        
        # Format delist date to the trading day before Massive says 
        # Massive returns the first day the stock is not trading, I want the last day it trades
        def formatDelistDate(row):
            if pd.isna(row.get("delisted_utc")):
                return pd.NA
            delistDate = pd.Timestamp(row["delisted_utc"], tz="UTC").tz_convert(NEW_YORK)
            return getPreviousTradingDay(delistDate, calendar)
        df["delistDate"] = df.apply(formatDelistDate, axis=1)

        # Pretty sector and industry keys for easier grouping and analysis later
        df["sector"] = df["sector"].apply(makeSectorOrIndustryKey)
        df["industry"] = df["industry"].apply(makeSectorOrIndustryKey)

        # Final other textual elements                               Regex = start string, 0 or more whitepace, end of string
        df[["website", "summary"]] = df[["website", "summary"]].replace(r"^\s*$", pd.NA, regex=True).fillna("unknown")
        df["cik"] = df["cik"].dropna()
        df["isin"] = df["isin"].fillna("unknown")

        df = df[[
            "ticker",
            "name",
            "exchange",
            "isAdrc",
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

        rowsToKeep = []

        # Filter out bad tickers that are not common stocks, but listed as such
        for cik, group in df.groupby("cik"):
            group = group.copy()

            group["secscore"] = group.apply(getSecurityScore, axis=1)
            validRows = group[group["secscore"] >= 0]

            if len(validRows) > 0:
                rowsToKeep.append(validRows)
            else:
                bestRow = group.sort_values(["secscore", "ticker"], ascending=[False, True]).head(1)
                rowsToKeep.append(bestRow)



        filteredDf = pd.concat(rowsToKeep, ignore_index=True)
        filteredDf.drop(columns=["secscore"], errors="ignore", inplace=True)

        finalDf = filteredDf.sort_values(["ticker", "name"]).reset_index(drop=True)
        finalDf.to_parquet(ALL_TICKERS_FILE, index=False)
        print(f"Saved final ticker metadata to {ALL_TICKERS_FILE} with {len(finalDf):,} tickers.")

        if os.path.exists(TEMP_TICKERS_FILE):
            os.remove(TEMP_TICKERS_FILE)
