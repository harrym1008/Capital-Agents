import json
import os
import threading
import time
from datetime import datetime, timedelta
import concurrent.futures
from dotenv import load_dotenv

from tqdm import tqdm
import pandas as pd

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.requests import StockBarsRequest, CorporateActionsRequest
from alpaca.data.enums import Adjustment, CorporateActionsType
from alpaca.data.timeframe import TimeFrame

from collectors.ratelimiter import GlobalRateLimiters, RateLimiter
from collectors.constants import *



NYSE_DIRECTORY = "data/ohlcv_nyse/"
NASDAQ_DIRECTORY = "data/ohlcv_nasdaq/"

OHLC_FILE_OUTPUT = "{}_ohlcv.parquet"
ACTIONS_FILE_OUTPUT = "{}_actions.json"



# When delistDate is None, it means the stock is still trading as of endDate
class SingleTickerDataCollector:
    def __init__(self, ticker, exchange, startDate, endDate, delistDate, 
                 priceClient: StockHistoricalDataClient, 
                 corpActionsClient: CorporateActionsClient, 
                 alpacaLimiter: RateLimiter):
        self.ticker = ticker
        self.exchange = exchange
        self.startDateStr = startDate
        self.endDateStr = endDate
        self.delistDateStr = delistDate

        self.startDate = datetime.strptime(startDate, "%Y-%m-%d")
        self.endDate = datetime.strptime(endDate, "%Y-%m-%d")
        self.delistDate = datetime.strptime(delistDate, "%Y-%m-%d") if delistDate else None

        self.priceClient = priceClient
        self.corpActionsClient = corpActionsClient
        self.limiter = alpacaLimiter


    def collect(self):
        timeRange = self.getTimeRange()
        
        priceDf, ipoDate = self.collectPriceData(timeRange)
        latestNameCheck, actions = self.collectCorporateActions(timeRange, priceDf)
        if not latestNameCheck:
            return False, None, None, None
        
        self.addCorpDataToDf(priceDf, actions)

        return True, priceDf, ipoDate, actions


    def getTimeRange(self):
        collectStart = self.startDateStr
        if self.delistDate and self.delistDate < self.endDate:
            collectEnd = self.delistDate.strftime("%Y-%m-%d")
        else:
            collectEnd = self.endDateStr

        if self.delistDate and self.delistDate < self.startDate:
            # This case should've been filtered out in tickerclient.py already
            return self.startDateStr, self.endDateStr

        return collectStart, collectEnd


    def collectPriceData(self, timeRange):
        collectStart, collectEnd = timeRange
        priceRequest = StockBarsRequest(
            symbol_or_symbols=self.ticker,
            start=collectStart,
            end=collectEnd,
            limit=10_000,
            timeframe=TimeFrame.Day,
            adjustment=Adjustment.RAW
        )

        self.limiter.wait()
        priceData = self.priceClient.get_stock_bars(priceRequest)
        df = priceData.df
        
        if df.empty:
            return pd.DataFrame(), None, {}
        
        df = df.reset_index()
        df.rename(columns={"timestamp": "date", "symbol": "ticker   "}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df[["date", "open", "high", "low", "close", "volume", "vwap"]]

        ipoDate = df["date"].min()
        if ipoDate <= FIRST_TRAD_DAY_AFTER_START:
            ipoDate = IPO_BEFORE_START_DATE

        return df, ipoDate        


    def collectCorporateActions(self, timeRange, df, nameChangeRerun=None):
        if nameChangeRerun is None:
            ticker = self.ticker
            ignoredNextTickers = []
        else:
            ticker = nameChangeRerun[0]
            ignoredNextTickers = nameChangeRerun[1]
        
        collectStart, collectEnd = timeRange
        request = CorporateActionsRequest(
            symbols=[ticker],
            start=collectStart,
            end=collectEnd,
            types=[
                # Splits
                CorporateActionsType.REVERSE_SPLIT, 
                CorporateActionsType.FORWARD_SPLIT, 
                # Cash dividends
                CorporateActionsType.CASH_DIVIDEND, 
                CorporateActionsType.STOCK_DIVIDEND,
                # Mergers and acquisitions
                CorporateActionsType.CASH_MERGER,
                CorporateActionsType.STOCK_MERGER,
                CorporateActionsType.STOCK_AND_CASH_MERGER,
                # Restructuring
                CorporateActionsType.SPIN_OFF,
                CorporateActionsType.WORTHLESS_REMOVAL,     # Bankruptcy
                CorporateActionsType.NAME_CHANGE
            ],
            limit=1000
        )
                
        self.limiter.wait()
        response = self.corpActionsClient.get_corporate_actions(request)
        data = response.data

        actions = {
            "splits": [],
            "dividends": [],
            "mergers": [],
            "restructures": []
        }

        # Process forward splits
        for split in data.get("forward_splits", []):
            actions["splits"].append({
                "type": CorporateActionsType.FORWARD_SPLIT,
                "date": split.ex_date.strftime("%Y-%m-%d"),
                "oldRate": split.old_rate,
                "newRate": split.new_rate
            })
        
        # Process reverse splits
        for split in data.get("reverse_splits", []):
            actions["splits"].append({
                "type": CorporateActionsType.REVERSE_SPLIT,
                "date": split.ex_date.strftime("%Y-%m-%d"),
                "oldRate": split.old_rate,
                "newRate": split.new_rate
            })
        
        # Process cash dividends
        for dividend in data.get("cash_dividends", []):
            actions["dividends"].append({
                "type": CorporateActionsType.CASH_DIVIDEND,
                "date": dividend.ex_date.strftime("%Y-%m-%d"),
                "rate": dividend.rate,
                "special": dividend.special
            })

        # Process stock dividends
        for dividend in data.get("stock_dividends", []):
            actions["dividends"].append({
                "type": CorporateActionsType.STOCK_DIVIDEND,
                "date": dividend.ex_date.strftime("%Y-%m-%d"),
                "rate": dividend.rate,
                "special": False
            })

        # Process cash mergers
        for merger in data.get("cash_mergers", []):
            actions["mergers"].append({
                "type": CorporateActionsType.CASH_MERGER,
                "date": merger.effective_date.strftime("%Y-%m-%d"),
                "rate": merger.rate,
                "acquireeTicker": merger.acquiree_symbol,
                "acquirerTicker": merger.acquirer_symbol
            })

        # Process stock mergers
        for merger in data.get("stock_mergers", []):
            actions["mergers"].append({
                "type": CorporateActionsType.STOCK_MERGER,
                "date": merger.effective_date.strftime("%Y-%m-%d"),
                "acquireeTicker": merger.acquiree_symbol,
                "acquireeRate": merger.acquiree_rate,
                "acquirerTicker": merger.acquirer_symbol,
                "acquirerRate": merger.acquirer_rate
            })

        # Process stock and cash mergers
        for merger in data.get("stock_and_cash_mergers", []):
            actions["mergers"].append({
                "type": CorporateActionsType.STOCK_AND_CASH_MERGER,
                "date": merger.effective_date.strftime("%Y-%m-%d"),
                "acquireeTicker": merger.acquiree_symbol,
                "acquireeRate": merger.acquiree_rate,
                "acquirerTicker": merger.acquirer_symbol,
                "acquirerRate": merger.acquirer_rate,
                "cashRate": merger.cash_rate
            })

        for spinOff in data.get("spin_offs", []):
            actions["restructures"].append({
                "type": CorporateActionsType.SPIN_OFF,
                "date": spinOff.ex_date.strftime("%Y-%m-%d"),
                "newTicker": spinOff.new_symbol,
                "newRate": spinOff.new_rate,
                "sourceTicker": spinOff.source_symbol,
                "sourceRate": spinOff.source_rate
            })

        for removal in data.get("worthless_removals", []):
            actions["restructures"].append({
                "type": CorporateActionsType.WORTHLESS_REMOVAL,
                "date": removal.process_date.strftime("%Y-%m-%d"),
                "ticker": removal.symbol
            })

        for nameChange in data.get("name_changes", []):
            actions["restructures"].append({
                "type": CorporateActionsType.NAME_CHANGE,
                "date": nameChange.process_date.strftime("%Y-%m-%d"),
                "oldTicker": nameChange.old_symbol,
                "newTicker": nameChange.new_symbol
            })


        # if there are any name changes: FIRST CHECK that this ticker is the most recent
        nameHasChanged = False
        for restructure in actions["restructures"]:
            if restructure["type"] == CorporateActionsType.NAME_CHANGE:
                nameHasChanged = True
                if restructure["oldTicker"] == ticker and restructure["newTicker"] not in ignoredNextTickers:
                    return False, []
                

        
        # If this point is reached, this ticker is the most recent name. If there are other name changes, must work backwards
        if nameHasChanged:
            mergedActions = {
                "splits": [],
                "dividends": [],
                "mergers": [],
                "restructures": []
            }
            for restructure in actions["restructures"]:
                if restructure["type"] == CorporateActionsType.NAME_CHANGE:
                    if restructure["newTicker"] == ticker:
                        # Need to recollect corporate actions for old ticker, to get any splits/dividends that would affect historical prices
                        rerunTicker = restructure["oldTicker"]
                        latestNameCheck, oldActions = self.collectCorporateActions(timeRange, df, 
                                                            nameChangeRerun=(rerunTicker, ignoredNextTickers + [ticker]))
                                                
                        # Merge splits and dividends from old ticker into current actions
                        for key in mergedActions:
                            mergedActions[key].extend(oldActions.get(key, []))

            for key in actions:
                actions[key].extend(mergedActions[key])

            # Remove duplicate name changes
            seenNameChanges = set()
            uniqueRestructures = []
            for restructure in actions["restructures"]:
                if restructure["type"] == CorporateActionsType.NAME_CHANGE:
                    nameChangeKey = (restructure["oldTicker"], restructure["newTicker"])
                    if nameChangeKey not in seenNameChanges:
                        seenNameChanges.add(nameChangeKey)
                        uniqueRestructures.append(restructure)
                else:
                    uniqueRestructures.append(restructure)
            actions["restructures"] = uniqueRestructures

        return True, actions


    def addCorpDataToDf(self, df, actions):
        # Add split factor column to adjust historical prices for splits
        df["splitFactor"] = 1.0
        df["corpActionToday"] = False

        for action in actions["splits"]:
            exDate = pd.Timestamp(action["date"], tz=NEW_YORK)
            factor = action["oldRate"] / action["newRate"]

            df.loc[df["date"] < exDate, "splitFactor"] *= factor

            # NaN the prices on the ex-date (Alpaca data is inaccurate)
            # Example: NVDA high at $195 on 2024-06-10 on day of 10:1 split, 
            # but should be around $123
            df.loc[df["date"] == exDate, ["open", "high", "low", "close", "vwap"]] = pd.NA

        for actionList in actions.values():
            for action in actionList:
                exDate = pd.Timestamp(action["date"], tz=NEW_YORK)
                df.loc[df["date"] == exDate, "corpActionToday"] = True

        return df
    



def threadWorker(ticker, exchange, startDate, endDate, delistDate, priceClient, corpActionsClient, limiter):
    collector = SingleTickerDataCollector(
        ticker=ticker,
        exchange=exchange,
        startDate=startDate,
        endDate=endDate,
        delistDate=delistDate,
        priceClient=priceClient,
        corpActionsClient=corpActionsClient,
        alpacaLimiter=limiter
    )
    passed, priceDf, ipoDate, actions = collector.collect()
    if not passed:
        return False

    if exchange == "XNAS":
        baseDir = NASDAQ_DIRECTORY
    else:
        baseDir = NYSE_DIRECTORY

    parquetPath = os.path.join(baseDir, OHLC_FILE_OUTPUT.format(ticker))
    jsonPath = os.path.join(baseDir, ACTIONS_FILE_OUTPUT.format(ticker))

    priceDf.to_parquet(parquetPath, index=False)

    with open(jsonPath, "w") as f:
        json.dump(actions, f, indent=4, default=str)
    
    return True, ticker, ipoDate



class OHLCVDataClient:
    def __init__(self, startDate, endDate, rateLimiterDatabase: GlobalRateLimiters = None):
        load_dotenv()

        self.alphaVantageApiKey = os.getenv("ALPHAVANTAGE_API_KEY")
        self.alpacaApiKey = os.getenv("ALPACA_API_KEY")
        self.alpacaApiSecret = os.getenv("ALPACA_API_SECRET")
        self.massiveApiKey = os.getenv("MASSIVE_API_KEY")

        self.startDate = startDate
        self.endDate = endDate
        self.limiters = rateLimiterDatabase if rateLimiterDatabase else GlobalRateLimiters()

        self.priceClient = StockHistoricalDataClient(self.alpacaApiKey, self.alpacaApiSecret)
        self.corpActionsClient = CorporateActionsClient(self.alpacaApiKey, self.alpacaApiSecret)


    def massDownload(self, updateIpoDates=False):
        tickersPath = "data/tickers.parquet"

        if not os.path.exists(tickersPath):
            raise FileNotFoundError(f"Cannot find {tickersPath}")
        
        tickersDf = pd.read_parquet(tickersPath)
        totalTickers = len(tickersDf)


        tickerRows = []
        for idx, row in tickersDf.iterrows():
            tickerRows.append({
                "ticker": row["ticker"],
                "exchange": row["exchange"],
                "startDate": self.startDate.strftime("%Y-%m-%d"),
                "endDate": self.endDate.strftime("%Y-%m-%d"),
                "delistDate": row["delistDate"] if not pd.isna(row["delistDate"]) else None,
                "rowIdx": idx
            })

        
        results = {}
        resultsLock = threading.Lock()
        completed = 0
        errors = 0


        print(f"\nBeginning mass download of OHLCV data for {totalTickers} tickers...\n")
        pbar = tqdm(
            total=totalTickers,
            desc="Downloading OHLCV for all tickers",
            smoothing=0.8,
            # bar_format="{desc}| {percentage:3.2f}% |{bar}| [{elapsed} elapsed, {remaining} remaining] ",
            colour="green"
        )

        def downloadAndTrack(item):
            nonlocal completed, errors
            try:
                passed, ticker, ipoDate = threadWorker(
                    item["ticker"], item["exchange"], item["startDate"], item["endDate"], item["delistDate"],
                    self.priceClient, self.corpActionsClient, self.limiters.alpacaLimiter
                )

                if not passed:
                    with resultsLock:
                        completed += 1
                        results[item["rowIndex"]] = None
                        pbar.set_postfix_str(f"{ticker:>5} was intentionally skipped")
                        pbar.update(1)
                    return

                with resultsLock:
                    completed += 1
                    results[item["rowIdx"]] = ipoDate
                    pbar.set_postfix_str(f"{ticker:>5}, {errors} errors so far")
                    pbar.update(1)

            except Exception as e:
                with resultsLock:
                    errors += 1
                    results[item["rowIndex"]] = None
                    print(f"  Error downloading data for {item['ticker']}: {e}\n")
                    pbar.set_postfix_str(f"ERROR: {ticker:>5} - {e}")
                    pbar.update(1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(downloadAndTrack, item) for item in tickerRows]
            concurrent.futures.wait(futures)

        pbar.close()


        if updateIpoDates:
            for idx, ipoDate in results.items():
                tickersDf.at[idx, "ipoDate"] = ipoDate

            tickersDf.to_parquet(tickersPath, index=False)
            print(f"\nUpdated IPO dates for {completed} tickers and saved to {tickersPath}.")


    