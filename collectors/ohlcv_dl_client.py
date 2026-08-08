import json
import requests
import os
import threading
import time
from datetime import datetime, timedelta
import concurrent.futures
from dotenv import load_dotenv

from tqdm import tqdm
import pandas as pd
import numpy as np
import yfinance as yf

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.requests import StockBarsRequest, CorporateActionsRequest
from alpaca.data.enums import Adjustment, CorporateActionsType
from alpaca.data.timeframe import TimeFrame

from collectors.rate_limiter import GlobalRateLimiters, RateLimiter
from collectors.constants import *



# When delistDate is None, it means the stock is still trading as of endDate
class SingleTickerDataCollector:
    def __init__(self, ticker, exchange, isAdrc, cik, startDate, endDate, delistDate, 
                 priceClient: StockHistoricalDataClient, 
                 corpActionsClient: CorporateActionsClient, 
                 alpacaLimiter: RateLimiter,
                 edgarLimiter: RateLimiter):
        self.ticker = ticker
        self.exchange = exchange
        self.isAdrc = isAdrc
        self.cik = cik
        self.startDateStr = startDate
        self.endDateStr = endDate
        self.delistDateStr = delistDate

        self.startDate =  pd.Timestamp(startDate, tz=NEW_YORK)
        self.endDate = pd.Timestamp(endDate, tz=NEW_YORK)
        self.delistDate = pd.Timestamp(delistDate, tz=NEW_YORK) if delistDate else None

        self.priceClient = priceClient
        self.corpActionsClient = corpActionsClient
        self.alpacaLimiter = alpacaLimiter
        self.edgarLimiter = edgarLimiter


    def collect(self):
        timeRange = self.getTimeRange()
        
        df, ipoDate = self.collectPriceData(timeRange)
        latestNameCheck, actions = self.collectCorporateActions(timeRange, df)
        if not latestNameCheck:
            return False, pd.DataFrame(), IPO_BEFORE_START_DATE, \
                {type_: [] for type_ in ["splits", "dividends", "mergers", "restructures"]}
        
        df = self.addCorpDataToDf(df, actions)
        df = self.addOutstandingSharesToDf(df, actions)

        return True, df, ipoDate, actions


    def getTimeRange(self):
        collectStart = self.startDateStr
        if self.delistDate and self.delistDate < self.endDate:
            collectEnd = self.delistDateStr
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

        self.alpacaLimiter.wait()
        priceData = self.priceClient.get_stock_bars(priceRequest)
        df = priceData.df
        
        if df.empty:
            return pd.DataFrame(), IPO_BEFORE_START_DATE
        
        df = df.reset_index()
        df.rename(columns={"timestamp": "date", "symbol": "ticker"}, inplace=True)
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
                
        self.alpacaLimiter.wait()
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
            # Alpaca's dividend data sometimes is incomplete, ex-date = process-date and missing record date.
            # Ex-div date is always accurate, so use this and set the process date to 4 weeks after ex-date (if they are the same)
            # Declaration date is always missing... so in the simulation, simulate announcement 3 weeks before ex-date

            if dividend.ex_date == dividend.process_date:
                dividend.process_date = dividend.ex_date + timedelta(weeks=4)            

            actions["dividends"].append({
                "type": CorporateActionsType.CASH_DIVIDEND,
                "date": dividend.ex_date.strftime("%Y-%m-%d"),      # "Date" = "Ex-dividend date"
                "rate": dividend.rate,
                "payDate": dividend.process_date.strftime("%Y-%m-%d"),
                "special": dividend.special
            })

        # Process stock dividends (do the same adjustment as cash dividends for missing process date and declaration date)
        for dividend in data.get("stock_dividends", []):
            if dividend.ex_date == dividend.process_date:
                dividend.process_date = dividend.ex_date + timedelta(weeks=4)    

            actions["dividends"].append({
                "type": CorporateActionsType.STOCK_DIVIDEND,
                "date": dividend.ex_date.strftime("%Y-%m-%d"),
                "rate": dividend.rate,
                "payDate": dividend.process_date.strftime("%Y-%m-%d"),
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
        # Also filter out stale name changes from different securities that reused this ticker
        nameHasChanged = False
        staleNameChanges = []
        for restructure in actions["restructures"]:
            if restructure["type"] == CorporateActionsType.NAME_CHANGE:
                nameHasChanged = True
                if restructure["oldTicker"] == ticker and restructure["newTicker"] not in ignoredNextTickers:
                    # Prevent ticker reuse
                    changeDateTs = pd.Timestamp(restructure["date"], tz=NEW_YORK)
                    if not df.empty and (df["date"] > changeDateTs).any():
                        staleNameChanges.append(restructure)
                        continue
                    return False, []

        # Remove stale name changes so the recursive pass below doesn't follow them
        if staleNameChanges:
            actions["restructures"] = [r for r in actions["restructures"] if r not in staleNameChanges]
            nameHasChanged = any(r["type"] == CorporateActionsType.NAME_CHANGE for r in actions["restructures"])

        
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

            # Order all actions by their date
            for key in actions:
                actions[key].sort(key=lambda x: pd.Timestamp(x["date"], tz=NEW_YORK))

        return True, actions


    def addCorpDataToDf(self, df, actions):
        # Add split factor column to adjust historical prices for splits
        df["splitFactor"] = 1.0
        df["corpActionToday"] = False

        for action in actions["splits"]:
            exDate = pd.Timestamp(action["date"], tz=NEW_YORK)
            factor = action["oldRate"] / action["newRate"]

            df.loc[df["date"] < exDate, "splitFactor"] *= factor

            # NaN the prices on the ex-date (Alpaca data is inaccurate) so they can be fwd filled
            # Example: NVDA high at $195 on 2024-06-10 on day of 10:1 split, 
            # but should be around $123
            df.loc[df["date"] == exDate, ["open", "high", "low", "close", "volume", "vwap"]] = pd.NA

        df["open", "high", "low", "close", "volume", "vwap"].bfill(inplace=True).ffill(inplace=True)


        for actionList in actions.values():
            for action in actionList:
                exDate = pd.Timestamp(action["date"], tz=NEW_YORK)
                df.loc[df["date"] == exDate, "corpActionToday"] = True

        return df
    

    def addOutstandingSharesToDf(self, df, actions):
        if pd.isna(self.cik):
            df["outstandingShares"] = pd.NA
            df["marketCap"] = "N/A"
            return df

        cikPadded = self.cik.zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cikPadded}.json"
        headers = {"User-Agent": SEC_EDGAR_IDENTITY}

        self.edgarLimiter.wait()
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()

        nameTagOutputs = {
            ("dei", "EntityCommonStockSharesOutstanding"): None,
            ("us-gaap", "CommonStockSharesOutstanding"): None,
            ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"): None,
            ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"): None,
        }

        for namespace, tagKey in nameTagOutputs:
            try:
                series = data["facts"][namespace][tagKey]["units"]["shares"]
                nameTagOutputs[(namespace, tagKey)] = series
            except Exception:
                continue

        # Overwrite shares outstanding dates from least reliable to most reliable:
        sharesByDate = {}

        # Least reliable: weighted average shares outstanding from US-GAAP
        for key in [("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
                     ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic")]:
            series = nameTagOutputs.get(key)
            if series is not None:
                for entry in series:
                    period = pd.Timestamp(entry["filed"], tz=NEW_YORK)
                    if self.startDate <= period <= self.endDate:
                        sharesByDate[period] = entry["val"]

        # Middle reliability: common stock shares outstanding from US-GAAP
        csSeries = nameTagOutputs.get(("us-gaap", "CommonStockSharesOutstanding"))
        if csSeries is not None:
            for entry in csSeries:
                period = pd.Timestamp(entry["filed"], tz=NEW_YORK)
                if self.startDate <= period <= self.endDate:
                    sharesByDate[period] = entry["val"]

        # Most reliable data is dei > EntityCommonStockSharesOutstanding
        deiSeries = nameTagOutputs.get(("dei", "EntityCommonStockSharesOutstanding"))
        if deiSeries is not None:
            for entry in deiSeries:
                period = pd.Timestamp(entry["filed"], tz=NEW_YORK)
                if self.startDate <= period <= self.endDate:
                    sharesByDate[period] = entry["val"]

        results = [{"date": date, "outstandingShares": value}
                   for date, value in sharesByDate.items()]        
        sharesDf = pd.DataFrame(results) if results else pd.DataFrame()

        if sharesDf.empty:
            df["outstandingShares"] = pd.NA
            df["marketCap"] = "N/A"
            return df

        sharesDf.sort_values("date", inplace=True)

        # Extrapolate ADR ratio from yfinance
        if self.isAdrc:
            try:
                yfTicker = yf.Ticker(self.ticker)
                yfShares = yfTicker.info.get("sharesOutstanding")

                if yfShares and yfShares > 0:
                    latestSecShares = sharesDf["outstandingShares"].iloc[-1]
                    adrRatio = yfShares / latestSecShares
                    sharesDf["outstandingShares"] *= adrRatio
                else:
                    raise Exception("YFinance shares outstanding data is missing or invalid")
            except Exception:
                df["outstandingShares"] = pd.NA
                df["marketCap"] = "N/A"
                return df            
        
        df = df.merge(sharesDf, on="date", how="left")
        # Forward fill first, then backfill (to fill it in before self.startDate)
        df["outstandingShares"] = df["outstandingShares"].ffill()
        df["outstandingShares"] = df["outstandingShares"].bfill()

        # On price splits, outstanding shares must be multiplied by the split factor to reflect the change in shares outstanding
        # until the next date with a split adjusted outstanding shares value
        for action in actions["splits"]:
            splitDate = pd.Timestamp(action["date"], tz=NEW_YORK)
            factor = action["newRate"] / action["oldRate"]

            # Find the next date 
            nextDate = sharesDf[sharesDf["date"] > splitDate]["date"].min()

            if pd.isna(nextDate):
                mask = df["date"] >= splitDate
            else:
                mask = (df["date"] >= splitDate) & (df["date"] < nextDate)

            df.loc[mask, "outstandingShares"] *= factor


        # Add formatted market cap column
        def formatMarketCap(marketCap):
            def clean(number):
                if number >= 100:
                    return f"{number:.0f}"
                if number >= 10:
                    return f"{number:.1f}"
                return f"{number:.2f}"
            
            if pd.isna(marketCap):
                return f"N/A"
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

        df["marketCapNumber"] = df["close"] * df["outstandingShares"]
        df["marketCap"] = df["marketCapNumber"].map(formatMarketCap)
        df.drop(columns=["marketCapNumber"], inplace=True)

        return df      
    

    
def threadWorker(ticker, exchange, isAdrc, cik, startDate, endDate, delistDate, priceClient, corpActionsClient, alpacaLimiter, edgarLimiter):
    collector = SingleTickerDataCollector(
        ticker=ticker,
        exchange=exchange,
        isAdrc=isAdrc,
        cik=cik,
        startDate=startDate,
        endDate=endDate,
        delistDate=delistDate,
        priceClient=priceClient,
        corpActionsClient=corpActionsClient,
        alpacaLimiter=alpacaLimiter,
        edgarLimiter=edgarLimiter
    )
    passed, priceDf, ipoDate, actions = collector.collect()
    if not passed:
        return False, ticker, IPO_BEFORE_START_DATE, None

    if exchange == "XNAS":
        baseDir = NASDAQ_DIRECTORY
    else:
        baseDir = NYSE_DIRECTORY

    parquetPath = os.path.join(baseDir, OHLC_FILE_OUTPUT.format(ticker))
    priceDf.to_parquet(parquetPath, index=False)

    actionRows = []

    for actionList in actions.values():
        for action in actionList:
            actionRows.append({
                "date": action.get("date"),
                "ticker": ticker,
                "exchange": exchange,
                "actionType": action.get("type").value,

                "oldRate": action.get("oldRate"),
                "newRate": action.get("newRate"),
                "rate": action.get("rate"),
                "payDate": action.get("payDate"),
                "special": action.get("special"),
                "acquireeTicker": action.get("acquireeTicker"),
                "acquireeRate": action.get("acquireeRate"),
                "acquirerTicker": action.get("acquirerTicker"),
                "acquirerRate": action.get("acquirerRate"),
                "cashRate": action.get("cashRate"),
                "oldTicker": action.get("oldTicker"),
                "newTicker": action.get("newTicker"),
                "sourceTicker": action.get("sourceTicker"),
                "sourceRate": action.get("sourceRate"),
            })
    
    return True, ticker, ipoDate, actionRows



class OHLCVDataClient:
    def __init__(self, startDate, endDate, rateLimiterDatabase: GlobalRateLimiters = None):
        load_dotenv()

        self.alphaVantageApiKey = os.getenv("ALPHAVANTAGE_API_KEY")
        self.alpacaApiKey = os.getenv("ALPACA_API_KEY")
        self.alpacaApiSecret = os.getenv("ALPACA_API_SECRET")
        self.massiveApiKey = os.getenv("MASSIVE_API_KEY")

        self.testedTickers = []
        self.startDate = startDate
        self.endDate = endDate
        self.limiters = rateLimiterDatabase if rateLimiterDatabase else GlobalRateLimiters()

        self.priceClient = StockHistoricalDataClient(self.alpacaApiKey, self.alpacaApiSecret)
        self.corpActionsClient = CorporateActionsClient(self.alpacaApiKey, self.alpacaApiSecret)



    def massDownload(self, updateIpoDates=False, threads=8, pbar=None):
        for directory in [NYSE_DIRECTORY, NASDAQ_DIRECTORY]:
            if not os.path.exists(directory):
                os.mkdir(directory)

            for filename in os.listdir(directory):
                if filename.endswith(".parquet"):
                    os.remove(os.path.join(directory, filename))

        if not os.path.exists(ALL_TICKERS_FILE):
            raise FileNotFoundError(f"Cannot find {ALL_TICKERS_FILE}")
        
        tickersDf = pd.read_parquet(ALL_TICKERS_FILE)
        totalTickers = len(tickersDf) if not self.testedTickers else len(self.testedTickers)


        tickerRows = []
        for idx, row in tickersDf.iterrows():
            if self.testedTickers and row["ticker"] not in self.testedTickers:
                continue

            tickerRows.append({
                "ticker": row["ticker"],
                "exchange": row["exchange"],
                "isAdrc": row["isAdrc"],
                "cik": row["cik"],
                "startDate": self.startDate,
                "endDate": self.endDate,
                "delistDate": row["delistDate"] if not pd.isna(row["delistDate"]) else None,
                "rowIdx": idx
            })
        
        ipoResults = {}
        allActionRows = []
        resultsLock = threading.Lock()
        completed = 0
        errors = 0
        skipped = 0
        exportInterval = 200
        lastExportCount = 0

        if pbar is None:
            pbar = tqdm(
                total=totalTickers,
                desc="Downloading OHLCV for all tickers",
                smoothing=0.1,
                # bar_format="{desc}| {percentage:3.2f}% |{bar}| [{elapsed} elapsed, {remaining} remaining] ",
                colour="green",
                dynamic_ncols=True
            )
        else:
            pbar.reset(total=totalTickers)
            pbar.set_description("OHLCV: Downloading")

        skippedTickers = []
        failedTickers = []

        def downloadAndTrack(item):
            nonlocal completed, errors, skipped, lastExportCount
            exportAfter = False
            try:
                passed, ticker, ipoDate, actionRows = threadWorker(
                    item["ticker"], item["exchange"], item["isAdrc"], item["cik"], item["startDate"], item["endDate"], item["delistDate"],
                    self.priceClient, self.corpActionsClient, self.limiters.alpacaLimiter, self.limiters.edgarLimiter
                )

                with resultsLock:
                    if not passed:
                        skipped += 1
                        skippedTickers.append(item["ticker"])
                        pbar.set_postfix_str(f"{ticker:>5} was intentionally skipped")
                    else:
                        completed += 1
                        ipoResults[item["rowIdx"]] = ipoDate
                        allActionRows.extend(actionRows)
                        pbar.set_postfix_str(f"{ticker:>5}, {errors} errors so far")

                    processed = completed + skipped + errors
                    if processed - lastExportCount >= exportInterval:
                        lastExportCount = processed
                        exportAfter = True

            except Exception as e:
                with resultsLock:
                    errors += 1
                    failedTickers.append(item["ticker"])
                    pbar.set_postfix_str(f"{item['ticker']:>5} FAILED: {e}")

            # Export outside the lock
            pbar.update(1)
            if exportAfter:
                exportActions(allActionRows)


        def exportActions(rows=None):
            if not rows:
                return pd.DataFrame()
            columns = list(rows[0].keys())
            actionsDf = pd.DataFrame(rows).reindex(columns=columns)
            actionsDf.sort_values(by=["date", "exchange", "ticker", "actionType"], ascending=True, inplace=True)
            actionsDf.to_parquet(CORP_ACTIONS_OUTPUT, index=False)
            return actionsDf


        if threads <= 1:
            for item in tickerRows:
                downloadAndTrack(item)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                futures = [executor.submit(downloadAndTrack, item) for item in tickerRows]
                concurrent.futures.wait(futures)

        pbar.set_description(f"OHLCV: Exported {completed} tickers ({errors} errors, {skipped} skipped)")
        print(f"\nOHLCV: Exported {completed} tickers ({errors} errors, {skipped} skipped)\n")

        # Export all corporate actions to one parquet file
        actionsDf = exportActions(allActionRows)
        # pbar.set_description(f"OHLCV: Saved {len(actionsDf)} corporate actions")


        tickerChangeMap = {}
        tickerChangeRows = []

        # Work through all ticker changes and build a mapping from old ticker to new, given a date
        for _, row in actionsDf.iterrows():
            if row["actionType"] == CorporateActionsType.NAME_CHANGE.value:
                changeDate = pd.Timestamp(row["date"], tz=NEW_YORK)
                oldTicker = row["oldTicker"]
                newTicker = row["newTicker"]

                # If there are multiple name changes, the most recent one should take precedence
                if oldTicker not in tickerChangeMap or changeDate > tickerChangeMap[oldTicker][1]:
                    tickerChangeMap[oldTicker] = (newTicker, changeDate)
        
        # Export ticker changes to a parquet file
        for oldTicker, (newTicker, changeDate) in tickerChangeMap.items():
            tickerChangeRows.append({
                "changeDate": changeDate,
                "oldTicker": oldTicker,
                "newTicker": newTicker
            })

        tickerChangesDf = pd.DataFrame(tickerChangeRows).reindex(columns=["changeDate", "oldTicker", "newTicker"])
        tickerChangesDf.sort_values(by=["changeDate", "oldTicker"], ascending=True, inplace=True)
        tickerChangesDf.to_parquet(TICKER_CHANGES_OUTPUT, index=False)
        pbar.set_description(f"OHLCV: Saved {len(tickerChangesDf)} ticker changes")


        if updateIpoDates:
            for idx, ipoDate in ipoResults.items():
                tickersDf.at[idx, "ipoDate"] = ipoDate.strftime("%Y-%m-%d")

            tickersDf.to_parquet(ALL_TICKERS_FILE, index=False)
            pbar.set_description(f"OHLCV: Updated IPO dates for {completed} tickers")

        if pbar is not None and not hasattr(pbar, '_external'):
            pbar.close()

    