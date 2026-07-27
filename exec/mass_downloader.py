import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.ohlcv_dl_client import OHLCVDataClient
from collectors.ticker_dl_client import TickerDataClient
from collectors.news_dl_client import NewsClient
from collectors.macro_dl_client import MacroDataClient
from collectors.shortdata_dl_client import ShortDataClient
from collectors.forex_dl_client import CurrencyDataClient
from collectors.rate_limiter import GlobalRateLimiters
from collectors.constants import *

from tqdm import tqdm
import concurrent.futures
import time



def runMassDownloadTool(presetDownloads=None):
    print("=" * 60)
    print("  Mass Download Tool")
    print("=" * 60, "\n")

    dateStartStr = pd.Timestamp(START_DATE_STR, tz=NEW_YORK).strftime("%d %b %Y")
    dateEndStr = pd.Timestamp(END_DATE_STR, tz=NEW_YORK).strftime("%d %b %Y")

    if presetDownloads is not None:
        downloading = {
            "tickers": {
                "confirm": presetDownloads.get("tickers", False),
                "desc": f"All tickers (listed and delisted) from the NYSE and NASDAQ from {dateStartStr} to {dateEndStr}"
            },
            "ohlcv": {
                "confirm": presetDownloads.get("ohlcv", False),
                "desc": f"All daily OHLCV values, and corporate actions, from {dateStartStr} to {dateEndStr} for all tickers"
            },
            "news": {
                "confirm": presetDownloads.get("news", False),
                "desc": f"All news articles from {dateStartStr} to {dateEndStr}"
            },
            "macro": {
                "confirm": presetDownloads.get("macro", False),
                "desc": f"Commodities, indices, forex and macroeconomic data from FRED from {dateStartStr} to {dateEndStr}"
            },
            "forex": {
                "confirm": presetDownloads.get("forex", False),
                "desc": f"Forex data for 20 currencies to USD from {dateStartStr} to {dateEndStr}"
            },
            "short": {
                "confirm": presetDownloads.get("short", False),
                "desc": f"FINRA short interest data from {'Jun 2021' if \
                            START_DATE < pd.Timestamp('2021-06-01', tz=NEW_YORK) else dateStartStr} to {dateEndStr}"
            }
        }
    else:
        downloading = {
            "tickers": {
                "confirm": input("Download tickers? (yes/no)        > ").lower() == "yes",
                "desc": f"All tickers (listed and delisted) from the NYSE and NASDAQ from {dateStartStr} to {dateEndStr}"
            },
            "ohlcv": {
                "confirm": input("Download OHLCV data? (yes/no)     > ").lower() == "yes",
                "desc": f"All daily OHLCV values, and corporate actions, from {dateStartStr} to {dateEndStr} for all tickers"
            },
            "news": {
                "confirm": input("Download news articles? (yes/no)  > ").lower() == "yes",
                "desc": f"All news articles from {dateStartStr} to {dateEndStr}"
            },
            "macro": {
                "confirm": input("Download macro data? (yes/no)     > ").lower() == "yes",
                "desc": f"Commodities, indices, forex and macroeconomic data from FRED from {dateStartStr} to {dateEndStr}"
            },
            "forex": {
                "confirm": input("Download forex data? (yes/no)     > ").lower() == "yes",
                "desc": f"Forex data for 20 currencies to USD from {dateStartStr} to {dateEndStr}"
            },
            "short": {
                "confirm": input("Download short data? (yes/no)     > ").lower() == "yes",
                "desc": f"FINRA short interest data from {'Jun 2021' if \
                            START_DATE < pd.Timestamp('2021-06-01', tz=NEW_YORK) else dateStartStr} to {dateEndStr}"
            }
        }

    if not any(d["confirm"] for d in downloading.values()):
        print("Nothing to download. Aborting.")
        exit()


    print(f"\nConfirmation! The following will be deleted and redownloaded:")
    for id, download in downloading.items():
        if download["confirm"]:
            print(f" - {download['desc']}")

    # print("\nThis WILL TAKE MULTIPLE HOURS!")
    # print("To confirm, type the following exactly: \"Proceed!\"")

    # if input("> ") != "Proceed!":
    #     print("Aborting.")
    #     exit()


    print()
    print( "=" * 60)
    print("Starting downloads in 5 seconds...")
    print("=" * 60, "\n")

    time.sleep(5)

    limiters = GlobalRateLimiters()

    # Step 1: Tickers MUST run first
    if downloading["tickers"]["confirm"]:
        tickerClient = TickerDataClient(START_DATE, END_DATE, limiters)
        tickerClient.massTickerDownloadWithData()
        tqdm.write(f"Completed downloading: {downloading['tickers']['desc']}\n")

    pbars = {}
    if downloading["ohlcv"]["confirm"]:
        pbars["ohlcv"] = tqdm(total=1, desc="[OHLCV]", position=1, leave=True, dynamic_ncols=True)
    if downloading["news"]["confirm"]:
        pbars["news"] = tqdm(total=1, desc="[NEWS]", position=2, leave=True, dynamic_ncols=True)
    if downloading["macro"]["confirm"]:
        pbars["macro"] = tqdm(total=1, desc="[MACRO]", position=3, leave=True, dynamic_ncols=True)
    if downloading["forex"]["confirm"]:
        pbars["forex"] = tqdm(total=1, desc="[FOREX]", position=4, leave=True, dynamic_ncols=True)
    if downloading["short"]["confirm"]:
        pbars["short"] = tqdm(total=1, desc="[SHORT]", position=5, leave=True, dynamic_ncols=True)
    for pbar in pbars.values():
        pbar._external = True

    # Step 2: OHLCV, News, Macro run in parallel (all share the same limiters instance)
    parallelTasks = []

    if downloading["ohlcv"]["confirm"]:
        parallelTasks.append(("ohlcv", OHLCVDataClient, (START_DATE_STR, END_DATE_STR, limiters),
                              lambda c: c.massDownload(True, threads=8, pbar=pbars["ohlcv"])))

    if downloading["news"]["confirm"]:
        parallelTasks.append(("news", NewsClient, (START_DATE_STR, END_DATE_STR, limiters),
                              lambda c: (c.threadedMassDownload(threads=8, pbar=pbars["news"]), c.buildInvertedIndex(pbar=pbars["news"]))))

    if downloading["macro"]["confirm"]:
        parallelTasks.append(("macro", MacroDataClient, (START_DATE_STR, END_DATE_STR, limiters),
                              lambda c: c.massDownload(pbar=pbars["macro"])))

    if downloading["forex"]["confirm"]:
        parallelTasks.append(("forex", CurrencyDataClient, (START_DATE_STR, END_DATE_STR, limiters),
                              lambda c: c.massDownload(pbar=pbars["forex"])))
        
    if downloading["short"]["confirm"]:
        parallelTasks.append(("short", ShortDataClient, (START_DATE.date(), END_DATE.date(), limiters),
                              lambda c: c.massDownload(pbar=pbars["short"])))

    if parallelTasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallelTasks)) as executor:
            futures = {}
            for name, clientClass, args, runner in parallelTasks:
                client = clientClass(*args)
                futures[executor.submit(runner, client)] = name

            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                pbars[name].close()
                try:
                    future.result()
                    tqdm.write(f"Completed downloading: {downloading[name]['desc']}\n")
                except Exception as e:
                    tqdm.write(f"ERROR in {name} download: {e}\n")

    # Close all progress bars
    for pbar in pbars.values():
        pbar.close()

    print("\n" + "=" * 60)
    print("All downloads complete!")
    print("=" * 60)



if __name__ == "__main__":
    runMassDownloadTool()