import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from collectors.ohlcv_dl_client import OHLCVDataClient
from collectors.ticker_client import TickerDataClient
from collectors.news_dl_client import NewsClient
from collectors.macro_dl_client import MacroDataClient
from collectors.rate_limiter import GlobalRateLimiters
from collectors.constants import *

from tqdm import tqdm
import concurrent.futures
import time


if __name__ == "__main__":

    print("=" * 60)
    print("Mass Download Tool")
    print("=" * 60, "\n")

    dateStart = pd.Timestamp(START_DATE_STR, tz=NEW_YORK).strftime("%d %b %Y")
    dateEnd = pd.Timestamp(END_DATE_STR, tz=NEW_YORK).strftime("%d %b %Y")

    downloading = {
        "tickers": {
            "confirm": input("Download tickers? (yes/no)        > ").lower() == "yes",
            "desc": f"All tickers (listed and delisted) from the NYSE and NASDAQ from {dateStart} to {dateEnd}"
        },
        "ohlcv": {
            "confirm": input("Download OHLCV data? (yes/no)     > ").lower() == "yes",
            "desc": f"All daily OHLCV values, and corporate actions, from {dateStart} to {dateEnd} for all tickers"
        },
        "news": {
            "confirm": input("Download news articles? (yes/no)  > ").lower() == "yes",
            "desc": f"All news articles from {dateStart} to {dateEnd}"
        },
        "macro": {
            "confirm": input("Download macro data? (yes/no)     > ").lower() == "yes",
            "desc": f"Commodities, indices, forex and macroeconomic data from FRED from {dateStart} to {dateEnd}"
        }
    }

    if not any(d["confirm"] for d in downloading.values()):
        print("Nothing to download. Aborting.")
        exit()


    print(f"\nConfirmation! The following will be deleted and redownloaded:")
    for category, download in downloading.items():
        if download["confirm"]:
            print(f" - {download['desc']}")

    print("\nThis WILL TAKE MULTIPLE HOURS!")
    print("To confirm, type the following exactly: \"Proceed!\"")

    if input("> ") != "Proceed!":
        print("Aborting.")
        exit()


    print()
    print( "=" * 60)
    print("Starting downloads in 5 seconds...")
    print("=" * 60, "\n")

    time.sleep(5)

    # Single shared rate limiter instance for ALL parallel threads
    limiters = GlobalRateLimiters()

    # Create positioned progress bars (one per category) to prevent overlap
    # position=0 is tickers (runs first), 1-3 are the parallel downloads
    pbars = {}
    if downloading["tickers"]["confirm"]:
        pbars["tickers"] = tqdm(total=1, desc="[TICKERS]", position=0, leave=True, dynamic_ncols=True)
    
    # Mark external so clients don't close them
    for pbar in pbars.values():
        pbar._external = True

    # Step 1: Tickers MUST run first (OHLCV depends on tickers.parquet)
    if downloading["tickers"]["confirm"]:
        tickerClient = TickerDataClient(START_DATE, END_DATE, limiters)
        tickerClient.massTickerDownloadWithData(pbar=pbars["tickers"])
        tqdm.write(f"Completed downloading: {downloading['tickers']['desc']}\n")

    if downloading["ohlcv"]["confirm"]:
        pbars["ohlcv"] = tqdm(total=1, desc="[OHLCV]", position=1, leave=True, dynamic_ncols=True)
    if downloading["news"]["confirm"]:
        pbars["news"] = tqdm(total=1, desc="[NEWS]", position=2, leave=True, dynamic_ncols=True)
    if downloading["macro"]["confirm"]:
        pbars["macro"] = tqdm(total=1, desc="[MACRO]", position=3, leave=True, dynamic_ncols=True)


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

    if parallelTasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallelTasks)) as executor:
            futures = {}
            for name, clientClass, args, runner in parallelTasks:
                client = clientClass(*args)
                futures[executor.submit(runner, client)] = name

            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
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
