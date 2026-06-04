from collectors.ohlcvclient import OHLCVDataClient
from collectors.tickerclient import TickerDataClient
from collectors.newsclient import NewsClient
from collectors.macroclient import MacroDataClient
from collectors.ratelimiter import GlobalRateLimiters
from collectors.constants import *


if __name__ == "__main__":
    
    print("=" * 60)
    print("Mass Download Tool")
    print("=" * 60, "\n")

    downloading = {
        "tickers": {
            "confirm": input("Download tickers? (yes/no)        > ").lower() == "yes",
            "desc": "All tickers (listed and delisted) from the NYSE and NASDAQ from 1 Jan 2016 to 31 May 2026"
        },
        "ohlcv": {
            "confirm": input("Download OHLCV data? (yes/no)     > ").lower() == "yes",
            "desc": "All daily OHLCV values, and corporate actions, from 1 Jan 2016 to 31 May 2026 for all tickers"
        },
        "news": {
            "confirm": input("Download news articles? (yes/no)  > ").lower() == "yes",
            "desc": "All news articles from 1 Jan 2016 to 31 May 2026"
        },
        "macro": {
            "confirm": input("Download macro data? (yes/no)     > ").lower() == "yes",
            "desc": "Commodities, indices, forex and macroeconomic data from FRED from 1 Jan 2016 to 31 May 2026"
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
    print("To confirm, type the following exactly: \"I wish to proceed.\"")

    if input("> ") != "I wish to proceed.":
        print("Aborting.")
        exit()


    print()
    print( "=" * 60)
    print("Starting downloads in 5 seconds...")
    print("=" * 60, "\n")

    import time
    time.sleep(5)

    limiters = GlobalRateLimiters()

    if downloading["tickers"]["confirm"]:
        tickerClient = TickerDataClient(START_DATE, END_DATE, limiters)
        tickerClient.massTickerDownloadWithData()
        print(f"Completed downloading: {downloading['tickers']['desc']}\n")

    if downloading["ohlcv"]["confirm"]:
        ohlcvClient = OHLCVDataClient(START_DATE_STR, END_DATE_STR, limiters)
        ohlcvClient.massDownload(True, threads=8)
        print(f"Completed downloading: {downloading['ohlcv']['desc']}\n")

    if downloading["news"]["confirm"]:
        newsClient = NewsClient(START_DATE_STR, END_DATE_STR, limiters)
        newsClient.threadedMassDownload(threads=8)
        newsClient.buildInvertedIndex()
        print(f"Completed downloading: {downloading['news']['desc']}\n")

    if downloading["macro"]["confirm"]:
        macroClient = MacroDataClient(START_DATE_STR, END_DATE_STR, limiters)
        macroClient.massDownload()
        print(f"Completed downloading: {downloading['macro']['desc']}\n")
