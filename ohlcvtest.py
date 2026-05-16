from collectors.ohlcvclient import OHLCVDataClient, threadWorker
from collectors.ratelimiter import GlobalRateLimiters

from collectors.constants import * 

import pandas as pd


def main():    
    tickers = pd.read_parquet("data/tickers.parquet")
    tickerCount = len(tickers)
    del tickers

    if input("ARE YOU SURE? This will download ALL DAILY OHLCV values, and corporate actions, from 1 Jan 2016 to 30 Apr 2026 "
             f"for {tickerCount:,} tickers! This will take MULTIPLE HOURS! (yes/no) ").lower() != "yes":
        print("Aborting.")
        return
    
    client = OHLCVDataClient(START_DATE_STR, END_DATE_STR, GlobalRateLimiters())


    threadWorker(
        "BINI", "XNAS", START_DATE_STR, END_DATE_STR, None, 
        client.priceClient, client.corpActionsClient, client.limiters.alpacaLimiter
    )


    threadWorker(
        "MULN", "XNAS", START_DATE_STR, END_DATE_STR, None, 
        client.priceClient, client.corpActionsClient, client.limiters.alpacaLimiter
    )


    threadWorker(
        "NETE", "XNAS", START_DATE_STR, END_DATE_STR, None, 
        client.priceClient, client.corpActionsClient, client.limiters.alpacaLimiter
    )


    threadWorker(
        "TWTR", "XNAS", START_DATE_STR, END_DATE_STR, None, 
        client.priceClient, client.corpActionsClient, client.limiters.alpacaLimiter
    )


    threadWorker(
        "NVDA", "XNAS", START_DATE_STR, END_DATE_STR, None, 
        client.priceClient, client.corpActionsClient, client.limiters.alpacaLimiter
    )


    threadWorker(
        "SNDK", "XNAS", START_DATE_STR, END_DATE_STR, None, 
        client.priceClient, client.corpActionsClient, client.limiters.alpacaLimiter
    )


    # client.massDownload(True)



if __name__ == "__main__":
    main()
