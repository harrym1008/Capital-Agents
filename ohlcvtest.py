from collectors.ohlcvclient import OHLCVDataClient, threadWorker
from collectors.ratelimiter import GlobalRateLimiters

from collectors.constants import * 

import pandas as pd


def main():    
    testedTickers = ["BINI", "MULN", "NETE", "TWTR", "NVDA", "SNDK"]

    tickers = pd.read_parquet("data/tickers.parquet")
    tickerCount = len(tickers) if not testedTickers else len(testedTickers)
    del tickers

    if input("ARE YOU SURE? This will download ALL DAILY OHLCV values, and corporate actions, from 1 Jan 2016 to 30 Apr 2026 "
             f"for {tickerCount:,} tickers! This will take MULTIPLE HOURS! (yes/no) ").lower() != "yes":
        print("Aborting.")
        return
    
    client = OHLCVDataClient(START_DATE_STR, END_DATE_STR, GlobalRateLimiters())
    client.testedTickers = testedTickers
    client.massDownload(True)



if __name__ == "__main__":
    main()
