import os
import json

from data.tickers import TickerClient, Exchange
from data.companyprofile import DescriptionClient, CompanyProfile


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    tickerClient = TickerClient(os.getenv("FINNHUB_API_KEY"))
    descriptionClient = DescriptionClient(os.getenv("FINNHUB_API_KEY"))

    allTickers = tickerClient.getAllTickers()
    nyseTickers = tickerClient.getFormattedTickersForExchange(allTickers, Exchange.NYSE)
    nasdaqTickers = tickerClient.getFormattedTickersForExchange(allTickers, Exchange.NASDAQ)
    # etfTickers = tickerClient.getFormattedTickersForExchange(allTickers, Exchange.ETF)

    print(f"Total NYSE Tickers: {len(nyseTickers)}")
    print(f"Total NASDAQ Tickers: {len(nasdaqTickers)}")
    # print(f"Total ETF Tickers: {len(etfTickers)}")

    exportData = {}

    for (exchange, tickers) in [(Exchange.NYSE, nyseTickers), (Exchange.NASDAQ, nasdaqTickers)]:  # , (Exchange.ETF, etfTickers)]:
        print(f"\n{exchange} Tickers:")
        profile = []
        for ticker in tickers[:5]:
            description = descriptionClient.getDescription(ticker)
            profile.append(description)
            print(f"{description.symbol}, ")
        print(" done.\n\n")

    outputFilename = "db/ticker.json"
    with open(outputFilename, "w") as f:
        json.dump(exportData, f, indent=4)

    
    

