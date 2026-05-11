import os
from enum import Enum

import finnhub


class Exchange(Enum):
    NYSE = "XNYS"
    NASDAQ = "XNAS"
    ETF = "ARCX"



class TickerClient:
    def __init__(self, apiKey):
        self.finnhubClient = finnhub.Client(apiKey)

    def getAllTickers(self):
        allTickers = self.finnhubClient.stock_symbols("US")
        allTickers.sort(key=lambda x: x["symbol"])
        return allTickers
    

    def formatEntry(self, ticker):
        return {
            "ticker": ticker["symbol"],
            "description": ticker["description"],
            "exchange": ticker["mic"],
            "type": ticker["type"]
        }


    def getFormattedSecuritiesForExchange(self, allTickers, exchange: Exchange):
        tickers = []

        if exchange == Exchange.ETF:
            for ticker in allTickers:
                if ticker["type"] == "ETP":
                    tickers.append(self.formatEntry(ticker))
        else:
            for ticker in allTickers:
                if ticker["mic"] == exchange.value and ticker["type"] == "Common Stock":
                    tickers.append(self.formatEntry(ticker))

        return tickers

        


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    client = TickerClient(os.getenv("FINNHUB_API_KEY"))

    allTickers = client.getAllTickers()
    nyseTickers = client.getFormattedTickersForExchange(allTickers, Exchange.NYSE)
    nasdaqTickers = client.getFormattedTickersForExchange(allTickers, Exchange.NASDAQ)
    etfTickers = client.getFormattedTickersForExchange(allTickers, Exchange.ETF)

    print(f"Total NYSE Tickers: {len(nyseTickers)}")
    print(f"Total NASDAQ Tickers: {len(nasdaqTickers)}")
    print(f"Total ETF Tickers: {len(etfTickers)}")
