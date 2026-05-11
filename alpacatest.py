from collectors.alpaca.alpacaclient import AlpacaStockPricesClient

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    apiKey = os.getenv("ALPACA_API_KEY")
    apiSecret = os.getenv("ALPACA_API_SECRET")

    client = AlpacaStockPricesClient(apiKey, apiSecret, "2016-01-01", "2026-04-30")
    tickers = ["NVDA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "BRK.B", "JPM", "JNJ", 
               "V", "PG", "UNH", "HD", "MA", "DIS", "NVDA", "PYPL", "BAC", "ADBE", "CMCSA", 
               "NFLX", "INTC", "TMO", "PFE", "CSCO", "PEP", "XOM", "KO", "ABT", "CVX",
               "NKE", "MRK", "WMT", "ORCL", "COST", "DHR", "MCD", "LLY", "MDT", "NEE",
               "AMGN", "BMY", "TXN", "QCOM", "UNP", "HON", "IBM", "SBUX", "LOW"]
    
    df, splitDatesByTicker = client.downloadOhlcAndSplitsBatch(tickers)
    
    print(df)
    print(splitDatesByTicker)
    