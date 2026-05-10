import os
from dotenv import load_dotenv
from data.dailynews import NewsClient

def main():
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")

    if not api_key or not api_secret:
        print("Error: ALPACA_API_KEY or ALPACA_API_SECRET not found in environment.")
        return

    client = NewsClient(api_key, api_secret)

    tickers = ["AAPL", "NVDA", "WMT", "SNAP"]

    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        try:
            news_articles = client.getNews(ticker, printProgress=True)
            print(f"Successfully fetched {len(news_articles)} articles.")
            
            if news_articles:
                print("First article sample:")
                first_article = news_articles[0]
                print(f"Timestamp: {first_article.get('created_at', 'N/A')}")
                print(f"Headline: {first_article.get('headline', 'N/A')}")
            else:
                print("No news articles found.")
        except Exception as e:
            print(f"An error occurred during the test for {ticker}: {e}")

if __name__ == "__main__":
    main()
