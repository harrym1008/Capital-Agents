import os
from datetime import datetime, timezone

# Alpaca's direct Python SDK has issues with pagination, so I will use raw requests
import requests

# from alpaca.data.historical.news import NewsClient as AlpacaNewsClient
# from alpaca.data.requests import NewsRequest as AlpacaNewsRequest

from data.ratelimiter import RateLimiter


class NewsClient:
    def __init__(self, apiKey, apiSecret):
        self.apiKey = apiKey
        self.apiSecret = apiSecret
        self.alpacaLimiter = RateLimiter("alpacaNews", 200, 60)  # 200 calls per min

        self.startDate = datetime.strptime("2020-01-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
        self.endDate = datetime.now(timezone.utc)


    def getNews(self, ticker, printProgress=False):
        allArticles = []
        nextPageToken = None

        url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.apiKey,
            "APCA-API-SECRET-KEY": self.apiSecret
        }

        if printProgress:
            print(f"Fetched 0 articles for {ticker}...", end="", flush=True)

        while True:        
            params = {
                "symbols": ticker,
                "start": self.startDate.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": self.endDate.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit": 50,
                "sort": "asc",
                "include_content": True
            }
            if nextPageToken:
                params["page_token"] = nextPageToken
            
            iters = 0
            pageSuccess = False

            while iters < 10:
                self.alpacaLimiter.wait()

                try:
                    response = requests.get(url, headers=headers, params=params)

                    if response.status_code == 429:
                        iters += 1
                        self.alpacaLimiter.got429(iters)
                        continue

                    response.raise_for_status() 
                    newsResponse = response.json()

                    articlesList = newsResponse.get("news", [])
                    allArticles.extend(articlesList)

                    nextPageToken = newsResponse.get("next_page_token", None)
                    pageSuccess = True

                    if printProgress:
                        mostRecentArticleDate = "N/A"
                        if articlesList:
                            try:
                                rawDate = articlesList[-1].get("created_at")
                                dtDate = datetime.strptime(rawDate, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                mostRecentArticleDate = dtDate.strftime("%B %Y")
                            except Exception:
                                pass
                        print(f"\rFetched {len(allArticles)} articles for {ticker}. Month: {mostRecentArticleDate}        ", end="", flush=True)

                    break

                except Exception as e:
                    iters += 1
                    self.alpacaLimiter.non429Error(e)

            if not pageSuccess:
                print(f"FAILED to fetch news page for {ticker} after 10 attempts! Moving on...")
                break

            if not nextPageToken:
                break

        return allArticles