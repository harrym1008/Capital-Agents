import os
from datetime import datetime

import yfinance as yf
import finnhub

from collectors.ratelimiter import GlobalRateLimiters


class CompanyProfile:
    def __init__(self, ticker, yfData, fhData):
        self.ticker = ticker
        self.prettyName = yfData.get("displayName", ticker)
        self.fullName = yfData.get("longName", ticker)
        self.website = yfData.get("website", "")
        self.industry = yfData.get("industryKey", "")
        self.sector = yfData.get("sectorKey", "")
        self.summary = yfData.get("summary", "")
        self.shares = yfData.get("sharesOutstanding", 0) 
        self.ipoTs = datetime.strptime(fhData.get("ipo", ""), "%Y-%m-%d") if fhData.get("ipo") else None
        self.ipoText = self.ipoTs.strftime("%Y-%m-%d") if self.ipoTs else "N/A"
        self.logo = fhData.get("logo", "")

    def toDict(self):
        return {
            "ticker": self.ticker,
            "prettyName": self.prettyName,
            "fullName": self.fullName,
            "website": self.website,
            "industry": self.industry,
            "sector": self.sector,
            "summary": self.summary,
            "shares": self.shares,
            "ipoText": self.ipoText,
            "ipoTs": self.ipoTs,
            "logo": self.logo
        }
    
    def __str__(self):
        s = f"""Company Profile for {self.prettyName} ({self.ticker}):
Full Name: {self.fullName}
Website: {self.website}
Industry: {self.industry}
Sector: {self.sector}
Business Summary: {self.summary[:128]}...
Outstanding Shares: {self.shares}
IPO Date: {self.ipoText}
Logo URL: {self.logo}
"""
        return s
    
    def shortStr(self):
        return f"{self.prettyName} ({self.ticker}) - {self.industry} in {self.sector}, IPOed on {self.ipoText}"
        


class DescriptionClient:
    def __init__(self, finnhubApiKey, rateLimiterDatabase: GlobalRateLimiters):
        self.finnhubClient = finnhub.Client(api_key=finnhubApiKey)
        self.yahooLimiter = rateLimiterDatabase.yFinanceLimiter
        self.finnhubLimiter = rateLimiterDatabase.finnhubLimiter


    def getYahooInfo(self, ticker):
        iters = 0
        self.yahooLimiter.wait()

        while iters < 10:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                return info
            
            except Exception as e:
                iters += 1
                if "429" in str(e):
                    self.yahooLimiter.got429(iters)
                else:
                    self.yahooLimiter.non429Error(e)

        print(f"FAILED to fetch Yahoo info for {ticker} after 10 attempts!") 
        return {}

    def getFinnhubInfo(self, ticker):
        iters = 0
        self.finnhubLimiter.wait()

        while iters < 10:
            try:
                info = self.finnhubClient.company_profile2(ticker=ticker)
                return info
            
            except Exception as e:
                iters += 1
                if "429" in str(e):
                    self.finnhubLimiter.got429(iters)
                else:
                    self.finnhubLimiter.non429Error(e)

        print(f"FAILED to fetch Finnhub info for {ticker} after 10 attempts!")
        return {}
            

    def getDescription(self, ticker):        
        yahooInfo = self.getYahooInfo(ticker)
        finnhubInfo = self.getFinnhubInfo(ticker)
        return CompanyProfile(ticker, yahooInfo, finnhubInfo)
        


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    client = DescriptionClient(finnhubApiKey=os.getenv("FINNHUB_API_KEY"))

    description = client.getDescription("NVDA")
    print(description)